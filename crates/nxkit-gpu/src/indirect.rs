//! The compute pre-pass of the GPU driven draws of a render pass.
//!
//! Two kinds of draw need their arguments changed on the GPU before Metal can draw them:
//!
//! - `MultiDraw{Arrays,Elements}IndirectCount` runs draw `i` only while `i` is below a count that the GPU wrote
//!   earlier in the frame (culling). Metal cannot skip draws by a count in memory. The pre-pass copies the arguments of
//!   every draw up to the maximum and writes zeros past the count, and the render pass issues one indirect draw for
//!   each copy. A draw of zero vertices or instances draws nothing.
//! - `DrawMeshTasksIndirect` reads `{count, first}`, where Metal's indirect mesh draw reads three threadgroup counts.
//!   The pre-pass writes `{count, 1, 1}` and the first workgroup after it. Without a task shader it writes
//!   [`mesh::INDIRECT_SPLITS`](crate::mesh::INDIRECT_SPLITS) grids of at most 1024 workgroups, each with its first
//!   workgroup, and the grids past the count are empty.
//!
//! The choice is a compute pre-pass and not an indirect command buffer: an ICB would need the same kernel to encode
//! the draws, plus GPU encoding of the whole pipeline and argument state, while the copies leave the draw on the normal
//! path of the render pass. The CPU never waits for the GPU. The pre-pass runs once for each render pass that has such
//! draws, before the pass, in the same command buffer: a compute pass cannot run inside a render pass. It waits for
//! all earlier work, and the render pass waits for it. The arguments must be final when the render pass starts, which
//! holds when a dispatch or copy wrote them. The pre-pass does not see a draw of the same pass that writes them through
//! a storage buffer.

use nxkit_metal::{
    ArgumentTable, CommandBuffer, ComputeCommand, Dependency, Scope, Scratch, Stages, Visibility,
};

use crate::detail::DrawArgs;
use crate::drawctx::Shared;
use crate::fastmap::{FastMap, FastSet};
use crate::mesh::{INDIRECT_SPLITS, MAX_GRID};
use crate::skip::Skip;
use crate::stats::EncodeStats;

/// The bytes from one copied record to the next. A record has at most five words and the first workgroup of a mesh
/// grid is at word 4.
pub(crate) const RECORD: u64 = 32;
pub(crate) use crate::geometry::MAX_MULTI;
/// The threads of a threadgroup of the kernel.
const GROUP: usize = 64;
/// The bytes of one job of the kernel: three addresses and four words.
const JOB_BYTES: usize = 40;

/// The kernel. A job copies `words` words of draw `i` from `rec` to `out`, or zeros when `i` is not below `*cnt`. A
/// job with no words is a mesh grid: `out` gets the count of `rec[0]` from `base` on, at most `limit`, as `{n, 1, 1}`,
/// and word 4 gets the first workgroup `rec[1] + base`. A job with no output is padding.
const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct job { device const uint *rec; device const uint *cnt; device uint *out; uint i, words, base, limit; };
kernel void nxkit_indirect(device const job *j [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    job m = j[t];
    if (!m.out) return;
    if (m.words == 0) {
        uint c = m.rec[0];
        m.out[0] = min(c > m.base ? c - m.base : 0u, m.limit);
        m.out[1] = 1u;
        m.out[2] = 1u;
        m.out[4] = m.rec[1] + m.base;
        return;
    }
    bool live = m.i < m.cnt[0];
    for (uint k = 0; k < m.words; k++) m.out[k] = live ? m.rec[k] : 0u;
}
"#;

/// Same-frame GPU conditional rendering kernel (`nxkit_cond_draw`). Evaluates two operands summing vis-buffer slots
/// plus CPU-known constants, then writes indirect draw or mesh grid arguments (`kind`: 0 arrays, 1 indexed, 2 mesh),
/// zeroing `instanceCount` or threadgroup count when false.
const COND_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct cond_job {
    device const ulong *vis;
    device const uint *src;
    device uint *out;
    ulong const_a;
    ulong const_b;
    uint offsets_a[4];
    uint offsets_b[4];
    uint count_a;
    uint count_b;
    uint mode;
    uint w0;
    uint w1;
    uint w2;
    uint w3;
    uint w4;
    uint kind;
    uint _pad;
};
kernel void nxkit_cond_draw(device const cond_job *j [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    cond_job m = j[t];
    if (!m.out) return;
    ulong a = m.const_a;
    for (uint k = 0; k < m.count_a; k++) a += m.vis[m.offsets_a[k]];
    ulong b = m.const_b;
    for (uint k = 0; k < m.count_b; k++) b += m.vis[m.offsets_b[k]];
    bool live = (m.mode == 0u) ? (a == b) : (a != b);
    if (m.kind == 2u) {
        uint c = live ? (m.src ? m.src[0] : m.w0) : 0u;
        uint first = m.src ? m.src[1] : m.w1;
        m.out[0] = min(c > m.w2 ? c - m.w2 : 0u, m.w3);
        m.out[1] = 1u;
        m.out[2] = 1u;
        m.out[4] = first + m.w2;
        return;
    }
    uint w0 = m.src ? m.src[0] : m.w0;
    uint w1 = m.src ? m.src[1] : m.w1;
    uint w2 = m.src ? m.src[2] : m.w2;
    uint w3 = m.src ? m.src[3] : m.w3;
    uint w4 = m.src ? (m.kind == 1u ? m.src[4] : 0u) : m.w4;
    m.out[0] = w0;
    m.out[1] = live ? w1 : 0u;
    m.out[2] = w2;
    m.out[3] = w3;
    if (m.kind == 1u) m.out[4] = w4;
}
"#;

const COND_JOB_BYTES: usize = 112;

struct CondJob {
    vis: u64,
    src: u64,
    cond: crate::queries::ResolvedGpuCond,
    words: [u32; 5],
    kind: u32,
}

/// One job of the kernel. `rec` and `cnt` are real addresses.
struct Job {
    rec: u64,
    cnt: u64,
    i: u32,
    words: u32,
    base: u32,
    limit: u32,
}

/// The copies of the GPU driven draws of one render pass, by draw ordinal: the address of the first record, or why the
/// draw has none.
#[derive(Default)]
pub(crate) struct Prepass {
    pub(crate) copies: FastMap<u32, Result<u64, Skip>>,
    /// Synthesised indirect argument records for draws gated by same-frame `GpuCond`, by draw ordinal.
    pub(crate) cond_indirect: FastMap<u32, u64>,
    /// Direct `MeshTasks` draws whose indirect arguments were written by `nxkit_cond_draw`.
    pub(crate) cond_mesh: FastSet<u32>,
    /// True when a draw of the pass reads arguments or data that earlier GPU work wrote, so the pass waits for it.
    pub(crate) gpu_driven: bool,
    /// Render stages that read records produced by `nxkit_cond_draw` (without waiting on `FRAGMENT` when not `gpu_driven`).
    pub(crate) cond_stages: Option<Stages>,
}

fn add_cond_stage(slot: &mut Option<Stages>, s: Stages) {
    *slot = Some(match *slot {
        Some(cur) => cur | s,
        None => s,
    });
}

/// The jobs of draw `d`, or why it gets none. `None` for a draw that needs no copy. The jobs hold real addresses: the
/// kernel loads through them.
fn jobs_of(
    at: &dyn Fn(u64, u64) -> Result<u64, Skip>,
    args: &DrawArgs,
    task: bool,
    out: &mut Vec<Job>,
) -> Option<Result<(), Skip>> {
    match *args {
        DrawArgs::MultiIndirectCount {
            index,
            indirect,
            count,
            max,
            stride,
        } => {
            let words = if index.is_some() { 5 } else { 4 };
            let n = max.min(MAX_MULTI);
            let Ok(cnt) = (if stride.is_multiple_of(4) {
                at(count, 4)
            } else {
                Err(Skip::Buffer)
            }) else {
                return Some(Err(crate::buf_skip(format_args!(
                    "multi-draw count {count:#x} stride {stride}"
                ))));
            };
            let first = out.len();
            for i in 0..n {
                let rec = indirect.wrapping_add(u64::from(i).wrapping_mul(stride));
                let Ok(real) = at(rec, 4 * u64::from(words)) else {
                    out.truncate(first);
                    return Some(Err(crate::buf_skip(format_args!(
                        "multi-draw record {i} at {rec:#x}"
                    ))));
                };
                out.push(Job {
                    rec: real,
                    cnt,
                    i,
                    words,
                    base: 0,
                    limit: 0,
                });
            }
            Some(Ok(()))
        }
        DrawArgs::MeshTasksIndirect { indirect } => {
            let Ok(real) = at(indirect, 8) else {
                return Some(Err(crate::buf_skip(format_args!(
                    "mesh draw arguments at {indirect:#x}"
                ))));
            };
            let (splits, limit) = if task {
                (1, u32::MAX)
            } else {
                (INDIRECT_SPLITS, MAX_GRID)
            };
            for k in 0..splits {
                out.push(Job {
                    rec: real,
                    cnt: real,
                    i: 0,
                    words: 0,
                    base: k * MAX_GRID,
                    limit,
                });
            }
            Some(Ok(()))
        }
        _ => None,
    }
}

/// Runs the pre-pass for the draws `range` of a render pass, when one of them needs it.
pub(crate) fn prepass(
    sh: &Shared<'_>,
    cb: &mut CommandBuffer,
    table: &ArgumentTable,
    scratch: &Scratch,
    queries: Option<&crate::queries::FrameQueries>,
    range: std::ops::Range<u32>,
    stats: &mut EncodeStats,
) -> Prepass {
    let mut out = Prepass::default();
    let mut jobs = Vec::new();
    // The draw ordinal and first job of each draw that gets copies.
    let mut firsts: Vec<(u32, usize)> = Vec::new();
    let mut cond_jobs: Vec<CondJob> = Vec::new();
    let mut cond_draw_ords: Vec<(u32, usize)> = Vec::new();
    let mut cond_mesh_ords: Vec<(u32, usize)> = Vec::new();
    let vis_addr = queries.and_then(|q| q.buffer()).map(|b| b.gpu_address());
    for n in range {
        let Some(d) = sh.detail.draws.get(n as usize) else {
            continue;
        };
        // A task or mesh shader reads what the culling of the frame wrote, through its storage buffers.
        out.gpu_driven |= d.args.is_indirect() || d.args.is_mesh();
        let cond = vis_addr
            .zip(queries.and_then(|q| q.gpu_cond(n)))
            .filter(|_| d.topology != crate::translate::QUADS);
        let mut handled_mesh_indirect = false;
        if let Some((vis, cond)) = cond {
            match d.args {
                DrawArgs::Arrays {
                    first,
                    count,
                    instances,
                    base_instance,
                } => {
                    let at = cond_jobs.len();
                    cond_jobs.push(CondJob {
                        vis,
                        src: 0,
                        cond,
                        words: [count, instances, first, base_instance, 0],
                        kind: 0,
                    });
                    cond_draw_ords.push((n, at));
                    add_cond_stage(&mut out.cond_stages, Stages::VERTEX);
                }
                DrawArgs::Indexed {
                    index_type,
                    count,
                    instances,
                    base_vertex,
                    base_instance,
                    ..
                } if index_type != 0 => {
                    let at = cond_jobs.len();
                    cond_jobs.push(CondJob {
                        vis,
                        src: 0,
                        cond,
                        words: [count, instances, 0, base_vertex as u32, base_instance],
                        kind: 1,
                    });
                    cond_draw_ords.push((n, at));
                    add_cond_stage(&mut out.cond_stages, Stages::VERTEX);
                }
                DrawArgs::ArraysIndirect { indirect } => {
                    if let Ok(src) = sh.indirect_real(indirect, 16) {
                        let at = cond_jobs.len();
                        cond_jobs.push(CondJob {
                            vis,
                            src,
                            cond,
                            words: [0; 5],
                            kind: 0,
                        });
                        cond_draw_ords.push((n, at));
                        add_cond_stage(&mut out.cond_stages, Stages::VERTEX);
                    }
                }
                DrawArgs::IndexedIndirect {
                    index_type,
                    indirect,
                    ..
                } if index_type != 0 => {
                    if let Ok(src) = sh.indirect_real(indirect, 20) {
                        let at = cond_jobs.len();
                        cond_jobs.push(CondJob {
                            vis,
                            src,
                            cond,
                            words: [0; 5],
                            kind: 1,
                        });
                        cond_draw_ords.push((n, at));
                        add_cond_stage(&mut out.cond_stages, Stages::VERTEX);
                    }
                }
                DrawArgs::MeshTasks { first, count } => {
                    let task = d.program.task.is_some();
                    let splits = crate::mesh::splits(count, task).max(1);
                    let limit = if task { u32::MAX } else { MAX_GRID };
                    let at = cond_jobs.len();
                    for k in 0..splits {
                        cond_jobs.push(CondJob {
                            vis,
                            src: 0,
                            cond,
                            words: [count, first, k * MAX_GRID, limit, 0],
                            kind: 2,
                        });
                    }
                    cond_mesh_ords.push((n, at));
                    add_cond_stage(&mut out.cond_stages, Stages::OBJECT | Stages::MESH);
                }
                DrawArgs::MeshTasksIndirect { indirect } => {
                    handled_mesh_indirect = true;
                    match sh.indirect_real(indirect, 8) {
                        Ok(src) => {
                            let task = d.program.task.is_some();
                            let splits = if task { 1 } else { INDIRECT_SPLITS };
                            let limit = if task { u32::MAX } else { MAX_GRID };
                            let at = cond_jobs.len();
                            for k in 0..splits {
                                cond_jobs.push(CondJob {
                                    vis,
                                    src,
                                    cond,
                                    words: [0, 0, k * MAX_GRID, limit, 0],
                                    kind: 2,
                                });
                            }
                            cond_mesh_ords.push((n, at));
                            add_cond_stage(&mut out.cond_stages, Stages::OBJECT | Stages::MESH);
                        }
                        Err(_) => {
                            out.copies.insert(
                                n,
                                Err(crate::buf_skip(format_args!(
                                    "mesh draw arguments at {indirect:#x}"
                                ))),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        if !handled_mesh_indirect {
            let at = jobs.len();
            match jobs_of(
                &|a, b| sh.indirect_real(a, b),
                &d.args,
                d.program.task.is_some(),
                &mut jobs,
            ) {
                Some(Ok(())) => firsts.push((n, at)),
                Some(Err(why)) => {
                    out.copies.insert(n, Err(why));
                }
                None => {}
            }
            if let DrawArgs::MultiIndirectCount { max, .. } = d.args {
                stats.kinds.clamped += max.saturating_sub(MAX_MULTI);
            }
        }
    }
    if !cond_jobs.is_empty()
        && let Ok(base) = run_cond(sh, cb, table, scratch, &cond_jobs)
    {
        for (n, k) in cond_draw_ords {
            out.cond_indirect.insert(n, base + RECORD * k as u64);
        }
        for (n, k) in cond_mesh_ords {
            out.copies.insert(n, Ok(base + RECORD * k as u64));
            out.cond_mesh.insert(n);
        }
        stats.compute_passes += 1;
        stats.kinds.prepasses += 1;
        stats.kinds.prepass_records += cond_jobs.len() as u32;
    }
    if firsts.is_empty() {
        return out;
    }
    if jobs.is_empty() {
        // Multi-draws with a maximum of 0: they draw nothing and need no kernel.
        out.copies
            .extend(firsts.into_iter().map(|(n, _)| (n, Ok(0))));
        return out;
    }
    let r = run(sh, cb, table, scratch, &jobs);
    for (n, at) in firsts {
        out.copies
            .insert(n, r.map(|base| base + RECORD * at as u64));
    }
    if r.is_ok() {
        stats.compute_passes += 1;
        stats.kinds.prepasses += 1;
        stats.kinds.prepass_records += jobs.len() as u32;
    }
    out
}

/// Evaluates same-frame `GpuCond` occlusion jobs on the GPU into `scratch` indirect draw records.
fn run_cond(
    sh: &Shared<'_>,
    cb: &mut CommandBuffer,
    table: &ArgumentTable,
    scratch: &Scratch,
    jobs: &[CondJob],
) -> Result<u64, Skip> {
    let pipeline = sh
        .pipelines
        .host_compute("nxkit-cond-draw", COND_MSL, "nxkit_cond_draw")?;
    let push = |b: &[u8]| {
        scratch
            .push(b)
            .map_err(|e| crate::buf_skip(format_args!("cond-draw {} bytes: {e:?}", b.len())))
    };
    let outputs = push(&vec![0u8; jobs.len() * RECORD as usize])?;
    let bytes = encode_cond_jobs(jobs, outputs);
    let list = push(&bytes)?;
    let mut after = Stages::FRAGMENT;
    if jobs.iter().any(|j| j.src != 0) {
        after = after | Stages::DISPATCH | Stages::BLIT;
    }
    let dep = Dependency {
        scope: Scope::FromEarlierPasses,
        after,
        before: Stages::DISPATCH,
        visibility: Visibility::Device,
    };
    let mut cp = cb.begin_compute_pass(table).map_err(|_| Skip::Encode)?;
    let groups = bytes.len() / (GROUP * COND_JOB_BYTES);
    for c in [
        ComputeCommand::Depend(dep),
        ComputeCommand::SetPipeline(pipeline),
        ComputeCommand::SetBuffer {
            index: 0,
            gpu_addr: list,
        },
        ComputeCommand::DispatchThreadgroups {
            groups: [groups as u32, 1, 1],
            threads: [GROUP as u32, 1, 1],
        },
    ] {
        cp.encode(&c).map_err(|_| Skip::Encode)?;
    }
    cp.end();
    Ok(outputs)
}

fn encode_cond_jobs(jobs: &[CondJob], outputs: u64) -> Vec<u8> {
    let padded = jobs.len().next_multiple_of(GROUP);
    let mut bytes = vec![0u8; padded * COND_JOB_BYTES];
    for (k, (j, b)) in jobs
        .iter()
        .zip(bytes.as_chunks_mut::<COND_JOB_BYTES>().0)
        .enumerate()
    {
        let dst = outputs + RECORD * k as u64;
        b[0..8].copy_from_slice(&j.vis.to_le_bytes());
        b[8..16].copy_from_slice(&j.src.to_le_bytes());
        b[16..24].copy_from_slice(&dst.to_le_bytes());
        b[24..32].copy_from_slice(&j.cond.const_a.to_le_bytes());
        b[32..40].copy_from_slice(&j.cond.const_b.to_le_bytes());
        for (idx, off) in j.cond.offsets_a.iter().enumerate() {
            b[40 + idx * 4..44 + idx * 4].copy_from_slice(&off.to_le_bytes());
        }
        for (idx, off) in j.cond.offsets_b.iter().enumerate() {
            b[56 + idx * 4..60 + idx * 4].copy_from_slice(&off.to_le_bytes());
        }
        let tail = [
            u32::from(j.cond.count_a),
            u32::from(j.cond.count_b),
            j.cond.mode,
            j.words[0],
            j.words[1],
            j.words[2],
            j.words[3],
            j.words[4],
            j.kind,
            0,
        ];
        for (w, v) in b[72..112].as_chunks_mut::<4>().0.iter_mut().zip(tail) {
            w.copy_from_slice(&v.to_le_bytes());
        }
    }
    bytes
}

/// Writes the jobs and their outputs into the scratch buffer and encodes the kernel. Gives the address of the output of
/// job 0. Job `k` writes at `RECORD * k` from there.
fn run(
    sh: &Shared<'_>,
    cb: &mut CommandBuffer,
    table: &ArgumentTable,
    scratch: &Scratch,
    jobs: &[Job],
) -> Result<u64, Skip> {
    let pipeline = sh
        .pipelines
        .host_compute("nxkit-indirect", MSL, "nxkit_indirect")?;
    let push = |b: &[u8]| {
        scratch
            .push(b)
            .map_err(|e| crate::buf_skip(format_args!("pre-pass {} bytes: {e:?}", b.len())))
    };
    let outputs = push(&vec![0u8; jobs.len() * RECORD as usize])?;
    let padded = jobs.len().next_multiple_of(GROUP);
    let mut bytes = vec![0u8; padded * JOB_BYTES];
    for (k, (j, b)) in jobs
        .iter()
        .zip(bytes.as_chunks_mut::<JOB_BYTES>().0)
        .enumerate()
    {
        let dst = outputs + RECORD * k as u64;
        b[0..8].copy_from_slice(&j.rec.to_le_bytes());
        b[8..16].copy_from_slice(&j.cnt.to_le_bytes());
        b[16..24].copy_from_slice(&dst.to_le_bytes());
        for (w, v) in b[24..40]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip([j.i, j.words, j.base, j.limit])
        {
            w.copy_from_slice(&v.to_le_bytes());
        }
    }
    let list = push(&bytes)?;
    let mut cp = cb.begin_compute_pass(table).map_err(|_| Skip::Encode)?;
    let groups = (padded / GROUP) as u32;
    for c in [
        ComputeCommand::WaitForEarlierPasses,
        ComputeCommand::SetPipeline(pipeline),
        ComputeCommand::SetBuffer {
            index: 0,
            gpu_addr: list,
        },
        ComputeCommand::DispatchThreadgroups {
            groups: [groups, 1, 1],
            threads: [GROUP as u32, 1, 1],
        },
    ] {
        cp.encode(&c).map_err(|_| Skip::Encode)?;
    }
    cp.end();
    Ok(outputs)
}

#[cfg(test)]
mod tests;
