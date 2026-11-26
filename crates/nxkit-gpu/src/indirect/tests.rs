//! The jobs of the pre-pass against pools whose real addresses differ from the guest addresses.

use std::sync::Arc;

use super::*;
use crate::geometry::word_loc;
use crate::pools::{Backing, PagePins, PoolSet};

struct Shifted(u64);

impl Backing for Shifted {
    fn real_base(&self) -> u64 {
        self.0
    }
}

struct NoPins;

impl PagePins for NoPins {
    fn pin(&self, _start: u64, _end: u64) {}
    fn unpin(&self, _start: u64, _end: u64) {}
}

const SHIFT: u64 = 0x1_0000_0000;

/// One pool at `0x4000` of `0x1000` bytes.
fn set() -> PoolSet<Shifted> {
    let set = PoolSet::new(
        0,
        0,
        |h: u64, _l: u64| Ok(Shifted(h + SHIFT)),
        Arc::new(NoPins),
    );
    set.register(0x4000, 0x1000, 0).unwrap();
    set
}

fn multi(indirect: u64, count: u64, max: u32) -> DrawArgs {
    DrawArgs::MultiIndirectCount {
        index: None,
        indirect,
        count,
        max,
        stride: 16,
    }
}

#[test]
fn the_jobs_of_a_multi_draw_hold_real_addresses() {
    let set = set();
    let snap = set.current();
    let at = |a, b| {
        word_loc(&*snap, a, b)
            .map(|l| l.real)
            .map_err(|_| Skip::Buffer)
    };
    let mut jobs = Vec::new();
    let r = jobs_of(&at, &multi(0x4100, 0x4000, 3), false, &mut jobs);
    assert!(matches!(r, Some(Ok(()))));
    assert_eq!(jobs.len(), 3);
    for (i, j) in jobs.iter().enumerate() {
        assert_eq!(j.rec, 0x4100 + 16 * i as u64 + SHIFT);
        assert_eq!(j.cnt, 0x4000 + SHIFT);
        assert_eq!((j.i, j.words), (i as u32, 4));
    }
}

#[test]
fn a_record_outside_the_pool_skips_the_draw_and_leaves_no_job() {
    let set = set();
    let snap = set.current();
    let at = |a, b| {
        word_loc(&*snap, a, b)
            .map(|l| l.real)
            .map_err(|_| Skip::Buffer)
    };
    let mut jobs = Vec::new();
    // record 1 starts at 0x4ff8 and ends past the pool
    let r = jobs_of(&at, &multi(0x4fe8, 0x4000, 3), false, &mut jobs);
    assert!(matches!(r, Some(Err(Skip::Buffer))));
    assert!(jobs.is_empty());
    // the count word lies in no pool
    let r = jobs_of(&at, &multi(0x4100, 0x9000, 2), false, &mut jobs);
    assert!(matches!(r, Some(Err(Skip::Buffer))));
    assert!(jobs.is_empty());
}

#[test]
fn the_grids_of_an_indirect_mesh_draw_read_the_real_arguments() {
    let set = set();
    let snap = set.current();
    let at = |a, b| {
        word_loc(&*snap, a, b)
            .map(|l| l.real)
            .map_err(|_| Skip::Buffer)
    };
    let mut jobs = Vec::new();
    let args = DrawArgs::MeshTasksIndirect { indirect: 0x4200 };
    assert!(matches!(jobs_of(&at, &args, true, &mut jobs), Some(Ok(()))));
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].rec, jobs[0].cnt), (0x4200 + SHIFT, 0x4200 + SHIFT));
    jobs.clear();
    assert!(matches!(
        jobs_of(&at, &args, false, &mut jobs),
        Some(Ok(()))
    ));
    assert_eq!(jobs.len(), INDIRECT_SPLITS as usize);
    assert!(jobs.iter().all(|j| j.rec == 0x4200 + SHIFT));
    let far = DrawArgs::MeshTasksIndirect { indirect: 0x9000 };
    assert!(matches!(
        jobs_of(&at, &far, false, &mut jobs),
        Some(Err(Skip::Buffer))
    ));
}

#[test]
fn a_draw_with_no_copy_has_no_jobs() {
    let set = set();
    let snap = set.current();
    let at = |a, b| {
        word_loc(&*snap, a, b)
            .map(|l| l.real)
            .map_err(|_| Skip::Buffer)
    };
    let mut jobs = Vec::new();
    let args = DrawArgs::ArraysIndirect { indirect: 0x4000 };
    assert!(jobs_of(&at, &args, false, &mut jobs).is_none());
}

#[test]
fn indirect_resolver_can_override_guest_pool_address() {
    let mut jobs = Vec::new();
    let args = DrawArgs::MeshTasksIndirect {
        indirect: 0x10_0000,
    };
    let at = |addr: u64, bytes: u64| {
        assert_eq!((addr, bytes), (0x10_0000, 8));
        Ok(0xdead_beef)
    };
    assert!(matches!(jobs_of(&at, &args, true, &mut jobs), Some(Ok(()))));
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].rec, jobs[0].cnt), (0xdead_beef, 0xdead_beef));
}

#[test]
fn gpu_cond_kernel_writes_or_zeroes_instance_count_from_vis_buffer() {
    use crate::pipelines::Pipelines;
    use crate::queries::ResolvedGpuCond;
    use nxkit_metal::{ArchiveConfig, ArgumentTable, Context, DataBuffer, FrameRing, Scratch};

    let Ok(ctx) = Context::system_default() else {
        return;
    };
    let Ok(pipelines) = Pipelines::new(&ctx, &ArchiveConfig::default()) else {
        return;
    };
    let pipeline = pipelines
        .host_compute("nxkit-cond-draw", COND_MSL, "nxkit_cond_draw")
        .unwrap();
    let vis = DataBuffer::new(&ctx, 64).unwrap();
    let out = DataBuffer::new(&ctx, 256).unwrap();
    let mesh_src = DataBuffer::new(&ctx, 64).unwrap();
    // Vis slot 0 = 0 (occluded), vis slot 1 = 42 (visible).
    vis.write(0, &0u64.to_le_bytes()).unwrap();
    vis.write(8, &42u64.to_le_bytes()).unwrap();
    // Indirect mesh source: count=500, first=7.
    mesh_src.write(0, &500u32.to_le_bytes()).unwrap();
    mesh_src.write(4, &7u32.to_le_bytes()).unwrap();
    let table = ArgumentTable::new(&ctx, 31, 0).unwrap();
    let scratch = Scratch::new(&ctx, 1 << 16).unwrap();
    let mut ring = FrameRing::new(&ctx).unwrap();
    let mut frame = ring.begin().unwrap();
    let cond_occluded = ResolvedGpuCond {
        mode: 1,
        offsets_a: [0, 0, 0, 0],
        count_a: 1,
        const_a: 0,
        offsets_b: [0; 4],
        count_b: 0,
        const_b: 0,
    };
    let cond_visible = ResolvedGpuCond {
        mode: 1,
        offsets_a: [1, 0, 0, 0],
        count_a: 1,
        const_a: 0,
        offsets_b: [0; 4],
        count_b: 0,
        const_b: 0,
    };
    let jobs = [
        // 0: Occluded Arrays draw -> instanceCount 0.
        CondJob {
            vis: vis.gpu_address(),
            src: 0,
            cond: cond_occluded,
            words: [120, 3, 10, 7, 0],
            kind: 0,
        },
        // 1: Visible Indexed draw -> keeps instances=2 and base_instance=5.
        CondJob {
            vis: vis.gpu_address(),
            src: 0,
            cond: cond_visible,
            words: [60, 2, 0, (-4i32) as u32, 5],
            kind: 1,
        },
        // 2: Occluded direct mesh draw -> threadgroups {0, 1, 1}, first 3.
        CondJob {
            vis: vis.gpu_address(),
            src: 0,
            cond: cond_occluded,
            words: [200, 3, 0, MAX_GRID, 0],
            kind: 2,
        },
        // 3: Visible direct mesh draw -> threadgroups {200, 1, 1}, first 3.
        CondJob {
            vis: vis.gpu_address(),
            src: 0,
            cond: cond_visible,
            words: [200, 3, 0, MAX_GRID, 0],
            kind: 2,
        },
        // 4: Occluded indirect mesh draw -> threadgroups {0, 1, 1}, first 7.
        CondJob {
            vis: vis.gpu_address(),
            src: mesh_src.gpu_address(),
            cond: cond_occluded,
            words: [0, 0, 0, MAX_GRID, 0],
            kind: 2,
        },
        // 5: Visible indirect mesh draw -> threadgroups {500, 1, 1}, first 7.
        CondJob {
            vis: vis.gpu_address(),
            src: mesh_src.gpu_address(),
            cond: cond_visible,
            words: [0, 0, 0, MAX_GRID, 0],
            kind: 2,
        },
    ];
    let job_bytes = encode_cond_jobs(&jobs, out.gpu_address());
    let list = scratch.push(&job_bytes).unwrap();
    let mut cp = frame.commands().begin_compute_pass(&table).unwrap();
    cp.encode(&ComputeCommand::SetPipeline(pipeline)).unwrap();
    cp.encode(&ComputeCommand::SetBuffer {
        index: 0,
        gpu_addr: list,
    })
    .unwrap();
    cp.encode(&ComputeCommand::DispatchThreadgroups {
        groups: [1, 1, 1],
        threads: [GROUP as u32, 1, 1],
    })
    .unwrap();
    cp.end();
    ctx.commit_residency();
    frame.submit().unwrap().wait().unwrap();

    let read_words = |idx: u64| -> [u32; 5] {
        let mut raw = [0u8; 20];
        out.read(idx * RECORD, &mut raw).unwrap();
        std::array::from_fn(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()))
    };
    assert_eq!(read_words(0), [120, 0, 10, 7, 0]);
    assert_eq!(read_words(1), [60, 2, 0, (-4i32) as u32, 5]);
    assert_eq!(read_words(2), [0, 1, 1, 0, 3]);
    assert_eq!(read_words(3), [200, 1, 1, 0, 3]);
    assert_eq!(read_words(4), [0, 1, 1, 0, 7]);
    assert_eq!(read_words(5), [500, 1, 1, 0, 7]);
}
