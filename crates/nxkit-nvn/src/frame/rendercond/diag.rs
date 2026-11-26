//! Render condition diagnostics and GPU address helper.

use super::*;

/// The commands of `frame` that a false condition leaves out, by kind: draws, clears, dispatches, copies.
pub fn disabled_counts(frame: &Frame) -> [u32; 4] {
    let mut n = [0u32; 4];
    for r in &frame.disabled {
        for i in r.clone() {
            if !frame.skips(i) {
                continue;
            }
            let k = match frame.cmds.get(i as usize) {
                Some(Cmd::Draw { .. }) => 0,
                Some(Cmd::ClearColor { .. } | Cmd::ClearDepthStencil { .. }) => 1,
                Some(Cmd::Dispatch { .. }) => 2,
                _ => 3,
            };
            n[k] += 1;
        }
    }
    n
}

/// The little-endian word at a GPU address, None outside guest memory.
pub(crate) fn read_gpu_u64<S: GpuSpace, M: GuestMem>(space: &S, mem: &M, addr: u64) -> Option<u64> {
    let host = space.host_address(GpuAddr::new(addr).ok()?, 8).ok()?;
    let mut b = [0u8; 8];
    mem.read(host, &mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// `NXKIT_DIAG=rendercond`: one line for a frame that set a condition or left something out (the first 300 such
/// frames, then every 60th), and the first evaluations with the counters they read. The backend adds a line for the
/// emulated counters it wrote (`nxkit-gpu`, `queries`).
pub(crate) fn frame_diag(rc: &mut RenderConds, frame: &Frame) {
    static ON: nxkit_abi::live::Memo = nxkit_abi::live::Memo::new();
    if !ON.flag(|| crate::diag::on("rendercond")) {
        return;
    }
    static FRAMES: AtomicU32 = AtomicU32::new(0);
    static LINES: AtomicU32 = AtomicU32::new(0);
    let s = rc.take_stats();
    let n = FRAMES.fetch_add(1, Ordering::Relaxed);
    for (mode, addr, a, b, r) in rc.evaluations.drain(..) {
        if crate::diag::first(&LINES, 200) {
            eprintln!(
                "DIAG rendercond frame {n}: if {} at {addr:#x}: counters {a:x?} and {b:x?} -> {}",
                if mode == RENDER_IF_EQUAL {
                    "equal"
                } else if mode == RENDER_IF_NOT_EQUAL {
                    "not equal"
                } else {
                    "unknown mode"
                },
                r.map_or("unknown, renders", |r| if r {
                    "renders"
                } else {
                    "left out"
                })
            );
        }
    }
    rc.evaluations.shrink_to(64);
    if s.is_empty() && frame.disabled.is_empty() {
        return;
    }
    static SHOWN: AtomicU32 = AtomicU32::new(0);
    let shown = SHOWN.fetch_add(1, Ordering::Relaxed);
    if shown < 300 || shown.is_multiple_of(60) {
        let [d, c, x, k] = disabled_counts(frame);
        let kinds: Vec<String> = s
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(k, n)| format!("{k:#x}:{n}"))
            .collect();
        let forced = if s.forced == [0; 3] {
            String::new()
        } else {
            format!(
                " | forced decisions rcflag {} rcunk {} rcall {}",
                s.forced[0], s.forced[1], s.forced[2]
            )
        };
        eprintln!(
            "DIAG rendercond frame {n}: enable {} (off {}), conditions true {} false {} unknown {} (read landed {} stale {} by place {}), reports value {} emulated {} (zero {}) gpu-only {} by type [{}] in {} slots | left out: draws {d} clears {c} dispatches {x} copies {k} in {} ranges of {} commands{forced}",
            s.enables,
            s.disables,
            s.true_,
            s.false_,
            s.unknown,
            s.landed,
            s.stale,
            s.placed,
            s.values,
            s.emulated,
            s.zero,
            s.counters,
            kinds.join(" "),
            frame.counters.slots,
            frame.disabled.len(),
            frame.cmds.len()
        );
    }
}
