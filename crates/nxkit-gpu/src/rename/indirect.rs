//! Indirect argument snapshots for CPU-written indirect draw and dispatch buffers.

use super::*;

pub(super) fn has_indirect(detail: &FrameDetail) -> bool {
    detail.draws.iter().any(|d| d.args.is_indirect())
        || detail.dispatches.iter().any(|d| d.indirect.is_some())
}

/// Captures exact indirect-argument records `(guest_addr, bytes, payload)` from CPU-written buffers so large
/// read-only indirect buffers (e.g. 15 MiB mesh draw args) stay protected against CPU overwrite even under overlap.
pub(super) fn collect_indirects(
    detail: &FrameDetail,
    ranges: &[CapRange],
    copy_w: &[(u64, u64)],
    gpu_cands: &[(u64, u64)],
    read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Vec<(u64, u32, Vec<u8>)> {
    use crate::detail::{DrawArgs, MAX_MULTI};
    if !has_indirect(detail) {
        return Vec::new();
    }
    let mut keys: Vec<(u64, u32)> = Vec::new();
    for d in &detail.draws {
        match d.args {
            DrawArgs::ArraysIndirect { indirect } => keys.push((indirect, 16)),
            DrawArgs::IndexedIndirect { indirect, .. } => keys.push((indirect, 20)),
            DrawArgs::MeshTasksIndirect { indirect } => keys.push((indirect, 8)),
            DrawArgs::MultiIndirectCount {
                index,
                indirect,
                count,
                max,
                stride,
            } => {
                keys.push((count, 4));
                if stride.is_multiple_of(4) {
                    let bytes = if index.is_some() { 20u32 } else { 16u32 };
                    let n = max.min(MAX_MULTI);
                    for i in 0..n {
                        keys.push((
                            indirect.wrapping_add(u64::from(i).wrapping_mul(stride)),
                            bytes,
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    for d in &detail.dispatches {
        if let Some(a) = d.indirect {
            keys.push((a, 12));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    let mut out = Vec::new();
    for (addr, len) in keys {
        if addr == 0 || len == 0 {
            continue;
        }
        let end = addr.saturating_add(u64::from(len));
        let span = (addr, end);
        let in_range = ranges.iter().any(|r| {
            r.kind != Kind::Probe
                && r.addr <= addr
                && end <= r.addr.saturating_add(u64::from(r.len))
        });
        if in_range
            || copy_w.iter().any(|&w| overlaps(span, w))
            || gpu_cands.iter().any(|&g| overlaps(span, g))
        {
            continue;
        }
        let mut buf = vec![0u8; len as usize];
        if read(addr, &mut buf) {
            out.push((addr, len, buf));
        }
    }
    out
}
