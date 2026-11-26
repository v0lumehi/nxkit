//! Multi-hop ARM64 `b` branch relay planning over safe unused executable caves inside `[0, p)`.

use std::collections::{BTreeMap, HashSet};

use crate::error::RewriteError;
use crate::scan::{code_ranges, fde_ranges};

/// Maximum backward byte displacement of ARM64 `b` (`-128 MiB`).
pub(crate) const B_REACH_BACK: i64 = 1 << 27;
/// Maximum forward byte displacement of ARM64 `b` (`+128 MiB - 4`).
pub(crate) const B_REACH_FWD: i64 = (1 << 27) - 4;
const MAX_HOPS: usize = 64;

const NOP: u32 = 0xd503_201f;
const BRK_0: u32 = 0xd420_0000;
const BRK_1: u32 = 0xd420_0020;

/// True if a single ARM64 `b` instruction at `from` reaches `to`.
pub(crate) fn b_reaches(from: i64, to: i64) -> bool {
    let off = to - from;
    (-B_REACH_BACK..=B_REACH_FWD).contains(&off) && off % 4 == 0
}

fn u32_at(img: &[u8], off: i64) -> u32 {
    usize::try_from(off)
        .ok()
        .and_then(|o| img.get(o..o + 4))
        .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn sext(v: u32, bits: u32) -> i64 {
    let s = i64::from(v);
    let m = 1i64 << (bits - 1);
    (s ^ m) - m
}

fn is_terminator_or_call(w: u32) -> bool {
    (w & 0xffff_fc1f) == 0xd65f_0000 // ret
        || (w >> 26) == 0x05 // b
        || (w >> 26) == 0x25 // bl
        || (w & 0xffff_fc1f) == 0xd61f_0000 // br
        || (w >> 21) == 0x6a1 // brk
        || (w >> 21) == 0x6a2 // hlt
        || (w >> 16) == 0x0000 // udf
        || w == 0xd69f_03e0 // eret
}

fn is_padding_word(w: u32, allow_zero: bool) -> bool {
    matches!(w, NOP | BRK_0 | BRK_1) || (allow_zero && w == 0)
}

fn in_sorted_ranges(ranges: &[(i64, i64)], addr: i64) -> bool {
    let idx = ranges.partition_point(|(s, _)| *s <= addr);
    idx > 0 && addr < ranges[idx - 1].1
}

/// Disjoint 4-byte aligned executable cave intervals `[start, end)` available for `b` relays.
#[derive(Default)]
pub(crate) struct CaveMap {
    free: BTreeMap<i64, i64>,
}

impl CaveMap {
    /// Collects safe unused executable padding in `[0, data_off_page)`:
    /// 1. The inter-segment gap `[text_end, ro_off)` when `ro_off > text_end`.
    /// 2. Alignment padding (`nop`, `brk #0/#1`, `0`) in inter-FDE gaps of `.text` that follow a
    ///    function terminator/call, lie outside `data_blocks`, are not in `excluded`, and are not
    ///    targeted by any branch, `adr`, or PC-relative literal load.
    pub(crate) fn build(
        img: &[u8],
        text_end: i64,
        ro_off: usize,
        data_off: usize,
        data_blocks: &[(i64, i64)],
        excluded: &HashSet<i64>,
    ) -> Result<CaveMap, RewriteError> {
        let mut map = CaveMap::default();
        let text_limit = if data_off > 0 {
            (data_off as i64 / 0x4000) * 0x4000
        } else {
            text_end
        };

        // 1. Inter-segment padding between .text and .rodata (inside r-x __NX_<TAG>_TEXT).
        let te = (text_end + 3) & !3;
        let re = if ro_off > 0 {
            (ro_off as i64).min(text_limit) & !3
        } else {
            te
        };
        if re > te {
            map.push_filtered(te, re, |a| !excluded.contains(&a));
        }

        // 2. Inter-FDE padding gaps inside .text.
        let raw_fdes = fde_ranges(img)?;
        if raw_fdes.is_empty() {
            return Ok(map);
        }
        let mut fdes: Vec<(i64, i64)> = Vec::new();
        for (s, e) in raw_fdes {
            let (s, e) = (s.max(0), e.min(text_end));
            if s >= e {
                continue;
            }
            match fdes.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => fdes.push((s, e)),
            }
        }
        if fdes.is_empty() {
            return Ok(map);
        }

        let targeted = collect_gap_targets(img, text_end, data_blocks, &fdes);
        let mut data_sorted = data_blocks.to_vec();
        data_sorted.sort_unstable();

        for i in 0..fdes.len() {
            let prev_start = fdes[i].0;
            let prev_end = fdes[i].1;
            let next_start = if i + 1 < fdes.len() {
                fdes[i + 1].0
            } else {
                text_end
            };
            let g_start = (prev_end + 3) & !3;
            let g_end = next_start & !3;
            if g_end <= g_start {
                continue;
            }
            let last_addr = (prev_end - 4) & !3;
            if last_addr < prev_start || !is_terminator_or_call(u32_at(img, last_addr)) {
                continue;
            }
            let allow_zero = g_end - g_start < 0x100;
            map.push_filtered(g_start, g_end, |a| {
                is_padding_word(u32_at(img, a), allow_zero)
                    && !in_sorted_ranges(&data_sorted, a)
                    && !excluded.contains(&a)
                    && !targeted.contains(&a)
            });
        }

        Ok(map)
    }

    fn push_filtered(&mut self, start: i64, end: i64, mut ok: impl FnMut(i64) -> bool) {
        let mut run_start: Option<i64> = None;
        let mut a = start;
        while a < end {
            if ok(a) {
                if run_start.is_none() {
                    run_start = Some(a);
                }
            } else if let Some(rs) = run_start.take() {
                self.insert_range(rs, a);
            }
            a += 4;
        }
        if let Some(rs) = run_start {
            self.insert_range(rs, end);
        }
    }

    fn insert_range(&mut self, mut start: i64, mut end: i64) {
        if start >= end {
            return;
        }
        if let Some((&ps, &pe)) = self.free.range(..=start).next_back()
            && pe >= start
        {
            start = ps;
            end = end.max(pe);
            self.free.remove(&ps);
        }
        while let Some((&ns, &ne)) = self.free.range(start..).next() {
            if ns > end {
                break;
            }
            end = end.max(ne);
            self.free.remove(&ns);
        }
        self.free.insert(start, end);
    }

    /// Largest available 4-byte-aligned address `r` with `lo_excl < r <= hi_incl`.
    fn last_in(&self, lo_excl: i64, hi_incl: i64) -> Option<i64> {
        let (&s, &e) = self.free.range(..=hi_incl).next_back()?;
        let r = hi_incl.min(e - 4) & !3;
        (r >= s && r > lo_excl).then_some(r)
    }

    /// Smallest available 4-byte-aligned address `r` with `lo_incl <= r < hi_excl`.
    fn first_in(&self, lo_incl: i64, hi_excl: i64) -> Option<i64> {
        let aligned_lo = (lo_incl + 3) & !3;
        if aligned_lo >= hi_excl {
            return None;
        }
        if let Some((&_s, &e)) = self.free.range(..=aligned_lo).next_back()
            && e - 4 >= aligned_lo
        {
            return Some(aligned_lo);
        }
        let (&s, _) = self.free.range((aligned_lo + 1)..hi_excl).next()?;
        Some(s)
    }

    fn remove_word(&mut self, addr: i64) {
        let Some((&s, &e)) = self.free.range(..=addr).next_back() else {
            return;
        };
        if addr < s || addr + 4 > e {
            return;
        }
        self.free.remove(&s);
        if addr > s {
            self.free.insert(s, addr);
        }
        if addr + 4 < e {
            self.free.insert(addr + 4, e);
        }
    }

    /// Finds the relay chain `[r_1, ..., r_k]` from `from` to `to` without mutating the map.
    pub(crate) fn find_hops(&self, from: i64, to: i64) -> Option<Vec<i64>> {
        let mut hops = Vec::new();
        let mut cur = from;
        while !b_reaches(cur, to) {
            if hops.len() >= MAX_HOPS {
                return None;
            }
            let next = if cur < to {
                self.last_in(cur, cur + B_REACH_FWD)?
            } else {
                self.first_in(cur - B_REACH_BACK, cur)?
            };
            hops.push(next);
            cur = next;
        }
        Some(hops)
    }

    /// Number of relay hops needed from `from` to `to`, or `None` if unreachable.
    pub(crate) fn hop_count(&self, from: i64, to: i64) -> Option<usize> {
        self.find_hops(from, to).map(|h| h.len())
    }

    /// Reserves and returns the relay chain `[r_1, ..., r_k]` from `from` to `to`.
    pub(crate) fn alloc_chain(&mut self, from: i64, to: i64) -> Result<Vec<i64>, RewriteError> {
        let hops = self.find_hops(from, to).ok_or_else(|| {
            RewriteError::Unsupported(format!(
                "ARM64 ±128 MiB branch reach exceeded with no relay space between {from:#x} and {to:#x}"
            ))
        })?;
        for &h in &hops {
            self.remove_word(h);
        }
        Ok(hops)
    }
}

/// Collects PC-relative targets inside `[0, text_end)` that fall outside all FDE ranges.
fn collect_gap_targets(
    img: &[u8],
    text_end: i64,
    data_blocks: &[(i64, i64)],
    fdes: &[(i64, i64)],
) -> HashSet<i64> {
    let mut out = HashSet::new();
    let mut mark = |addr: i64, words: i64| {
        let base = addr & !3;
        for k in 0..words {
            let a = base + 4 * k;
            if a >= 0 && a < text_end && !in_sorted_ranges(fdes, a) {
                out.insert(a);
            }
        }
    };
    for (cs, ce) in code_ranges(data_blocks, text_end) {
        let mut a = (cs + 3) & !3;
        let end = ce & !3;
        while a < end {
            let w = u32_at(img, a);
            if w & 0x7c00_0000 == 0x1400_0000 {
                mark(a + sext(w & 0x3ff_ffff, 26) * 4, 1);
            } else if w & 0xff00_0010 == 0x5400_0000 || w & 0x7e00_0000 == 0x3400_0000 {
                mark(a + sext((w >> 5) & 0x7_ffff, 19) * 4, 1);
            } else if w & 0x7e00_0000 == 0x3600_0000 {
                mark(a + sext((w >> 5) & 0x3fff, 14) * 4, 1);
            } else if w & 0x9f00_0000 == 0x1000_0000 {
                let imm = sext((((w >> 5) & 0x7_ffff) << 2) | ((w >> 29) & 3), 21);
                mark(a + imm, 2);
            } else if w & 0x3b00_0000 == 0x1800_0000 {
                mark(a + sext((w >> 5) & 0x7_ffff, 19) * 4, 4);
            }
            a += 4;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_hop_allocation_bridges_more_than_256_mib() {
        let mut map = CaveMap::default();
        map.insert_range(0x6f0_0000, 0x6f0_0020);
        map.insert_range(0xe00_0000, 0xe00_0020);
        map.insert_range(0x1500_0000, 0x1500_0020);

        let hops = map.alloc_chain(-0x100_0000, 0x1a00_0000).unwrap();
        assert_eq!(hops, vec![0x6f0_001c, 0xe00_001c, 0x1500_001c]);
        let mut prev = -0x100_0000;
        for &h in &hops {
            assert!(b_reaches(prev, h));
            prev = h;
        }
        assert!(b_reaches(prev, 0x1a00_0000));

        let rev = map.alloc_chain(0x1a00_0000, -0x100_0000).unwrap();
        assert_eq!(rev, vec![0x1500_0000, 0xe00_0000, 0x6f0_0000]);
        let mut prev = 0x1a00_0000;
        for &h in &rev {
            assert!(b_reaches(prev, h));
            prev = h;
        }
        assert!(b_reaches(prev, -0x100_0000));
    }

    #[test]
    fn missing_cave_returns_unsupported() {
        let mut map = CaveMap::default();
        map.insert_range(0x100_0000, 0x100_0010);
        let err = map.alloc_chain(0, 0x1400_0000).unwrap_err();
        assert!(matches!(err, RewriteError::Unsupported(_)));
    }
}
