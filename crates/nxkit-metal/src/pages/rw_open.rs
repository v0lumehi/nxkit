//! Step 10: writable `read_open` false-sharing pages.
//!
//! When a `read_open` (`snap != 0`) GPU-written page receives repeated emulated stores to `!stale` blocks and has no
//! recent CPU writes into `stale` blocks, leaving it `PROT_RW` with a shadow copy removes kernel trap overhead while
//! preserving `stale`-block isolation.

use super::*;

/// Consecutive flushes with emulated stores before a `read_open` page stays `PROT_RW`.
const EMU_HOT_AFTER: u8 = 2;
/// Flushes a page stays out of `rw_open` after a GPU frame overwrites a CPU-written stale block (`won != 0`).
const LATE_WRITER_COOLDOWN: u32 = 8;

/// True while CPU writes to `stale` blocks were recently overwritten on page `i` (`won != 0`), so it briefly stays `PROT_READ`.
pub(super) fn late_writer_active(i: &PageInfo) -> bool {
    let until = i.late_writer.load(Ordering::Relaxed);
    if until == 0 {
        return false;
    }
    let cur = (SHARED.flushed.load(Ordering::Relaxed) & FRAME_MASK) as u32;
    let diff = until.wrapping_sub(cur) & (FRAME_MASK as u32);
    diff > 0 && diff <= LATE_WRITER_COOLDOWN + 1
}

fn any_bit(m: &Mask) -> bool {
    m.iter().any(|&w| w != 0)
}

fn all_bits(m: &Mask) -> bool {
    m.iter().all(|&w| w == u64::MAX)
}

fn diff_masks(live: &[u8], shadow: &[u8], stale: &Mask) -> (bool, bool, Mask) {
    let blocks = (live.len() as u64 / BLOCK) as usize;
    let mut dirty_stale = false;
    let mut dirty_clean = false;
    let mut dirty_stale_mask: Mask = [0; 4];
    for b in 0..blocks {
        let r = b * BLOCK as usize..(b + 1) * BLOCK as usize;
        if live[r.clone()] == shadow[r] {
            continue;
        }
        if has(stale, b) {
            dirty_stale = true;
            dirty_stale_mask[b / 64] |= 1 << (b % 64);
        } else {
            dirty_clean = true;
        }
    }
    (dirty_stale, dirty_clean, dirty_stale_mask)
}

fn read_page_copy(ptr: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    // SAFETY: `ptr` points to a readable mapping (`alias` or `guest` while readable) of one full host page `len`.
    let live = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    out.copy_from_slice(live);
    out
}

fn protect_back(pg: u32, snap_nonzero: bool, timed: bool) {
    let prot = if snap_nonzero { PROT_READ } else { PROT_NONE };
    protect_pages(pg, pg + 1, prot, timed);
}

/// Diffs and stages pages currently in `cc.rw_open`. Returns `(staged, demoted)`.
/// Pages leaving `runs_w` or losing `read_open` eligibility are protected and demoted cleanly.
pub(super) fn stage_rw_open(
    cc: &mut Ctrl,
    runs_w: &[(u32, u32)],
    alias: u64,
    eligible_mode: bool,
    stager: &mut dyn Stager,
) -> (u32, u32) {
    if cc.rw_open.is_empty() {
        return (0, 0);
    }
    let p = page() as usize;
    let mut staged = 0u32;
    let mut demoted = 0u32;
    let mut remove: Vec<(u32, bool)> = Vec::new();
    let keys: Vec<u32> = cc.rw_open.keys().copied().collect();

    for pg in keys {
        let cur = word(pg as usize).map_or(FREE, |w| w.load(Ordering::Acquire));
        let state = cur & STATE_MASK;
        let info_p = info(pg as usize);
        let in_w = in_runs(runs_w, pg);
        let can_stay = eligible_mode
            && in_w
            && state == GPU
            && alias != 0
            && info_p.is_some_and(|i| {
                i.snap.load(Ordering::Acquire) != 0
                    && !late_writer_active(i)
                    && !all_bits(&i.stale())
            });

        if !can_stay {
            let snap_nz = info_p.is_some_and(|i| i.snap.load(Ordering::Acquire) != 0);
            if state == GPU {
                protect_back(pg, snap_nz, true);
            }
            let Some(h) = cc.rw_open.remove(&pg) else {
                continue;
            };
            if let Some(x) = cc.emu_streak.get_mut(pg as usize) {
                *x = 0;
            }
            if state != FREE
                && let Some(i) = info_p
            {
                let i_emu_zeroed = i.emu.swap(0, Ordering::AcqRel);
                let src = if alias != 0 {
                    Some(alias + u64::from(pg) * p as u64)
                } else {
                    matches!(state, CLEAN | DIRTY | OPEN)
                        .then(|| SHARED.base.load(Ordering::Acquire) + u64::from(pg) * p as u64)
                };
                if let Some(src_ptr) = src {
                    let live = read_page_copy(src_ptr, p);
                    let need_stage = h.shadow.is_empty()
                        || live[..] != h.shadow[..]
                        || cc.carry.contains_key(&pg)
                        || (i_emu_zeroed & 1 != 0);
                    if need_stage {
                        let stale = i.stale();
                        if !h.shadow.is_empty() {
                            let (ds, _, dsm) = diff_masks(&live, &h.shadow, &stale);
                            if ds {
                                mark_stale_own(i, &dsm);
                            }
                        }
                        stage_carry(stager, &mut cc.carry, pg, &live, i.take_skip());
                        staged += 1;
                    }
                }
            }
            demoted += 1;
            continue;
        }

        let Some(i) = info_p else {
            remove.push((pg, false));
            continue;
        };
        let Some(h) = cc.rw_open.get_mut(&pg) else {
            continue;
        };
        let ptr = alias + u64::from(pg) * p as u64;
        if h.shadow.is_empty() {
            continue;
        }
        let live = read_page_copy(ptr, p);
        let stale = i.stale();
        let (dirty_stale, dirty_clean, dirty_stale_mask) = diff_masks(&live, &h.shadow, &stale);
        if dirty_stale {
            mark_stale_own(i, &dirty_stale_mask);
        }
        let carry_has = cc.carry.contains_key(&pg);
        if dirty_stale || dirty_clean || carry_has {
            h.quiet = 0;
            let blocks = p / BLOCK as usize;
            for b in 0..blocks {
                if !has(&stale, b) || has(&dirty_stale_mask, b) {
                    let r = b * BLOCK as usize..(b + 1) * BLOCK as usize;
                    h.shadow[r.clone()].copy_from_slice(&live[r]);
                }
            }
            i.emu.store(0, Ordering::Release);
            stage_carry(stager, &mut cc.carry, pg, &live, i.take_skip());
            staged += 1;
        } else {
            h.quiet = h.quiet.saturating_add(1);
            if h.quiet >= HOT_QUIET {
                remove.push((pg, true));
            }
        }
    }

    for (pg, prot_read) in remove {
        cc.rw_open.remove(&pg);
        if let Some(x) = cc.emu_streak.get_mut(pg as usize) {
            *x = 0;
        }
        if prot_read && let Some(i) = info(pg as usize) {
            let snap_nz = i.snap.load(Ordering::Acquire) != 0;
            protect_back(pg, snap_nz, true);
        }
        demoted += 1;
    }
    (staged, demoted)
}

pub(super) fn demote_won(pg: u32, i: &PageInfo, cc_rw_open: &mut BTreeMap<u32, Hot>) {
    let until = ((SHARED
        .flushed
        .load(Ordering::Relaxed)
        .wrapping_add(u64::from(LATE_WRITER_COOLDOWN)))
        & FRAME_MASK) as u32;
    i.late_writer.store(until.max(1), Ordering::Relaxed);
    if cc_rw_open.remove(&pg).is_some() {
        protect_pages(pg, pg + 1, PROT_READ, false);
        if i.emu.fetch_or(1, Ordering::AcqRel) & 1 == 0 {
            log_dirty(pg as usize);
        }
    }
}

fn mark_stale_own(i: &PageInfo, dirty_stale_mask: &Mask) {
    let mut owned = 0u64;
    for (k, &m) in dirty_stale_mask.iter().enumerate() {
        if m != 0 {
            i.own[k].fetch_or(m, Ordering::AcqRel);
            owned += u64::from(m.count_ones());
        }
    }
    if owned > 0 {
        i.own_frame.fetch_max(
            (SHARED.flushed.load(Ordering::Acquire) & FRAME_MASK) as u32,
            Ordering::AcqRel,
        );
        SHARED.late_owned.fetch_add(owned, Ordering::Relaxed);
    }
}

/// Updates consecutive emulated-store streaks for pages in `runs_w`, returning newly promoted pages.
pub(super) fn update_emu_streaks(
    cc: &mut Ctrl,
    emu: &[u32],
    runs_w: &[(u32, u32)],
    eligible_mode: bool,
) -> Vec<u32> {
    if !eligible_mode {
        cc.prev_emu.clear();
        return Vec::new();
    }
    for &pg in &cc.prev_emu {
        if emu.binary_search(&pg).is_err()
            && let Some(x) = cc.emu_streak.get_mut(pg as usize)
        {
            *x = 0;
        }
    }
    let mut promoted = Vec::new();
    for &pg in emu {
        if cc.rw_open.contains_key(&pg) {
            continue;
        }
        let Some(x) = cc.emu_streak.get_mut(pg as usize) else {
            continue;
        };
        if !in_runs(runs_w, pg) {
            *x = 0;
            continue;
        }
        let Some(i) = info(pg as usize) else {
            *x = 0;
            continue;
        };
        if i.snap.load(Ordering::Acquire) == 0 || late_writer_active(i) || all_bits(&i.stale()) {
            *x = 0;
            continue;
        }
        *x = x.saturating_add(1);
        if *x >= EMU_HOT_AFTER {
            cc.rw_open.insert(
                pg,
                Hot {
                    shadow: Vec::new(),
                    quiet: 0,
                },
            );
            promoted.push(pg);
        }
    }
    cc.prev_emu.clone_from(&emu.to_vec());
    promoted
}

/// Stages any modified `rw_open` pages during `flush_late`.
pub(super) fn stage_rw_open_late(
    cc: &mut Ctrl,
    alias: u64,
    stager: &mut dyn Stager,
    staged_out: &mut Vec<u32>,
) {
    if cc.rw_open.is_empty() || alias == 0 {
        return;
    }
    let p = page() as usize;
    let keys: Vec<u32> = cc.rw_open.keys().copied().collect();
    for pg in keys {
        let Some(i) = info(pg as usize) else {
            continue;
        };
        let Some(h) = cc.rw_open.get_mut(&pg) else {
            continue;
        };
        if h.shadow.is_empty() {
            continue;
        }
        let ptr = alias + u64::from(pg) * p as u64;
        let live = read_page_copy(ptr, p);
        let stale = i.stale();
        let (dirty_stale, dirty_clean, dirty_stale_mask) = diff_masks(&live, &h.shadow, &stale);
        if dirty_stale {
            mark_stale_own(i, &dirty_stale_mask);
        } else if !dirty_clean && !cc.carry.contains_key(&pg) {
            continue;
        }
        let blocks = p / BLOCK as usize;
        for b in 0..blocks {
            if !has(&stale, b) || has(&dirty_stale_mask, b) {
                let r = b * BLOCK as usize..(b + 1) * BLOCK as usize;
                h.shadow[r.clone()].copy_from_slice(&live[r]);
            }
        }
        i.emu.store(0, Ordering::Release);
        stage_carry(stager, &mut cc.carry, pg, &live, i.take_skip());
        staged_out.push(pg);
    }
}

/// Before `late_copy_back` copies `stale` blocks `m` into `guest`, checks whether CPU writes touched any block in `m`
/// on an `rw_open` page. If so, marks `i.own` (`late_copy_back` handles carry/demote only when GPU `won != 0`).
pub(super) fn check_late_copy(
    pg: u32,
    i: &PageInfo,
    m: &Mask,
    guest: u64,
    cc_rw_open: &mut BTreeMap<u32, Hot>,
) {
    let Some(h) = cc_rw_open.get(&pg) else {
        return;
    };
    if h.shadow.is_empty() {
        return;
    }
    let p = page() as usize;
    let blocks = p / BLOCK as usize;
    // SAFETY: `guest` points to a readable mapping of page `pg` claimed in `SYNC`.
    let live = unsafe { std::slice::from_raw_parts(guest as *const u8, p) };
    let mut dsm: Mask = [0; 4];
    for b in 0..blocks {
        if !has(m, b) {
            continue;
        }
        let r = b * BLOCK as usize..(b + 1) * BLOCK as usize;
        if live[r.clone()] != h.shadow[r] {
            dsm[b / 64] |= 1 << (b % 64);
        }
    }
    if any_bit(&dsm) {
        mark_stale_own(i, &dsm);
    }
}

/// Updates `h.shadow` for `copy` blocks after `late_copy_back` copies them from `src` to `guest`.
pub(super) fn sync_shadow_after_copy(
    pg: u32,
    src: u64,
    copy: &Mask,
    cc_rw_open: &mut BTreeMap<u32, Hot>,
) {
    let Some(h) = cc_rw_open.get_mut(&pg) else {
        return;
    };
    if h.shadow.is_empty() {
        return;
    }
    copy_blocks(src, h.shadow.as_mut_ptr() as u64, copy);
}

/// Called when `late_copy_back` drops the snapshot (`revoke_read`) on page `pg`.
pub(super) fn on_revoke(pg: u32, i: &PageInfo, cc_rw_open: &mut BTreeMap<u32, Hot>) {
    if cc_rw_open.remove(&pg).is_some() && i.emu.fetch_or(1, Ordering::AcqRel) & 1 == 0 {
        log_dirty(pg as usize);
    }
}

/// Syncs `stale` blocks into `h.shadow` when `flush_mirrored` Step 3 runs `copy_stale` on an `rw_open` page.
pub(super) fn sync_shadow_stale(
    pg: u32,
    alias: u64,
    stale: &Mask,
    cc_rw_open: &mut BTreeMap<u32, Hot>,
) {
    let Some(h) = cc_rw_open.get_mut(&pg) else {
        return;
    };
    if h.shadow.is_empty() {
        return;
    }
    copy_blocks(
        alias + u64::from(pg) * page(),
        h.shadow.as_mut_ptr() as u64,
        stale,
    );
}

/// Protection for page `pg` in `flush_mirrored` Step 3 when `read_open` is active.
pub(super) fn step3_prot(pg: u32, cc: &mut Ctrl) -> c_int {
    let i = info(pg as usize);
    let snap = i.is_some_and(|i| i.snap.load(Ordering::Acquire) != 0);
    let rw_ok = snap
        && i.is_some_and(|i| !late_writer_active(i) && !all_bits(&i.stale()))
        && cc.rw_open.get(&pg).is_some_and(|h| !h.shadow.is_empty());
    if rw_ok {
        PROT_RW
    } else {
        if cc.rw_open.remove(&pg).is_some()
            && let Some(x) = cc.emu_streak.get_mut(pg as usize)
        {
            *x = 0;
        }
        if snap { PROT_READ } else { PROT_NONE }
    }
}
