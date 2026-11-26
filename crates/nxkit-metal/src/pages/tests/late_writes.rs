//! Tests of results one frame late: CPU writes while a frame runs.

use super::*;

#[test]
fn a_cpu_write_during_a_frame_survives_the_copy_back_of_the_frame_and_is_staged_at_the_next_flush()
{
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    pg.set_adaptive(true, false);
    let mut st = Collect::default();
    let _ = pg.flush_mirrored(2, &[], &[(m.addr(1) + 64, 128)], None, &mut st);
    mir.store(1, 70, 0xB1);
    mir.store(1, 130, 0xB2);
    let s0 = pg.stats();
    m.store(1, 70, 0x55);
    let s1 = pg.stats();
    assert_eq!(s1.late_writes - s0.late_writes, 1);
    assert_eq!(s1.late_owned - s0.late_owned, 1, "one 64-byte block");
    assert_eq!(s1.writer_waits, s0.writer_waits);
    assert_eq!(m.load(1, 70), 0x55, "the CPU reads what it wrote");
    pg.complete(2);
    wait_clean(pg, &m);
    assert_eq!(
        m.load(1, 70),
        0x55,
        "the copy-back of frame 2 left the block the CPU wrote"
    );
    assert_eq!(m.load(1, 130), 0xB2, "and copied the other block");
    assert!(pg.stats().late_kept > s1.late_kept);
    // The next flush stages the page with the CPU's byte.
    let mut st3 = Collect::default();
    let f = pg.flush_mirrored(3, &[], &[(m.addr(1) + 64, 128)], None, &mut st3);
    assert_eq!(f.staged, 1);
    let i = st3
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("the page is staged");
    assert_eq!(st3.pages[i].1[70], 0x55);
    assert_eq!(
        st3.skips[i].1, [0; 4],
        "the page is current in guest memory: nothing is skipped"
    );
    pg.complete(3);
    pg.set_adaptive(false, false);
    done(pg);
}

#[test]
fn a_write_staged_while_its_frame_runs_skips_the_blocks_only_the_frame_wrote_and_survives_the_frames_completion()
 {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    let mut st = Collect::default();
    let _ = pg.flush_mirrored(2, &[], &[(m.addr(1) + 64, 128)], None, &mut st);
    let mut c2 = late_start(pg, &mir, 2);
    // Frame 2 writes block 2 only. The CPU writes block 1.
    mir.store(1, 130, 0xB2);
    m.store(1, 70, 0x55);
    // The CPU runs ahead: the flush of frame 3 stages the page while frame 2 runs.
    let mut st3 = Collect::default();
    let f = pg.flush_mirrored(3, &[], &[(m.addr(1) + 64, 128)], None, &mut st3);
    let mut c3 = late_start(pg, &mir, 3);
    late_end(&mut c2, &mir);
    assert_eq!(f.emu_staged, 1);
    let i = st3
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("the page is staged");
    assert_eq!(st3.pages[i].1[70], 0x55);
    // Block 1 (bytes 64 to 127) is the CPU's and goes along. Block 2 (128 to 191) is the frame's and is skipped.
    assert_eq!(
        st3.skips[i].1[0], 0b100,
        "only the block of the frame is skipped"
    );
    // The frame completes while the page is tagged by frame 3: its copy-back keeps the staged block.
    pg.complete(2);
    assert_eq!(
        m.load(1, 70),
        0x55,
        "the copy-back of frame 2 does not take back what the CPU wrote before frame 3"
    );
    assert_eq!(m.load(1, 130), 0xB2, "the other block is frame 2's");
    // The staged block is not staged again at the next flush: a frame may have written it since.
    let mut st4 = Collect::default();
    let _ = pg.flush_mirrored(4, &[], &[(m.addr(1) + 64, 128)], None, &mut st4);
    assert!(st4.pages.iter().all(|p| p.0 != 1), "nothing new to stage");
    late_end(&mut c3, &mir);
    pg.complete(3);
    pg.complete(4);
    done(pg);
}

#[test]
fn an_atomic_add_during_a_frame_lands_on_guest_memory_and_is_staged() {
    use std::arch::asm;
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = emulating(4);
    pg.set_late(true);
    let mut st = Collect::default();
    // Frame 1's result: a counter of 10 at byte 64.
    let _ = pg.flush_mirrored(1, &[], &[(m.addr(1) + 64, 64)], None, &mut st);
    for (k, b) in 10u32.to_le_bytes().iter().enumerate() {
        mir.store(1, 64 + k as u64, *b);
    }
    pg.complete(1);
    assert_eq!(m.load(1, 64), 10);
    let _ = pg.flush_mirrored(2, &[], &[(m.addr(1) + 64, 64)], None, &mut st);
    let mut c2 = late_start(pg, &mir, 2);
    for (k, b) in 99u32.to_le_bytes().iter().enumerate() {
        mir.store(1, 64 + k as u64, *b);
    }
    let s0 = pg.stats();
    let p = m.addr(1) as usize + 64;
    let old: u32;
    // SAFETY: inside page 1 of the test mapping, aligned for a word.
    unsafe {
        asm!(".arch_extension lse", "ldaddal {one:w}, {old:w}, [{p}]",
            p = in(reg) p, one = in(reg) 5u32, old = out(reg) old, options(nostack));
    }
    assert_eq!(
        old, 10,
        "the atomic operates on the last completed frame's value"
    );
    let s1 = pg.stats();
    assert_eq!(s1.late_writes - s0.late_writes, 1);
    assert_eq!(s1.writer_waits, s0.writer_waits);
    let mut st3 = Collect::default();
    let _ = pg.flush_mirrored(3, &[], &[(m.addr(1) + 64, 64)], None, &mut st3);
    let i = st3
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("the page is staged");
    assert_eq!(
        u32::from_le_bytes(st3.pages[i].1[64..68].try_into().unwrap()),
        15,
        "the sum is staged"
    );
    assert_eq!(st3.skips[i].1[0] & 0b10, 0, "the block is not skipped");
    late_end(&mut c2, &mir);
    pg.complete(2);
    // Frame 2 changed the counter: its value is the result the reader gets one frame late, and the CPU's sum is in
    // the mirror for frame 3.
    let read = u32::from_le_bytes([m.load(1, 64), m.load(1, 65), m.load(1, 66), m.load(1, 67)]);
    assert_eq!(read, 99, "the frame's 99 wins for the reader");
    pg.complete(3);
    done(pg);
}

#[test]
fn with_late_results_off_an_access_to_written_bytes_waits() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    pg.set_late(false);
    let mut st = Collect::default();
    let _ = pg.flush_mirrored(2, &[], &[(m.addr(1) + 64, 128)], None, &mut st);
    mir.store(1, 70, 0xB1);
    let s0 = pg.stats();
    let base = m.addr(1) as usize;
    let reader = std::thread::spawn(move || {
        // SAFETY: inside page 1 of the test mapping, which outlives the joined thread.
        unsafe { ((base + 70) as *const u8).read_volatile() }
    });
    std::thread::sleep(std::time::Duration::from_millis(60));
    assert!(!reader.is_finished());
    pg.complete(2);
    assert_eq!(reader.join().unwrap(), 0xB1);
    assert_eq!(pg.stats().late_reads, s0.late_reads);
    done(pg);
}

#[test]
fn rw_open_false_sharing_page_promotes_and_stages_without_traps() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    pg.set_adaptive(false, true);
    pg.set_read_open(true);
    let written = [(m.addr(1) + 64, 128)];

    // Flush 1 & 2 with emulated stores on false-sharing byte 512 (block 8, outside stale blocks 1..=2).
    let mut st1 = Collect::default();
    let _ = pg.flush_mirrored(2, &[], &written, None, &mut st1);
    let mut c2 = late_start(pg, &mir, 2);
    m.store(1, 512, 0x11);
    late_end(&mut c2, &mir);
    pg.complete(2);

    let mut st2 = Collect::default();
    let _ = pg.flush_mirrored(3, &[], &written, None, &mut st2);
    let mut c3 = late_start(pg, &mir, 3);
    m.store(1, 512, 0x22);
    late_end(&mut c3, &mir);
    pg.complete(3);

    // Second consecutive emulated-store flush promotes page 1 to `rw_open` (`PROT_RW`).
    let mut st3 = Collect::default();
    let _ = pg.flush_mirrored(4, &[], &written, None, &mut st3);
    let mut c4 = late_start(pg, &mir, 4);
    mir.store(1, 70, 0xC1);
    let emu0: u64 = pg.stats().emulated.iter().sum();
    m.store(1, 512, 0x33);
    m.store(1, 520, 0x44);
    let emu1: u64 = pg.stats().emulated.iter().sum();
    assert_eq!(
        emu1 - emu0,
        0,
        "rw_open false-sharing stores take zero traps"
    );
    late_end(&mut c4, &mir);
    pg.complete(4);
    assert_eq!(
        m.load(1, 70),
        0xC1,
        "late_copy_back still updates stale blocks"
    );
    assert_eq!(m.load(1, 512), 0x33, "CPU false-sharing byte preserved");

    // Frame 4 completed cleanly, so blocks 1..=2 are no longer stale at Flush 5;
    // when Flush 6 overlaps in-flight Frame 5, stale blocks 1..=2 ARE skipped.
    let mut st4 = Collect::default();
    let f4 = pg.flush_mirrored(5, &[], &written, None, &mut st4);
    assert!(f4.hot_staged >= 1);
    let idx = st4
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("rw_open page staged");
    assert_eq!(st4.pages[idx].1[512], 0x33);
    assert_eq!(st4.pages[idx].1[520], 0x44);
    assert_eq!(st4.skips[idx].1[0] & 0b110, 0, "completed stale cleared");

    let mut c5 = late_start(pg, &mir, 5);
    m.store(1, 512, 0x34);
    let mut st5 = Collect::default();
    let _ = pg.flush_mirrored(6, &[], &written, None, &mut st5);
    let idx5 = st5
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("overlapping rw_open page staged");
    assert_eq!(
        st5.skips[idx5].1[0] & 0b110,
        0b110,
        "in-flight stale blocks skipped"
    );
    late_end(&mut c5, &mir);
    pg.complete(5);
    pg.complete(6);
    pg.set_read_open(false);
    pg.set_adaptive(false, false);
    done(pg);
}

#[test]
fn rw_open_page_demotes_when_cpu_touches_stale_block() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    pg.set_adaptive(false, true);
    pg.set_read_open(true);
    let written = [(m.addr(1) + 64, 128)];

    for (f, v) in [(2u64, 0x11u8), (3, 0x22)] {
        let mut st = Collect::default();
        let _ = pg.flush_mirrored(f, &[], &written, None, &mut st);
        let mut c = late_start(pg, &mir, f);
        m.store(1, 512, v);
        late_end(&mut c, &mir);
        pg.complete(f);
    }

    // (a) CPU writes stale block 1 (byte 70) + false-sharing byte 512 while GPU does NOT change block 1.
    // Page stays rw_open (PROT_RW), stages CPU writes, and subsequent stores take 0 traps.
    let mut st3 = Collect::default();
    let _ = pg.flush_mirrored(4, &[], &written, None, &mut st3);
    let mut c4 = late_start(pg, &mir, 4);
    mir.store(1, 130, 0xD2);
    m.store(1, 70, 0x77);
    m.store(1, 512, 0x55);
    late_end(&mut c4, &mir);
    pg.complete(4);
    assert_eq!(m.load(1, 70), 0x77, "CPU stale-block write kept");
    assert_eq!(m.load(1, 130), 0xD2, "GPU stale-block write copied");

    let mut st4 = Collect::default();
    let _ = pg.flush_mirrored(5, &[], &written, None, &mut st4);
    assert!(
        st4.pages
            .iter()
            .any(|p| p.0 == 1 && p.1[70] == 0x77 && p.1[512] == 0x55)
    );
    let emu0: u64 = pg.stats().emulated.iter().sum();
    m.store(1, 512, 0x56);
    assert_eq!(
        pg.stats().emulated.iter().sum::<u64>() - emu0,
        0,
        "no won change -> stays rw_open"
    );

    // (b) GPU DOES change CPU-written stale block 1 in frame 5 (`start != end` => `won != 0`).
    // Next flush detects dirty_stale, then completion lifts CPU bytes to carry and demotes to PROT_READ.
    let mut c5 = late_start(pg, &mir, 5);
    let _ = pg.flush_mirrored(6, &[], &written, None, &mut Collect::default());
    m.store(1, 70, 0x78);
    mir.store(1, 70, 0xE1);
    late_end(&mut c5, &mir);
    pg.complete(5);
    assert_eq!(m.load(1, 70), 0xE1, "GPU won in guest memory");

    let mut st6 = Collect::default();
    let _ = pg.flush_mirrored(7, &[], &written, None, &mut st6);
    assert!(
        st6.pages.iter().any(|p| p.0 == 1 && p.1[70] == 0x78),
        "carried CPU write staged"
    );
    let emu2: u64 = pg.stats().emulated.iter().sum();
    m.store(1, 512, 0x66);
    let emu3: u64 = pg.stats().emulated.iter().sum();
    assert!(emu3 > emu2, "won != 0 demoted page back to PROT_READ");
    pg.complete(6);
    pg.complete(7);
    pg.set_read_open(false);
    pg.set_adaptive(false, false);
    done(pg);
}

#[test]
fn rw_open_page_demotes_cleanly_when_no_longer_gpu_written() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    pg.set_adaptive(false, true);
    pg.set_read_open(true);
    let written = [(m.addr(1) + 64, 128)];

    for (f, v) in [(2u64, 0x11u8), (3, 0x22)] {
        let mut st = Collect::default();
        let _ = pg.flush_mirrored(f, &[], &written, None, &mut st);
        let mut c = late_start(pg, &mir, f);
        m.store(1, 512, v);
        late_end(&mut c, &mir);
        pg.complete(f);
    }

    let mut st3 = Collect::default();
    let _ = pg.flush_mirrored(4, &[], &written, None, &mut st3);
    let mut c4 = late_start(pg, &mir, 4);
    m.store(1, 512, 0x88);
    late_end(&mut c4, &mir);
    pg.complete(4);

    // Flush 5 does NOT write page 1: it must demote, stage 0x88, and re-protect the page.
    let mut st4 = Collect::default();
    let f4 = pg.flush_mirrored(5, &[], &[], None, &mut st4);
    assert!(f4.demoted >= 1);
    let idx = st4
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("final rw_open changes staged on demote");
    assert_eq!(st4.pages[idx].1[512], 0x88);
    pg.complete(5);
    pg.set_read_open(false);
    pg.set_adaptive(false, false);
    done(pg);
}

#[test]
fn overlapping_frames_clear_completed_stale_blocks_on_late_copy_back() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0x91, 0x92);
    // Warm up snapshot so page 1 participates in late_copy_back.
    let w1 = [(m.addr(1) + 64, 64)];
    let _ = pg.flush_mirrored(2, &[], &w1, None, &mut Collect::default());
    let mut c2 = late_start(pg, &mir, 2);
    late_end(&mut c2, &mir);
    pg.complete(2);

    // Frame 3 writes block 1 (64..128), overlapping Frame 4 which writes block 2 (128..192).
    let _ = pg.flush_mirrored(
        3,
        &[],
        &[(m.addr(1) + 64, 64)],
        None,
        &mut Collect::default(),
    );
    let mut c3 = late_start(pg, &mir, 3);
    mir.store(1, 70, 0xB3);
    let _ = pg.flush_mirrored(
        4,
        &[],
        &[(m.addr(1) + 128, 64)],
        None,
        &mut Collect::default(),
    );
    let mut c4 = late_start(pg, &mir, 4);
    mir.store(1, 130, 0xC4);
    late_end(&mut c3, &mir);

    // Frame 3 completes while Frame 4 is still in flight: block 1 must be cleared from `stale`,
    // while block 2 remains stale.
    pg.complete(3);
    assert_eq!(m.load(1, 70), 0xB3, "completed frame 3 block 1 copied");

    let s0 = pg.stats();
    m.store(1, 70, 0xD5);
    let s1 = pg.stats();
    assert_eq!(
        s1.late_writes - s0.late_writes,
        0,
        "block 1 was cleared from stale on frame 3 completion"
    );

    m.store(1, 130, 0xD6);
    let s2 = pg.stats();
    assert!(
        s2.late_writes - s1.late_writes > 0,
        "block 2 from in-flight frame 4 must remain stale"
    );

    late_end(&mut c4, &mir);
    pg.complete(4);
    let mut st = Collect::default();
    let _ = pg.flush_mirrored(5, &[], &[], None, &mut st);
    assert!(st.pages.iter().any(|p| p.0 == 1 && p.1[70] == 0xD5));
    done(pg);
}

#[test]
fn a_cpu_reset_after_completion_is_staged_without_exposing_the_reset_to_late_readers() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    // Frame 2 modifies byte 70 (0xA1 -> 0x10) and leaves byte 130 (0xA2) unchanged, then completes.
    let _ = pg.flush_mirrored(
        2,
        &[],
        &[(m.addr(1) + 64, 128)],
        None,
        &mut Collect::default(),
    );
    let mut c2 = late_start(pg, &mir, 2);
    mir.store(1, 70, 0x10);
    late_end(&mut c2, &mir);
    pg.complete(2);

    // CPU reads frame 2's result, resets byte 70 to 0, and writes 0x77 into unchanged block 2 (byte 130).
    assert_eq!(m.load(1, 70), 0x10);
    m.store(1, 70, 0);
    m.store(1, 130, 0x77);

    // Frame 3 flushes: the reset (0) and 0x77 must go into the mirror, while guest memory restores GPU's 0x10
    // so late readers during frame 3 do not read the CPU's reset.
    let mut st3 = Collect::default();
    let _ = pg.flush_mirrored(3, &[], &[(m.addr(1) + 64, 128)], None, &mut st3);
    let idx = st3
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("page 1 staged for frame 3");
    assert_eq!(st3.pages[idx].1[70], 0, "reset staged for GPU frame 3");
    assert_eq!(st3.pages[idx].1[130], 0x77, "non-GPU-changed block staged");
    assert_eq!(
        st3.skips[idx].1[0] & 0b110,
        0,
        "completed blocks not skipped"
    );

    let s0 = pg.stats();
    assert_eq!(
        m.load(1, 70),
        0x10,
        "late reader while frame 3 runs sees frame 2's 0x10, not CPU's reset 0"
    );
    assert_eq!(pg.stats().writer_waits, s0.writer_waits, "no writer wait");
    assert_eq!(m.load(1, 130), 0x77, "non-GPU-changed block keeps 0x77");

    let mut c3 = late_start(pg, &mir, 3);
    mir.store(1, 70, 0x20);
    late_end(&mut c3, &mir);
    pg.complete(3);
    assert_eq!(m.load(1, 70), 0x20);
    done(pg);
}

#[test]
fn a_cpu_reset_after_completion_on_rw_open_syncs_shadow_and_does_not_restage_restored_bytes() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0xA1, 0xA2);
    pg.set_adaptive(false, true);
    pg.set_read_open(true);
    let written = [(m.addr(1) + 64, 128)];

    for (f, v) in [(2u64, 0x11u8), (3, 0x22)] {
        let _ = pg.flush_mirrored(f, &[], &written, None, &mut Collect::default());
        let mut c = late_start(pg, &mir, f);
        m.store(1, 512, v);
        late_end(&mut c, &mir);
        pg.complete(f);
    }

    // Frame 4 runs on rw_open page and changes byte 70 to 0x10, then completes before CPU resets byte 70.
    let _ = pg.flush_mirrored(4, &[], &written, None, &mut Collect::default());
    let mut c4 = late_start(pg, &mir, 4);
    mir.store(1, 70, 0x10);
    late_end(&mut c4, &mir);
    pg.complete(4);

    assert_eq!(m.load(1, 70), 0x10);
    m.store(1, 70, 0);

    // Frame 5 flushes: rw_open stages 0 for frame 5, then restores 0x10 in guest & shadow.
    let mut st5 = Collect::default();
    let _ = pg.flush_mirrored(5, &[], &written, None, &mut st5);
    let idx = st5
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("rw_open page staged");
    assert_eq!(st5.pages[idx].1[70], 0, "reset staged for frame 5");
    assert_eq!(
        m.load(1, 70),
        0x10,
        "late reader sees 0x10 while frame 5 runs"
    );

    let mut c5 = late_start(pg, &mir, 5);
    late_end(&mut c5, &mir);
    pg.complete(5);

    // Without another CPU write, flush 6 must NOT re-stage the restored 0x10 because shadow is in sync.
    let mut st6 = Collect::default();
    let _ = pg.flush_mirrored(6, &[], &written, None, &mut st6);
    assert!(
        st6.pages.iter().all(|p| p.0 != 1),
        "restored 0x10 must not be re-staged as a CPU write"
    );

    pg.complete(6);
    pg.set_read_open(false);
    pg.set_adaptive(false, false);
    done(pg);
}

#[test]
fn stage_carry_preserves_partial_cpu_write_in_same_block() {
    let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let (m, mir, pg) = late_page(0x31, 0x32);
    let written = [(m.addr(1) + 64, 64)];

    // Teach page 1 as a late page via frame 2.
    let _ = pg.flush_mirrored(2, &[], &written, None, &mut Collect::default());
    let mut c2 = late_start(pg, &mir, 2);
    late_end(&mut c2, &mir);
    pg.complete(2);

    // Frame 3 runs; while it runs, CPU writes two offsets inside block 1 (bytes 68 and 72).
    let _ = pg.flush_mirrored(3, &[], &written, None, &mut Collect::default());
    let mut c3 = late_start(pg, &mir, 3);
    mir.store(1, 68, 0xAA);
    mir.store(1, 72, 0xBB);
    m.store(1, 68, 0x11);
    m.store(1, 72, 0x22);
    late_end(&mut c3, &mir);
    // Completion of frame 3 sees GPU changed block 1, overwrites guest with frame 3's bytes and stashes carry.
    pg.complete(3);
    assert_eq!(m.load(1, 68), 0xAA);
    assert_eq!(m.load(1, 72), 0xBB);

    // Before frame 4 flushes, CPU modifies ONLY another offset in the same 64B block (byte 76), leaving 68/72 untouched.
    m.store(1, 76, 0x33);
    let mut st4 = Collect::default();
    let _ = pg.flush_mirrored(4, &[], &written, None, &mut st4);
    let idx = st4
        .pages
        .iter()
        .position(|p| p.0 == 1)
        .expect("page 1 staged");
    assert_eq!(
        st4.pages[idx].1[68], 0x11,
        "earlier carried CPU byte restored"
    );
    assert_eq!(
        st4.pages[idx].1[72], 0x22,
        "earlier carried CPU byte restored"
    );
    assert_eq!(
        st4.pages[idx].1[76], 0x33,
        "newer partial CPU byte preserved"
    );

    pg.complete(4);
    done(pg);
}
