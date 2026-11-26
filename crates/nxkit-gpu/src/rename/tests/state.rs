//! Tests of the state over frames: learning, checks, watched ranges and snapshots.

use super::*;

#[test]
fn a_sum_that_the_cpu_resets_is_learned_and_then_renamed() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = state_with_sum();
    let mut renamed_at = None;
    for fence in 1..=300u64 {
        let f = play(&mut st, &g, fence, 3, true);
        if !f.renamed.is_empty() {
            renamed_at.get_or_insert(fence);
            assert_eq!(f.renamed[0].addr, SUM);
            assert_eq!(f.renamed[0].start.len(), 0x40);
            // The frame is renamed: the capture holds the reset word.
            assert_eq!(&f.renamed[0].start[..4], &[0, 0, 0, 0]);
        }
    }
    // Only one frame in LEARN_PERIOD is judged, at the flush after it: the flush after the eighth clean frame already
    // reads a copy.
    assert_eq!(renamed_at, Some(LEARN_PERIOD * u64::from(LEARN_FRAMES) + 1));
    assert_eq!(st.cands[&SUM].learn.class(), Class::Rename);
}

#[test]
fn a_flush_that_overtakes_the_wait_is_not_judged() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = state_with_sum();
    // The CPU flushes the next frame before the wait for the one before: `post` is not read in time.
    for fence in 1..=200u64 {
        g.set(SUM, &[0]);
        let f = st.flush(
            fence,
            &[(SUM, 0x40)],
            &|r| vec![false; r.len()],
            &|| Vec::new(),
            &|a, b| g.read(a, b),
        );
        assert!(f.renamed.is_empty());
        g.set(SUM, &[5]);
        // The wait for the frame before the newest: the newest flush has not finished.
        st.waited(fence - 1, fence, &|a, b| g.read(a, b));
    }
    assert_eq!(st.cands[&SUM].learn.class(), Class::Learning);
}

#[test]
fn state_that_the_gpu_keeps_is_never_renamed() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = state_with_sum();
    for fence in 1..=200u64 {
        // The CPU does not reset the word: the GPU keeps adding.
        let f = play(&mut st, &g, fence, 3, false);
        // A judged frame may give the range a probe copy. Nothing else is renamed.
        assert!(f.renamed.iter().all(|r| r.kind == Kind::Probe));
    }
    assert_eq!(st.cands[&SUM].learn.class(), Class::Never);
}

#[test]
fn a_renamed_range_that_turns_into_state_is_demoted_by_the_check() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = state_with_sum();
    st.fast_only = true;
    let mut fence = 0;
    while st.cands[&SUM].learn.class() != Class::Rename {
        fence += 1;
        play(&mut st, &g, fence, 3, true);
    }
    // Renamed frames: the host copies and writes back. Here the range is only counted until the check is due.
    let mut checked = false;
    for _ in 0..RECHECK_EVERY + 40 {
        fence += 1;
        // From now on the CPU keeps what the GPU wrote.
        let f = play(&mut st, &g, fence, 3, false);
        if st.cands[&SUM].learn.sampling() {
            checked = true;
        }
        if f.sampling && st.cands[&SUM].learn.class() == Class::Never {
            break;
        }
    }
    assert!(checked);
    assert_eq!(st.cands[&SUM].learn.class(), Class::Never);
}

#[test]
fn a_range_that_no_frame_binds_is_forgotten_in_time() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = state_with_sum();
    for fence in 1..=3600u64 {
        st.flush(
            fence,
            &[],
            &|r| vec![false; r.len()],
            &|| Vec::new(),
            &|a, b| g.read(a, b),
        );
    }
    assert!(st.cands.is_empty());
}

#[test]
fn a_binding_larger_than_the_range_grows_it_and_starts_again() {
    let mut st = state_with_sum();
    st.cands.get_mut(&SUM).unwrap().learn = renamed();
    st.candidate(SUM, 0x80);
    assert_eq!(st.cands[&SUM].size, 0x80);
    assert_eq!(st.cands[&SUM].learn.class(), Class::Learning);
}

#[test]
fn a_watched_range_reports_what_the_cpu_wrote_while_its_frame_ran() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = State {
        watch: true,
        ..State::default()
    };
    st.watch_read_only(SUM, 0x40);
    // The frame that is judged: the 16th flush.
    for fence in 1..LEARN_PERIOD {
        st.flush(
            fence,
            &[(SUM, 0x40)],
            &|r| vec![false; r.len()],
            &|| Vec::new(),
            &|a, b| g.read(a, b),
        );
    }
    let fence = LEARN_PERIOD;
    let f = st.flush(
        fence,
        &[(SUM, 0x40)],
        &|r| vec![false; r.len()],
        &|| Vec::new(),
        &|a, b| g.read(a, b),
    );
    // A read-only range is measured when the wait begins and needs no full wait.
    assert!(f.watching && !f.sampling);
    // The wait begins, the CPU has written word 1 meanwhile.
    g.set(SUM + 4, &[7]);
    st.before_wait(fence, &|a, b| g.read(a, b));
    let e = &st.inflight[&SUM];
    assert_eq!((e.frames, e.hit, e.words), (1, 1, 1));
    assert!(e.ro);
}

#[test]
fn a_read_only_range_that_the_cpu_writes_in_flight_becomes_a_snapshot() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = State::default();
    st.watch_read_only(SUM, 0x40);
    run_to(&mut st, &g, LEARN_PERIOD - 1);
    judged_frame(&mut st, &g, LEARN_PERIOD, true);
    // One judged frame with a CPU write in flight is enough.
    assert_eq!(st.cands[&SUM].learn.class(), Class::Snapshot);
    // The next flush announces it and copies the range.
    let f = st.flush(
        LEARN_PERIOD + 1,
        &[(SUM, 0x40)],
        &|r| vec![false; r.len()],
        &|| Vec::new(),
        &|a, b| g.read(a, b),
    );
    assert!(f.news.iter().any(|n| n.2 == Class::Snapshot));
    assert_eq!(f.renamed.len(), 1);
    assert_eq!(f.renamed[0].kind, Kind::Snapshot);
    assert_eq!(f.snapshot_bytes, 0x40);
}

#[test]
fn a_read_only_range_that_the_cpu_leaves_alone_is_watched_and_then_rests() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = State {
        fast_only: true,
        ..State::default()
    };
    st.watch_read_only(SUM, 0x40);
    let mut fence = 0;
    // A range that the CPU writes only in some frames is not given up for one quiet frame.
    for _ in 0..SETTLE_PROBES {
        fence += LEARN_PERIOD;
        run_to(&mut st, &g, fence - 1);
        let f = judged_frame(&mut st, &g, fence, false);
        // A read-only range needs no full wait.
        assert!(f.watching && !f.sampling);
    }
    // After that it rests.
    fence += LEARN_PERIOD;
    run_to(&mut st, &g, fence - 1);
    let f = judged_frame(&mut st, &g, fence, false);
    assert!(!f.watching);
    assert_eq!(st.cands[&SUM].learn.class(), Class::Watch);
}

#[test]
fn a_snapshot_that_a_shader_writes_is_never() {
    let mut st = State::default();
    st.watch_read_only(SUM, 0x40);
    st.cands.get_mut(&SUM).unwrap().learn.set(Class::Snapshot);
    assert!(st.is_snapshot(SUM));
    st.candidate(SUM, 0x40);
    assert_eq!(st.cands[&SUM].learn.class(), Class::Never);
    assert!(!st.is_snapshot(SUM));
}

#[test]
fn a_snapshot_that_a_copy_writes_is_never() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = State::default();
    st.watch_read_only(SUM, 0x40);
    st.cands.get_mut(&SUM).unwrap().learn.set(Class::Snapshot);
    let f = st.flush(
        1,
        &[(SUM, 0x40)],
        &|r| vec![false; r.len()],
        &|| vec![(SUM + 8, SUM + 16)],
        &|a, b| g.read(a, b),
    );
    assert!(f.renamed.is_empty());
    assert_eq!(st.cands[&SUM].learn.class(), Class::Never);
}

#[test]
fn snapshots_over_the_budget_of_a_frame_are_left_out() {
    let g = Guest(RefCell::new(vec![0; 0x400]));
    let mut st = State::default();
    let big = (SNAPSHOT_BUDGET / 2 + 1) as u32;
    // Reads past the small guest fail. The state only needs the sizes, so the reader says yes.
    let read = |_: u64, b: &mut [u8]| {
        b.fill(0);
        true
    };
    for k in 0..2u64 {
        let a = 0x1000_0000 * (k + 1);
        st.watch_read_only(a, big);
        st.cands.get_mut(&a).unwrap().learn.set(Class::Snapshot);
    }
    let _ = &g;
    let bound = [(0x1000_0000, big), (0x2000_0000, big)];
    let f = st.flush(1, &bound, &|r| vec![false; r.len()], &|| Vec::new(), &read);
    assert_eq!(f.renamed.len(), 1);
    assert_eq!(f.over_budget, 1);
    assert_eq!(f.snapshot_bytes, u64::from(big));
}

#[test]
fn the_masks_of_the_cpu_and_the_gpu_are_found_from_the_three_copies() {
    // Word 0: the GPU wrote it after the wait began. Word 1: the CPU wrote it in flight. Word 2: nobody.
    let pre = words(&[0, 1, 2]);
    let atwait = words(&[0, 5, 2]);
    assert_eq!(cpu_mask(&pre, &atwait), vec![0b010]);
    assert_eq!(
        gpu_words(&words(&[9, 5, 2]), &words(&[0, 5, 2])),
        vec![0b001]
    );
    let mut m = vec![0b100];
    assert!(union_mask(&mut m, &[0b001, 0b1]));
    assert_eq!(m, vec![0b101, 1]);
    assert!(!union_mask(&mut m, &[0b100]));
    assert!(is_empty_mask(&[0, 0]) && !is_empty_mask(&[0, 2]));
}

#[test]
fn reprobe_exceeds_slow_period() {
    const {
        assert!(REPROBE > SLOW_PERIOD);
    }
}

#[test]
fn probe_cursor_rotates_touched_when_watch_budget_is_exceeded() {
    let mut st = State {
        fast_only: true,
        ..State::default()
    };
    let half = (WATCH_BUDGET / 2) as u32;
    let addrs = [0x1000_0000u64, 0x2000_0000, 0x3000_0000];
    for a in addrs {
        st.watch_read_only(a, half);
    }
    let bound: Vec<(u64, u32)> = addrs.iter().map(|&a| (a, half)).collect();
    let read = |_: u64, b: &mut [u8]| {
        b.fill(0);
        true
    };
    run_to_bound(&mut st, &bound, LEARN_PERIOD - 1, &read);
    for a in addrs {
        let c = st.cands.get_mut(&a).unwrap();
        c.sample = None;
        c.rest_until = 0;
    }
    st.flush(
        LEARN_PERIOD,
        &bound,
        &|r| vec![false; r.len()],
        &|| Vec::new(),
        &read,
    );
    assert!(st.cands[&addrs[0]].sample.is_some());
    assert!(st.cands[&addrs[1]].sample.is_some());
    assert!(st.cands[&addrs[2]].sample.is_none());
    assert_eq!(st.probe_cursor, addrs[2]);
    run_to_bound(&mut st, &bound, 2 * LEARN_PERIOD - 1, &read);
    for a in addrs {
        let c = st.cands.get_mut(&a).unwrap();
        c.sample = None;
        c.rest_until = 0;
    }
    st.flush(
        2 * LEARN_PERIOD,
        &bound,
        &|r| vec![false; r.len()],
        &|| Vec::new(),
        &read,
    );
    assert!(st.cands[&addrs[2]].sample.is_some());
}

fn run_to_bound(
    st: &mut State,
    bound: &[(u64, u32)],
    last: u64,
    read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    for fence in st.frame + 1..=last {
        st.flush(
            fence,
            bound,
            &|r| vec![false; r.len()],
            &|| Vec::new(),
            read,
        );
    }
}

#[test]
fn task_and_mesh_sbuf_bindings_are_adopted_and_rw_tracked() {
    use crate::detail::{DrawArgs, Stage, UboDetail};
    let Ok(ctx) = nxkit_metal::Context::system_default() else {
        return;
    };
    let mem = Arc::new(nxkit_metal::GuestMemory::allocate(&ctx, 1 << 22).unwrap());
    let guest = nxkit_metal::GuestSpace::over(&mem).unwrap();
    let base = guest.address_map().gpu_base();
    let renamer = Renamer::new(&ctx, guest);
    let mut detail = FrameDetail::default();
    let mut draw = crate::walk::empty_draw();
    draw.program.task = Some(Arc::from("task_prog"));
    draw.program.mesh = Some(Arc::from("mesh_prog"));
    draw.args = DrawArgs::MeshTasksIndirect {
        indirect: base + 0x1000,
    };
    draw.sbufs.push(UboDetail {
        stage: Stage::Task,
        slot: 0,
        address: base + 0x2000,
        size: 0x100,
        snapshot: None,
    });
    draw.sbufs.push(UboDetail {
        stage: Stage::Mesh,
        slot: 1,
        address: base + 0x4000,
        size: 0x100,
        snapshot: None,
    });
    detail.draws.push(Arc::new(draw));
    renamer.note_rw_with(&detail, None, &|key, vertex, slot| {
        assert!(!vertex);
        matches!((key, slot), ("mesh_prog", 1))
    });
    let mut st = lock(&renamer.state);
    assert_eq!(st.cands[&(base + 0x2000)].learn.class(), Class::Watch);
    assert_eq!(st.cands[&(base + 0x4000)].learn.class(), Class::Learning);
    let (bindings, bound) = st.bindings(&detail);
    assert_eq!(bindings.len(), 2);
    assert!(bound.contains(&(base + 0x2000, 0x100)));
    assert!(bound.contains(&(base + 0x4000, 0x100)));
}

#[test]
fn collect_indirects_captures_cpu_records_and_begin_slot_filters_gpu_writes() {
    use crate::detail::{DrawArgs, Stage, UboDetail};
    let Ok(ctx) = nxkit_metal::Context::system_default() else {
        return;
    };
    let mem = Arc::new(nxkit_metal::GuestMemory::allocate(&ctx, 1 << 24).unwrap());
    let guest = nxkit_metal::GuestSpace::over(&mem).unwrap();
    let base = guest.address_map().gpu_base();
    let renamer = Renamer::new(&ctx, guest.clone());
    let cpu_ind = base + 0x80_0000;
    let gpu_ind = base + 0x90_0000;
    guest
        .write(cpu_ind - base, &[42, 0, 0, 0, 7, 0, 0, 0])
        .unwrap();
    guest
        .write(gpu_ind - base, &[9, 0, 0, 0, 1, 0, 0, 0])
        .unwrap();
    let mut detail = FrameDetail::default();
    for ind in [cpu_ind, gpu_ind] {
        let mut draw = crate::walk::empty_draw();
        draw.program.mesh = Some(Arc::from("mesh_ind"));
        draw.args = DrawArgs::MeshTasksIndirect { indirect: ind };
        draw.sbufs.push(UboDetail {
            stage: Stage::Mesh,
            slot: 0,
            address: ind,
            size: 8,
            snapshot: None,
        });
        detail.draws.push(Arc::new(draw));
    }
    let (cap, _) = renamer.at_flush(&detail, 1);
    let cap = Arc::new(cap.expect("indirects must capture even when cands is empty"));
    assert_eq!(cap.indirects.len(), 2);
    // Simulate guest CPU overwriting the record immediately after flush returns.
    guest.write(cpu_ind - base, &[0; 8]).unwrap();
    let slotted = renamer.begin_slot(0, Some(&cap), &[], &[(gpu_ind, gpu_ind + 8)]);
    assert!(slotted.indirect_snaps.contains_key(&(cpu_ind, 8)));
    assert!(!slotted.indirect_snaps.contains_key(&(gpu_ind, 8)));
    let snap_addr = slotted.indirect_snaps[&(cpu_ind, 8)];
    let buf = slotted.buf.expect("slot buf");
    let off = snap_addr - buf.gpu_address();
    let mut got = [0u8; 8];
    buf.read(off, &mut got).unwrap();
    assert_eq!(got, [42, 0, 0, 0, 7, 0, 0, 0]);
}

#[test]
fn read_packed_handles_sparse_and_dense_multiword_masks() {
    let empty: Vec<u32> = vec![0, 0];
    let (vals, pref) = read_packed(&|_, _| panic!("no reads on empty mask"), 0, &empty).unwrap();
    assert!(vals.is_empty());
    assert_eq!(pref, vec![0, 0]);

    // Sparse across word boundaries: bit 31 of word 0 (w=31), bit 0 of word 1 (w=32), and bit 5 of word 2 (w=69).
    let mask = [1u32 << 31, 1u32, 1u32 << 5];
    let mem: Vec<u8> = (0u32..96).flat_map(|w| w.to_le_bytes()).collect();
    let reader = |addr: u64, dst: &mut [u8]| {
        let start = (addr - 0x1000) as usize;
        dst.copy_from_slice(&mem[start..start + dst.len()]);
        true
    };
    let (vals, pref) = read_packed(&reader, 0x1000, &mask).unwrap();
    assert_eq!(vals, vec![31, 32, 69]);
    assert_eq!(pref, vec![0, 1, 2]);
}

#[test]
fn large_rw_buffer_up_to_max_bytes_stays_candidate_and_demote_stops_snapshot_oscillation() {
    assert_eq!(MAX_BYTES, SNAPSHOT_MAX_BYTES);
    let size = 196_608u32;
    let g = Guest(RefCell::new(vec![0u8; size as usize + 0x200]));

    let mut rw_st = State::default();
    rw_st.candidate(SUM, size);
    assert!(rw_st.cands.contains_key(&SUM));

    // Once promoted to `Snapshot` with accumulated hits, `demote` must clear `ro` so flushes do not re-promote it.
    let mut ro_st = State::default();
    ro_st.watch_read_only(SUM, size);
    let cand = ro_st.cands.get_mut(&SUM).unwrap();
    cand.learn.set(Class::Snapshot);
    cand.hits = SNAPSHOT_HITS;
    assert!(ro_st.demote(SUM));
    assert_eq!(ro_st.cands[&SUM].learn.class(), Class::Never);
    assert!(!ro_st.cands[&SUM].ro);
    assert!(ro_st.cands[&SUM].sample.is_none());

    let bound = [(SUM, size)];
    for fence in 1..=LEARN_PERIOD * 2 {
        let f = ro_st.flush(
            fence,
            &bound,
            &|r| vec![false; r.len()],
            &|| Vec::new(),
            &|a, b| g.read(a, b),
        );
        assert!(f.renamed.iter().all(|r| r.kind != Kind::Snapshot));
        assert!(!ro_st.is_snapshot(SUM));
        assert_eq!(ro_st.cands[&SUM].learn.class(), Class::Never);
    }
}
