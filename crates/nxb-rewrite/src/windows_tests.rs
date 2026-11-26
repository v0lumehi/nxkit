//! Exclusive windows on synthetic code: the cases of "Branches within an exclusive window" in `docs/reference/rewriter.md`.

use std::collections::BTreeMap;

use nxb_a64::{Insn, decode, encode};

use crate::analyse::analyse;
use crate::rw::Rw;
use crate::scan::{Kind, Scan};

const RET: u32 = 0xd65f_03c0;
const CLREX: u32 = 0xd503_3f5f;
/// `ldxr w9, [x8]`
const LDXR: u32 = 0x885f_7d09;
/// `ldxr w11, [x8]`, the first instruction of a retry block
const LDXR_RETRY: u32 = 0x885f_7d0b;
/// `stxr w10, w9, [x8]`
const STXR: u32 = 0x8800_7c00 | 10 << 16 | 8 << 5 | 9;
/// `add w9, w9, w18`, the instruction that makes a window a site
const ADD_W18: u32 = 0x0b00_0000 | 18 << 16 | 9 << 5 | 9;
/// `cmp w9, w18`
const CMP_W18: u32 = 0x6b00_001f | 18 << 16 | 9 << 5;
/// `cmp w11, w12`
const CMP_RETRY: u32 = 0x6b00_001f | 12 << 16 | 11 << 5;
const MOV_W9_W1: u32 = 0x2a01_03e0 | 9;

/// The word index `to` seen from index `at`, as a 19-bit field.
fn imm19(at: usize, to: usize) -> u32 {
    (((to as i64 - at as i64) as u32) & 0x7ffff) << 5
}

/// `cbnz w10, to`
fn cbnz_w10(at: usize, to: usize) -> u32 {
    0x3500_0000 | imm19(at, to) | 10
}

/// `b.<cond>`: eq 0, ne 1.
fn bcond(cond: u32, at: usize, to: usize) -> u32 {
    0x5400_0000 | imm19(at, to) | cond
}

fn b(at: usize, to: usize) -> u32 {
    encode::b(4 * at as i64, 4 * to as i64).unwrap()
}

fn decode_all(words: &[u32]) -> Vec<Insn> {
    words
        .iter()
        .enumerate()
        .map(|(i, w)| decode(*w, 4 * i as u64).unwrap())
        .collect()
}

fn rw_of(words: &[u32]) -> Rw {
    let mut img = vec![0u8; 0x1000];
    for (i, w) in words.iter().enumerate() {
        img[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    Rw {
        name: "t".into(),
        img,
        text_end: 4 * words.len() as i64,
        far: false,
        front: 0,
        big_text: false,
        two: false,
        split: 0,
        back_min: 0,
        key_page_off: 0x8000,
        image_size: 0,
        patch_off: 0x10000,
        back_off: 0,
        file_image_len: 0,
        ro_off: 0,
        data_off: 0,
        front_area: Vec::new(),
        back_area: Vec::new(),
        in_back: false,
        caves: None,
        cave_data_blocks: Vec::new(),
        cave_excluded: std::collections::HashSet::new(),
        cave_fdes: Vec::new(),
        cave_code_ranges: Vec::new(),
        gates: Vec::new(),
        nfix: 0,
        stats: Vec::new(),
    }
}

/// Analyses `words` as functions starting at the word indexes `starts`, with the word `site` using x18.
fn run(words: &[u32], starts: &[usize], site: usize) -> Result<crate::analyse::Analysis, String> {
    let rw = rw_of(words);
    let scan = Scan {
        starts: starts.iter().map(|s| 4 * *s as i64).collect(),
        data_blocks: Vec::new(),
        kinds: BTreeMap::from([(4 * site as i64, (Kind::X18, 0))]),
        spin_cand: BTreeMap::new(),
    };
    analyse(&rw, &scan).map_err(|e| e.to_string())
}

/// A window whose `b.ne` goes to its own store: `ldxr; cmp; b.ne store; mov; stxr; cbnz head`.
fn inner_branch() -> Vec<u32> {
    vec![
        LDXR,
        CMP_W18,
        bcond(1, 2, 4),
        MOV_W9_W1,
        STXR,
        cbnz_w10(5, 0),
        RET,
    ]
}

/// A window whose `b` skips an instruction: `ldxr; b store; add; stxr; cbnz head`.
fn with_b() -> Vec<u32> {
    vec![LDXR, b(1, 3), ADD_W18, STXR, cbnz_w10(4, 0), RET]
}

#[test]
fn a_branch_to_the_store_inside_its_own_window_is_not_a_retry() {
    let a = run(&inner_branch(), &[0], 1).unwrap();
    assert_eq!(a.windows.len(), 1);
    assert!(a.retries.is_empty());
}

#[test]
fn a_b_inside_a_window_goes_to_the_copy() {
    let mut rw = rw_of(&with_b());
    let seg = decode_all(&with_b())[..5].to_vec();
    let (start, lo, hi) = rw.site_window(&seg).unwrap();
    assert_eq!((lo, hi), (0, 20));
    let copy0 = start + 4 * 6;
    let at = |k: i64| (copy0 + 4 * k - rw.patch_off) as usize;
    let word = u32::from_le_bytes(rw.front_area[at(1)..at(1) + 4].try_into().unwrap());
    let ins = decode(word, (copy0 + 4) as u64).unwrap();
    assert_eq!((ins.mn, ins.target()), ("b", Some((copy0 + 12) as u64)));
}

#[test]
fn a_b_out_of_a_window_goes_through_an_exit_stub() {
    // The reservation is cleared before the b, so the stub's store loses nothing.
    let words = [LDXR, ADD_W18, CLREX, b(3, 10), STXR, cbnz_w10(5, 0), RET];
    let mut rw = rw_of(&words);
    let seg = decode_all(&words)[..6].to_vec();
    let (start, _, _) = rw.site_window(&seg).unwrap();
    let copy0 = start + 4 * 6;
    let word_at = |addr: i64| {
        let o = (addr - rw.patch_off) as usize;
        u32::from_le_bytes(rw.front_area[o..o + 4].try_into().unwrap())
    };
    // The fall-through stub comes first, the stub of the b after it.
    let stub = copy0 + 4 * 6 + 16;
    let ins = decode(word_at(copy0 + 12), (copy0 + 12) as u64).unwrap();
    assert_eq!((ins.mn, ins.target()), ("b", Some(stub as u64)));
    let out = decode(word_at(stub + 8), (stub + 8) as u64).unwrap();
    assert_eq!((out.mn, out.target()), ("b", Some(40)));
}

#[test]
fn a_window_with_a_b_may_not_leave_with_a_live_reservation() {
    let words = [LDXR, ADD_W18, b(2, 10), STXR, cbnz_w10(4, 0), RET];
    let mut rw = rw_of(&words);
    let seg = decode_all(&words)[..5].to_vec();
    let e = rw.site_window(&seg).unwrap_err().to_string();
    assert!(e.contains("live reservation"), "{e}");
}

#[test]
fn a_branch_from_another_function_into_the_window_is_refused() {
    // The second function starts at word 6 and jumps to the store, word 3.
    let mut words = with_b();
    words.extend([b(6, 3), RET]);
    let e = run(&words, &[0, 6], 2).unwrap_err();
    assert!(e.contains("from another function"), "{e}");
}

#[test]
fn the_window_with_a_b_is_accepted_when_no_other_function_enters_it() {
    let mut words = with_b();
    words.extend([RET, RET]);
    let a = run(&words, &[0, 6], 2).unwrap();
    assert_eq!(a.windows.len(), 1);
}

/// A window `ldxr; add; stxr; cbnz <back>` and a retry block `ldxr; cmp; b.eq store` after it.
fn retry(back: usize) -> Vec<u32> {
    vec![
        LDXR,
        ADD_W18,
        STXR,
        cbnz_w10(3, back),
        RET,
        LDXR_RETRY,
        CMP_RETRY,
        bcond(0, 7, 2),
        RET,
    ]
}

#[test]
fn a_retry_block_is_still_a_retry_block() {
    let a = run(&retry(0), &[0], 1).unwrap();
    assert_eq!(a.retries.len(), 1);
}

#[test]
fn a_retry_copy_may_not_branch_into_the_window() {
    let e = run(&retry(1), &[0], 1).unwrap_err();
    assert!(e.contains("retry copy branches into window"), "{e}");
}
