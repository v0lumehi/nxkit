//! Tests that need no game data.

use nxb_a64::{decode, encode};
use nxb_rewrite::{Options, RewriteError, patches::PatchTable, rewrite};

#[test]
fn patch_table_parses_rows() {
    let t = PatchTable::parse(
        "file\ta.py\tab\ncuts\tb1\t16\t00000000\t1f2003d5\tnop out\nframe\tb2\ncutsempty\tb3\nrule\tsdk\tsdk\tnode\t0\t0\t240\taa,bb\nrcut\tsdk\tsdk\tnode\t0\t0\t4\t00000000\t1f2003d5\twhy\n",
    )
    .unwrap_or_default();
    assert_eq!(t.cuts["b1"][0].off, 16);
    assert!(t.cuts["b3"].is_empty());
    assert!(t.frame.contains("b2"));
    assert_eq!(t.rules["sdk"]["sdk"][0].fns[0].fps, ["aa", "bb"]);
}

#[test]
fn garbage_is_refused_not_panicked() {
    let r = rewrite(&[0u8; 64], &Options::default(), &PatchTable::default());
    assert!(matches!(r, Err(RewriteError::Failed(m)) if m.contains("not an NSO")));
}

const RET: u32 = 0xd65f_03c0;
const NOP: u32 = 0xd503_201f;
const SVC_0: u32 = 0xd400_0001;
const MRS_X0_TPIDRRO: u32 = 0xd53b_d060;
const ADD_X1_X18_4: u32 = 0x9100_1241;

fn put32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn get32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Builds an uncompressed NSO0 with `.text` (`0..text_size`), `.rodata` (`ro_off..ro_off+0x4000`),
/// `.data` (`data_off..data_off+0x4000`), and `bss` bytes, plus `MOD0` and `.eh_frame_hdr`.
/// Each entry in `funcs` is `(offset, words)`. Gaps between `funcs` are filled with `gap_word`.
fn make_nso(
    text_size: usize,
    ro_off: usize,
    data_off: usize,
    bss: usize,
    gap_word: u32,
    funcs: &[(usize, Vec<u32>)],
) -> Vec<u8> {
    let img_len = data_off + 0x4000;
    let mut nso = vec![0u8; 0x100 + img_len];
    nso[0..4].copy_from_slice(b"NSO0");
    // text: file_off=0x100, mem_off=0, size=text_size
    put32(&mut nso, 0x10, 0x100);
    put32(&mut nso, 0x14, 0);
    put32(&mut nso, 0x18, text_size as u32);
    // rodata: file_off=0x100+ro_off, mem_off=ro_off, size=0x4000
    put32(&mut nso, 0x20, (0x100 + ro_off) as u32);
    put32(&mut nso, 0x24, ro_off as u32);
    put32(&mut nso, 0x28, 0x4000);
    // data: file_off=0x100+data_off, mem_off=data_off, size=0x4000
    put32(&mut nso, 0x30, (0x100 + data_off) as u32);
    put32(&mut nso, 0x34, data_off as u32);
    put32(&mut nso, 0x38, 0x4000);
    put32(&mut nso, 0x3c, bss as u32);
    nso[0x40..0x50].copy_from_slice(&[0xab; 0x10]);

    let img = &mut nso[0x100..];
    for chunk in img[..text_size].as_chunks_mut::<4>().0 {
        chunk.copy_from_slice(&gap_word.to_le_bytes());
    }
    for (off, words) in funcs {
        for (i, w) in words.iter().enumerate() {
            put32(img, off + 4 * i, *w);
        }
    }

    // Place MOD0, CIE, FDEs, and .eh_frame_hdr inside .rodata.
    let mod0 = ro_off;
    if let Ok(b0) = encode::b(0, 0x40) {
        put32(img, 0, b0);
    }
    put32(img, 4, mod0 as u32);
    img[mod0..mod0 + 4].copy_from_slice(b"MOD0");
    let eh_hdr = ro_off + 0x100;
    let eh_hdr_end = eh_hdr + 12 + funcs.len() * 8;
    put32(img, mod0 + 16, (eh_hdr - mod0) as u32);
    put32(img, mod0 + 20, (eh_hdr_end - mod0) as u32);

    // CIE at ro_off + 0x40
    let cie = ro_off + 0x40;
    put32(img, cie, 16);
    put32(img, cie + 4, 0);
    img[cie + 8] = 1;
    img[cie + 9..cie + 12].copy_from_slice(b"zR\0");
    img[cie + 12] = 1; // code_align
    img[cie + 13] = 0x7c; // data_align
    img[cie + 14] = 30; // ret_reg
    img[cie + 15] = 1; // aug_len
    img[cie + 16] = 0x03; // udata4

    // .eh_frame_hdr
    img[eh_hdr] = 1;
    img[eh_hdr + 1] = 0x03;
    img[eh_hdr + 2] = 0x03;
    img[eh_hdr + 3] = 0x03;
    put32(img, eh_hdr + 4, cie as u32);
    put32(img, eh_hdr + 8, funcs.len() as u32);

    let mut fde_cursor = eh_hdr_end + 16;
    for (i, (off, words)) in funcs.iter().enumerate() {
        let fde = fde_cursor;
        fde_cursor += 16;
        put32(img, fde, 12);
        put32(img, fde + 4, (fde + 4 - cie) as u32);
        put32(img, fde + 8, *off as u32);
        put32(img, fde + 12, (words.len() * 4) as u32);

        put32(img, eh_hdr + 12 + i * 8, *off as u32);
        put32(img, eh_hdr + 12 + i * 8 + 4, fde as u32);
    }

    nso
}

fn parse_fixups(text: &str) -> std::collections::HashMap<String, i64> {
    let mut m = std::collections::HashMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("# ") else {
            continue;
        };
        if let Some((k, v)) = rest.split_once('\t') {
            let v = v.trim();
            let (neg, digits) = v.strip_prefix('-').map_or((false, v), |d| (true, d));
            if let Ok(n) = digits
                .strip_prefix("0x")
                .map_or_else(|| digits.parse::<i64>(), |h| i64::from_str_radix(h, 16))
            {
                m.insert(k.trim().to_string(), if neg { -n } else { n });
            }
        }
    }
    m
}

/// Reads a 32-bit instruction from the virtual address `va` of a rewritten two-area module.
fn read_va(
    bin: &[u8],
    patch_off: i64,
    file_image_len: usize,
    back_off: i64,
    back_len: usize,
    va: i64,
) -> Result<u32, String> {
    if va >= 0 && (va as usize) < file_image_len {
        return Ok(get32(bin, va as usize));
    }
    let front_len = bin.len() - file_image_len - back_len;
    if va >= patch_off && va < patch_off + front_len as i64 {
        let off = file_image_len + (va - patch_off) as usize;
        return Ok(get32(bin, off));
    }
    if va >= back_off && va < back_off + back_len as i64 {
        let off = bin.len() - back_len + (va - back_off) as usize;
        return Ok(get32(bin, off));
    }
    Err(format!("va {va:#x} outside mapped executable ranges"))
}

fn follow_to_stub(
    bin: &[u8],
    patch_off: i64,
    file_image_len: usize,
    back_off: i64,
    back_len: usize,
    start: i64,
) -> Result<(i64, usize), String> {
    let mut pc = start;
    let mut hops = 0usize;
    loop {
        if pc < 0 || pc >= back_off {
            return Ok((pc, hops));
        }
        let w = read_va(bin, patch_off, file_image_len, back_off, back_len, pc)?;
        let ins = decode(w, pc as u64).ok_or_else(|| format!("invalid insn at {pc:#x}"))?;
        assert_eq!(ins.mn, "b", "expected branch at {pc:#x} ({})", ins.mn);
        let tgt = ins
            .target()
            .ok_or_else(|| format!("no target at {pc:#x}"))? as i64;
        let disp = tgt - pc;
        assert!((-(1 << 27)..(1 << 27)).contains(&disp));
        pc = tgt;
        hops += 1;
        assert!(hops < 16, "too many hops from {start:#x}");
    }
}

fn follow_to_addr(
    bin: &[u8],
    patch_off: i64,
    file_image_len: usize,
    back_off: i64,
    back_len: usize,
    start: i64,
    want: i64,
) -> Result<usize, String> {
    let mut pc = start;
    let mut hops = 0usize;
    loop {
        if pc == want {
            return Ok(hops);
        }
        let w = read_va(bin, patch_off, file_image_len, back_off, back_len, pc)?;
        let ins = decode(w, pc as u64).ok_or_else(|| format!("invalid insn at {pc:#x}"))?;
        assert_eq!(
            ins.mn, "b",
            "expected return branch at {pc:#x} ({})",
            ins.mn
        );
        let tgt = ins
            .target()
            .ok_or_else(|| format!("no target at {pc:#x}"))? as i64;
        let disp = tgt - pc;
        assert!((-(1 << 27)..(1 << 27)).contains(&disp));
        pc = tgt;
        hops += 1;
        assert!(hops < 16, "too many return hops from {start:#x}");
    }
}

#[test]
fn large_module_with_dead_zone_uses_multi_hop_relays_and_back_area() {
    // 230 MiB .text + 125 MiB .bss -> back_off = 0x1630_0000, back_min = 0xe40_0000 (> 0x7e0_0000).
    let text_size = 0xe60_0000;
    let ro_off = 0xe60_0000;
    let data_off = 0xe60_4000;
    let bss = 0x7cf_0000;
    let site_front = 0x1000usize;
    let site_mid1 = 0x900_0000usize;
    let site_mid2 = 0xc00_0000usize;
    let site_back = 0xe50_0000usize;
    // Place padding islands (FDEs ending in RET followed by NOPs) inside reach windows;
    // fill giant unused gaps with undecodable 0x0001_0000 so `data_blocks` skips them quickly.
    let funcs = vec![
        (site_front, vec![SVC_0, RET]),
        (0x6ef_0000, vec![RET, NOP, NOP, NOP, NOP, NOP, NOP, NOP]),
        (0x6ef_0020, vec![RET]),
        (site_mid1, vec![MRS_X0_TPIDRRO, RET]),
        (site_mid2, vec![ADD_X1_X18_4, RET]),
        (0xe41_0000, vec![RET, NOP, NOP, NOP, NOP, NOP, NOP, NOP]),
        (0xe41_0020, vec![RET]),
        (site_back, vec![SVC_0, RET]),
    ];
    let mut nso = make_nso(text_size, ro_off, data_off, bss, 0x0001_0000, &funcs);
    // Keep the NOP padding right after the anchor FDEs outside their FDE lengths.
    let eh_hdr = 0x100 + ro_off + 0x100;
    for idx in [1usize, 5] {
        let fde_off = get32(&nso, eh_hdr + 12 + idx * 8 + 4) as usize;
        put32(&mut nso, 0x100 + fde_off + 12, 4);
    }
    let opts = Options {
        name: "main".into(),
        ..Options::default()
    };
    let out = rewrite(&nso, &opts, &PatchTable::default()).expect("large module should rewrite");
    let fixups_bytes = &out
        .files
        .iter()
        .find(|(n, _)| n == "main.fixups.tsv")
        .unwrap()
        .1;
    let fixups = parse_fixups(&String::from_utf8_lossy(fixups_bytes));
    let bin = &out
        .files
        .iter()
        .find(|(n, _)| n == "main.patched.bin")
        .unwrap()
        .1;

    let patch_off = fixups["PATCH_OFF"];
    let file_image_len = fixups["FILE_IMAGE_LEN"] as usize;
    let back_off = fixups["BACK_OFF"];
    let back_len = fixups["BACK_LEN"] as usize;
    assert!(patch_off < 0);
    assert!(back_off - 0x800_0000 + 0x10_0000 >= 0xe3e_0000);
    assert!(back_len > 0);

    let cases = [
        (site_front, 36),
        (site_mid1, 28),
        (site_mid2, 36),
        (site_back, 36),
    ];
    for (site, tail_off) in cases {
        let (stub, hops) = follow_to_stub(
            bin,
            patch_off,
            file_image_len,
            back_off,
            back_len,
            site as i64,
        )
        .unwrap();
        if site == site_front {
            assert!(stub < 0);
            assert_eq!(hops, 1);
        } else if site == site_back {
            assert!(stub >= back_off);
            assert_eq!(hops, 1);
        } else {
            assert!(
                hops >= 2,
                "dead-zone site {site:#x} should use >=1 relay hop (got {hops})"
            );
        }
        let ret_hops = follow_to_addr(
            bin,
            patch_off,
            file_image_len,
            back_off,
            back_len,
            stub + tail_off,
            site as i64 + 4,
        )
        .unwrap();
        if site == site_mid1 || site == site_mid2 {
            assert!(ret_hops >= 2, "return from {site:#x} should also relay");
        }
    }
}

#[test]
fn oversized_module_without_relay_caves_returns_unsupported() {
    // 130 MiB .text filled with non-padding `yield` words (inside a single FDE covering all of .text)
    // plus 225 MiB .bss -> back_min = 0xe40_0000, ro_off == text_size -> zero relay caves in the dead zone.
    let text_size = 0x820_0000;
    let ro_off = 0x820_0000;
    let data_off = 0x820_4000;
    let bss = 0xe0f_0000;
    let site_mid = 0x810_0000usize;
    let mut nso = make_nso(
        text_size,
        ro_off,
        data_off,
        bss,
        0xd503_203f, // `yield`: decodable (not data_blocks), not a padding word
        &[(0, vec![RET]), (site_mid, vec![SVC_0, RET])],
    );
    // Cover all of `.text` with 1 FDE so there are no inter-FDE gaps at all.
    let eh_hdr = 0x100 + ro_off + 0x100;
    put32(&mut nso, eh_hdr + 8, 1);
    let fde_off = get32(&nso, eh_hdr + 12 + 4) as usize;
    put32(&mut nso, 0x100 + fde_off + 8, 0);
    put32(&mut nso, 0x100 + fde_off + 12, text_size as u32);
    let opts = Options {
        name: "main".into(),
        ..Options::default()
    };
    let err = rewrite(&nso, &opts, &PatchTable::default())
        .err()
        .expect("should fail");
    assert!(
        matches!(err, RewriteError::Unsupported(_)),
        "expected Unsupported, got {err:?}"
    );
}
