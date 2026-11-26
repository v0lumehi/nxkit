//! NSZ (compressed NCAs) on synthetic packages: block and solid streams give the NCA's original bytes, the extraction
//! from an NSZ equals the extraction from the NSP of the same NCAs, and hostile NCZ files are refused.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use std::sync::Arc;

use common::*;
use nxb_extract::Request;
use nxb_extract::aes::Aes128;
use nxb_extract::ncz::NczStorage;
use nxb_extract::storage::{MemStorage, Storage};

/// zstd of `data`: the `zstd` command when installed, else a frame of one raw block per 128 KiB.
fn zstd(data: &[u8]) -> Vec<u8> {
    use std::io::Write;
    if let Ok(mut c) = std::process::Command::new("zstd")
        .args(["-3", "-c", "-q"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
    {
        let mut i = c.stdin.take().unwrap();
        let d = data.to_vec();
        let w = std::thread::spawn(move || i.write_all(&d));
        let out = c.wait_with_output().unwrap().stdout;
        w.join().unwrap().unwrap();
        return out;
    }
    let mut f = vec![0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x58];
    let n = data.chunks(1 << 17).count().max(1);
    for (i, c) in data.chunks(1 << 17).enumerate() {
        f.extend_from_slice(&(((c.len() as u32) << 3) | u32::from(i + 1 == n)).to_le_bytes()[..3]);
        f.extend_from_slice(c);
    }
    f
}

/// What an `.ncz` holds for an NCA: header, section table, then blocks (`Some(exponent)`) or one stream.
fn ncz(nca: &[u8], plain: &[Plain], block: Option<u32>) -> Vec<u8> {
    let mut v = nca[..0x4000].to_vec();
    v.extend_from_slice(b"NCZSECTN");
    v.extend_from_slice(&(plain.len() as u64).to_le_bytes());
    let mut body = Vec::new();
    for (start, bytes, fs) in plain {
        v.extend_from_slice(&start.to_le_bytes());
        v.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        v.extend_from_slice(&3u64.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&CTR_KEY);
        let mut up: [u8; 8] = fs[0x140..0x148].try_into().unwrap();
        up.reverse();
        v.extend_from_slice(&up);
        v.extend_from_slice(&[0u8; 8]);
        body.extend_from_slice(bytes);
    }
    match block {
        None => v.extend_from_slice(&zstd(&body)),
        Some(exp) => {
            let bs = 1usize << exp;
            let blocks: Vec<Vec<u8>> = body
                .chunks(bs)
                .map(|b| {
                    let z = zstd(b);
                    if z.len() >= b.len() { b.to_vec() } else { z }
                })
                .collect();
            v.extend_from_slice(b"NCZBLOCK");
            v.extend_from_slice(&[2, 1, 0, exp as u8]);
            v.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
            v.extend_from_slice(&(body.len() as u64).to_le_bytes());
            for b in &blocks {
                v.extend_from_slice(&(b.len() as u32).to_le_bytes());
            }
            for b in &blocks {
                v.extend_from_slice(b);
            }
        }
    }
    v
}

fn open(bytes: Vec<u8>) -> nxb_extract::Result<NczStorage> {
    NczStorage::open(Arc::new(MemStorage(bytes)), "x")
}

fn request(keys: &std::path::Path, base: &std::path::Path, out: &std::path::Path) -> Request {
    Request {
        keys: keys.to_path_buf(),
        base: base.to_path_buf(),
        out_base: out.to_path_buf(),
        updates: vec![],
        threads: 2,
        mem_limit: 1 << 28,
        cache_chunks: 16,
        digests: None,
        html: false,
        base_outputs: true,
        split: None,
        app: None,
        program: None,
        discard_images: false,
        allow_damaged: false,
        manifest: false,
        progress: None,
        empty_romfs: false,
    }
}

/// The package's NCAs as `.ncz` files (the `Title` helper builds plain NCAs: rebuild each with its sections).
fn compressed_title(block: Option<u32>) -> (Files, Files) {
    let t = title(0x80, 0x0100_0000_0000_1000, 0, 7, true);
    let mut nsz = Vec::new();
    for (name, bytes) in &t.files {
        // Program and Control NCAs have sections with CTR, rebuild their plain layout from the NCA itself
        if name.ends_with(".cnmt.nca") {
            nsz.push((name.clone(), bytes.clone()));
            continue;
        }
        let plain = plain_sections(bytes);
        let n = format!("{}.ncz", name.trim_end_matches(".nca"));
        nsz.push((n, ncz(bytes, &plain, block)));
    }
    (t.files, nsz)
}

/// The decrypted sections of an NCA made by `common::nca` (read through the key area as the extractor does).
fn plain_sections(nca: &[u8]) -> Vec<Plain> {
    let mut h = nca[..0xC00].to_vec();
    xts_decrypt(&mut h);
    let mut out = Vec::new();
    for i in 0..4 {
        let u = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap()) as u64 * 0x200;
        let (s, e) = (u(0x240 + 16 * i), u(0x244 + 16 * i));
        if e > s {
            let fs = h[0x400 + 0x200 * i..0x600 + 0x200 * i].to_vec();
            let mut up: [u8; 8] = fs[0x140..0x148].try_into().unwrap();
            up.reverse();
            let mut d = nca[s as usize..e as usize].to_vec();
            Aes128::new(&CTR_KEY).ctr(up, s, &mut d);
            out.push((s, d, fs));
        }
    }
    out
}

fn xts_decrypt(h: &mut [u8]) {
    let k = |s: &[u8]| Aes128::new(&s.try_into().unwrap());
    nxb_extract::aes::xts_decrypt(&k(&HEADER_KEY[..16]), &k(&HEADER_KEY[16..]), h, 0x200, 0);
}

#[test]
fn block_and_solid_streams_show_the_original_nca() {
    let t = title(0x80, 0x0100_0000_0000_1000, 0, 7, true);
    let (name, nca) = &t.files[0];
    assert!(name.ends_with(".nca"));
    let plain = plain_sections(nca);
    for block in [None, Some(14), Some(16), Some(20)] {
        let s = open(ncz(nca, &plain, block)).unwrap();
        assert_eq!(s.size(), nca.len() as u64);
        let mut all = vec![0u8; nca.len()];
        s.read_at(0, &mut all).unwrap();
        assert!(all == *nca, "{block:?}: the whole NCA differs");
        // reads across the header edge, block edges and section edges
        for (at, n) in [
            (0x3FF0usize, 0x30usize),
            (0x4000, 1),
            (nca.len() - 5, 5),
            (0x4000 + (1 << 14) - 3, 9),
            (0x5001, 0x2fff),
        ] {
            let mut b = vec![0u8; n];
            s.read_at(at as u64, &mut b).unwrap();
            assert!(b == nca[at..at + n], "{block:?}: {at:#x}+{n:#x}");
        }
        assert!(s.read_at(nca.len() as u64 - 4, &mut [0u8; 5]).is_err());
    }
}

#[test]
fn extraction_from_an_nsz_equals_extraction_from_the_nsp() {
    let dir = tmp("nsz");
    let keys = write_keys(&dir);
    let (plain_files, _) = compressed_title(None);
    std::fs::write(dir.join("g.nsp"), pfs0(&plain_files)).unwrap();
    nxb_extract::extract(&request(&keys, &dir.join("g.nsp"), &dir.join("a"))).unwrap();
    for (k, block) in [None, Some(14), Some(20)].into_iter().enumerate() {
        let (_, nsz) = compressed_title(block);
        let p = dir.join(format!("g{k}.nsz"));
        std::fs::write(&p, pfs0(&nsz)).unwrap();
        let out = dir.join(format!("o{k}"));
        nxb_extract::extract(&request(&keys, &p, &out)).unwrap();
        same_outputs(&dir.join("a"), &out);
    }
    cleanup(&dir);
}

#[test]
fn a_big_solid_stream_goes_through_an_unlinked_temporary_file() {
    // 70 MiB of data in one CTR section: more than the in-memory limit
    let mut header = vec![0u8; 0x4000];
    header[0x300] = 1;
    let mut data = vec![0u8; 70 << 20];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i as u32).wrapping_mul(2654435761) as u8 & 0x0f;
    }
    let key = [5u8; 16];
    let mut v = header.clone();
    v.extend_from_slice(b"NCZSECTN");
    v.extend_from_slice(&1u64.to_le_bytes());
    v.extend_from_slice(&0x4000u64.to_le_bytes());
    v.extend_from_slice(&(data.len() as u64).to_le_bytes());
    v.extend_from_slice(&3u64.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes());
    v.extend_from_slice(&key);
    v.extend_from_slice(&[9, 8, 7, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
    v.extend_from_slice(&zstd(&data));
    let mut original = header;
    let mut enc = data;
    Aes128::new(&key).ctr([9, 8, 7, 6, 5, 4, 3, 2], 0x4000, &mut enc);
    original.extend_from_slice(&enc);
    let dir = tmp("nsz-spill");
    nxb_extract::ncz::set_spill_dir(&dir);
    let s = open(v).unwrap();
    let mut got = vec![0u8; 4096];
    for at in [
        0x4000u64,
        0x4000 + (33 << 20) + 17,
        original.len() as u64 - 4096,
    ] {
        s.read_at(at, &mut got).unwrap();
        assert!(got == original[at as usize..at as usize + 4096]);
    }
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "the temporary file is unlinked"
    );
    cleanup(&dir);
}

#[test]
fn hostile_ncz_files_are_refused() {
    let t = title(0x80, 0x0100_0000_0000_1000, 0, 7, false);
    let nca = &t.files[0].1;
    let plain = plain_sections(nca);
    let solid = ncz(nca, &plain, None);
    let block = ncz(nca, &plain, Some(14));
    assert!(open(solid.clone()).is_ok() && open(block.clone()).is_ok());
    let sec = 0x4000usize;
    // short file, no magic, huge section count, a section that does not start where the last ended, a huge size
    assert!(open(solid[..0x4000].to_vec()).is_err());
    let mut b = solid.clone();
    b[sec..sec + 8].copy_from_slice(b"NOTNCZ!!");
    assert!(open(b).is_err());
    for (at, v) in [
        (sec + 8, u64::MAX),
        (sec + 8, 0),
        (sec + 8, 1 << 21),
        (sec + 16, 0x3fff),
        (sec + 24, u64::MAX),
        (sec + 24, 1 << 50),
    ] {
        let mut b = solid.clone();
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        assert!(open(b).is_err(), "{at:#x} = {v:#x}");
    }
    // overlapping sections are refused even if both start at or after 0x4000
    {
        let mut b = vec![0u8; 0x4000];
        b.extend_from_slice(b"NCZSECTN");
        b.extend_from_slice(&2u64.to_le_bytes());
        for (start, len) in [(0x4000u64, 0x2000u64), (0x5000, 0x2000)] {
            b.extend_from_slice(&start.to_le_bytes());
            b.extend_from_slice(&len.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b.extend_from_slice(&[0u8; 0x28]);
        }
        b.extend_from_slice(&zstd(&vec![0u8; 0x4000]));
        assert!(open(b).is_err());
    }
    // block header: block size exponent, count and total size that do not match, a compressed size past the block
    let bh = sec + 16 + 0x40 * plain.len();
    for (at, len, v) in [
        (bh + 11, 1usize, 40u64),
        (bh + 11, 1, 3),
        (bh + 12, 4, 1),
        (bh + 12, 4, u32::MAX as u64),
        (bh + 16, 8, 5),
        (bh + 24, 4, 1 << 20),
        (bh + 24, 4, 0),
    ] {
        let mut b = block.clone();
        b[at..at + len].copy_from_slice(&v.to_le_bytes()[..len]);
        let r = open(b).and_then(|s| s.read_vec(0x4000, 16));
        assert!(r.is_err(), "{at:#x} = {v:#x}");
    }
    // a solid stream that is cut, or that decodes to more or less than the sections hold
    let cut = open(solid[..solid.len() - 20].to_vec()).unwrap();
    assert!(cut.read_vec(0x4000, 16).is_err());
    let mut longer = solid.clone();
    let last = sec + 16 + 0x40 * (plain.len() - 1) + 8;
    let n = plain[plain.len() - 1].1.len() as u64;
    longer[last..last + 8].copy_from_slice(&(n + 0x200).to_le_bytes());
    assert!(open(longer).unwrap().read_vec(0x4000, 16).is_err());
    let mut shorter = solid.clone();
    shorter[last..last + 8].copy_from_slice(&(n - 0x200).to_le_bytes());
    assert!(open(shorter).unwrap().read_vec(0x4000, 16).is_err());
}

#[test]
fn non_contiguous_ncz_sections_roundtrip() {
    let mut header = vec![0u8; 0x4000];
    header[0x200..0x204].copy_from_slice(b"NCA3");
    let key = [0x3cu8; 16];
    let p0 = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let p1 = [8u8, 7, 6, 5, 4, 3, 2, 1];
    let sec0: Vec<u8> = (0..0x2000usize).map(|i| (i * 7 + 3) as u8).collect();
    let sec1: Vec<u8> = (0..0x4000usize).map(|i| (i * 13 + 9) as u8).collect();
    // Leading gap 0x4000..0x8000 and middle gap 0xa000..0xe000.
    let mut expected = header.clone();
    expected.resize(0x12000, 0);
    expected[0x8000..0xa000].copy_from_slice(&sec0);
    Aes128::new(&key).ctr(p0, 0x8000, &mut expected[0x8000..0xa000]);
    expected[0xe000..0x12000].copy_from_slice(&sec1);
    Aes128::new(&key).ctr(p1, 0xe000, &mut expected[0xe000..0x12000]);

    let build_table = || {
        let mut v = header.clone();
        v.extend_from_slice(b"NCZSECTN");
        v.extend_from_slice(&2u64.to_le_bytes());
        for (off, data, prefix) in [(0x8000u64, &sec0, p0), (0xe000u64, &sec1, p1)] {
            v.extend_from_slice(&off.to_le_bytes());
            v.extend_from_slice(&(data.len() as u64).to_le_bytes());
            v.extend_from_slice(&3u64.to_le_bytes());
            v.extend_from_slice(&0u64.to_le_bytes());
            v.extend_from_slice(&key);
            v.extend_from_slice(&prefix);
            v.extend_from_slice(&[0u8; 8]);
        }
        v
    };
    let build_blocks = |v: &mut Vec<u8>, body: &[u8]| {
        let exp = 12u32;
        let blocks: Vec<Vec<u8>> = body
            .chunks(1usize << exp)
            .map(|b| {
                let z = zstd(b);
                if z.len() >= b.len() { b.to_vec() } else { z }
            })
            .collect();
        v.extend_from_slice(b"NCZBLOCK");
        v.extend_from_slice(&[2, 1, 0, exp as u8]);
        v.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
        v.extend_from_slice(&(body.len() as u64).to_le_bytes());
        for b in &blocks {
            v.extend_from_slice(&(b.len() as u32).to_le_bytes());
        }
        for b in &blocks {
            v.extend_from_slice(b);
        }
    };

    let mut packed_body = sec0.clone();
    packed_body.extend_from_slice(&sec1);
    let mut flat_body = vec![0u8; 0x12000 - 0x4000];
    flat_body[0x4000..0x6000].copy_from_slice(&sec0);
    flat_body[0xa000..0xe000].copy_from_slice(&sec1);

    let mut packed_solid = build_table();
    packed_solid.extend_from_slice(&zstd(&packed_body));
    let mut packed_block = build_table();
    build_blocks(&mut packed_block, &packed_body);
    let mut flat_block = build_table();
    build_blocks(&mut flat_block, &flat_body);

    for (label, bytes) in [
        ("packed-solid", packed_solid),
        ("packed-block", packed_block),
        ("flat-block", flat_block),
    ] {
        let s = open(bytes).unwrap();
        assert_eq!(s.size(), expected.len() as u64, "{label}");
        let mut all = vec![0u8; expected.len()];
        s.read_at(0, &mut all).unwrap();
        assert!(all == expected, "{label}: whole NCA mismatch");
        for (at, len) in [(0x3ff0u64, 0x200usize), (0x9f00, 0x300), (0xdf00, 0x300)] {
            let mut slice = vec![0u8; len];
            s.read_at(at, &mut slice).unwrap();
            assert!(
                slice == expected[at as usize..at as usize + len],
                "{label}: {at:#x}+{len:#x}"
            );
        }
    }
}
