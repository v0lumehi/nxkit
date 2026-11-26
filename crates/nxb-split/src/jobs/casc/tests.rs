#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use super::*;
use crate::SplitError;
use crate::jobs::{NroSet, deferred, own, plan_with};
use crate::prim::{CS_MAGIC, CTL_MAGIC, GFX_MAGIC};
use crate::shaders::Profile;
use crate::zstd::Dicts;

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut pos = 0;
    while pos < data.len() || pos == 0 {
        let end = (pos + 65535).min(data.len());
        let last = u8::from(end == data.len());
        let chunk = &data[pos..end];
        let len = chunk.len() as u16;
        out.push(last);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
        pos = end;
        if data.is_empty() {
            break;
        }
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn make_blte(chunks: &[&[u8]], decomp_sizes: &[usize]) -> Vec<u8> {
    let hs = 12 + 24 * chunks.len();
    let mut out = Vec::new();
    out.extend_from_slice(&blte::MAGIC);
    out.extend_from_slice(&(hs as u32).to_be_bytes());
    out.push(0x0f);
    let n = chunks.len() as u32;
    out.extend_from_slice(&n.to_be_bytes()[1..]);
    for (c, &ds) in chunks.iter().zip(decomp_sizes) {
        out.extend_from_slice(&(c.len() as u32).to_be_bytes());
        out.extend_from_slice(&(ds as u32).to_be_bytes());
        out.extend_from_slice(&[0u8; 16]);
    }
    for c in chunks {
        out.extend_from_slice(c);
    }
    out
}

fn make_casc_entry(ekey_byte: u8, payload: &[u8]) -> Vec<u8> {
    let total = CASC_HDR_LEN + payload.len();
    let mut out = vec![ekey_byte; 16];
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 10]);
    out.extend_from_slice(payload);
    out
}

fn make_raw_vs_pair(tag: u8) -> (Vec<u8>, Vec<u8>) {
    let nctl = 0x880usize;
    let ncode = 0x80usize;
    let mut ctl = vec![0u8; nctl];
    ctl[..4].copy_from_slice(&CTL_MAGIC);
    ctl[0x20..0x24].copy_from_slice(&(nctl as u32).to_le_bytes());
    ctl[0x704..0x708].copy_from_slice(&(ncode as u32).to_le_bytes());
    ctl[0x100] = tag;

    let mut code = vec![0u8; ncode];
    code[..4].copy_from_slice(&GFX_MAGIC);
    // sph word at +0x30: stage = 1 (vs) in bits [13:10]
    code[0x30..0x34].copy_from_slice(&(1u32 << 10).to_le_bytes());
    code[0x40] = tag;
    (ctl, code)
}

fn make_raw_cs_pair(tag: u8) -> (Vec<u8>, Vec<u8>) {
    let nctl = 0x880usize;
    let ncode = 0x60usize;
    let mut ctl = vec![0u8; nctl];
    ctl[..4].copy_from_slice(&CTL_MAGIC);
    ctl[0x20..0x24].copy_from_slice(&(nctl as u32).to_le_bytes());
    ctl[0x704..0x708].copy_from_slice(&(ncode as u32).to_le_bytes());
    ctl[0x748..0x754].copy_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    ctl[0x100] = tag;

    let mut code = vec![0u8; ncode];
    code[..4].copy_from_slice(&CS_MAGIC);
    code[0x20] = tag;
    (ctl, code)
}

fn make_casc_idx() -> Vec<u8> {
    let mut idx = vec![0u8; 0x30];
    idx[0..4].copy_from_slice(&0x10u32.to_le_bytes());
    idx[8..10].copy_from_slice(&7u16.to_le_bytes());
    idx[12] = 4;
    idx[13] = 5;
    idx[14] = 9;
    idx[15] = 0x1e;
    idx
}

#[test]
fn plans_and_scans_switch_casc_data_and_skips_nlg_and_non_casc() {
    // 1. Archive header entry (0x40 bytes total) + GLSLC vs shader in multi-chunk BLTE
    let vs_glslc = include_bytes!("../../../tests/data/glslc_a_vs.bin");
    let mid = vs_glslc.len() / 2;
    let mut c0 = vec![b'N'];
    c0.extend_from_slice(&vs_glslc[..mid]);
    let mut c1 = vec![b'Z'];
    c1.extend_from_slice(&zlib_stored(&vs_glslc[mid..]));
    let blte_glslc = make_blte(&[&c0, &c1], &[mid, vs_glslc.len() - mid]);

    // 2. Single-chunk BLTE (header_size = 0) with a raw CS shader (code right after ctl, 256B aligned)
    let (cs_ctl, cs_code) = make_raw_cs_pair(0x22);
    let aligned = (cs_ctl.len() + 255) & !255;
    let mut raw_cs_blob = vec![0u8; aligned + cs_code.len()];
    raw_cs_blob[..cs_ctl.len()].copy_from_slice(&cs_ctl);
    raw_cs_blob[aligned..].copy_from_slice(&cs_code);
    let mut single_blte = Vec::from(blte::MAGIC);
    single_blte.extend_from_slice(&0u32.to_be_bytes());
    single_blte.push(b'Z');
    single_blte.extend_from_slice(&zlib_stored(&raw_cs_blob));

    // 3. Encrypted BLTE entry (must be skipped cleanly)
    let enc_blte = make_blte(&[b"E\x08\0\0\0\0\0\0\0\0\x04\0\0\0\0Ssecret"], &[6]);

    // 4. Multi-chunk BLTE wrapping a raw VS shader
    let (vs_ctl, vs_code) = make_raw_vs_pair(0x33);
    let mut raw_vs_blob = vs_ctl.clone();
    raw_vs_blob.extend_from_slice(&vs_code);
    let mut c_vs = vec![b'N'];
    c_vs.extend_from_slice(&raw_vs_blob);
    let blte_vs = make_blte(&[&c_vs], &[raw_vs_blob.len()]);

    let mut data_file = make_casc_entry(0, &[0u8; 0x22]); // archive header entry (0x40 total)
    data_file.extend_from_slice(&make_casc_entry(1, &blte_glslc));
    data_file.extend_from_slice(&[0u8; 32]); // alignment padding
    data_file.extend_from_slice(&make_casc_entry(2, &single_blte));
    data_file.extend_from_slice(&make_casc_entry(3, &enc_blte));
    data_file.extend_from_slice(&make_casc_entry(4, &blte_vs));

    let files = [
        File {
            rel: "data/00-00000001.data".into(),
            src: own(data_file),
            live: true,
        },
        File {
            rel: "data/00-00000001.idx".into(),
            src: own(make_casc_idx()),
            live: true,
        },
        File {
            rel: "lm/rom.dict".into(),
            src: own(vec![0x58, 0x24, 0xf3, 0xa9]),
            live: true,
        },
        File {
            rel: "lm/rom.data".into(),
            src: own(vec![0u8; 128]),
            live: true,
        },
        File {
            rel: "other/plain.data".into(),
            src: own(b"not casc at all".to_vec()),
            live: true,
        },
    ];

    let plan = plan_with(&files, &NroSet::default()).unwrap();
    assert_eq!(plan.jobs.len(), 1);
    assert!(plan.taken.contains(&0));
    assert!(plan.taken.contains(&1));
    assert!(!plan.taken.contains(&2));
    assert!(!plan.taken.contains(&3));
    assert!(!plan.taken.contains(&4));

    let dicts = Dicts::new();
    let ctx = Ctx {
        files: &files,
        dicts: &dicts,
    };
    let mut notes = Vec::new();
    let out = (plan.jobs[0].run)(&ctx, &mut notes).unwrap();
    assert!(notes.is_empty());
    assert_eq!(out.found.len(), 3);
    assert_eq!(out.found[0].stage, "vs");
    assert_eq!(out.found[1].stage, "cs");
    assert_eq!(out.found[2].stage, "vs");

    let prof = Profile::from_game_tsv("shader_groups\t/data/=data\n");
    for f in &out.found {
        assert_eq!(prof.group(&f.place), "data");
    }
}

#[test]
fn deferred_and_classic_data_000_and_streaming_buffer_boundary() {
    assert!(deferred("data/00-00000001.data", &[]));
    assert!(deferred("data/00-00000001.idx", &[]));
    assert!(deferred("Data/data/data.000", &[]));
    assert!(!deferred("Data/data/data.txt", &[]));

    // Put a valid CASC entry just past the 4 MiB StreamBuf boundary without calling whole()
    struct StrictStream(Vec<u8>);
    impl Src for StrictStream {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<(), SplitError> {
            crate::src::Mem(&self.0).read_at(off, buf)
        }
    }

    let (vs_ctl, vs_code) = make_raw_vs_pair(0x55);
    let mut blob = vs_ctl;
    blob.extend_from_slice(&vs_code);
    let mut c0 = vec![b'N'];
    c0.extend_from_slice(&blob);
    let blte_entry = make_blte(&[&c0], &[blob.len()]);
    let first = make_casc_entry(1, &blte_entry);

    let gap = BUF_SIZE + 17;
    let mut big = Vec::with_capacity(gap + first.len() * 2);
    big.extend_from_slice(&first);
    big.resize(gap, 0);
    // Second entry with a different tag across the 4 MiB boundary
    let (cs_ctl, cs_code) = make_raw_cs_pair(0x66);
    let mut blob2 = cs_ctl;
    blob2.extend_from_slice(&cs_code);
    let mut c1 = vec![b'N'];
    c1.extend_from_slice(&blob2);
    let blte_entry2 = make_blte(&[&c1], &[blob2.len()]);
    big.extend_from_slice(&make_casc_entry(2, &blte_entry2));

    let files = [File {
        rel: "Data/data/data.000".into(),
        src: Arc::new(StrictStream(big)),
        live: true,
    }];
    let plan = plan_with(&files, &NroSet::default()).unwrap();
    assert_eq!(plan.jobs.len(), 1);
    let dicts = Dicts::new();
    let ctx = Ctx {
        files: &files,
        dicts: &dicts,
    };
    let out = (plan.jobs[0].run)(&ctx, &mut Vec::new()).unwrap();
    assert_eq!(out.found.len(), 2);
    assert_eq!(out.found[0].stage, "vs");
    assert_eq!(out.found[1].stage, "cs");
}
