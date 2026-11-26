use super::{Budget, CHUNK, Job, WHOLE, put};
use crate::nup::{FileRec, MAX_IN_MEMORY, Nup};
use crate::storage::FileStorage;
use nxb_model::Sha256;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

/// Builds a valid multi-block zstd frame of `blocks * 128 KiB` using RLE blocks so a >64 MiB payload takes ~2 KiB.
fn multi_block_zstd(blocks: usize) -> (Vec<u8>, [u8; 32], u64) {
    const BLK: usize = 1 << 17;
    let total = (blocks * BLK) as u64;
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0xE0];
    frame.extend_from_slice(&total.to_le_bytes());
    let mut h = Sha256::new();
    let mut chunk = vec![0u8; BLK];
    for i in 0..blocks {
        let byte = (i & 0xFF) as u8;
        let last = u32::from(i + 1 == blocks);
        let hdr = ((BLK as u32) << 3) | (1 << 1) | last;
        frame.extend_from_slice(&hdr.to_le_bytes()[..3]);
        frame.push(byte);
        chunk.fill(byte);
        h.update(&chunk);
    }
    (frame, h.finalize().0, total)
}

#[test]
fn streaming_zstd_unpack_handles_large_files_and_releases_budget_on_error() {
    let dir = std::env::temp_dir().join(format!("nxb-nup-stream-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let (frame, sha, total) = multi_block_zstd((WHOLE as usize >> 17) + 8);
    assert!(total > WHOLE);

    let container = dir.join("big.nup");
    let off = 0x100u64;
    let mut raw = vec![0u8; off as usize];
    raw.extend_from_slice(&frame);
    std::fs::write(&container, &raw).unwrap();

    let st = Arc::new(FileStorage::open(&container).unwrap());
    let mut nup = Nup {
        path: container.clone(),
        st,
        toc_sha: [0u8; 32],
        packages: Vec::new(),
        dirs: Vec::new(),
        files: vec![
            FileRec {
                name: b"big.ucas".to_vec(),
                size: total,
                sha,
                off,
                stored: frame.len() as u64,
                zstd: true,
            },
            FileRec {
                name: b"huge.bin".to_vec(),
                size: MAX_IN_MEMORY + 1,
                sha: [0u8; 32],
                off,
                stored: frame.len() as u64,
                zstd: true,
            },
        ],
        slots: Vec::new(),
    };

    // read_checked refuses files above MAX_IN_MEMORY while streaming unpack handles > WHOLE.
    let err = nup.read_checked(1, true).unwrap_err().to_string();
    assert!(err.contains("too large to read into memory"), "{err}");

    let budget = Budget {
        used: Mutex::new(0),
        cv: Condvar::new(),
        cap: CHUNK as u64 * 2,
    };
    let out_path: PathBuf = dir.join("big.ucas");
    let job = Job {
        file: 0,
        path: out_path.clone(),
        rel: "big.ucas".to_string(),
        split: false,
    };
    let (mem, held) = put(&nup, &job, true, &budget).unwrap();
    assert!(mem.is_none());
    assert_eq!(held, 0);
    assert_eq!(*budget.used.lock().unwrap(), 0);
    assert_eq!(std::fs::metadata(&out_path).unwrap().len(), total);

    // Hash mismatch fails and still returns the budget.
    nup.files[0].sha[0] ^= 0xFF;
    let err = put(&nup, &job, true, &budget).unwrap_err().to_string();
    assert!(err.contains("SHA-256"), "{err}");
    assert_eq!(*budget.used.lock().unwrap(), 0);

    // Truncated zstd frame fails and still returns the budget.
    nup.files[0].sha = sha;
    nup.files[0].stored -= 1;
    assert!(put(&nup, &job, true, &budget).is_err());
    assert_eq!(*budget.used.lock().unwrap(), 0);

    let _ = std::fs::remove_dir_all(&dir);
}
