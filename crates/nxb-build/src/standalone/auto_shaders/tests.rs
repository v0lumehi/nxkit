use super::*;
#[cfg(feature = "air-front")]
#[allow(
    dead_code,
    reason = "the staged translator test uses the other record cases"
)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nxb-stages/tests/domain_job/fixture.rs"
    ));
}

#[cfg(feature = "air-front")]
#[test]
fn encoded_direct_patch_records_use_normal_standalone_packaging() {
    const CONTROL_SIZE_FIELD: std::ops::Range<usize> = 0x28..0x2c;
    const CODE_SIZE_FIELD: std::ops::Range<usize> = 0x2c..0x30;
    const NVN1_CODE_PREFIX_BYTES: u32 = 0x30;
    let mut output = records::Outputs::default();
    for (name, stage, key, code) in [
        (
            "vs_cloud.bin",
            "vs",
            "0101010101010101",
            fixture::vs_capture(),
        ),
        (
            "ds_cloud.bin",
            "ds",
            "0303030303030303",
            fixture::ds_partial_x(),
        ),
        (
            "ps_cloud.bin",
            "ps",
            "0202020202020202",
            fixture::fs_xyzw(0),
        ),
        (
            "ps_cloud.bin",
            "ps",
            "0404040404040404",
            fixture::fs_xyzw(1),
        ),
    ] {
        let mut record = fixture::record(&code);
        record[..8].copy_from_slice(b"BTCDBTCD");
        record[CONTROL_SIZE_FIELD].copy_from_slice(&(fixture::CONTROL_LEN as u32).to_le_bytes());
        record[CODE_SIZE_FIELD]
            .copy_from_slice(&(NVN1_CODE_PREFIX_BYTES + code.len() as u32).to_le_bytes());
        output
            .packages
            .entry(name.into())
            .or_default()
            .extend(record);
        let previous = usize::from(key == "0404040404040404");
        output
            .rows
            .push((name.into(), previous, stage, key.into(), "synthetic".into()));
    }
    // The normal standalone builder consumes the same package bytes and map rows
    // as after discovery. No supplied MSL, resource facts, or guessed pair table.
    let files: BTreeMap<_, _> = compile_packages(output, vec![], false)
        .unwrap()
        .into_iter()
        .collect();
    for key in [
        "0101010101010101",
        "0202020202020202",
        "0303030303030303",
        "0404040404040404",
    ] {
        let base = format!("shaders/{key}_b");
        assert!(files[&format!("{base}.metallib")].starts_with(b"MTLB"));
        let json = std::str::from_utf8(&files[&format!("{base}.json")]).unwrap();
        let reflection = parse(json).unwrap();
        let stage = match key {
            "0101010101010101" => "vs",
            "0303030303030303" => "ds",
            _ => "ps",
        };
        assert_eq!(
            reflection.get("stage").and_then(J::as_str),
            Some(stage),
            "{base}"
        );
        assert_eq!(
            reflection.get("entry").and_then(J::as_str),
            Some("main0"),
            "{base}"
        );
    }
}

#[test]
fn rejects_unsafe_names_and_oversized_nso() {
    assert!(safe_name("../main").is_err());
    assert!(safe_name("main").is_ok());
    assert!(check_nso_size(b"NSO0").is_err());
    let mut nso = vec![0; 0x40];
    nso[..4].copy_from_slice(b"NSO0");
    nso[0x18..0x1c].copy_from_slice(&((INPUT_LIMIT as u32) + 1).to_le_bytes());
    assert!(check_nso_size(&nso).is_err());
}

#[test]
fn incomplete_discovery_does_not_produce_a_successful_shader_stage() {
    let result = compile(
        vec![],
        &BTreeSet::new(),
        vec![],
        vec!["[RED] /content.pak: its shaders are left out".into()],
        false,
    );
    assert!(
        result
            .unwrap_err()
            .starts_with("unsupported: incomplete shader discovery")
    );
}

#[test]
fn empty_discovery_produces_tables() {
    let files = discover(&[], &[], false).unwrap();
    assert!(
        files
            .iter()
            .any(|(name, _)| name == "shaders/shader_map.tsv")
    );
    assert!(!files.iter().any(|(name, _)| name.ends_with(".metallib")));
}

#[test]
fn file_src_reads_concurrently_and_bounds_checks() {
    use src::Src;
    let dir = std::env::temp_dir().join(format!("nxb-filesrc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("blob.bin");
    let payload: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    std::fs::write(&path, &payload).unwrap();
    let fs = Arc::new(FileSrc::open(path, payload.len() as u64).unwrap());
    assert!(fs.bytes().is_none());
    assert_eq!(fs.len(), payload.len() as u64);
    assert!(fs.read_at(payload.len() as u64, &mut [0; 1]).is_err());
    std::thread::scope(|s| {
        for t in 0..4 {
            let fs = fs.clone();
            let payload = &payload;
            s.spawn(move || {
                let mut head = [0u8; 16];
                fs.read_at(0, &mut head).unwrap();
                assert_eq!(&head, &payload[..16]);
                let mut mid = vec![0u8; 64];
                let off = 100 + t * 32;
                fs.read_at(off as u64, &mut mid).unwrap();
                assert_eq!(&mid, &payload[off..off + 64]);
            });
        }
    });
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn discover_dir_writes_tables_to_out_dir() {
    let dir = std::env::temp_dir().join(format!("nxb-discover-dir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (romfs, overlay, out) = (dir.join("romfs"), dir.join("overlay"), dir.join("run"));
    std::fs::create_dir_all(&romfs).unwrap();
    std::fs::create_dir_all(&overlay).unwrap();
    std::fs::write(romfs.join("a.bin"), b"hello").unwrap();
    std::fs::write(overlay.join("b.bin"), b"world").unwrap();
    std::fs::write(overlay.join(crate::resource_patch::STAMP), b"stamp").unwrap();
    let count = discover_dir(&[], &romfs, Some(&overlay), false, 2, &out).unwrap();
    assert_eq!(count, 0);
    assert!(out.join("shaders/shader_map.tsv").is_file());
    assert!(out.join("shaders/shader_facts.tsv").is_file());
    assert!(out.join("shaders/shader_pairs.tsv").is_file());
    assert!(out.join("shaders/pso_pairs.tsv").is_file());
    std::fs::remove_dir_all(&dir).unwrap();
}
