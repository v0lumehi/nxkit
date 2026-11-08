use super::*;

fn temp() -> TempDir {
    let root = std::env::var_os("NXB_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    TempDir::new_in(&root).unwrap()
}

fn arguments(source: &Path, out: &Path, scale: &str) -> Vec<String> {
    [
        "fixed-scale",
        "--title-id",
        "0400486024DEC000",
        "--romfs",
        source.to_str().unwrap(),
        "--scale",
        scale,
        "--out",
        out.to_str().unwrap(),
        "--zstd-library",
        "unused-native-library",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

#[test]
fn unsupported_assets_publish_no_overlay_and_leave_the_input_unchanged() {
    let t = temp();
    let source = t.path().join("source");
    std::fs::create_dir_all(source.join("Pack")).unwrap();
    let archive = source.join("Pack/Bootup.Nin_Ounce_NVN2.pack.zs");
    let bytes = b"synthetic unsupported archive, not game data";
    std::fs::write(&archive, bytes).unwrap();
    let out = t.path().join("overlay");
    assert_eq!(main(&arguments(&source, &out, "0.666667")), 4);
    assert!(!out.exists());
    assert_eq!(std::fs::read(archive).unwrap(), bytes);
}

#[test]
fn an_existing_output_is_not_replaced() {
    let t = temp();
    let out = t.path().join("existing");
    std::fs::create_dir(&out).unwrap();
    std::fs::write(out.join("keep.txt"), "user data").unwrap();
    assert_eq!(main(&arguments(&t.path().join("missing"), &out, "0.75")), 1);
    assert_eq!(
        std::fs::read_to_string(out.join("keep.txt")).unwrap(),
        "user data"
    );
}

#[test]
fn invalid_scale_and_duplicate_options_are_bad_arguments() {
    let t = temp();
    let source = t.path().join("missing");
    let out = t.path().join("overlay");
    for scale in ["0", "-1", "NaN", "inf", "1.01"] {
        assert_eq!(main(&arguments(&source, &out, scale)), 2);
        assert!(!out.exists());
    }
    let mut args = arguments(&source, &out, "0.75");
    args.extend(["--scale".into(), "0.5".into()]);
    assert_eq!(main(&args), 2);
    assert!(!out.exists());
}

#[test]
fn a_title_without_recorded_resource_edits_is_not_patched() {
    let t = temp();
    let out = t.path().join("overlay");
    let mut args = arguments(t.path(), &out, "0.75");
    args[2] = "01007EF00011E000".into();
    assert_eq!(main(&args), 4);
    assert!(!out.exists());
}

/// Oracle comparison with the retained overlay. No game data lives in this tree.
/// Set NXB_PARITY_RESOLUTION_ROMFS to the source image or its source-slices folder.
/// Set NXB_PARITY_RESOLUTION_REFERENCE to the retained compressed Bootup overlay.
/// Set NXB_PARITY_ZSTD_LIBRARY to the trusted libzstd 1.5.7 library.
#[test]
fn real_overlay_matches_reference() {
    let (Some(source), Some(reference), Some(library)) = (
        std::env::var_os("NXB_PARITY_RESOLUTION_ROMFS"),
        std::env::var_os("NXB_PARITY_RESOLUTION_REFERENCE"),
        std::env::var_os("NXB_PARITY_ZSTD_LIBRARY"),
    ) else {
        eprintln!(
            "skipped: set NXB_PARITY_RESOLUTION_ROMFS, NXB_PARITY_RESOLUTION_REFERENCE and NXB_PARITY_ZSTD_LIBRARY"
        );
        return;
    };
    let catalog = nxtitles::catalog().unwrap();
    let patch = catalog
        .resolve(&TitleFacts::new(TitleId(0x0400_4860_24DE_C000)))
        .fixed_resolution()
        .unwrap()
        .value;
    let t = temp();
    let out = t.path().join("overlay");
    prepare(
        Path::new(&source),
        &out,
        Path::new(&library),
        patch,
        0.666667,
    )
    .unwrap();
    let actual = std::fs::read(out.join(patch.archive)).unwrap();
    let expected = std::fs::read(Path::new(&reference)).unwrap();
    assert!(
        actual == expected,
        "compressed overlay differs from the reference"
    );
}
