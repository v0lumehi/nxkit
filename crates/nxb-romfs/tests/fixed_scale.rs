//! Verified resource edits preserve every byte outside the declared float words.

use nxb_model::FileDigest;
use nxb_romfs::patch::fixed_scale;

fn original() -> Vec<u8> {
    vec![
        b'Y', b'B', 7, 0, 0x11, 0x22, 0x33, 0x44, 0, 0, 0, 0x3f, // 0.5
        95, 0, 0, 0, // A threshold, not a scale.
        0, 0, 0x80, 0x3f, // 1.0
        108, 0, 0, 0, // Another threshold.
    ]
}

#[test]
fn only_the_selected_float_words_change() {
    let mut data = original();
    let digest = FileDigest::of(&data).sha256;
    fixed_scale(&mut data, digest, &[8, 16], 0.75).unwrap();
    assert_eq!(
        data,
        [
            b'Y', b'B', 7, 0, 0x11, 0x22, 0x33, 0x44, 0, 0, 0x40, 0x3f, 95, 0, 0, 0, 0, 0, 0x40,
            0x3f, 108, 0, 0, 0,
        ]
    );
}

#[test]
fn changed_source_bytes_are_refused_without_mutation() {
    let mut data = original();
    let digest = FileDigest::of(&data).sha256;
    data[4] ^= 1;
    let before = data.clone();
    assert!(fixed_scale(&mut data, digest, &[8, 16], 0.75).is_err());
    assert_eq!(data, before);
}

#[test]
fn all_offsets_are_checked_before_any_word_is_written() {
    for offsets in [
        vec![8, 24],
        vec![8, usize::MAX],
        vec![8, 17],
        vec![8, 8],
        vec![16, 8],
        vec![],
    ] {
        let mut data = original();
        let before = data.clone();
        let digest = FileDigest::of(&data).sha256;
        assert!(
            fixed_scale(&mut data, digest, &offsets, 0.75).is_err(),
            "{offsets:?}"
        );
        assert_eq!(data, before, "{offsets:?}");
    }
}

#[test]
fn invalid_scales_leave_the_source_unchanged() {
    for scale in [0.0, -0.5, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut data = original();
        let before = data.clone();
        let digest = FileDigest::of(&data).sha256;
        assert!(
            fixed_scale(&mut data, digest, &[8, 16], scale).is_err(),
            "{scale}"
        );
        assert_eq!(data, before, "{scale}");
    }
}
