//! In-place resource edits for an exact, recorded source asset.

use nxb_model::{Digest, FileDigest};

/// Fixes the declared little-endian float words to `scale` without changing the layout.
///
/// The source digest must match before any edit. Offsets must be nonempty, aligned,
/// strictly increasing and in bounds. All checks precede the first write, so an error
/// leaves `data` unchanged. A scale is finite and greater than zero, up to one.
pub fn fixed_scale(
    data: &mut [u8],
    source: Digest,
    offsets: &[usize],
    scale: f32,
) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
        return Err("scale must be finite and greater than 0, up to 1".into());
    }
    if offsets.is_empty() {
        return Err("the resource has no declared scale words".into());
    }
    let mut previous = None;
    for &at in offsets {
        if at % 4 != 0
            || at.checked_add(4).is_none_or(|end| end > data.len())
            || previous.is_some_and(|last| last >= at)
        {
            return Err(format!("invalid scale word offset {at}"));
        }
        previous = Some(at);
    }
    let actual = FileDigest::of(data).sha256;
    if actual != source {
        return Err(format!(
            "unsupported resource digest {actual}, expected {source}"
        ));
    }
    let bytes = scale.to_le_bytes();
    for &at in offsets {
        data[at..at + 4].copy_from_slice(&bytes);
    }
    Ok(())
}
