//! Resource facts for an opt-in fixed-resolution overlay.

use nxb_model::Digest;

/// The exact asset and shared float words that control one game's resolution scale.
///
/// A title supplies the paths and source digests. The core reads and edits the files.
/// An unrecognized archive, dictionary or member is not patched. These offsets apply
/// only to the recorded member bytes, not to a guessed layout of another update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedResolution {
    /// The compressed SARC path from the RomFS root.
    pub archive: &'static str,
    /// The SHA-256 of that compressed archive.
    pub archive_sha256: Digest,
    /// The member whose scale words change.
    pub member: &'static str,
    /// The SHA-256 of the original, uncompressed member.
    pub member_sha256: Digest,
    /// Shared little-endian float offsets within the member, in increasing order.
    pub scale_offsets: &'static [usize],
    /// The compressed SARC containing the game's zstd dictionaries.
    pub dictionary_archive: &'static str,
    /// The dictionary member used by the source archive.
    pub dictionary_member: &'static str,
    /// The SHA-256 of the uncompressed dictionary bytes.
    pub dictionary_sha256: Digest,
}
