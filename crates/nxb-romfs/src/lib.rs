//! RomFS metadata and verified resource edits.
//!
//! [`patch`] edits declared float words in a source asset with a known digest.
//! The image tables, layout and identity modules remain planned.

pub mod patch;

/// Nintendo RomFS directory and file tables.
pub mod tables {}

/// `ImageLayout::plan`: offsets, index and header of a `DMROMFS1` image.
pub mod layout {}

/// `romfs.id` of a layout.
pub mod identity {}
