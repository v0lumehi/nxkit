//! The renamed ranges of a frame (`crate::rename`), which the frame detail carries from the flush to the encoder.

/// How a frame binds a range that it has a copy of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    /// Written by the GPU, re-initialised by the CPU: the end value goes back in the wait.
    Rename,
    /// Only read: nothing goes back.
    Snapshot,
    /// The GPU merges the copy with guest memory at the start and the end of the frame.
    Merge,
    /// A written range in a judged frame: the GPU reads and writes a copy that it fills from guest memory at the start
    /// (a second copy keeps the start state), and writes the words that it changed back at the end. Guest memory is
    /// written by the CPU only while the frame runs, so the host can tell the CPU's words from the GPU's.
    Probe,
}

/// A renamed range of one frame with the guest bytes at its flush.
#[derive(Debug)]
pub(crate) struct CapRange {
    pub(crate) addr: u64,
    /// The bytes of the range.
    pub(crate) len: u32,
    /// The bytes at the flush: the whole range for `Rename` and `Snapshot`, nothing for the others.
    pub(crate) start: Vec<u8>,
    pub(crate) kind: Kind,
    /// For a merge range, one bit per word: `cmask` the words that the CPU writes in flight, `gmask` the words that only
    /// the GPU writes (`P`: the kernel at the end of the frame writes these into guest memory) and `rmask` the words that
    /// both write (`R`: the host writes the GPU's value back in the wait, if the CPU has not written the word since).
    pub(crate) cmask: Vec<u32>,
    pub(crate) gmask: Vec<u32>,
    pub(crate) rmask: Vec<u32>,
    /// For a merge range: the bytes of the words of `cmask` at the flush, packed, and for each block of 32 words the
    /// number of such words before it.
    pub(crate) cvals: Vec<u32>,
    pub(crate) cprefix: Vec<u32>,
}

/// The renamed ranges of a frame, taken at its flush on the CPU thread.
#[derive(Debug)]
pub(crate) struct Capture {
    /// The fence of the frame.
    pub(crate) fence: u64,
    pub(crate) ranges: Vec<CapRange>,
    /// `NXKIT_DIAG=rename` prints this frame's capture and write-back.
    pub(crate) sampled: bool,
    /// The number of CPU waits so far at the flush. Two frames with the same number have no wait between them.
    pub(crate) epoch: u64,
    /// Snapshots of CPU-written indirect argument records `(guest_addr, bytes, payload)`.
    pub(crate) indirects: Vec<(u64, u32, Vec<u8>)>,
}
