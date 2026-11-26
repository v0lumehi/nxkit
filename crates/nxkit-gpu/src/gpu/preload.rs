//! The preload thread: the uniform snapshots of a frame go to the scratch memory once, before the chunks start.

use std::thread::JoinHandle;

use super::*;

/// The scratch address of each uniform snapshot, by the address of its bytes.
pub(super) type Preloaded = FastMap<usize, u64>;

/// Starts the thread that pushes every uniform snapshot of `detail` to `scratch`.
pub(super) fn start(
    detail: &FrameDetail,
    scratch: &Arc<Scratch>,
) -> std::io::Result<Option<JoinHandle<Preloaded>>> {
    let mut seen: FastSet<usize> = FastSet::default();
    let mut snaps: Vec<Arc<[u8]>> = Vec::new();
    let mut prev_draw = 0usize;
    for d in &detail.draws {
        let ptr = Arc::as_ptr(d) as usize;
        if ptr == prev_draw {
            continue;
        }
        prev_draw = ptr;
        for u in d.ubos.iter().chain(d.sbufs.iter()) {
            if let Some(b) = &u.snapshot
                && seen.insert(b.as_ptr() as usize)
            {
                snaps.push(Arc::clone(b));
            }
        }
    }
    for d in &detail.dispatches {
        for u in &d.ubos {
            if let Some(b) = &u.snapshot
                && seen.insert(b.as_ptr() as usize)
            {
                snaps.push(Arc::clone(b));
            }
        }
    }
    if snaps.is_empty() {
        return Ok(None);
    }
    let scratch = scratch.clone();
    std::thread::Builder::new()
        .name("nxkit-preload".into())
        .spawn(move || {
            // Every uniform snapshot of the frame goes to the scratch memory once. The chunks then only look the
            // addresses up. When each chunk uploaded on first use, all chunk threads locked the shared map twice per
            // uniform of every draw.
            let mut preloaded: Preloaded =
                FastMap::with_capacity_and_hasher(snaps.len(), Default::default());
            for bytes in &snaps {
                if let Ok(a) = scratch.push(bytes) {
                    preloaded.insert(bytes.as_ptr() as usize, a);
                }
            }
            preloaded
        })
        .map(Some)
}

/// The addresses the thread collected, or none when it did not start or panicked.
pub(super) fn finish(thread: std::io::Result<Option<JoinHandle<Preloaded>>>) -> Preloaded {
    match thread {
        Ok(Some(h)) => h.join().unwrap_or_default(),
        _ => FastMap::default(),
    }
}
