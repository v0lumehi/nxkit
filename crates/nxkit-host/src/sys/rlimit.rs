//! Raises the open-file soft limit (`RLIMIT_NOFILE`) at host startup.
//!
//! Apps launched by `launchd` (from Finder or a launcher) start with a soft limit of 256 descriptors while
//! `rlim_max` is `RLIM_INFINITY`. Parallel shader/pipeline prewarm, archive shards, and AppKit's lazy `dlopen`
//! of `libobjc-trampolines.dylib` easily exhaust 256 descriptors (`EMFILE`), causing `TrampolinePointerWrapper`
//! or Metal binary-archive loading to fail fatally.

use std::ffi::c_int;

const RLIMIT_NOFILE: c_int = 8;
const TARGETS: &[u64] = &[16_384, 10_240, 4_096];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RLimit {
    rlim_cur: u64,
    rlim_max: u64,
}

unsafe extern "C" {
    fn getrlimit(resource: c_int, rlp: *mut RLimit) -> c_int;
    fn setrlimit(resource: c_int, rlp: *const RLimit) -> c_int;
}

fn candidates(cur: u64, max: u64) -> impl Iterator<Item = u64> {
    let mut prev = None;
    TARGETS.iter().filter_map(move |&t| {
        let want = t.min(max);
        if want <= cur || prev.is_some_and(|p| want >= p) {
            return None;
        }
        prev = Some(want);
        Some(want)
    })
}

/// Raises the process open-file soft limit (`RLIMIT_NOFILE`) toward `16384` (never above `rlim_max`, never
/// lowering an existing higher limit). Logs a one-line warning if the limit stays below `4096`.
pub fn raise_open_files() {
    let mut rl = RLimit::default();
    // SAFETY: `rl` is a valid writable `struct rlimit` of `<sys/resource.h>`.
    if unsafe { getrlimit(RLIMIT_NOFILE, &mut rl) } != 0 {
        return;
    }
    if rl.rlim_cur >= TARGETS[0] {
        return;
    }
    for want in candidates(rl.rlim_cur, rl.rlim_max) {
        let next = RLimit {
            rlim_cur: want,
            rlim_max: rl.rlim_max,
        };
        // SAFETY: `next` is a valid readable `struct rlimit` with `rlim_cur <= rlim_max`.
        if unsafe { setrlimit(RLIMIT_NOFILE, &next) } == 0 {
            return;
        }
    }
    if rl.rlim_cur < 4_096 {
        eprintln!(
            "nxkit-host: open file limit {} could not be raised",
            rl.rlim_cur
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rlimit_layout_matches_sdk() {
        assert_eq!(std::mem::size_of::<RLimit>(), 16);
    }

    #[test]
    fn candidates_raise_clamped_and_never_lower() {
        assert_eq!(
            candidates(256, u64::MAX).collect::<Vec<_>>(),
            [16_384, 10_240, 4_096]
        );
        assert_eq!(candidates(256, 8_192).collect::<Vec<_>>(), [8_192, 4_096]);
        assert!(candidates(256, 256).collect::<Vec<_>>().is_empty());
        assert!(candidates(65_536, u64::MAX).collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn raise_open_files_reaches_target_within_hard_limit() {
        raise_open_files();
        let mut rl = RLimit::default();
        // SAFETY: `rl` is a valid writable `struct rlimit`.
        assert_eq!(unsafe { getrlimit(RLIMIT_NOFILE, &mut rl) }, 0);
        assert!(rl.rlim_cur >= 4_096.min(rl.rlim_max));
    }
}
