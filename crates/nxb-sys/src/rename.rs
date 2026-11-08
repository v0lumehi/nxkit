//! Atomic publication without replacing a path another process already created.

use std::io;
use std::path::Path;

/// Renames `src` to `dst` only when `dst` does not exist, including an empty directory.
///
/// Apple platforms use `renamex_np(RENAME_EXCL)`. Other platforms return `Unsupported`
/// rather than fall back to a check followed by a replacing rename.
pub fn exclusive(src: &Path, dst: &Path) -> io::Result<()> {
    imp::exclusive(src, dst)
}

#[cfg(target_vendor = "apple")]
mod imp {
    use std::ffi::{CString, c_char};
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    const RENAME_EXCL: u32 = 0x0000_0004;

    unsafe extern "C" {
        fn renamex_np(src: *const c_char, dst: *const c_char, flags: u32) -> i32;
    }

    pub(super) fn exclusive(src: &Path, dst: &Path) -> io::Result<()> {
        let src = CString::new(src.as_os_str().as_bytes())?;
        let dst = CString::new(dst.as_os_str().as_bytes())?;
        // SAFETY: both NUL-terminated paths live through the call. RENAME_EXCL makes
        // destination existence and the rename one atomic filesystem operation.
        if unsafe { renamex_np(src.as_ptr(), dst.as_ptr(), RENAME_EXCL) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
mod imp {
    use std::io;
    use std::path::Path;

    pub(super) fn exclusive(_src: &Path, _dst: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic exclusive rename is not supported on this platform",
        ))
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp() -> io::Result<std::path::PathBuf> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "nxb-rename-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path)?;
        Ok(path)
    }

    #[test]
    fn an_empty_destination_created_before_publication_is_preserved() {
        let root = temp().unwrap();
        let src = root.join("staged");
        let dst = root.join("other-process");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("data"), b"candidate").unwrap();
        std::fs::create_dir(&dst).unwrap();
        assert_eq!(
            exclusive(&src, &dst).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(src.join("data")).unwrap(), b"candidate");
        assert!(std::fs::read_dir(&dst).unwrap().next().is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_absent_destination_receives_the_complete_directory() {
        let root = temp().unwrap();
        let src = root.join("staged");
        let dst = root.join("published");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("data"), b"complete").unwrap();
        exclusive(&src, &dst).unwrap();
        assert!(!src.exists());
        assert_eq!(std::fs::read(dst.join("data")).unwrap(), b"complete");
        std::fs::remove_dir_all(root).unwrap();
    }
}
