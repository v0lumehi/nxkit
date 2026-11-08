//! Optional libzstd 1.5.7 compression with the reference's one-shot dictionary API.
//!
//! The library is loaded only when requested. It is a trusted native dependency,
//! not a game resource. No application or shader build links against libzstd.

use std::path::Path;

/// Compresses with `ZSTD_compress_usingDict`, not the CLI's streaming encoder.
///
/// `library` names the installed, trusted libzstd 1.5.7 library. Its version is
/// checked before compression. Loading is currently supported on macOS only.
pub fn compress_using_dict(
    library: &Path,
    source: &[u8],
    dictionary: &[u8],
    level: i32,
) -> Result<Vec<u8>, String> {
    imp::compress(library, source, dictionary, level)
}

#[cfg(target_vendor = "apple")]
mod imp {
    use super::*;
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::ptr::NonNull;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    struct Library(NonNull<c_void>);

    fn loader_error() -> String {
        // SAFETY: dlerror returns null or the loader's NUL-terminated error string.
        let error = unsafe { dlerror() };
        if error.is_null() {
            "dynamic library operation failed".into()
        } else {
            // SAFETY: the loader owns the string until the next loader operation.
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        }
    }

    impl Library {
        fn open(path: &Path) -> Result<Self, String> {
            let name = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| "library path contains NUL")?;
            // SAFETY: name is terminated. The caller selects a trusted native library.
            // Darwin dlfcn.h defines RTLD_NOW=2 and RTLD_LOCAL=4.
            let handle = unsafe { dlopen(name.as_ptr(), 2 | 4) };
            NonNull::new(handle).map(Self).ok_or_else(loader_error)
        }

        fn symbol(&self, name: &CStr) -> Result<*mut c_void, String> {
            // SAFETY: self owns an open handle and name is terminated.
            let address = unsafe { dlsym(self.0.as_ptr(), name.as_ptr()) };
            if address.is_null() {
                Err(loader_error())
            } else {
                Ok(address)
            }
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: contexts and function calls finish before this owned handle closes.
            unsafe { dlclose(self.0.as_ptr()) };
        }
    }

    type Free = unsafe extern "C" fn(*mut c_void) -> usize;

    struct Context<'a> {
        pointer: NonNull<c_void>,
        free: Free,
        _library: &'a Library,
    }

    impl Drop for Context<'_> {
        fn drop(&mut self) {
            // SAFETY: the context belongs to this still-open library and is freed once.
            unsafe { (self.free)(self.pointer.as_ptr()) };
        }
    }

    pub(super) fn compress(
        library: &Path,
        source: &[u8],
        dictionary: &[u8],
        level: i32,
    ) -> Result<Vec<u8>, String> {
        let library = Library::open(library)?;
        macro_rules! load {
            ($name:literal, $signature:ty) => {{
                // SAFETY: these exported signatures come from libzstd's public zstd.h.
                // The owned library outlives every call and context below.
                unsafe { std::mem::transmute::<*mut c_void, $signature>(library.symbol($name)?) }
            }};
        }
        let version = load!(c"ZSTD_versionNumber", unsafe extern "C" fn() -> u32);
        // SAFETY: version is the public no-argument version query.
        let version = unsafe { version() };
        if version != 10507 {
            return Err(format!(
                "libzstd 1.5.7 is required, got version number {version}"
            ));
        }
        let create = load!(c"ZSTD_createCCtx", unsafe extern "C" fn() -> *mut c_void);
        let free = load!(c"ZSTD_freeCCtx", Free);
        let bound = load!(c"ZSTD_compressBound", unsafe extern "C" fn(usize) -> usize);
        let is_error = load!(c"ZSTD_isError", unsafe extern "C" fn(usize) -> u32);
        let error_name = load!(
            c"ZSTD_getErrorName",
            unsafe extern "C" fn(usize) -> *const c_char
        );
        let compress = load!(
            c"ZSTD_compress_usingDict",
            unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                usize,
                *const c_void,
                usize,
                *const c_void,
                usize,
                c_int,
            ) -> usize
        );
        // SAFETY: create allocates an independent context or returns null.
        let pointer =
            NonNull::new(unsafe { create() }).ok_or("libzstd context allocation failed")?;
        let context = Context {
            pointer,
            free,
            _library: &library,
        };
        // SAFETY: the size query accesses no buffers.
        let capacity = unsafe { bound(source.len()) };
        // SAFETY: is_error accepts every size_t result.
        if unsafe { is_error(capacity) } != 0 {
            return Err("libzstd input is too large".into());
        }
        let mut output = Vec::<u8>::with_capacity(capacity);
        // SAFETY: the independent context is live. Each input has the stated length.
        // output reserves capacity writable bytes. No Rust code reads them yet.
        let size = unsafe {
            compress(
                context.pointer.as_ptr(),
                output.as_mut_ptr().cast(),
                capacity,
                source.as_ptr().cast(),
                source.len(),
                dictionary.as_ptr().cast(),
                dictionary.len(),
                level,
            )
        };
        // SAFETY: is_error accepts every size_t result.
        if unsafe { is_error(size) } != 0 {
            // SAFETY: the public error query returns a static terminated string.
            let error = unsafe { CStr::from_ptr(error_name(size)) };
            return Err(format!("libzstd compression: {}", error.to_string_lossy()));
        }
        if size > capacity {
            return Err("libzstd returned an invalid output size".into());
        }
        // SAFETY: a successful compression initialized exactly size bytes within capacity.
        unsafe { output.set_len(size) };
        Ok(output)
    }
}

#[cfg(not(target_vendor = "apple"))]
mod imp {
    use super::*;
    pub(super) fn compress(_: &Path, _: &[u8], _: &[u8], _: i32) -> Result<Vec<u8>, String> {
        Err("fixed-scale resource recompression requires macOS".into())
    }
}
