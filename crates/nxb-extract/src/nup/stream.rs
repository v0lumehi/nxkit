//! Streaming zstd extraction for large NUP members.

use std::path::Path;

use nxb_model::Sha256;
use nxb_split::SplitError;

use super::{Nup, bad};
use crate::error::{ExtractError, Result};
use crate::storage::Storage;

/// Stream-decompresses a zstd-compressed file larger than `WHOLE` directly to `out` without buffering the whole
/// uncompressed file in memory, verifying size and (when `verify` is set) SHA-256.
pub(super) fn put_zstd_stream_into(
    n: &Nup,
    file: usize,
    path: &Path,
    rel: &str,
    out: &mut std::fs::File,
    verify: bool,
) -> Result<()> {
    const BUF: usize = 1 << 20;
    let f = &n.files[file];
    let werr = |e: std::io::Error| ExtractError::io(format!("write {}", path.display()), e);
    let mut src_at = 0u64;
    let mut ahead = Vec::new();
    let mut ahead_at = 0usize;
    let mut read_err: Option<ExtractError> = None;
    let mut write_err: Option<ExtractError> = None;
    let mut input = |buf: &mut [u8]| -> std::result::Result<(), SplitError> {
        let mut done = 0usize;
        while done < buf.len() {
            if ahead_at == ahead.len() {
                let k = (f.stored - src_at).min(BUF as u64) as usize;
                if k == 0 {
                    return Err(SplitError::Format("input ends inside a frame".into()));
                }
                ahead.resize(k, 0);
                if let Err(e) = n.st.read_at(f.off + src_at, &mut ahead) {
                    read_err = Some(e);
                    return Err(SplitError::Format("read failed".into()));
                }
                src_at += k as u64;
                ahead_at = 0;
            }
            let m = (ahead.len() - ahead_at).min(buf.len() - done);
            buf[done..done + m].copy_from_slice(&ahead[ahead_at..ahead_at + m]);
            ahead_at += m;
            done += m;
        }
        Ok(())
    };
    let mut h = verify.then(Sha256::new);
    let mut written = 0u64;
    let mut sink = |chunk: &[u8]| -> std::result::Result<(), SplitError> {
        if let Some(h) = h.as_mut() {
            h.update(chunk);
        }
        if let Err(e) = std::io::Write::write_all(out, chunk).map_err(werr) {
            write_err = Some(e);
            return Err(SplitError::Format("write failed".into()));
        }
        written += chunk.len() as u64;
        Ok(())
    };
    if let Err(e) = nxb_split::zstd::decompress_stream(&mut input, f.stored, f.size, &mut sink) {
        if let Some(err) = read_err.or(write_err) {
            return Err(err);
        }
        return Err(bad(&n.path, format!("{rel}: {e}")));
    }
    if written != f.size || h.is_some_and(|h| h.finalize().0 != f.sha) {
        return Err(bad(
            &n.path,
            format!("{rel} does not match its size or SHA-256 (damaged file)"),
        ));
    }
    Ok(())
}
