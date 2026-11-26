//! NCZ: an NCA compressed by `nsz` (the files of an NSZ or XCZ). Layout, as `nsz` writes it:
//!
//! - the NCA's first 0x4000 bytes as they were (header and section headers, still encrypted).
//! - `NCZSECTN`, a section count and per section (offset, size, crypto type, padding, key, counter), 0x40 bytes each:
//!   the sections tile the rest of the NCA, and the data after the header is stored *decrypted*.
//! - either `NCZBLOCK` (block size, block count, the compressed size of each block, then the blocks: one zstd frame
//!   each, or the bytes as they are when compression did not make them smaller) or one zstd stream (solid).
//!
//! [`NczStorage`] shows the original NCA: the data is decompressed and, in sections with crypto type 3 or 4, encrypted
//! again with AES-CTR (what `nsz -D` writes), so everything above it reads the NCA as from an NSP. Blocks are decoded
//! on demand. A solid stream cannot be read at random, so it is decoded once on first use into memory (up to
//! 64 MiB) or an unlinked temporary file in the spill folder ([`set_spill_dir`]).

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::aes::Aes128;
use crate::error::{ExtractError, Result};
use crate::storage::{FileStorage, MemStorage, Storage};

/// The NCA header and section headers that `nsz` keeps as they were.
const HEADER: u64 = 0x4000;
/// Largest NCA accepted (the declared size, the data must still be there to decode).
const MAX_NCA: u64 = 256 << 30;
/// Solid streams up to this size are decoded into memory.
const SPILL_MEM: u64 = 64 << 20;

static SPILL_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
static SPILL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Where decoded solid streams go when they are too big for memory (an NCA of 20 GB needs 20 GB there). The default
/// is the system's temporary folder.
pub fn set_spill_dir(dir: &Path) {
    if let Ok(mut g) = SPILL_DIR.lock() {
        *g = Some(dir.to_path_buf());
    }
}

fn bad(id: &str, m: impl std::fmt::Display) -> ExtractError {
    ExtractError::format(format!("{id}.ncz: {m}"))
}

fn le64(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8).map_or(0, |s| {
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        u64::from_le_bytes(a)
    })
}

/// One section of the section table.
struct Sec {
    start: u64,
    end: u64,
    stream_off: u64,
    ctr: bool,
    key: [u8; 16],
    prefix: [u8; 8],
}

#[derive(Clone)]
struct SolidDecoded {
    storage: Arc<dyn Storage>,
    packed: bool,
    encrypted: bool,
}

enum Body {
    Blocks {
        src: Arc<dyn Storage>,
        shift: u32,
        total: u64,
        packed: bool,
        /// Start of each block in `src`, then the end of the last.
        at: Vec<u64>,
        cache: Mutex<VecDeque<(u64, Arc<Vec<u8>>)>>,
    },
    Solid {
        src: Arc<dyn Storage>,
        at: u64,
        stream_sum: u64,
        decoded: Mutex<Option<SolidDecoded>>,
    },
}

/// The NCA inside an NCZ, as it was before compression.
pub struct NczStorage {
    id: String,
    header: Vec<u8>,
    secs: Vec<Sec>,
    size: u64,
    body: Body,
}

impl NczStorage {
    /// Opens the NCZ file `src` (the entry `<id>.ncz` of the package).
    pub fn open(src: Arc<dyn Storage>, id: &str) -> Result<NczStorage> {
        let n = src.size();
        if n < HEADER + 16 {
            return Err(bad(id, "shorter than its header"));
        }
        let header = src.read_vec(0, HEADER as usize)?;
        let t = src.read_vec(HEADER, 16)?;
        if &t[..8] != b"NCZSECTN" {
            return Err(bad(id, "no NCZSECTN table"));
        }
        let count = le64(&t, 8);
        if count == 0 || count > 1 << 20 {
            return Err(bad(id, format!("{count} sections")));
        }
        let table = src
            .read_vec(HEADER + 16, (count * 0x40) as usize)
            .map_err(|_| bad(id, "section table past the end"))?;
        let mut secs: Vec<Sec> = Vec::with_capacity(count as usize);
        let mut stream_sum = 0u64;
        for e in table.chunks(0x40) {
            let (off, size, kind) = (le64(e, 0), le64(e, 8), le64(e, 16));
            if off < HEADER {
                return Err(bad(id, format!("section offset {off:#x} before 0x4000")));
            }
            let end = off
                .checked_add(size)
                .filter(|&x| x <= MAX_NCA)
                .ok_or_else(|| bad(id, "section size"))?;
            let next_stream = stream_sum
                .checked_add(size)
                .filter(|&x| x <= MAX_NCA)
                .ok_or_else(|| bad(id, "section size"))?;
            let (mut key, mut prefix) = ([0u8; 16], [0u8; 8]);
            key.copy_from_slice(&e[0x20..0x30]);
            prefix.copy_from_slice(&e[0x30..0x38]);
            secs.push(Sec {
                start: off,
                end,
                stream_off: stream_sum,
                ctr: kind == 3 || kind == 4,
                key,
                prefix,
            });
            stream_sum = next_stream;
        }
        secs.sort_by_key(|s| (s.start, s.end));
        if secs.windows(2).any(|w| w[1].start < w[0].end) {
            return Err(bad(id, "overlapping sections"));
        }
        let size = secs.last().map_or(HEADER, |s| s.end);
        let span_len = size - HEADER;
        let p = HEADER + 16 + count * 0x40;
        let magic = src.read_vec(p, 8).unwrap_or_default();
        let body = if magic == b"NCZBLOCK" {
            let h = src
                .read_vec(p, 24)
                .map_err(|_| bad(id, "block header past the end"))?;
            let (exp, nblocks, total) = (
                h[11],
                u64::from(u32::from_le_bytes([h[12], h[13], h[14], h[15]])),
                le64(&h, 16),
            );
            if !(12..=30).contains(&exp) || h[8] != 2 && h[8] != 1 {
                return Err(ExtractError::Unsupported(format!(
                    "{id}.ncz: block header version {} with block size 2^{exp}",
                    h[8]
                )));
            }
            let packed = if total == span_len {
                false
            } else if total == stream_sum {
                true
            } else {
                return Err(bad(id, "block header does not match the section table"));
            };
            if h[9] != 1 || nblocks != total.div_ceil(1 << exp) {
                return Err(bad(id, "block header does not match the section table"));
            }
            let sizes = src
                .read_vec(p + 24, (nblocks * 4) as usize)
                .map_err(|_| bad(id, "block sizes past the end"))?;
            let mut pos = p + 24 + nblocks * 4;
            let mut at = vec![pos];
            for (i, s) in sizes.chunks(4).enumerate() {
                let cs = u64::from(u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
                let want = (1u64 << exp).min(total - (i as u64) * (1 << exp));
                if cs == 0 || cs > want {
                    return Err(bad(id, format!("block {i} of {cs} bytes holds {want}")));
                }
                pos += cs;
                at.push(pos);
            }
            if pos > n {
                return Err(bad(id, "blocks past the end"));
            }
            Body::Blocks {
                src,
                shift: u32::from(exp),
                total,
                packed,
                at,
                cache: Mutex::new(VecDeque::new()),
            }
        } else {
            Body::Solid {
                src,
                at: p,
                stream_sum,
                decoded: Mutex::new(None),
            }
        };
        Ok(NczStorage {
            id: id.to_string(),
            header,
            secs,
            size,
            body,
        })
    }

    /// Encrypts `buf`, which holds the NCA bytes at `abs`, in the sections with crypto type 3 or 4.
    fn encrypt(&self, abs: u64, buf: &mut [u8]) {
        let end = abs + buf.len() as u64;
        let mut i = self.secs.partition_point(|s| s.end <= abs);
        while let Some(s) = self.secs.get(i).filter(|s| s.start < end) {
            let (a, b) = (s.start.max(abs), s.end.min(end));
            if s.ctr && a < b {
                Aes128::new(&s.key).ctr(
                    s.prefix,
                    a,
                    &mut buf[(a - abs) as usize..(b - abs) as usize],
                );
            }
            i += 1;
        }
    }

    fn block(&self, i: u64) -> Result<Arc<Vec<u8>>> {
        let Body::Blocks {
            src,
            shift,
            total,
            at,
            cache,
            ..
        } = &self.body
        else {
            return Err(bad(&self.id, "not block compressed"));
        };
        if let Ok(c) = cache.lock()
            && let Some(b) = c.iter().find(|b| b.0 == i)
        {
            return Ok(b.1.clone());
        }
        let want = (1u64 << shift).min(total - i * (1 << shift));
        let raw = src.read_vec(
            at[i as usize],
            (at[i as usize + 1] - at[i as usize]) as usize,
        )?;
        let data = if raw.len() as u64 == want {
            raw
        } else {
            nxb_split::zstd::decompress_limited(&raw, want)
                .map_err(|e| bad(&self.id, format!("block {i}: {e}")))?
        };
        if data.len() as u64 != want {
            return Err(bad(
                &self.id,
                format!("block {i} decodes to {} bytes, not {want}", data.len()),
            ));
        }
        let b = Arc::new(data);
        if let Ok(mut c) = cache.lock() {
            if c.len() >= 16 {
                c.pop_front();
            }
            c.push_back((i, b.clone()));
        }
        Ok(b)
    }

    /// Decodes the solid stream once. Contiguous streams are cached already CTR-encrypted; non-contiguous streams keep
    /// plaintext so [`Self::read_at`] can zero-fill gaps and encrypt at true NCA offsets.
    fn solid(&self) -> Result<SolidDecoded> {
        let Body::Solid {
            src,
            at,
            stream_sum,
            decoded,
        } = &self.body
        else {
            return Err(bad(&self.id, "not a solid stream"));
        };
        let mut g = decoded
            .lock()
            .map_err(|_| bad(&self.id, "decoder state poisoned"))?;
        if let Some(d) = g.as_ref() {
            return Ok(d.clone());
        }
        let span_len = self.size - HEADER;
        let contiguous = *stream_sum == span_len;
        let mut pos = *at;
        let n = src.size();
        // read ahead in 8 MiB pieces: the decoder asks for a few bytes at a time
        let (mut ahead, mut ahead_at) = (Vec::<u8>::new(), 0usize);
        let mut input = |buf: &mut [u8]| -> std::result::Result<(), nxb_split::SplitError> {
            let mut done = 0;
            while done < buf.len() {
                if ahead_at == ahead.len() {
                    let k = (n - pos).min(8 << 20) as usize;
                    ahead = src
                        .read_vec(pos, k)
                        .map_err(|e| nxb_split::SplitError::Format(e.to_string()))?;
                    pos += k as u64;
                    ahead_at = 0;
                    if k == 0 {
                        return Err(nxb_split::SplitError::Format("input ends".into()));
                    }
                }
                let m = (ahead.len() - ahead_at).min(buf.len() - done);
                buf[done..done + m].copy_from_slice(&ahead[ahead_at..ahead_at + m]);
                ahead_at += m;
                done += m;
            }
            Ok(())
        };
        let (mut mem, mut file) = (Vec::new(), None);
        if span_len > SPILL_MEM {
            let dir = SPILL_DIR
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_else(std::env::temp_dir);
            let path = dir.join(format!(
                ".nxb-ncz-{}-{}.tmp",
                std::process::id(),
                SPILL_COUNT.fetch_add(1, Ordering::Relaxed)
            ));
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| ExtractError::io(format!("create {}", path.display()), e))?;
            let _ = std::fs::remove_file(&path); // unlinked at once: nothing is left behind after a crash
            file = Some(FileStorage::from_file(f, span_len, "decoded NCZ"));
        }
        let mut written = 0u64;
        let mut failure = None;
        let mut sink = |b: &[u8]| -> std::result::Result<(), nxb_split::SplitError> {
            let mut v = b.to_vec();
            if contiguous {
                self.encrypt(HEADER + written, &mut v);
            }
            match &file {
                Some(f) => f
                    .write_at(written, &v)
                    .map_err(|e| nxb_split::SplitError::Format(e.to_string()))?,
                None => mem.extend_from_slice(&v),
            }
            written += v.len() as u64;
            Ok(())
        };
        let r = nxb_split::zstd::decompress_stream(&mut input, n - at, span_len, &mut sink);
        if let Err(e) = r {
            failure = Some(e);
        }
        if let Some(e) = failure {
            return Err(bad(&self.id, format!("solid stream: {e}")));
        }
        let packed = if contiguous {
            if written != span_len {
                return Err(bad(
                    &self.id,
                    format!("solid stream decodes to {written} bytes, not {span_len}"),
                ));
            }
            false
        } else if written == *stream_sum {
            true
        } else if written == span_len {
            false
        } else {
            return Err(bad(
                &self.id,
                format!("solid stream decodes to {written} bytes, not {stream_sum} or {span_len}"),
            ));
        };
        let storage: Arc<dyn Storage> = match file {
            Some(f) => Arc::new(f),
            None => Arc::new(MemStorage(mem)),
        };
        let res = SolidDecoded {
            storage,
            packed,
            encrypted: contiguous,
        };
        *g = Some(res.clone());
        Ok(res)
    }

    fn read_blocks_range(&self, shift: u32, stream_off: u64, out: &mut [u8]) -> Result<()> {
        let mut pos = 0usize;
        while pos < out.len() {
            let s = stream_off + pos as u64;
            let b = self.block(s >> shift)?;
            let within = (s & ((1 << shift) - 1)) as usize;
            let n = (b.len() - within).min(out.len() - pos);
            out[pos..pos + n].copy_from_slice(&b[within..within + n]);
            pos += n;
        }
        Ok(())
    }

    fn read_packed(
        &self,
        from: u64,
        rest: &mut [u8],
        mut read_stream: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        rest.fill(0);
        let end = from + rest.len() as u64;
        let mut i = self.secs.partition_point(|s| s.end <= from);
        while let Some(s) = self.secs.get(i).filter(|s| s.start < end) {
            let (a, b) = (s.start.max(from), s.end.min(end));
            if a < b {
                let off = s.stream_off + (a - s.start);
                read_stream(off, &mut rest[(a - from) as usize..(b - from) as usize])?;
            }
            i += 1;
        }
        Ok(())
    }
}

impl Storage for NczStorage {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .filter(|&e| e <= self.size)
            .ok_or_else(|| {
                bad(
                    &self.id,
                    format!("read of {} bytes at {offset:#x} past the end", buf.len()),
                )
            })?;
        let head = (HEADER.min(end).saturating_sub(offset)) as usize;
        if head > 0 {
            buf[..head].copy_from_slice(&self.header[offset as usize..offset as usize + head]);
        }
        if head == buf.len() {
            return Ok(());
        }
        let (from, rest) = (offset + head as u64, &mut buf[head..]);
        match &self.body {
            Body::Solid { .. } => {
                let s = self.solid()?;
                if s.packed {
                    self.read_packed(from, rest, |off, dst| s.storage.read_at(off, dst))?;
                } else {
                    s.storage.read_at(from - HEADER, rest)?;
                }
                if !s.encrypted {
                    self.encrypt(from, rest);
                }
                Ok(())
            }
            Body::Blocks { shift, packed, .. } => {
                if *packed {
                    self.read_packed(from, rest, |off, dst| {
                        self.read_blocks_range(*shift, off, dst)
                    })?;
                } else {
                    self.read_blocks_range(*shift, from - HEADER, rest)?;
                }
                self.encrypt(from, rest);
                Ok(())
            }
        }
    }
}
