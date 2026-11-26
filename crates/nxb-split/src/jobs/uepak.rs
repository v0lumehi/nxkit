//! Unreal Engine pak files (`uepak.py`: `Pak`, `footer`, `inline_shaders`), read by ranges, with no key in the split
//! and with the key from main in the build's Unreal decode ([`super::uejob`]). Blocks in other methods than stored,
//! Zlib and LZ4 (LZ5, Oodle) need the game's own decoder (`crate::oodle`).

use std::collections::HashMap;
use std::sync::Arc;

use super::R;
use crate::SplitError;
use crate::rd::{sl, u32le};
use crate::src::Src;

pub(super) fn ve<T>(m: impl Into<String>) -> R<T> {
    Err(SplitError::Red(format!("ValueError: {}", m.into())))
}

/// `class R`: a reader over a table, a read past its end is the error of a damaged file.
pub(super) struct Rd<'a> {
    b: &'a [u8],
    o: usize,
    what: String,
}

impl<'a> Rd<'a> {
    pub(super) fn new(b: &'a [u8], o: usize, what: impl Into<String>) -> Rd<'a> {
        Rd {
            b,
            o,
            what: what.into(),
        }
    }
    pub(super) fn raw(&mut self, n: usize) -> R<&'a [u8]> {
        if self.o > self.b.len() || n > self.b.len() - self.o {
            return ve(format!(
                "{}: {n:#x} bytes at {:#x} do not fit in its {:#x}",
                self.what,
                self.o,
                self.b.len()
            ));
        }
        let v = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(v)
    }
    pub(super) fn pos(&self) -> usize {
        self.o
    }
    pub(super) fn le(&mut self, n: usize) -> R<u64> {
        Ok(self
            .raw(n)?
            .iter()
            .rev()
            .fold(0u64, |a, &x| a << 8 | u64::from(x)))
    }
    pub(super) fn i32(&mut self) -> R<i32> {
        Ok(self.le(4)? as u32 as i32)
    }
    pub(super) fn fstr(&mut self) -> R<String> {
        let n = self.i32()?;
        if n < 0 {
            let b = self.raw(n.unsigned_abs() as usize * 2)?;
            let u: Vec<u16> = b
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let s =
                String::from_utf16(&u).map_err(|_| SplitError::Red("UnicodeDecodeError".into()))?;
            return Ok(s
                .chars()
                .take(s.chars().count().saturating_sub(1))
                .collect());
        }
        if n == 0 {
            return Ok(String::new());
        }
        let b = self.raw(n as usize)?;
        Ok(String::from_utf8_lossy(&b[..b.len() - 1]).into_owned())
    }
}

/// An entry of a pak.
#[derive(Clone, Debug, Default)]
pub(super) struct Ent {
    off: u64,
    pub(super) raw: u64,
    size: u64,
    method: String,
    enc: bool,
    block_size: u64,
    blocks: Vec<(u64, u64)>,
}

const PAK_MAGIC: u32 = 0x5A6F_12E1;

/// A pak opened for reading by ranges.
pub(super) struct Pak {
    src: Arc<dyn Src>,
    rel: String,
    pub(super) version: u32,
    methods: Vec<String>,
    pub(super) mount: String,
    pub(super) names: Vec<String>,
    legacy: HashMap<String, Ent>,
    enc_blob: Vec<u8>,
    idx: HashMap<String, u32>,
    key: Option<crate::aes256::Aes256>,
}

/// `footer`: (encrypted index, index offset, index size, method names, version).
pub(super) fn footer(s: &dyn Src, rel: &str) -> R<(bool, u64, u64, Vec<String>, u32)> {
    let size = s.len();
    let mut found = None;
    for (back, foot) in [(0x2cu64, 0x2du64), (0xc8, 0xc8 + 21), (0xcc, 0xcc + 21)] {
        if size < foot {
            continue;
        }
        let h = s.read_up(size - back, 8)?;
        if u32le(&h, 0)? as u32 == PAK_MAGIC {
            found = Some((back, u32le(&h, 4)? as u32));
            break;
        }
    }
    let Some((back, ver)) = found else {
        return ve("no pak footer");
    };
    if ((ver == 4 || ver == 11) && ((ver == 4) != (back == 0x2c))) || (ver == 8 && back == 0x2c) {
        return ve(format!(
            "a version {ver} pak footer with its magic {back:#x} bytes from the end, where that version has none"
        ));
    }
    let (enc, ioff, isz, methods);
    if ver == 4 {
        let ft = s.read_up(size - 0x2d, 0x2d)?;
        (ioff, isz) = (crate::rd::u64le(&ft, 9)?, crate::rd::u64le(&ft, 17)?);
        enc = ft[0] != 0;
        methods = (1..8)
            .map(|i| match i {
                1 => "Zlib".to_string(),
                2 => "Gzip".into(),
                3 => "LZ5".into(),
                4 => "Custom".into(),
                n => format!("flags {n}"),
            })
            .collect();
    } else if ver != 8 && ver != 11 {
        return ve(format!(
            "{rel}: pak version {ver}, only 4, 8 and 11 are read"
        ));
    } else {
        let ft = s.read_up(size - (back + 21), (back + 21) as usize)?;
        let k = 21;
        (ioff, isz) = (
            crate::rd::u64le(&ft, k + 8)?,
            crate::rd::u64le(&ft, k + 16)?,
        );
        enc = ft[k - 1] != 0;
        methods = (0..5)
            .map(|i| {
                let b = sl(&ft, k + 44 + 32 * i, k + 76 + 32 * i);
                let end = b.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                b[..end].iter().map(|&c| c as char).collect()
            })
            .collect();
    }
    if ioff.saturating_add(isz) > size {
        return ve(format!(
            "the index ({isz:#x} bytes at {ioff:#x}) runs past the end of the pak"
        ));
    }
    Ok((enc, ioff, isz, methods, ver))
}

pub(super) fn range(s: &dyn Src, off: u64, n: u64, what: &str) -> R<Vec<u8>> {
    if off.saturating_add(n) > s.len() {
        return ve(format!(
            "{what}: {n:#x} bytes at {off:#x} do not lie inside the file"
        ));
    }
    s.read_up(off, n as usize)
}

impl Pak {
    /// `Pak(path)` without a key.
    pub(super) fn open(src: Arc<dyn Src>, rel: &str) -> R<Pak> {
        Pak::open_keyed(src, rel, None)
    }

    /// `Pak(path, key)`.
    pub(super) fn open_keyed(src: Arc<dyn Src>, rel: &str, key: Option<&[u8; 32]>) -> R<Pak> {
        let (enc, ioff, isz, methods, version) = footer(src.as_ref(), rel)?;
        if enc && key.is_none() {
            return ve(format!(
                "{rel}: the index is encrypted and no key was given"
            ));
        }
        let mut pk = Pak {
            src: src.clone(),
            rel: rel.to_string(),
            version,
            methods,
            mount: String::new(),
            names: Vec::new(),
            legacy: HashMap::new(),
            enc_blob: Vec::new(),
            idx: HashMap::new(),
            key: key.map(crate::aes256::Aes256::new),
        };
        let ib = pk.read_at(ioff, isz, enc)?;
        let mut r = Rd::new(&ib, 0, format!("{rel}: the index"));
        pk.mount = r.fstr()?;
        let count = r.i32()?;
        if version == 4 || version == 8 {
            for _ in 0..count {
                let name = r.fstr()?;
                let (off, size, raw, meth) = (r.le(8)?, r.le(8)?, r.le(8)?, r.i32()?);
                r.raw(20)?;
                let mut blocks = Vec::new();
                if meth != 0 {
                    for _ in 0..r.i32()? {
                        blocks.push((r.le(8)?, r.le(8)?));
                    }
                }
                let (e, bsz) = (r.le(1)?, r.le(4)?);
                let head = 53
                    + if meth != 0 {
                        4 + 16 * blocks.len() as u64
                    } else {
                        0
                    };
                let mut bl: Vec<(u64, u64)> = blocks
                    .iter()
                    .map(|&(s, t)| {
                        if version < 5 {
                            (s.wrapping_sub(off), t.wrapping_sub(s))
                        } else {
                            (s, t.wrapping_sub(s))
                        }
                    })
                    .collect();
                if bl.is_empty() {
                    bl = vec![(head, size)];
                }
                let method = if meth == 0 {
                    String::new()
                } else if version < 8 {
                    match meth {
                        1 => "Zlib".to_string(),
                        2 => "Gzip".into(),
                        3 => "LZ5".into(),
                        4 => "Custom".into(),
                        n => format!("flags {n}"),
                    }
                } else if meth > 0 && (meth as usize) <= pk.methods.len() {
                    pk.methods[meth as usize - 1].clone()
                } else {
                    format!("method {meth}")
                };
                pk.names.push(name.clone());
                pk.legacy.insert(
                    name,
                    Ent {
                        off,
                        raw,
                        size,
                        method,
                        enc: e != 0,
                        block_size: bsz,
                        blocks: bl,
                    },
                );
            }
            return Ok(pk);
        }
        r.le(8)?;
        if r.i32()? != 0 {
            r.le(8)?;
            r.le(8)?;
            r.raw(20)?;
        }
        if r.i32()? == 0 {
            return ve(format!("{rel}: no full directory index"));
        }
        let (doff, dsz) = (r.le(8)?, r.le(8)?);
        r.raw(20)?;
        let n = r.i32()?.max(0) as usize;
        pk.enc_blob = r.raw(n)?.to_vec();
        if r.i32()? != 0 {
            return ve(format!("{rel}: unencoded entries are not read"));
        }
        let db = pk.read_at(doff, dsz, enc)?;
        let mut d = Rd::new(&db, 0, format!("{rel}: the directory index"));
        for _ in 0..d.i32()? {
            let dn = d.fstr()?;
            for _ in 0..d.i32()? {
                let fnm = d.fstr()?;
                let v = d.i32()? as u32;
                let key = format!("{dn}{fnm}").trim_start_matches('/').to_string();
                if pk.idx.insert(key.clone(), v).is_none() {
                    pk.names.push(key);
                }
            }
        }
        Ok(pk)
    }

    pub(super) fn entry(&self, path: &str) -> R<Ent> {
        if self.version == 4 || self.version == 8 {
            return self
                .legacy
                .get(path)
                .cloned()
                .ok_or_else(|| SplitError::Red(format!("KeyError: {path:?}")));
        }
        let at = *self
            .idx
            .get(path)
            .ok_or_else(|| SplitError::Red(format!("KeyError: {path:?}")))?
            as usize;
        let mut r = Rd::new(
            &self.enc_blob,
            at,
            format!("{}: the entry of {path}", self.rel),
        );
        let v = r.le(4)?;
        let mut bsz = (v & 0x3f) << 11;
        if v & 0x3f == 0x3f {
            bsz = r.le(4)?;
        }
        let (nblk, enc, meth) = (
            (v >> 6) & 0xffff,
            (v >> 22) & 1,
            ((v >> 23) & 0x3f) as usize,
        );
        let off = r.le(if (v >> 31) & 1 == 1 { 4 } else { 8 })?;
        let raw = r.le(if (v >> 30) & 1 == 1 { 4 } else { 8 })?;
        let size = if meth != 0 {
            r.le(if (v >> 29) & 1 == 1 { 4 } else { 8 })?
        } else {
            raw
        };
        let mut cur = 53 + if meth != 0 { 4 + 16 * nblk } else { 0 };
        let mut blocks = Vec::new();
        if nblk == 1 && bsz == 0 {
            bsz = raw;
        }
        if nblk == 1 && enc == 0 {
            blocks.push((cur, size));
        } else {
            for _ in 0..nblk {
                let s = r.le(4)?;
                blocks.push((cur, s));
                cur += if enc == 1 { (s + 15) & !15 } else { s };
            }
        }
        if meth > self.methods.len() || (meth != 0 && self.methods[meth - 1].is_empty()) {
            return ve(format!(
                "{}: {path} names compression method {meth}, which the footer does not",
                self.rel
            ));
        }
        Ok(Ent {
            off,
            raw,
            size,
            method: if meth != 0 {
                self.methods[meth - 1].clone()
            } else {
                String::new()
            },
            enc: enc == 1,
            block_size: bsz,
            blocks,
        })
    }

    fn read_at(&self, off: u64, n: u64, enc: bool) -> R<Vec<u8>> {
        if !enc {
            return range(self.src.as_ref(), off, n, &self.rel);
        }
        let Some(k) = &self.key else {
            return ve(format!(
                "{}: {n:#x} encrypted bytes at {off:#x} and no key was given",
                self.rel
            ));
        };
        let mut b = range(
            self.src.as_ref(),
            off,
            n.saturating_add(15) & !15,
            &self.rel,
        )?;
        k.decrypt_ecb(&mut b);
        b.truncate(n as usize);
        Ok(b)
    }

    /// `Pak.read`: a file's bytes, block by block.
    pub(super) fn read(&self, path: &str) -> R<Vec<u8>> {
        let e = self.entry(path)?;
        if e.method.is_empty() {
            return self.read_at(e.off + 53, e.size, e.enc);
        }
        let (mut out, mut left) = (Vec::new(), e.raw);
        for (i, &(s, n)) in e.blocks.iter().enumerate() {
            let k = left.min(e.block_size);
            let b = self.read_at(e.off + s, n, e.enc)?;
            let d = match e.method.as_str() {
                "Zlib" => crate::zlibx::zlib_stream(&b, (k as usize).max(1)),
                "LZ4" => {
                    if k > 255 * b.len() as u64 {
                        return ve(format!(
                            "{path}: an LZ4 block of {:#x} bytes cannot hold {k:#x}",
                            b.len()
                        ));
                    }
                    crate::lz4::block_upto(&b, k as usize).ok()
                }
                m => Some(crate::oodle::decode(
                    &b,
                    k as usize,
                    &format!("{}: {path} ({m})", self.rel),
                )?),
            };
            let Some(d) = d else {
                return ve(format!("{}: {path} block {i} does not decode", self.rel));
            };
            if d.len() as u64 != k {
                return ve(format!(
                    "{}: {path} block {i} decodes to {:#x} bytes, not {k:#x}",
                    self.rel,
                    d.len()
                ));
            }
            out.extend(d);
            left -= k;
        }
        Ok(out)
    }
}

/// `inline_shaders`: whether a pak keeps its shaders inline in its cooked packages. `beside` tells whether the
/// containers in its folder hold them (`inline_in_containers`), for a pak that names only the global shader cache.
pub(super) fn inline_shaders(s: &dyn Src, rel: &str, beside: &dyn Fn() -> bool) -> bool {
    let f = || -> R<bool> {
        let (enc, ioff, isz, _, ver) = footer(s, rel)?;
        if ver == 4 || ver == 8 {
            return Ok(true);
        }
        if enc {
            return Ok(false);
        }
        let ib = range(s, ioff, isz, rel)?;
        let mut r = Rd::new(&ib, 0, rel);
        r.fstr()?;
        r.raw(12)?;
        if r.i32()? != 0 {
            r.raw(16)?;
            r.raw(20)?;
        }
        if r.i32()? == 0 {
            return Ok(false);
        }
        let (doff, dsz) = (r.le(8)?, r.le(8)?);
        let d = range(s, doff, dsz, rel)?;
        let has = |p: &[u8]| crate::shaders::memfind(&d, p).is_some();
        Ok(!has(b".ushaderbytecode\0")
            && (has(b".uexp\0") || (has(b"GlobalShaderCache-") && beside())))
    };
    f().unwrap_or(false)
}

/// Posix `os.path.normpath`.
pub(super) fn normpath(p: &str) -> String {
    if p.is_empty() {
        return ".".into();
    }
    let slashes = if p.starts_with("//") && !p.starts_with("///") {
        2
    } else {
        usize::from(p.starts_with('/'))
    };
    let mut new: Vec<&str> = Vec::new();
    for c in p.split('/') {
        if c.is_empty() || c == "." {
            continue;
        }
        if c != ".." || (slashes == 0 && new.is_empty()) || new.last() == Some(&"..") {
            new.push(c);
        } else if !new.is_empty() {
            new.pop();
        }
    }
    let s = format!("{}{}", "/".repeat(slashes), new.join("/"));
    if s.is_empty() { ".".into() } else { s }
}
