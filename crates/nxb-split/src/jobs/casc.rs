//! Blizzard CASC `.data` archives (`00-00000001.data`, `data.000`, ...) with BLTE containers (`crate::blte`).
//!
//! Each `.data` file is streamed in bounded chunks from offset 0, stepping over 30-byte CASC local entry headers
//! and BLTE containers so multi-gigabyte archives are never loaded into memory whole. Sibling CASC `.idx` files
//! are marked taken alongside the `.data` files.

use std::collections::HashSet;
use std::sync::Arc;

use super::{Ctx, File, Job, JobOut, Plan, R};
use crate::bahs::BahsVar;
use crate::blte;
use crate::glslc::{glslc_shaders, stage};
use crate::prim::{CS_MAGIC, CTL_MAGIC, Found, GFX_MAGIC, check, find_all, memfind, u32le};
use crate::shaders::scan_blob;
use crate::src::Src;
use crate::zstd::Dicts;

const GLSLC: [u8; 4] = [0x91, 0x68, 0x86, 0x19];
const CASC_HDR_LEN: usize = 0x1e;
const BUF_SIZE: usize = 4 << 20;
const PROBE_SIZE: usize = 64 << 10;

fn is_casc_data_name(rel: &str) -> bool {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    name.ends_with(".data")
        || (name.starts_with("data.")
            && name.len() > 5
            && name[5..].bytes().all(|b| b.is_ascii_digit()))
}

fn dir_of(rel: &str) -> &str {
    rel.rsplit_once('/').map_or("", |x| x.0)
}

fn is_casc_idx(src: &dyn Src) -> R<bool> {
    if src.len() < 0x20 {
        return Ok(false);
    }
    let h = src.read_up(0, 0x20)?;
    let hs = u32le(&h, 0).unwrap_or(0);
    let ver = h.get(8..10).map_or(0, |b| u16::from_le_bytes([b[0], b[1]]));
    Ok(hs == 0x10
        && matches!(ver, 5 | 7 | 8)
        && h[12] == 4
        && h[13] == 5
        && h[14] == 9
        && h[15] <= 32)
}

fn valid_casc_entry(buf: &[u8], off: usize, file_len: u64, base_off: u64) -> Option<(u64, u64)> {
    let hdr = buf.get(off..off + CASC_HDR_LEN + 9)?;
    let total_size = u32le(hdr, 0x10)? as u64;
    if total_size < CASC_HDR_LEN as u64 + 9 || base_off + off as u64 + total_size > file_len {
        return None;
    }
    let blte_off = off + CASC_HDR_LEN;
    let hs = blte::header_len(&buf[blte_off..])?;
    if hs == 0 {
        return Some((CASC_HDR_LEN as u64, total_size));
    }
    let blte_slice = buf.get(blte_off..blte_off + hs)?;
    let parsed = blte::parse_header(blte_slice)?;
    let ps = parsed.payload_size()? as u64;
    (CASC_HDR_LEN as u64 + ps <= total_size).then_some((CASC_HDR_LEN as u64, total_size))
}

fn is_casc_data(src: &dyn Src) -> R<bool> {
    let len = src.len();
    if len < 0x28 {
        return Ok(false);
    }
    let probe = src.read_up(0, PROBE_SIZE)?;
    if valid_casc_entry(&probe, 0, len, 0).is_some() {
        return Ok(true);
    }
    // Many CASC data files begin with a 30-byte entry for the archive header (typically 0x2e or 0x40 bytes).
    if let Some(skip0) = u32le(&probe, 0x10)
        && (CASC_HDR_LEN as u32..=0x1000).contains(&skip0)
        && (skip0 as usize) < probe.len()
        && valid_casc_entry(&probe, skip0 as usize, len, 0).is_some()
    {
        return Ok(true);
    }
    for at in find_all(&probe, &blte::MAGIC) {
        if at >= CASC_HDR_LEN && valid_casc_entry(&probe, at - CASC_HDR_LEN, len, 0).is_some() {
            return Ok(true);
        }
        if let Some(hs) = blte::header_len(&probe[at..])
            && hs >= 36
            && let Some(hdr) = probe.get(at..at + hs).and_then(blte::parse_header)
            && let Some(ps) = hdr.payload_size()
            && at as u64 + ps as u64 <= len
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Scans raw NVN control sections paired with adjacent `GFX_MAGIC` / `CS_MAGIC` code areas inside a decoded BLTE entry.
fn scan_adjacent_raw(d: &[u8], label: &str, out: &mut Vec<Found>) -> R<()> {
    for c in find_all(d, &CTL_MAGIC) {
        if c & 3 != 0 || c + 0x708 > d.len() {
            continue;
        }
        let nctl = u32le(d, c + 0x20).unwrap_or(0) as usize;
        let ncode = u32le(d, c + 0x704).unwrap_or(0) as usize;
        if !(0x760..=0x2000).contains(&nctl)
            || nctl & 3 != 0
            || c + nctl > d.len()
            || ncode < 4
            || ncode > d.len()
        {
            continue;
        }
        let ctl = &d[c..c + nctl];
        let mut cands = Vec::new();
        for align in [1usize, 4, 16, 32, 64, 128, 256] {
            let k = (c + nctl + align - 1) & !(align - 1);
            if !cands.contains(&k) {
                cands.push(k);
            }
        }
        if let Some(before) = c.checked_sub(ncode) {
            cands.push(before);
            for align in [16usize, 32, 64, 128, 256] {
                let k = before & !(align - 1);
                if !cands.contains(&k) {
                    cands.push(k);
                }
            }
        }
        for k in cands {
            let Some(code) = d.get(k..k + ncode) else {
                continue;
            };
            if !code.starts_with(&GFX_MAGIC) && !code.starts_with(&CS_MAGIC) {
                continue;
            }
            if let Some(st) = stage(ctl, code)
                && check(st, ctl, code).is_none()
            {
                out.push(Found {
                    place: format!("{label}#RAW@{c:#x}"),
                    stage: st,
                    ctl: ctl.to_vec(),
                    code: code.to_vec(),
                });
                break;
            }
        }
    }
    Ok(())
}

fn scan_decoded(
    decoded: &[u8],
    label: &str,
    dicts: &Dicts,
    out: &mut Vec<Found>,
    var: &mut BahsVar,
) -> R<()> {
    let owned: Vec<u8>;
    let bytes = if decoded.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        owned = crate::zstd::decompress_dict(decoded, dicts)?;
        &owned[..]
    } else if decoded.starts_with(&[0x1f, 0x8b, 0x08]) {
        owned = crate::misc::ungzip(decoded, label)?;
        &owned[..]
    } else {
        decoded
    };
    let before = out.len();
    scan_blob(bytes, label, out, var, 0)?;
    if out.len() == before && bytes.get(..4) != Some(b"MSHD") && memfind(bytes, &GLSLC).is_some() {
        out.extend(glslc_shaders(bytes, label, false, &[])?);
    }
    if out.len() == before && memfind(bytes, &CTL_MAGIC).is_some() {
        scan_adjacent_raw(bytes, label, out)?;
    }
    Ok(())
}

struct StreamBuf<'a> {
    src: &'a dyn Src,
    len: u64,
    off: u64,
    data: Vec<u8>,
}

impl<'a> StreamBuf<'a> {
    fn new(src: &'a dyn Src) -> Self {
        Self {
            src,
            len: src.len(),
            off: 0,
            data: Vec::new(),
        }
    }

    fn slice_at(&mut self, at: u64, min_bytes: usize) -> R<&[u8]> {
        let want = min_bytes.min(self.len.saturating_sub(at) as usize);
        if at < self.off || at + want as u64 > self.off + self.data.len() as u64 {
            let read_len = BUF_SIZE.max(want).min(self.len.saturating_sub(at) as usize);
            self.data = self.src.read_up(at, read_len)?;
            self.off = at;
        }
        let rel = (at - self.off) as usize;
        Ok(&self.data[rel..])
    }
}

/// Checks whether `buf[0..]` starts a valid BLTE container (with or without a preceding 30-byte CASC header).
/// Returns `Some((blte_len, advance_from_pos))` on match.
fn match_blte_at(
    src: &dyn Src,
    file_len: u64,
    blte_off: u64,
    max_len: Option<usize>,
    head: &[u8],
) -> R<Option<(usize, u64)>> {
    let Some(hs) = blte::header_len(head) else {
        return Ok(None);
    };
    if hs == 0 {
        let Some(len) = max_len.filter(|&n| n > 8 && blte_off + n as u64 <= file_len) else {
            return Ok(None);
        };
        return Ok(Some((len, len as u64)));
    }
    if blte_off + hs as u64 > file_len {
        return Ok(None);
    }
    let hdr_bytes = if head.len() >= hs {
        head[..hs].to_vec()
    } else {
        src.read_up(blte_off, hs)?
    };
    let Some(hdr) = blte::parse_header(&hdr_bytes) else {
        return Ok(None);
    };
    let Some(ps) = hdr.payload_size() else {
        return Ok(None);
    };
    if blte_off + ps as u64 > file_len {
        return Ok(None);
    }
    if let Some(m) = max_len
        && ps > m
    {
        return Ok(None);
    }
    Ok(Some((ps, ps as u64)))
}

fn decode_and_scan(
    src: &dyn Src,
    rel: &str,
    blte_off: u64,
    blen: usize,
    dicts: &Dicts,
    out: &mut Vec<Found>,
    var: &mut BahsVar,
) -> R<()> {
    let raw = src.read_up(blte_off, blen)?;
    let label = format!("/{rel}@{blte_off:#x}");
    if let Some(dec) = blte::decode(&raw, &label)? {
        scan_decoded(&dec, &label, dicts, out, var)?;
    }
    Ok(())
}

fn scan_data_file(src: &dyn Src, rel: &str, dicts: &Dicts) -> R<JobOut> {
    let (mut out, mut var) = (Vec::new(), BahsVar::new());
    let file_len = src.len();
    let mut sb = StreamBuf::new(src);
    let mut pos = 0u64;

    while pos + 9 <= file_len {
        let win = sb.slice_at(pos, CASC_HDR_LEN + 12)?;
        if win.len() >= CASC_HDR_LEN + 9
            && let Some(total) = u32le(win, 0x10).map(|n| n as u64)
            && total >= CASC_HDR_LEN as u64 + 9
            && pos + total <= file_len
            && win[CASC_HDR_LEN..CASC_HDR_LEN + 4] == blte::MAGIC
        {
            let max_blte = (total - CASC_HDR_LEN as u64) as usize;
            let blte_off = pos + CASC_HDR_LEN as u64;
            if let Some((blen, _)) = match_blte_at(
                src,
                file_len,
                blte_off,
                Some(max_blte),
                &win[CASC_HDR_LEN..],
            )? {
                decode_and_scan(src, rel, blte_off, blen, dicts, &mut out, &mut var)?;
                pos += total;
                continue;
            }
        }
        if pos == 0
            && let Some(total) = u32le(win, 0x10).map(|n| n as u64)
            && (CASC_HDR_LEN as u64..=0x1000).contains(&total)
            && total < file_len
            && win.get(CASC_HDR_LEN..CASC_HDR_LEN + 4) != Some(&blte::MAGIC[..])
        {
            pos = total;
            continue;
        }
        if win.starts_with(&blte::MAGIC)
            && let Some((blen, adv)) = match_blte_at(src, file_len, pos, None, win)?
        {
            decode_and_scan(src, rel, pos, blen, dicts, &mut out, &mut var)?;
            pos += adv;
            continue;
        }
        let Some(hit) = memfind(win, &blte::MAGIC) else {
            let step = win.len().saturating_sub(3) as u64;
            if step == 0 {
                break;
            }
            pos += step;
            continue;
        };
        if hit == 0 {
            pos += 4;
            continue;
        }
        let bpos = pos + hit as u64;
        if bpos >= CASC_HDR_LEN as u64 {
            let cpos = bpos - CASC_HDR_LEN as u64;
            let cwin = sb.slice_at(cpos, CASC_HDR_LEN + 12)?;
            if let Some(total) = u32le(cwin, 0x10).map(|n| n as u64)
                && total >= CASC_HDR_LEN as u64 + 9
                && cpos + total <= file_len
                && let Some((blen, _)) = match_blte_at(
                    src,
                    file_len,
                    bpos,
                    Some((total - CASC_HDR_LEN as u64) as usize),
                    &cwin[CASC_HDR_LEN..],
                )?
            {
                decode_and_scan(src, rel, bpos, blen, dicts, &mut out, &mut var)?;
                pos = cpos + total;
                continue;
            }
        }
        let bwin = sb.slice_at(bpos, 12)?;
        if let Some((blen, adv)) = match_blte_at(src, file_len, bpos, None, bwin)? {
            decode_and_scan(src, rel, bpos, blen, dicts, &mut out, &mut var)?;
            pos = bpos + adv;
            continue;
        }
        pos = bpos + 4;
    }

    Ok(JobOut { found: out, var })
}

/// Plans jobs for Blizzard CASC `.data` archives and marks sibling CASC `.idx` files as taken.
pub fn plan(files: &[File], p: &mut Plan) -> R<()> {
    let mut casc_dirs: HashSet<String> = HashSet::new();
    for (i, f) in files.iter().enumerate() {
        if p.taken.contains(&i)
            || !f.live
            || !is_casc_data_name(&f.rel)
            || super::nlg::covered(&f.rel, files)
            || !is_casc_data(f.src.as_ref())?
        {
            continue;
        }
        p.taken.insert(i);
        casc_dirs.insert(dir_of(&f.rel).to_string());
        let src: Arc<dyn Src> = f.src.clone();
        let rel = f.rel.clone();
        let rel2 = rel.clone();
        p.jobs.push(Job {
            rel,
            run: Box::new(move |ctx: &Ctx<'_>, _| scan_data_file(src.as_ref(), &rel2, ctx.dicts)),
        });
    }
    if !casc_dirs.is_empty() {
        for (i, f) in files.iter().enumerate() {
            if !p.taken.contains(&i)
                && f.rel.ends_with(".idx")
                && casc_dirs.contains(dir_of(&f.rel))
                && is_casc_idx(f.src.as_ref())?
            {
                p.taken.insert(i);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
