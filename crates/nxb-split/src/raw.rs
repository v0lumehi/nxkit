//! `raw_shaders`: control sections in main that no GLSLC output or BNSH file holds, with their code areas found by the
//! relocations or the `adrp` + `add` / `adr` pairs that build their addresses (`rel_targets`, `adrp_targets`).

use std::collections::{HashMap, HashSet};

use crate::SplitError;
use crate::glslc::{layout, stage};
use crate::prim::{CS_MAGIC, CTL_MAGIC, Found, GFX_MAGIC, check, find_all};
use crate::rd::{u32le, u64le};

type R<T> = Result<T, SplitError>;

/// Ordered map: Python dicts keep the first insertion's place and the last value.
#[derive(Default)]
struct Rel {
    keys: Vec<u64>,
    val: HashMap<u64, u64>,
}

impl Rel {
    fn put(&mut self, k: u64, v: u64) {
        if self.val.insert(k, v).is_none() {
            self.keys.push(k);
        }
    }
}

/// `rel_targets`: a module image's relative relocations (slot to target).
fn rel_targets(img: &[u8]) -> R<Rel> {
    let m = u32le(img, 4)?;
    let o = (m as i64 + i64::from(u32le(img, m + 4)? as u32 as i32)) as usize;
    let mut t: HashMap<i64, u64> = HashMap::new();
    let mut o = o;
    while o + 16 <= img.len() {
        let (tag, v) = (u64le(img, o)? as i64, u64le(img, o + 8)?);
        o += 16;
        if tag == 0 {
            break;
        }
        t.entry(tag).or_insert(v);
    }
    let mut out = Rel::default();
    if let Some(&base) = t.get(&7) {
        for i in (0..t.get(&8).copied().unwrap_or(0) as usize).step_by(24) {
            let at = base as usize + i;
            let (off, info, add) = (u64le(img, at)?, u64le(img, at + 8)?, u64le(img, at + 16)?);
            if info & 0xffff_ffff == 0x403 {
                out.put(off, add);
            }
        }
    }
    let mut nxt = 0u64;
    if let Some(&base) = t.get(&36) {
        for i in (0..t.get(&35).copied().unwrap_or(0) as usize).step_by(8) {
            let e = u64le(img, base as usize + i)?;
            if e & 1 == 0 {
                out.put(e, u64le(img, e as usize)?);
                nxt = e + 8;
                continue;
            }
            for b in 0..63 {
                if e >> (b + 1) & 1 == 1 {
                    let s = nxt + 8 * b;
                    out.put(s, u64le(img, s as usize)?);
                }
            }
            nxt += 8 * 63;
        }
    }
    Ok(out)
}

fn adrp_imm21(x: u32) -> i64 {
    let raw = (((x >> 5) & 0x7ffff) << 2 | ((x >> 29) & 3)) as i32;
    i64::from((raw << 11) >> 11)
}

fn is_code_magic(d: &[u8], at: usize) -> bool {
    matches!(
        d.get(at..at + 4),
        Some(m) if m == GFX_MAGIC || m == CS_MAGIC
    )
}

/// `adrp_targets`: addresses in `[lo, hi)` (or pointing to a control section / shader code magic anywhere in `d`)
/// built with `adrp` + `add` or `adr`, with the instruction offsets that materialize them.
fn adrp_targets(
    d: &[u8],
    lo: i64,
    hi: i64,
    cset: &HashSet<usize>,
) -> (HashMap<i64, Vec<i64>>, Vec<i64>) {
    let w: Vec<u32> = d
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let mut out: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut code_targets: Vec<i64> = Vec::new();
    let mut record = |t: i64, at: i64| {
        if t < 0 || t as usize >= d.len() {
            return;
        }
        let code_hit = is_code_magic(d, t as usize);
        if (lo <= t && t < hi) || cset.contains(&(t as usize)) || code_hit {
            let e = out.entry(t).or_default();
            if code_hit && e.is_empty() {
                code_targets.push(t);
            }
            e.push(at);
        }
    };
    for (i, &x) in w.iter().enumerate() {
        if x & 0x9f00_0000 == 0x1000_0000 {
            let t = (i as i64 * 4) + adrp_imm21(x);
            record(t, i as i64 * 4);
            continue;
        }
        if x & 0x9f00_0000 != 0x9000_0000 {
            continue;
        }
        let rd = x & 31;
        let page = ((i as i64 * 4) & !0xfff) + (adrp_imm21(x) << 12);
        if page + 0x1000 <= 0 || page >= d.len() as i64 {
            continue;
        }
        for (j, &y) in w.iter().skip(i + 1).take(8).enumerate() {
            if y & 0xffc0_0000 == 0x9100_0000 && (y >> 5) & 31 == rd {
                let t = page + i64::from((y >> 10) & 0xfff);
                record(t, (i as i64 + 1 + j as i64) * 4);
                break;
            }
        }
    }
    (out, code_targets)
}

fn valid_pair(ctl: &[u8], code: &[u8]) -> Option<&'static str> {
    if code.is_empty() || code.windows(4).any(|w| w == CTL_MAGIC) {
        return None;
    }
    let st = stage(ctl, code)?;
    check(st, ctl, code).is_none().then_some(st)
}

fn adjacent_candidates(c: usize, nctl: usize, n: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for align in [16usize, 1, 4, 8, 32, 64, 128, 256, 512, 4096] {
        let k = ((c + nctl + align - 1) & !(align - 1)) as i64;
        if !out.contains(&k) {
            out.push(k);
        }
    }
    let before = c as i64 - n as i64;
    if !out.contains(&before) {
        out.push(before);
    }
    if before >= 0 {
        for align in [16usize, 32, 64, 128, 256] {
            let k = ((before as usize) & !(align - 1)) as i64;
            if !out.contains(&k) {
                out.push(k);
            }
        }
    }
    out
}

/// `raw_shaders`. `known`: control sections the other readers took from this main, left out.
pub fn raw_shaders(d: &[u8], label: &str, known: &HashSet<&[u8]>) -> R<Vec<Found>> {
    let mut skip: Vec<(usize, usize)> = Vec::new();
    for h in find_all(d, &[0x91, 0x68, 0x86, 0x19]) {
        let (size, _, first, _) = layout(d, h);
        let min = if first == 0xa8 { 0 } else { 0x760 };
        if min < size && size <= (d.len() - h) as u64 {
            skip.push((h, h + size as usize));
        }
    }
    for h in find_all(d, b"BNSH\0\0\0\0") {
        skip.push((h, h + u32le(d, h + 0x1c)?));
    }
    let mut ctls = Vec::new();
    for c in find_all(d, &CTL_MAGIC) {
        if c & 3 != 0 || skip.iter().any(|&(a, b)| a <= c && c < b) {
            continue;
        }
        let n = u32le(d, c + 0x20)?;
        if !known.contains(d.get(c..c + n).unwrap_or(&d[c.min(d.len())..])) {
            ctls.push(c);
        }
    }
    if ctls.is_empty() {
        return Ok(Vec::new());
    }
    let rel = rel_targets(d)?;
    let cset: HashSet<usize> = ctls.iter().copied().collect();
    let mut ctl_slots: HashMap<usize, Vec<u64>> = HashMap::new();
    let mut code_of: HashMap<usize, u64> = HashMap::new();
    for &s in &rel.keys {
        let c = rel.val[&s] as usize;
        if cset.contains(&c) {
            ctl_slots.entry(c).or_default().push(s);
            if s >= 8
                && rel.val.contains_key(&(s - 8))
                && !code_of.contains_key(&c)
                && crate::prim::u32le(d, s as usize + 8) == crate::prim::u32le(d, c + 0x704)
            {
                code_of.insert(c, rel.val[&(s - 8)]);
            }
        }
    }
    for &c in &ctls {
        if code_of.contains_key(&c) {
            continue;
        }
        let Some(slots) = ctl_slots.get(&c) else {
            continue;
        };
        let (Some(nctl), Some(n)) = (
            crate::prim::u32le(d, c + 0x20),
            crate::prim::u32le(d, c + 0x704),
        ) else {
            continue;
        };
        let Some(ctl) = d.get(c..c + nctl as usize) else {
            continue;
        };
        'slot: for &s in slots {
            for delta in [8i64, -8, 16, -16, 24, -24, 32, -32] {
                let Some(ns) = s.checked_add_signed(delta) else {
                    continue;
                };
                let Some(&k) = rel.val.get(&ns) else {
                    continue;
                };
                if k as usize == c {
                    continue;
                }
                if let Some(code) = d.get(k as usize..k as usize + n as usize)
                    && valid_pair(ctl, code).is_some()
                {
                    code_of.insert(c, k);
                    break 'slot;
                }
            }
        }
    }
    let (built, code_targets) = if ctls.iter().any(|c| !code_of.contains_key(c)) {
        let (mn, mx) = (
            ctls.iter().min().copied().unwrap_or(0),
            ctls.iter().max().copied().unwrap_or(0),
        );
        adrp_targets(d, mn as i64 - 0x10_0000, mx as i64 + 0x10_0000, &cset)
    } else {
        (HashMap::new(), Vec::new())
    };
    let mut out = Vec::new();
    for &c in &ctls {
        let (nctl, n) = (u32le(d, c + 0x20)?, u32le(d, c + 0x704)?);
        let ctl = d.get(c..c + nctl).unwrap_or(&d[c.min(d.len())..]);
        let near = |k: i64| -> i64 {
            let (a, b) = (&built[&k], &built[&(c as i64)]);
            a.iter()
                .flat_map(|x| b.iter().map(move |y| (x - y).abs()))
                .min()
                .unwrap_or(i64::MAX)
        };
        let mut cands: Vec<i64> = if let Some(&k) = code_of.get(&c) {
            vec![k as i64]
        } else {
            let adj = adjacent_candidates(c, nctl, n);
            let mut v: Vec<i64> = adj
                .iter()
                .copied()
                .filter(|k| {
                    built.contains_key(k) && built.contains_key(&(c as i64)) && near(*k) < 0x400
                })
                .collect();
            v.sort_by_key(|&k| near(k));
            if built.contains_key(&(c as i64)) {
                let mut non_adj: Vec<i64> = code_targets
                    .iter()
                    .copied()
                    .filter(|&k| k != c as i64 && !v.contains(&k) && near(k) < 0x400)
                    .collect();
                non_adj.sort_by_key(|&k| near(k));
                v.extend(non_adj);
            }
            v
        };
        cands.dedup();
        let mut why = Some(
            "no code area: no data record points at one, and the code builds the address of none right after or right before it",
        );
        for k in cands {
            let code = if k >= 0 {
                d.get(k as usize..k as usize + n)
                    .unwrap_or(&d[(k as usize).min(d.len())..])
            } else {
                &[][..]
            };
            if k >= 0
                && let Some(st) = valid_pair(ctl, code)
            {
                out.push(Found {
                    place: format!("{label}#RAW@{c:#x}"),
                    stage: st,
                    ctl: ctl.to_vec(),
                    code: code.to_vec(),
                });
                why = None;
                break;
            }
        }
        if let Some(w) = why {
            println!("[RED] {label} control section @{c:#x}: {w}; left out");
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn encode_adrp(rd: u32, pc: usize, target_page: usize) -> u32 {
        let delta = ((target_page as i64 - (pc as i64 & !0xfff)) >> 12) as i32;
        let imm21 = (delta as u32) & 0x1f_ffff;
        let immlo = imm21 & 3;
        let immhi = (imm21 >> 2) & 0x7ffff;
        0x9000_0000 | (immlo << 29) | (immhi << 5) | (rd & 31)
    }

    fn encode_add(rd: u32, rn: u32, imm12: usize) -> u32 {
        0x9100_0000 | (((imm12 as u32) & 0xfff) << 10) | ((rn & 31) << 5) | (rd & 31)
    }

    fn encode_adr(rd: u32, pc: usize, target: usize) -> u32 {
        let delta = (target as i64 - pc as i64) as i32;
        let imm21 = (delta as u32) & 0x1f_ffff;
        let immlo = imm21 & 3;
        let immhi = (imm21 >> 2) & 0x7ffff;
        0x1000_0000 | (immlo << 29) | (immhi << 5) | (rd & 31)
    }

    fn empty_mod0_image(len: usize) -> Vec<u8> {
        let mut img = vec![0u8; len];
        img[0x04..0x08].copy_from_slice(&0x40u32.to_le_bytes());
        img[0x40..0x44].copy_from_slice(b"MOD0");
        img[0x44..0x48].copy_from_slice(&0x10u32.to_le_bytes());
        img
    }

    fn write_vs(img: &mut [u8], ctl_at: usize, code_at: usize, tag: u8) {
        let nctl = 0x880usize;
        let ncode = 0x80usize;
        img[ctl_at..ctl_at + 4].copy_from_slice(&CTL_MAGIC);
        img[ctl_at + 0x20..ctl_at + 0x24].copy_from_slice(&(nctl as u32).to_le_bytes());
        img[ctl_at + 0x704..ctl_at + 0x708].copy_from_slice(&(ncode as u32).to_le_bytes());
        img[ctl_at + 0x100] = tag;

        img[code_at..code_at + 4].copy_from_slice(&GFX_MAGIC);
        img[code_at + 0x30..code_at + 0x34].copy_from_slice(&(1u32 << 10).to_le_bytes());
        img[code_at + 0x40] = tag;
    }

    #[test]
    fn matches_backward_adrp_adr_256_align_and_non_adjacent_code() {
        let mut img = empty_mod0_image(0x20_0000);
        // 1. Control section at 0x4000, code area at 0x18_0000 (>1 MiB away!), loaded by backward ADRP+ADD at 0x1a_0000
        write_vs(&mut img, 0x4000, 0x18_0000, 1);
        let insns = [
            encode_adrp(0, 0x1a_0000, 0x4000),
            encode_add(0, 0, 0),
            encode_adrp(1, 0x1a_0008, 0x18_0000),
            encode_add(1, 1, 0),
        ];
        for (i, w) in insns.iter().enumerate() {
            img[0x1a_0000 + 4 * i..0x1a_0004 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }

        // 2. Control section at 0x8000, 256B-aligned code right after (0x8900), loaded by ADR within +-1 MiB
        write_vs(&mut img, 0x8000, 0x8900, 2);
        let adr_insns = [encode_adr(2, 0x1000, 0x8000), encode_adr(3, 0x1004, 0x8900)];
        for (i, w) in adr_insns.iter().enumerate() {
            img[0x1000 + 4 * i..0x1004 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }

        let found = raw_shaders(&img, "main", &HashSet::new()).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].place, "main#RAW@0x4000");
        assert_eq!(found[0].code[0x40], 1);
        assert_eq!(found[1].place, "main#RAW@0x8000");
        assert_eq!(found[1].code[0x40], 2);
    }

    #[test]
    fn matches_ctl_first_relocated_record() {
        let mut img = empty_mod0_image(0x10000);
        // Dynamic table at 0x50: DT_RELA=0x200, DT_RELASZ=48
        img[0x50..0x58].copy_from_slice(&7u64.to_le_bytes());
        img[0x58..0x60].copy_from_slice(&0x200u64.to_le_bytes());
        img[0x60..0x68].copy_from_slice(&8u64.to_le_bytes());
        img[0x68..0x70].copy_from_slice(&48u64.to_le_bytes());

        // Struct at 0x2000: { ctl_ptr (0x4000), code_ptr (0x7000) } — ctl first, no size field
        for (i, (off, add)) in [(0x2000u64, 0x4000u64), (0x2008, 0x7000)]
            .into_iter()
            .enumerate()
        {
            let at = 0x200 + 24 * i;
            img[at..at + 8].copy_from_slice(&off.to_le_bytes());
            img[at + 8..at + 16].copy_from_slice(&0x403u64.to_le_bytes());
            img[at + 16..at + 24].copy_from_slice(&add.to_le_bytes());
        }
        write_vs(&mut img, 0x4000, 0x7000, 7);

        let found = raw_shaders(&img, "main", &HashSet::new()).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].place, "main#RAW@0x4000");
        assert_eq!(found[0].code[0x40], 7);
    }
}
