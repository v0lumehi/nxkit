//! Locates `OodleLZ_Decompress` in loaded Switch guest modules without symbol exports.
//!
//! When a game statically links `liboo2core` into `main` (or a `subsdk*`), the host looks for
//! Oodle's string references in `.rodata` (`"OodleLZ_Decompress"`, `"Oodle Data Compression"`,
//! `"Oodle2"`, `"OodleLZ"`) and maps `ADRP`+`ADD` / `ADR` sites to either the enclosing
//! 14-argument decoder function or an adjacent `BL` target, falling back to a 14-argument
//! call-site + callee stack-argument signature scan when `.rodata` names `"Oodle"`.

use std::collections::HashMap;

/// Read-only view of one loaded guest module's `.text` and `.rodata` segments.
pub struct ModuleView<'a> {
    /// Module name (`"main"`, `"subsdk0"`, ...).
    pub name: &'a str,
    /// Guest base address.
    pub base: u64,
    /// Byte offset of `.text` from `base`.
    pub text_offset: u64,
    /// `.text` bytes.
    pub text: &'a [u8],
    /// Byte offset of `.rodata` from `base`.
    pub rodata_offset: u64,
    /// `.rodata` bytes.
    pub rodata: &'a [u8],
}

/// Located Oodle decoder entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundDecoder {
    /// Module that holds the decoder.
    pub module: String,
    /// Guest address of `OodleLZ_Decompress`.
    pub addr: u64,
    /// Strategy that found it (`string-callsite`, `string-body`, or `signature-callsite`).
    pub how: &'static str,
}

fn word_at(b: &[u8], idx: usize) -> Option<u32> {
    let at = idx.checked_mul(4)?;
    let s = b.get(at..at + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn sign_ext(v: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v as i64) << shift) >> shift
}

fn adrp_page(w: u32, pc: u64) -> Option<(u32, u64)> {
    if (w & 0x9F00_0000) != 0x9000_0000 {
        return None;
    }
    let rd = w & 31;
    if rd == 31 {
        return None;
    }
    let imm21 = ((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2);
    let page = ((pc & !0xFFF) as i64).wrapping_add(sign_ext(imm21, 21) << 12);
    u64::try_from(page).ok().map(|p| (rd, p))
}

fn adr_target(w: u32, pc: u64) -> Option<(u32, u64)> {
    if (w & 0x9F00_0000) != 0x1000_0000 {
        return None;
    }
    let rd = w & 31;
    if rd == 31 {
        return None;
    }
    let imm21 = ((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2);
    let addr = (pc as i64).wrapping_add(sign_ext(imm21, 21));
    u64::try_from(addr).ok().map(|a| (rd, a))
}

fn add_imm12(w: u32) -> Option<(u32, u32, u64)> {
    if (w & 0xFFC0_0000) != 0x9100_0000 {
        return None;
    }
    let (rd, rn) = (w & 31, (w >> 5) & 31);
    if rd == 31 || rn == 31 {
        return None;
    }
    Some((rd, rn, u64::from((w >> 10) & 0xFFF)))
}

fn bl_target(w: u32, pc: u64) -> Option<u64> {
    if (w & 0xFC00_0000) != 0x9400_0000 {
        return None;
    }
    let off = sign_ext(w & 0x03FF_FFFF, 26) << 2;
    u64::try_from((pc as i64).wrapping_add(off)).ok()
}

fn is_boundary(w: u32) -> bool {
    w == 0
        || (w & 0xFFFF_FC1F) == 0xD65F_0000 // RET
        || (w & 0xFC00_0000) == 0x1400_0000 // B
        || (w & 0xFFFF_FC1F) == 0xD61F_0000 // BR
        || (w & 0xFFE0_001F) == 0xD420_0000 // BRK
}

fn is_prologue(w: u32) -> bool {
    let stp_pre_sp = (w & 0xFFC0_03E0) == 0xA980_03E0 && ((w >> 15) & 0x7F) >= 0x40;
    let sub_sp = (w & 0xFF80_03FF) == 0xD100_03FF;
    stp_pre_sp || sub_sp
}

fn is_pac_or_bti(w: u32) -> bool {
    w == 0xD503_237F || (w & 0xFFFF_FF3F) == 0xD503_241F
}

/// Finds the start VA of the function enclosing `idx` in `text` (at `text_va`).
fn enclosing_func(text: &[u8], text_va: u64, idx: usize) -> Option<u64> {
    let mut k = idx;
    let floor = idx.saturating_sub(4096);
    let mut candidate = None;
    while k > floor {
        let Some(w) = word_at(text, k) else { break };
        if k < idx && is_boundary(w) {
            return candidate;
        }
        if is_prologue(w) {
            let start = if k > 0 && word_at(text, k - 1).is_some_and(is_pac_or_bti) {
                k - 1
            } else {
                k
            };
            candidate = Some(text_va + (start as u64) * 4);
            if start == 0 || word_at(text, start - 1).is_some_and(is_boundary) {
                return candidate;
            }
        }
        k -= 1;
    }
    if k == 0 && word_at(text, 0).is_some_and(|w| is_prologue(w) || is_pac_or_bti(w)) {
        return Some(text_va);
    }
    candidate
}

/// Stack-frame adjustment at function entry (`SUB sp, sp, #imm` or `STP ..., [sp, #-imm]!`).
fn stack_alloc(w: u32) -> Option<i64> {
    if (w & 0xFF80_03FF) == 0xD100_03FF {
        let sh = (w >> 22) & 1;
        let imm = i64::from((w >> 10) & 0xFFF) << (12 * sh);
        return Some(imm);
    }
    if (w & 0xFFC0_03E0) == 0xA980_03E0 {
        let imm7 = sign_ext((w >> 15) & 0x7F, 7) * 8;
        if imm7 < 0 {
            return Some(-imm7);
        }
    }
    None
}

/// Caller-stack offsets read relative to incoming `sp` (`[sp + off]` or `[x29 + off]`).
fn stack_load_offsets(w: u32, sp_depth: i64, fp_depth: Option<i64>) -> [Option<i64>; 2] {
    let base_depth = |rn: u32| -> Option<i64> {
        if rn == 31 {
            Some(sp_depth)
        } else if rn == 29 {
            fp_depth
        } else {
            None
        }
    };
    // LDR Xt, [Xn, #uimm12*8]
    if (w & 0xFFC0_0000) == 0xF940_0000
        && let Some(d) = base_depth((w >> 5) & 31)
    {
        let off = i64::from((w >> 10) & 0xFFF) * 8 - d;
        return [Some(off), None];
    }
    // LDR Wt, [Xn, #uimm12*4]
    if (w & 0xFFC0_0000) == 0xB940_0000
        && let Some(d) = base_depth((w >> 5) & 31)
    {
        let off = i64::from((w >> 10) & 0xFFF) * 4 - d;
        return [Some(off), None];
    }
    // LDP Xt1, Xt2, [Xn, #simm7*8]
    if (w & 0xFFC0_0000) == 0xA940_0000
        && let Some(d) = base_depth((w >> 5) & 31)
    {
        let off = sign_ext((w >> 15) & 0x7F, 7) * 8 - d;
        return [Some(off), Some(off + 8)];
    }
    [None, None]
}

/// True when `func_va` looks like `OodleLZ_Decompress`: reads multiple caller stack arguments in `[0..=40]`.
fn callee_reads_oodle_stack_args(text: &[u8], text_va: u64, func_va: u64) -> bool {
    let Some(off) = func_va.checked_sub(text_va) else {
        return false;
    };
    if off % 4 != 0 {
        return false;
    }
    let start = (off / 4) as usize;
    let mut sp_depth = 0i64;
    let mut fp_depth: Option<i64> = None;
    let mut seen = [false; 6];
    for i in start..(start + 96).min(text.len() / 4) {
        let Some(w) = word_at(text, i) else { break };
        if i > start && is_boundary(w) {
            break;
        }
        if let Some(a) = stack_alloc(w) {
            sp_depth += a;
            continue;
        }
        // ADD x29, sp, #imm12
        if (w & 0xFFC0_03FF) == 0x9100_03FD {
            let imm = i64::from((w >> 10) & 0xFFF);
            fp_depth = Some(sp_depth - imm);
            continue;
        }
        for o in stack_load_offsets(w, sp_depth, fp_depth)
            .into_iter()
            .flatten()
        {
            if (0..=40).contains(&o) && o % 8 == 0 {
                seen[(o / 8) as usize] = true;
            }
        }
    }
    let count = seen.iter().filter(|&&b| b).count();
    count >= 2 && (seen[5] || seen[3] || seen[4])
}

/// Scores a `BL` site at `idx` as a call to `OodleLZ_Decompress` (14 arguments, `[sp+0..sp+40]`).
fn callsite_score(text: &[u8], idx: usize) -> u32 {
    let mut slots = [false; 6];
    let mut thread_phase_hint = false;
    let mut fuzz_or_crc_or_null = 0u32;
    for k in (idx.saturating_sub(24)..idx).rev() {
        let Some(w) = word_at(text, k) else { break };
        if is_boundary(w) || bl_target(w, 0).is_some() {
            break;
        }
        // STR Xt, [sp, #uimm12*8]
        if (w & 0xFFC0_03E0) == 0xF900_03E0 {
            let off = ((w >> 10) & 0xFFF) * 8;
            if off <= 40 && off % 8 == 0 {
                slots[(off / 8) as usize] = true;
            }
        } else if (w & 0xFFC0_03E0) == 0xB900_03E0 {
            // STR Wt, [sp, #uimm12*4]
            let off = ((w >> 10) & 0xFFF) * 4;
            if off <= 40 && off % 8 == 0 {
                slots[(off / 8) as usize] = true;
            }
        } else if (w & 0xFFC0_03E0) == 0xA900_03E0 {
            // STP Xt1, Xt2, [sp, #simm7*8]
            let off = sign_ext((w >> 15) & 0x7F, 7) * 8;
            if (0..=32).contains(&off) && off % 8 == 0 {
                slots[(off / 8) as usize] = true;
                slots[(off / 8) as usize + 1] = true;
            }
        }
        // MOV w4/w5/w6/x7, #imm or wzr/xzr
        let rd = w & 31;
        if matches!(rd, 4..=7) {
            if (w & 0xFFFF_FFE0) == 0xAA1F_03E0 || (w & 0xFFFF_FFE0) == 0x2A1F_03E0 {
                fuzz_or_crc_or_null += 1;
            } else if (w & 0xFF80_0000) == 0x5280_0000 || (w & 0xFF80_0000) == 0xD280_0000 {
                let imm = (w >> 5) & 0xFFFF;
                if imm <= 3 {
                    fuzz_or_crc_or_null += 1;
                }
            }
        }
        // MOV wN, #3 (threadPhase = OodleLZ_Decode_Unthreaded)
        if ((w & 0xFF80_0000) == 0x5280_0000 || (w & 0xFF80_0000) == 0xD280_0000)
            && ((w >> 5) & 0xFFFF) == 3
        {
            thread_phase_hint = true;
        }
    }
    if !slots[5] {
        return 0;
    }
    let stack_count = slots.iter().filter(|&&b| b).count() as u32;
    if stack_count < 3 {
        return 0;
    }
    stack_count + fuzz_or_crc_or_null + u32::from(thread_phase_hint)
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + needle.len() <= hay.len() {
        if let Some(pos) = hay[i..].windows(needle.len()).position(|w| w == needle) {
            out.push(i + pos);
            i += pos + needle.len();
        } else {
            break;
        }
    }
    out
}

/// Finds `text` instruction indices that compute any VA in `targets` via `ADRP`+`ADD` or `ADR`.
fn find_string_refs(text: &[u8], text_va: u64, targets: &[u64]) -> Vec<usize> {
    let mut out = Vec::new();
    let n = text.len() / 4;
    let mut page = [None::<(usize, u64)>; 31];
    for i in 0..n {
        let pc = text_va + (i as u64) * 4;
        let Some(w) = word_at(text, i) else { break };
        if let Some((rd, addr)) = adr_target(w, pc) {
            if targets.contains(&addr) {
                out.push(i);
            }
            page[rd as usize] = None;
            continue;
        }
        if let Some((rd, p)) = adrp_page(w, pc) {
            page[rd as usize] = Some((i, p));
            continue;
        }
        if let Some((rd, rn, lo)) = add_imm12(w) {
            if let Some((at, p)) = page[rn as usize]
                && i.saturating_sub(at) <= 8
            {
                let addr = p + lo;
                if targets.contains(&addr) {
                    out.push(i);
                }
            }
            page[rd as usize] = None;
            continue;
        }
        if is_boundary(w) || bl_target(w, 0).is_some() {
            page = [None; 31];
        } else {
            let rd = (w & 31) as usize;
            if rd < 31 {
                page[rd] = None;
            }
        }
    }
    out
}

fn search_module(m: &ModuleView<'_>) -> Option<FoundDecoder> {
    let text_va = m.base + m.text_offset;
    let ro_va = m.base + m.rodata_offset;
    let n = m.text.len() / 4;
    // Anchor strings in priority order: exact function name first, then library strings.
    let anchors: &[&[u8]] = &[
        b"OodleLZ_Decompress",
        b"Oodle Data Compression",
        b"Oodle2",
        b"OodleLZ",
    ];
    for pat in anchors {
        let mut target_vas = Vec::new();
        for off in find_bytes(m.rodata, pat) {
            // Walk back to the start of the C string when inside a message.
            let start = m.rodata[..off]
                .iter()
                .rposition(|&b| b == 0)
                .map_or(0, |p| p + 1);
            target_vas.push(ro_va + off as u64);
            if start != off {
                target_vas.push(ro_va + start as u64);
            }
        }
        if target_vas.is_empty() {
            continue;
        }
        let refs = find_string_refs(m.text, text_va, &target_vas);
        for &idx in &refs {
            let lo = idx.saturating_sub(32);
            let hi = (idx + 16).min(n);
            for k in lo..hi {
                let pc = text_va + (k as u64) * 4;
                if let Some(w) = word_at(m.text, k)
                    && let Some(target) = bl_target(w, pc)
                    && callee_reads_oodle_stack_args(m.text, text_va, target)
                {
                    return Some(FoundDecoder {
                        module: m.name.to_string(),
                        addr: target,
                        how: "string-callsite",
                    });
                }
            }
        }
        for &idx in &refs {
            if let Some(func_va) = enclosing_func(m.text, text_va, idx)
                && callee_reads_oodle_stack_args(m.text, text_va, func_va)
            {
                return Some(FoundDecoder {
                    module: m.name.to_string(),
                    addr: func_va,
                    how: "string-body",
                });
            }
        }
    }
    if find_bytes(m.rodata, b"Oodle").is_empty() {
        return None;
    }
    let mut scores: HashMap<u64, u32> = HashMap::new();
    for i in 0..n {
        let pc = text_va + (i as u64) * 4;
        let Some(w) = word_at(m.text, i) else { break };
        let Some(target) = bl_target(w, pc) else {
            continue;
        };
        let sc = callsite_score(m.text, i);
        if sc >= 5 && callee_reads_oodle_stack_args(m.text, text_va, target) {
            *scores.entry(target).or_insert(0) += sc;
        }
    }
    scores
        .into_iter()
        .max_by_key(|&(addr, sc)| (sc, std::cmp::Reverse(addr)))
        .map(|(addr, _)| FoundDecoder {
            module: m.name.to_string(),
            addr,
            how: "signature-callsite",
        })
}

/// Searches `modules` (in order, `main` first) for `OodleLZ_Decompress`.
pub fn find_decoder(modules: &[ModuleView<'_>]) -> Option<FoundDecoder> {
    let mut ordered: Vec<&ModuleView<'_>> = modules.iter().filter(|m| m.name == "main").collect();
    ordered.extend(modules.iter().filter(|m| m.name != "main"));
    for m in ordered {
        if let Some(found) = search_module(m) {
            return Some(found);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_adrp(rd: u32, pc: u64, target: u64) -> u32 {
        let delta = ((target & !0xFFF) as i64 - (pc & !0xFFF) as i64) >> 12;
        let imm21 = (delta as u32) & 0x1F_FFFF;
        0x9000_0000 | ((imm21 & 3) << 29) | (((imm21 >> 2) & 0x7FFFF) << 5) | (rd & 31)
    }

    fn enc_add(rd: u32, rn: u32, imm12: u64) -> u32 {
        0x9100_0000 | (((imm12 as u32) & 0xFFF) << 10) | ((rn & 31) << 5) | (rd & 31)
    }

    fn enc_bl(pc: u64, target: u64) -> u32 {
        let delta = (target as i64 - pc as i64) >> 2;
        0x9400_0000 | ((delta as u32) & 0x03FF_FFFF)
    }

    fn words_to_bytes(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Encodes a stub `OodleLZ_Decompress` at `func_idx` that saves frame and reads `[sp+frame+24]` and `[sp+frame+40]`.
    fn push_oodle_callee(ws: &mut Vec<u32>) -> usize {
        let start = ws.len();
        ws.push(0xA9BF_7BFD); // stp x29, x30, [sp, #-16]!
        ws.push(0xD101_03FF); // sub sp, sp, #0x40 (total frame = 0x50 = 80)
        // ldr x8, [sp, #80 + 24] -> uimm12 = 104 / 8 = 13
        ws.push(0xF940_03E8 | (13 << 10));
        // ldr w9, [sp, #80 + 40] -> uimm12 = 120 / 4 = 30
        ws.push(0xB940_03E9 | (30 << 10));
        ws.push(0xD65F_03C0); // ret
        start
    }

    #[test]
    fn finds_decoder_via_string_referenced_in_body() {
        let base = 0x10_0000_0000u64;
        let text_off = 0x1000u64;
        let ro_off = 0x4000u64;
        let ro = b"\0OodleLZ_Decompress: bad input\0".to_vec();
        let str_va = base + ro_off + 1;
        let mut ws = Vec::new();
        let func_idx = push_oodle_callee(&mut ws);
        // Replace ret with adrp+add to string, then ret.
        ws.pop();
        let adrp_pc = base + text_off + (ws.len() as u64) * 4;
        ws.push(enc_adrp(0, adrp_pc, str_va));
        ws.push(enc_add(0, 0, str_va & 0xFFF));
        ws.push(0xD65F_03C0);
        let text = words_to_bytes(&ws);
        let m = ModuleView {
            name: "main",
            base,
            text_offset: text_off,
            text: &text,
            rodata_offset: ro_off,
            rodata: &ro,
        };
        let found = find_decoder(&[m]).expect("found");
        assert_eq!(found.addr, base + text_off + (func_idx as u64) * 4);
        assert_eq!(found.how, "string-body");
    }

    #[test]
    fn finds_decoder_via_wrapper_callsite_and_string() {
        let base = 0x10_0000_0000u64;
        let text_off = 0x1000u64;
        let ro_off = 0x4000u64;
        let ro = b"OodleLZ_Decompress failed\0".to_vec();
        let str_va = base + ro_off;
        let mut ws = Vec::new();
        let oodle_idx = push_oodle_callee(&mut ws);
        let oodle_va = base + text_off + (oodle_idx as u64) * 4;
        // Wrapper function calls BL oodle_va then references string.
        ws.push(0xA9BF_7BFD);
        let bl_pc = base + text_off + (ws.len() as u64) * 4;
        ws.push(enc_bl(bl_pc, oodle_va));
        let adrp_pc = base + text_off + (ws.len() as u64) * 4;
        ws.push(enc_adrp(1, adrp_pc, str_va));
        ws.push(enc_add(1, 1, str_va & 0xFFF));
        ws.push(0xD65F_03C0);
        let text = words_to_bytes(&ws);
        let m = ModuleView {
            name: "main",
            base,
            text_offset: text_off,
            text: &text,
            rodata_offset: ro_off,
            rodata: &ro,
        };
        let found = find_decoder(&[m]).expect("found");
        assert_eq!(found.addr, oodle_va);
        assert_eq!(found.how, "string-callsite");
    }

    #[test]
    fn finds_decoder_via_fourteen_arg_signature_when_only_oodle_string_exists() {
        let base = 0x10_0000_0000u64;
        let text_off = 0x1000u64;
        let ro_off = 0x4000u64;
        let ro = b"some unreferenced Oodle marker\0".to_vec();
        let mut ws = Vec::new();
        let oodle_idx = push_oodle_callee(&mut ws);
        let oodle_va = base + text_off + (oodle_idx as u64) * 4;
        // Caller sets x4=1, x5=0, x6=0, x7=0, stp xzr, xzr, [sp], str xzr, [sp, #16], mov w8, #3, str w8, [sp, #40], bl
        ws.push(0x5280_0024); // mov w4, #1
        ws.push(0xAA1F_03E5); // mov x5, xzr
        ws.push(0xAA1F_03E6); // mov x6, xzr
        ws.push(0xAA1F_03E7); // mov x7, xzr
        ws.push(0xA900_7FFF); // stp xzr, xzr, [sp]
        ws.push(0xF900_0BFF); // str xzr, [sp, #16]
        ws.push(0x5280_0068); // mov w8, #3
        ws.push(0xB900_2BE8); // str w8, [sp, #40] (40/4 = 10)
        let bl_pc = base + text_off + (ws.len() as u64) * 4;
        ws.push(enc_bl(bl_pc, oodle_va));
        ws.push(0xD65F_03C0);
        let text = words_to_bytes(&ws);
        let m = ModuleView {
            name: "main",
            base,
            text_offset: text_off,
            text: &text,
            rodata_offset: ro_off,
            rodata: &ro,
        };
        let found = find_decoder(&[m]).expect("found");
        assert_eq!(found.addr, oodle_va);
        assert_eq!(found.how, "signature-callsite");
    }

    #[test]
    fn returns_none_when_no_oodle_is_present() {
        let text = words_to_bytes(&[0xD65F_03C0]);
        let m = ModuleView {
            name: "main",
            base: 0x10_0000_0000,
            text_offset: 0x1000,
            text: &text,
            rodata_offset: 0x2000,
            rodata: b"no compression library here\0",
        };
        assert_eq!(find_decoder(&[m]), None);
    }
}
