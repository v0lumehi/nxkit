//! Spin waits, part two: the copy of a region in the patch area, its spin gate and the entries from the original code.

use std::collections::HashMap;

use nxb_a64::{Insn, encode};

use crate::error::fail;
use crate::rw::{Gate, R, Rw, reach};
use crate::spin::{NK, Region, norm};

/// The direct branch `x` re-encoded at `pc` with target `tgt`.
fn spin_rebranch(x: &Insn, pc: i64, tgt: i64) -> R<u32> {
    let w = x.word;
    let off = tgt - pc;
    if x.mn == "b" {
        return reach(encode::b(pc, tgt));
    }
    if matches!(x.mn, "tbz" | "tbnz") {
        if !(-(1 << 15)..(1 << 15)).contains(&off) || off % 4 != 0 {
            return fail(format!(
                "assertion failed: spin wait tbz out of range {pc:#x}"
            ));
        }
        return Ok((w & !(0x3fff << 5)) | ((((off >> 2) as u32) & 0x3fff) << 5));
    }
    if !(-(1 << 20)..(1 << 20)).contains(&off) || off % 4 != 0 {
        return fail(format!(
            "assertion failed: spin wait branch out of range {pc:#x}"
        ));
    }
    Ok((w & !(0x7_ffff << 5)) | ((((off >> 2) as u32) & 0x7_ffff) << 5))
}

enum Info {
    None,
    Cpy(i64),
    Orig(i64),
}

impl Rw {
    fn patch_len(&self) -> usize {
        if self.in_back {
            self.back_area.len()
        } else {
            self.front_area.len()
        }
    }

    fn patch_base(&self) -> i64 {
        if self.in_back {
            self.back_off
        } else {
            self.patch_off
        }
    }

    fn patch_write(&mut self, idx: usize, w: u32) {
        let a = if self.in_back {
            &mut self.back_area
        } else {
            &mut self.front_area
        };
        if let Some(s) = a.get_mut(idx..idx + 4) {
            s.copy_from_slice(&w.to_le_bytes());
        }
    }

    /// Emits one found region.
    pub(crate) fn spin_emit(&mut self, g: &Region) -> R<()> {
        let t = g.t;
        self.use_area(t);
        let mut cpy: HashMap<i64, i64> = HashMap::new();
        let mut fixes: Vec<(usize, Option<Insn>, Info)> = Vec::new();
        let mut tramp: HashMap<i64, i64> = HashMap::new();
        for (i, a) in g.r.iter().enumerate() {
            let n = &g.nodes[a];
            cpy.insert(*a, self.here());
            let x = &n.ins[0];
            match n.kind {
                NK::Rmw => {
                    for y in &n.ins[..3] {
                        self.emit(y.word);
                    }
                    fixes.push((self.patch_len(), Some(n.ins[3].clone()), Info::Cpy(*a)));
                    self.emit(0);
                }
                NK::Acq | NK::Rel => {
                    cpy.insert(*a, self.here() + 4 * n.pre as i64);
                    for y in &n.ins {
                        self.emit(y.word);
                    }
                }
                NK::Tst => {
                    for y in &n.ins[..2] {
                        self.emit(y.word);
                    }
                    fixes.push((self.patch_len(), Some(n.ins[2].clone()), Info::Orig(*a)));
                    self.emit(0);
                }
                NK::Pc => {
                    let rd = x.reg(0).as_ref().and_then(norm).unwrap_or(0) as u32;
                    let tgt = x.imm(1).unwrap_or(0);
                    let w = reach(encode::adrp(rd, self.here(), tgt))?;
                    self.emit(w);
                    if x.mn == "adr" {
                        self.emit(encode::addi(rd, rd, (tgt & 0xfff) as u32, 0));
                    }
                }
                NK::Br => {
                    fixes.push((self.patch_len(), Some(x.clone()), Info::None));
                    self.emit(0);
                }
                _ => self.emit(x.word),
            }
            if !(n.kind == NK::Br && x.mn == "b") {
                let ft = *a + 4 * (n.ins.len() as i64 - n.pre as i64);
                if !(i + 1 < g.r.len() && g.r[i + 1] == ft) {
                    fixes.push((self.patch_len(), None, Info::Cpy(ft)));
                    self.emit(0);
                }
            }
        }
        let pause = self.here();
        self.emit(encode::stp_pre(29, 30, 31, -16));
        self.emit_gate_call()?;
        self.emit(encode::blr(30));
        let ret = self.here();
        self.emit(encode::ldp_post(29, 30, 31, 16));
        self.gates.push(Gate {
            ret,
            kind: "spin".into(),
            arg: 0,
            orig: t,
        });
        self.emit_b(cpy[&t])?;
        for (idx, x, info) in fixes {
            let pc = self.patch_base() + idx as i64;
            let w = match (&x, &info) {
                (None, Info::Cpy(v)) => match cpy.get(v).copied() {
                    Some(tgt) => reach(encode::b(pc, tgt))?,
                    None => self.branch_word(pc, *v)?,
                },
                (None, _) => return fail("spin fix"),
                (Some(x), Info::Orig(o)) => {
                    let d = self.spin_out(&mut tramp, *o)?;
                    spin_rebranch(x, pc, d)?
                }
                (Some(x), Info::Cpy(a)) => spin_rebranch(x, pc, cpy[a])?,
                (Some(x), Info::None) => {
                    let tg = x.target().unwrap_or(0) as i64;
                    let cond = x.mn != "b";
                    if tg == t && g.valid.contains(&(x.addr as i64)) {
                        spin_rebranch(x, pc, pause)?
                    } else if cpy.contains_key(&tg) && tg != t {
                        spin_rebranch(x, pc, cpy[&tg])?
                    } else if cond {
                        let d = self.spin_out(&mut tramp, tg)?;
                        spin_rebranch(x, pc, d)?
                    } else {
                        self.branch_word(pc, tg)?
                    }
                }
            };
            self.patch_write(idx, w);
        }
        for b in &g.valid {
            let x = &g.nodes[b].ins[0];
            if x.mn == "b" {
                self.put_branch(*b, cpy[&t])?;
            } else {
                let st = self.here();
                let w = spin_rebranch(x, self.here(), cpy[&t])?;
                self.emit(w);
                self.emit_b(*b + 4)?;
                self.put_branch(*b, st)?;
            }
        }
        self.stat("spin");
        Ok(())
    }

    fn spin_out(&mut self, tramp: &mut HashMap<i64, i64>, t: i64) -> R<i64> {
        if let Some(v) = tramp.get(&t) {
            return Ok(*v);
        }
        let h = self.here();
        tramp.insert(t, h);
        self.emit_b(t)?;
        Ok(h)
    }
}
