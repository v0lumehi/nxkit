//! Unreal Engine paks with inline shaders and IoStore containers beside them (`ue_legacy_jobs`, `ue_legacy_shaders`,
//! `ue_io_shaders`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::uepak::{Pak, inline_shaders, normpath};
use super::uerec::inline_records;
use super::uetoc::{Toc, dirname, inline_in_containers, mount_order};
use super::{Ctx, File, Job, JobOut, Plan, R};
use crate::SplitError;
use crate::bahs::BahsVar;
use crate::src::Src;

/// Whether a pak entry can hold inline Switch shader records (`LSLG`).
///
/// UE 4.15+ cooked assets (EDL) split exports into `.uexp`, and global inline shaders live in
/// `GlobalShaderCache-*`. Skipping non-shader assets (such as ICU `.dict`, `.locres`, `.ini`,
/// `.ubulk`, or `.uasset` when `.uexp` / UE5 IoStore is present) avoids needlessly decompressing
/// non-shader entries (for example Oodle-compressed `.dict` files in UE5 paks).
fn may_hold_inline_shaders(name: &str, allow_uasset: bool) -> bool {
    let lower = name.to_ascii_lowercase();
    let base = name.rsplit('/').next().unwrap_or(name);
    lower.ends_with(".uexp")
        || base.starts_with("GlobalShaderCache-")
        || (allow_uasset && (lower.ends_with(".uasset") || lower.ends_with(".umap")))
}

fn legacy_shaders(pk: &Pak, rel: &str, names: &[String]) -> R<JobOut> {
    let allow_uasset = pk.version < 11
        && !pk
            .names
            .iter()
            .any(|n| n.to_ascii_lowercase().ends_with(".uexp"));
    let mut out = Vec::new();
    for name in names {
        if !may_hold_inline_shaders(name, allow_uasset) {
            continue;
        }
        let d = pk.read(name)?;
        let n = normpath(&format!("{}{}", pk.mount, name));
        let label = n.trim_start_matches("../");
        inline_records(&d, &format!("/{rel}#{label}"), &mut out)?;
    }
    Ok(JobOut {
        found: out,
        var: BahsVar::new(),
    })
}

/// A container with its `.ucas` partitions.
type Opened = (Toc, Vec<Option<Arc<dyn Src>>>);

/// `ue_legacy_jobs`.
pub fn plan(files: &[File], p: &mut Plan) -> R<()> {
    let beside = |rel: &str| {
        inline_in_containers(
            files
                .iter()
                .filter(|x| x.rel.ends_with(".utoc") && dirname(&x.rel) == dirname(rel))
                .map(|x| (x.rel.as_str(), x.src.as_ref())),
        )
    };
    let paks: Vec<usize> = (0..files.len())
        .filter(|&i| {
            files[i].rel.ends_with(".pak")
                && inline_shaders(files[i].src.as_ref(), &files[i].rel, &|| {
                    beside(&files[i].rel)
                })
        })
        .collect();
    if paks.is_empty() {
        return Ok(());
    }
    // (dir, normalised path) -> (pak, name, raw size): a patch pak replaces
    let mut view: BTreeMap<(String, String), (usize, String, u64)> = BTreeMap::new();
    let mut sorted = paks.clone();
    sorted.sort_by(|&a, &b| {
        (files[a].rel.ends_with("_P.pak"), &files[a].rel)
            .cmp(&(files[b].rel.ends_with("_P.pak"), &files[b].rel))
    });
    let mut paks_open: HashMap<usize, Arc<Pak>> = HashMap::new();
    for &i in &sorted {
        let f = &files[i];
        let r = Pak::open(f.src.clone(), &f.rel).and_then(|pk| {
            let mut v = Vec::new();
            for name in &pk.names {
                v.push((name.clone(), pk.entry(name)?.raw));
            }
            Ok((pk, v))
        });
        match r {
            Ok((pk, v)) => {
                for (name, raw) in v {
                    view.insert(
                        (
                            dirname(&f.rel).to_string(),
                            normpath(&format!("{}{name}", pk.mount)),
                        ),
                        (i, name, raw),
                    );
                }
                paks_open.insert(i, Arc::new(pk));
            }
            Err(SplitError::Red(e)) => p.notes.push(format!(
                "[RED] /{}: the pak cannot be read ({e}); its shaders are left out",
                f.rel
            )),
            Err(e) => return Err(e),
        }
    }
    let folders: HashSet<&str> = paks.iter().map(|&i| dirname(&files[i].rel)).collect();
    p.taken.extend(paks.iter().copied());
    let mut tocs: Vec<usize> = (0..files.len())
        .filter(|&i| files[i].rel.ends_with(".utoc") && folders.contains(dirname(&files[i].rel)))
        .collect();
    tocs.sort_by_key(|&i| mount_order(&files[i].rel));
    // (dir, chunk id) -> (toc, chunk index, length)
    let mut chunks: BTreeMap<(String, [u8; 12]), (usize, usize, u64)> = BTreeMap::new();
    let mut tocs_open: HashMap<usize, Arc<Opened>> = HashMap::new();
    for i in tocs {
        let f = &files[i];
        let t = match Toc::load(f.src.as_ref(), &f.rel) {
            Ok(t) => t,
            Err(SplitError::Red(e)) => {
                p.notes.push(format!(
                    "[RED] /{}: the container cannot be read ({e}); its shaders are left out",
                    f.rel
                ));
                continue;
            }
            Err(e) => return Err(e),
        };
        let bundles = t.export_bundles();
        if bundles.is_empty() {
            continue;
        }
        if t.encrypted {
            p.notes.push(format!("[RED] /{}: the container is encrypted, which is not read here; its shaders are left out", f.rel));
            continue;
        }
        p.taken.insert(i);
        let mut ucas = Vec::new();
        for up in t.ucas_paths() {
            let ui = files.iter().position(|x| x.rel == up);
            if let Some(ui) = ui {
                p.taken.insert(ui);
            }
            ucas.push(ui.map(|ui| files[ui].src.clone()));
        }
        for &b in &bundles {
            chunks.insert(
                (dirname(&f.rel).to_string(), t.ids[b]),
                (i, b, t.ranges[b].1),
            );
        }
        tocs_open.insert(i, Arc::new((t, ucas)));
    }
    // legacy files, then export bundles: each in jobs of about equal raw size
    let step = view.values().map(|v| v.2).sum::<u64>() as f64 / 64.0;
    let mut items: Vec<&(usize, String, u64)> = view.values().collect();
    items.sort_by(|a, b| (&files[a.0].rel, &a.1).cmp(&(&files[b.0].rel, &b.1)));
    let (mut cur, mut order, mut size) = (
        HashMap::<usize, Vec<String>>::new(),
        Vec::<usize>::new(),
        0u64,
    );
    let flush = |cur: &mut HashMap<usize, Vec<String>>, order: &mut Vec<usize>, p: &mut Plan| {
        for i in order.drain(..) {
            let names = cur.remove(&i).unwrap_or_default();
            let (pk, rel) = (paks_open[&i].clone(), files[i].rel.clone());
            let r2 = rel.clone();
            p.jobs.push(Job {
                rel,
                run: Box::new(move |_: &Ctx<'_>, _| legacy_shaders(&pk, &r2, &names)),
            });
        }
    };
    for (i, name, raw) in items {
        if !cur.contains_key(i) {
            order.push(*i);
        }
        cur.entry(*i).or_default().push(name.clone());
        size += raw;
        if size as f64 >= step {
            flush(&mut cur, &mut order, p);
            size = 0;
        }
    }
    flush(&mut cur, &mut order, p);
    let step = chunks.values().map(|v| v.2).sum::<u64>() as f64 / 64.0;
    let mut items: Vec<&(usize, usize, u64)> = chunks.values().collect();
    items.sort_by(|a, b| (&files[a.0].rel, a.1).cmp(&(&files[b.0].rel, b.1)));
    let (mut cur, mut order, mut size) = (
        HashMap::<usize, Vec<usize>>::new(),
        Vec::<usize>::new(),
        0u64,
    );
    let flush = |cur: &mut HashMap<usize, Vec<usize>>, order: &mut Vec<usize>, p: &mut Plan| {
        for i in order.drain(..) {
            let idxs = cur.remove(&i).unwrap_or_default();
            let (t, rel) = (tocs_open[&i].clone(), files[i].rel.clone());
            let r2 = rel.clone();
            p.jobs.push(Job {
                rel,
                run: Box::new(move |_: &Ctx<'_>, _| {
                    let names = t.0.names()?;
                    let mut out = Vec::new();
                    for &c in &idxs {
                        let d = t.0.read(c, &t.1)?;
                        let nm = names.get(&c).cloned().unwrap_or_else(|| {
                            t.0.ids[c].iter().map(|b| format!("{b:02x}")).collect()
                        });
                        inline_records(&d, &format!("/{r2}#{nm}"), &mut out)?;
                    }
                    Ok(JobOut {
                        found: out,
                        var: BahsVar::new(),
                    })
                }),
            });
        }
    };
    for (i, c, raw) in items {
        if !cur.contains_key(i) {
            order.push(*i);
        }
        cur.entry(*i).or_default().push(*c);
        size += raw;
        if size as f64 >= step {
            flush(&mut cur, &mut order, p);
            size = 0;
        }
    }
    flush(&mut cur, &mut order, p);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::own;
    use crate::oodle::{Answer, Decoder, set_local_decoder};
    use crate::zstd::Dicts;

    fn fstr(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as i32 + 1).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        out.push(0);
    }

    /// Builds a minimal v11 pak with method 1 = `"Oodle"`. Each entry is `(dir, file, oodle, data)`.
    fn build_v11_pak(entries: &[(&str, &str, bool, &[u8])]) -> Vec<u8> {
        let mut pak = Vec::new();
        let mut enc_blob = Vec::new();
        let mut by_dir: BTreeMap<&str, Vec<(&str, u32)>> = BTreeMap::new();
        for &(dir, file, oodle, data) in entries {
            let off = pak.len() as u32;
            pak.extend_from_slice(&[0u8; 53]);
            if oodle {
                pak.extend_from_slice(&1u32.to_le_bytes());
                pak.extend_from_slice(&[0u8; 16]);
            }
            pak.extend_from_slice(data);
            let at = enc_blob.len() as u32;
            if oodle {
                let v = (1u32 << 31) | (1 << 30) | (1 << 29) | (1 << 23) | (1 << 6) | 1;
                enc_blob.extend_from_slice(&v.to_le_bytes());
                enc_blob.extend_from_slice(&off.to_le_bytes());
                enc_blob.extend_from_slice(&(data.len() as u32).to_le_bytes());
                enc_blob.extend_from_slice(&(data.len() as u32).to_le_bytes());
                enc_blob.extend_from_slice(&(data.len() as u32).to_le_bytes());
            } else {
                let v = (1u32 << 31) | (1 << 30);
                enc_blob.extend_from_slice(&v.to_le_bytes());
                enc_blob.extend_from_slice(&off.to_le_bytes());
                enc_blob.extend_from_slice(&(data.len() as u32).to_le_bytes());
            }
            by_dir.entry(dir).or_default().push((file, at));
        }
        let doff = pak.len() as u64;
        let mut dir_idx = Vec::new();
        dir_idx.extend_from_slice(&(by_dir.len() as i32).to_le_bytes());
        for (dir, files) in by_dir {
            fstr(&mut dir_idx, dir);
            dir_idx.extend_from_slice(&(files.len() as i32).to_le_bytes());
            for (file, at) in files {
                fstr(&mut dir_idx, file);
                dir_idx.extend_from_slice(&at.to_le_bytes());
            }
        }
        let dsz = dir_idx.len() as u64;
        pak.extend_from_slice(&dir_idx);
        let ioff = pak.len() as u64;
        let mut idx = Vec::new();
        fstr(&mut idx, "../../../");
        idx.extend_from_slice(&0i32.to_le_bytes());
        idx.extend_from_slice(&0u64.to_le_bytes());
        idx.extend_from_slice(&0i32.to_le_bytes());
        idx.extend_from_slice(&1i32.to_le_bytes());
        idx.extend_from_slice(&doff.to_le_bytes());
        idx.extend_from_slice(&dsz.to_le_bytes());
        idx.extend_from_slice(&[0u8; 20]);
        idx.extend_from_slice(&(enc_blob.len() as i32).to_le_bytes());
        idx.extend_from_slice(&enc_blob);
        idx.extend_from_slice(&0i32.to_le_bytes());
        let isz = idx.len() as u64;
        pak.extend_from_slice(&idx);
        let mut ft = vec![0u8; 0xcc + 21];
        let k = 21usize;
        ft[k..k + 4].copy_from_slice(&0x5A6F_12E1u32.to_le_bytes());
        ft[k + 4..k + 8].copy_from_slice(&11u32.to_le_bytes());
        ft[k + 8..k + 16].copy_from_slice(&ioff.to_le_bytes());
        ft[k + 16..k + 24].copy_from_slice(&isz.to_le_bytes());
        ft[k + 44..k + 49].copy_from_slice(b"Oodle");
        pak.extend_from_slice(&ft);
        pak
    }

    #[test]
    fn inline_shader_entry_filter_skips_non_shader_assets() {
        assert!(may_hold_inline_shaders("Game/Content/M.uexp", false));
        assert!(may_hold_inline_shaders("Game/Content/M.UEXP", false));
        assert!(may_hold_inline_shaders(
            "Engine/GlobalShaderCache-SF_SWITCH_2.bin",
            false
        ));
        assert!(!may_hold_inline_shaders(
            "Engine/Content/Internationalization/icudt64l/brkitr/laodict.dict",
            false
        ));
        assert!(!may_hold_inline_shaders("Game/Content/M.uasset", false));
        assert!(may_hold_inline_shaders("Game/Content/M.uasset", true));
        assert!(may_hold_inline_shaders("Game/Content/Map.umap", true));
    }

    #[test]
    fn oodle_non_shader_entries_do_not_trigger_oodle_decode() {
        set_local_decoder(Some(None));
        let pak = build_v11_pak(&[
            (
                "Engine/Content/Internationalization/icudt64l/brkitr/",
                "laodict.dict",
                true,
                b"oodle-dict-bytes",
            ),
            ("Game/Content/", "M.uexp", false, b"no-shaders-here"),
            ("Game/Content/", "M.uasset", true, b"oodle-uasset"),
        ]);
        let files = vec![File {
            rel: "PJ045/Content/Paks/PJ045-Sage.pak".into(),
            src: own(pak),
            live: true,
        }];
        let mut p = Plan::default();
        plan(&files, &mut p).expect("plan");
        assert!(!p.jobs.is_empty());
        let dicts = Dicts::new();
        let ctx = Ctx {
            files: &files,
            dicts: &dicts,
        };
        for j in &p.jobs {
            let out = (j.run)(&ctx, &mut Vec::new()).expect("non-shader Oodle entries skipped");
            assert!(out.found.is_empty());
        }
        set_local_decoder(None);
    }

    #[test]
    fn oodle_shader_cache_still_requires_decoder() {
        set_local_decoder(Some(None));
        let pak = build_v11_pak(&[
            ("Game/Content/", "M.uexp", false, b"plain"),
            (
                "Engine/",
                "GlobalShaderCache-SF_SWITCH_2.bin",
                true,
                b"compressed-global",
            ),
        ]);
        let files = vec![File {
            rel: "PJ045/Content/Paks/PJ045-Sage.pak".into(),
            src: own(pak),
            live: true,
        }];
        let mut p = Plan::default();
        plan(&files, &mut p).expect("plan");
        let dicts = Dicts::new();
        let ctx = Ctx {
            files: &files,
            dicts: &dicts,
        };
        let errs: Vec<_> = p
            .jobs
            .iter()
            .filter_map(|j| (j.run)(&ctx, &mut Vec::new()).err())
            .collect();
        assert!(
            errs.iter().any(|e| matches!(e, SplitError::Unsupported(_))),
            "{errs:?}"
        );

        struct Copy;
        impl Decoder for Copy {
            fn ask(&self, comp: &[u8], n: usize) -> Answer {
                assert_eq!(comp.len(), n);
                Answer::Bytes(comp.to_vec())
            }
            fn said(&self) -> String {
                String::new()
            }
        }
        set_local_decoder(Some(Some(Arc::new(Copy))));
        for j in &p.jobs {
            (j.run)(&ctx, &mut Vec::new()).expect("decodes with local decoder");
        }
        set_local_decoder(None);
    }
}
