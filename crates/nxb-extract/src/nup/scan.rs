//! Container scanning for base, update, and DLC packages in a `.nup` file.

use super::*;

/// What a container holds: the title ID, the base package (none when an update holds the whole game), the updates by
/// version and the DLC by title ID.
pub struct Scan {
    /// Program ID of the game.
    pub app: u64,
    /// Base package index.
    pub base: Option<usize>,
    /// Updates: version -> package index.
    pub updates: BTreeMap<u32, usize>,
    /// DLC: title ID -> package index.
    pub dlc: BTreeMap<u64, usize>,
}

/// Complete addon trees, keyed by title and version. No delta layering is implied.
/// Reserved `addon_update` records are refused because NUP v1 defines no extraction semantics for them.
pub fn addon_records(n: &Nup) -> Result<BTreeMap<(u64, u32), usize>> {
    let mut records = BTreeMap::new();
    for package in &n.packages {
        if package.kind == 3 {
            return Err(ExtractError::Unsupported(
                "addon_update is reserved by NUP v1 and has no defined extraction semantics".into(),
            ));
        }
        if package.kind != ADDON {
            continue;
        }
        let index = package.title_id & 0xfff;
        if !(1..=2000).contains(&index) || (package.title_id & !0xfff) < 0x1000 {
            return Err(bad(
                &n.path,
                format!("invalid DLC title ID {:016X}", package.title_id),
            ));
        }
        if package.romfs == NO_DIR {
            return Err(bad(
                &n.path,
                format!("DLC {:016X} has no RomFS", package.title_id),
            ));
        }
        if records
            .insert((package.title_id, package.version), package.index)
            .is_some()
        {
            return Err(bad(
                &n.path,
                format!(
                    "DLC {:016X} version {} is duplicated",
                    package.title_id, package.version
                ),
            ));
        }
    }
    Ok(records)
}

fn is_valid_base_id(app: u64) -> bool {
    (app & 0xFFF == 0) || ((app >> 56) == 0x04 && (app & 0xFFF) < 0x800)
}

/// The engine's `scan`: the same checks and the same error texts.
pub fn scan(n: &Nup) -> Result<Scan> {
    scan_for(n, false, None)
}

/// Like [`scan`]. With `newest_dlc` (the generic path), [`addon_records`] reads the DLC records. Then several complete
/// versions of one DLC title can be present, and the scan selects the newest.
pub fn scan_with(n: &Nup, newest_dlc: bool) -> Result<Scan> {
    scan_for(n, newest_dlc, None)
}

/// Like [`scan_with`], optionally selecting one base title ID (`want`) when the container holds multiple base packages
/// (such as a Switch 2 Edition NUP with both a Switch 1 base and a Switch 2 Edition base).
pub fn scan_for(n: &Nup, newest_dlc: bool, want: Option<u64>) -> Result<Scan> {
    let p = &n.path;
    let bases: Vec<&Package> = n.packages.iter().filter(|k| k.kind == BASE).collect();
    let updates: Vec<&Package> = n.packages.iter().filter(|k| k.kind == UPDATE).collect();
    if bases.is_empty() && (updates.is_empty() || updates.iter().any(|k| k.flags & STANDALONE == 0))
    {
        return Err(bad(
            p,
            "no base game: pack the base game and its update into one NUP, or use updates marked standalone_update with complete exefs/ and romfs/",
        ));
    }
    let pkg_pid = |k: &Package| -> Result<Option<u64>> {
        if k.exefs != NO_DIR
            && let Some(i) = n.child(k.exefs, b"main.npdm")
        {
            return Ok(Some(program_id(
                &n.read(i)?,
                &format!("{} exefs/main.npdm", k.name),
            )?));
        }
        Ok(None)
    };
    let multi_base = bases.len() > 1;
    let (app, base, scoped_updates) = if multi_base {
        let mut candidates = Vec::with_capacity(bases.len());
        for b in &bases {
            let pid = pkg_pid(b)?.unwrap_or(b.title_id);
            candidates.push((*b, pid));
        }
        let (selected, app) = if let Some(w) = want {
            let matching: Vec<(&Package, u64)> = candidates
                .iter()
                .copied()
                .filter(|&(_, pid)| pid == w)
                .collect();
            match matching.as_slice() {
                [] => {
                    let have: Vec<String> = candidates
                        .iter()
                        .map(|(_, id)| format!("{id:016X}"))
                        .collect();
                    return Err(bad(
                        p,
                        format!(
                            "no base package with title ID {w:016X} (have: {})",
                            have.join(", ")
                        ),
                    ));
                }
                [one] => *one,
                _ => {
                    return Err(bad(
                        p,
                        format!("multiple base packages with title ID {w:016X}"),
                    ));
                }
            }
        } else {
            let sw2: Vec<(&Package, u64)> = candidates
                .iter()
                .copied()
                .filter(|&(_, pid)| (pid >> 56) == 0x04)
                .collect();
            if sw2.len() == 1 {
                sw2[0]
            } else {
                let have: Vec<String> = candidates
                    .iter()
                    .map(|(_, id)| format!("{id:016X}"))
                    .collect();
                return Err(bad(
                    p,
                    format!(
                        "multiple base packages ({}); pass --title-id to select one",
                        have.join(", ")
                    ),
                ));
            }
        };
        if !is_valid_base_id(app) {
            return Err(bad(
                p,
                format!("base program ID {app:016X} is not a base title ID"),
            ));
        }
        let mut has_npdm = pkg_pid(selected)?.is_some();
        let mut scoped_updates = Vec::new();
        for k in &updates {
            if let Some(upid) = pkg_pid(k)? {
                if upid == app {
                    has_npdm = true;
                    scoped_updates.push(*k);
                } else if k.title_id == app || k.title_id == (app | 0x800) {
                    return Err(bad(
                        p,
                        format!(
                            "base and updates have different program IDs: {app:016X}, {upid:016X}"
                        ),
                    ));
                }
            } else if k.title_id == app || k.title_id == (app | 0x800) {
                scoped_updates.push(*k);
            }
        }
        if !has_npdm {
            return Err(bad(p, "no exefs/main.npdm in the base or updates"));
        }
        let base = (n.tree_bytes(selected.root, 0) > 0).then_some(selected.index);
        (app, base, scoped_updates)
    } else {
        let base = bases
            .first()
            .filter(|b| n.tree_bytes(b.root, 0) > 0)
            .map(|b| b.index);
        let mut ids = std::collections::BTreeSet::new();
        for k in n
            .packages
            .iter()
            .filter(|k| (k.kind == BASE || k.kind == UPDATE) && k.exefs != NO_DIR)
        {
            if let Some(pid) = pkg_pid(k)? {
                ids.insert(pid);
            }
        }
        if ids.is_empty() {
            return Err(bad(p, "no exefs/main.npdm in the base or updates"));
        }
        if ids.len() != 1 {
            let l: Vec<String> = ids.iter().map(|x| format!("{x:016X}")).collect();
            return Err(bad(
                p,
                format!(
                    "base and updates have different program IDs: {}",
                    l.join(", ")
                ),
            ));
        }
        let app = ids.into_iter().next().unwrap_or(0);
        if let Some(w) = want
            && w != app
        {
            return Err(bad(
                p,
                format!("no base package with title ID {w:016X} (have: {app:016X})"),
            ));
        }
        if !is_valid_base_id(app) {
            return Err(bad(
                p,
                format!("base program ID {app:016X} is not a base title ID"),
            ));
        }
        (app, base, updates)
    };
    let is_dlc_of = |title: u64| -> bool {
        if app & 0xFFF == 0 {
            let d = title.wrapping_sub(app).wrapping_sub(0x1000);
            (1..=2000).contains(&d) && title >= app.saturating_add(0x1000)
        } else {
            let base_id = app & !0xFFF;
            let idx = title & 0xFFF;
            (1..=2000).contains(&idx)
                && (title & !0xFFF) >= base_id.saturating_add(0x1000)
                && (title & !0xFFF) <= base_id.saturating_add(0xF000)
        }
    };
    let family = |title: u64| {
        if !is_dlc_of(title) {
            return Err(bad(p, format!("{title:016X} is not a DLC of {app:016X}")));
        }
        Ok(())
    };
    let mut dlc = BTreeMap::new();
    if newest_dlc {
        // Ordered title/version keys make the newest complete tree win for each title.
        for ((title, _), index) in addon_records(n)? {
            if multi_base && !is_dlc_of(title) {
                continue;
            }
            family(title)?;
            dlc.insert(title, index);
        }
    }
    let mut vers = BTreeMap::new();
    for k in scoped_updates {
        if vers.contains_key(&k.version) {
            return Err(bad(p, format!("two updates of version {}", k.version)));
        }
        if k.version == 0 {
            return Err(bad(p, "an update of version 0"));
        }
        vers.insert(k.version, k.index);
    }
    if !newest_dlc {
        for k in &n.packages {
            if k.kind == ADDON {
                if multi_base && !is_dlc_of(k.title_id) {
                    continue;
                }
                family(k.title_id)?;
                if k.romfs == NO_DIR {
                    return Err(bad(p, format!("DLC {:016X} has no romfs/", k.title_id)));
                }
                if dlc.insert(k.title_id, k.index).is_some() {
                    return Err(bad(p, format!("DLC {:016X} more than once", k.title_id)));
                }
            }
        }
    }
    Ok(Scan {
        app,
        base,
        updates: vers,
        dlc,
    })
}
