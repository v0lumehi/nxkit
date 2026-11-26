//! Discovery and parallel shader compilation directly to an output directory.

use super::*;

fn discover_outputs(
    modules: &[(String, Vec<u8>)],
    files: &[jobs::File],
    threads: usize,
) -> Result<(records::Outputs, Vec<String>)> {
    let profile = shaders::Profile::default();
    let (mut results, common) = scan_modules(modules, &profile)?;
    let mut notes = Vec::new();
    let mut dicts = zstd::Dicts::new();
    for file in files {
        let head = file.src.read_up(0, 4).map_err(|e| e.to_string())?;
        if head == [0x37, 0xa4, 0x30, 0xec] {
            let raw = file.src.whole().map_err(|e| e.to_string())?;
            dicts.add(&raw).map_err(|e| e.to_string())?;
        } else if file.rel == "Pack/ZsDic.pack.zs" {
            let raw = file.src.whole().map_err(|e| e.to_string())?;
            let pack = zstd::decompress(&raw).map_err(|e| e.to_string())?;
            for (_, data) in shaders::sarc_members(&pack, &file.rel).map_err(|e| e.to_string())? {
                if data.starts_with(&[0x37, 0xa4, 0x30, 0xec]) {
                    dicts.add(data).map_err(|e| e.to_string())?;
                }
            }
        }
    }
    let plan = jobs::plan(files).map_err(|e| e.to_string())?;
    notes.extend(plan.notes);
    let ctx = jobs::Ctx {
        files,
        dicts: &dicts,
    };
    let job_results = run_workers(threads, plan.jobs.len(), |k| {
        let job = &plan.jobs[k];
        let mut job_notes = Vec::new();
        let out = (job.run)(&ctx, &mut job_notes).map_err(|e| format!("{}: {e}", job.rel))?;
        let res = scanfile::finish(&profile, out.found, &out.var).map_err(|e| e.to_string())?;
        Ok((res, job_notes))
    })?;
    for (res, job_notes) in job_results {
        notes.extend(job_notes);
        if let Some(result) = res {
            results.push(result);
        }
    }
    let env = scanfile::Env {
        dicts: &dicts,
        common: common.as_ref(),
        trees: None,
    };
    let untaken: Vec<usize> = (0..files.len())
        .filter(|i| !plan.taken.contains(i))
        .collect();
    let pass1 = run_workers(threads, untaken.len(), |idx| {
        let i = untaken[idx];
        let file = &files[i];
        let label = format!("/{}", file.rel);
        let mut file_notes = Vec::new();
        let result = if file.live {
            let head = file.src.read_up(0, 4).map_err(|e| e.to_string())?;
            if head == jobs::nlg::DICT {
                let (found, var) =
                    jobs::nlg::shaders(files, i, &label).map_err(|e| e.to_string())?;
                scanfile::finish(&profile, found, &var)
            } else if jobs::nlg::covered(&file.rel, files) {
                Ok(None)
            } else {
                scanfile::scan_file(
                    &profile,
                    file.src.as_ref(),
                    &label,
                    &file.rel,
                    &env,
                    &mut file_notes,
                )
            }
        } else {
            scanfile::scan_file(
                &profile,
                file.src.as_ref(),
                &label,
                &file.rel,
                &env,
                &mut file_notes,
            )
        }
        .map_err(|e| format!("{}: {e}", file.rel))?;
        Ok((i, result, file_notes))
    })?;
    let mut deferred = Vec::new();
    let mut plain = Vec::new();
    let mut trees = unity::ser::Trees::new();
    for (i, result, file_notes) in pass1 {
        notes.extend(file_notes);
        if let Some(result) = result {
            trees.extend(result.trees.clone());
            if result.defer {
                deferred.push(i);
            } else {
                plain.push((i, result));
            }
        }
    }
    if !deferred.is_empty() {
        for (key, value) in unity::ser::known_trees().map_err(|e| e.to_string())? {
            trees.entry(key).or_insert(value);
        }
        let env = scanfile::Env {
            trees: Some(&trees),
            ..env
        };
        let pass2 = run_workers(threads, deferred.len(), |idx| {
            let i = deferred[idx];
            let file = &files[i];
            let mut file_notes = Vec::new();
            let result = scanfile::scan_file(
                &profile,
                file.src.as_ref(),
                &format!("/{}", file.rel),
                &file.rel,
                &env,
                &mut file_notes,
            )
            .map_err(|e| format!("{}: {e}", file.rel))?;
            if let Some(ref r) = result
                && r.defer
            {
                return Err(format!("{}: missing Unity shader type trees", file.rel));
            }
            Ok((i, result, file_notes))
        })?;
        for (i, result, file_notes) in pass2 {
            notes.extend(file_notes);
            if let Some(result) = result {
                plain.push((i, result));
            }
        }
    }
    // Deferred Unity scans keep their original file order at the merge.
    plain.sort_by_key(|(i, _)| *i);
    results.extend(plain.into_iter().map(|(_, result)| result));
    let output = records::build(&profile, &results).map_err(|e| e.to_string())?;
    Ok((output, notes))
}

#[cfg(test)]
pub(in crate::standalone) fn discover(
    modules: &[(String, Vec<u8>)],
    resources: &[(String, Vec<u8>)],
    ios: bool,
) -> Result<Vec<(String, Vec<u8>)>> {
    let mut total = 0usize;
    let mut files = Vec::new();
    if resources.len() > 20000 {
        return Err("too many shader source files".into());
    }
    for (path, data) in resources {
        let rel = path.strip_prefix("romfs/").unwrap_or(path);
        total = total
            .checked_add(data.len())
            .ok_or("shader source size overflow")?;
        if total > INPUT_LIMIT {
            return Err("shader source files exceed 256 MiB".into());
        }
        files.push(jobs::File {
            rel: rel.into(),
            live: jobs::deferred(rel, &data[..data.len().min(16)]),
            src: Arc::new(Owned(data.clone())),
        });
    }
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    let (output, notes) = discover_outputs(modules, &files, 1)?;
    compile_packages(output, notes, ios)
}

/// Discovers shaders from `modules` and disk `romfs_dir` (+ optional `overlay`) without size or
/// count limits, compiles them across `threads` workers, and writes `shaders/*` directly under
/// `out_dir`. Returns the number of `.metallib` files written.
pub(in crate::standalone) fn discover_dir(
    modules: &[(String, Vec<u8>)],
    romfs_dir: &Path,
    overlay: Option<&Path>,
    ios: bool,
    threads: usize,
    out_dir: &Path,
) -> Result<usize> {
    let files = walk_dir_files(romfs_dir, overlay)?;
    let (output, mut notes) = discover_outputs(modules, &files, threads)?;
    let prepared = prepare_jobs(output, &mut notes)?;
    if let Some(note) = notes.iter().find(|note| note.starts_with("[RED]")) {
        return Err(format!("unsupported: incomplete shader discovery: {note}"));
    }
    let shaders_dir = out_dir.join("shaders");
    std::fs::create_dir_all(&shaders_dir).map_err(|e| format!("{}: {e}", shaders_dir.display()))?;
    for (rel, bytes) in prepared.tables {
        let path = out_dir.join(&rel);
        std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    if !notes.is_empty() {
        let warn = shaders_dir.join("warnings.txt");
        std::fs::write(&warn, notes.join("\n").as_bytes())
            .map_err(|e| format!("{}: {e}", warn.display()))?;
    }
    let mut ps_jobs = Vec::new();
    let mut other_jobs = Vec::new();
    for entry in prepared.jobs {
        if entry.job.stage == "ps" {
            ps_jobs.push(entry);
        } else {
            other_jobs.push(entry);
        }
    }
    ps_jobs.sort_by(|a, b| a.job.key.cmp(&b.job.key));
    other_jobs.sort_by(|a, b| a.job.key.cmp(&b.job.key));
    let chain = nxb_shader::chain::standalone_version();
    let empty_inputs = BTreeMap::new();
    let libs_written = AtomicUsize::new(0);
    let write_outputs = |files: Vec<NamedFile>| -> Result<()> {
        for (rel, bytes) in files {
            if rel.ends_with(".metallib") {
                libs_written.fetch_add(1, Ordering::Relaxed);
            }
            let path = out_dir.join(&rel);
            std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(())
    };
    let ps_results = run_workers(threads, ps_jobs.len(), |i| {
        let entry = &ps_jobs[i];
        let (stage_inputs, out_files) =
            compile_work(entry, &chain, &prepared.pairs, &empty_inputs, ios)?;
        write_outputs(out_files)?;
        Ok((entry.job.key.clone(), stage_inputs))
    })?;
    let inputs: BTreeMap<String, StageInputs> = ps_results.into_iter().collect();
    run_workers(threads, other_jobs.len(), |i| {
        let entry = &other_jobs[i];
        let (_, out_files) = compile_work(entry, &chain, &prepared.pairs, &inputs, ios)?;
        write_outputs(out_files)?;
        Ok(())
    })?;
    Ok(libs_written.load(Ordering::Relaxed))
}
