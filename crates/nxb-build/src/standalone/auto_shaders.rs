//! In-memory orchestration of nxkit's unmodified split, job and compiler APIs.
use nxb_pyjson::{Value as J, dumps, parse};
type Result<T> = std::result::Result<T, String>;
const INPUT_LIMIT: usize = 256 * 1024 * 1024;
fn file(name: impl Into<String>, data: &[u8]) -> (String, Vec<u8>) {
    (name.into(), data.to_vec())
}
fn safe_name(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 128
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err("invalid module or shader name".into());
    }
    Ok(())
}
pub(super) fn check_nso_size(nso: &[u8]) -> Result<()> {
    if !nso.starts_with(b"NSO0") {
        return Err("missing NSO0 header".into());
    }
    for i in 0..3 {
        let at = 0x14 + i * 16;
        let words = nso.get(at..at + 8).ok_or("truncated NSO")?;
        let offset =
            u32::from_le_bytes(words[..4].try_into().map_err(|_| "invalid NSO offset")?) as u64;
        let size =
            u32::from_le_bytes(words[4..].try_into().map_err(|_| "invalid NSO size")?) as u64;
        if offset + size > INPUT_LIMIT as u64 {
            return Err("NSO memory image exceeds 256 MiB".into());
        }
    }
    Ok(())
}
use nxb_split::{jobs, records, scanfile, shaders, src, unity, zstd};
use nxb_xlate::msl::{self, StageInputs};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(test)]
const SHADER_LIMIT: usize = 20000;

#[cfg(test)]
struct Owned(Vec<u8>);
#[cfg(test)]
impl src::Src for Owned {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn bytes(&self) -> Option<&[u8]> {
        Some(&self.0)
    }
    fn read_at(
        &self,
        offset: u64,
        data: &mut [u8],
    ) -> std::result::Result<(), nxb_split::SplitError> {
        src::Mem(&self.0).read_at(offset, data)
    }
}

/// Disk-backed `Src` that caches the 16-byte header inline and serves larger range reads via
/// thread-safe positional I/O (`FileExt::read_exact_at`), returning `bytes() -> None` so large
/// RomFS assets stream without loading the whole game into RAM.
struct FileSrc {
    path: PathBuf,
    size: u64,
    head: [u8; 16],
}

impl FileSrc {
    fn open(path: PathBuf, size: u64) -> Result<Self> {
        use std::io::Read;
        let mut head = [0u8; 16];
        let n = (size as usize).min(16);
        if n > 0 {
            let mut f =
                std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            f.read_exact(&mut head[..n])
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(Self { path, size, head })
    }

    fn head_slice(&self) -> &[u8] {
        &self.head[..(self.size as usize).min(16)]
    }
}

impl src::Src for FileSrc {
    fn len(&self) -> u64 {
        self.size
    }

    fn read_at(
        &self,
        offset: u64,
        data: &mut [u8],
    ) -> std::result::Result<(), nxb_split::SplitError> {
        use nxb_extract::fileext::FileExt;
        let Some(end) = offset.checked_add(data.len() as u64) else {
            return Err(nxb_split::SplitError::Red(
                "read past the end of the file".into(),
            ));
        };
        if end > self.size {
            return Err(nxb_split::SplitError::Red(
                "read past the end of the file".into(),
            ));
        }
        if data.is_empty() {
            return Ok(());
        }
        if end <= 16 {
            data.copy_from_slice(&self.head[offset as usize..end as usize]);
            return Ok(());
        }
        std::fs::File::open(&self.path)
            .and_then(|f| f.read_exact_at(data, offset))
            .map_err(|e| nxb_split::SplitError::Format(format!("{}: {e}", self.path.display())))
    }
}

struct Work {
    job: nxb_xlate::job::Job,
    code: Vec<u8>,
    source: String,
    no_facts: bool,
}
fn work(value: &J, code: Vec<u8>, source: String) -> Result<Work> {
    let job = nxb_xlate::job::parse_jobs(&dumps(&J::List(vec![value.clone()])))?
        .into_iter()
        .next()
        .ok_or("missing shader job")?;
    safe_name(&job.key)?;
    Ok(Work {
        job,
        code,
        source,
        no_facts: value.get("no_facts") == Some(&J::Bool(true)),
    })
}

fn walk_dir_files(romfs_dir: &Path, overlay: Option<&Path>) -> Result<Vec<jobs::File>> {
    let mut entries_map = BTreeMap::<String, (PathBuf, u64)>::new();
    let mut scan = |root: &Path, skip_stamp: bool| -> Result<()> {
        if !root.is_dir() {
            return Ok(());
        }
        let mut pending = vec![(root.to_path_buf(), String::new())];
        while let Some((dir, rel)) = pending.pop() {
            for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
                let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "RomFS filename is not UTF-8")?;
                let next = if rel.is_empty() {
                    name
                } else {
                    format!("{rel}/{name}")
                };
                if skip_stamp && next == crate::resource_patch::STAMP {
                    continue;
                }
                let kind = entry.file_type().map_err(|e| e.to_string())?;
                if kind.is_dir() {
                    pending.push((entry.path(), next));
                } else if kind.is_file() {
                    let size = entry
                        .metadata()
                        .map_err(|e| format!("{}: {e}", entry.path().display()))?
                        .len();
                    entries_map.insert(next, (entry.path(), size));
                }
            }
        }
        Ok(())
    };
    scan(romfs_dir, false)?;
    if let Some(o) = overlay {
        scan(o, true)?;
    }
    let mut files = Vec::with_capacity(entries_map.len());
    for (rel, (path, size)) in entries_map {
        let file_src = FileSrc::open(path, size)?;
        let live = jobs::deferred(&rel, file_src.head_slice());
        files.push(jobs::File {
            rel,
            live,
            src: Arc::new(file_src),
        });
    }
    Ok(files)
}

type NamedFile = (String, Vec<u8>);
type ModuleScan = (Vec<records::FileResult>, Option<HashMap<u32, String>>);

fn scan_modules(modules: &[(String, Vec<u8>)], profile: &shaders::Profile) -> Result<ModuleScan> {
    let mut results = Vec::new();
    let mut common = None;
    for (name, nso) in modules {
        safe_name(name)?;
        check_nso_size(nso)?;
        let image = nxb_split::nso::image(nso).map_err(|e| format!("{name}: {e}"))?;
        if name == "main" {
            common = unity::bundle::common_strings(&image).ok();
        }
        let mut found = Vec::new();
        let mut variations = shaders::BahsVar::new();
        shaders::scan_blob(&image, name, &mut found, &mut variations, 0)
            .map_err(|e| e.to_string())?;
        found
            .extend(nxb_split::misc::main_extras(&image, name, &found).map_err(|e| e.to_string())?);
        let pairs = shaders::var_pairs(profile, &found, &variations);
        results.push(records::FileResult {
            keys: records::uniq(profile, found, true).map_err(|e| e.to_string())?,
            pairs,
            ..Default::default()
        });
    }
    Ok((results, common))
}

fn run_workers<T: Send>(
    threads: usize,
    count: usize,
    f: impl Fn(usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let workers = threads.max(1).min(count);
    if workers == 1 {
        return (0..count).map(f).collect();
    }
    let slots: Vec<Mutex<Option<Result<T>>>> = (0..count).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    let r = f(i);
                    if let Ok(mut g) = slots[i].lock() {
                        *g = Some(r);
                    }
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|s| {
            s.into_inner()
                .ok()
                .flatten()
                .unwrap_or_else(|| Err("worker failed to produce result".into()))
        })
        .collect()
}

mod discover;
#[cfg(test)]
pub(super) use discover::discover;
pub(super) use discover::discover_dir;

struct PreparedJobs {
    jobs: Vec<Work>,
    pairs: BTreeSet<(String, String)>,
    tables: Vec<NamedFile>,
}

fn prepare_jobs(output: records::Outputs, notes: &mut Vec<String>) -> Result<PreparedJobs> {
    let mut jobs = Vec::new();
    let mut offsets = BTreeMap::new();
    for (package, index, _, key, _) in &output.rows {
        let start = *offsets.entry(package).or_insert(0usize);
        let raw = output
            .packages
            .get(package)
            .and_then(|p| p.get(start..))
            .ok_or("missing shader package")?;
        let rd = |at: usize| -> Result<usize> {
            let bytes = raw.get(at..at + 4).ok_or("truncated shader package")?;
            Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| "invalid shader size")?) as usize)
        };
        let first_size = rd(0x28)?;
        let second_size = rd(0x2c)?;
        let end = 0x38usize
            .checked_add(first_size)
            .and_then(|n| n.checked_add(second_size))
            .ok_or("shader size overflow")?;
        let code = raw.get(..end).ok_or("truncated shader code")?;
        let fact = output
            .facts
            .get(key)
            .map(|s| parse(s).map_err(|e| format!("{e:?}")))
            .transpose()?;
        let job =
            nxb_shader::job::job_for(key, package, code, fact.as_ref(), "", Default::default())
                .map_err(|e| e.to_string())?;
        notes.extend(job.notes);
        jobs.push(work(&job.json, job.code, format!("{package}#{index}"))?);
        offsets.insert(package, start + end);
    }
    let mut seen = BTreeSet::new();
    for entry in &jobs {
        if !seen.insert(&entry.job.key) {
            return Err(format!("duplicate shader key {}", entry.job.key));
        }
    }
    let pairs_text = |pairs: &BTreeSet<(String, String)>| {
        pairs
            .iter()
            .map(|(a, b)| format!("{a}\t{b}\n"))
            .collect::<String>()
    };
    let map = output
        .rows
        .iter()
        .map(|(p, i, s, k, src)| format!("{p}\t{i}\t{s}\t{k}\t{src}\n"))
        .collect::<String>();
    let facts = output
        .facts
        .iter()
        .map(|(k, v)| format!("{k}\t{v}\n"))
        .collect::<String>();
    let tables = vec![
        file("shaders/shader_map.tsv", map.as_bytes()),
        file("shaders/shader_facts.tsv", facts.as_bytes()),
        file(
            "shaders/shader_pairs.tsv",
            pairs_text(&output.pairs).as_bytes(),
        ),
        file(
            "shaders/pso_pairs.tsv",
            pairs_text(&output.pso_pairs).as_bytes(),
        ),
    ];
    let pairs = output.pairs.union(&output.pso_pairs).cloned().collect();
    Ok(PreparedJobs {
        jobs,
        pairs,
        tables,
    })
}

fn has_unsupported(s: &str) -> bool {
    s.starts_with("unsupported") || s.contains("unsupported:")
}

fn format_translate_err(key: &str, reason: &str, detail: &str) -> String {
    if has_unsupported(reason) || has_unsupported(detail) {
        format!("unsupported: shader {key}: {reason}: {detail}")
    } else {
        format!("shader {key}: {reason}: {detail}")
    }
}

fn format_compile_err(key: &str, suffix: &str, err: &str) -> String {
    if has_unsupported(err) {
        format!("unsupported: shader {key}{suffix} compilation: {err}")
    } else {
        format!("shader {key}{suffix} compilation: {err:?}")
    }
}

fn compile_work(
    entry: &Work,
    chain: &str,
    pairs: &BTreeSet<(String, String)>,
    inputs: &BTreeMap<String, StageInputs>,
    ios: bool,
) -> Result<(StageInputs, Vec<NamedFile>)> {
    let key = &entry.job.key;
    let partners = pairs
        .iter()
        .filter(|(v, _)| v == key)
        .filter_map(|(_, p)| inputs.get(p).cloned())
        .collect::<Vec<_>>();
    let ctx = msl::Context {
        chain,
        source: entry.source.clone(),
        partners: &partners,
        no_facts: entry.no_facts,
        ..Default::default()
    };
    let translated = msl::translate(&entry.job, &entry.code, &ctx)
        .map_err(|e| format_translate_err(key, &e.reason, &e.detail))?;
    let mut files = Vec::new();
    for lib in translated.libraries {
        let stem = format!("shaders/{key}{}", lib.suffix);
        let mut metallib = compile_raw(lib.msl.as_bytes(), ios)
            .map_err(|e| format_compile_err(key, &lib.suffix, &e))?;
        if lib.suffix == "_b"
            && let Some(side) = &translated.side
        {
            let names = side
                .alpha
                .iter()
                .chain(&side.logic)
                .cloned()
                .collect::<Vec<_>>();
            if !names.is_empty() {
                let (base, side) =
                    nxb_shader::side::cut(&metallib, &names).map_err(|e| e.to_string())?;
                metallib = base;
                files.push(file(format!("shaders/{key}_bx.metallib"), &side));
            }
        }
        files.push(file(format!("{stem}.metal"), lib.msl.as_bytes()));
        files.push(file(format!("{stem}.metallib"), &metallib));
        if let Some(reflection) = lib.reflection {
            files.push(file(
                format!("{stem}.json"),
                dumps(&reflection.to_value()).as_bytes(),
            ));
        }
    }
    Ok((translated.inputs, files))
}

#[cfg(test)]
fn compile_packages(
    output: records::Outputs,
    mut notes: Vec<String>,
    ios: bool,
) -> Result<Vec<(String, Vec<u8>)>> {
    if output.rows.len() > SHADER_LIMIT {
        return Err("too many discovered shaders".into());
    }
    let prepared = prepare_jobs(output, &mut notes)?;
    compile(prepared.jobs, &prepared.pairs, prepared.tables, notes, ios)
}

#[cfg(test)]
fn compile(
    mut jobs: Vec<Work>,
    pairs: &BTreeSet<(String, String)>,
    mut files: Vec<(String, Vec<u8>)>,
    notes: Vec<String>,
    ios: bool,
) -> Result<Vec<(String, Vec<u8>)>> {
    // The shared scanner marks unreadable or omitted content as RED. Generic builds have no partial mode.
    if let Some(note) = notes.iter().find(|note| note.starts_with("[RED]")) {
        return Err(format!("unsupported: incomplete shader discovery: {note}"));
    }
    let mut seen = BTreeSet::new();
    for entry in &jobs {
        if !seen.insert(&entry.job.key) {
            return Err(format!("duplicate shader key {}", entry.job.key));
        }
    }
    // Fragment inputs must be known before their vertex/mesh partners translate.
    jobs.sort_by(|a, b| (a.job.stage != "ps", &a.job.key).cmp(&(b.job.stage != "ps", &b.job.key)));
    let chain = nxb_shader::chain::standalone_version();
    let mut inputs = BTreeMap::<String, StageInputs>::new();
    for entry in jobs {
        let (stage_inputs, out_files) = compile_work(&entry, &chain, pairs, &inputs, ios)?;
        inputs.insert(entry.job.key, stage_inputs);
        files.extend(out_files);
    }
    if !notes.is_empty() {
        files.push(file("shaders/warnings.txt", notes.join("\n").as_bytes()));
    }
    Ok(files)
}

// Keep compiler validation and panic handling in the shared tool layer.
#[cfg(feature = "air-front")]
fn compile_raw(msl: &[u8], ios: bool) -> Result<Vec<u8>> {
    use nxb_tool::FrontEnd;
    nxb_tool::airfront::RawFrontEnd::new(ios)
        .compile(msl)
        .map(|output| output.metallib)
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "air-front"))]
fn compile_raw(_msl: &[u8], _ios: bool) -> Result<Vec<u8>> {
    Err("unsupported: generic shader compilation requires the air-front feature".into())
}

#[cfg(test)]
mod tests;
