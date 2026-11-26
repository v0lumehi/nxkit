//! Loose `romfs/` folder support for [`crate::split::split`] (used when a NUP's deferred split runs later).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nxb_model::{Digest, sha256};
use nxb_split::SplitError;
use nxb_split::records::Outputs;
use nxb_split::src::Src;

use crate::error::{ExtractError, Result};
use crate::fileext::FileExt;
use crate::post::{PostFile, stream_digest};
use crate::split::{SplitRequest, Templates, VersionIn};

struct LooseSrc {
    path: PathBuf,
    size: u64,
}

impl Src for LooseSrc {
    fn len(&self) -> u64 {
        self.size
    }
    fn read_at(&self, off: u64, buf: &mut [u8]) -> std::result::Result<(), SplitError> {
        if off.saturating_add(buf.len() as u64) > self.size {
            return Err(SplitError::Red("read past the end of the file".into()));
        }
        if buf.is_empty() {
            return Ok(());
        }
        File::open(&self.path)
            .and_then(|f| f.read_exact_at(buf, off))
            .map_err(|e| SplitError::Format(format!("read {}: {e}", self.path.display())))
    }
}

fn walk_romfs(root: &Path) -> Result<Vec<(String, PathBuf, u64)>> {
    let mut out = Vec::new();
    let mut pending = vec![(root.to_path_buf(), String::new())];
    while let Some((dir, rel)) = pending.pop() {
        for e in std::fs::read_dir(&dir)
            .map_err(|e| ExtractError::io(format!("read {}", dir.display()), e))?
        {
            let e = e.map_err(|e| ExtractError::io(format!("read {}", dir.display()), e))?;
            let name = e
                .file_name()
                .into_string()
                .map_err(|_| ExtractError::format("non-UTF-8 RomFS filename"))?;
            let next = if rel.is_empty() {
                name
            } else {
                format!("{rel}/{name}")
            };
            let ft = e
                .file_type()
                .map_err(|er| ExtractError::io(format!("stat {}", e.path().display()), er))?;
            if ft.is_dir() {
                pending.push((e.path(), next));
            } else if ft.is_file() {
                let sz = e
                    .metadata()
                    .map_err(|er| ExtractError::io(format!("stat {}", e.path().display()), er))?
                    .len();
                out.push((next, e.path(), sz));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Runs [`crate::split::split`] over version folders that hold a loose `romfs/` directory (NUP unpacks).
pub(crate) fn split_loose(req: &SplitRequest) -> Result<(Outputs, Vec<String>)> {
    let mut t = Templates::new(&req.game_tsv)?;
    let mut versions = Vec::new();
    let mut jobs: Vec<(String, PathBuf, u64)> = Vec::new();
    for (i, v) in req.versions.iter().enumerate() {
        let tag = if i == 0 {
            String::new()
        } else {
            v.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let mainp = v.join("ExeFS/main");
        let main = std::fs::read(&mainp)
            .map_err(|e| ExtractError::io(format!("read {}", mainp.display()), e))?;
        if i == 0 {
            t.set_common(&main);
        }
        let files = walk_romfs(&v.join("romfs"))?;
        for (p, path, size) in &files {
            if p == "Pack/ZsDic.pack.zs" {
                let pack = std::fs::read(path)
                    .map_err(|e| ExtractError::io("read Pack/ZsDic.pack.zs", e))?;
                t.add_dict_pack(&pack)?;
            }
            let label = if tag.is_empty() {
                format!("/{p}")
            } else {
                format!("/{tag}/{p}")
            };
            jobs.push((label, path.clone(), *size));
        }
        versions.push((tag, main, files));
    }
    let digests: Vec<Mutex<Option<Digest>>> = jobs.iter().map(|_| Mutex::new(None)).collect();
    if let Some(c) = req.progress.as_deref() {
        c.total
            .store(jobs.iter().map(|j| j.2).sum(), Ordering::Relaxed);
    }
    let next = AtomicUsize::new(0);
    let fatal: Mutex<Option<ExtractError>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..req.threads.max(1) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some((label, path, size)) = jobs.get(k) else {
                        break;
                    };
                    let mut f = match File::open(path) {
                        Ok(f) => f,
                        Err(e) => {
                            if let Ok(mut g) = fatal.lock() {
                                *g = Some(ExtractError::io(format!("open {label}"), e));
                            }
                            break;
                        }
                    };
                    let mut head = vec![0u8; (*size as usize).min(16)];
                    if let Err(e) = f.read_exact_at(&mut head, 0) {
                        if let Ok(mut g) = fatal.lock() {
                            *g = Some(ExtractError::io(format!("read {label}"), e));
                        }
                        break;
                    }
                    let d = if nxb_split::jobs::deferred(label.trim_start_matches('/'), &head) {
                        match stream_digest(&f, 0, *size) {
                            Ok(d) => {
                                t.defer(label, &head, d);
                                d
                            }
                            Err(e) => {
                                if let Ok(mut g) = fatal.lock() {
                                    *g = Some(e);
                                }
                                break;
                            }
                        }
                    } else {
                        use std::io::Read;
                        let mut data = Vec::with_capacity(*size as usize);
                        if let Err(e) = f.read_to_end(&mut data) {
                            if let Ok(mut g) = fatal.lock() {
                                *g = Some(ExtractError::io(format!("read {label}"), e));
                            }
                            break;
                        }
                        let d = sha256(&data);
                        t.scan(label, &data, d);
                        d
                    };
                    if let Ok(mut g) = digests[k].lock() {
                        *g = Some(d);
                    }
                    if let Some(c) = req.progress.as_deref() {
                        c.done.fetch_add(*size, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    if let Some(e) = fatal.into_inner().ok().flatten() {
        return Err(e);
    }
    let mut ds = digests.into_iter().map(|m| m.into_inner().ok().flatten());
    let mut vin = Vec::new();
    let mut post = Vec::new();
    for (tag, main, files) in versions {
        let mut fl = Vec::new();
        for (p, path, size) in files {
            let d = ds
                .next()
                .flatten()
                .ok_or_else(|| ExtractError::format("a split job did not finish"))?;
            let label = if tag.is_empty() {
                format!("/{p}")
            } else {
                format!("/{tag}/{p}")
            };
            post.push(PostFile {
                label,
                src: Arc::new(LooseSrc { path, size }),
                digest: d,
            });
            fl.push((p, d));
        }
        vin.push(VersionIn {
            tag,
            main,
            files: fl,
        });
    }
    t.post(post, req.threads)?;
    t.merge(&vin, &req.outputs)
}
