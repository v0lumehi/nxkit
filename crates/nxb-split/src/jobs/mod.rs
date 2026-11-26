//! The second kind of split job: archives that are read by entry ranges straight from the image (`*_jobs` and
//! `*_shaders` of the reference). A plan lists the jobs in the reference's order, each job scans one range of entries
//! and gives the shaders it found, the archives it takes are left out of the plain file scan.

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::collections::HashSet;
use std::sync::Arc;

use crate::SplitError;
use crate::bahs::BahsVar;
use crate::shaders::Found;
use crate::src::Src;
use crate::zstd::Dicts;

pub mod arh;
pub mod bdt;
pub mod casc;
pub mod cpk;
pub mod cra;
pub mod frostbite;
pub mod kpka;
pub mod linkdata;
pub mod nlg;
pub mod pkz;
pub mod smash;
pub mod trinity;
pub mod ue;
pub mod uejob;
mod uelib;
mod uepak;
mod uerec;
pub use uerec::ue_shaders;
mod uetoc;
pub mod zip;

type R<T> = Result<T, SplitError>;

/// A file of the RomFS images (all versions) as the planners see it.
pub struct File {
    /// Path with the version directory in front of later versions (`rel` of the reference, no leading slash).
    pub rel: String,
    /// Its bytes by range.
    pub src: Arc<dyn Src>,
    /// Whether the hook kept it back for the planners (its first bytes name an archive family).
    pub live: bool,
}

impl File {
    fn head(&self, n: usize) -> R<Vec<u8>> {
        self.src.as_ref().read_up(0, n)
    }
}

/// The NROs of the game's `DM_NRO_DIR`, sorted by name: the CRA archives take their file table from them.
#[derive(Default)]
pub struct NroSet {
    /// The folder as the reference names it in its message (`None` when the variable is unset).
    pub dir: String,
    /// The bytes of each `*.nro` file.
    pub files: Vec<Arc<Vec<u8>>>,
}

/// What a job reads from.
pub struct Ctx<'a> {
    /// Every file.
    pub files: &'a [File],
    /// The zstd dictionaries.
    pub dicts: &'a Dicts,
}

/// What one job found.
pub struct JobOut {
    /// The shaders, in the reference's order.
    pub found: Vec<Found>,
    /// BAHS variation groups of the found places.
    pub var: BahsVar,
}

/// A job's work: lines for `[RED]` notes go to the vector.
pub type Run = Box<dyn Fn(&Ctx<'_>, &mut Vec<String>) -> R<JobOut> + Send + Sync>;

/// One scan job.
pub struct Job {
    /// The `rel` the reference names the job by.
    pub rel: String,
    /// The work.
    pub run: Run,
}

/// The planned jobs and the files they take.
#[derive(Default)]
pub struct Plan {
    /// In the reference's order.
    pub jobs: Vec<Job>,
    /// Indices of the files the jobs read (no plain scan for them).
    pub taken: HashSet<usize>,
    /// Lines of the plan that go to the log.
    pub notes: Vec<String>,
}

/// Splits `items` as the planners do: a job ends when its stored size reaches `total / parts`.
pub(crate) fn ranges<T: Clone>(items: &[T], size_of: impl Fn(&T) -> u64, step: f64) -> Vec<Vec<T>> {
    let (mut out, mut cur, mut size) = (Vec::new(), Vec::new(), 0u64);
    for e in items {
        cur.push(e.clone());
        size += size_of(e);
        if size as f64 >= step {
            out.push(std::mem::take(&mut cur));
            size = 0;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Whether the hook keeps a file back for the planners (they need every file's range reader, siblings and all).
pub fn deferred(rel: &str, head: &[u8]) -> bool {
    let ext = |e: &str| rel.ends_with(e);
    let name = rel.rsplit('/').next().unwrap_or(rel);
    ext(".data")
        || ext(".idx")
        || (name.starts_with("data.")
            && name.len() > 5
            && name[5..].bytes().all(|b| b.is_ascii_digit()))
        || ext(".patch")
        || ext(".debug")
        || ext(".nxpc")
        || head.starts_with(&[0x58, 0x24, 0xf3, 0xa9])
        || name == "DATA1.bin"
        || name == "DATA0.bin"
        || (name.starts_with("LI_") && name.ends_with(".IDX"))
        || (name.starts_with("LD_") && name.ends_with(".BIN"))
        || (ext(".pak") && !head.is_empty())
        || ext(".utoc")
        || ext(".ucas")
        || ext(".trpfs")
        || ext(".trpfd")
        || ext(".bdt")
        || ext(".bhd5")
        || (ext(".arc") && head.starts_with(&smash::MAGIC))
        || (ext(".pkz") && head.starts_with(b"pkzl"))
        || (rel.to_lowercase().ends_with(".cpk") && head.starts_with(b"CPK "))
        || head.starts_with(b"CRA.")
        || head.starts_with(b"PK\x03\x04")
        || ext(".arh")
        || ext(".ard")
        || ext(".toc")
        || ext(".cas")
}

/// The plan over every file, in the reference's order of families (`big` of `main()`).
pub fn plan(files: &[File]) -> R<Plan> {
    plan_with(files, &NroSet::default())
}

/// [`plan`] with the NROs of `DM_NRO_DIR`.
pub fn plan_with(files: &[File], nros: &NroSet) -> R<Plan> {
    let mut p = Plan::default();
    linkdata::plan(files, &mut p)?;
    kpka::plan(files, &mut p)?;
    trinity::plan(files, &mut p)?;
    ue::plan(files, &mut p)?;
    bdt::plan(files, &mut p)?;
    smash::plan(files, &mut p)?;
    pkz::plan(files, &mut p)?;
    cpk::plan(files, &mut p)?;
    cra::plan(files, nros, &mut p)?;
    zip::plan(files, &mut p)?;
    arh::plan(files, &mut p)?;
    frostbite::plan(files, &mut p)?;
    casc::plan(files, &mut p)?;
    Ok(p)
}

/// Bytes owned in memory, for the planners' tests.
#[cfg(test)]
pub(crate) struct Owned(pub Vec<u8>);

#[cfg(test)]
impl Src for Owned {
    fn bytes(&self) -> Option<&[u8]> {
        Some(&self.0)
    }
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, off: u64, buf: &mut [u8]) -> R<()> {
        crate::src::Mem(&self.0).read_at(off, buf)
    }
}

/// A test file's source.
#[cfg(test)]
pub(crate) fn own(b: Vec<u8>) -> Arc<dyn Src> {
    Arc::new(Owned(b))
}
