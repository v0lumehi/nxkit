//! Opt-in resource overlays. Stock extraction, shaders and application builds stay unchanged.

use crate::host::{TempDir, rom::Rom};
use nxb_model::{Digest, FileDigest, HashingWriter, TitleId};
use nxtitle::{FixedResolution, TitleFacts};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

const MAX_RESOURCE: u64 = 64 << 20;
const USAGE: &str = "usage: nxb romfs fixed-scale --title-id ID --romfs IMAGE_OR_FOLDER --scale F --zstd-library LIB --out NEW_DIR";

fn digest_matches(bytes: &[u8], expected: Digest, what: &str) -> Result<(), String> {
    let actual = FileDigest::of(bytes).sha256;
    if actual != expected {
        return Err(format!(
            "unsupported {what} digest {actual}, expected {expected}"
        ));
    }
    Ok(())
}

fn relative(path: &str) -> Result<(), String> {
    if path.is_empty()
        || !Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(format!("invalid resource path {path:?}"));
    }
    Ok(())
}

fn resource(rom: &Rom, path: &str) -> Result<Vec<u8>, String> {
    relative(path)?;
    let wanted = format!("/{path}");
    let file = rom
        .files()
        .iter()
        .find(|f| f.path == wanted)
        .ok_or_else(|| format!("missing RomFS resource {path}"))?;
    if file.size > MAX_RESOURCE {
        return Err(format!(
            "unsupported resource size {} for {path}",
            file.size
        ));
    }
    rom.read(file, 0, file.size)
}

fn member_range(data: &[u8], name: &str) -> Result<std::ops::Range<usize>, String> {
    let mut found = None;
    for (member, bytes) in
        nxb_split::shaders::sarc_members(data, "resource archive").map_err(|e| e.to_string())?
    {
        if member != name {
            continue;
        }
        if found.is_some() {
            return Err(format!("duplicate archive member {name}"));
        }
        // The reader returns slices of `data`, so this preserves the original SARC layout.
        let start = bytes.as_ptr() as usize - data.as_ptr() as usize;
        found = Some(start..start + bytes.len());
    }
    found.ok_or_else(|| format!("missing archive member {name}"))
}

fn output_path(source: &Path, out: &Path) -> Result<PathBuf, String> {
    if std::fs::symlink_metadata(out).is_ok() {
        return Err(format!("output already exists: {}", out.display()));
    }
    let name = out
        .file_name()
        .ok_or_else(|| "output must name a new directory".to_string())?;
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    if source.is_dir() {
        let source = std::fs::canonicalize(source).map_err(|e| e.to_string())?;
        if parent.starts_with(&source) {
            return Err("output must be outside the source RomFS folder".into());
        }
    }
    Ok(parent.join(name))
}

/// Prepares a fixed-scale overlay and `env.txt` for the host's `DM_ROMFS_OVERLAY`.
///
/// `out` must be a new directory whose parent exists, outside a loose input folder.
/// Only recorded source assets are accepted. Nothing is published until validation
/// and recompression finish. The original RomFS and all other members stay unchanged.
/// `zstd_library` names the trusted libzstd 1.5.7 native dependency on macOS.
pub fn prepare(
    source: &Path,
    out: &Path,
    zstd_library: &Path,
    patch: FixedResolution,
    scale: f32,
) -> Result<FileDigest, String> {
    if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
        return Err("scale must be finite and greater than 0, up to 1".into());
    }
    relative(patch.archive)?;
    let out = output_path(source, out)?;
    let rom = Rom::open(source)?;
    let compressed = resource(&rom, patch.archive)?;
    digest_matches(&compressed, patch.archive_sha256, "archive")?;
    let dictionary_pack = resource(&rom, patch.dictionary_archive)?;
    let dictionary_sarc =
        nxb_split::zstd::decompress(&dictionary_pack).map_err(|e| e.to_string())?;
    let dictionary = &dictionary_sarc[member_range(&dictionary_sarc, patch.dictionary_member)?];
    digest_matches(dictionary, patch.dictionary_sha256, "dictionary")?;
    let mut dictionaries = nxb_split::zstd::Dicts::new();
    let dictionary_id = dictionaries.add(dictionary).map_err(|e| e.to_string())?;
    if nxb_split::zstd::dict_id(&compressed) != dictionary_id {
        return Err("unsupported archive dictionary ID".into());
    }
    let mut archive =
        nxb_split::zstd::decompress_dict(&compressed, &dictionaries).map_err(|e| e.to_string())?;
    let range = member_range(&archive, patch.member)?;
    nxb_romfs::patch::fixed_scale(
        &mut archive[range],
        patch.member_sha256,
        patch.scale_offsets,
        scale,
    )?;

    let parent = out
        .parent()
        .ok_or_else(|| "output has no parent".to_string())?;
    let scratch = TempDir::new_in(parent)?;
    let encoded = nxb_sys::zstd::compress_using_dict(zstd_library, &archive, dictionary, 19)?;
    let checked =
        nxb_split::zstd::decompress_dict(&encoded, &dictionaries).map_err(|e| e.to_string())?;
    if checked != archive {
        return Err("recompressed archive did not round-trip".into());
    }
    let staged = scratch.path().join("overlay");
    let archive_out = staged.join(patch.archive);
    let directory = archive_out
        .parent()
        .ok_or_else(|| "archive has no parent".to_string())?;
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let file = std::fs::File::create(&archive_out).map_err(|e| e.to_string())?;
    let mut writer = HashingWriter::new(file);
    writer.write_all(&encoded).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    let (_, digest) = writer.finish();
    let value = out
        .to_str()
        .ok_or_else(|| "overlay path must be UTF-8".to_string())?;
    if value.contains(['\r', '\n']) {
        return Err("overlay path cannot contain a newline".into());
    }
    std::fs::write(
        staged.join("env.txt"),
        format!("DM_ROMFS_OVERLAY={value}\n"),
    )
    .map_err(|e| e.to_string())?;
    nxb_sys::rename::exclusive(&staged, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(digest)
}

struct Args {
    title: TitleId,
    source: PathBuf,
    out: PathBuf,
    zstd_library: PathBuf,
    scale: f32,
}

fn parse(args: &[String]) -> Result<Args, String> {
    if args.first().map(String::as_str) != Some("fixed-scale") {
        return Err(USAGE.into());
    }
    let mut values = std::collections::BTreeMap::new();
    let rest = &args[1..];
    for pair in rest.chunks(2) {
        let [name, value] = pair else {
            return Err(USAGE.into());
        };
        if !matches!(
            name.as_str(),
            "--title-id" | "--romfs" | "--out" | "--scale" | "--zstd-library"
        ) || values.insert(name.as_str(), value.as_str()).is_some()
        {
            return Err(format!("invalid or duplicate option {name}\n{USAGE}"));
        }
    }
    let get = |name| {
        values
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing {name}\n{USAGE}"))
    };
    let title = get("--title-id")?;
    if title.len() != 16 {
        return Err("title ID must have 16 hex digits".into());
    }
    let title =
        TitleId(u64::from_str_radix(title, 16).map_err(|_| "title ID must have 16 hex digits")?);
    let scale: f32 = get("--scale")?.parse().map_err(|_| "invalid scale")?;
    if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
        return Err("scale must be finite and greater than 0, up to 1".into());
    }
    Ok(Args {
        title,
        source: get("--romfs")?.into(),
        out: get("--out")?.into(),
        zstd_library: get("--zstd-library")?.into(),
        scale,
    })
}

/// `nxb romfs fixed-scale`: 2 bad arguments, 4 unsupported assets, 1 other errors.
pub fn main(args: &[String]) -> i32 {
    let args = match parse(args) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return 2;
        }
    };
    let result = (|| {
        let catalog = nxtitles::catalog()?;
        let resolved = catalog.resolve(&TitleFacts::new(args.title));
        let patch = resolved
            .fixed_resolution()
            .ok_or_else(|| format!("unsupported fixed resolution for title {}", args.title))?;
        prepare(
            &args.source,
            &args.out,
            &args.zstd_library,
            patch.value,
            args.scale,
        )
    })();
    match result {
        Ok(digest) => {
            println!(
                "overlay: {}\narchive SHA-256: {}\nenvironment: {}",
                args.out.display(),
                digest.sha256,
                args.out.join("env.txt").display()
            );
            0
        }
        Err(error) => {
            eprintln!("[RED] {error}");
            if error.starts_with("unsupported") {
                4
            } else {
                1
            }
        }
    }
}

#[cfg(test)]
mod tests;
