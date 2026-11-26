//! The link stage: the dylib with every module's segments at its fixed address (R48), and the files of the run folder
//! that describe the modules.

use super::bundle;
use super::macho::{self, Seg};
use super::manifest;
use super::modules::Built;
use std::path::Path;

/// The image's identity: the first 16 bytes of a hash of the build IDs in load order.
pub(crate) fn uuid(mods: &[(Built, u64)]) -> [u8; 16] {
    let mut h = nxb_model::Sha256::new();
    for (b, _) in mods {
        h.update(b.build_id.as_bytes());
        h.update(&[0]);
    }
    let d = h.finalize().0;
    let mut u = [0u8; 16];
    u.copy_from_slice(&d[..16]);
    u
}

/// The text of `nxgame.dylib`: every segment of every module, by address.
pub(crate) fn dylib(mods: &[(Built, u64)]) -> Result<Vec<u8>, String> {
    let mut segs: Vec<Seg> = Vec::new();
    for (b, base) in mods {
        for g in &b.shape.segments {
            let addr = base.checked_add_signed(g.offset).ok_or_else(|| {
                format!(
                    "module {}: segment {} leaves the address space",
                    b.name, g.name
                )
            })?;
            let data = b
                .image
                .get(g.file_off..g.file_off + g.file_len)
                .ok_or_else(|| {
                    format!(
                        "module {}: segment {} reads {:#x} bytes at {:#x} of a {:#x} byte image",
                        b.name,
                        g.name,
                        g.file_len,
                        g.file_off,
                        b.image.len()
                    )
                })?;
            segs.push(Seg {
                name: &g.name,
                addr,
                data,
                mem_size: g.mem_size,
                exec: g.exec,
            });
        }
    }
    segs.sort_by_key(|s| s.addr);
    macho::write(&segs, uuid(mods))
}

fn put(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes `modules/<name>.{bin,imports.json,exports.json,fixups.tsv,gates.tsv}` into the run folder.
pub(crate) fn module_files(
    run: &Path,
    mods: &[(Built, u64)],
    mut progress: impl FnMut(u64, u64),
) -> Result<(), String> {
    let dir = run.join("modules");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (i, (b, _)) in mods.iter().enumerate() {
        progress(i as u64, mods.len() as u64);
        let at = |s: &str| dir.join(format!("{}.{s}", b.name));
        put(&at("bin"), &b.image)?;
        put(
            &at("imports.json"),
            manifest::imports_json(&b.symbols.imports).as_bytes(),
        )?;
        put(
            &at("exports.json"),
            manifest::symbols_json(&b.symbols.exports).as_bytes(),
        )?;
        put(&at("fixups.tsv"), b.fixups.as_bytes())?;
        put(&at("gates.tsv"), &b.gates)?;
    }
    Ok(())
}

/// Writes and signs `nxgame.dylib`. Returns its size. `progress` gets the steps done (of 3) and what comes next.
pub(crate) fn write_dylib(
    run: &Path,
    mods: &[(Built, u64)],
    mut progress: impl FnMut(u64, &str),
) -> Result<u64, String> {
    progress(0, "linking");
    let bytes = dylib(mods)?;
    let path = run.join("nxgame.dylib");
    progress(1, "writing");
    put(&path, &bytes)?;
    progress(2, "signing");
    bundle::sign(&path)?;
    Ok(bytes.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standalone::elf::Symbols;
    use crate::standalone::layout::{self, Fixups};

    #[test]
    fn links_two_area_module_into_macho_dylib() {
        let fixups = "# PATCH_OFF\t-0x20000\n# KEY_PAGE_OFF\t0x40000\n# KEY_PAGE_SIZE\t0x4000\n# TEXT_END\t0x10000\n# RO_OFF\t0x10000\n# DATA_OFF\t0x20000\n# IMAGE_SIZE\t0x68000\n# FILE_IMAGE_LEN\t0x30000\n# BACK_OFF\t0x60000\n# BACK_LEN\t0x8000\n";
        let mut image = vec![0u8; 0x3c000];
        image[0..4].copy_from_slice(&0xd65f_03c0u32.to_le_bytes()); // TEXT
        image[0x30000..0x30004].copy_from_slice(&0x1400_0001u32.to_le_bytes()); // PATCH
        image[0x34000..0x34004].copy_from_slice(&0xd503_201fu32.to_le_bytes()); // BACK
        let shape = layout::shape("main", 0, image.len(), &Fixups::parse(fixups)).unwrap();
        let bases = layout::bases(std::slice::from_ref(&shape));
        let built = Built {
            name: "main".into(),
            build_id: "0123456789abcdef0123456789abcdef".into(),
            shape,
            parts: [(0, 0x10000), (0x10000, 0x10000), (0x20000, 0x10000)],
            bss: 0x10000,
            image,
            symbols: Symbols::default(),
            fixups: fixups.into(),
            gates: Vec::new(),
        };
        let bytes = dylib(&[(built, bases[0])]).unwrap();
        // Verify Mach-O header and that all 4 module segment names are present.
        assert_eq!(&bytes[0..4], &0xfeed_facfu32.to_le_bytes());
        for seg_name in [
            b"__NX_MAIN_PATCH\0" as &[u8],
            b"__NX_MAIN_TEXT\0\0",
            b"__NX_MAIN_DATA\0\0",
            b"__NX_MAIN_BACK\0\0",
        ] {
            assert!(
                bytes.windows(seg_name.len()).any(|w| w == seg_name),
                "missing segment {:?}",
                String::from_utf8_lossy(seg_name)
            );
        }
    }
}
