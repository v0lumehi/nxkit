//! Opt-in Mac producer built on the same title-independent analysis used by other targets.
use super::{Say, analysis, auto_shaders, bundle, extract, io, link, stage, vk_shaders};
use crate::Args;
use std::path::Path;

fn validate(a: &Args) -> Result<(), String> {
    if !matches!(a.own_front.as_str(), "" | "on") {
        return Err(
            "--auto-analyze requires --own-front on (no verdict-file or Apple compiler fallback)"
                .into(),
        );
    }
    if a.partial
        || a.pipeline_list.is_some()
        || a.manifest.is_some()
        || a.allow_damaged
        || a.gpu.as_deref().is_some_and(|s| s != "none")
    {
        return Err("--auto-analyze does not accept damaged data, shader subsets, pipeline lists or GPU archives".into());
    }
    Ok(())
}

fn put(run: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let rel = nxb_extract::safe_path::rel("run folder", name, false).map_err(|e| e.to_string())?;
    let path = run.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    }
    std::fs::write(&path, bytes).map_err(|e| io(&path, e))
}

fn publish(partial: &Path, out: &Path) -> Result<(), String> {
    super::replaceable(out)?;
    if out.exists() {
        std::fs::remove_dir_all(out).map_err(|e| io(out, e))?;
    }
    std::fs::rename(partial, out).map_err(|e| io(out, e))
}

pub(super) fn run(a: &Args, p: &Say) -> Result<(), String> {
    validate(a).map_err(|e| crate::exit::fail(2, e))?;
    if a.own_front.is_empty() && std::env::var("NXB_OWN_FRONT").is_ok_and(|v| v == "0") {
        return Err(crate::exit::fail(
            2,
            "--auto-analyze requires the own shader front end. Use --own-front on to override NXB_OWN_FRONT=0",
        ));
    }
    if a.dry_run {
        p.say("auto-analysis: discover and extract packages or NUP, analyze modules/NPDM/NVN, rewrite without title patches, translate shaders, link and bundle for Mac. Runtime compatibility is not certified.");
        return Ok(());
    }
    if let Some(out) = &a.out {
        super::replaceable(out)?;
    }
    p.plan(5);
    let game = stage(p, "extract", "validate application metadata", || {
        let game = extract::extract(a, p)?;
        let note = format!("{} [{:016X}] v{}", game.name, game.title_id, game.version);
        Ok((game, note))
    })?;
    let name = a.app_name.as_deref().unwrap_or(&game.name);
    let out = a
        .out
        .clone()
        .unwrap_or_else(|| super::home(a).join(format!("{}.app", bundle::file_name(name))));
    super::replaceable(&out)?;
    let mut partial = out.clone().into_os_string();
    partial.push(".partial");
    let partial = std::path::PathBuf::from(partial);
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let host = bundle::host(std::env::var("NXB_NXKIT_HOST").ok().as_deref(), &exe)?;
    let id = a
        .bundle_id
        .clone()
        .unwrap_or_else(|| bundle::default_bundle_id(game.title_id));
    let analyzed = stage(
        p,
        "analysis",
        "derive and rewrite modules without executable title patches",
        || {
            let analyzed = analysis::analyze(&game)?;
            let note = format!(
                "{} modules, NPDM main thread, NVN name candidates",
                analyzed.modules.len()
            );
            Ok((analyzed, note))
        },
    )?;
    match (analyzed.engine, analyzed.engine_candidates.as_slice()) {
        (Some(id), _) => p.say(&format!(
            "engine family {id}, found from the markers {}: the family's frame limit and render size arguments apply",
            analyzed.engine_hits.join(", ")
        )),
        (None, []) => p.say("no engine family found from the modules and the RomFS"),
        (None, many) => p.say(&format!(
            "no engine family: the markers {} fit {}, and nothing settles it",
            analyzed.engine_hits.join(", "),
            many.join(", ")
        )),
    }
    let vulkan = analyzed.graphics == nxtitle::GraphicsApi::Vulkan;
    if vulkan && !analyzed.replaced.is_empty() {
        p.say(&format!(
            "the Vulkan layer of the host replaces the modules {}: they are left out",
            analyzed.replaced.join(", ")
        ));
    }
    let _ = std::fs::remove_dir_all(&partial);
    if let Some(parent) = partial.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    }
    std::fs::create_dir(&partial).map_err(|e| io(&partial, e))?;
    let run = stage(p, "link", "make Mac app and sign guest dylib", || {
        let run = bundle::skeleton(
            &partial,
            &host,
            &bundle::info_plist(&bundle::file_name(name), &id, &game.version.to_string()),
        )?;
        put(&run, "game.json", analyzed.manifest(&game).as_bytes())?;
        put(
            &run,
            nxkit_abi::launch::FILE,
            analyzed.launcher(&game).as_bytes(),
        )?;
        put(&run, "main.npdm", &game.npdm)?;
        for (name, bytes) in &game.files {
            put(&run, name, bytes)?;
        }
        if let Some(romfs_dir) = &game.romfs_dir {
            super::link(
                p,
                a.threads,
                romfs_dir,
                game.romfs_overlay.as_deref(),
                &run.join("romfs"),
                "linking RomFS",
            )?;
        } else {
            std::fs::create_dir_all(run.join("romfs")).map_err(|e| io(&run.join("romfs"), e))?;
        }
        for (index, aoc_dir) in &game.aoc_dirs {
            let rel = format!("aoc/{index}");
            if aoc_dir.is_dir() {
                super::link(p, a.threads, aoc_dir, None, &run.join(&rel), "linking DLC")?;
            } else {
                std::fs::create_dir_all(run.join(&rel)).map_err(|e| io(&run.join(&rel), e))?;
            }
        }
        link::module_files(&run, &analyzed.modules, |_, _| {})?;
        link::write_dylib(&run, &analyzed.modules, |_, _| {})?;
        Ok((run, "Mac host and guest linked".into()))
    })?;
    stage(
        p,
        "shaders",
        if vulkan {
            "translate the SPIR-V shaders"
        } else {
            "discover and translate shaders for macOS"
        },
        || {
            if vulkan {
                let scratch = crate::host::TempDir::new_in(
                    a.work_dir.as_deref().unwrap_or(&std::env::temp_dir()),
                )?;
                let d = vk_shaders::run(a, scratch.path(), p, &run)?;
                return Ok(((), d.summary));
            }
            // Only a game on NVN has NVN shaders to split and translate
            if analyzed.graphics != nxtitle::GraphicsApi::Nvn {
                return Ok((
                    (),
                    format!("none: the game renders with {}", analyzed.graphics.id()),
                ));
            }
            let romfs_fallback = run.join("romfs");
            let romfs_dir = game.romfs_dir.as_deref().unwrap_or(&romfs_fallback);
            let count = auto_shaders::discover_dir(
                &game.modules,
                romfs_dir,
                game.romfs_overlay.as_deref(),
                false,
                a.threads,
                &run,
            )?;
            Ok((
                (),
                format!("{count} Metal libraries (zero does not prove no shaders are needed)"),
            ))
        },
    )?;
    stage(p, "bundle", "publish new app", || {
        let old_save = out.join("Contents/Resources/nxkit/save");
        if old_save.is_dir() {
            let new_save = run.join("save");
            if new_save.exists() {
                return Err(format!(
                    "{}: the new app already has a save folder",
                    new_save.display()
                ));
            }
            std::fs::rename(&old_save, &new_save).map_err(|e| io(&new_save, e))?;
            p.say(&format!(
                "save: kept from the old app ({})",
                new_save.display()
            ));
        }
        match super::fonts::write(&run.join("font")) {
            Ok(s) => p.say(&s),
            Err(e) => p.say(&format!(
                "warning: no shared fonts, button icons draw as \"?\": {e}"
            )),
        }
        let icon_jpg = run.join("icon.jpg");
        if icon_jpg.is_file() {
            let icon_scratch = crate::host::TempDir::new_in(
                a.work_dir.as_deref().unwrap_or(&std::env::temp_dir()),
            )?;
            match super::icon::write(
                &icon_jpg,
                &partial.join("Contents/Resources"),
                icon_scratch.path(),
            ) {
                Ok(s) => p.say(&s),
                Err(e) => p.say(&format!("warning: no app icon: {e}")),
            }
        } else {
            p.say("warning: no app icon, the control data has no icon.jpg");
        }
        publish(&partial, &out)?;
        Ok(((), out.display().to_string()))
    })?;
    p.say("Built without executable title patches. Launch and game compatibility are not verified by this build.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_paths_are_owned_and_stay_inside_the_run_folder() {
        let root = std::env::temp_dir().join(format!("nxb-generic-output-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        let out = root.join("Game.app");
        let partial = root.join("Game.app.partial");
        assert!(super::super::replaceable(&out).is_ok());
        std::fs::create_dir(&partial).unwrap();
        put(&partial, "/nested/file", b"data").unwrap();
        assert_eq!(std::fs::read(partial.join("nested/file")).unwrap(), b"data");
        assert!(put(&partial, "../escape", b"data").is_err());
        std::fs::create_dir(&out).unwrap();
        assert!(publish(&partial, &out).is_err());
        assert!(partial.join("nested/file").is_file());
        // An existing nxkit app is replaceable and replaced by publish.
        let host = out.join("Contents/MacOS/nxkit-host");
        std::fs::create_dir_all(host.parent().unwrap()).unwrap();
        std::fs::write(&host, b"old-host").unwrap();
        publish(&partial, &out).unwrap();
        assert!(out.join("nested/file").is_file());
        assert!(!out.join("Contents/MacOS/nxkit-host").exists());
        assert!(!partial.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_unsupported_input_modes() {
        let mut a = Args {
            nsp_dir: "game.nsp".into(),
            keys: Some("prod.keys".into()),
            ..Args::default()
        };
        assert!(validate(&a).is_ok());
        a.own_front = "off".into();
        assert!(validate(&a).is_err());
        a.own_front = "verdicts.json".into();
        assert!(validate(&a).is_err());
        a.own_front = "on".into();
        assert!(validate(&a).is_ok());
        a.pipeline_list = Some("pipelines.seen".into());
        assert!(validate(&a).is_err());
        a.pipeline_list = None;
        a.resource_patches.push(("x".into(), "y".into()));
        assert!(validate(&a).is_ok());
        a.resource_patches.clear();
        a.nsp_dir = "packages".into();
        assert!(validate(&a).is_ok());
    }
}
