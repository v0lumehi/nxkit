//! `nxkit-host <run folder> [--fullscreen]` runs a game. `nxkit-host probe <run folder>` prints the R18 measurements.
//! With no arguments, the host uses the run folder in its application bundle. `--save-sync-report` and
//! `--save-sync-send` are the cloud save entries for a launcher (`savesync.rs`).

#[cfg(any(target_os = "macos", test))]
fn bundled_run_folder(executable: &std::path::Path) -> Result<std::path::PathBuf, String> {
    use std::ffi::OsStr;

    let macos = executable.parent();
    let contents = macos.and_then(std::path::Path::parent);
    let app = contents.and_then(std::path::Path::parent);
    if !executable.is_absolute()
        || executable.file_name() != Some(OsStr::new("nxkit-host"))
        || macos.and_then(std::path::Path::file_name) != Some(OsStr::new("MacOS"))
        || contents.and_then(std::path::Path::file_name) != Some(OsStr::new("Contents"))
        || app.and_then(std::path::Path::extension) != Some(OsStr::new("app"))
    {
        return Err(format!(
            "{} is not in a generated application bundle. Supply a run folder explicitly.",
            executable.display()
        ));
    }
    let Some(contents) = contents else {
        return Err("The application bundle has no Contents folder.".into());
    };
    let root = contents.join("Resources/nxkit");
    if !root.join("game.json").is_file() {
        return Err(format!(
            "The application bundle has no run manifest at {}.",
            root.join("game.json").display()
        ));
    }
    Ok(root)
}

/// Filters out macOS LaunchServices and Cocoa defaults flags from the argument list.
#[cfg(any(target_os = "macos", test))]
fn filter_launch_args(args: &[String]) -> Vec<&str> {
    let mut out = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg.starts_with("-psn_") {
            i += 1;
            continue;
        }
        if arg.starts_with("-NS") || arg.starts_with("-Apple") {
            i += 1;
            if let Some(next) = args.get(i).map(String::as_str) {
                let is_value_token = !next.starts_with('-')
                    && next != "probe"
                    && next != "home-preview"
                    && next != "oodle-serve"
                    && !next.contains('/')
                    && !std::path::Path::new(next).is_dir();
                if is_value_token {
                    i += 1;
                }
            }
            continue;
        }
        out.push(arg);
        i += 1;
    }
    out
}

#[cfg(target_os = "macos")]
fn resolve_bundled_run_folder_or_exit(usage: &str) -> std::path::PathBuf {
    match std::env::current_exe()
        .map_err(|e| format!("Cannot locate nxkit-host: {e}"))
        .and_then(|executable| bundled_run_folder(&executable))
    {
        Ok(root) => root,
        Err(e) => {
            eprintln!("nxkit-host: {e}\n{usage}");
            nxkit_host::stop::log_error(&e);
            std::process::exit(64);
        }
    }
}

/// The allocator with the spy of `NXKIT_MEM_ALLOC` (see `docs/diagnostics.md`).
#[cfg(target_os = "macos")]
#[global_allocator]
static ALLOC: nxkit_host::sys::allocspy::Spy = nxkit_host::sys::allocspy::Spy;

#[cfg(target_os = "macos")]
fn main() {
    use std::path::PathBuf;

    use nxkit_host::run::Options;

    nxkit_host::sys::rlimit::raise_open_files();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let filtered = filter_launch_args(&args);
    let usage = "usage: nxkit-host [--fullscreen] <run folder>\n       nxkit-host probe <run folder>\n       \
         nxkit-host oodle-serve <run folder>\n       nxkit-host home-preview <folder>\n       \
         nxkit-host --save-sync-report[=<ms>] [<run folder>]\n       \
         nxkit-host --save-sync-send [<run folder>]";
    if let Some(code) = save_sync(&filtered, usage) {
        std::process::exit(code);
    }
    let result = match filtered.as_slice() {
        [] => nxkit_host::run::run(&Options {
            root: resolve_bundled_run_folder_or_exit(usage),
            fullscreen: false,
        }),
        ["--fullscreen"] => nxkit_host::run::run(&Options {
            root: resolve_bundled_run_folder_or_exit(usage),
            fullscreen: true,
        }),
        ["probe", root] => nxkit_host::probe::run(&PathBuf::from(*root)),
        ["oodle-serve", root] => nxkit_host::oodle::serve(&PathBuf::from(*root)),
        ["home-preview", dir] => match nxkit_host::sys::homepreview::run(&PathBuf::from(*dir)) {
            Ok(files) => {
                for f in files {
                    println!("{}", f.display());
                }
                Ok(())
            }
            Err(e) => {
                eprintln!("nxkit-host: {e}");
                std::process::exit(1);
            }
        },
        ["--fullscreen", root] | [root, "--fullscreen"] => nxkit_host::run::run(&Options {
            root: PathBuf::from(*root),
            fullscreen: true,
        }),
        [root]
            if *root != "probe"
                && *root != "home-preview"
                && *root != "oodle-serve"
                && !root.starts_with('-') =>
        {
            nxkit_host::run::run(&Options {
                root: PathBuf::from(*root),
                fullscreen: false,
            })
        }
        _ => {
            eprintln!("{usage}");
            nxkit_host::stop::log_error(&format!("invalid arguments: {filtered:?}"));
            std::process::exit(64);
        }
    };
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => nxkit_host::stop::fatal(&e.to_string()),
    }
}

/// The cloud save entries: `--save-sync-report[=<ms>]` prints the report line, `--save-sync-send` sends what the game
/// left. Both use the run folder of the bundle unless one is given. `None` for other arguments.
#[cfg(target_os = "macos")]
fn save_sync(args: &[&str], usage: &str) -> Option<i32> {
    use std::path::PathBuf;
    let first = *args.first()?;
    let report = first.strip_prefix("--save-sync-report");
    if report.is_none() && first != "--save-sync-send" {
        return None;
    }
    let root = match args.get(1..) {
        Some([root]) => Ok(PathBuf::from(*root)),
        Some([]) => std::env::current_exe()
            .map_err(|e| format!("Cannot locate nxkit-host: {e}"))
            .and_then(|executable| bundled_run_folder(&executable)),
        _ => Err(String::new()),
    };
    let root = match root {
        Ok(r) => r,
        Err(e) => {
            eprintln!("nxkit-host: {e}\n{usage}");
            nxkit_host::stop::log_error(if e.is_empty() {
                "invalid save-sync arguments"
            } else {
                &e
            });
            return Some(64);
        }
    };
    let env = nxkit_host::hostenv::HostEnv::from_env(&nxkit_abi::env::Env::capture());
    Some(match report {
        Some("") => nxkit_host::savesync::report_cli(&root, &env, std::time::Duration::ZERO),
        Some(ms) => match ms.strip_prefix('=').and_then(|m| m.parse::<u64>().ok()) {
            Some(ms) => {
                nxkit_host::savesync::report_cli(&root, &env, std::time::Duration::from_millis(ms))
            }
            None => {
                eprintln!("{usage}");
                nxkit_host::stop::log_error(&format!("invalid arguments: {args:?}"));
                64
            }
        },
        None => nxkit_host::savesync::send_cli(&root, &env),
    })
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("nxkit-host runs on macOS only");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::{bundled_run_folder, filter_launch_args};
    use std::path::Path;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn filters_launch_services_and_cocoa_defaults_flags() {
        let args = strings(&["-psn_0_12345"]);
        assert!(filter_launch_args(&args).is_empty());

        let args = strings(&[
            "-psn_0_12345",
            "-NSDocumentRevisionsDebugMode",
            "YES",
            "-AppleTextDirection",
            "YES",
            "--fullscreen",
        ]);
        assert_eq!(filter_launch_args(&args), vec!["--fullscreen"]);

        let args = strings(&[
            "-NSDocumentRevisionsDebugMode",
            "YES",
            "probe",
            "/path/to/run",
        ]);
        assert_eq!(filter_launch_args(&args), vec!["probe", "/path/to/run"]);

        let args = strings(&["-psn_0_12345", "probe", "/path/to/run"]);
        assert_eq!(filter_launch_args(&args), vec!["probe", "/path/to/run"]);

        let args = strings(&[
            "-AppleLanguages",
            "(en)",
            "-NSDocumentRevisionsDebugMode",
            "--fullscreen",
            "/path/to/run",
        ]);
        assert_eq!(
            filter_launch_args(&args),
            vec!["--fullscreen", "/path/to/run"]
        );

        let args = strings(&["-NSDocumentRevisionsDebugMode", "probe", "/path/to/run"]);
        assert_eq!(filter_launch_args(&args), vec!["probe", "/path/to/run"]);

        let args = strings(&["-AppleTextDirection", "home-preview", "/tmp/dir"]);
        assert_eq!(filter_launch_args(&args), vec!["home-preview", "/tmp/dir"]);

        let args = strings(&["-NSDocumentRevisionsDebugMode", "/path/to/run"]);
        assert_eq!(filter_launch_args(&args), vec!["/path/to/run"]);

        let args = strings(&["-NSDocumentRevisionsDebugMode", "."]);
        assert_eq!(filter_launch_args(&args), vec!["."]);

        let args = strings(&["-AppleTextDirection"]);
        assert!(filter_launch_args(&args).is_empty());
    }

    #[test]
    fn rejects_paths_outside_generated_bundles() {
        for path in [
            "/usr/local/bin/nxkit-host",
            "/tmp/Game/Contents/MacOS/nxkit-host",
            "/tmp/Game.app/Other/MacOS/nxkit-host",
            "/tmp/Game.app/Contents/bin/nxkit-host",
            "/tmp/Game.app/Contents/MacOS/other",
            "Game.app/Contents/MacOS/nxkit-host",
        ] {
            assert!(
                bundled_run_folder(Path::new(path))
                    .unwrap_err()
                    .contains("Supply a run folder explicitly")
            );
        }
    }

    #[test]
    fn resolves_bundle_resources_and_requires_a_manifest() {
        let temp = std::env::temp_dir().join(format!("nxkit-host-bundle-{}", std::process::id()));
        let app = temp.join("Game With Spaces.app");
        let executable = app.join("Contents/MacOS/nxkit-host");
        let root = app.join("Contents/Resources/nxkit");
        std::fs::create_dir_all(&root).unwrap();
        assert!(
            bundled_run_folder(&executable)
                .unwrap_err()
                .contains("no run manifest")
        );
        std::fs::write(root.join("game.json"), "{}").unwrap();
        assert_eq!(bundled_run_folder(&executable).unwrap(), root);
        std::fs::remove_dir_all(temp).unwrap();
    }
}
