//! The app bundle of the standalone build: `Info.plist`, `Contents/MacOS/nxkit-host` as the main program, and the
//! ad hoc signatures (R21).

use std::path::{Path, PathBuf};
use std::process::Command;

const HOST: &str = "nxkit-host";

/// The `nxkit-host` program: `NXB_NXKIT_HOST` or the file beside `exe` (`cargo build -p nxkit-host` puts it next to
/// `nxb` in `target/<profile>/`).
pub(crate) fn host(env: Option<&str>, exe: &Path) -> Result<PathBuf, String> {
    let path = match env.filter(|e| !e.is_empty()) {
        Some(e) => PathBuf::from(e),
        None => exe.parent().unwrap_or(Path::new(".")).join(HOST),
    };
    if path.is_file() {
        return Ok(path);
    }
    Err(format!(
        "{}: the app needs the nxkit-host program and it is missing (build it with `cargo build -p nxkit-host`, or set NXB_NXKIT_HOST)",
        path.display()
    ))
}

/// A name that is safe as a file name: no `/` and no `:`, which Finder shows as one another.
pub(crate) fn file_name(title: &str) -> String {
    let n: String = title
        .chars()
        .filter(|c| !matches!(c, '/' | ':' | '\\' | '\0') && !c.is_control())
        .collect();
    let n = n.trim().trim_start_matches('.').trim().to_string();
    if n.is_empty() { "Game".to_string() } else { n }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The bundle identifier of a title when `--bundle-id` is not given.
pub(crate) fn default_bundle_id(title_id: u64) -> String {
    format!("org.nxkit.game.{title_id:016x}")
}

/// `Info.plist`.
pub(crate) fn info_plist(name: &str, bundle_id: &str, version: &str) -> String {
    let rows = [
        ("CFBundleExecutable", HOST),
        ("CFBundleIdentifier", bundle_id),
        ("CFBundleName", name),
        ("CFBundleDisplayName", name),
        ("CFBundlePackageType", "APPL"),
        // `Contents/Resources/AppIcon.icns`, the cover of the game (standalone::icon)
        ("CFBundleIconFile", super::icon::NAME),
        ("CFBundleInfoDictionaryVersion", "6.0"),
        ("CFBundleVersion", version),
        ("CFBundleShortVersionString", version),
        ("LSMinimumSystemVersion", "14.0"),
        ("LSApplicationCategoryType", "public.app-category.games"),
        // Head tracking of spatial audio (`NXKIT_HEAD_TRACKING`) reads the motion sensors of AirPods.
        (
            "NSMotionUsageDescription",
            "Spatial audio follows the movement of your head.",
        ),
    ];
    let mut s = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n",
    );
    for (k, v) in rows {
        s.push_str(&format!(
            "\t<key>{k}</key>\n\t<string>{}</string>\n",
            escape(v)
        ));
    }
    s.push_str("\t<key>NSHighResolutionCapable</key>\n\t<true/>\n</dict>\n</plist>\n");
    s
}

/// Signs `path` ad hoc (`codesign --force --sign -`). The Mac runs no unsigned arm64 code.
pub(crate) fn sign(path: &Path) -> Result<(), String> {
    let o = Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(path)
        .output()
        .map_err(|e| format!("codesign: {e}"))?;
    if o.status.success() {
        return Ok(());
    }
    Err(format!(
        "codesign {}: {}",
        path.display(),
        String::from_utf8_lossy(&o.stderr).trim()
    ))
}

/// Lays out `MacOS/nxkit-host`, signs the program, and then writes `Info.plist` in the bundle `app`.
/// Signing before writing `Info.plist` signs only the Mach-O binary (`Sealed Resources=none`) without creating a stale
/// `Contents/_CodeSignature/CodeResources` before resources are populated. Returns the run folder,
/// `Contents/Resources/nxkit`.
pub(crate) fn skeleton(app: &Path, host: &Path, plist: &str) -> Result<PathBuf, String> {
    let (contents, run) = (app.join("Contents"), app.join("Contents/Resources/nxkit"));
    let macos = contents.join("MacOS");
    for d in [&macos, &run] {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let to = macos.join(HOST);
    let bytes = std::fs::read(host).map_err(|e| format!("{}: {e}", host.display()))?;
    std::fs::write(&to, &bytes).map_err(|e| format!("{}: {e}", to.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&to, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("{}: {e}", to.display()))?;
    }
    let _ = std::fs::remove_file(macos.join("._nxkit-host"));
    sign(&to)?;
    let plist_path = contents.join("Info.plist");
    std::fs::write(&plist_path, plist).map_err(|e| format!("{}: {e}", plist_path.display()))?;
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_loses_slashes_and_colons() {
        assert_eq!(file_name("Some Game: The Sequel"), "Some Game The Sequel");
        assert_eq!(file_name("../x/y"), "xy");
        assert_eq!(file_name(":"), "Game");
    }

    #[test]
    fn the_plist_names_the_host_and_escapes() {
        let p = info_plist("A & B", "org.nxkit.game.1", "1114112");
        assert!(p.contains("<key>CFBundleExecutable</key>\n\t<string>nxkit-host</string>"));
        assert!(p.contains("A &amp; B"));
    }

    #[test]
    fn a_missing_host_is_named() {
        let e = host(None, Path::new("/nonexistent/nxb")).unwrap_err();
        assert!(e.contains("NXB_NXKIT_HOST"));
        assert!(host(Some("/nonexistent/h"), Path::new("/x/nxb")).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn skeleton_signs_host_before_plist_and_cleans_appledouble() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("nxb-bundle-skel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let app = root.join("Game.app");
        let macos = app.join("Contents/MacOS");
        std::fs::create_dir_all(&macos).unwrap();
        std::fs::write(macos.join("._nxkit-host"), b"appledouble").unwrap();
        let exe = std::env::current_exe().unwrap();
        let plist = info_plist("Game", "org.nxkit.game.1", "0");
        let run = skeleton(&app, &exe, &plist).unwrap();
        assert_eq!(run, app.join("Contents/Resources/nxkit"));
        assert!(!macos.join("._nxkit-host").exists());
        assert!(!app.join("Contents/_CodeSignature").exists());
        assert!(app.join("Contents/Info.plist").is_file());
        let mode = std::fs::metadata(macos.join(HOST))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
