//! Process stop. Any thread asks, the main thread ends the application, and a watchdog ends the process when that
//! does not happen.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once, OnceLock};
use std::time::Duration;

/// How long the application gets to stop before the watchdog ends the process.
const GRACE: Duration = Duration::from_secs(5);

fn cell() -> &'static Arc<AtomicBool> {
    static CELL: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(AtomicBool::new(false)))
}

/// The stop flag. The window loop reads it on every display tick.
pub fn flag() -> Arc<AtomicBool> {
    cell().clone()
}

/// True once a stop was requested.
pub fn requested() -> bool {
    cell().load(Ordering::Acquire)
}

/// Asks the host to stop. The reason goes to stderr. The first call arms the watchdog.
pub fn request(why: &str) {
    static ARM: Once = Once::new();
    ARM.call_once(|| {
        eprintln!("nxkit-host: stopping, {why}");
        cell().store(true, Ordering::Release);
        std::thread::spawn(|| {
            std::thread::sleep(GRACE);
            eprintln!(
                "nxkit-host: the window did not stop in {} s, ending the process",
                GRACE.as_secs()
            );
            std::process::exit(0);
        });
    });
}

/// Stops the calling thread for good. The process ends from the main thread.
pub fn park_forever() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

/// Best-effort append of a fatal error line to `~/Library/Logs/nxkit/last-error.log`.
pub fn log_error(message: &str) {
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    log_error_in_home(std::path::Path::new(&home), message);
}

fn log_error_in_home(home: &std::path::Path, message: &str) {
    use std::io::Write;

    let dir = home.join("Library/Logs/nxkit");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("last-error.log"))
    else {
        return;
    };
    let _ = writeln!(file, "nxkit-host: fatal: {message}");
}

/// Ends the process with an error message. Used where the host cannot go on and cannot return an error (R15).
pub fn fatal(message: &str) -> ! {
    eprintln!("nxkit-host: fatal: {message}");
    log_error(message);
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::log_error_in_home;

    #[test]
    fn appends_fatal_lines_to_last_error_log() {
        let temp = std::env::temp_dir().join(format!("nxkit-host-stop-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        log_error_in_home(&temp, "first failure");
        log_error_in_home(&temp, "second failure");

        let log_path = temp.join("Library/Logs/nxkit/last-error.log");
        let contents = std::fs::read_to_string(&log_path);
        assert_eq!(
            contents.ok().as_deref(),
            Some("nxkit-host: fatal: first failure\nnxkit-host: fatal: second failure\n")
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn ignores_io_errors_without_panicking() {
        let temp =
            std::env::temp_dir().join(format!("nxkit-host-stop-log-err-{}", std::process::id()));
        assert!(std::fs::write(&temp, "not a directory").is_ok());
        log_error_in_home(&temp, "should not panic");
        let _ = std::fs::remove_file(&temp);
    }
}
