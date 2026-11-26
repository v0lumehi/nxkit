//! The serve client of the game's own decoder: runs `nxkit-host oodle-serve <run_dir>` as an
//! on-demand worker pool and installs it as [`nxb_split::oodle::Decoder`] for the duration of a split.
//!
//! Request: `u64` stored length, `u64` raw length (little-endian), stored bytes.
//! Answer: `u64` count, then that many decoded bytes (`0` means the decoder refused the stream).

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nxb_split::oodle::{Answer, Decoder};

/// How to start a Mac host Oodle worker.
#[derive(Clone, Debug)]
pub struct HostSpec {
    /// Path to `nxkit-host`.
    pub host: PathBuf,
    /// Leading subcommand arguments (normally `["oodle-serve"]`).
    pub args: Vec<OsString>,
    /// Run folder (`game.json`, `modules/`, `nxgame.dylib`).
    pub run_dir: PathBuf,
    /// Environment variables passed to the host process.
    pub env: Vec<(OsString, OsString)>,
    /// Folder for `oodle_serve_<n>.log` and scratch `NXKIT_SAVE`.
    pub logs: PathBuf,
}

impl HostSpec {
    /// Standard spec for `nxkit-host oodle-serve <run_dir>`.
    pub fn for_run(host: PathBuf, run_dir: PathBuf, logs: PathBuf) -> Self {
        let mut env = Vec::new();
        for k in ["PATH", "HOME", "TMPDIR"] {
            if let Some(v) = std::env::var_os(k) {
                env.push((OsString::from(k), v));
            }
        }
        Self {
            host,
            args: vec![OsString::from("oodle-serve")],
            run_dir,
            env,
            logs,
        }
    }
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    log: PathBuf,
}

/// Pool of `nxkit-host oodle-serve` worker processes started on demand.
pub struct HostDecoder {
    spec: HostSpec,
    idle: Mutex<Vec<Proc>>,
    started: AtomicUsize,
    said: Mutex<String>,
}

fn log_text(p: &Path) -> String {
    std::fs::read(p)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default()
}

fn reap(mut c: Child, t: Duration) {
    let t0 = Instant::now();
    while t0.elapsed() < t {
        if let Ok(Some(_)) = c.try_wait() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = c.kill();
    let _ = c.wait();
}

impl HostDecoder {
    /// Creates a decoder pool configured by `spec`.
    pub fn new(spec: HostSpec) -> Self {
        Self {
            spec,
            idle: Mutex::new(Vec::new()),
            started: AtomicUsize::new(0),
            said: Mutex::new(String::new()),
        }
    }

    /// Number of host worker processes spawned so far.
    pub fn started(&self) -> usize {
        self.started.load(Ordering::Relaxed)
    }

    fn spawn(&self) -> Result<Proc, String> {
        let n = self.started.fetch_add(1, Ordering::Relaxed);
        std::fs::create_dir_all(&self.spec.logs)
            .map_err(|e| format!("{}: {e}", self.spec.logs.display()))?;
        let save = self.spec.logs.join(format!("save_{n}"));
        let _ = std::fs::create_dir_all(&save);
        let log = self.spec.logs.join(format!("oodle_serve_{n}.log"));
        let f = std::fs::File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let mut child = Command::new(&self.spec.host)
            .args(&self.spec.args)
            .arg(&self.spec.run_dir)
            .env_clear()
            .envs(self.spec.env.iter().map(|(k, v)| (k, v)))
            .env("NXKIT_SAVE", &save)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(f)
            .spawn()
            .map_err(|e| format!("{}: {e}", self.spec.host.display()))?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            return Err("the Mac host's pipes are missing".into());
        };
        Ok(Proc {
            child,
            stdin,
            stdout,
            log,
        })
    }

    fn take(&self) -> Result<Proc, String> {
        let p = self.idle.lock().ok().and_then(|mut v| v.pop());
        match p {
            Some(p) => Ok(p),
            None => self.spawn(),
        }
    }

    fn gone(&self, p: Proc) -> String {
        let Proc {
            child,
            stdin,
            stdout,
            log,
        } = p;
        drop(stdin);
        drop(stdout);
        reap(child, Duration::from_secs(2));
        let t = log_text(&log);
        if let Ok(mut s) = self.said.lock() {
            s.clone_from(&t);
        }
        t
    }
}

fn exchange(p: &mut Proc, comp: &[u8], n: usize) -> Option<u64> {
    let mut head = [0u8; 16];
    head[..8].copy_from_slice(&(comp.len() as u64).to_le_bytes());
    head[8..].copy_from_slice(&(n as u64).to_le_bytes());
    p.stdin.write_all(&head).ok()?;
    p.stdin.write_all(comp).ok()?;
    p.stdin.flush().ok()?;
    let mut got = [0u8; 8];
    p.stdout.read_exact(&mut got).ok()?;
    Some(u64::from_le_bytes(got))
}

impl Decoder for HostDecoder {
    fn ask(&self, comp: &[u8], n: usize) -> Answer {
        let mut p = match self.take() {
            Ok(p) => p,
            Err(e) => return Answer::Gone(e),
        };
        match exchange(&mut p, comp, n) {
            Some(0) => {
                let t = log_text(&p.log);
                if let Ok(mut s) = self.said.lock() {
                    *s = t;
                }
                if let Ok(mut v) = self.idle.lock() {
                    v.push(p);
                }
                Answer::Refused
            }
            Some(k) if k == n as u64 => {
                let mut out = vec![0u8; n];
                if p.stdout.read_exact(&mut out).is_err() {
                    return Answer::Gone(self.gone(p));
                }
                if let Ok(mut v) = self.idle.lock() {
                    v.push(p);
                }
                Answer::Bytes(out)
            }
            _ => Answer::Gone(self.gone(p)),
        }
    }

    fn said(&self) -> String {
        self.said.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

impl Drop for HostDecoder {
    fn drop(&mut self) {
        let procs = self
            .idle
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default();
        for p in procs {
            drop(p.stdin);
            drop(p.stdout);
            reap(p.child, Duration::from_secs(5));
        }
    }
}

/// Installs [`HostDecoder`] globally for the lifetime of this guard.
pub struct Installed(Arc<HostDecoder>);

impl Installed {
    /// Creates and installs a [`HostDecoder`] from `spec`.
    pub fn new(spec: HostSpec) -> Self {
        let d = Arc::new(HostDecoder::new(spec));
        nxb_split::oodle::set_decoder(Some(d.clone()));
        Self(d)
    }

    /// Returns the underlying [`HostDecoder`].
    pub fn decoder(&self) -> &HostDecoder {
        &self.0
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        nxb_split::oodle::set_decoder(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAKE: &str = r#"#!/usr/bin/perl
binmode STDIN; binmode STDOUT; $| = 1;
print STDERR "fake host up\n";
while (read(STDIN, my $h, 16) == 16) {
  my ($c, $n) = unpack("Q<Q<", $h);
  read(STDIN, my $d, $c);
  if ($c == 1) { print pack("Q<", 0); next; }
  if ($c == 2) { print STDERR "no streaming decoder to find\n"; exit 1; }
  print pack("Q<", $n) . scalar reverse($d);
}
"#;

    fn spec(dir: &Path) -> Option<HostSpec> {
        if !Path::new("/usr/bin/perl").exists() {
            return None;
        }
        std::fs::create_dir_all(dir).ok()?;
        let host = dir.join("fakehost");
        std::fs::write(&host, FAKE).ok()?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&host, std::fs::Permissions::from_mode(0o755)).ok()?;
        Some(HostSpec {
            host,
            args: Vec::new(),
            run_dir: dir.to_path_buf(),
            env: vec![("PATH".into(), "/usr/bin:/bin".into())],
            logs: dir.join("logs"),
        })
    }

    #[test]
    fn the_host_decodes_refuses_and_goes() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-oodle-host");
        let _ = std::fs::remove_dir_all(&dir);
        let Some(s) = spec(&dir) else { return };
        let d = HostDecoder::new(s);
        assert_eq!(d.ask(b"abc", 3), Answer::Bytes(b"cba".to_vec()));
        assert_eq!(d.ask(b"wxyz", 4), Answer::Bytes(b"zyxw".to_vec()));
        assert_eq!(d.started(), 1);
        assert_eq!(d.ask(b"a", 9), Answer::Refused);
        assert!(d.said().contains("fake host up"));
        let a = d.ask(b"ab", 9);
        assert!(
            matches!(&a, Answer::Gone(t) if t.contains("no streaming decoder to find")),
            "{a:?}"
        );
        assert_eq!(d.ask(b"xy z", 4), Answer::Bytes(b"z yx".to_vec()));
        assert_eq!(d.started(), 2);
        let answers: Vec<Answer> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..4u8)
                .map(|i| {
                    let d = &d;
                    sc.spawn(move || d.ask(&[i, 1, 2], 3))
                })
                .collect();
            hs.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        assert_eq!(answers.len(), 4);
        assert!(answers.iter().all(|a| matches!(a, Answer::Bytes(_))));
        drop(d);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
