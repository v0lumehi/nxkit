//! Tool mode (`nxkit-host oodle-serve <run folder>`) that loads the game's rewritten modules
//! and decodes Oodle streams fed over stdin/stdout using the game's own `OodleLZ_Decompress`.
//!
//! Wire protocol per request:
//! - Request: `u64` little-endian `comp_len`, `u64` little-endian `raw_len`, then `comp_len` bytes.
//! - Response: `u64` little-endian `decoded_len` (`raw_len` on success, `0` when refused), followed
//!   by `decoded_len` bytes when non-zero. Clean EOF on stdin exits with status 0.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::boot::Boot;
use crate::error::{HostError, Result};
use crate::gate::{self, CallArgs};
use crate::hostenv::RunSettings;
use crate::manifest::Game;
use crate::module::Module;
use crate::oodle_find::{FoundDecoder, ModuleView, find_decoder};
use crate::refresh::{Refresh, Request as RefreshRequest};
use crate::{guestsym, loader, place, seams, sys, thread};

const MAX_STREAM: usize = 512 << 20;
const SCRATCH_SIZE: usize = 16 << 20;

fn parse_override(raw: &str, modules: &[Module]) -> Result<u64> {
    if let Some((name, off)) = raw.split_once('+') {
        let m = modules
            .iter()
            .find(|m| m.name == name)
            .ok_or_else(|| HostError::App(format!("NXKIT_OODLE_ADDR: unknown module {name}")))?;
        let off = u64::from_str_radix(off.trim().trim_start_matches("0x"), 16)
            .map_err(|_| HostError::App(format!("NXKIT_OODLE_ADDR: bad hex offset {off}")))?;
        return Ok(m.base.wrapping_add(off));
    }
    u64::from_str_radix(raw.trim().trim_start_matches("0x"), 16)
        .map_err(|_| HostError::App(format!("NXKIT_OODLE_ADDR: bad hex address {raw}")))
}

fn locate_decoder(modules: &[Module]) -> Result<FoundDecoder> {
    if let Ok(raw) = std::env::var("NXKIT_OODLE_ADDR")
        && !raw.trim().is_empty()
    {
        return Ok(FoundDecoder {
            module: "override".into(),
            addr: parse_override(&raw, modules)?,
            how: "env",
        });
    }
    for sym in ["OodleLZ_Decompress", "_OodleLZ_Decompress"] {
        if let Some(addr) = guestsym::find(sym) {
            return Ok(FoundDecoder {
                module: "export".into(),
                addr,
                how: "symbol",
            });
        }
    }
    let mut ordered: Vec<&Module> = modules.iter().filter(|m| m.name == "main").collect();
    ordered.extend(modules.iter().filter(|m| m.name != "main"));
    for m in ordered {
        let found = nxkit_abi::guest::with_slice(m.base + m.text.offset, m.text.size, |text| {
            nxkit_abi::guest::with_slice(m.base + m.rodata.offset, m.rodata.size, |rodata| {
                find_decoder(&[ModuleView {
                    name: &m.name,
                    base: m.base,
                    text_offset: m.text.offset,
                    text,
                    rodata_offset: m.rodata.offset,
                    rodata,
                }])
            })
            .ok()
            .flatten()
        })
        .ok()
        .flatten();
        if let Some(f) = found {
            return Ok(f);
        }
    }
    Err(HostError::App("no streaming decoder to find".into()))
}

/// Runs the binary `oodle-serve` request loop over `input` and `output`, calling `decode(comp, raw, scratch)`.
fn serve_loop(
    mut input: impl Read,
    mut output: impl Write,
    mut decode: impl FnMut(&[u8], &mut [u8], &mut [u8]) -> usize,
) -> std::io::Result<()> {
    let mut scratch = vec![0u8; SCRATCH_SIZE];
    let mut comp = Vec::new();
    let mut raw = Vec::new();
    let mut head = [0u8; 16];
    loop {
        let mut read = 0usize;
        while read < head.len() {
            match input.read(&mut head[read..]) {
                Ok(0) if read == 0 => return Ok(()),
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "truncated request header",
                    ));
                }
                Ok(n) => read += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        let comp_len = u64::from_le_bytes([
            head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7],
        ]);
        let raw_len = u64::from_le_bytes([
            head[8], head[9], head[10], head[11], head[12], head[13], head[14], head[15],
        ]);
        let Ok(clen) = usize::try_from(comp_len) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "compressed length overflow",
            ));
        };
        if clen > MAX_STREAM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "compressed stream exceeds limit",
            ));
        }
        comp.resize(clen, 0);
        input.read_exact(&mut comp)?;
        let Ok(rlen) = usize::try_from(raw_len) else {
            output.write_all(&0u64.to_le_bytes())?;
            output.flush()?;
            continue;
        };
        if rlen == 0 || rlen > MAX_STREAM {
            output.write_all(&0u64.to_le_bytes())?;
            output.flush()?;
            continue;
        }
        raw.resize(rlen, 0);
        let got = decode(&comp, &mut raw, &mut scratch);
        if got == rlen {
            output.write_all(&(rlen as u64).to_le_bytes())?;
            output.write_all(&raw[..rlen])?;
        } else {
            output.write_all(&0u64.to_le_bytes())?;
        }
        output.flush()?;
    }
}

/// Loads the game in `root`, locates `OodleLZ_Decompress`, and serves decode requests over stdin/stdout.
pub fn serve(root: &Path) -> Result<()> {
    let proto_out = sys::logpipe::take_protocol_stdout()
        .map_err(|e| HostError::App(format!("cannot prepare oodle-serve stdout: {e}")))?;
    let snapshot = nxkit_abi::env::Env::capture();
    let run = RunSettings::from_env(&snapshot);
    run.install()
        .map_err(|e| HostError::Settings(e.to_string()))?;
    let env = &run.host;
    let game = Game::read(root, env)?;
    let mask = loader::service_mask(&game.services)?;
    if mask & nxkit_rt::Service::Kernel.bit() == 0 {
        return Err(HostError::Manifest(
            "\"services\" must include \"kernel\"".into(),
        ));
    }
    let placed = place::place(&game)?;
    let mem = placed.mem;
    let (layout, settings) = (mem.layout, placed.game.settings.clone());
    let screen = Arc::new(nxkit_nvn::screen::Screen::new(None));
    let refresh = Arc::new(Refresh::new(
        RefreshRequest::from_settings(&settings, env),
        screen,
    ));
    mem.seal()?;
    nxkit_rt::nxkit_init(mask);
    let arena = crate::obswire::arena_size(&placed.game, &run.ue, layout.alias.len)?;
    sys::kernel_init(thread::guest_thread_entry, &layout, &placed.game, arena)?;
    seams::register_kernel()?;
    seams::register_server(mask)?;
    seams::init_services(&placed.game, mask, env, &refresh)?;
    let program_args = placed.game.program_args(placed.game.settings.render_dims());
    let mut loaded = loader::load(placed, mask)?;
    guestsym::set(std::mem::take(&mut loaded.exports));
    let found = locate_decoder(&loaded.game.modules)?;
    eprintln!(
        "nxkit-host: oodle-serve using {} ({}) at {:#x}",
        found.module, found.how, found.addr
    );
    let main_handle = seams::attach_main_thread()?;
    let thread_env = thread::ThreadEnv::new()?;
    let sp = layout.main_stack.end();
    if std::env::var_os("NXKIT_OODLE_NO_BOOT").is_none_or(|v| v != "1")
        && let Ok(boot) = Boot::resolve(std::mem::take(&mut loaded.inits), &program_args)
    {
        boot.init(main_handle, sp);
    }
    let stdin = BufReader::new(std::io::stdin().lock());
    let stdout = BufWriter::new(proto_out);
    let res = serve_loop(stdin, stdout, |comp, raw, scratch| {
        let args = CallArgs {
            x: [
                comp.as_ptr() as usize as u64,
                comp.len() as u64,
                raw.as_mut_ptr() as usize as u64,
                raw.len() as u64,
                1, // OodleLZ_FuzzSafe_Yes
                0, // OodleLZ_CheckCRC_No
                0, // OodleLZ_Verbosity_None
                0, // decBufBase
            ],
            stack: [
                0, // decBufSize
                0, // fpCallback
                0, // callbackUserData
                scratch.as_mut_ptr() as usize as u64,
                scratch.len() as u64,
                3, // OodleLZ_Decode_Unthreaded
            ],
        };
        usize::try_from(gate::call(found.addr, &args, sp)).unwrap_or(0)
    })
    .map_err(|e| HostError::App(format!("oodle-serve io: {e}")));
    thread_env.release();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn req(comp: &[u8], raw_len: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(comp.len() as u64).to_le_bytes());
        out.extend_from_slice(&raw_len.to_le_bytes());
        out.extend_from_slice(comp);
        out
    }

    #[test]
    fn serve_loop_decodes_refuses_and_stops_on_eof() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&req(b"abc", 3));
        wire.extend_from_slice(&req(b"bad", 4));
        wire.extend_from_slice(&req(b"hi", 2));
        let mut out = Vec::new();
        serve_loop(Cursor::new(wire), &mut out, |comp, raw, scratch| {
            assert_eq!(scratch.len(), SCRATCH_SIZE);
            if comp == b"bad" {
                return 0;
            }
            for (dst, &b) in raw.iter_mut().zip(comp.iter().rev()) {
                *dst = b;
            }
            comp.len()
        })
        .expect("serve_loop");
        let mut want = Vec::new();
        want.extend_from_slice(&3u64.to_le_bytes());
        want.extend_from_slice(b"cba");
        want.extend_from_slice(&0u64.to_le_bytes());
        want.extend_from_slice(&2u64.to_le_bytes());
        want.extend_from_slice(b"ih");
        assert_eq!(out, want);
    }
}
