# Title layer

Three layers decide how one game is built and run.

- The core is the rest of the workspace. It knows no game and no engine.
- An engine family holds what is true for every game of one engine. The family `agl` is Nintendo EPD's engine on sead, agl and gsys.
- A title module holds what is true for one game only.

The API is the crate `nxtitle`. The crate `nxtitles` lists the built-in families and title modules.

## Folders

```
titles/
  nxtitle/              the API: traits, hooks, evidence, the catalog
  nxtitles/             the registry, one cargo feature per module
  families/<id>/        crate nxfamily-<id>
  games/<title id>/     crate nxgame-<short name>, with the game's profile.toml
```

## Who answers a question

A family and a title answer questions through hooks. `None` means no opinion.

- A single answer comes from the title first, then the family, then the core default.
- A list adds up. The family's entries come first, then the title's.

| hook | trait | answer |
|---|---|---|
| `shader_role` | `BuildHooks` | what a shader does for the frame, for example the game's own upscaler |
| `seeds` | `BuildHooks` | shaders to prepare before the first frame |
| `rewrite_rules` | `BuildHooks` | rewriter rules as TOML text |
| `fixed_resolution` | `BuildHooks` | recorded resource hashes and scale-field offsets for an explicit fixed-resolution overlay |
| `needs` | `RuntimeHooks` | host behaviours the game depends on, with the setting |
| `upscaler` | `RuntimeHooks` | keep the game's upscaler, render at the output size, or use MetalFX |
| `pacing` | `RuntimeHooks` | the game, the display or a fixed rate paces the frames |
| `runtime` | `RuntimeHooks` | whether nxnative or nxkit runs one service of the game |

A role names what a shader does. The core decides what to do with it. A title module does not change the core's code path for a shader.

## Which engine runs a game

The host is split into services (`Service`): `fs`, `input`, `audio`, `video`, `system`, `offline`, `pacing`, `kernel` and `gpu`. nxnative runs every service by default. A family or title answers `runtime` for a service with `Runtime::Nxkit`, and the nxkit runtime runs that service for the game. `Resolved::nxkit_services` lists them.

The plan for the nxkit runtime and for applying needs is [plans/runtime-hooks.md](plans/runtime-hooks.md).

## Evidence

Every answer is a `Fact`, a value with its `Evidence`.

- `Tested` names a test in this workspace.
- `Recorded` names a record outside the workspace, for example a game README in nxnative at a commit, and the day it was read.
- `Unverified` says why the fact is believed and what would check it.

The core can refuse an unverified answer, for example in a release build.

## Rules for a module

1. A title module does not work around a core bug. A fix that helps more than one game goes into the core.
2. A fact about engine code goes into the family, even when one game showed it. Its evidence names that game.
3. A fact about one game's content stays in the title module.
4. A title names its family by id. It does not depend on the family crate. Without the family, the title still works with the core defaults.
5. Only `nxtitles` depends on family and title crates. The core depends on `nxtitle` and `nxtitles`. `cargo run -p xtask -- layers` checks this.
6. The committed part of a game's profile (design 2.11) is `profile.toml` in the title module's folder. Keys that a capability check decides are not in it. A family or title states them as needs.

## Adding a game

1. Make `titles/games/<title id>/` with a crate `nxgame-<short name>` of layer `T`.
2. Implement `Title`, `BuildHooks` and `RuntimeHooks`. Leave every hook without an answer at its default.
3. Add `profile.toml` with `title_id` and `[declared] title`. A test parses it with `nxb-profile` and expects no problems.
4. Add the crate to `[workspace.dependencies]` and add a feature for it in `titles/nxtitles/Cargo.toml` and `catalog()`.

A new family follows the same steps in `titles/families/<id>/` with `Family` in place of `Title`.

## The Legend of Zelda: Ocarina of Time (`0400486024DEC000`)

The Nintendo Switch 2 game on NVN2, family `agl`. The records are in `seedds/nxnative` `dm/games/0400486024DEC000/README.md`.

- The family marks the `agl_ndsr_vr` networks as the game's upscaler. The archive path that identifies them is unverified.
- The title states `gpu_timestamps`, `draw_barriers`, `mem_map_limit`, `render_cond`, `tex_flush_upload` and `tex_dirty_upload`. The last three come from agl's depth prepass and the terrain maps the CPU writes. They are title facts, because Breath of the Wild sets none of them.

### Fixed-resolution overlay

`nxb romfs fixed-scale` pins the recorded dynamic-resolution palette to a chosen scale.
The command is opt-in. It does not alter the source image, shader caches or stock builds.
It preserves resolution initialization, timing and load thresholds.

The recorded assets are from v131072. The command checks the Bootup archive, dictionary
member and target member hashes. Another update is not supported until its assets are recorded.
It edits ten float fields in place. All other archive bytes stay unchanged before compression.

On macOS, install libzstd 1.5.7 and select its trusted native library file explicitly.
For Homebrew, the file is `$(brew --prefix zstd)/lib/libzstd.dylib`.
The command checks the loaded version. It uses the same one-shot dictionary API as the
reference overlay, not the CLI's streaming encoder.

```sh
nxb romfs fixed-scale \
  --title-id 0400486024DEC000 \
  --romfs "$ROMFS" \
  --scale 0.666667 \
  --zstd-library "$ZSTD_LIBRARY" \
  --out "$NEW_OVERLAY"
```

`ROMFS` can name a RomFS image or a loose folder. `NEW_OVERLAY` must not exist.
Its parent must exist and must be outside a loose source folder.
The scale must be finite, greater than zero and no greater than one.

The output contains `Pack/Bootup.Nin_Ounce_NVN2.pack.zs` and `env.txt`.
Use that file as the host's environment file, or set its `DM_ROMFS_OVERLAY` value
when you launch the development host. The host must support that option.
The command does not launch the game or add the overlay to an app bundle.
Omit the environment override to restore stock resources.

A fixed scale does not guarantee an output resolution, a frame rate or stable frame times.
A higher scale can increase GPU work.

The real-data test reads `NXB_PARITY_RESOLUTION_ROMFS`, `NXB_PARITY_RESOLUTION_REFERENCE`
and `NXB_PARITY_ZSTD_LIBRARY`. It compares the complete compressed output at scale
0.666667 against the retained reference overlay. It skips itself when these inputs are absent.
No game files belong in the repository.

Open questions:

- Is the game logic time-stepped, so that a pacing other than the game's own is safe?
- Should the host keep `agl_ndsr_vr`, render at the output size, or use MetalFX?
- Which facts of the family hold for Tears of the Kingdom and other agl games?

## The Legend of Zelda: Breath of the Wild (`01007EF00011E000`)

The Nintendo Switch game on NVN, update 1.9.0 (v1114112), family `agl`. The records are in `seedds/nxnative` `dm/games/01007EF00011E000/`.

- The title states `gpu_timestamps segment`, `fps_target 60`, `present_snap`, `pso_mtl4` and `screen_aspect`.
- `fps_target 60` needs the `FRAME_STEP` stubs of the game's `patches.py`. `nxb` applies them from `data/rewrite-rules`.
- The game is the pilot for the nxkit runtime. nxnative runs every service of it.
