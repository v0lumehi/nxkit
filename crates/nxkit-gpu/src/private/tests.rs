//! Tests of the private GPU memory: tracking, and the guest and frame helpers of the other tests.

use super::*;
use crate::detail::{DispatchDetail, UboDetail};

mod early;
mod mirror;
mod staging;

#[test]
fn unknown_programs_count_as_writing_and_known_read_only_ones_do_not() {
    let mut rw = Rw::default();
    let slot_stage = 3 | (Stage::Fragment as u32) << 24;
    assert!(rw.written(Some(7), slot_stage), "unknown");
    assert!(rw.written(None, slot_stage), "no program (mesh stage)");
    rw.known.insert((7, slot_stage), false);
    assert!(!rw.written(Some(7), slot_stage));
    rw.known.insert((8, slot_stage), true);
    assert!(rw.written(Some(8), slot_stage));
}

#[test]
fn the_written_ranges_cover_stores_copies_and_learned_pools() {
    use crate::detail::{CopyDetail, DispatchDetail, UboDetail};
    let ubo = |address, size, slot| UboDetail {
        stage: Stage::Compute,
        slot,
        address,
        size,
        snapshot: None,
    };
    let detail = FrameDetail {
        dispatches: vec![DispatchDetail {
            program: "cs".into(),
            push: None,
            groups: [1, 1, 1],
            indirect: None,
            ubos: Vec::new(),
            sbufs: vec![ubo(0x1000, 64, 0), ubo(0x2000, 64, 1)],
            textures: Vec::new(),
            samplers: Vec::new(),
            images: Vec::new(),
        }],
        copies: vec![
            CopyDetail {
                kind: CopyKind::BufferFill,
                dst_addr: 0x3000,
                bytes: 32,
                ..Default::default()
            },
            CopyDetail {
                kind: CopyKind::TextureToTexture,
                dst_addr: 0x9000,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut rw = Rw::default();
    // Slot 1 of the compute program is read only.
    rw.known
        .insert((hash_str("cs"), 1 | (Stage::Compute as u32) << 24), false);
    let r = written_ranges(&detail, &[(0x5000, 0x6000)], &rw);
    assert!(r.contains(&(0x1000, 64)), "unknown slot counts as writing");
    assert!(
        !r.iter().any(|x| x.0 == 0x2000),
        "read only slot is left out"
    );
    assert!(r.contains(&(0x3000, 32)));
    assert!(
        !r.iter().any(|x| x.0 == 0x9000),
        "a texture copy writes no buffer"
    );
    assert!(r.contains(&(0x5000, 0x1000)), "a learned pool is whole");
}

#[test]
fn mesh_draw_written_ranges_respect_task_mesh_and_fragment_reflection() {
    use crate::detail::{
        BlendDetail, DepthStencilDetail, DrawDetail, DynamicDetail, ProgramRef, RasterDetail,
    };
    let sbuf = |stage, slot, address| UboDetail {
        stage,
        slot,
        address,
        size: 64,
        snapshot: None,
    };
    let draw = DrawDetail {
        program: ProgramRef {
            vertex: Arc::from(""),
            fragment: Some(Arc::from("fs_prog")),
            task: Some(Arc::from("ts_prog")),
            mesh: Some(Arc::from("ms_prog")),
            geometry: None,
        },
        push: None,
        blend: [BlendDetail::default(); 8],
        write_masks: [0xf; 8],
        depth_stencil: DepthStencilDetail::default(),
        raster: RasterDetail::default(),
        topology: 4,
        attributes: Vec::new(),
        streams: Vec::new(),
        args: DrawArgs::MeshTasks { first: 0, count: 1 },
        ubos: Vec::new(),
        sbufs: vec![
            sbuf(Stage::Task, 0, 0x1_0000),
            sbuf(Stage::Mesh, 0, 0x2_0000),
            sbuf(Stage::Mesh, 1, 0x3_0000),
            sbuf(Stage::Fragment, 0, 0x4_0000),
        ],
        textures: Vec::new(),
        samplers: Vec::new(),
        images: Vec::new(),
        dynamic: DynamicDetail::default(),
        feedback_input: false,
        capture: None,
    };
    let mut rw = Rw::default();
    let ts_h = rw.hash_of(draw.program.task.as_ref().unwrap());
    let ms_h = rw.hash_of(draw.program.mesh.as_ref().unwrap());
    let fs_h = rw.hash_of(draw.program.fragment.as_ref().unwrap());
    rw.known.insert((ts_h, (Stage::Task as u32) << 24), false);
    rw.known.insert((ms_h, (Stage::Mesh as u32) << 24), false);
    rw.known
        .insert((ms_h, 1 | (Stage::Mesh as u32) << 24), true);
    rw.known
        .insert((fs_h, (Stage::Fragment as u32) << 24), false);
    let detail = FrameDetail {
        draws: vec![Arc::new(draw)],
        ..Default::default()
    };
    let r = written_ranges(&detail, &[], &rw);
    assert!(
        !r.iter().any(|&(a, _)| a == 0x1_0000),
        "read-only task sbuf should be skipped"
    );
    assert!(
        !r.iter().any(|&(a, _)| a == 0x2_0000),
        "read-only mesh sbuf should be skipped"
    );
    assert!(r.contains(&(0x3_0000, 64)), "rw mesh sbuf must be included");
    assert!(
        !r.iter().any(|&(a, _)| a == 0x4_0000),
        "read-only fragment sbuf should be skipped"
    );
}

// ----- the mirror -----

use super::TEST_TURN as TURN;
const VA: u64 = nxkit_metal::GUEST_VA_BASE;

struct Guest {
    mem: Arc<GuestMemory>,
    map: AddressMap,
    ctx: Context,
}

fn guest(len: u64) -> Option<Guest> {
    let ctx = Context::system_default().ok()?;
    let mem = Arc::new(GuestMemory::allocate(&ctx, len).ok()?);
    let map = AddressMap::new(mem.host_base(), VA, mem.len()).ok()?;
    FORCE.with(|f| f.set(Some(Mode::Mirror)));
    FAILED.store(false, Ordering::Relaxed);
    attach(&ctx, mem.host_base(), mem.len(), VA);
    // These tests are about the waits for a writer. The late results are tested in `nxkit-metal`.
    Pages::global().set_late(false);
    Some(Guest { mem, map, ctx })
}

fn dispatch(addr: u64, size: u32) -> FrameDetail {
    let ubo = |address, size, slot| UboDetail {
        stage: Stage::Compute,
        slot,
        address,
        size,
        snapshot: None,
    };
    FrameDetail {
        dispatches: vec![DispatchDetail {
            program: "cs".into(),
            push: None,
            groups: [1, 1, 1],
            indirect: None,
            ubos: vec![ubo(addr + 0x8000, 64, 0)],
            sbufs: vec![ubo(addr, size, 0)],
            textures: Vec::new(),
            samplers: Vec::new(),
            images: Vec::new(),
        }],
        ..Default::default()
    }
}

fn pools_of(g: &Guest, o: Arc<GuestOverlay>) -> crate::guestpools::GuestPools {
    crate::guestpools::GuestPools::whole(VA, g.mem.len(), g.mem.clone()).with_overlay(o)
}

#[test]
fn parse_mode_defaults_to_mirror_and_accepts_zero_and_track() {
    assert_eq!(parse_mode(None), Mode::Mirror);
    assert_eq!(parse_mode(Some("")), Mode::Mirror);
    assert_eq!(parse_mode(Some("1")), Mode::Mirror);
    assert_eq!(parse_mode(Some("0")), Mode::Off);
    assert_eq!(parse_mode(Some("track")), Mode::Track);
}
