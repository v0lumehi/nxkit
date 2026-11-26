//! AIR emission tests for `texture2d_ms`.

use air_emit::{compile_source, Target};

const PRE: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

fn emit_with(body: &str, target: &Target) -> String {
    compile_source(&format!("{PRE}{body}"), target, "t.metal").expect("emits")
}

fn has(ir: &str, needle: &str) {
    assert!(ir.contains(needle), "missing `{needle}` in\n{ir}");
}

#[test]
fn emits_reads_queries_and_metadata_for_macos() {
    let ir = emit_with(
        "struct spvDescriptorSetBuffer0 {
             texture2d_ms<float> tf [[id(0)]];
             texture2d_ms<half> th [[id(1)]];
             texture2d_ms<int> ti [[id(2)]];
             texture2d_ms<uint> tu [[id(3)]];
         };
         struct O { float4 c0 [[color(0)]]; };
         fragment O main0(float4 pos [[position]], uint sid [[sample_id]], constant spvDescriptorSetBuffer0& d [[buffer(0)]]) {
             O o = {};
             uint2 px = uint2(pos.xy);
             float4 a = d.tf.read(px, sid);
             half4 b = d.th.read(px, 1u);
             int4 c = d.ti.read(px, 0u);
             uint4 u = d.tu.read(px, 0u);
             uint w = d.tf.get_width();
             uint h = d.tf.get_height();
             uint s = d.tf.get_num_samples();
             o.c0 = a + float4(b) + float4(c) + float4(u) + float4(w, h, s, 0u);
             return o;
         }",
        &Target::macos(),
    );

    has(
        &ir,
        "%\"struct.metal::texture2d_ms\" = type { %struct._texture_2d_ms_t addrspace(1)* }",
    );
    has(&ir, "%struct._texture_2d_ms_t = type opaque");

    has(
        &ir,
        "declare { <4 x float>, i8 } @air.read_texture_2d_ms.v4f32(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly, %struct._sampler_t addrspace(2)*, <2 x i32>, i32, i32)",
    );
    has(
        &ir,
        "declare { <4 x half>, i8 } @air.read_texture_2d_ms.v4f16(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly, %struct._sampler_t addrspace(2)*, <2 x i32>, i32, i32)",
    );
    has(
        &ir,
        "declare { <4 x i32>, i8 } @air.read_texture_2d_ms.s.v4i32(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly, %struct._sampler_t addrspace(2)*, <2 x i32>, i32, i32)",
    );
    has(
        &ir,
        "declare { <4 x i32>, i8 } @air.read_texture_2d_ms.u.v4i32(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly, %struct._sampler_t addrspace(2)*, <2 x i32>, i32, i32)",
    );
    has(
        &ir,
        "declare i32 @air.get_width_texture_2d_ms(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly)",
    );
    has(
        &ir,
        "declare i32 @air.get_height_texture_2d_ms(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly)",
    );
    has(
        &ir,
        "declare i32 @air.get_num_samples_texture_2d_ms(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly)",
    );

    has(
        &ir,
        "!\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.arg_type_name\", !\"texture2d_ms<float, read>\", !\"air.arg_name\", !\"tf\"",
    );
    has(
        &ir,
        "!\"_ZTSN5metal12texture2d_msIfLNS_6accessE1ELNS_16memory_coherenceE0EvEE\"",
    );
    has(&ir, "!\"__metal_texture_2d_ms_t\"");
}

#[test]
fn ios_target_omits_read_sampler_for_texture2d_ms() {
    let ir = emit_with(
        "struct spvDescriptorSetBuffer0 { texture2d_ms<float> t [[id(0)]]; };
         fragment float4 main0(float4 pos [[position]], constant spvDescriptorSetBuffer0& d [[buffer(0)]]) {
             uint w = d.t.get_width();
             return d.t.read(uint2(pos.xy), 0u) + float4(w);
         }",
        &Target::ios(),
    );
    assert!(!ir.contains("air.get_read_sampler"), "{ir}");
    has(
        &ir,
        "declare { <4 x float>, i8 } @air.read_texture_2d_ms.v4f32(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly, <2 x i32>, i32, i32)",
    );
    has(&ir, ", i32 0, i32 1) #");
    has(
        &ir,
        "declare i32 @air.get_width_texture_2d_ms(%struct._texture_2d_ms_t addrspace(1)* nocapture readonly)",
    );
}

#[test]
fn emits_metal_array_of_texture2d_ms() {
    let ir = emit_with(
        "struct spvPool_ms { uint n; ulong seen; array<texture2d_ms<float>, 2> e; };
         struct spvDescriptorSetBuffer0 {
             constant spvPool_ms* p [[id(0)]];
         };
         fragment float4 main0(float4 pos [[position]], constant spvDescriptorSetBuffer0& d [[buffer(0)]]) {
             return d.p->e[min(1u, d.p->n)].read(uint2(pos.xy), 2u);
         }",
        &Target::macos(),
    );
    has(
        &ir,
        "%struct.spvPool_ms = type { i32, i64, [2 x %\"struct.metal::texture2d_ms\"] }",
    );
    has(
        &ir,
        "!\"air.texture\", !\"air.location_index\", i32 2, i32 2, !\"air.read\", !\"air.arg_type_name\", !\"array<texture2d_ms<float, read>, 2>\", !\"air.arg_name\", !\"e\"",
    );
    has(
        &ir,
        "!\"_ZTSN5metal5arrayINS_12texture2d_msIfLNS_6accessE1ELNS_16memory_coherenceE0EvEELm2EEE\"",
    );
}
