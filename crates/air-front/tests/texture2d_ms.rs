//! End-to-end `air-front` tests for `texture2d_ms`.

use air_front::{build, build_raw, compile, Ir, Target, Tools};

const SHADER: &str = "#include <metal_stdlib>
using namespace metal;
struct spvDescriptorSetBuffer0 {
    texture2d_ms<float> tf [[id(0)]];
    texture2d_ms<half, access::read> th [[id(1)]];
    texture2d_ms<int> ti [[id(2)]];
    texture2d_ms<uint> tu [[id(3)]];
    array<texture2d_ms<float>, 2> arr [[id(4)]];
};
struct O { float4 c0 [[color(0)]]; };
fragment O main0(float4 pos [[position]], uint sid [[sample_id]], constant spvDescriptorSetBuffer0& d [[buffer(0)]]) {
    O o = {};
    uint2 px = uint2(pos.xy);
    float4 a = d.tf.read(px, sid);
    half4 b = d.th.read(px, 0u);
    int4 c = d.ti.read(px, 0u);
    uint4 u = d.tu.read(px, 0u);
    float4 e = d.arr[1].read(px, 1u);
    uint dims = d.tf.get_width() + d.tf.get_height() + d.tf.get_num_samples();
    o.c0 = a + float4(b) + float4(c) + float4(u) + e + float4(dims);
    return o;
}
";

#[test]
fn compiles_to_bitcode_and_raw_metallib_on_all_targets() {
    for target in [Target::macos(), Target::macos_for(26), Target::ios()] {
        let ir = compile(SHADER.as_bytes(), &target).expect("compiles");
        assert!(
            matches!(ir, Ir::Bitcode(_)),
            "expected bitcode for {target:?}"
        );
        let lib = build_raw(SHADER.as_bytes(), &target).expect("raw library");
        let parsed = air_lib::parse(&lib).expect("valid metallib");
        assert_eq!(parsed.functions.len(), 1);
        assert_eq!(parsed.functions[0].name(), b"main0");
    }
}

#[test]
#[ignore = "needs Xcode's Metal toolchain"]
fn builds_library_with_apple_back_half() {
    let tools = Tools::find("macosx").expect("xcrun");
    let lib = build(SHADER.as_bytes(), &Target::macos(), &tools).expect("builds");
    let parsed = air_lib::parse(&lib).expect("metallib");
    assert_eq!(parsed.functions[0].name(), b"main0");
}
