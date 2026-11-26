//! Lowering tests for `texture2d_ms`.

use msl_sema::hir::{Expr, ExprKind, Module, Query, Stmt, TexOp};
use msl_sema::types::{Access, Scalar, Tex, TexKind, Type};
use msl_sema::{analyze, Error};

const PRE: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

fn lower(body: &str) -> Result<Module, Error> {
    let src = format!("{PRE}{body}");
    let msl_parse::Outcome::Checked(c) = msl_parse::check_source(&src) else {
        panic!("parse failed for:\n{src}");
    };
    assert!(c.reasons.is_empty(), "rejected: {:?}", c.reasons);
    analyze(&c.module)
}

#[test]
fn lowers_read_and_dimension_queries() {
    let m = lower(
        "struct spvDescriptorSetBuffer0 {
             texture2d_ms<float> tf [[id(0)]];
             texture2d_ms<half, access::read> th [[id(1)]];
             texture2d_ms<int> ti [[id(2)]];
             texture2d_ms<uint> tu [[id(3)]];
         };
         fragment float4 main0(constant spvDescriptorSetBuffer0& d [[buffer(0)]], int2 px [[position]]) {
             float4 a = d.tf.read(px, 0);
             half4 b = d.th.read(uint2(px), 1u);
             int4 c = d.ti.read(uint2(px), 0u);
             uint4 u = d.tu.read(uint2(px), 0u);
             uint n = d.tf.get_width() + d.tf.get_height() + d.tf.get_num_samples();
             return a + float4(b) + float4(c) + float4(u) + float4(n);
         }",
    )
    .expect("lowers");

    let ab = m
        .types
        .structs
        .iter()
        .find(|s| &*s.name == "spvDescriptorSetBuffer0")
        .expect("argbuf");
    assert_eq!(
        ab.fields[0].ty,
        Type::Texture(Tex {
            kind: TexKind::T2dMs,
            elem: Scalar::Float,
            access: Access::Read,
        })
    );
    assert_eq!(
        m.types.msl_name(&ab.fields[0].ty),
        "texture2d_ms<float, read>"
    );

    let f = &m.funcs[0];
    let reads: Vec<_> = f
        .body
        .iter()
        .filter_map(|s| match s {
            Stmt::Expr(Expr {
                kind: ExprKind::Assign(_, rhs),
                ..
            }) => match &rhs.kind {
                ExprKind::Texture {
                    tex_ty,
                    op: TexOp::Read { .. },
                    ..
                } => Some((*tex_ty, rhs)),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(reads.len(), 4);
    let (tex_ty, init_a) = reads[0];
    let ExprKind::Texture {
        op:
            TexOp::Read {
                coord,
                array,
                lod,
                sample,
            },
        ..
    } = &init_a.kind
    else {
        unreachable!()
    };
    assert_eq!(tex_ty.kind, TexKind::T2dMs);
    assert_eq!(coord.ty, Type::Vector(Scalar::Uint, 2));
    assert!(matches!(coord.kind, ExprKind::Convert(_)));
    assert!(array.is_none());
    assert!(lod.is_none());
    let sample = sample.as_ref().expect("sample index");
    assert_eq!(sample.ty, Type::UINT);
    assert!(matches!(sample.kind, ExprKind::Convert(_)));

    let queries: Vec<Query> = f
        .body
        .iter()
        .filter_map(|s| match s {
            Stmt::Expr(e) => collect_queries(e),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(
        queries,
        vec![Query::Width, Query::Height, Query::NumSamples]
    );
}

fn collect_queries(e: &Expr) -> Option<Vec<Query>> {
    let mut out = Vec::new();
    walk_queries(e, &mut out);
    (!out.is_empty()).then_some(out)
}

fn walk_queries(e: &Expr, out: &mut Vec<Query>) {
    match &e.kind {
        ExprKind::Texture {
            op: TexOp::Query { what, .. },
            ..
        } => out.push(*what),
        ExprKind::Assign(_, rhs) => walk_queries(rhs, out),
        ExprKind::Binary(_, a, b) => {
            walk_queries(a, out);
            walk_queries(b, out);
        }
        _ => {}
    }
}

#[test]
fn lowers_metal_array_of_texture2d_ms() {
    let m = lower(
        "struct spvDescriptorSetBuffer0 {
             array<texture2d_ms<float>, 2> t [[id(0)]];
         };
         fragment float4 main0(constant spvDescriptorSetBuffer0& d [[buffer(0)]], float4 pos [[position]]) {
             return d.t[1].read(uint2(pos.xy), 0u);
         }",
    )
    .expect("lowers");
    let ab = &m.types.structs[0];
    assert!(ab.fields[0].metal_array);
    assert_eq!(
        m.types.field_msl_name(&ab.fields[0]),
        "array<texture2d_ms<float, read>, 2>"
    );
}

#[test]
fn rejects_unsupported_access_and_methods() {
    for (body, expected) in [
        (
            "struct B { texture2d_ms<float, access::sample> t [[id(0)]]; };
             fragment float4 f(constant B& b [[buffer(0)]]) { return b.t.read(uint2(0u), 0u); }",
            "unsupported:texture2d_ms:access:sample",
        ),
        (
            "struct B { texture2d_ms<float> t [[id(0)]]; };
             fragment float4 f(constant B& b [[buffer(0)]], sampler s) { return b.t.sample(s, float2(0.0)); }",
            "unsupported:method:sample:texture2d_ms",
        ),
        (
            "struct B { texture2d_ms<float> t [[id(0)]]; };
             fragment void f(constant B& b [[buffer(0)]]) { b.t.write(float4(0.0), uint2(0u)); }",
            "unsupported:method:write:texture2d_ms",
        ),
        (
            "struct B { texture2d_ms<float> t [[id(0)]]; };
             fragment uint f(constant B& b [[buffer(0)]]) { return b.t.get_num_mip_levels(); }",
            "unsupported:method:get_num_mip_levels:texture2d_ms",
        ),
        (
            "struct B { texture2d_ms<float> t [[id(0)]]; };
             fragment uint f(constant B& b [[buffer(0)]]) { return b.t.get_width(0u); }",
            "invalid:query-arg:texture2d_ms",
        ),
        (
            "struct B { texture2d_ms<float> t [[id(0)]]; };
             fragment float4 f(constant B& b [[buffer(0)]]) { return b.t.read(uint2(0u)); }",
            "invalid:read sample",
        ),
        (
            "struct B { texture2d<float> t [[id(0)]]; };
             fragment uint f(constant B& b [[buffer(0)]]) { return b.t.get_num_samples(); }",
            "unsupported:method:get_num_samples:texture2d",
        ),
    ] {
        let err = lower(body).unwrap_err();
        assert_eq!(err.key(), expected, "for:\n{body}");
    }
}
