//! Tests of the syntax tree, the construct counters and the recursion-limit emulation.

use msl_parse::ast::*;
use msl_parse::ty::{self, Scalar, Space, Ty};
use msl_parse::{check_source, Outcome};

fn checked(src: &str) -> msl_parse::Checked {
    match check_source(src) {
        Outcome::Checked(c) => c,
        other => panic!("not checked: {other:?}"),
    }
}

fn function<'m>(m: &'m Module, name: &str) -> &'m Function {
    m.items
        .iter()
        .find_map(|it| match it {
            Item::Function(f) if &*f.sig.name == name => Some(&**f),
            _ => None,
        })
        .expect("function")
}

#[test]
fn tree_of_a_small_function() {
    let c = checked("float f(float x, int i) { float y = x * 2.0 + float(i); return y; }");
    assert!(c.reasons.is_empty());
    let f = function(&c.module, "f");
    assert_eq!(f.sig.ret, ty::FLOAT);
    assert_eq!(f.sig.params.len(), 2);
    assert_eq!(f.body.stmts.len(), 2);
    let Stmt::Decl(d) = &f.body.stmts[0] else {
        panic!("decl")
    };
    assert_eq!(&*d.vars[0].name, "y");
    let Init::Expr(e) = &d.vars[0].init else {
        panic!("init")
    };
    // (x * 2.0) + float(i): left-associative with `*` binding tighter
    let ExprKind::Binary {
        op: BinOp::Add,
        lhs,
        rhs,
    } = &e.kind
    else {
        panic!("add")
    };
    assert!(matches!(lhs.kind, ExprKind::Binary { op: BinOp::Mul, .. }));
    assert!(matches!(rhs.kind, ExprKind::Construct { .. }));
    assert_eq!(e.ty, ty::FLOAT);
    let Stmt::Return(Some(r)) = &f.body.stmts[1] else {
        panic!("return")
    };
    // a local variable is an lvalue: Ref(thread, float)
    assert_eq!(r.ty, Ty::Ref(Space::Thread, std::rc::Rc::new(ty::FLOAT)));
}

#[test]
fn spans_cover_the_source() {
    let src = "float f(float x) { return x + 1.0; }";
    let c = checked(src);
    let f = function(&c.module, "f");
    assert_eq!(&src[f.span.range()], src);
    let Stmt::Return(Some(e)) = &f.body.stmts[0] else {
        panic!()
    };
    assert_eq!(&src[e.span.range()], "x + 1.0");
}

#[test]
fn left_associativity_and_precedence() {
    let c = checked("int f(int a, int b, int c) { return a - b - c << 1 == 0 ? a : b; }");
    let f = function(&c.module, "f");
    let Stmt::Return(Some(e)) = &f.body.stmts[0] else {
        panic!()
    };
    let ExprKind::Cond { cond, .. } = &e.kind else {
        panic!("cond")
    };
    let ExprKind::Binary {
        op: BinOp::Eq, lhs, ..
    } = &cond.kind
    else {
        panic!("==")
    };
    let ExprKind::Binary {
        op: BinOp::Shl,
        lhs,
        ..
    } = &lhs.kind
    else {
        panic!("<<")
    };
    let ExprKind::Binary {
        op: BinOp::Sub,
        lhs,
        ..
    } = &lhs.kind
    else {
        panic!("-")
    };
    assert!(
        matches!(lhs.kind, ExprKind::Binary { op: BinOp::Sub, .. }),
        "(a - b) - c"
    );
}

#[test]
fn rejected_items_stay_in_the_tree() {
    let c = checked("void f(foo x) { }\nfloat g() { return 1.0; }");
    assert_eq!(c.reasons.get("type:foo"), 1);
    assert!(matches!(&c.module.items[0], Item::Rejected { reason, .. } if reason == "type:foo"));
    assert!(matches!(&c.module.items[1], Item::Function(_)));
}

#[test]
fn counters_match_the_prototype_spelling() {
    let c = checked("float f(float x) { float _1 = x; _1 = _1 * 2.0; return _1; }");
    assert_eq!(c.stats.get("function"), 1);
    assert_eq!(c.stats.get("decl:ssa"), 1);
    assert_eq!(c.stats.get("assign:ssa-reassign"), 1);
    assert_eq!(c.stats.get("op:*"), 1);
    assert_eq!(c.stats.get("stmt:return"), 1);
    assert_eq!(c.stats.get("ssa-names-reassigned"), 1);
}

#[test]
fn types_of_literals() {
    let c = checked(
        "void f() { long a = 3000000000; uint b = 0x80000000; half h = 1.0h; ulong u = 1ul; }",
    );
    assert!(c.reasons.is_empty());
    let _ = Scalar::Long;
}

fn nested_parens(n: usize) -> String {
    format!(
        "float f(float x) {{ return {}x{}; }}",
        "(".repeat(n),
        ")".repeat(n)
    )
}

#[test]
fn recursion_limit_matches_the_prototype() {
    // The prototype (Python 3.14, pool worker, limit 1000) accepts 57 nested parentheses
    // here and fails with RecursionError at 58 (tools/nesting.py measures this).
    assert!(check_source(&nested_parens(57)).accepted());
    assert!(matches!(
        check_source(&nested_parens(58)),
        Outcome::Recursion
    ));
    let blocks = |n: usize| format!("void f() {{ {}{} }}", "{ ".repeat(n), "}".repeat(n));
    assert!(check_source(&blocks(493)).accepted());
    assert!(matches!(check_source(&blocks(494)), Outcome::Recursion));
}

#[test]
fn deep_input_does_not_overflow_the_stack() {
    // Far beyond the limit: must end in Recursion, not a stack overflow. Run on a thread with
    // the default 2 MiB stack to show that the emulated limit bounds the recursion.
    let h =
        std::thread::spawn(|| matches!(check_source(&nested_parens(100_000)), Outcome::Recursion));
    assert!(h.join().unwrap());
}

#[test]
fn texture2d_ms_defaults_to_read_access_and_types_methods() {
    let c = checked(
        "uint f(texture2d_ms<float> tf, texture2d_ms<half, access::read> th, uint2 px) { \
         float4 a = tf.read(px, 0u); \
         half4 b = th.read(px, 1); \
         return tf.get_width() + tf.get_height() + tf.get_num_samples() + uint(a.x) + uint(b.x); \
         }",
    );
    assert!(c.reasons.is_empty(), "{:?}", c.reasons);
    let f = function(&c.module, "f");
    assert_eq!(
        f.sig.params[0].ty,
        Ty::Tex {
            kind: "texture2d_ms".into(),
            el: Scalar::Float,
            access: "read".into(),
        }
    );
    assert_eq!(
        f.sig.params[1].ty,
        Ty::Tex {
            kind: "texture2d_ms".into(),
            el: Scalar::Half,
            access: "read".into(),
        }
    );
    assert_eq!(f.sig.ret, ty::UINT);
}
