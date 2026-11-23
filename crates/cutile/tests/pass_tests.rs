use crate::kernel::{Access, Arith, Expr, Kernel, Parameter, ScalarType, ValueType};
use crate::pass::{EliminateDeadParams, Pass, Pipeline, normalize};

fn literal(value: f32) -> Expr {
    Expr::F32(value.to_bits())
}

fn load(name: &str) -> Expr {
    Expr::Load(name.to_string())
}

fn param(name: &str, ty: ValueType) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty,
    }
}

fn out_buf() -> Parameter {
    param(
        "out",
        ValueType::Buf {
            element: ScalarType::F32,
            access: Access::ReadWrite,
        },
    )
}

fn extent() -> Parameter {
    param("extent", ValueType::Scalar(ScalarType::Usize))
}

fn binary(op: Arith, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// 定律表逐条验证，并顺带钉住两条边界：`x + 0.0` 必须原样保留（`-0.0` 反例），
/// 子项先规约好的结果必须在同一趟里被父节点用上。
#[test]
fn laws_normalize_bottom_up() {
    let add = binary(Arith::Add, load("a"), literal(0.0));
    let cases = [
        (
            Expr::Neg(Box::new(Expr::Neg(Box::new(load("a"))))),
            load("a"),
        ),
        (binary(Arith::Mul, load("a"), literal(1.0)), load("a")),
        (binary(Arith::Mul, literal(1.0), load("a")), load("a")),
        (binary(Arith::Div, load("a"), literal(1.0)), load("a")),
        (binary(Arith::Sub, load("a"), literal(0.0)), load("a")),
        (binary(Arith::Add, literal(0.5), literal(0.5)), literal(1.0)),
        (Expr::Neg(Box::new(literal(2.0))), literal(-2.0)),
        (Expr::Abs(Box::new(literal(-2.5))), literal(2.5)),
        (
            Expr::Min {
                left: Box::new(literal(1.0)),
                right: Box::new(literal(3.0)),
            },
            literal(1.0),
        ),
        (
            Expr::Max {
                left: Box::new(literal(1.0)),
                right: Box::new(literal(3.0)),
            },
            literal(3.0),
        ),
        (
            binary(
                Arith::Mul,
                load("a"),
                binary(Arith::Add, literal(0.5), literal(0.5)),
            ),
            load("a"),
        ),
        (add.clone(), add),
    ];
    for (input, expected) in cases {
        assert_eq!(normalize(input), expected);
    }
}

#[test]
fn eliminate_dead_params_drops_unused_scalars_only() {
    let kernel = Kernel::map(
        "keep",
        vec![
            out_buf(),
            extent(),
            param("unused", ValueType::Scalar(ScalarType::F32)),
            param("used", ValueType::Scalar(ScalarType::F32)),
        ],
        "i",
        "extent",
        "out",
        Expr::Scalar("used".to_string()),
    );
    let kernel = EliminateDeadParams.run(kernel).expect("eliminate");
    let names: Vec<&str> = kernel
        .params
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect();
    assert_eq!(names, ["out", "extent", "used"]);
}

#[test]
fn standard_pipeline_is_idempotent() {
    let kernel = Kernel::map(
        "clean",
        vec![
            out_buf(),
            extent(),
            param("unused", ValueType::Scalar(ScalarType::F32)),
        ],
        "i",
        "extent",
        "out",
        Expr::Neg(Box::new(Expr::Neg(Box::new(literal(2.0))))),
    );
    let once = Pipeline::standard().run(kernel.clone()).expect("once");
    let twice = Pipeline::standard().run(once.clone()).expect("twice");
    assert_eq!(once, twice);
}

#[test]
fn run_rejects_invalid_kernels() {
    let kernel = Kernel::map(
        "bad",
        vec![
            out_buf(),
            param("extent", ValueType::Scalar(ScalarType::F32)),
        ],
        "i",
        "extent",
        "out",
        literal(0.0),
    );
    let error = Pipeline::new()
        .run(kernel)
        .expect_err("non-usize extent is rejected");
    assert!(error.to_string().contains("loop extent"));
}
