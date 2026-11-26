//! 一元组合子闭集：逐族发射文本与类型闸反向用例。

use super::*;

/// `output[i] = op(input[i])` 内核：输入只读、输出可变，`extent` 为 `usize` 长度。
fn map_unary(op: UnaryFn) -> Kernel {
    Kernel::map(
        "unary",
        vec![
            buf("input", Access::Read),
            buf("out", Access::ReadWrite),
            scalar("extent", ScalarType::Usize),
        ],
        "i",
        "extent",
        "out",
        Expr::Unary {
            op,
            child: Box::new(Expr::Load("input".to_string())),
        },
    )
}

testkit::test_cases! {
/// 一元闭集逐族发射成对应 `<math.h>` 库调用；表驱动遍历，新增算子只在闭集里登记一行。
fn unary_ops_emit_their_math_libcalls() {
    let cases = [
        (UnaryFn::Abs, "fabsf"),
        (UnaryFn::Sqrt, "sqrtf"),
        (UnaryFn::Exp, "expf"),
        (UnaryFn::Ln, "logf"),
        (UnaryFn::Sin, "sinf"),
        (UnaryFn::Cos, "cosf"),
        (UnaryFn::Ceil, "ceilf"),
        (UnaryFn::Floor, "floorf"),
        (UnaryFn::Round, "roundf"),
        (UnaryFn::Trunc, "truncf"),
    ];
    for (op, call) in cases {
        let source = map_unary(op).emit_c().expect("unary emits C");
        assert!(source.contains("#include <math.h>"), "{source}");
        assert!(source.contains(&format!("{call}(input[i])")), "{source}");
    }
}

/// 一元组合子的子表达式必须算 `f32`：套在 `usize` 标量上在校验期就被打回。
fn unary_rejects_non_f32_operand() {
    let mut kernel = map_unary(UnaryFn::Abs);
    kernel.write = Write::Map(Expr::Unary {
        op: UnaryFn::Abs,
        child: Box::new(Expr::Scalar("extent".to_string())),
    });
    let error = kernel.verify().expect_err("usize operand is rejected");
    assert!(error.to_string().contains("f32"), "{error}");
}
}
