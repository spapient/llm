//! 训练测试共用的夹具：`loss = add_f32(p, x)` 的自动求导图与形状表。
//!
//! 在 crate 根挂一次（`src/lib.rs`），各测试模块经 `crate::support` 引用：同构的装配只留一处，
//! 不必在每个内联 `mod tests` 里另写一遍路径。各处只用到其中一部分，故放开未使用告警。

#![allow(dead_code)]

use autograd::{Sizes, Tape, Var};
use tensor::{DType, Shape};

/// 批量登记形状：`(变量, 元素数)` 逐项按 f32 写进新表。
pub(crate) fn sizes_of(tape: &Tape, vars: &[(Var, usize)]) -> Sizes {
    let mut sizes = Sizes::new(tape.vars());
    for &(var, len) in vars {
        sizes.set(var, Shape::new([len]), DType::F32);
    }
    sizes
}

/// `loss = add_f32(p, x)` 的夹具图：`p` 是梯度终点参数、`x` 是常量输入叶子，三者都按 `width`
/// 宽登记；返回图、形状表与三个变量。
pub(crate) fn add_graph(width: usize) -> (Tape, Sizes, Var, Var, Var) {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let x = tape.leaf();
    let loss = tape.record("add_f32", &[p, x]).expect("记录 loss");
    let sizes = sizes_of(&tape, &[(p, width), (x, width), (loss, width)]);
    (tape, sizes, p, x, loss)
}
