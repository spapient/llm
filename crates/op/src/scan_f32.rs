//! `scan_f32`：β 指数梯形扫描（§5.6）的离散递推 `s_t = a·s_{t-1} + b·x_t`。
//!
//! 连续动力学 `h' = λ h + b x` 经 [`crate::scan::exp_trapezoid`] 离散成对角半群步 `(a, b)`；
//! 本算子把这两个离散系数固定为模板常量，内核沿时间轴串行递推。扫描的**第二份实现**（幺半群
//! 前缀、Blelloch 并行扫描、结合律与金标验证）住在 [`crate::scan`]，本模块只声明这份算子的
//! 形状、代价、样本与标量参考，`#[annotation::model]` 装配成 `MODEL`。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用 unary 骨架：时间轴 `n` 既是输入轴也是输出轴。
pub use crate::unary::{AXES, project_shape};

/// 对角衰减系数 `ā = exp(λ·dt)`（模板常量 `{{a}}`）。
const DECAY: f32 = 0.5;
/// 输入系数 `b̄ = expm1(λ·dt)/λ · b`（模板常量 `{{b}}`）。
const INPUT: f32 = 1.0;

/// 标量代价：每步一次乘加（2 flop）、读一次 `x`、写一次 `y`。
///
/// # Errors
///
/// 形状缺轴 `n` 时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    cost_n(shape, "scan", 2.0, 1)
}

/// 确定性样本：`x`、`y` 两个等长缓冲，标量 `N` 是时间步数。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            x_buffer(len),
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：`s = ā·s + b̄·x[t]`，初态 `s = 0`，与内核同一公式。
///
/// # Errors
///
/// 样本缺 `x` 缓冲时返回说明。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let mut state = 0.0f32;
    Ok(x.iter()
        .map(|value| {
            state = DECAY * state + INPUT * value;
            state
        })
        .collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出不等长时返回说明。
pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    let mut state = 0.0f32;
    for (out, value) in out.iter_mut().zip(x) {
        state = DECAY * state + INPUT * value;
        *out = state;
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Scan;

/// β 指数梯形扫描的具体语义别名。
#[annotation::model(name = "scan_f32", shape = [4096, 4096], dtype = "f32", skeleton = "scan")]
pub type ScanF32 = Scan;

#[cfg(test)]
#[path = "../tests/scan_f32_tests.rs"]
mod tests;
