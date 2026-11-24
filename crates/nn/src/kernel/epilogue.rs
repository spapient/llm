//! 写回尾块执行：把 `prof::epilogue::Step` 闭集按声明顺序施加到 f32 瓦片，算术只转发 `gemm` 的拥有方。
//!
//! 六步闭集来自 [`prof::epilogue::Step`]：`bias` / `act` / `residual` / `scale` / `quant` / `store`。
//! 本模块不持有任何算术——激活逐个取 [`crate::unary`] 的标量核，量化用 [`crate::dequant::SCALE`]
//! 做对称 round-trip，残差与缩放直接就地累加。契约层（`prof`）只声明"有哪些步"，执行层在这里
//! 决定"每一步落在哪个既有内核上"。

use diagnostic::{Code, Error};
use prof::atom::{ActKind, DType};
use prof::epilogue::Step;

/// 按 `steps` 的顺序把写回尾块施加到 `out`（行主序，列宽 `cols`）。
///
/// * `bias`：长度 `cols` 的列偏置，按列广播；缺省跳过。
/// * `residual`：长度等于 `out` 的残差输入；缺省跳过。
/// * `act:<kind>`：逐元素施加 `crate::{kind}_f32` 的标量核，算术归 [`crate::unary`]。
/// * `scale:<α>`：逐元素乘 `α`。
/// * `quant:<i8|i4|i2>`：量化到对称整数网格再反量化（步长 [`crate::dequant::SCALE`]，误差 ≤ 半步长）。
/// * `store`：无副作用标记，本层空操作。
///
/// # Errors
///
/// `bias` / `residual` 长度不符、`quant` 位宽超出 `i8|i4|i2` 时返回 [`Code::Invalid`]。
pub fn apply(
    steps: &[Step],
    out: &mut [f32],
    cols: usize,
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
) -> diagnostic::Result<()> {
    for step in steps {
        apply_step(step, out, cols, bias, residual)?;
    }
    Ok(())
}

/// 单步分派：前三个是几何/标量步（`bias` / `residual` / `scale`），其余交给 [`apply_tail`]，
/// 把六臂 match 压成两段、每段 ≤5 圈复杂度。
fn apply_step(
    step: &Step,
    out: &mut [f32],
    cols: usize,
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
) -> diagnostic::Result<()> {
    match step {
        Step::Bias => add_bias(out, cols, bias),
        Step::Residual => add_residual(out, residual),
        Step::Scale(alpha) => scale_tile(out, *alpha),
        _ => apply_tail(step, out),
    }
}

/// 后三个是激活 / 量化 / 无副作用步；`_` 臂不可达（已由 [`apply_step`] 收走），防御性报错。
fn apply_tail(step: &Step, out: &mut [f32]) -> diagnostic::Result<()> {
    match step {
        Step::Act(kind) => {
            activate(out, *kind);
            Ok(())
        }
        Step::Quant(dtype) => quant_round_trip(out, *dtype),
        Step::Store => Ok(()),
        _ => Err(Error::new(Code::Invalid, "写回步骤分派不可达")),
    }
}

/// 列偏置广播：`out[r][c] += bias[c]`。缺省即空操作。
fn add_bias(out: &mut [f32], cols: usize, bias: Option<&[f32]>) -> diagnostic::Result<()> {
    match bias {
        Some(bias) => add_bias_cols(out, cols, bias),
        None => Ok(()),
    }
}

/// 有偏置的广播加：先校验列宽，再逐行累加。
fn add_bias_cols(out: &mut [f32], cols: usize, bias: &[f32]) -> diagnostic::Result<()> {
    if bias.len() != cols {
        return Err(shape_err("bias", cols, bias.len()));
    }
    for row in out.chunks_mut(cols) {
        for (x, b) in row.iter_mut().zip(bias) {
            *x += *b;
        }
    }
    Ok(())
}

/// 残差加：`out[i] += residual[i]`。缺省即空操作。
fn add_residual(out: &mut [f32], residual: Option<&[f32]>) -> diagnostic::Result<()> {
    let Some(residual) = residual else {
        return Ok(());
    };
    if residual.len() != out.len() {
        return Err(shape_err("residual", out.len(), residual.len()));
    }
    for (x, r) in out.iter_mut().zip(residual) {
        *x += *r;
    }
    Ok(())
}

/// 标量缩放：`out[i] *= alpha`。
fn scale_tile(out: &mut [f32], alpha: f32) -> diagnostic::Result<()> {
    for x in out.iter_mut() {
        *x *= alpha;
    }
    Ok(())
}

/// 激活分派：按 [`SCALARS`] 取标量核，就地施加（不建副本、不另分配）。
fn activate(out: &mut [f32], kind: ActKind) {
    let Some((_, apply)) = SCALARS.iter().find(|(have, _)| *have == kind) else {
        return;
    };
    for x in out.iter_mut() {
        *x = apply(*x);
    }
}

/// 一个标量激活核：`f32 -> f32`，取对拍侧 [`crate::unary::Unary::apply_f32`]。
type Scalar = fn(f32) -> f32;

/// 五个标量核的登记表：`ActKind` 的闭集成员与 `crate::unary` 的 f32 标量核一一对上，
/// 覆盖关系由 `epilogue_tests` 的 `every_act_kind_has_a_scalar_kernel` 钉住。
const SCALARS: [(ActKind, Scalar); 5] = [
    (
        ActKind::Relu,
        <crate::relu_f32::ReluF32 as crate::unary::Unary>::apply_f32,
    ),
    (
        ActKind::Gelu,
        <crate::gelu_f32::GeluF32 as crate::unary::Unary>::apply_f32,
    ),
    (
        ActKind::Silu,
        <crate::silu_f32::SiluF32 as crate::unary::Unary>::apply_f32,
    ),
    (
        ActKind::LeakyRelu,
        <crate::leaky_relu_f32::LeakyReluF32 as crate::unary::Unary>::apply_f32,
    ),
    (
        ActKind::HardSigmoid,
        <crate::hard_sigmoid_f32::HardSigmoidF32 as crate::unary::Unary>::apply_f32,
    ),
];

/// 量化到对称整数网格再反量化（round-trip）：`out[i] = SCALE * round(out[i] / SCALE)`，饱和到
/// `[-(2^(b-1)), 2^(b-1)-1]`。步长 [`crate::dequant::SCALE`]，最大误差 0.5 步长。
fn quant_round_trip(out: &mut [f32], dtype: DType) -> diagnostic::Result<()> {
    let (min, max) = quant_bounds(dtype)?;
    for x in out.iter_mut() {
        let code = (*x / crate::dequant::SCALE).round().clamp(min, max);
        *x = crate::dequant::SCALE * code;
    }
    Ok(())
}

/// 对称网格边界：`i2 → [-2, 1]`、`i4 → [-8, 7]`、`i8 → [-128, 127]`；其余位宽拒绝。
fn quant_bounds(dtype: DType) -> diagnostic::Result<(f32, f32)> {
    // 定义域是这条对称网格接受的三个精度；位宽从 `DType::bits` 取，不在此再写一遍字面量。
    let bits = match dtype {
        DType::I2 | DType::I4 | DType::I8 => dtype.bits(),
        _ => {
            return Err(Error::new(
                Code::Invalid,
                format!("`quant` 只接受 i8/i4/i2，`{}` 不是", dtype.name()),
            ));
        }
    };
    let half = 1u32 << (bits - 1);
    Ok((-(half as f32), (half - 1) as f32))
}

/// 形状不符的错误。
fn shape_err(what: &str, expected: usize, got: usize) -> Error {
    Error::new(
        Code::Invalid,
        format!("{what} 长度不一致：期望 {expected}、实际 {got}"),
    )
}

#[cfg(test)]
mod tests {
    include!("../../tests/kernel_epilogue_tests.rs");
}
