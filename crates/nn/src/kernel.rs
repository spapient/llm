//! 稠密计算核：矩阵乘族（`gemm` / `gemv` / `matmul`）、激活（`embed`/`rmsnorm`/`rope`/`softmax`）、归一化（`norm`）、
//! 反量化（`dequant`）、逐元素（`elementwise` / `unary`）、归约（`reduce`）与 SIMD/SME 通道（`fast`）。

// 不稳定特性一律挂在 `cfg(nightly)` 上（由本 crate 的 `build.rs` 按 `rustc --version` 判定）：
// nightly 上走手写车道，stable 上走 `gemv::simd` 的标量退路，两边的能力差异只在
// 「向量化没向量化」，不在语义。这样 `rust-toolchain.toml` 不必把整个工作区钉在某个 nightly。
//
// 本 crate 只用到 `portable_simd`（`std::simd`，用在 `gemv::simd` 的解包点积）。它**不按目标门控**：
// 架构无关的 16 车道源码，aarch64 上落成 NEON、x86_64 上落成 SSE2 基线，两个目标编译同一份算术。
// SME 与 i8 dotprod 通道本体已迁入 `prof::native`，各自的 `sme` / `dotprod` `target_feature` 与
// 运行时探测门在 `prof` 的 `lib.rs` 按 nightly + aarch64 声明；本 crate 经 `cutile::native`
// 转发调用（nn → cutile → prof），不再持有这些门，也不直接 import `prof` 的内核层。

pub mod clamp {
    //! 逐元素裁剪：`y[n] = clamp(x[n], lo, hi)`；`lo` / `hi` 取模块常量（f32 标量入参通道未贯通，同 `dequant_i8` 口径）。

    use crate::model::{Buf, Cost, Sample, ShapeCtx, cost_n, x_buf};
    use diagnostic::{Code, Error};
    use tensor::View;

    /// 单轴定义与形状投影复用 unary 骨架：本文件只声明 clamp 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 样本固定的下界（内核模板里写同一常量）。
    pub const LO: f32 = -1.0;
    /// 样本固定的上界（内核模板里写同一常量）。
    pub const HI: f32 = 1.0;

    /// 标量代价：每元素两次比较（下界、上界），读一次写一次。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "clamp", 2.0, 1)
    }

    /// 确定性样本：`x` 夹具、`y` 零缓冲；`lo` / `hi` 取模块常量（签名通道只支持整数标量）。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![x_buf(len), Buf::f32("y", vec![0.0; len])],
        }
    }

    /// 标量界前置：两界有限且下界不大于上界。
    ///
    /// # Errors
    ///
    /// 任一界非有限，或下界大于上界时返回说明。
    fn check_bounds(lo: f32, hi: f32) -> diagnostic::Result<()> {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(Error::new(Code::Invalid, "clamp 界必须是有限值"));
        }
        if lo > hi {
            return Err(Error::new(
                Code::Invalid,
                format!("clamp 下界 {lo} 大于上界 {hi}"),
            ));
        }
        Ok(())
    }

    /// 标量参考：逐元素裁剪，与内核同一公式，界取模块常量。
    ///
    /// # Errors
    ///
    /// 样本缺 `x`，或界非法时返回说明。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        check_bounds(LO, HI)?;
        Ok(x.iter().map(|value| value.clamp(LO, HI)).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// `x` 与 `out` 不等长，或界非法时返回说明。
    pub fn forward(lo: f32, hi: f32, x: View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
        let x = x.as_slice();
        check_bounds(lo, hi)?;
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        for (place, value) in out.iter_mut().zip(x) {
            *place = value.clamp(lo, hi);
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Clamp;

    /// clamp 的具体语义别名。
    #[annotation::model(name = "clamp_f32", shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
    pub type ClampF32 = Clamp;

    // 基座契约（ADR 0050）：本模块自带具体入口，直接转发；图输入只有 `x`，`lo` / `hi` 取模块常量。
    crate::operator_here!(Clamp, "clamp_f32", 1, (4096, 4096), false, reference);
}
pub mod dequant;
pub mod elementwise;
pub mod embed {
    //! Embedding 行查表：`y[n, :] = table[idx[n], :]`（行向量 gather）。
    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx};
    use diagnostic::{Code, Error};
    use tensor::View;

    /// 权重表行数（闭集内的固定档）。
    pub const VOCAB: usize = 64;

    /// 两条基本轴：token 轴 `n`（输入）、行宽轴 `d`（输出）。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'n',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'd',
        },
    ];

    /// `(m, n)` 即 `(token 数 n, 行宽 d)`。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(tokens: usize, dim: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('n', tokens), ('d', dim)])
    }

    /// 标量代价：每个 token 搬一行 `D`（读表一行 + 写输出一行），下标转换按 1 次/token 估算。
    ///
    /// 常驻表按被访问的行数计价，不把整张词表读一遍（表流量口径属于代价模型，层不重复推）。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        let n = shape
            .get('n')
            .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `n`"))?;
        let d = shape
            .get('d')
            .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `d`"))?;
        let elements = (n as u64) * (d as u64);
        Ok(Cost {
            flops: n as f64,
            read: elements * 4,
            write: elements * 4,
        })
    }

    /// 确定性样本：`idx`（f32 承载的整数下标，长度 n）、`table`（长度 `V·D`）、`y`（长度 `n·D`）。
    pub fn sample(tokens: usize, dim: usize) -> Sample {
        let table = (0..VOCAB * dim)
            .map(|idx| (idx % VOCAB) as f32 * 0.5)
            .collect();
        Sample {
            m: tokens,
            n: dim,
            scalars: vec![("N", tokens as i64), ("V", VOCAB as i64), ("D", dim as i64)],
            bufs: vec![
                Buf::f32("idx", (0..tokens).map(|i| (i % VOCAB) as f32).collect()),
                Buf::f32("table", table),
                Buf::f32("y", vec![0.0; tokens * dim]),
            ],
        }
    }

    /// 标量参考：与内核同一行 gather 公式。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let idx = sample.f32("idx")?;
        let table = sample.f32("table")?;
        gather(idx, table, sample.n)
    }

    /// 校验行宽、表长和全部索引；通过后才允许参考物化或内核写入。
    fn validate_table(idx: &[f32], table: &[f32], dim: usize) -> diagnostic::Result<usize> {
        if dim == 0 {
            return Err(Error::new(Code::Invalid, "embed 行宽不能为零"));
        }
        if !table.len().is_multiple_of(dim) {
            return Err(Error::new(
                Code::Invalid,
                format!("embedding 表长 {} 不是行宽 {dim} 的整数倍", table.len()),
            ));
        }
        let rows = table.len() / dim;
        for &value in idx {
            let idx = value as usize;
            if idx >= rows {
                return Err(Error::new(
                    Code::Invalid,
                    format!("embedding 下标 {idx} 越界（词表 {rows}）"),
                ));
            }
        }
        Ok(rows)
    }

    /// 共享行查表：参考与内核同一套校验和索引语义。
    fn gather(idx: &[f32], table: &[f32], dim: usize) -> diagnostic::Result<Vec<f32>> {
        validate_table(idx, table, dim)?;
        let mut output = Vec::with_capacity(idx.len() * dim);
        for &value in idx {
            let start = value as usize * dim;
            output.extend_from_slice(&table[start..start + dim]);
        }
        Ok(output)
    }

    /// 内核本体：结果写进调用方给的缓冲，不分配临时输出。
    ///
    /// 行宽 `D` 由输出长度与 token 数推出：`out.len() == idx.len()·D`；表长必须是 `D` 的整数倍。
    ///
    /// # Errors
    ///
    /// 缓冲长度不自洽、表长不符或下标越界时返回说明。
    pub fn forward(
        idx: View<'_, f32>,
        table: View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let idx = idx.as_slice();
        let table = table.as_slice();
        let tokens = idx.len();
        if tokens == 0 {
            return Err(Error::new(Code::Invalid, "embed 索引为空"));
        }
        if !out.len().is_multiple_of(tokens) {
            return Err(Error::new(
                Code::Invalid,
                format!("输出长度 {} 不能按 token 数 {tokens} 整除出行宽", out.len()),
            ));
        }
        let dim = out.len() / tokens;
        validate_table(idx, table, dim)?;
        for (&value, output) in idx.iter().zip(out.chunks_exact_mut(dim)) {
            let start = value as usize * dim;
            output.copy_from_slice(&table[start..start + dim]);
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Embed;

    /// Embedding 的具体语义别名。
    #[annotation::model(name = "embed_f32", shape = [8, 64], dtype = "f32", skeleton = "embed")]
    pub type EmbedF32 = Embed;

    crate::operator_here!(Embed, "embed_f32", 2, (8, 64), true, reference);
}
pub mod epilogue;
pub mod fast;
pub mod gemm;
pub mod gemv;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod rmsnorm {
    //! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
    use crate::model::{Buf, Cost, Sample, ShapeCtx, cost_n, x_buf};
    use diagnostic::{Code, Error};
    use tensor::View;

    /// 数值稳定项：均方根里加的小常数。
    pub const EPS: f32 = 1e-6;

    // 单轴定义与形状投影复用 unary 骨架：本文件只声明 rmsnorm 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：两趟读 `x`、一趟读 `gamma`、一趟写 `y`，运算按每元素约 4 次（平方、加法、
    /// 除法、乘法）估算。这里给出可审计的显式口径，不伪装成一条 einsum。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "rmsnorm", 4.0, 3)
    }

    /// 确定性样本：`x`、`gamma`、`y` 三个等长缓冲。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                x_buf(len),
                Buf::f32("gamma", crate::model::linear(len, 0.01, 0.5)),
                Buf::f32("y", vec![0.0; len]),
            ],
        }
    }

    /// 公共前置：输入与权重等长且非空；输出长度由对应入口单独核对。
    fn check(x: &[f32], gamma: &[f32]) -> diagnostic::Result<()> {
        if x.len() != gamma.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x 与 gamma 长度不一致：{} 与 {}", x.len(), gamma.len()),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "rmsnorm 输入不能为空：均方根无定义",
            ));
        }
        Ok(())
    }

    /// 归一化第一趟：平方和。累加是串行依赖，保留逐元素循环；参考与内核共用同一份口径。
    fn sum_sq(x: &[f32]) -> f32 {
        let mut sum = 0.0;
        for value in x {
            sum += value * value;
        }
        sum
    }

    /// 校验数值稳定项；有限且非负才允许进入参考或内核。
    fn validate_epsilon(epsilon: f32) -> diagnostic::Result<()> {
        if epsilon.is_finite() && epsilon >= 0.0 {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            "rmsnorm epsilon 必须是有限非负数",
        ))
    }

    /// 一次平方根和倒数，供整行乘法复用，避免每元素一次除法。
    fn inverse_rms(x: &[f32], epsilon: f32) -> f32 {
        (sum_sq(x) / (x.len() as f32) + epsilon).sqrt().recip()
    }

    /// 标量参考：两趟，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 输入为空或权重长度不符时返回说明。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        check(x, gamma)?;
        let scale = inverse_rms(x, EPS);
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(value, weight)| weight * value * scale)
            .collect())
    }

    /// 单行数值核：调用方已完成校验；批路径复用它以避免逐行重复检查。
    fn normalize(x: &[f32], gamma: &[f32], out: &mut [f32], epsilon: f32) {
        let scale = inverse_rms(x, epsilon);
        for ((out, x), gamma) in out.iter_mut().zip(x).zip(gamma) {
            *out = gamma * x * scale;
        }
    }

    /// 使用默认 [`EPS`] 的内核入口。
    ///
    /// # Errors
    ///
    /// 三个缓冲不等长或输入为空时返回说明。
    pub fn forward(
        x: View<'_, f32>,
        gamma: View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        forward_with_epsilon(x, gamma, out, EPS)
    }

    /// 可配置 epsilon 的内核入口；全部校验完成后才开始写入。
    ///
    /// # Errors
    ///
    /// 三个缓冲不等长、输入为空或 epsilon 非法时返回说明。
    pub fn forward_with_epsilon(
        x: View<'_, f32>,
        gamma: View<'_, f32>,
        out: &mut [f32],
        epsilon: f32,
    ) -> diagnostic::Result<()> {
        validate_epsilon(epsilon)?;
        let x = x.as_slice();
        let gamma = gamma.as_slice();
        check(x, gamma)?;
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
            ));
        }
        normalize(x, gamma, out, epsilon);
        Ok(())
    }

    /// 批前置：只在本入口校验一次，行宽与非空约束覆盖所有后续分段。
    fn check_batch(x: &[f32], gamma: &[f32], out: &[f32], row_dim: usize) -> Result<(), Error> {
        check_row_config(gamma, row_dim)?;
        check_batch_bufs(x, out, row_dim)
    }

    /// 行配置只需校验一次，与整批输入输出缓冲判据分离。
    fn check_row_config(gamma: &[f32], row_dim: usize) -> Result<(), Error> {
        if row_dim == 0 {
            return Err(Error::new(Code::Invalid, "rmsnorm 行宽不能为 0"));
        }
        if gamma.len() != row_dim {
            return Err(Error::new(
                Code::Invalid,
                format!("gamma 长度 {} 不等于行宽 {row_dim}", gamma.len()),
            ));
        }
        Ok(())
    }

    /// 批缓冲必须等长、非空且能按行宽完整分段。
    fn check_batch_bufs(x: &[f32], out: &[f32], row_dim: usize) -> Result<(), Error> {
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
            ));
        }
        if x.is_empty() || !x.len().is_multiple_of(row_dim) {
            return Err(Error::new(Code::Invalid, "rmsnorm 批缓冲不是行宽整数倍"));
        }
        Ok(())
    }

    /// 使用默认 [`EPS`] 的批量入口。
    ///
    /// # Errors
    ///
    /// 行宽、缓冲或默认配置不合法时返回说明。
    pub fn forward_batch(
        x: View<'_, f32>,
        gamma: View<'_, f32>,
        out: &mut [f32],
        row_dim: usize,
    ) -> diagnostic::Result<()> {
        forward_batch_with_epsilon(x, gamma, out, row_dim, EPS)
    }

    /// 可配置 epsilon 的批量入口；每段独立归一化并复用 gamma。
    ///
    /// # Errors
    ///
    /// 行宽为 0、gamma 与行宽不等长、缓冲不完整或 epsilon 非法时返回说明。
    pub fn forward_batch_with_epsilon(
        x: View<'_, f32>,
        gamma: View<'_, f32>,
        out: &mut [f32],
        row_dim: usize,
        epsilon: f32,
    ) -> diagnostic::Result<()> {
        validate_epsilon(epsilon)?;
        let x = x.as_slice();
        let gamma = gamma.as_slice();
        check_batch(x, gamma, out, row_dim)?;
        for (x, out) in x.chunks_exact(row_dim).zip(out.chunks_exact_mut(row_dim)) {
            normalize(x, gamma, out, epsilon);
        }
        Ok(())
    }

    /// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
    #[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
    pub type RmsNormF32 = RmsNorm;

    crate::operator_here!(RmsNorm, "rmsnorm_f32", 2, (4096, 4096), true, reference);

    /// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
    pub struct RmsNorm;
}
pub mod rope;
pub mod scalar;
pub mod softmax {
    //! Softmax：`y[n] = exp(x[n] - max(x)) / sum_k exp(x[k] - max(x))`。
    use crate::model::{Cost, Sample, ShapeCtx, cost_n};
    use diagnostic::{Code, Error};
    use tensor::View;

    // 单轴定义与形状投影复用 unary 骨架：本文件只声明 softmax 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：三趟读 `x`、一趟写 `y`，每元素约 3 次运算（max、exp、除法）。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "softmax", 3.0, 3)
    }

    /// 确定性样本：`x`、`y` 两个等长缓冲。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample::f32_io(len, len, len, len, vec![("N", len as i64)])
    }

    /// 标量参考：三趟，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 输入为空时返回说明：空行没有可归一化的分母。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "softmax 输入不能为空：归一化分母无定义",
            ));
        }
        let max = x.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        let exps: Vec<f32> = x
            .iter()
            .map(|value| {
                let e = (value - max).exp();
                sum += e;
                e
            })
            .collect();
        Ok(exps.iter().map(|e| e / sum).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出等长，或输入为空时返回说明：空行没有可归一化的分母。
    pub fn forward(x: View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
        let x = x.as_slice();
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "softmax 输入不能为空：归一化分母无定义",
            ));
        }
        let max = x.iter().copied().fold(f32::MIN, f32::max);
        // 一趟 `exp`：先用写入的指数值累加分母，再原地除掉。旧实现算了两遍 `exp`
        // （一遍求 `sum`、一遍求结果），而 `exp` 正是这里最贵的指令；`(x-max).exp()`
        // 的取值与从左到右的累加顺序都和旧实现一致，所以输出逐位不变。
        let mut sum = 0.0;
        for (out, x) in out.iter_mut().zip(x) {
            let e = (x - max).exp();
            sum += e;
            *out = e;
        }
        for out in out.iter_mut() {
            *out /= sum;
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Softmax;

    /// Softmax 的具体语义别名。
    #[annotation::model(name = "softmax_f32", shape = [4096, 4096], dtype = "f32", skeleton = "softmax")]
    pub type SoftmaxF32 = Softmax;

    crate::operator_here!(Softmax, "softmax_f32", 1, (4096, 4096), true, reference);
}
pub mod unary;

// 一元 / 二元 / 归约 / 反量化 / GEMV / GEMM 骨架内的声明式子模块，re-export 到 crate 根
// 保持 `<crate>::<name>::MODEL` / `forward` 路径。
pub use clamp as clamp_f32;
pub use dequant::{dequant_i2, dequant_i4, i8 as dequant_i8};
pub use elementwise::{
    add_f32, add_f32x4, div_f32, max_f32, min_f32, mul_f32, remainder_f32, residual_add_f32,
    sub_f32,
};
pub use epilogue::apply as apply_epilogue;
pub use gemm::{f32 as gemm_f32, i2 as gemm_i2, i4 as gemm_i4, i8 as gemm_i8};
pub use gemv::{f32 as gemv_f32, i2 as gemv_i2, i4 as gemv_i4, i8 as gemv_i8};
pub use matmul::i8 as matmul_i8;
pub use reduce::{reduce_max_f32, reduce_prod_f32, reduce_sum_f32};
pub use scalar::Scalar;
pub use unary::abs_f16;
pub use unary::ops::{
    abs_f32, ceil_f32, cos_f32, exp_f32, exp2_f32, floor_f32, gelu_f32, hard_sigmoid_f32,
    leaky_relu_f32, log_f32, log1p_f32, neg_f32, recip_f32, relu_f32, round_f32, rsqrt_f32,
    scale_f32, sigmoid_f32, sign_f32, silu_f32, sin_f32, sqrt_f32, square_f32, tanh_f32, trunc_f32,
};
pub use {embed as embed_f32, rmsnorm as rmsnorm_f32, rope as rope_f32, softmax as softmax_f32};

// 契约层在 `prof`：内核与本 crate 的骨架宏都按 crate 根解析契约路径，这里把它们搬回来，
// 于是内核源码里写的是 `crate::model::…` / `crate::operator_via!`，而不是到处写依赖 crate 名。
pub use prof::same_bytes;
// `$crate::formula_cell!` / `$crate::operator_via!` 在骨架宏里按 crate 根解析，这里把 spec 导出的
// 宏搬回本 crate 根：内核里几十处 `#[model]` 展开都不必改路径。
pub use model::quant;
pub use prof::tpl::formula;
pub use prof::{einsum, model, operator, sample, sch, sig, tpl};
pub use prof::{formula_cell, operator_here, operator_via};
pub use sch::layout;

#[cfg(test)]
#[path = "../tests/kernel_prop_numeric.rs"]
mod prop_numeric;

#[cfg(test)]
mod pointwise_tests {
    include!("../tests/kernel_pointwise_tests.rs");
}
