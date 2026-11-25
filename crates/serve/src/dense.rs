//! 稠密、量化与逐元素算子的签名适配；算术与几何语义均由 `nn` 持有。

use super::{SITE, input_f32, output_f32};
use diagnostic::{Code, Error, Result};
use engine::exec::{Launch, View, ViewMut};

/// 取第 `slot` 路输入为打包字节切片；缺失或形态不符给 `Invalid`。
fn input_u8<'a>(name: &str, inputs: &[View<'a>], slot: usize) -> Result<&'a [u8]> {
    inputs.get(slot).and_then(View::u8).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{name}` 第 {slot} 路输入不是字节缓冲"),
        )
        .with(SITE)
    })
}

/// 一元/二元适配共享同一宏：列表只登记入口，拆包与输出检查不重复。
macro_rules! elementwise {
    ($($run:ident),* ; unary) => {$(
        pub(super) fn $run(_: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let input = input_f32(stringify!($run), inputs, 0)?;
            let output = output_f32(stringify!($run), outputs, 0)?;
            nn::$run::forward(tensor::View::one(input), output)
        }
    )*};
    ($($run:ident),* ; binary) => {$(
        pub(super) fn $run(_: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let left = input_f32(stringify!($run), inputs, 0)?;
            let right = input_f32(stringify!($run), inputs, 1)?;
            let output = output_f32(stringify!($run), outputs, 0)?;
            nn::$run::forward(tensor::View::one(left), tensor::View::one(right), output)
        }
    )*};
}

elementwise!(gelu_f32, reduce_max_f32, reduce_sum_f32, scale, silu_f32, softmax_f32; unary);
elementwise!(add_f32, mul_f32, residual_add_f32, embed_f32; binary);

/// RMSNorm 按 `gamma` 行宽处理连续行批，段内校验由 `nn` 统一执行。
pub(super) fn rmsnorm_f32(
    _: Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
) -> Result<()> {
    let x = input_f32("rmsnorm_f32", inputs, 0)?;
    let gamma = input_f32("rmsnorm_f32", inputs, 1)?;
    let out = output_f32("rmsnorm_f32", outputs, 0)?;
    nn::rmsnorm_f32::forward_batch(
        tensor::View::one(x),
        tensor::View::one(gamma),
        out,
        gamma.len(),
    )
}

/// 稠密投影：单行复用 GEMV，行批按 `Launch` 几何选择共享权重的矩阵路径。
pub(super) fn gemv_f32(
    launch: Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
) -> Result<()> {
    let (x, weights, out) = gemv_inputs(inputs, outputs)?;
    let m = launch.block[1] as usize;
    if m == 0 || x.len() == m {
        return nn::gemv_f32::forward(tensor::View::one(x), weights, out);
    }
    project_rows(x, weights, out, m, launch.block[0] as usize)
}

/// 取出投影两个读入与一个写出的 `f32` 视图。
fn gemv_inputs<'a, 'b>(
    inputs: &'b [View<'_>],
    outputs: &'a mut [ViewMut<'_>],
) -> Result<(&'b [f32], &'b [f32], &'a mut [f32])> {
    Ok((
        input_f32("gemv_f32", inputs, 0)?,
        input_f32("gemv_f32", inputs, 1)?,
        output_f32("gemv_f32", outputs, 0)?,
    ))
}

/// 行批投影：输入按 `m` 切行、输出按 `n` 切行，权重视图始终保持 `[n, m]`。
pub(super) fn project_rows(
    x: &[f32],
    weights: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
) -> Result<()> {
    let rows = row_batch_rows(x, out, m, n)?;
    if rows == 1 {
        return nn::gemv_f32::forward(tensor::View::one(x), weights, out);
    }
    project_batch(x, weights, out, m, n, rows)
}

/// 多行矩阵路径：几何视图只在合法行批上构造，按行数选择共享内核。
fn project_batch(
    x: &[f32],
    weights: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    rows: usize,
) -> Result<()> {
    let xv = tensor::View::matrix(rows, m, x)?;
    let wv = tensor::View::matrix(n, m, weights)?;
    if rows >= nn::fast::MIN_ROWS {
        return nn::fast::gemm_f32_tn(xv, wv, out);
    }
    nn::matmul::forward_tn_tiled(xv, wv, out)
}

/// 行批几何校验：输入必须含整数行，输出必须与 `rows × n` 完全一致。
fn row_batch_rows(x: &[f32], out: &[f32], m: usize, n: usize) -> Result<usize> {
    if m == 0 || !x.len().is_multiple_of(m) || out.len() != (x.len() / m) * n {
        return Err(Error::new(Code::Invalid, "gemv 行批几何不自洽").with(super::SITE));
    }
    Ok(x.len() / m)
}

/// 量化 GEMV 的单行与行批函数指针形态。
type GemvQuantFn =
    fn(tensor::View<'_, f32>, &[u8], tensor::View<'_, f32>, &mut [f32]) -> Result<()>;
/// 量化 GEMV 通用桥：单行与行批共用视图拆包，具体算术由两个函数指针注入。
fn gemv_quant(
    launch: Launch,
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    single: GemvQuantFn,
    batch: GemvQuantFn,
) -> Result<()> {
    let m = launch.block[1] as usize;
    if m == 0 || x.len() == m {
        return single(
            tensor::View::one(x),
            weights,
            tensor::View::one(scales),
            out,
        );
    }
    batch(
        tensor::View::matrix(x.len() / m, m, x)?,
        weights,
        tensor::View::one(scales),
        out,
    )
}

/// 量化 GEMV 表项：`x: f32, w: u8, scales: f32 → y: f32` 只写一次。
macro_rules! gemv_quant_kernel {
    ($($run:ident => $batch:path),* $(,)?) => {$(
        pub(super) fn $run(launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let name = stringify!($run);
            let x = input_f32(name, inputs, 0)?;
            let weights = input_u8(name, inputs, 1)?;
            let scales = input_f32(name, inputs, 2)?;
            let out = output_f32(name, outputs, 0)?;
            gemv_quant(launch, x, weights, scales, out, nn::$run::forward, $batch)
        }
    )*};
}

gemv_quant_kernel!(
    gemv_i2 => nn::gemv_i2::forward_batch,
    gemv_i4 => nn::gemv_i4::forward_batch,
    gemv_i8 => nn::gemv_i8::forward_batch,
);

/// 打包反量化与逐元素反量化共享声明宏，样本常量仍由 `nn` 持有。
macro_rules! dequant {
    ($($run:ident),* ; packed) => {$(
        pub(super) fn $run(_: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let q = input_u8(stringify!($run), inputs, 0)?;
            let out = output_f32(stringify!($run), outputs, 0)?;
            nn::$run::forward(nn::$run::SCALE, nn::$run::ZERO, q, out)
        }
    )*};
}

dequant!(dequant_i2, dequant_i4; packed);

/// `i8` 码点以 `f32` 缓冲表示，输入 lane 与打包格式不同。
pub(super) fn dequant_i8(
    _: Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
) -> Result<()> {
    let q = input_f32("dequant_i8", inputs, 0)?;
    let out = output_f32("dequant_i8", outputs, 0)?;
    nn::dequant_i8::forward(
        nn::dequant_i8::SCALE,
        nn::dequant_i8::ZERO,
        tensor::View::one(q),
        out,
    )
}

/// 量化 GEMM 算术入口：两路打包字节与静态几何 `(m, n)` 到浮点输出。
type GemmQuantFn = fn(&[u8], &[u8], usize, usize, &mut [f32]) -> Result<()>;

/// 量化 GEMM 的公共几何推导：`K` 与位宽来自 `nn::gemm::Gemm`。
fn gemm_quant<G: nn::gemm::Gemm<Elem = u8>>(
    a: &[u8],
    b: &[u8],
    out: &mut [f32],
    run: GemmQuantFn,
) -> Result<()> {
    let row = nn::gemm::K / nn::gemm::codes_per_byte(G::BITS);
    run(a, b, a.len() / row, b.len() / row, out)
}

/// 量化 GEMM 表项：标记类型提供位宽，适配器不复制第二条位宽常量。
macro_rules! gemm_quant_kernel {
    ($(($run:ident, $marker:ty)),* $(,)?) => {$(
        pub(super) fn $run(_: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let name = stringify!($run);
            let a = input_u8(name, inputs, 0)?;
            let b = input_u8(name, inputs, 1)?;
            let out = output_f32(name, outputs, 0)?;
            gemm_quant::<$marker>(a, b, out, nn::$run::forward)
        }
    )*};
}

gemm_quant_kernel!(
    (gemm_i2, nn::gemm_i2::GemmI2),
    (gemm_i4, nn::gemm_i4::GemmI4),
    (gemm_i8, nn::gemm_i8::GemmI8),
);
