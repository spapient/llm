//! 注意力 / 解码 / 潜空间注意力与量化 GEMV 的视图适配：几何从视图形状取，结果写进调用方缓冲。

use crate::kernel::{Inputs, View, ViewMut, batches, output_slice};
use diagnostic::Result;

/// 稠密自注意力内核入口：`q/k/v` 是二维只读视图，几何从视图形状取，结果写进调用方缓冲。
type AttentionKernel = fn(
    tensor::View<'_, f32>,
    tensor::View<'_, f32>,
    tensor::View<'_, f32>,
    &mut [f32],
) -> Result<()>;

/// decode 注意力内核入口：`q/k/v` 是二维只读视图，查询/历史行数与头宽从视图形状取。
type DecodeKernel = fn(
    tensor::View<'_, f32>,
    tensor::View<'_, f32>,
    tensor::View<'_, f32>,
    &mut [f32],
) -> Result<()>;

/// 量化 GEMV 适配：`[x: f32, w: u8, scales: f32] → y: f32`。三档位宽同一条形参表，
/// 只差一行算术入口，因此适配只写一份。
macro_rules! quant_gemv {
    ($($run:ident),* $(,)?) => {$(
        pub(crate) fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (x, w, scales) = <(f32, u8, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            nn::$run::forward(
                tensor::View::one(x),
                w,
                tensor::View::one(scales),
                output_slice::<f32>(stringify!($run), output)?,
            )
        }
    )*};
}

quant_gemv!(gemv_i2, gemv_i4, gemv_i8);

/// 自注意力三路适配的公共几何：头宽取声明宽度，`heads` 由 q 长按头宽分段；两版只差算术入口。
fn attention_paths(
    inputs: &[View<'_>],
    output: &mut ViewMut<'_>,
    run: AttentionKernel,
) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "attention_f32")?;
    let out = output_slice::<f32>("attention_f32", output)?;
    let d = nn::conv::attention::MODEL.shape.1;
    let s = batches("attention_f32", q.len(), d)?;
    let (qv, kv, vv) = tensor::View::matrix_triple(s, d, q, k, v)?;
    run(qv, kv, vv, out)
}

/// `[q, k, v: f32] → y: f32`：尺寸由声明宽度推导；算术走稠密分块 tile。
pub(crate) fn attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    attention_paths(inputs, output, nn::conv::attention::forward)
}

/// 同形参，走 online softmax 版（`nn::conv::attention::flash_forward`）。
pub(crate) fn flash_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    attention_paths(inputs, output, nn::conv::attention::flash_forward)
}

/// decode 注意力适配的公共几何：`q_rows` 由 q 长、`hist` 由 k 长按头宽 `d` 推出。
fn decode_paths(inputs: &[View<'_>], output: &mut ViewMut<'_>, run: DecodeKernel) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "decode_attention_f32")?;
    let out = output_slice::<f32>("decode_attention_f32", output)?;
    let d = nn::conv::decode_attention::MODEL.shape.1;
    let (q_rows, hist) = decode_geom(q.len(), k.len(), d)?;
    let (qv, kv, vv) = decode_triple(q, k, v, q_rows, hist, d)?;
    run(qv, kv, vv, out)
}

/// decode 几何：查询行数与历史行数各按头宽反推。
fn decode_geom(q_len: usize, k_len: usize, d: usize) -> Result<(usize, usize)> {
    let q_rows = batches("decode_attention_f32", q_len, d)?;
    let hist = batches("decode_attention_f32", k_len, d)?;
    Ok((q_rows, hist))
}

/// decode 三视图：`q` 是 `q_rows×d`，`k/v` 是 `hist×d`。
fn decode_triple<'a>(
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    q_rows: usize,
    hist: usize,
    d: usize,
) -> Result<(
    tensor::View<'a, f32>,
    tensor::View<'a, f32>,
    tensor::View<'a, f32>,
)> {
    Ok((
        tensor::View::matrix(q_rows, d, q)?,
        tensor::View::matrix(hist, d, k)?,
        tensor::View::matrix(hist, d, v)?,
    ))
}

/// decode 注意力：`q/k/v → out`。
pub(crate) fn decode_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    decode_paths(inputs, output, nn::conv::attention::decode_forward)
}

/// 同 decode 几何，走 online softmax 版 `nn::conv::attention::decode_flash_forward`。
pub(crate) fn decode_flash_attention_f32(
    inputs: &[View<'_>],
    output: &mut ViewMut<'_>,
) -> Result<()> {
    decode_paths(inputs, output, nn::conv::attention::decode_flash_forward)
}

/// 潜空间注意力适配：`[q, k, v: f32] → y: f32`，行宽取声明宽度，输出宽由输出长度反推。
///
/// 与 `attention_f32` 的区别只有输出宽可以更窄（上投影折在口径差里），所以适配层不做任何
/// 头数/压缩比的假设——全部几何都由张量长度与声明宽度给出。
pub(crate) fn latent_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "latent_attention_f32")?;
    let out = output_slice::<f32>("latent_attention_f32", output)?;
    let w = nn::conv::attention::latent::MODEL.shape.1;
    let (r, t) = latent_geom(q.len(), k.len(), w)?;
    let (qv, kv) = row_pair(q, k, r, t, w)?;
    nn::conv::attention::latent_forward(qv, kv, v, out)
}

/// 潜空间几何：查询行数与键行数各按行宽反推。
fn latent_geom(q_len: usize, k_len: usize, w: usize) -> Result<(usize, usize)> {
    let r = batches("latent_attention_f32", q_len, w)?;
    let t = batches("latent_attention_f32", k_len, w)?;
    Ok((r, t))
}

/// 两条同宽行主序视图：`a` 是 `ra×w`、`b` 是 `rb×w`。
fn row_pair<'a>(
    a: &'a [f32],
    b: &'a [f32],
    ra: usize,
    rb: usize,
    w: usize,
) -> Result<(tensor::View<'a, f32>, tensor::View<'a, f32>)> {
    Ok((
        tensor::View::matrix(ra, w, a)?,
        tensor::View::matrix(rb, w, b)?,
    ))
}
