//! CPU 推理内核桥：把 `op` 的标量参考适配成 engine 可登记的 [`Program`]。
//!
//! 落点裁决见 `docs/adr/0025-runtime-backend-abstraction.md`；设备数据与机制的边界见
//! `docs/ARCHITECTURE.md` 第一、二节。
//! 端点契约、路由与线格式保持模型无关；CPU 模型实现作为注入 [`Predict`](crate::Predict)
//! 的提供者收在本模块内，真设备后端仍收在独立边界 crate，不回流这里。
//!
//! # 职责
//!
//! - 编译期锁定的算子表：表内每个名字都必须在 `op::REGISTRY` 内（`const` 子集断言）；
//!   表未覆盖的算子不登记，执行时由 [`CpuExecutor`](engine::exec::CpuExecutor) 按名拒绝，
//!   不留静默空入口。
//! - 适配函数只做拆视图 + 调 `op::<op>::forward`，算术不重写。模式复刻 `train::KERNELS`
//!   （`crates/train/src/kernel.rs`），但不带 train 的 `Lane` 适配层：缓冲形态用 engine
//!   自己的 [`View`] / [`ViewMut`]，f32 与打包字节两路分开取切片。
//! - 层栈装配、驻留参数表与前向宿主收在 [`assembly`](crate::assembly) 模块，本文件只留内核适配表。
//!
//! # 不拥有
//!
//! 执行/调度/KV 簿记（`engine`）、算子声明与参考算术（`op`）、层形状语义（`nn`）、
//! 权重字节读取（`weights`）。本模块不出现设备厂商或后端类型的特判分支。
//!
//! # 范围
//!
//! 接纯切片签名的 f32 算子，以及 i4/i8 量化 lane（`gemv_i4`/`gemv_i8`/`dequant_i4`/
//! `dequant_i8`）；量化权重文件侧的半精度与零点解码仍留后，本轮量化通路以内存夹具对拍
//! 验证。`rope_f32` 单序列位置走标量槽回填；批形态逐序列位置表走独立输入缓冲（ADR 0043 §5）。
//! decode 变长历史段紧排布局已定（ADR 0046 §7，op `*_batch_var` 入口就位），serve 批装填 / 写回
//! 随块链整批刀落地。

use diagnostic::{Code, Error, Result};
use engine::exec::{Executor, Launch, Program, View, ViewMut};

pub(crate) const SITE: &str = "serve::model";

/// 注意力内核入口：`q/k/v → out`，序列长 `s` 与头宽 `d` 随几何传入。
type AttentionKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize) -> Result<()>;

/// decode 注意力内核入口：`q/k/v → out`，查询行数 `q_rows`、历史行数 `hist` 与头宽 `d` 随几何传入。
type DecodeKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()>;

/// decode 批段内核入口：`q/k/v → out`，`hist` 为逐序列段长表（ragged 紧排），头数/头宽静态传入。
type BatchDecodeKernel =
    fn(&[f32], &[f32], &[f32], &mut [f32], usize, &[usize], usize, usize) -> Result<()>;

/// decode 拆出的三路读入与一路写出视图。
type DecodeViews<'a, 'b> = (&'a [f32], &'a [f32], &'a [f32], &'b mut [f32]);

/// 适配入口：拆好视图后调用算子参考实现；launch 承载逐步变化的标量槽（如 rope 位置）。
type KernelFn = fn(Launch, &[View<'_>], &mut [ViewMut<'_>]) -> Result<()>;

/// 取第 `slot` 路输入为 `f32` 切片；编号缺失或形态不符给 `Invalid`。
fn input_f32<'a>(name: &str, inputs: &[View<'a>], slot: usize) -> Result<&'a [f32]> {
    inputs.get(slot).and_then(View::f32).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{name}` 第 {slot} 路输入不是浮点缓冲"),
        )
        .with(SITE)
    })
}

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

/// 取第 `slot` 路输出为可写 `f32` 切片；缺失或形态不符给 `Invalid`。
fn output_f32<'a>(
    name: &str,
    outputs: &'a mut [ViewMut<'_>],
    slot: usize,
) -> Result<&'a mut [f32]> {
    outputs
        .get_mut(slot)
        .and_then(ViewMut::f32_mut)
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("`{name}` 第 {slot} 路输出不是浮点缓冲"),
            )
            .with(SITE)
        })
}

/// 表内一行：算子名（与 `op::REGISTRY` 同名）+ 缓冲级函数指针。
struct Entry {
    op: &'static str,
    run: KernelFn,
}

/// 算子适配宏：按 arity 分支展开拆切片 + 调 `op::<op>::forward`。
macro_rules! kernel {
    ($($run:ident),* ; unary) => {$(
        fn $run(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let input = input_f32(stringify!($run), inputs, 0)?;
            let output = output_f32(stringify!($run), outputs, 0)?;
            op::$run::forward(input, output)
        }
    )*};
    ($($run:ident),* ; binary) => {$(
        fn $run(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let left = input_f32(stringify!($run), inputs, 0)?;
            let right = input_f32(stringify!($run), inputs, 1)?;
            let output = output_f32(stringify!($run), outputs, 0)?;
            op::$run::forward(left, right, output)
        }
    )*};
}

kernel!(
    gelu_f32,
    reduce_max_f32,
    reduce_sum_f32,
    scale_f32,
    silu_f32,
    softmax_f32; unary
);
kernel!(
    add_f32,
    mul_f32,
    residual_add_f32,
    embed_f32; binary
);

/// rmsnorm 适配：行宽取 `gamma` 长度；整块 `[batch, row_dim]` 段连续，段内独立 RMS 归一化，
/// `gamma` 按行宽复用；单序列 `batch=1` 退化为单行 `forward`。段内校验交 [`forward_batch`]。
fn rmsnorm_f32(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let x = input_f32("rmsnorm_f32", inputs, 0)?;
    let gamma = input_f32("rmsnorm_f32", inputs, 1)?;
    let out = output_f32("rmsnorm_f32", outputs, 0)?;
    op::rmsnorm_f32::forward_batch(x, gamma, out, gamma.len())
}

/// 注意力三路适配：`q/k/v → out`；头宽 `d` 取算子静态语义表，序列长 `s` 由 `q` 长推出，
/// 形状不符交由算子参考校验。朴素与 Flash 同几何，只换内核算式。
fn attention_paths(
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
    run: AttentionKernel,
) -> Result<()> {
    let q = input_f32("attention", inputs, 0)?;
    let k = input_f32("attention", inputs, 1)?;
    let v = input_f32("attention", inputs, 2)?;
    let out = output_f32("attention", outputs, 0)?;
    let d = op::attention::MODEL.shape.1;
    let s = q.len() / d;
    run(q, k, v, out, s, d)
}

/// 朴素自注意力适配：走稠密分块 tile 路径。
fn attention_f32(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    attention_paths(inputs, outputs, op::attention::tiled::forward)
}

/// prefill 自注意力适配：走分块 tile 路径。
fn flash_attention_f32(
    _launch: Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
) -> Result<()> {
    attention_paths(inputs, outputs, op::attention::tiled::flash_tiled_forward)
}

/// decode 三路适配：`q/hist_k/hist_v → out`；第四路输入为 ragged 段长表（ADR 0046 §8）走批段
/// 入口，缺省回退单序列 heads 入口。头数/头宽静态取 Launch 几何，段长表折 usize，形状不符交算子校验。
fn decode_paths(
    launch: &Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
    single: DecodeKernel,
    batch: BatchDecodeKernel,
) -> Result<()> {
    let (q, k, v, out) = decode_views(inputs, outputs)?;
    match inputs
        .get(3)
        .and_then(View::f32)
        .filter(|table| !table.is_empty())
    {
        Some(hist_table) => decode_batch(launch, q, k, v, out, hist_table, batch),
        None => decode_single(q, k, v, out, single),
    }
}

/// 拆出 decode 三路读入与一路写出视图；编号缺失或形态不符交输入助手报 `Invalid`。
fn decode_views<'a, 'b>(
    inputs: &'a [View<'a>],
    outputs: &'b mut [ViewMut<'_>],
) -> Result<DecodeViews<'a, 'b>> {
    let q = input_f32("decode_attention", inputs, 0)?;
    let k = input_f32("decode_attention", inputs, 1)?;
    let v = input_f32("decode_attention", inputs, 2)?;
    let out = output_f32("decode_attention", outputs, 0)?;
    Ok((q, k, v, out))
}

/// 单序列路径：头宽取算子静态语义表，头数/查询行数/历史行数由缓冲长度反推。
fn decode_single(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    single: DecodeKernel,
) -> Result<()> {
    let d = op::decode_attention::MODEL.shape.1;
    let heads = q.len() / d;
    let q_rows = q.len() / (heads * d);
    let hist = k.len() / (heads * d);
    single(q, k, v, out, q_rows, hist, d)
}

/// 批段路径：头数/头宽取 Launch 静态几何，段长表逐序列给历史行数；查询行数由 q 总长反推。
fn decode_batch(
    launch: &Launch,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    hist_table: &[f32],
    batch: BatchDecodeKernel,
) -> Result<()> {
    let head_dim = launch.block[1] as usize;
    if head_dim == 0 {
        return Err(Error::new(Code::Invalid, "decode 批段头宽为 0").with(SITE));
    }
    let heads = launch.block[0] as usize / head_dim;
    let hist: Vec<usize> = hist_table.iter().map(|&h| h as usize).collect();
    if hist.is_empty() {
        return Err(Error::new(Code::Invalid, "decode 批段长表为空").with(SITE));
    }
    let q_rows = q.len() / (hist.len() * heads * head_dim);
    batch(q, k, v, out, q_rows, &hist, head_dim, heads)
}

fn decode_attention_f32(
    launch: Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
) -> Result<()> {
    decode_paths(
        &launch,
        inputs,
        outputs,
        op::attention::decode_forward_heads,
        op::attention::decode_forward_batch_var,
    )
}

fn decode_flash_attention_f32(
    launch: Launch,
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
) -> Result<()> {
    decode_paths(
        &launch,
        inputs,
        outputs,
        op::attention::decode_flash_forward_heads,
        op::attention::decode_flash_forward_batch_var,
    )
}

/// 稠密投影适配：单向量走原 gemv；行批整块按输入宽 `m`（Launch 第二维）切段，逐行共享同一份
/// 驻留权重调 gemv。`m=0` 是装配期占位，退回单向量几何（长度自洽交由内核校验）。
fn gemv_f32(launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let (x, weights, out) = gemv_inputs(inputs, outputs)?;
    let m = launch.block[1] as usize;
    if m == 0 || x.len() == m {
        return op::gemv_f32::forward(x, weights, out);
    }
    project_rows(x, weights, out, m, launch.block[0] as usize)
}

/// 取出投影两个读入与一个写出的 f32 视图。
fn gemv_inputs<'a, 'b>(
    inputs: &'b [View<'_>],
    outputs: &'a mut [ViewMut<'_>],
) -> Result<(&'b [f32], &'b [f32], &'a mut [f32])> {
    let x = input_f32("gemv_f32", inputs, 0)?;
    let weights = input_f32("gemv_f32", inputs, 1)?;
    let out = output_f32("gemv_f32", outputs, 0)?;
    Ok((x, weights, out))
}

/// 行批投影：整块 `x` 按 `m` 切段、`out` 按 `n` 切段。
fn project_rows(x: &[f32], weights: &[f32], out: &mut [f32], m: usize, n: usize) -> Result<()> {
    let rows = row_batch_rows(x, out, m, n)?;
    if rows == 1 {
        return op::gemv_f32::forward(x, weights, out);
    }
    if rows >= op::fast::MIN_ROWS {
        op::fast::gemm_f32_tn(x, weights, rows, n, m, out);
        return Ok(());
    }
    op::matmul::forward_tn_tiled(x, weights, rows, n, m, out)
}

/// 行批几何校验并给出输出行数：`x` 是 `rows × m`，`out` 必须是 `rows × n`。
fn row_batch_rows(x: &[f32], out: &[f32], m: usize, n: usize) -> Result<usize> {
    if m == 0 || !x.len().is_multiple_of(m) || out.len() != (x.len() / m) * n {
        return Err(Error::new(Code::Invalid, "gemv 行批几何不自洽").with(SITE));
    }
    Ok(x.len() / m)
}

/// 量化 GEMV 单行算术入口：一个输入向量 → 一个输出向量。
type GemvQuantFn = fn(x: &[f32], w: &[u8], s: &[f32], out: &mut [f32]) -> Result<()>;

/// 量化 GEMV 行批算术入口：`rows × m` 的输入 → `rows × n` 的输出。
type GemvQuantBatchFn =
    fn(x: &[f32], w: &[u8], s: &[f32], geo: op::gemv::Geometry, out: &mut [f32]) -> Result<()>;

/// `[x: f32, w: u8, scales: f32] → y: f32`：量化 GEMV 通用桥接，按批行数分流。
fn gemv_quant(
    launch: Launch,
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
    single: GemvQuantFn,
    batch: GemvQuantBatchFn,
) -> Result<()> {
    let m = launch.block[1] as usize;
    if m == 0 || x.len() == m {
        return single(x, weights, scales, out);
    }
    batch(
        x,
        weights,
        scales,
        op::gemv::Geometry::new(x.len() / m, m, launch.block[0] as usize),
        out,
    )
}

/// 量化 GEMV 桥接适配：第 0 路 x、第 1 路打包权重、第 2 路标度、第 0 路输出。
macro_rules! gemv_quant_kernel {
    ($($run:ident => $batch:path),*) => {$(
        fn $run(launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let name = stringify!($run);
            let x = input_f32(name, inputs, 0)?;
            let weights = input_u8(name, inputs, 1)?;
            let scales = input_f32(name, inputs, 2)?;
            let out = output_f32(name, outputs, 0)?;
            gemv_quant(launch, x, weights, scales, out, op::$run::forward, $batch)
        }
    )*};
}

gemv_quant_kernel!(
    gemv_i2 => op::gemv_i2::forward_batch,
    gemv_i4 => op::gemv_i4::forward_batch,
    gemv_i8 => op::gemv_i8::forward_batch
);

/// 反量化桥接：打包字节 lane（i2/i4）输入 u8；i8 逐元素 lane 输入 f32。
macro_rules! dequant_packed {
    ($($run:ident),*) => {$(
        fn $run(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let name = stringify!($run);
            let q = input_u8(name, inputs, 0)?;
            let out = output_f32(name, outputs, 0)?;
            op::$run::forward(op::$run::SCALE, op::$run::ZERO, q, out)
        }
    )*};
}

dequant_packed!(dequant_i2, dequant_i4);

/// `[q: f32] → y: f32`：i8 逐元素反量化，scale/zero 取样本常量。
fn dequant_i8(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let q = input_f32("dequant_i8", inputs, 0)?;
    let out = output_f32("dequant_i8", outputs, 0)?;
    op::dequant_i8::forward(op::dequant_i8::SCALE, op::dequant_i8::ZERO, q, out)
}

/// 量化 GEMM 算术入口：两路打包字节 + 静态几何 `(m, n)` → 浮点输出。
type GemmQuantFn = fn(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) -> Result<()>;

/// 量化 GEMM 通用桥接：`A: u8, B: u8 → Y: f32`，几何从静态收缩轴 `K` 与缓冲长度反推。
fn gemm_quant<G: op::gemm::Gemm<Elem = u8>>(
    a: &[u8],
    b: &[u8],
    out: &mut [f32],
    run: GemmQuantFn,
) -> Result<()> {
    let cpb = op::gemm::codes_per_byte(G::BITS);
    let row = op::gemm::K / cpb;
    let m = a.len() / row;
    let n = b.len() / row;
    run(a, b, m, n, out)
}

/// 量化 GEMM 桥接适配：两路打包字节输入、一路浮点输出。
macro_rules! gemm_quant_kernel {
    ($(($run:ident, $marker:ty)),*) => {$(
        fn $run(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
            let name = stringify!($run);
            let a = input_u8(name, inputs, 0)?;
            let b = input_u8(name, inputs, 1)?;
            let out = output_f32(name, outputs, 0)?;
            gemm_quant::<$marker>(a, b, out, op::$run::forward)
        }
    )*};
}

gemm_quant_kernel!(
    (gemm_i2, op::gemm_i2::GemmI2),
    (gemm_i4, op::gemm_i4::GemmI4),
    (gemm_i8, op::gemm_i8::GemmI8)
);

/// 按头宽选 RoPE 入口：头宽 0 或等于整段（单头）走单头旋转，否则按头分段重计频率。
fn run_rope(pos: usize, x: &[f32], out: &mut [f32], head_dim: usize) -> Result<()> {
    if head_dim == 0 || head_dim == x.len() {
        op::rope_f32::forward(pos, x, out)
    } else {
        op::rope_f32::forward_heads(pos, x, out, head_dim)
    }
}

/// RoPE 适配：单序列位置走标量槽（ADR 0043 §1）；批形态第二路输入为逐序列位置表
/// （ADR 0043 §5），元素折 f32、内核 `as usize`，批节点绑表后不再读标量槽。头宽取 Launch
/// 几何第二维，单头走单头旋转，多头按头宽分段重计频率。
fn rope_f32(launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let x = input_f32("rope_f32", inputs, 0)?;
    let out = output_f32("rope_f32", outputs, 0)?;
    let head_dim = launch.block[1] as usize;
    match inputs
        .get(1)
        .and_then(View::f32)
        .filter(|table| !table.is_empty())
    {
        Some(table) => rope_batch(table, x, out, head_dim),
        None => rope_single(&launch, x, out, head_dim),
    }
}

/// 批位置表通道：`positions[b]` 折 usize，段内复用多头旋转编排。
fn rope_batch(table: &[f32], x: &[f32], out: &mut [f32], head_dim: usize) -> Result<()> {
    let positions: Vec<usize> = table.iter().map(|&pos| pos as usize).collect();
    op::rope_f32::forward_batch(&positions, x, out, head_dim)
}

/// 单序列标量槽通道：未填位置给 `Invalid`；头宽 0 或等于整段走单头旋转。
fn rope_single(launch: &Launch, x: &[f32], out: &mut [f32], head_dim: usize) -> Result<()> {
    let pos = launch
        .scalars()
        .rope_position()
        .ok_or_else(|| Error::new(Code::Invalid, "rope_f32 位置标量未填").with(SITE))?;
    run_rope(pos as usize, x, out, head_dim)
}

/// 内核表展开：每条例子取适配函数标识符，算子名由 `stringify!` 派生，避免名字与函数名各写一份。
/// CPU 内核表：`op::REGISTRY` 的 f32 纯切片子集，不是全覆盖。
const KERNELS: &[Entry] = &op::kernel_table![
    add_f32,
    attention_f32,
    decode_attention_f32,
    decode_flash_attention_f32,
    dequant_i2,
    dequant_i4,
    dequant_i8,
    embed_f32,
    flash_attention_f32,
    gelu_f32,
    gemm_i2,
    gemm_i4,
    gemm_i8,
    gemv_f32,
    gemv_i2,
    gemv_i4,
    gemv_i8,
    mul_f32,
    reduce_max_f32,
    reduce_sum_f32,
    residual_add_f32,
    rmsnorm_f32,
    rope_f32,
    scale_f32,
    silu_f32,
    softmax_f32,
];

const _: () = all_in_registry(KERNELS);

/// 编译期断言：表内每个算子名都能在 `op::REGISTRY` 里找到（子集即可，不必等长）；
/// 归属查询复用上游 `op::contains`，本层不自抄 const 字节比较。
const fn all_in_registry(entries: &[Entry]) {
    let mut index = 0;
    while index < entries.len() {
        assert!(
            op::contains(entries[index].op),
            "serve::model 内核表含注册表之外的算子"
        );
        index += 1;
    }
}

/// 表条目对 engine [`Program`] 契约的适配：名字随表走，执行入口落到适配函数。
struct Kernel {
    entry: &'static Entry,
}

impl Program for Kernel {
    fn name(&self) -> &str {
        self.entry.op
    }

    fn run(&self, launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        (self.entry.run)(launch, inputs, outputs)
    }
}

/// 把表内全部 CPU 内核登记进执行器；重名由 [`Executor::register_program`] 拒绝。
pub fn register_cpu<E: Executor>(executor: &mut E) -> Result<()> {
    for entry in KERNELS {
        executor.register_program(Box::new(Kernel { entry }))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/model_tests.rs"]
mod tests;
