//! CPU 推理内核桥：把 `op` 的标量参考适配成 engine 可登记的 [`Program`]。
//!
//! 落点裁决见 [`backend-runtime-design.md`](../../docs/backend-runtime-design.md) §4：
//! 端点契约、路由与线格式保持模型无关；CPU 模型实现作为注入 [`Predict`](crate::Predict)
//! 的提供者收在本模块内，真设备后端仍按 ADR 0025 收在独立边界 crate，不回流这里。
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
//! 验证。`rope_f32` 的位置走标量槽回填；decode 变长历史几何仍随后续 KV 工作落地（设计文档 §5.1）。

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Launch, Program, View, ViewMut};

pub(crate) const SITE: &str = "serve::model";

/// 注意力内核入口：`q/k/v → out`，序列长 `s` 与头宽 `d` 随几何传入。
type AttentionKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize) -> Result<()>;

/// decode 注意力内核入口：`q/k/v → out`，查询行数 `q_rows`、历史行数 `hist` 与头宽 `d` 随几何传入。
type DecodeKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()>;

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
///
/// - `unary`：一进一出；
/// - `binary`：两进一出（两路是向量+向量，或向量/索引 + 驻留权重整块，算子签名相同）。
///
/// 适配函数名即算子名，注册名由 [`kernel_table!`] 从标识符派生，不再手抄第二份字符串。
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
    rmsnorm_f32,
    embed_f32,
    gemv_f32; binary
);

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

fn attention_f32(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    attention_paths(inputs, outputs, op::attention::forward)
}

fn flash_attention_f32(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    attention_paths(inputs, outputs, op::attention::flash_forward)
}

/// decode 三路适配：`q/hist_k/hist_v → out`；头宽 `d` 取算子静态语义表，查询行数 `q_rows` 由
/// `q.len()/d` 推出，历史行数 `hist` 由 `k.len()/d` 推出，形状不符交算子参考校验。
fn decode_paths(inputs: &[View<'_>], outputs: &mut [ViewMut<'_>], run: DecodeKernel) -> Result<()> {
    let q = input_f32("decode_attention", inputs, 0)?;
    let k = input_f32("decode_attention", inputs, 1)?;
    let v = input_f32("decode_attention", inputs, 2)?;
    let out = output_f32("decode_attention", outputs, 0)?;
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = q.len() / d;
    let hist = k.len() / d;
    run(q, k, v, out, q_rows, hist, d)
}

fn decode_attention_f32(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    decode_paths(inputs, outputs, op::attention::decode_forward)
}

fn decode_flash_attention_f32(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    decode_paths(inputs, outputs, op::attention::decode_flash_forward)
}

/// `[x: f32, w: u8, scales: f32] → y: f32`：4 位量化乘加，与 `op::gemv_i4::forward` 同一段算术。
fn gemv_i4(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let x = input_f32("gemv_i4", inputs, 0)?;
    let weights = input_u8("gemv_i4", inputs, 1)?;
    let scales = input_f32("gemv_i4", inputs, 2)?;
    let out = output_f32("gemv_i4", outputs, 0)?;
    op::gemv_i4::forward(x, weights, scales, out)
}

/// 同上，8 位量化：`op::gemv_i8::forward`。
fn gemv_i8(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let x = input_f32("gemv_i8", inputs, 0)?;
    let weights = input_u8("gemv_i8", inputs, 1)?;
    let scales = input_f32("gemv_i8", inputs, 2)?;
    let out = output_f32("gemv_i8", outputs, 0)?;
    op::gemv_i8::forward(x, weights, scales, out)
}

/// `[q: u8] → y: f32`：打包字节每字节两个 nibble，scale/zero 取样本常量。
fn dequant_i4(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let q = input_u8("dequant_i4", inputs, 0)?;
    let out = output_f32("dequant_i4", outputs, 0)?;
    op::dequant_i4::forward(op::dequant_i4::SCALE, op::dequant_i4::ZERO, q, out)
}

/// `[q: f32] → y: f32`：scale/zero 取样本常量。
fn dequant_i8(_launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let q = input_f32("dequant_i8", inputs, 0)?;
    let out = output_f32("dequant_i8", outputs, 0)?;
    op::dequant_i8::forward(op::dequant_i8::SCALE, op::dequant_i8::ZERO, q, out)
}

/// RoPE 适配：位置走标量槽（D9 / ADR 0043），未填位置给 `Invalid`；旋转算术走 `op::rope_f32`。
fn rope_f32(launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
    let x = input_f32("rope_f32", inputs, 0)?;
    let out = output_f32("rope_f32", outputs, 0)?;
    let pos = launch.scalars().rope_position().ok_or_else(|| {
        Error::new(Code::Invalid, "rope_f32 位置标量未填").with(SITE)
    })?;
    op::rope_f32::forward(pos as usize, x, out)
}

/// 内核表展开：每条例子取适配函数标识符，算子名由 `stringify!` 派生，避免名字与函数名各写一份。
macro_rules! kernel_table {
    ($( $run:ident ),* $(,)?) => {
        &[ $( Entry { op: stringify!($run), run: $run } ),* ]
    };
}

/// CPU 内核表：`op::REGISTRY` 的 f32 纯切片子集，不是全覆盖。
///
/// 对照 `train` 的参考全覆盖表（24/24 同序锁死）：本表只登记 serve CPU 通路实际用到的纯切片
/// f32 适配器，未覆盖的算子执行时由 [`CpuExecutor`](engine::exec::CpuExecutor) 按名拒绝。
/// 新算子进注册表时，两处表的职责不要混写——train 补参考覆盖，serve 只接自己通路用到的。
const KERNELS: &[Entry] = kernel_table![
    add_f32,
    attention_f32,
    decode_attention_f32,
    decode_flash_attention_f32,
    dequant_i4,
    dequant_i8,
    embed_f32,
    flash_attention_f32,
    gelu_f32,
    gemv_f32,
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

/// 把表内全部 CPU 内核登记进执行器；重名由 [`CpuExecutor::register`] 拒绝。
///
/// 装配侧在构造 CPU 模型宿主时调用一次；之后执行器只收图与操作数编号，不再改内核表。
pub fn register_cpu(executor: &mut CpuExecutor) -> Result<()> {
    for entry in KERNELS {
        executor.register(Box::new(Kernel { entry }))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/model_tests.rs"]
mod tests;
