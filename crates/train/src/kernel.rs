//! 标量参考内核：把 [`crate::shape`] 的视图适配到 `op` 的标量参考实现。
//!
//! 算术全部复用 `op` 自己的参考语义；本层不定义算子、不决定显存计划。内核表长度与顺序同
//! `op::REGISTRY` 在编译期锁死，"注册了算子却没有参考实现"是编译错误。

use crate::shape::{Inputs, View, ViewMut, output_slice};
use crate::store::SITE;
use diagnostic::{Code, Error, Result};

/// 一步前向：`op(inputs) → output`；契约是"输入此刻都在"与"输出要被完整写入"。
pub trait Kernel {
    /// 用 `inputs` 算出 `output`；出错时这块输出缓冲会被还回池子，不会留下半个值。
    fn compute(&mut self, op: &str, inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()>;
}

/// 标量参考内核：不碰设备，逐元素算，算术复用 `op` 的参考实现（同一份语义的两次执行）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reference;

impl Kernel for Reference {
    fn compute(&mut self, op: &str, inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
        match KERNELS.iter().find(|entry| entry.op == op) {
            Some(entry) => (entry.run)(inputs, output),
            None => {
                let covered = KERNELS.iter().map(|entry| entry.op).collect::<Vec<_>>();
                let message = format!(
                    "`{op}` 没有标量内核；`train::Reference` 覆盖 {}",
                    covered.join(", ")
                );
                Err(Error::new(Code::Unsupported, message).with(SITE))
            }
        }
    }
}

/// 输出长度按行宽 `k` 求批次数；`k` 为 0 或不能整除时显式报错，不把维度错误静默成 0。
fn batches(op: &str, len: usize, k: usize) -> Result<usize> {
    if k == 0 || !len.is_multiple_of(k) {
        return Err(Error::new(
            Code::Invalid,
            format!("`{op}` 输出 {len} 个元素不能按行宽 {k} 整除"),
        )
        .with(SITE));
    }
    Ok(len / k)
}

/// 内核表的一行：算子名 + 缓冲级入口（函数指针：表在编译期定长，按名字线性查一次）。
struct Entry {
    /// 算子名，必须与 [`op::REGISTRY`] 同名同序。
    op: &'static str,
    /// 把视图拆成算子的形参，再调算子自己的标量实现。
    run: fn(&[View<'_>], &mut ViewMut<'_>) -> Result<()>,
}

/// 标量内核表：算子名由适配函数的标识符派生（两者同名是同一条判据）。
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => {
        [$(Entry { op: stringify!($run), run: $run }),*]
    };
}

/// 一元 `[f32] → [f32]` 适配：套用算子的标量实现。
macro_rules! unary {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a,) = <(f32,) as Inputs>::unpack(inputs, stringify!($run))?;
            op::$run::forward(a, output_slice::<f32>(stringify!($run), output)?)
        }
    )*};
}

/// 二元 `[f32, f32] → [f32]` 适配。
macro_rules! binary {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a, b) = <(f32, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            op::$run::forward(a, b, output_slice::<f32>(stringify!($run), output)?)
        }
    )*};
}

/// 矩阵乘适配：`m` 由 `K` 与缓冲长度推导（`i4` 每字节两个 nibble），`n` 由输出长度推导。
macro_rules! gemm {
    ($(($run:ident, $lane:ident, $div:expr)),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a, b) = <($lane, $lane) as Inputs>::unpack(inputs, stringify!($run))?;
            let out = output_slice::<f32>(stringify!($run), output)?;
            let m = a.len() / $div;
            let rows = batches(stringify!($run), out.len(), m)?;
            op::$run::forward(a, b, m, rows, out)
        }
    )*};
}

unary!(
    gelu_f32,
    silu_f32,
    softmax_f32,
    reduce_max_f32,
    reduce_sum_f32,
    scale_f32
);
binary!(
    add_f32,
    embed_f32,
    mul_f32,
    rmsnorm_f32,
    residual_add_f32
);

/// 稠密向量投影：输入列表按算子操作数声明 `[weights, x]` 排列。
fn gemv_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (weights, x) = <(f32, f32) as Inputs>::unpack(inputs, "gemv_f32")?;
    op::gemv_f32::forward(x, weights, output_slice::<f32>("gemv_f32", output)?)
}
gemm!(
    (gemm_f32, f32, op::gemm_f32::K),
    (gemm_i2, u8, op::gemm_i2::K / 4),
    (gemm_i4, u8, op::gemm_i4::K / 2),
    (gemm_i8, u8, op::gemm_i8::K),
);

/// `[x: f32, w: u8, scales: f32] → y: f32`，与 `op::gemv_i2::forward` 同一段算术。
fn gemv_i2(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (x, w, scales) = <(f32, u8, f32) as Inputs>::unpack(inputs, "gemv_i2")?;
    op::gemv_i2::forward(x, w, scales, output_slice::<f32>("gemv_i2", output)?)
}

/// `[x: f32, w: u8, scales: f32] → y: f32`，与 `op::gemv_i4::forward` 同一段算术。
fn gemv_i4(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (x, w, scales) = <(f32, u8, f32) as Inputs>::unpack(inputs, "gemv_i4")?;
    op::gemv_i4::forward(x, w, scales, output_slice::<f32>("gemv_i4", output)?)
}

/// 同上，量化 8 位：`op::gemv_i8::forward`。
fn gemv_i8(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (x, w, scales) = <(f32, u8, f32) as Inputs>::unpack(inputs, "gemv_i8")?;
    op::gemv_i8::forward(x, w, scales, output_slice::<f32>("gemv_i8", output)?)
}

/// `[q, k, v: f32] → y: f32`：尺寸由声明宽度推导。
fn attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "attention_f32")?;
    let out = output_slice::<f32>("attention_f32", output)?;
    let d = op::attention::MODEL.shape.1;
    let heads = batches("attention_f32", q.len(), d)?;
    op::attention::forward(q, k, v, out, heads, d)
}

/// 同形参，走 online softmax 版 `op::attention::flash_forward`。
fn flash_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "flash_attention_f32")?;
    let out = output_slice::<f32>("flash_attention_f32", output)?;
    let d = op::attention::MODEL.shape.1;
    let heads = batches("flash_attention_f32", q.len(), d)?;
    op::attention::flash_forward(q, k, v, out, heads, d)
}

/// decode 注意力：`q/k/v → out`；`q_rows` 由 q 长、`hist` 由 k 长按头宽 `d` 推出。
fn decode_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "decode_attention_f32")?;
    let out = output_slice::<f32>("decode_attention_f32", output)?;
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = batches("decode_attention_f32", q.len(), d)?;
    let hist = batches("decode_attention_f32", k.len(), d)?;
    op::attention::decode_forward(q, k, v, out, q_rows, hist, d)
}

/// 同 decode 几何，走 online softmax 版 `op::attention::decode_flash_forward`。
fn decode_flash_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "decode_flash_attention_f32")?;
    let out = output_slice::<f32>("decode_flash_attention_f32", output)?;
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = batches("decode_flash_attention_f32", q.len(), d)?;
    let hist = batches("decode_flash_attention_f32", k.len(), d)?;
    op::attention::decode_flash_forward(q, k, v, out, q_rows, hist, d)
}

/// `[q: u8] → y: f32`：打包字节每字节四个 2 位码，参数取样本常量。
fn dequant_i2(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q,) = <(u8,) as Inputs>::unpack(inputs, "dequant_i2")?;
    let out = output_slice::<f32>("dequant_i2", output)?;
    op::dequant_i2::forward(op::dequant_i2::SCALE, op::dequant_i2::ZERO, q, out)
}

/// `[q: u8] → y: f32`：打包字节每字节两个 nibble，参数取样本常量。
fn dequant_i4(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q,) = <(u8,) as Inputs>::unpack(inputs, "dequant_i4")?;
    let out = output_slice::<f32>("dequant_i4", output)?;
    op::dequant_i4::forward(op::dequant_i4::SCALE, op::dequant_i4::ZERO, q, out)
}

/// `[q: f32] → y: f32`：参数取样本常量。
fn dequant_i8(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q,) = <(f32,) as Inputs>::unpack(inputs, "dequant_i8")?;
    let out = output_slice::<f32>("dequant_i8", output)?;
    op::dequant_i8::forward(op::dequant_i8::SCALE, op::dequant_i8::ZERO, q, out)
}

/// `[a: f32] → y: f32`：位置取样本约定的 7。
fn rope_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a,) = <(f32,) as Inputs>::unpack(inputs, "rope_f32")?;
    op::rope_f32::forward(7, a, output_slice::<f32>("rope_f32", output)?)
}

/// 标量内核表：长度就是 [`op::REGISTRY`] 的长度，名字与顺序由下面的 `const` 断言锁住。
const KERNELS: [Entry; op::REGISTRY.len()] = kernel_table![
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
    gemm_f32,
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

const _: () = same_order(&KERNELS, &op::REGISTRY);

/// 编译期断言：内核表与算子注册表逐条同名（长度已由类型锁住）。
const fn same_order(entries: &[Entry], registry: &[op::Registration]) {
    let mut index = 0;
    while index < entries.len() {
        assert!(
            same_str(entries[index].op, registry[index].name),
            "train::Reference 的内核表与 op::REGISTRY 名字或顺序漂移"
        );
        index += 1;
    }
}

/// `const` 上下文里没有可用的 `str` 比较，逐字节比。
const fn same_str(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}
