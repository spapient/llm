//! 标量参考内核：把 [`crate::shape`] 的视图适配到 `op` 的标量参考实现。

use crate::shape::{Inputs, View, ViewMut, output_slice};
use crate::store::SITE;
use diagnostic::{Code, Error, Result};

/// 一步前向：`op(inputs) → output`；契约是"输入此刻都在"与"输出要被完整写入"。
pub trait Kernel {
    /// 用 `inputs` 算出 `output`；出错时这块输出缓冲会被还回池子，不会留下半个值。
    fn compute(&mut self, op: &str, inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()>;

    /// 按算子实例几何算一步前向。默认忽略几何、退回声明档语义：只有能解析实例几何的执行器
    /// 才覆盖它，其余实现保持既有行为，不因新增一条通道而被迫改动。
    fn compute_shaped(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
        geometry: Option<&crate::Geometry>,
        running: Option<&mut crate::Running>,
    ) -> Result<()> {
        let _ = (geometry, running);
        self.compute(op, inputs, output)
    }
}

/// 标量参考内核：不碰设备，逐元素算，算术复用 `op` 的参考实现（同一份语义的两次执行）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reference;

impl Kernel for Reference {
    fn compute_shaped(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
        geometry: Option<&crate::Geometry>,
        running: Option<&mut crate::Running>,
    ) -> Result<()> {
        match geometry {
            Some(geometry) => self.shaped(op, inputs, output, geometry, running),
            None => self.compute(op, inputs, output),
        }
    }

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

impl Reference {
    /// 按实例几何前向：算术复用 `op::conv` / `op::norm` 的真实现，与声明档内核同一份算术。
    fn shaped(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
        geometry: &crate::Geometry,
        running: Option<&mut crate::Running>,
    ) -> Result<()> {
        let mut slices = Vec::with_capacity(inputs.len());
        for (idx, view) in inputs.iter().enumerate() {
            let slice = view.f32().ok_or_else(|| {
                Error::new(Code::Invalid, format!("`{op}` 的第 {idx} 个输入不是 f32")).with(SITE)
            })?;
            slices.push(slice);
        }
        geometry.forward_with(&slices, output_slice::<f32>(op, output)?, running)
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
/// 一元 `[f32] → [f32]` 适配：套用算子的标量实现。
macro_rules! unary {
    ($($run:ident => $module:ident),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a,) = <(f32,) as Inputs>::unpack(inputs, stringify!($run))?;
            op::$module::forward(a, output_slice::<f32>(stringify!($run), output)?)
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

/// 归一化 `[x, gamma, beta] → [y]` 适配：三个 f32 输入、一个 f32 输出；
/// 声明档几何（通道数、空间、批大小）固定在算子模块里，由层侧的真实几何另行决定。
macro_rules! norm {
    ($($run:ident => $module:ident),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (x, gamma, beta) =
                <(f32, f32, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            op::$module::forward(x, gamma, beta, output_slice::<f32>(stringify!($run), output)?)
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
    adaptive_pool_f32 => adaptive_pool,
    avg_pool2d_f32 => avg_pool2d,
    exp_f32 => exp_f32,
    gelu_f32 => gelu_f32,
    hard_sigmoid_f32 => hard_sigmoid_f32,
    leaky_relu_f32 => leaky_relu_f32,
    interpolate_f32 => interpolate,
    log_f32 => log_f32,
    max_pool1d_f32 => max_pool1d,
    max_pool2d_f32 => max_pool2d,
    silu_f32 => silu_f32,
    softmax_f32 => softmax_f32,
    reduce_max_f32 => reduce_max_f32,
    reduce_sum_f32 => reduce_sum_f32,
    scale_f32 => scale_f32,
    scan_f32 => scan_f32
);
binary!(
    add_f32,
    div_f32,
    embed_f32,
    max_f32,
    min_f32,
    mul_f32,
    rmsnorm_f32,
    residual_add_f32,
    sub_f32
);

norm!(
    batch_norm_f32 => batch_norm,
    group_norm_f32 => group_norm,
    instance_norm_f32 => instance_norm,
    layer_norm_f32 => layer_norm
);

/// 二维卷积适配：`[x, weight, bias] → y`，声明档单样本几何由算子模块定。
fn conv2d_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (x, weight, bias) = <(f32, f32, f32) as Inputs>::unpack(inputs, "conv2d_f32")?;
    op::conv2d::forward(x, weight, bias, output_slice::<f32>("conv2d_f32", output)?)
}

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

/// 大 K 稠密 GEMM：收缩轴取生产档 [`op::matmul::K`]，行数由输入长度推出。
fn matmul_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a, b) = <(f32, f32) as Inputs>::unpack(inputs, "matmul_f32")?;
    let out = output_slice::<f32>("matmul_f32", output)?;
    let (m, n) = matrix_geometry("matmul_f32", a.len(), out.len(), op::matmul::K)?;
    op::fast::gemm_f32_checked(a, b, m, n, op::matmul::K, out)
}

/// 大 K 对称 int8 GEMM：输入是承载有符号码的 `u8`，`i32` 精确累加后转 `f32`。
fn matmul_i8(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a, b) = <(u8, u8) as Inputs>::unpack(inputs, "matmul_i8")?;
    let out = output_slice::<f32>("matmul_i8", output)?;
    let (m, n) = matrix_geometry("matmul_i8", a.len(), out.len(), op::matmul_i8::K)?;
    let mut acc = vec![0i32; m * n];
    let (a, b) = (signed(a), signed(b));
    op::fast::gemm_i8_checked(&a, &b, m, n, op::matmul_i8::K, &mut acc)?;
    widen(acc, out);
    Ok(())
}

/// `i32` 精确累加结果 → `f32` 输出：算子的输出类型在声明里，累加精度在这里。
fn widen(acc: Vec<i32>, out: &mut [f32]) {
    for (slot, value) in out.iter_mut().zip(acc) {
        *slot = value as f32;
    }
}

/// 行主序 GEMM 的行列数：`m = 左操作数长度 / k`，`n = 输出长度 / m`。
fn matrix_geometry(site: &str, lhs: usize, out: usize, k: usize) -> Result<(usize, usize)> {
    let m = lhs / k;
    Ok((m, batches(site, out, m.max(1))?))
}

/// `u8` 承载的有符号码 → `i8`：字节相同、解释不同，i2/i4/i8 三档共用这一处转换。
fn signed(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|&x| x as i8).collect()
}

/// 稠密自注意力内核入口：`q/k/v → out`，查询行数 `heads` 与头宽 `d` 随几何传入。
type AttentionKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize) -> Result<()>;

/// decode 注意力内核入口：`q/k/v → out`，查询行数 `q_rows`、历史行数 `hist` 与头宽 `d` 随几何传入。
type DecodeKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()>;

/// 量化 GEMV 适配：`[x: f32, w: u8, scales: f32] → y: f32`。三档位宽同一条形参表，
/// 只差一行算术入口，因此适配只写一份。
macro_rules! quant_gemv {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (x, w, scales) = <(f32, u8, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            op::$run::forward(x, w, scales, output_slice::<f32>(stringify!($run), output)?)
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
    let d = op::attention::MODEL.shape.1;
    let heads = batches("attention_f32", q.len(), d)?;
    run(q, k, v, out, heads, d)
}

/// `[q, k, v: f32] → y: f32`：尺寸由声明宽度推导；算术走稠密分块 tile。
fn attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    attention_paths(inputs, output, op::attention::tiled::forward)
}

/// 同形参，走 online softmax 的分块 tile 版（`op::attention::tiled::flash_tiled_forward`）。
fn flash_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    attention_paths(inputs, output, op::attention::tiled::flash_tiled_forward)
}

/// decode 注意力适配的公共几何：`q_rows` 由 q 长、`hist` 由 k 长按头宽 `d` 推出。
fn decode_paths(inputs: &[View<'_>], output: &mut ViewMut<'_>, run: DecodeKernel) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "decode_attention_f32")?;
    let out = output_slice::<f32>("decode_attention_f32", output)?;
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = batches("decode_attention_f32", q.len(), d)?;
    let hist = batches("decode_attention_f32", k.len(), d)?;
    run(q, k, v, out, q_rows, hist, d)
}

/// decode 注意力：`q/k/v → out`。
fn decode_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    decode_paths(inputs, output, op::attention::decode_forward)
}

/// 同 decode 几何，走 online softmax 版 `op::attention::decode_flash_forward`。
fn decode_flash_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    decode_paths(inputs, output, op::attention::decode_flash_forward)
}

/// 潜空间注意力适配：`[q, k, v: f32] → y: f32`，行宽取声明宽度，输出宽由输出长度反推。
///
/// 与 `attention_f32` 的区别只有输出宽可以更窄（上投影折在口径差里），所以适配层不做任何
/// 头数/压缩比的假设——全部几何都由张量长度与声明宽度给出。
fn latent_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q, k, v) = <(f32, f32, f32) as Inputs>::unpack(inputs, "latent_attention_f32")?;
    let out = output_slice::<f32>("latent_attention_f32", output)?;
    let w = op::attention::latent::MODEL.shape.1;
    op::attention::latent_forward(q, k, v, out, w)
}

/// 反量化适配：`[q] → y: f32`，scale/zero 取样本常量；打包与逐元素两档只差输入 lane。
macro_rules! dequant {
    ($(($run:ident, $lane:ty)),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (q,) = <($lane,) as Inputs>::unpack(inputs, stringify!($run))?;
            let out = output_slice::<f32>(stringify!($run), output)?;
            op::$run::forward(op::$run::SCALE, op::$run::ZERO, q, out)
        }
    )*};
}

dequant!((dequant_i2, u8), (dequant_i4, u8), (dequant_i8, f32));

/// `[a: f32] → y: f32`：位置取样本约定的 7。
fn rope_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a,) = <(f32,) as Inputs>::unpack(inputs, "rope_f32")?;
    op::rope_f32::forward(7, a, output_slice::<f32>("rope_f32", output)?)
}

/// 标量内核表：长度就是 [`op::REGISTRY`] 的长度，名字与顺序由下面的 `const` 断言锁住。
const KERNELS: [Entry; op::REGISTRY.len()] = op::kernel_table![
    adaptive_pool_f32,
    add_f32,
    attention_f32,
    avg_pool2d_f32,
    batch_norm_f32,
    conv2d_f32,
    decode_attention_f32,
    decode_flash_attention_f32,
    dequant_i2,
    dequant_i4,
    dequant_i8,
    div_f32,
    embed_f32,
    exp_f32,
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
    group_norm_f32,
    hard_sigmoid_f32,
    instance_norm_f32,
    interpolate_f32,
    latent_attention_f32,
    layer_norm_f32,
    leaky_relu_f32,
    log_f32,
    matmul_f32,
    matmul_i8,
    max_f32,
    max_pool1d_f32,
    max_pool2d_f32,
    min_f32,
    mul_f32,
    reduce_max_f32,
    reduce_sum_f32,
    residual_add_f32,
    rmsnorm_f32,
    rope_f32,
    scale_f32,
    scan_f32,
    silu_f32,
    softmax_f32,
    sub_f32,
];

const _: () = same_order(&KERNELS, &op::REGISTRY);

/// 编译期断言：内核表与算子注册表逐条同名（长度已由类型锁住）。
const fn same_order(entries: &[Entry], registry: &[op::Registration]) {
    let mut idx = 0;
    while idx < entries.len() {
        assert!(
            same_str(entries[idx].op, registry[idx].name),
            "train::Reference 的内核表与 op::REGISTRY 名字或顺序漂移"
        );
        idx += 1;
    }
}

/// `const` 上下文逐字节比：唯一实现是 [`op::same_bytes`]，本层只重导出别名，
/// 内核表与 [`op::REGISTRY`] 的名字对拍不再自带第二份比较逻辑。
pub(crate) use op::same_bytes as same_str;

pub mod shape;
