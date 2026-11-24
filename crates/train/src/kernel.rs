//! 缓冲形态、视图与标量参考内核：`f32` 与按位打包 `u8` 两条 lane 的静态契约，以及把视图适配到
//! `nn` 参考实现的标量内核表。

use std::cell::RefCell;
use std::rc::Rc;

use crate::pool::SITE;
use diagnostic::{Code, Error, Result};
use nn::kernel_table;
use storage::{LaneBuf, LaneView, LaneViewMut};
use tensor::DType;

/// 一步前向：`op(inputs) → output`；契约是"输入此刻都在"与"输出要被完整写入"。
pub trait Kernel {
    /// 用 `inputs` 算出 `output`；出错时这块输出缓冲会被还回池子，不会留下半个值。
    fn compute_step(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
    ) -> Result<()>;

    /// 按算子实例几何算一步前向。默认忽略几何、退回声明档语义：只有能解析实例几何的执行器
    /// 才覆盖它，其余实现保持既有行为，不因新增一条通道而被迫改动。
    fn compute_shaped_step(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
        geometry: Option<&crate::Geometry>,
        running: Option<&mut crate::Running>,
    ) -> Result<()> {
        let _ = (geometry, running);
        self.compute_step(op, inputs, output)
    }
}

/// 标量参考内核：不碰设备，逐元素算，算术复用 `nn` 的参考实现（同一份语义的两次执行）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reference;

impl Kernel for Reference {
    fn compute_shaped_step(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
        geometry: Option<&crate::Geometry>,
        running: Option<&mut crate::Running>,
    ) -> Result<()> {
        match geometry {
            Some(geometry) => self.shaped(op, inputs, output, geometry, running),
            None => self.compute_step(op, inputs, output),
        }
    }

    fn compute_step(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
    ) -> Result<()> {
        if let Some(entry) = KERNELS.iter().find(|entry| entry.op == op) {
            (entry.run)(inputs, output)
        } else {
            let covered = KERNELS.iter().map(|entry| entry.op).collect::<Vec<_>>();
            let message = format!(
                "`{op}` 没有标量内核；`train::Reference` 覆盖 {}",
                covered.join(", ")
            );
            Err(Error::new(Code::Unsupported, message).with(SITE))
        }
    }
}

impl Reference {
    /// 按实例几何前向：算术复用 `nn::conv` / `nn::kernel::norm` 的真实现，与声明档内核同一份算术。
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
pub(crate) fn batches(op: &str, len: usize, k: usize) -> Result<usize> {
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
    /// 算子名，必须与 [`nn::registry::REGISTRY`] 同名同序。
    op: &'static str,
    /// 把视图拆成算子的形参，再调算子自己的标量实现。
    run: fn(&[View<'_>], &mut ViewMut<'_>) -> Result<()>,
}

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

/// 单行内核声明的适配器：声明种类决定拆包与调用形态，自定义函数只登记表项。
macro_rules! kernel_adapter {
    (unary!($run:ident => $module:ident)) => {
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a,) = <(f32,) as Inputs>::unpack(inputs, stringify!($run))?;
            nn::$module::forward(
                tensor::View::one(a),
                output_slice::<f32>(stringify!($run), output)?,
            )
        }
    };
    (binary!($run:ident)) => {
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a, b) = <(f32, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            nn::$run::forward(
                tensor::View::one(a),
                tensor::View::one(b),
                output_slice::<f32>(stringify!($run), output)?,
            )
        }
    };
    (norm!($run:ident => $module:ident)) => {
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (x, gamma, beta) = <(f32, f32, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            nn::$module::forward(
                tensor::View::one(x),
                tensor::View::one(gamma),
                tensor::View::one(beta),
                output_slice::<f32>(stringify!($run), output)?,
            )
        }
    };
    (gemm!($run:ident, f32, $div:expr)) => {
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a, b) = <(f32, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            let out = output_slice::<f32>(stringify!($run), output)?;
            let m = a.len() / $div;
            let rows = batches(stringify!($run), out.len(), m)?;
            nn::$run::forward(
                tensor::View::matrix(m, $div, a)?,
                tensor::View::matrix($div, rows, b)?,
                out,
            )
        }
    };
    (gemm!($run:ident, u8, $div:expr)) => {
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (a, b) = <(u8, u8) as Inputs>::unpack(inputs, stringify!($run))?;
            let out = output_slice::<f32>(stringify!($run), output)?;
            let m = a.len() / $div;
            let rows = batches(stringify!($run), out.len(), m)?;
            nn::$run::forward(a, b, m, rows, out)
        }
    };
    (dequant!($run:ident, $lane:ty)) => {
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (q,) = <($lane,) as Inputs>::unpack(inputs, stringify!($run))?;
            let out = output_slice::<f32>(stringify!($run), output)?;
            nn::$run::forward(nn::$run::SCALE, nn::$run::ZERO, q, out)
        }
    };
    (no_scalar!($run:ident => $dtype:expr)) => {
        fn $run(_inputs: &[View<'_>], _output: &mut ViewMut<'_>) -> Result<()> {
            Err(no_storage($dtype))
        }
    };
    (custom!($run:ident)) => {};
}

/// 唯一内核目录：一行同时生成适配函数和表项，表序与 `nn::registry::REGISTRY` 编译期互锁。
macro_rules! kernels {
    ($($kind:ident!($run:ident $($args:tt)*)),* $(,)?) => {
        $(kernel_adapter!($kind!($run $($args)*));)*
        const KERNELS: [Entry; nn::registry::REGISTRY.len()] = kernel_table![$($run),*];
    };
}

kernels!(
    no_scalar!(abs_f16 => DType::F16),
    unary!(abs_f32 => abs_f32),
    unary!(adaptive_pool_f32 => adaptive_pool),
    binary!(add_f32),
    custom!(attention_f32),
    unary!(avg_pool2d_f32 => avg_pool2d),
    norm!(batch_norm_f32 => batch_norm),
    unary!(ceil_f32 => ceil_f32),
    custom!(clamp_f32),
    custom!(conv2d_f32),
    unary!(cos_f32 => cos_f32),
    custom!(decode_attention_f32),
    custom!(decode_flash_attention_f32),
    dequant!(dequant_i2, u8),
    dequant!(dequant_i4, u8),
    custom!(dequant_i8),
    binary!(div_f32),
    binary!(embed_f32),
    unary!(exp2_f32 => exp2_f32),
    unary!(exp_f32 => exp_f32),
    custom!(flash_attention_f32),
    unary!(floor_f32 => floor_f32),
    unary!(gelu_f32 => gelu_f32),
    gemm!(gemm_f32, f32, nn::gemm_f32::K),
    gemm!(gemm_i2, u8, nn::gemm_i2::K / 4),
    gemm!(gemm_i4, u8, nn::gemm_i4::K / 2),
    gemm!(gemm_i8, u8, nn::gemm_i8::K),
    custom!(gemv_f32),
    custom!(gemv_i2),
    custom!(gemv_i4),
    custom!(gemv_i8),
    norm!(group_norm_f32 => group_norm),
    unary!(hard_sigmoid_f32 => hard_sigmoid_f32),
    norm!(instance_norm_f32 => instance_norm),
    unary!(interpolate_f32 => interpolate),
    custom!(latent_attention_f32),
    norm!(layer_norm_f32 => layer_norm),
    unary!(leaky_relu_f32 => leaky_relu_f32),
    unary!(log1p_f32 => log1p_f32),
    unary!(log_f32 => log_f32),
    custom!(matmul_f32),
    custom!(matmul_i8),
    binary!(max_f32),
    unary!(max_pool1d_f32 => max_pool1d),
    unary!(max_pool2d_f32 => max_pool2d),
    binary!(min_f32),
    binary!(mul_f32),
    unary!(neg_f32 => neg_f32),
    unary!(recip_f32 => recip_f32),
    unary!(reduce_max_f32 => reduce_max_f32),
    unary!(reduce_prod_f32 => reduce_prod_f32),
    unary!(reduce_sum_f32 => reduce_sum_f32),
    unary!(relu_f32 => relu_f32),
    binary!(remainder_f32),
    binary!(residual_add_f32),
    binary!(rmsnorm_f32),
    custom!(rope_f32),
    unary!(round_f32 => round_f32),
    unary!(rsqrt_f32 => rsqrt_f32),
    unary!(scale_f32 => scale_f32),
    unary!(scan_f32 => scan_f32),
    unary!(sigmoid_f32 => sigmoid_f32),
    unary!(sign_f32 => sign_f32),
    unary!(silu_f32 => silu_f32),
    unary!(sin_f32 => sin_f32),
    unary!(softmax_f32 => softmax_f32),
    unary!(sqrt_f32 => sqrt_f32),
    unary!(square_f32 => square_f32),
    binary!(sub_f32),
    unary!(tanh_f32 => tanh_f32),
    unary!(trunc_f32 => trunc_f32),
);

/// 二维卷积适配：`[x, weight, bias] → y`，声明档单样本几何由算子模块定。
fn conv2d_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (x, weight, bias) = <(f32, f32, f32) as Inputs>::unpack(inputs, "conv2d_f32")?;
    nn::conv::conv2d::forward(
        tensor::View::one(x),
        tensor::View::one(weight),
        tensor::View::one(bias),
        output_slice::<f32>("conv2d_f32", output)?,
    )
}

/// 稠密向量投影：输入列表按算子操作数声明 `[weights, x]` 排列。
fn gemv_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (weights, x) = <(f32, f32) as Inputs>::unpack(inputs, "gemv_f32")?;
    nn::gemv_f32::forward(
        tensor::View::one(x),
        weights,
        output_slice::<f32>("gemv_f32", output)?,
    )
}

/// 大 K 稠密 GEMM：收缩轴取生产档 [`nn::matmul::K`]，行数由输入长度推出。
fn matmul_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a, b) = <(f32, f32) as Inputs>::unpack(inputs, "matmul_f32")?;
    let out = output_slice::<f32>("matmul_f32", output)?;
    let (m, n) = matrix_geometry("matmul_f32", a.len(), out.len(), nn::matmul::K)?;
    let (av, bv) = matrix_pair(a, b, m, n, nn::matmul::K)?;
    nn::fast::gemm_f32_checked(av, bv, out)
}

/// 大 K 对称 int8 GEMM：输入是承载有符号码的 `u8`，`i32` 精确累加后转 `f32`。
fn matmul_i8(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a, b) = <(u8, u8) as Inputs>::unpack(inputs, "matmul_i8")?;
    let out = output_slice::<f32>("matmul_i8", output)?;
    let (m, n) = matrix_geometry("matmul_i8", a.len(), out.len(), nn::matmul_i8::K)?;
    let (a, b) = (signed(a), signed(b));
    let (av, bv) = matrix_pair(&a, &b, m, n, nn::matmul_i8::K)?;
    finish_i8(av, bv, out, m, n)
}

/// `i32` 累加缓冲：跑 int8 GEMM 后把累加结果拓宽成 f32 输出。
fn finish_i8(
    av: tensor::View<'_, i8>,
    bv: tensor::View<'_, i8>,
    out: &mut [f32],
    m: usize,
    n: usize,
) -> Result<()> {
    let mut acc = vec![0i32; m * n];
    nn::fast::gemm_i8_checked(av, bv, &mut acc)?;
    widen(acc, out);
    Ok(())
}

/// 两行主序矩阵视图对：`a` 是 `m×k`、`b` 是 `k×n`。
fn matrix_pair<'a, K: tensor::Element>(
    a: &'a [K],
    b: &'a [K],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(tensor::View<'a, K>, tensor::View<'a, K>)> {
    Ok((
        tensor::View::matrix(m, k, a)?,
        tensor::View::matrix(k, n, b)?,
    ))
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

/// i8 反量化：f32 承载量化整数，输入包只读张量视图。
fn dequant_i8(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (q,) = <(f32,) as Inputs>::unpack(inputs, "dequant_i8")?;
    let out = output_slice::<f32>("dequant_i8", output)?;
    nn::dequant_i8::forward(
        nn::dequant_i8::SCALE,
        nn::dequant_i8::ZERO,
        tensor::View::one(q),
        out,
    )
}

/// clamp：`[x] → y`，界取模块常量 `LO` / `HI`（签名通道只支持整数标量）。
fn clamp_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (x,) = <(f32,) as Inputs>::unpack(inputs, "clamp_f32")?;
    let out = output_slice::<f32>("clamp_f32", output)?;
    nn::clamp_f32::forward(
        nn::clamp_f32::LO,
        nn::clamp_f32::HI,
        tensor::View::one(x),
        out,
    )
}

/// `[a: f32] → y: f32`：位置取样本约定的 7。
fn rope_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a,) = <(f32,) as Inputs>::unpack(inputs, "rope_f32")?;
    nn::rope_f32::forward(
        7,
        tensor::View::one(a),
        output_slice::<f32>("rope_f32", output)?,
    )
}

// ---- 反向共享的旋转与正交化数值核心 ----

/// `rope_f32` 的固定位置：前向与反向同为 7（标量槽尚未接进训练内核）。
const ROPE_POS: f32 = 7.0;

/// 旋转表缓存槽：段长 + 该段长下的 `(sin, cos)` 表。
type RotationCache = RefCell<Option<(usize, Rc<[(f32, f32)]>)>>;

thread_local! {
    /// 段长 → 旋转表。`θ_i = ROPE_POS·base^(-2i/d)` 只依赖 `d`，与数据无关，所以 `(sin, cos)`
    /// 也可以一次性算好：逐对重算 `powf` 与 `sin_cos` 是这条路径上唯一的纯浪费。
    static ROTATIONS: RotationCache = const { RefCell::new(None) };
}

/// 段长 `d` 的旋转表：`d/2` 对 `(sin θ_i, cos θ_i)`，与逐对计算的 `angle.sin_cos()` 逐位相同。
fn rotations(d: usize) -> Rc<[(f32, f32)]> {
    ROTATIONS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some((cached, table)) = slot.as_ref()
            && *cached == d
        {
            return Rc::clone(table);
        }
        let table: Rc<[(f32, f32)]> = nn::rope_f32::frequencies(d)
            .iter()
            .map(|&freq| (ROPE_POS * freq).sin_cos())
            .collect();
        *slot = Some((d, Rc::clone(&table)));
        table
    })
}

/// RoPE 反向本体：逐对把上游梯度按同一角度旋转回去；末尾不成对的元素原样透传。
pub(crate) fn rope_grad(x: &[f32], grad: &[f32]) -> Vec<f32> {
    let d = x.len();
    let pairs = d / 2;
    let mut dx = vec![0.0; d];
    let rotations = rotations(d);
    for (i, &(sin, cos)) in rotations.iter().enumerate() {
        dx[2 * i] = grad[2 * i] * cos + grad[2 * i + 1] * sin;
        dx[2 * i + 1] = -grad[2 * i] * sin + grad[2 * i + 1] * cos;
    }
    debug_assert_eq!(rotations.len(), pairs);
    if d % 2 == 1 {
        dx[d - 1] = grad[d - 1];
    }
    dx
}

/// Newton–Schulz 的迭代次数：Muon 参考实现取 5，够把 `[0.3, 3]` 的谱压到 1 附近。
const NS_STEPS: usize = 5;

/// 五次迭代的多项式系数 `(a, b, c)`：`X ← aX + b(XA) + c(XA²)`，`A` 是小边上的 Gram。
const NS_COEF: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// 近似正交化：先按 Frobenius 范数归一，再迭代 [`NS_STEPS`] 次 Newton–Schulz。
pub(crate) fn orthogonalize(grad: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let norm = grad.iter().fold(0.0_f32, |acc, v| acc + v * v).sqrt();
    let scale = 1.0 / (norm + eps);
    let mut x: Vec<f32> = grad.iter().map(|v| v * scale).collect();
    for _ in 0..NS_STEPS {
        let (gram, side) = if rows >= cols {
            (gram_of_columns(&x, rows, cols), Side::Right)
        } else {
            (gram_of_rows(&x, rows, cols), Side::Left)
        };
        let squared = square(&gram, side.order(rows, cols));
        x = advance(&x, rows, cols, &gram, &squared, side);
    }
    x
}

/// `XᵀX`（`cols × cols`）：行数不少于列数时的小边。
fn gram_of_columns(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0_f32; cols * cols];
    for row in 0..rows {
        let lane = &x[row * cols..(row + 1) * cols];
        for col in 0..cols {
            let left = lane[col];
            if left == 0.0 {
                continue;
            }
            for inner in 0..cols {
                gram[col * cols + inner] += left * lane[inner];
            }
        }
    }
    gram
}

/// `X Xᵀ`（`rows × rows`）：列数多于行数时的小边。
fn gram_of_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0_f32; rows * rows];
    for outer in 0..rows {
        for inner in 0..rows {
            let mut acc = 0.0_f32;
            for col in 0..cols {
                acc += x[outer * cols + col] * x[inner * cols + col];
            }
            gram[outer * rows + inner] = acc;
        }
    }
    gram
}

/// 方阵自乘 `M·M`（行主序）。Gram 对称且阶数是小边，直接写三重循环。
fn square(matrix: &[f32], order: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; order * order];
    for row in 0..order {
        for inner in 0..order {
            let left = matrix[row * order + inner];
            if left == 0.0 {
                continue;
            }
            for col in 0..order {
                out[row * order + col] += left * matrix[inner * order + col];
            }
        }
    }
    out
}

/// Gram 乘在 `X` 的哪一边：小边是列时乘在右边（`XA`），小边是行时乘在左边（`AX`）。
#[derive(Clone, Copy)]
enum Side {
    /// `A` 是 `rows × rows`（小边是行），乘在 `X` 左边。
    Left,
    /// `A` 是 `cols × cols`（小边是列），乘在 `X` 右边。
    Right,
}

impl Side {
    /// 小边的阶数：Gram 与 `A²` 都是这个阶数的方阵。
    fn order(self, rows: usize, cols: usize) -> usize {
        match self {
            Side::Left => rows,
            Side::Right => cols,
        }
    }
}

/// 一步 `X ← aX + b(AX) + c(A²X)`；`side` 为 [`Side::Right`] 时是 `X ← aX + b(XA) + c(XA²)`。
/// 两者遍历的是同一块 `X`，只是 `A` 乘在另一侧：沿哪条轴收缩、Gram 槽位怎么排跟着 `side` 变。
fn advance(
    x: &[f32],
    rows: usize,
    cols: usize,
    gram: &[f32],
    squared: &[f32],
    side: Side,
) -> Vec<f32> {
    let (a, b, c) = NS_COEF;
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = a * x[row * cols + col];
            for inner in 0..side.order(rows, cols) {
                let (value, slot) = match side {
                    Side::Left => (x[inner * cols + col], row * rows + inner),
                    Side::Right => (x[row * cols + inner], inner * cols + col),
                };
                acc += b * value * gram[slot] + c * value * squared[slot];
            }
            out[row * cols + col] = acc;
        }
    }
    out
}

const _: () = same_order(&KERNELS, &nn::registry::REGISTRY);

/// 编译期断言：内核表与算子注册表逐条同名（长度已由类型锁住）。
const fn same_order(entries: &[Entry], registry: &[prof::catalogue::Registration]) {
    let mut idx = 0;
    while idx < entries.len() {
        assert!(
            same_str(entries[idx].op, registry[idx].name),
            "train::Reference 的内核表与 nn::registry::REGISTRY 名字或顺序漂移"
        );
        idx += 1;
    }
}

/// `const` 上下文逐字节比：唯一实现是 [`nn::same_bytes`]，本层只重导出别名，
/// 内核表与 [`nn::registry::REGISTRY`] 的名字对拍不再自带第二份比较逻辑。
pub(crate) use nn::same_bytes as same_str;

// ---- 形态、视图与输入拆包 --------------------------------------------------
//
// `f32` 与按位打包 `u8` 两条 lane 的静态契约：形态标签、只读/可写视图、以及按声明 lane 把输入
// 拆成切片的类型级表示。

/// 一块零值缓冲：分配失败是错误而不是 abort——长度可能来自不受信任的权重文件头。
pub(crate) fn zeroed<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len).map_err(|_| {
        let message = format!("要一块 {len} 个元素的缓冲区：超出可用内存或地址空间");
        Error::new(Code::Invalid, message).with(SITE)
    })?;
    buf.resize(len, value);
    Ok(buf)
}

/// 缓冲的存储形态：[`storage::Lane`] 的 train 别名；元素类型 → 形态的映射留在本层。
pub type Kind = storage::Lane;

/// 元素类型 → 存储形态；没有标量存储的类型（`F16` / `BF16` 等）为 `None`。
pub(crate) const fn lane_of(dtype: DType) -> Option<Kind> {
    match dtype {
        DType::F32 => Some(Kind::F32),
        DType::I8 | DType::U4 => Some(Kind::U8),
        _ => None,
    }
}

/// 密封：`Lane` 的合法实现者只有下面那两个，闭集与 [`DType`] 一一对应。
mod sealed {
    /// 不导出、也不可在 crate 外实现。
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for u8 {}
}

/// 一条存储 lane：`f32` 标量或按位打包的 `u8`，只承载"形态标签"与"按 lane 取切片"两件静态事实。
pub trait Lane: sealed::Sealed + Copy + Default + 'static {
    /// 该 lane 的存储形态；[`Inputs`] 用它把类型拼成常量签名，也用它报错。
    const KIND: Kind;

    /// 从输入视图里取这一条 lane 的切片；形态不符时为 `None`。
    fn slice<'b>(view: &'b View<'b>) -> Option<&'b [Self]>;

    /// 输出侧的同一条契约。
    fn slice_mut<'a, 'b: 'a>(view: &'a mut ViewMut<'b>) -> Option<&'a mut [Self]>;
}

/// 两条 lane 的取值：形态不符就是 `None`，与视图的变体一一对应。
macro_rules! lane {
    ($lane:ty, $kind:ident) => {
        impl Lane for $lane {
            const KIND: Kind = Kind::$kind;

            fn slice<'b>(view: &'b View<'b>) -> Option<&'b [Self]> {
                match view {
                    View::$kind(values) => Some(values.borrowed_slice()),
                    _ => None,
                }
            }

            fn slice_mut<'a, 'b: 'a>(view: &'a mut ViewMut<'b>) -> Option<&'a mut [Self]> {
                match view {
                    ViewMut::$kind(values) => Some(values.as_slice_mut()),
                    _ => None,
                }
            }
        }
    };
}

lane!(f32, F32);
lane!(u8, U8);

/// "这个元素类型没有标量存储"：`F16` / `BF16` 的唯一结论。
pub(crate) fn no_storage(dtype: DType) -> Error {
    let message = format!("{dtype:?} 没有标量存储：本层的缓冲只有 f32 与按位打包的 u8");
    Error::new(Code::Unsupported, message).with(SITE)
}

/// 一块真实缓冲：与登记的 [`DType`] 一一对应的两种形态（常驻值与激活共用）。
/// 公共定义在 [`storage::LaneBuf`]，本别名保留训练缝既有命名。
pub type Data = LaneBuf;

/// 只读视图：一次 [`Kernel::compute_step`] 的输入。公共定义在 [`storage::LaneView`]。
pub type View<'a> = LaneView<'a>;

/// 可写视图：一次 [`Kernel::compute_step`] 的输出。公共定义在 [`storage::LaneViewMut`]。
pub type ViewMut<'a> = LaneViewMut<'a>;

/// 与登记的元数据是否相容：登记的 [`DType`] 折成形态后必须与这块缓冲同一条 lane。
pub(crate) fn fits(data: &Data, dtype: DType) -> bool {
    lane_of(dtype) == Some(data.lane())
}

/// 一组输入 lane 的**类型级**表示：`(f32, u8, f32)` ↔ `(&[f32], &[u8], &[f32])`。
pub trait Inputs {
    /// 拆包后的借用形态。
    type Slices<'a>;

    /// 逐格 lane，顺序即参数顺序。
    const KINDS: &'static [Kind];

    /// 按声明的 lane 把 `inputs` 拆成切片；元数或某一格形态不符时返回 [`Code::Invalid`]。
    fn unpack<'a>(inputs: &'a [View<'a>], op: &str) -> Result<Self::Slices<'a>>;
}

/// 某一格不是内核要的 lane。
fn wrong_lane(op: &str, slot: &str, expected: Kind, actual: Kind) -> Error {
    let slot = format!("{slot}是 {}，内核要 {}", actual.name(), expected.name());
    Error::new(Code::Invalid, format!("`{op}` 的{slot}")).with(SITE)
}

/// 把逐格 lane 拼成 `f32, u8, f32`。
fn joined(kinds: impl Iterator<Item = Kind>) -> String {
    kinds.map(Kind::name).collect::<Vec<_>>().join(", ")
}

/// 元数不符：把两边逐格的 lane 摆在一起。
fn wrong_arity(op: &str, expected: &[Kind], inputs: &[View<'_>]) -> Error {
    let want = joined(expected.iter().copied());
    let got = joined(inputs.iter().map(View::lane));
    Error::new(
        Code::Invalid,
        format!("`{op}` 的输入必须是 [{want}]，实际是 [{got}]"),
    )
    .with(SITE)
}

/// 取第 `idx` 格输入并断言它是 `L`。
pub(crate) fn input_slice<'a, L: Lane>(
    op: &str,
    idx: usize,
    view: &'a View<'a>,
) -> Result<&'a [L]> {
    let slot = format!("第 {idx} 个输入");
    L::slice(view).ok_or_else(|| wrong_lane(op, &slot, L::KIND, view.lane()))
}

/// 取输出并断言它是 `L`。
pub(crate) fn output_slice<'a, L: Lane>(
    op: &str,
    view: &'a mut ViewMut<'_>,
) -> Result<&'a mut [L]> {
    let actual = view.lane();
    L::slice_mut(view).ok_or_else(|| wrong_lane(op, "输出", L::KIND, actual))
}

/// 元数 1..=3 的拆包：元数、顺序与形态在同一次单态化里核对。
macro_rules! inputs {
($(($lane:ident, $value:ident): $idx:literal),+ $(,)?) => {
    impl<$($lane: Lane),+> Inputs for ($($lane,)+) {
        type Slices<'a> = ($(&'a [$lane],)+);
        const KINDS: &'static [Kind] = &[$($lane::KIND),+];

        fn unpack<'a>(inputs: &'a [View<'a>], op: &str) -> Result<Self::Slices<'a>> {
            let [$($value),+] = inputs else {
                return Err(wrong_arity(op, Self::KINDS, inputs));
            };
            Ok(($(input_slice::<$lane>(op, $idx, $value)?,)+))
        }
    }
};
}

inputs!((A, a): 0);
inputs!((A, a): 0, (B, b): 1);
inputs!((A, a): 0, (B, b): 1, (C, c): 2);
