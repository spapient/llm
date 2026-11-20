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

    /// 按算子实例几何算一步前向。默认忽略几何、退回声明档语义：只有能解析实例几何的执行器
    /// 才覆盖它，其余实现保持既有行为，不因新增一条通道而被迫改动。
    fn compute_shaped(
        &mut self,
        op: &str,
        inputs: &[View<'_>],
        output: &mut ViewMut<'_>,
        geometry: Option<&crate::Geometry>,
    ) -> Result<()> {
        let _ = geometry;
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
    ) -> Result<()> {
        match geometry {
            Some(geometry) => self.shaped(op, inputs, output, geometry),
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
    ) -> Result<()> {
        let mut slices = Vec::with_capacity(inputs.len());
        for (index, view) in inputs.iter().enumerate() {
            let slice = view.f32().ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!("`{op}` 的第 {index} 个输入不是 f32"),
                )
                .with(SITE)
            })?;
            slices.push(slice);
        }
        geometry.forward(&slices, output_slice::<f32>(op, output)?)
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

/// 归一化 `[x, gamma, beta] → [y]` 适配：三个 f32 输入、一个 f32 输出；
/// 声明档几何（通道数、空间、批大小）固定在算子模块里，由层侧的真实几何另行决定。
macro_rules! norm {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
            let (x, gamma, beta) =
                <(f32, f32, f32) as Inputs>::unpack(inputs, stringify!($run))?;
            op::$run::forward(x, gamma, beta, output_slice::<f32>(stringify!($run), output)?)
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
    adaptive_pool_f32,
    avg_pool2d_f32,
    exp_f32,
    gelu_f32,
    hard_sigmoid_f32,
    leaky_relu_f32,
    interpolate_f32,
    log_f32,
    max_pool1d_f32,
    max_pool2d_f32,
    silu_f32,
    softmax_f32,
    reduce_max_f32,
    reduce_sum_f32,
    scale_f32,
    scan_f32
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
    batch_norm_f32,
    group_norm_f32,
    instance_norm_f32,
    layer_norm_f32
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

/// 大 K 稠密 GEMM：收缩轴取生产档 [`op::matmul::K`]，行数由输入长度推出。
///
/// 与上面的 `gemm!` 区别只在收缩轴不再固定为 `K = 8`：这里给的是 prefill 主链路的形状，
/// 尺寸校验交给算子自己的入口（三条缓冲长度必须同时自洽）。
fn matmul_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a, b) = <(f32, f32) as Inputs>::unpack(inputs, "matmul_f32")?;
    let out = output_slice::<f32>("matmul_f32", output)?;
    let (m, n) = matrix_geometry("matmul_f32", a.len(), out.len(), op::matmul::K)?;
    op::matmul::forward_tiled(a, b, m, n, op::matmul::K, out)
}

/// 大 K 对称 int8 GEMM：输入是承载有符号码的 `u8`，`i32` 精确累加后转 `f32`。
///
/// 与 [`op::matmul_i8::reference`] 同一段算术（后者从样本读，这里从视图读）；累加器是 i32，
/// 所以对拍逐位相等，不像 `f32` 两版要留容差。
fn matmul_i8(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let (a, b) = <(u8, u8) as Inputs>::unpack(inputs, "matmul_i8")?;
    let out = output_slice::<f32>("matmul_i8", output)?;
    let (m, n) = matrix_geometry("matmul_i8", a.len(), out.len(), op::matmul_i8::K)?;
    let mut acc = vec![0i32; m * n];
    let (a, b) = (signed(a), signed(b));
    op::matmul_i8::forward_tiled(&a, &b, m, n, op::matmul_i8::K, &mut acc)?;
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
///
/// 稠密与 int8 两条参考内核共用同一处推导——收缩轴 `k` 是两者唯一的差别，算式不该抄两遍。
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
    attention_paths(inputs, output, op::attention_tiled::forward)
}

/// 同形参，走 online softmax 的分块 tile 版（`op::attention_tiled::flash_tiled_forward`）。
fn flash_attention_f32(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    attention_paths(inputs, output, op::attention_tiled::flash_tiled_forward)
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
    let mut index = 0;
    while index < entries.len() {
        assert!(
            same_str(entries[index].op, registry[index].name),
            "train::Reference 的内核表与 op::REGISTRY 名字或顺序漂移"
        );
        index += 1;
    }
}

/// `const` 上下文逐字节比：唯一实现是 [`op::same_bytes`]，本层只重导出别名，
/// 内核表与 [`op::REGISTRY`] 的名字对拍不再自带第二份比较逻辑。
pub(crate) use op::same_bytes as same_str;

pub mod shape {
    //! 缓冲形态、视图与输入拆包：`f32` 与按位打包 `u8` 两条 lane 的静态契约。
    //!
    //! 这里只承载"元素类型 → 形态"、不可变/可变视图与元数 1..=3 的类型级拆包；算子适配在
    //! [`crate::kernel`]，分配复用在 [`crate::pool`]。

    use crate::store::SITE;
    use diagnostic::{Code, Error, Result};
    use tensor::DType;

    /// 一块零值缓冲：分配失败是错误而不是 abort——长度可能来自不受信任的权重文件头。
    pub(crate) fn zeroed<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(len).map_err(|_| {
            let message = format!("要一块 {len} 个元素的缓冲区：超出可用内存或地址空间");
            Error::new(Code::Invalid, message).with(SITE)
        })?;
        buffer.resize(len, value);
        Ok(buffer)
    }

    /// 一块 lane 的字节数。
    fn bytes_of<L: Lane>(values: &[L]) -> u64 {
        size_of_val(values) as u64
    }

    /// 缓冲的存储形态：[`Lane`] 的运行时标签，只在"元素类型 → 形态"与视图标记两处出现。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Kind {
        /// 32 位浮点。
        F32,
        /// 按位打包的整型。
        U8,
    }

    impl Kind {
        /// 元素类型 → 存储形态；没有标量存储的类型（`F16` / `BF16` 等）为 `None`。
        pub(crate) const fn of(dtype: DType) -> Option<Self> {
            match dtype {
                DType::F32 => Some(Self::F32),
                DType::I8 | DType::U4 => Some(Self::U8),
                _ => None,
            }
        }

        /// 名字：只用于错误说明。
        const fn name(self) -> &'static str {
            match self {
                Self::F32 => "f32",
                Self::U8 => "u8",
            }
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
        fn slice<'b>(view: View<'b>) -> Option<&'b [Self]>;

        /// 输出侧的同一条契约。
        fn slice_mut<'a, 'b: 'a>(view: &'a mut ViewMut<'b>) -> Option<&'a mut [Self]>;
    }

    /// 两条 lane 的取值：形态不符就是 `None`，与视图的变体一一对应。
    macro_rules! lane {
        ($lane:ty, $kind:ident) => {
            impl Lane for $lane {
                const KIND: Kind = Kind::$kind;

                fn slice<'b>(view: View<'b>) -> Option<&'b [Self]> {
                    match view {
                        View::$kind(values) => Some(values),
                        _ => None,
                    }
                }

                fn slice_mut<'a, 'b: 'a>(view: &'a mut ViewMut<'b>) -> Option<&'a mut [Self]> {
                    match view {
                        ViewMut::$kind(values) => Some(values),
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
    #[derive(Debug, Clone, PartialEq)]
    pub enum Data {
        /// 32 位浮点。
        F32(Vec<f32>),
        /// 按位打包的整型（`I8` 或 `U4`），字节序与内核一致。
        U8(Vec<u8>),
    }

    impl Data {
        /// 这块缓冲的存储形态。
        pub fn kind(&self) -> Kind {
            match self {
                Self::F32(_) => Kind::F32,
                Self::U8(_) => Kind::U8,
            }
        }

        /// 字节数：`size_of_val` 在编译期就知道元素宽度，热路径上没有分支。
        pub fn bytes(&self) -> u64 {
            match self {
                Self::F32(values) => bytes_of(values),
                Self::U8(values) => bytes_of(values),
            }
        }

        /// 只读视图：[`crate::Kernel`] 看到的输入。
        pub fn view(&self) -> View<'_> {
            match self {
                Self::F32(values) => View::F32(values),
                Self::U8(values) => View::U8(values),
            }
        }

        /// 可变视图：[`crate::Kernel`] 写的输出。
        pub fn view_mut(&mut self) -> ViewMut<'_> {
            match self {
                Self::F32(values) => ViewMut::F32(values),
                Self::U8(values) => ViewMut::U8(values),
            }
        }

        /// 与登记的元数据是否相容：登记的 [`DType`] 折成形态后必须与这块缓冲同一条 lane。
        pub(crate) fn fits(&self, dtype: DType) -> bool {
            Kind::of(dtype) == Some(self.kind())
        }
    }

    /// 只读视图：一次 [`crate::Kernel::compute`] 的输入。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum View<'a> {
        /// 32 位浮点。
        F32(&'a [f32]),
        /// 按位打包的整型。
        U8(&'a [u8]),
    }

    impl<'a> View<'a> {
        /// 这一格是哪个存储形态。
        pub fn kind(&self) -> Kind {
            match self {
                Self::F32(_) => Kind::F32,
                Self::U8(_) => Kind::U8,
            }
        }

        /// 当成 `f32` 读；不是浮点缓冲时为 `None`。
        pub fn f32(&self) -> Option<&'a [f32]> {
            <f32 as Lane>::slice(*self)
        }

        /// 当成按位打包的整型读；不是 `u8` 缓冲时为 `None`。
        pub fn u8(&self) -> Option<&'a [u8]> {
            <u8 as Lane>::slice(*self)
        }
    }

    /// 可写视图：一次 [`crate::Kernel::compute`] 的输出。
    #[derive(Debug, PartialEq)]
    pub enum ViewMut<'a> {
        /// 32 位浮点。
        F32(&'a mut [f32]),
        /// 按位打包的整型。
        U8(&'a mut [u8]),
    }

    impl<'a> ViewMut<'a> {
        /// 这一格是哪个存储形态。
        pub fn kind(&self) -> Kind {
            match self {
                Self::F32(_) => Kind::F32,
                Self::U8(_) => Kind::U8,
            }
        }

        /// 当成 `f32` 写；不是浮点缓冲时为 `None`。
        pub fn f32(&mut self) -> Option<&mut [f32]> {
            <f32 as Lane>::slice_mut(self)
        }

        /// 当成按位打包的整型写；不是 `u8` 缓冲时为 `None`。
        pub fn u8(&mut self) -> Option<&mut [u8]> {
            <u8 as Lane>::slice_mut(self)
        }
    }

    /// 一组输入 lane 的**类型级**表示：`(f32, u8, f32)` ↔ `(&[f32], &[u8], &[f32])`。
    ///
    /// 元数与每一格的 lane 都是类型的一部分：长度/顺序/形态的拆包由这里的单态化实现完成，能核对的
    /// 都在编译期核对，不能核对的集中到一处报错。
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
        let got = joined(inputs.iter().map(View::kind));
        Error::new(
            Code::Invalid,
            format!("`{op}` 的输入必须是 [{want}]，实际是 [{got}]"),
        )
        .with(SITE)
    }

    /// 取第 `index` 格输入并断言它是 `L`。
    pub(crate) fn input_slice<'a, L: Lane>(
        op: &str,
        index: usize,
        view: &'a View<'a>,
    ) -> Result<&'a [L]> {
        let slot = format!("第 {index} 个输入");
        L::slice(*view).ok_or_else(|| wrong_lane(op, &slot, L::KIND, view.kind()))
    }

    /// 取输出并断言它是 `L`。
    pub(crate) fn output_slice<'a, L: Lane>(
        op: &str,
        view: &'a mut ViewMut<'_>,
    ) -> Result<&'a mut [L]> {
        let actual = view.kind();
        L::slice_mut(view).ok_or_else(|| wrong_lane(op, "输出", L::KIND, actual))
    }

    /// 元数 1..=3 的拆包：元数、顺序与形态在同一次单态化里核对。
    macro_rules! inputs {
    ($(($lane:ident, $value:ident): $index:literal),+ $(,)?) => {
        impl<$($lane: Lane),+> Inputs for ($($lane,)+) {
            type Slices<'a> = ($(&'a [$lane],)+);
            const KINDS: &'static [Kind] = &[$($lane::KIND),+];

            fn unpack<'a>(inputs: &'a [View<'a>], op: &str) -> Result<Self::Slices<'a>> {
                let [$($value),+] = inputs else {
                    return Err(wrong_arity(op, Self::KINDS, inputs));
                };
                Ok(($(input_slice::<$lane>(op, $index, $value)?,)+))
            }
        }
    };
}

    inputs!((A, a): 0);
    inputs!((A, a): 0, (B, b): 1);
    inputs!((A, a): 0, (B, b): 1, (C, c): 2);
}
