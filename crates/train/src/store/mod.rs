//! 训练侧的激活缓冲：[`Buffers`] 把 [`Store`] 落到真实内存，算的部分交给 [`Kernel`]。
//!
//! **拥有**：`Buffers`、`Kernel`、`Reference`、`Data`、`View`、`ViewMut`，以及承载静态契约的
//! `Kind`、`Lane`、`Inputs`。**不拥有**：显存计划（`autograd::Plan`）、反向调度、形状表、算子语义
//! （`op`）。**依赖**：`autograd`、`tensor`、`op`、`diagnostic`。
//!
//! 三条硬约束：缓冲只有 `f32` 与按位打包的 `u8` 两种形态（`unsafe_code = deny`，"把字节重解释成
//! 浮点"这条路关着，`F16` / `BF16` 于是在编译期就被拒绝）；内核表与 [`op::REGISTRY`] 同名同序，
//! "注册了算子却没有参考实现"是编译错误；`live` / `peak` 都是实测值，与 [`autograd::Plan`] 的承诺
//! 对账，而计划保留的检查点必须在重放开始前就在场。
//!
//! # 例子
//!
//! ```
//! use autograd::{Sizes, Store, Tape};
//! use llm_train::{Buffers, Data, Reference};
//! use tensor::{DType, Shape};
//!
//! let mut tape = Tape::new();
//! let (x, w) = (tape.leaf(), tape.leaf());
//! let y = tape.record("gemv_f32", &[x, w]).expect("记录");
//! let mut sizes = Sizes::new(tape.vars());
//! sizes.set(x, Shape::new([2]), DType::F32);
//! sizes.set(w, Shape::new([2, 2]), DType::F32);
//! sizes.set(y, Shape::new([2]), DType::F32);
//!
//! let mut store = Buffers::new(sizes, Reference);
//! store.load(x, Data::F32(vec![1.0, 2.0])).expect("输入");
//! store.load(w, Data::F32(vec![1.0, 0.0, 0.0, 1.0])).expect("权重");
//! store.compute("gemv_f32", &[x, w], y).expect("可算");
//! assert!(store.value(y).and_then(|view| view.f32()).is_some());
//! ```

use autograd::{Sizes, Store, Var};
use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

/// 本模块的诊断站点：出错时用它标明"是这一层拒绝了这次请求"。
const SITE: &str = "train::Buffers";

/// 一块零值缓冲：分配失败是错误而不是 abort——长度可能来自不受信任的权重文件头。
fn zeroed<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
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
    const fn of(dtype: DType) -> Option<Self> {
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
fn no_storage(dtype: DType) -> Error {
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

    /// 只读视图：[`Kernel`] 看到的输入。
    pub fn view(&self) -> View<'_> {
        match self {
            Self::F32(values) => View::F32(values),
            Self::U8(values) => View::U8(values),
        }
    }

    /// 可变视图：[`Kernel`] 写的输出。
    pub fn view_mut(&mut self) -> ViewMut<'_> {
        match self {
            Self::F32(values) => ViewMut::F32(values),
            Self::U8(values) => ViewMut::U8(values),
        }
    }

    /// 与登记的元数据是否相容：登记的 [`DType`] 折成形态后必须与这块缓冲同一条 lane。
    fn fits(&self, dtype: DType) -> bool {
        Kind::of(dtype) == Some(self.kind())
    }
}

/// 只读视图：一次 [`Kernel::compute`] 的输入。
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

/// 可写视图：一次 [`Kernel::compute`] 的输出。
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
fn input_slice<'a, L: Lane>(op: &str, index: usize, view: &'a View<'a>) -> Result<&'a [L]> {
    let slot = format!("第 {index} 个输入");
    L::slice(*view).ok_or_else(|| wrong_lane(op, &slot, L::KIND, view.kind()))
}

/// 取输出并断言它是 `L`。
fn output_slice<'a, L: Lane>(op: &str, view: &'a mut ViewMut<'_>) -> Result<&'a mut [L]> {
    let actual = view.kind();
    L::slice_mut(view).ok_or_else(|| wrong_lane(op, "输出", L::KIND, actual))
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
    gemv_f32,
    mul_f32,
    rmsnorm_f32,
    residual_add_f32
);
gemm!(
    (gemm_f32, f32, op::gemm_f32::K),
    (gemm_i4, u8, op::gemm_i4::K / 2),
    (gemm_i8, u8, op::gemm_i8::K),
);

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
    dequant_i4,
    dequant_i8,
    embed_f32,
    flash_attention_f32,
    gelu_f32,
    gemm_f32,
    gemm_i4,
    gemm_i8,
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

/// 同尺寸缓冲的空闲表：按**正好相等的长度**分档，只复用等长的块（池容量永不超峰值）。
#[derive(Debug, Default)]
struct Pool {
    f32s: Vec<(usize, Vec<Vec<f32>>)>,
    u8s: Vec<(usize, Vec<Vec<u8>>)>,
}

/// 一条 lane 的三件套：取、还、数（查找是一次线性扫描，档数由形状的个数决定，通常是个位数）。
macro_rules! buckets {
    ($(($acquire:ident, $give:ident, $pooled:ident, $lane:ty, $pile:ident)),* $(,)?) => {$(
        impl Pool {
            /// 取一块长度正好是 `len` 的缓冲：池里有就复用（只对齐长度），否则零值分配。
            fn $acquire(&mut self, len: usize) -> Result<Vec<$lane>> {
                let pile = &mut self.$pile;
                let free = pile.iter_mut().find(|(size, _)| *size == len);
                match free.and_then(|(_, blocks)| blocks.pop()) {
                    Some(mut values) => {
                        values.truncate(len);
                        Ok(values)
                    }
                    None => zeroed(len, Default::default()),
                }
            }

            /// 还回一块空闲块，按它的长度归到对应档。
            fn $give(&mut self, buffer: Vec<$lane>) {
                let len = buffer.len();
                let pile = &mut self.$pile;
                match pile.iter_mut().find(|(size, _)| *size == len) {
                    Some((_, blocks)) => blocks.push(buffer),
                    None => pile.push((len, vec![buffer])),
                }
            }

            /// 档里现成可复用的块数。
            fn $pooled(&self) -> usize {
                self.$pile.iter().map(|(_, blocks)| blocks.len()).sum()
            }
        }
    )*};
}

buckets!(
    (acquire_f32, give_f32, pooled_f32, f32, f32s),
    (acquire_u8, give_u8, pooled_u8, u8, u8s),
);

impl Pool {
    /// 一块 `dtype` 的零值缓冲：先试空闲表，没有再向分配器要（"取多少元素"只在这里算一次）。
    fn block(&mut self, dtype: DType, shape: &Shape) -> Result<Data> {
        match Kind::of(dtype) {
            Some(Kind::F32) => Ok(Data::F32(self.acquire_f32(shape.numel())?)),
            Some(Kind::U8) => Ok(Data::U8(self.acquire_u8(shape.element_bytes(dtype))?)),
            None => Err(no_storage(dtype)),
        }
    }

    /// 把一块缓冲放回空闲表。
    fn recycle(&mut self, block: Data) {
        match block {
            Data::F32(values) => self.give_f32(values),
            Data::U8(values) => self.give_u8(values),
        }
    }

    /// 空闲表里的块数。
    fn pooled(&self) -> usize {
        self.pooled_f32() + self.pooled_u8()
    }
}

/// 以 [`Var`] 为下标的真实缓冲：`autograd::Store` 在训练侧的实现。
///
/// 一个训练步里只有一个实例，逐图复用：`load` 装常驻值、`compute` 算激活、`free` 还激活，
/// 步与步之间用 [`Buffers::clear`] 一次性归还；块本身来自内部空闲表，稳定态不再向系统要内存。
#[derive(Debug)]
pub struct Buffers<K: Kernel> {
    sizes: Sizes,
    kernel: K,
    slots: Vec<Option<Data>>,
    pool: Pool,
    live: u64,
    peak: u64,
}

impl<K: Kernel> Buffers<K> {
    /// 由形状表与内核构造：槽位数与 `sizes.len()` 同口径（即 `Tape::vars()`）。
    pub fn new(sizes: Sizes, kernel: K) -> Self {
        Self {
            slots: vec![None; sizes.len()],
            sizes,
            kernel,
            pool: Pool::default(),
            live: 0,
            peak: 0,
        }
    }

    /// 装入一个常驻值（参数、输入、或外部算好的结果）。
    ///
    /// 登记的元数据必须与实际数据一致：装错类型不会在训练中途才暴露成一串无意义的数；同一个
    /// 变量装第二次会先把旧的还回空闲表。
    ///
    /// # Errors
    ///
    /// 变量超出形状表的槽位、该变量没有登记形状与元素类型（都是 [`Code::Internal`]：调用方漏了
    /// `Sizes::set`）、数据与登记的（形状，元素类型）不相容（[`Code::Invalid`]）时返回说明。
    pub fn load(&mut self, var: Var, data: Data) -> Result<()> {
        let slot = self.slot(var)?;
        let Some((shape, dtype)) = self.sizes.get(var) else {
            let message = format!(
                "变量 {} 没有登记形状与元素类型：装入的数据无从校验（`Sizes` 必须按 `Tape::vars()` 构造并逐项 `set`）",
                var.id()
            );
            return Err(Error::new(Code::Internal, message).with(SITE));
        };
        if !data.fits(dtype) || data.bytes() != shape.element_bytes(dtype) as u64 {
            return Err(mismatch(var, shape, dtype, &data));
        }
        self.install(slot, data);
        Ok(())
    }

    /// 读一个变量当前的值；不在场（从未算过、或已被交还）时为 `None`。
    pub fn value(&self, var: Var) -> Option<View<'_>> {
        self.slots
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map(Data::view)
    }

    /// 当前占用的字节数（含常驻值）。
    pub fn live(&self) -> u64 {
        self.live
    }

    /// 占用字节数的峰值：与 [`autograd::Plan::peak`] 同口径之处是"都按真实占用量统计"，
    /// 差一处是这里也含常驻的参数与输入。
    pub fn peak(&self) -> u64 {
        self.peak
    }

    /// 把峰值重新对齐到当前占用：分段测量（前向一段、重放一段）时各测各的。
    pub fn reset_peak(&mut self) {
        self.peak = self.live;
    }

    /// 空闲表里现成可复用的块数（诊断用：稳定态应当不增长）。
    pub fn pooled(&self) -> usize {
        self.pool.pooled()
    }

    /// 一次性归零：换训练步、或换图时调用，空闲表与峰值统计一起重置。
    pub fn clear(&mut self) {
        self.slots.fill(None);
        self.pool = Pool::default();
        self.live = 0;
        self.peak = 0;
    }

    /// 形状表。
    pub fn sizes(&self) -> &Sizes {
        &self.sizes
    }

    /// 变量对应的槽位；越界说明 `Sizes` 与 `Tape` 不同源。
    fn slot(&self, var: Var) -> Result<usize> {
        let index = var.id() as usize;
        if index >= self.slots.len() {
            let message = format!(
                "变量 {} 超出形状表的 {} 个槽位：Sizes 必须按 Tape::vars() 构造",
                var.id(),
                self.slots.len()
            );
            return Err(Error::new(Code::Internal, message).with(SITE));
        }
        Ok(index)
    }

    /// 输出槽位与登记的（形状，元素类型）：两者缺一都说明 `Sizes` 与 `Tape` 不同源。
    fn target(&self, var: Var) -> Result<(usize, Shape, DType)> {
        let slot = self.slot(var)?;
        let (shape, dtype) = self.sizes.get(var).ok_or_else(|| unregistered(var))?;
        Ok((slot, shape.clone(), dtype))
    }

    /// 装入一块缓冲：交还旧的、记账、占住槽位。
    fn install(&mut self, slot: usize, block: Data) {
        self.release(slot);
        self.live += block.bytes();
        self.peak = self.peak.max(self.live);
        self.slots[slot] = Some(block);
    }

    /// 交还一个槽位：`live` 当场降下来，块回到空闲表。
    fn release(&mut self, slot: usize) {
        if let Some(block) = self.slots[slot].take() {
            self.live -= block.bytes();
            self.pool.recycle(block);
        }
    }
}

impl<K: Kernel> Store for Buffers<K> {
    fn compute(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()> {
        let (slot, shape, dtype) = self.target(output)?;
        let views = views(&self.slots, output, inputs)?;
        let mut block = self.pool.block(dtype, &shape)?;
        if let Err(error) = self.kernel.compute(op, &views, &mut block.view_mut()) {
            self.pool.recycle(block);
            return Err(error);
        }
        drop(views);
        self.install(slot, block);
        Ok(())
    }

    fn free(&mut self, var: Var) {
        let slot = var.id() as usize;
        if slot < self.slots.len() {
            self.release(slot);
        }
    }
}

/// 一次重算的全部输入视图：缺一个就是"重放读到了已释放的值"。
fn views<'a>(slots: &'a [Option<Data>], output: Var, inputs: &[Var]) -> Result<Vec<View<'a>>> {
    inputs
        .iter()
        .map(|input| view_of(slots, output, *input))
        .collect()
}

/// 一个输入视图；不在场时把两个编号都写进错误。
fn view_of<'a>(slots: &'a [Option<Data>], output: Var, input: Var) -> Result<View<'a>> {
    slots
        .get(input.id() as usize)
        .and_then(Option::as_ref)
        .map(Data::view)
        .ok_or_else(|| missing_input(output, input))
}

/// 没登记形状就分配不了缓冲：前向记录时漏了 `Sizes::set`。
fn unregistered(var: Var) -> Error {
    let message = format!(
        "变量 {} 没有登记形状，无法为它分配缓冲：前向记录时漏了 Sizes::set",
        var.id()
    );
    Error::new(Code::Internal, message).with(SITE)
}

/// 装入的数据与登记的（形状，元素类型）不相容。
fn mismatch(var: Var, shape: &Shape, dtype: DType, data: &Data) -> Error {
    let label = match data {
        Data::F32(_) => " f32",
        Data::U8(_) => " u8",
    };
    let message = format!(
        "变量 {} 登记为 {dtype:?} 的 {:?}（{} 字节），装入的却是 {} 字节的{label:?}",
        var.id(),
        shape.dims(),
        shape.element_bytes(dtype),
        data.bytes()
    );
    Error::new(Code::Invalid, message).with(SITE)
}

/// 一个输入不在场：重放读到了已交还（或从未算过）的值。
fn missing_input(output: Var, input: Var) -> Error {
    let message = format!(
        "重算 {} 时输入 {} 不在场（已被交还或从未算过）",
        output.id(),
        input.id()
    );
    Error::new(Code::Missing, message).with(SITE)
}

#[cfg(test)]
mod tests;
