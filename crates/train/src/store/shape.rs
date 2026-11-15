//! 缓冲形态、视图与输入拆包：`f32` 与按位打包 `u8` 两条 lane 的静态契约。
//!
//! 这里只承载"元素类型 → 形态"、不可变/可变视图与元数 1..=3 的类型级拆包；算子适配在
//! [`super::kernel`]，分配复用在 [`super::pool`]。

use super::SITE;
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
    pub(crate) fn fits(&self, dtype: DType) -> bool {
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
