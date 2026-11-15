//! 逻辑张量：`DType` + `Shape` + `Strides` + 只读字节视图 [`Tensor`] 与拥有形态 [`OwnedTensor`]。
//!
//! **拥有**：`Tensor`、`OwnedTensor`、`View`、`Shape`、`Strides`、`Element` 与计算层子集 [`SUPPORTED`]。**不拥有**：
//! 元素类型闭集（L1 `dtype`，这里只再导出 `DType`）、kernel 侧的操作数视图
//! （DSL 的宿主 API，两者语义不同，见 `docs/adr/0003-kernel-and-runtime-types.md`）、
//! 布局搬运（`storage`）。**依赖**：`dtype`、`diagnostic`。
//! **被依赖**：`storage`、`weights`、`kv`、`autograd`、`data`、`nn`、`engine`、`llm`。
//!
//! 这个类型只描述"数据长什么样、放在哪几个字节里"，**不做 `&mut` 分区**——分区是 DSL 的所有权
//! 规则（cutile README §4.4，P0 尚未实现），两套并发规则会互相打架。
//!
//! # 例子
//!
//! ```
//! use llm_tensor::{DType, Shape, Tensor};
//!
//! let shape = Shape::new([2, 3]);
//! let bytes = [0u8; 24]; // 2 * 3 个 f32
//! let tensor = Tensor::new(DType::F32, shape, &bytes).expect("形状与缓冲匹配");
//! assert_eq!(tensor.numel(), 6);
//! assert_eq!(tensor.strides().as_slice(), [3, 1], "行主序连续布局");
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};

// 元素类型的唯一事实源在 L1 `dtype`：全精度 22 变体，这里不再自定义枚举。
pub use dtype::DType;

/// 计算层当前可消费的元素子集：闭集里的其他类型可以被容器存储，却不能构造进运行时张量。
///
/// 旧设计里 `DType` 本身只有 5 个变体、类型系统天然收口；枚举上移到 L1 后，改用这张
/// 子集表在构造点显式校验，新增一种被内核支持的类型只需在这里加一项。
pub const SUPPORTED: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::I8, DType::U4];

/// 判定某个 `dtype` 是否属于计算层支持子集。
#[must_use]
pub fn supports(dtype: DType) -> bool {
    SUPPORTED.contains(&dtype)
}

/// 形状：每个轴的长度。空形状表示标量，`numel()` 为 1。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// 由各轴长度构造。
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// 轴数（rank）。标量为 0。
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// 某一轴的长度；越界返回 `None`。
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// 元素个数：各轴长度之积，空形状为 1。
    ///
    /// 溢出时**饱和**到 `usize::MAX`，不回绕：形状可以来自权重文件头这类不受信任的输入，
    /// 回绕会让一个巨大形状报出很小的 `numel`（例如 `[2^32, 2^32]` 得到 0），于是
    /// [`Tensor::new`] / [`View::new`] 的"缓冲够长"判定被绕过，得到一个形状与缓冲不符的
    /// 张量。饱和之后下游只会按"缓冲不够"拒绝，而 `checked_mul` 的调用点照旧能看见溢出。
    pub fn numel(&self) -> usize {
        self.0
            .iter()
            .fold(1usize, |acc, dim| acc.saturating_mul(*dim))
    }

    /// 该元素类型下需要的字节数（按位向上取整，`U4` 会小于 `numel`）。
    ///
    /// 公式的唯一实现在 L1 的 [`DType::bytes_for`]：这里只把 `usize` 的元素数搬进它接受
    /// 的 `u64`，再把结果搬回来——两处各写一遍 `(numel * bits).div_ceil(8)` 迟早会漂移，
    /// 而装载方的缓冲校验正是按 `numel` 与字节数比对。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        let numel = u64::try_from(self.numel()).unwrap_or(u64::MAX);
        usize::try_from(dtype.bytes_for(numel)).unwrap_or(usize::MAX)
    }
}

/// 步长：沿每个轴前进一个下标要走多少个元素。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<usize>);

impl Strides {
    /// 由各轴步长构造（不做合法性校验：非连续布局也是合法布局）。
    pub fn new(strides: impl Into<Vec<usize>>) -> Self {
        Self(strides.into())
    }

    /// 行主序连续布局：最后一轴步长为 1，向前累乘。
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![0; dims.len()];
        let mut acc = 1usize;
        for axis in (0..dims.len()).rev() {
            strides[axis] = acc;
            acc = acc.saturating_mul(dims[axis]);
        }
        Self(strides)
    }

    /// 各轴步长。
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// 某一轴的步长；越界返回 `None`。
    pub fn get(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }
}

/// 逻辑张量：元数据 + 只读字节视图。
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// 由元数据与连续布局的缓冲构造：`dtype` 必须在 [`SUPPORTED`] 子集内、`data` 必须够长。
    pub fn new(dtype: DType, shape: Shape, data: &'a [u8]) -> Result<Self> {
        if !supports(dtype) {
            return Err(Error::new(
                Code::Invalid,
                format!("{dtype:?} 不在计算层支持的元素子集内"),
            )
            .with("tensor::Tensor::new"));
        }
        let required = shape.element_bytes(dtype);
        if data.len() < required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {} 字节，{dtype:?} 的 {:?} 需要 {required} 字节",
                    data.len(),
                    shape.dims()
                ),
            )
            .with("tensor::Tensor::new"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            dtype,
            shape,
            strides,
            data,
        })
    }

    /// 元素类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 只读字节视图。
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }
}

/// 元素类型：把 Rust 标量映射到 [`DType`] 闭集，供 [`View`] 做单态化视图。
///
/// 只实现到 `DType` 已有的宽度：`f16` / `bf16` 没有稳定的 Rust 标量类型，
/// `U4` 是按位打包的存储形态（不是一种元素宽度），两者都不适合做视图的元素类型。
///
/// 映射必须**逐个精确**：`u8` 就是 [`DType::U8`]，不是 `I8`。量化权重常以 `u8` 缓冲搬运，
/// 但那是装载方的字节视图（[`Tensor`]），“语义是有符号 int8” 得在反量化时显式说出来；
/// 让无符号标量冒充有符号类型，会让上层按键取内核时静默选错分支。
pub trait Element: Copy + 'static {
    /// 对应的元素类型。
    const DTYPE: DType;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
}

impl Element for i8 {
    const DTYPE: DType = DType::I8;
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;
}

/// 泛型只读视图：形状 + 步长 + 调用方的 `&[E]`。
///
/// 与 [`Tensor`] 的分工：`Tensor` 是**字节**视图（权重、打包类型、跨语言边界），
/// `View` 是**元素**视图（内核与算子直接计算）。`View` 不分配、不做类型双关、
/// 不装 `unsafe`：它只接受调用方已经持有的 `&[E]`，因此“零拷贝”是类型保证，
/// 不是注释承诺；代价是它不负责把字节转成元素（那是装载方的职责）。
#[derive(Debug, Clone)]
pub struct View<'a, E: Element> {
    shape: Shape,
    strides: Strides,
    data: &'a [E],
}

impl<'a, E: Element> View<'a, E> {
    /// 由形状与连续布局的缓冲构造：元素个数必须与形状一致。
    pub fn new(shape: Shape, data: &'a [E]) -> Result<Self> {
        let required = shape.numel();
        if data.len() != required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {} 个元素，形状 {:?} 需要 {required} 个",
                    data.len(),
                    shape.dims()
                ),
            )
            .with("tensor::View::new"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// 由任意合法步长构造：转置、行/列切片这类非连续布局也是合法视图。
    ///
    /// 校验的三件事：步长轴数必须与形状一致；零元素视图必须借用空缓冲（与 [`View::new`]
    /// 同一口径）；按最大下标算出的最高偏移必须落在 `data` 内。校验用 `checked_mul`，
    /// 形状异常时返回错误而不是溢出。
    pub fn strided(shape: Shape, strides: Strides, data: &'a [E]) -> Result<Self> {
        let view = Self {
            shape,
            strides,
            data,
        };
        view.check_rank()?;
        view.check_empty()?;
        view.check_reach()?;
        Ok(view)
    }

    /// 轴数：步长必须与形状逐轴对应，否则下面的偏移计算没有意义。
    fn check_rank(&self) -> Result<()> {
        if self.strides.as_slice().len() == self.shape.rank() {
            return Ok(());
        }
        Err(Self::invalid(format!(
            "步长有 {} 个轴，形状 {:?} 有 {} 个轴",
            self.strides.as_slice().len(),
            self.shape.dims(),
            self.shape.rank()
        )))
    }

    /// 零元素口径：形状里有 0 个元素时只接受空缓冲（与 [`View::new`] 同一口径）。
    ///
    /// 回答「零元素视图借了什么」，不回答「最高偏移落在哪」——没有下标可走。
    fn check_empty(&self) -> Result<()> {
        if self.shape.numel() != 0 || self.data.is_empty() {
            return Ok(());
        }
        Err(Self::invalid(format!(
            "零元素视图不该借用 {} 个元素的缓冲",
            self.data.len()
        )))
    }

    /// 可达性：每轴最后一位走出的总偏移必须算得出、且落在 `data` 内。
    ///
    /// 回答「这套步长能否在缓冲里走完」，不回答「布局是否连续」。
    fn check_reach(&self) -> Result<()> {
        if self.shape.numel() == 0 {
            return Ok(());
        }
        let last: Vec<usize> = self
            .shape
            .dims()
            .iter()
            .map(|dimension| dimension - 1)
            .collect();
        let highest = self.offset(&last).ok_or_else(|| {
            Self::invalid(format!(
                "形状 {:?} 与步长 {:?} 算出溢出",
                self.shape.dims(),
                self.strides.as_slice()
            ))
        })?;
        if highest < self.data.len() {
            return Ok(());
        }
        Err(Self::invalid(format!(
            "最高偏移 {highest} 超出缓冲（{} 个元素）：步长与形状对不上",
            self.data.len()
        )))
    }

    /// 本视图的 `Invalid` 错误：四处校验共用同一条构造点标记，文案由调用点给出。
    fn invalid(message: String) -> Error {
        Error::new(Code::Invalid, message).with("tensor::View::strided")
    }

    /// 元素类型（来自 [`Element::DTYPE`]，与 `DType` 闭集对齐）。
    pub fn dtype(&self) -> DType {
        E::DTYPE
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 调用方的缓冲本身（同一个指针，不是副本）。
    pub fn as_slice(&self) -> &'a [E] {
        self.data
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }

    /// 按下标取值；秩不符或越界返回 `None`（不 panic）。
    pub fn at(&self, index: &[usize]) -> Option<E> {
        self.data.get(self.offset(index)?).copied()
    }

    /// 单轴的偏移增量：下标必须落在该轴长度内，`position * step` 溢出则 `None`。
    ///
    /// 回答「这一轴走多远」，不回答「走到的是否是合法的整体偏移」。
    fn axis_offset(&self, axis: usize, position: usize) -> Option<usize> {
        let dimension = self.shape.dim(axis)?;
        let step = self.strides.get(axis)?;
        if position < dimension {
            position.checked_mul(step)
        } else {
            None
        }
    }

    /// 一组下标 → 元素偏移：秩不符、任一轴越界或累加溢出都是 `None`。
    ///
    /// 这是 [`View::at`]、[`View::row`] 与 [`View::strided`] 唯一的「下标 → 偏移」路径；
    /// 它回答「落在哪个偏移」，不回答「这个偏移是否在 `data` 内」——
    /// 缓冲边界由每个调用点按自己的口径判定。
    fn offset(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.rank() {
            return None;
        }
        index
            .iter()
            .enumerate()
            .try_fold(0usize, |acc, (axis, &position)| {
                acc.checked_add(self.axis_offset(axis, position)?)
            })
    }

    /// 二维视图的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
    ///
    /// 只在列步长为 1 时给出切片：`&[E]` 这个类型本身就承诺"这一段是连续的"，
    /// 非连续的行（例如转置视图的列）必须走 [`View::at`]，不能假装它是一片内存。
    pub fn row(&self, row: usize) -> Option<&'a [E]> {
        let [_, columns] = <[usize; 2]>::try_from(self.shape.dims()).ok()?;
        if self.strides.get(1) != Some(1) {
            return None;
        }
        let start = self.axis_offset(0, row)?;
        self.data.get(start..start.checked_add(columns)?)
    }
}

/// 拥有缓冲的可写张量：元数据 + 持有字节向量。
///
/// 与只读 [`Tensor<'a>`] 视图互补：`Tensor` 是调用方持有的零拷贝读路径，`OwnedTensor`
/// 是分配出来、可写入的拥有形态。`dtype` / `shape` / `strides` 与设备无关；设备地址
/// 藏在分配器返回的句柄里，本层不泄漏（ADR 0025 §2）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensor {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    bytes: Vec<u8>,
}

impl OwnedTensor {
    /// 按形状分配并清零；`dtype` 必须在 [`SUPPORTED`] 子集内。
    pub fn zeros(dtype: DType, shape: Shape) -> Result<Self> {
        let required = shape.element_bytes(dtype);
        Self::from_bytes(dtype, shape, vec![0u8; required])
    }

    /// 由已有字节向量构造：`dtype` 受支持且缓冲够长，布局取行主序连续。
    pub fn from_bytes(dtype: DType, shape: Shape, bytes: Vec<u8>) -> Result<Self> {
        if !supports(dtype) {
            return Err(Error::new(
                Code::Invalid,
                format!("{dtype:?} 不在计算层支持的元素子集内"),
            )
            .with("tensor::OwnedTensor::from_bytes"));
        }
        let required = shape.element_bytes(dtype);
        if bytes.len() < required {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 {} 字节，{dtype:?} 的 {:?} 需要 {required} 字节",
                    bytes.len(),
                    shape.dims()
                ),
            )
            .with("tensor::OwnedTensor::from_bytes"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            dtype,
            shape,
            strides,
            bytes,
        })
    }

    /// 元素类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 只读字节。
    pub fn data(&self) -> &[u8] {
        &self.bytes
    }

    /// 可写字节。
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// 借出只读字节视图（零拷贝读路径）；元数据构造时已校验缓冲够长，这里只做一次同口径复核。
    pub fn view(&self) -> Result<Tensor<'_>> {
        Tensor::new(self.dtype, self.shape.clone(), &self.bytes)
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }
}

#[cfg(test)]
mod tests;
