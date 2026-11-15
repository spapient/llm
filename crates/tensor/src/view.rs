//! 泛型只读元素视图：[`Shape`] + [`Strides`] + 调用方的 `&[E]`。
//!
//! 与 [`crate::Tensor`] 的分工：`Tensor` 是**字节**视图（权重、打包类型、跨语言边界），
//! `View` 是**元素**视图（内核与算子直接计算）。`View` 不分配、不做类型双关、
//! 不装 `unsafe`：它只接受调用方已经持有的 `&[E]`，因此"零拷贝"是类型保证，
//! 不是注释承诺；代价是它不负责把字节转成元素（那是装载方的职责）。

use diagnostic::{Code, Error, Result};

use crate::DType;
use crate::element::Element;
use crate::shape::Shape;
use crate::strides::Strides;

/// 泛型只读视图：形状 + 步长 + 调用方的 `&[E]`。
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
    fn invalid(message: impl Into<String>) -> Error {
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
        self.strides.is_contiguous(&self.shape)
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
