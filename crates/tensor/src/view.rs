//! 泛型只读元素视图：[`Shape`] + [`Strides`] + 调用方的 `&[E]`。

use diagnostic::{Code, Error, Result};

use crate::DType;
use crate::element::Element;
use crate::shape::Shape;
use crate::shape::Strides;

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
    fn check_reach(&self) -> Result<()> {
        if self.shape.numel() == 0 {
            return Ok(());
        }
        let highest = self.highest_offset().ok_or_else(|| {
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

    /// 最高下标（每轴最后一位）走出的总偏移；任一轴越界或累加溢出则 `None`。
    fn highest_offset(&self) -> Option<usize> {
        self.shape
            .dims()
            .iter()
            .zip(self.strides.as_slice())
            .try_fold(0usize, |acc, (&dim, &step)| {
                let position = dim.checked_sub(1)?;
                acc.checked_add(position.checked_mul(step)?)
            })
    }

    /// 本视图的 `Invalid` 错误：四处校验共用同一条构造点标记，文案由调用点给出。
    fn invalid(message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, message).with("tensor::View::strided")
    }

    tensor_meta!(elem);

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
    pub fn row(&self, row: usize) -> Option<&'a [E]> {
        let [_, columns] = <[usize; 2]>::try_from(self.shape.dims()).ok()?;
        if self.strides.get(1) != Some(1) {
            return None;
        }
        let start = self.axis_offset(0, row)?;
        self.data.get(start..start.checked_add(columns)?)
    }
}
