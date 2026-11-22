//! 编译期秩 + 元素类型张量：秩 `D` 与元素类型 `E` 都是编译期事实，形状与步长是栈上的
//! `[usize; D]`（零堆分配），秩错在编译期拦下。这是对 Burn `Tensor<const D, K>`「类型即约束」的
//! 采纳：保留其连续 / 非连续视图、stride、reshape、permute 全套布局能力，却不引入后端 trait 的
//! 方法爆炸——算子仍由 `op` 数据驱动、后端仍在运行期选择（ADR 0053 / ADR 0046）。
//!
//! 与运行期 [`Shape`](crate::Shape) / [`Strides`](crate::Strides)（`Vec<usize>`）并存：静态形状
//! （权重、固定激活、层定义）用这里的栈数组，运行期 batch / segment 几何仍走运行期元数据。

use diagnostic::{Code, Error, Result};

use crate::element::Element;
use crate::shape::element_bytes;
use crate::{DType, Order};

/// 编译期秩的形状：`D` 个轴长存在栈上。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape<const D: usize> {
    dims: [usize; D],
}

impl<const D: usize> Shape<D> {
    /// 由各轴长度构造。
    pub const fn new(dims: [usize; D]) -> Self {
        Self { dims }
    }

    /// 各轴长度。
    pub const fn dims(&self) -> &[usize; D] {
        &self.dims
    }

    /// 轴数（编译期常量 `D`）。
    pub const fn rank(&self) -> usize {
        D
    }

    /// 某一轴的长度；越界返回 `None`。
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    /// 元素个数：各轴长度之积，空形状（标量）为 1。
    pub fn numel(&self) -> usize {
        crate::product(&self.dims)
    }

    /// 该元素类型下需要的字节数（按位向上取整，`U4` 会小于 `numel`）。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        element_bytes(self.numel(), dtype)
    }

    /// 交换两轴的长度；轴越界返回 `None`。
    pub fn transpose(&self, a: usize, b: usize) -> Option<Self> {
        let mut dims = self.dims;
        if a < D && b < D {
            dims.swap(a, b);
            return Some(Self { dims });
        }
        None
    }

    /// 按 `order` 重排各轴长度；`order` 必须是 `0..D` 的一个排列。
    pub fn permute(&self, order: [usize; D]) -> Option<Self> {
        if !is_permutation(&order) {
            return None;
        }
        let mut dims = [0; D];
        for axis in 0..D {
            dims[axis] = self.dims[order[axis]];
        }
        Some(Self { dims })
    }

    /// 等元素数地改形状；元素数不符返回 `None`。
    pub fn reshape<const D2: usize>(&self, dims: [usize; D2]) -> Option<Shape<D2>> {
        let reshaped = Shape::<D2>::new(dims);
        (reshaped.numel() == self.numel()).then_some(reshaped)
    }
}

/// 编译期秩的步长：与 [`Shape`] 同秩的栈数组。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Strides<const D: usize> {
    steps: [usize; D],
}

impl<const D: usize> Strides<D> {
    /// 由各轴步长构造（不做合法性校验：非连续布局也是合法布局）。
    pub const fn new(steps: [usize; D]) -> Self {
        Self { steps }
    }

    /// 各轴步长。
    pub const fn steps(&self) -> &[usize; D] {
        &self.steps
    }

    /// 某一轴的步长；越界返回 `None`。
    pub fn get(&self, axis: usize) -> Option<usize> {
        self.steps.get(axis).copied()
    }

    /// 按排列顺序推出的连续步长：行主序反向累乘，列主序正向累乘。
    pub fn ordered(shape: &Shape<D>, order: Order) -> Self {
        let mut steps = [1; D];
        let mut acc = 1usize;
        match order {
            Order::RowMajor => {
                for (step, dim) in steps.iter_mut().zip(shape.dims().iter()).rev() {
                    *step = acc;
                    acc = acc.saturating_mul(*dim);
                }
            }
            Order::ColMajor => {
                for (step, dim) in steps.iter_mut().zip(shape.dims().iter()) {
                    *step = acc;
                    acc = acc.saturating_mul(*dim);
                }
            }
        }
        Self { steps }
    }

    /// 行主序连续布局：最后一轴步长为 1，向前累乘。
    pub fn contiguous(shape: &Shape<D>) -> Self {
        Self::ordered(shape, Order::RowMajor)
    }

    /// 是否为行主序连续布局：与 [`contiguous`](Self::contiguous) 同一套期望累积值。
    #[must_use]
    pub fn is_contiguous(&self, shape: &Shape<D>) -> bool {
        *self == Self::contiguous(shape)
    }

    /// 交换两轴的步长；轴越界返回 `None`。
    pub fn transpose(&self, a: usize, b: usize) -> Option<Self> {
        let mut steps = self.steps;
        if a < D && b < D {
            steps.swap(a, b);
            return Some(Self { steps });
        }
        None
    }

    /// 按 `order` 重排各轴步长；`order` 必须是 `0..D` 的一个排列。
    pub fn permute(&self, order: [usize; D]) -> Option<Self> {
        if !is_permutation(&order) {
            return None;
        }
        let mut steps = [0; D];
        for axis in 0..D {
            steps[axis] = self.steps[order[axis]];
        }
        Some(Self { steps })
    }
}

/// 编译期秩 + 元素类型的只读视图：形状、步长与调用方的 `&[E]`。
#[derive(Debug, Clone)]
pub struct Tensor<'a, const D: usize, E: Element> {
    shape: Shape<D>,
    strides: Strides<D>,
    data: &'a [E],
}

impl<'a, const D: usize, E: Element> Tensor<'a, D, E> {
    /// 由形状与连续布局的缓冲构造：元素个数必须与形状一致。
    pub fn new(dims: [usize; D], data: &'a [E]) -> Result<Self> {
        let shape = Shape::new(dims);
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
            .with("tensor::typed::Tensor::new"));
        }
        let strides = Strides::contiguous(&shape);
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// 由任意合法步长构造：转置、切片这类非连续布局也是合法视图。
    pub fn strided(shape: Shape<D>, strides: Strides<D>, data: &'a [E]) -> Result<Self> {
        let tensor = Self {
            shape,
            strides,
            data,
        };
        tensor.check_empty()?;
        tensor.check_reach()?;
        Ok(tensor)
    }

    /// 零元素口径：形状里有 0 个元素时只接受空缓冲（与 [`Tensor::new`] 同一口径）。
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
                self.strides.steps()
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
        let mut idx = [0usize; D];
        for (slot, dim) in idx.iter_mut().zip(self.shape.dims().iter()) {
            *slot = dim.checked_sub(1)?;
        }
        self.offset(idx)
    }

    /// 本视图的 `Invalid` 错误：三处校验共用同一条构造点标记，文案由调用点给出。
    fn invalid(message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, message).with("tensor::typed::Tensor::strided")
    }

    /// 形状。
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// 步长。
    pub fn strides(&self) -> &Strides<D> {
        &self.strides
    }

    /// 元素类型（泛型 `E::DTYPE`）。
    pub fn dtype(&self) -> DType {
        E::DTYPE
    }

    /// 元素个数。
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// 是否为行主序连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// 调用方的缓冲本身（同一个指针，不是副本）。
    pub fn as_slice(&self) -> &'a [E] {
        self.data
    }

    /// 按下标取值；任一轴越界返回 `None`（不 panic）。
    pub fn at(&self, idx: [usize; D]) -> Option<E> {
        self.data.get(self.offset(idx)?).copied()
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

    /// 一组下标 → 元素偏移：任一轴越界或累加溢出都是 `None`。
    fn offset(&self, idx: [usize; D]) -> Option<usize> {
        idx.into_iter()
            .enumerate()
            .try_fold(0usize, |acc, (axis, position)| {
                acc.checked_add(self.axis_offset(axis, position)?)
            })
    }

    /// 交换两轴：转置是零拷贝的元数据重排。
    pub fn transpose(&self, a: usize, b: usize) -> Option<Self> {
        let shape = self.shape.transpose(a, b)?;
        let strides = self.strides.transpose(a, b)?;
        Some(Self {
            shape,
            strides,
            data: self.data,
        })
    }

    /// 按 `order` 重排各轴：置换是零拷贝的元数据重排。
    pub fn permute(&self, order: [usize; D]) -> Option<Self> {
        let shape = self.shape.permute(order)?;
        let strides = self.strides.permute(order)?;
        Some(Self {
            shape,
            strides,
            data: self.data,
        })
    }

    /// 等元素数地改形状：仅连续布局可用（否则需先物化成连续缓冲）。
    pub fn reshape<const D2: usize>(&self, dims: [usize; D2]) -> Option<Tensor<'a, D2, E>> {
        if !self.is_contiguous() {
            return None;
        }
        let shape = Shape::<D2>::new(dims);
        if shape.numel() != self.numel() {
            return None;
        }
        let strides = Strides::contiguous(&shape);
        Some(Tensor {
            shape,
            strides,
            data: self.data,
        })
    }
}

impl<'a, E: Element> Tensor<'a, 2, E> {
    /// 二维视图的第 `row` 行；列不连续时返回 `None`。
    pub fn row(&self, row: usize) -> Option<&'a [E]> {
        if self.strides.get(1) != Some(1) {
            return None;
        }
        let start = self.axis_offset(0, row)?;
        self.data
            .get(start..start.checked_add(self.shape.dims()[1])?)
    }
}

/// `order` 是否是 `0..D` 的一个排列（每个值恰好出现一次）。
fn is_permutation<const D: usize>(order: &[usize; D]) -> bool {
    (0..D).all(|value| order.contains(&value))
}
