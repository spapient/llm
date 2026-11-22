//! 逻辑张量：`Tensor<B, R, K>`——存储、秩、元素类型各占一个泛型位。
//!
//! | 位置 | 约束 | 管什么 | 两种形态 |
//! |---|---|---|---|
//! | `B` | [`Store`] | 元素存在哪 | 借来的 `&'a [K]` / 自己持有的 [`Owned<K>`](Owned) |
//! | `R` | [`Rank`] | 秩从哪来 | `[usize; D]` 编译期定长 / `Vec<usize>` 运行期长度 |
//! | `K` | [`Element`] | 元素是什么 | `f32` / `i8` / `u8`，精度钉在类型上 |
//!
//! 与 Burn `Tensor<B, const D, K>` 只差两处，都是刻意的：`B` 是存储而不是后端算子集（算子由
//! `op` 数据驱动注册表给出，泛型派发会带来 `FloatTensorOps` 那种百方法接口，见 ADR 0053），
//! `R` 是 `[usize; D]` 的超集（运行期 batch / segment 几何也要进同一个类型，ADR 0046）。
//! 其余全部继承：连续与非连续布局、stride、transpose、permute、reshape 都是零拷贝的元数据
//! 重排；形状、元素数、可达性与溢出在构造期一次拦下。
//!
//! 偏移、可达性、连续性与取行只有 [`crate::layout`] 一份实现——两种秩、两种存储共用它。

use core::marker::PhantomData;

use diagnostic::Result;

use crate::invalid;

use crate::element::Element;
use crate::layout;
use crate::rank::Rank;
use crate::shape::element_bytes;
use crate::store::{Owned, Store, StoreMut};
use crate::{DType, product};

/// 逻辑张量：元数据 + 元素存储。
///
/// 两个默认参数给出运行期秩 + `f32`：`Tensor<&[f32]>` 是 2 维以内的任意秩 f32 借来视图，
/// 而 `Tensor<&[f32], [usize; 2], f32>` 把秩 2 钉进类型。
#[derive(Debug, Clone)]
pub struct Tensor<B, R = Vec<usize>, K = f32> {
    dims: R,
    steps: R,
    buf: B,
    /// 零尺寸，把 `K` 钉在类型里；`fn() -> K` 位置避免自动 trait 泄漏到 `Tensor`。
    marker: PhantomData<fn() -> K>,
}

/// 借用调用方缓冲的只读张量：运行期秩 + 借来的元素。
pub type View<'a, K> = Tensor<&'a [K], Vec<usize>, K>;

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 连续行主序张量：元素个数必须与形状一致。
    pub fn new(dims: impl Into<R>, buf: B) -> Result<Self> {
        Self::build(dims.into(), buf, "tensor::Tensor::new")
    }

    /// 任意合法步长：转置、行/列切片这类非连续布局也是合法张量。
    pub fn strided(dims: impl Into<R>, steps: impl Into<R>, buf: B) -> Result<Self> {
        let dims = dims.into();
        let steps = steps.into();
        layout::check(
            dims.as_slice(),
            steps.as_slice(),
            product(dims.as_slice()),
            buf.slice().len(),
            "tensor::Tensor::strided",
        )?;
        Ok(assemble(dims, steps, buf))
    }

    /// 两种形态共用的连续构造：先核元素数，再推连续步长。
    fn build(dims: R, buf: B, marker: &str) -> Result<Self> {
        let numel = product(dims.as_slice());
        let len = buf.slice().len();
        if len != numel {
            return Err(invalid(
                format!(
                    "缓冲 {len} 个元素，形状 {:?} 需要 {numel} 个（{:?} 共 {} 字节）",
                    dims.as_slice(),
                    K::DTYPE,
                    element_bytes(numel, K::DTYPE)
                ),
                marker,
            ));
        }
        let steps = R::contiguous(dims.as_slice()).ok_or_else(|| {
            invalid(
                format!("形状 {:?} 的轴数与秩见证不符", dims.as_slice()),
                marker,
            )
        })?;
        Ok(assemble(dims, steps, buf))
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// 各轴步长。
    pub fn strides(&self) -> &[usize] {
        self.steps.as_slice()
    }

    /// 轴数（rank）。标量为 0。
    pub fn rank(&self) -> usize {
        self.dims.as_slice().len()
    }

    /// 元素个数：各轴长度之积，标量为 1。
    pub fn numel(&self) -> usize {
        product(self.dims.as_slice())
    }

    /// 元素类型（`K` 的常量，不是字段）。
    pub fn dtype(&self) -> DType {
        K::DTYPE
    }

    /// 存储里的全部元素——借来视图就是调用方的缓冲本身，不是副本。
    pub fn as_slice(&self) -> &[K] {
        self.buf.slice()
    }

    /// 是否为行主序连续布局：期望步长由 [`Rank::contiguous`] 给出，与构造期同一套累积规则。
    pub fn is_contiguous(&self) -> bool {
        layout::is_contiguous(self.dims.as_slice(), self.steps.as_slice())
    }

    /// 按下标取值；秩不符或越界返回 `None`（不 panic）。
    pub fn at(&self, idx: impl AsRef<[usize]>) -> Option<K> {
        let offset = layout::offset(self.dims.as_slice(), self.steps.as_slice(), idx.as_ref())?;
        self.buf.slice().get(offset).copied()
    }

    /// 二维张量的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
    pub fn row(&self, row: usize) -> Option<&[K]> {
        layout::row(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.buf.slice(),
            row,
        )
    }

    /// 交换两轴：零拷贝的元数据重排，与存储形态无关。
    pub fn transpose(self, a: usize, b: usize) -> Option<Self> {
        Some(assemble(
            self.dims.swapped(a, b)?,
            self.steps.swapped(a, b)?,
            self.buf,
        ))
    }

    /// 按 `order` 重排各轴：置换是零拷贝的元数据重排。
    pub fn permute(self, order: impl AsRef<[usize]>) -> Option<Self> {
        let order = order.as_ref();
        if !is_permutation(order, self.rank()) {
            return None;
        }
        Some(assemble(
            self.dims.permuted(order)?,
            self.steps.permuted(order)?,
            self.buf,
        ))
    }

    /// 广播到更大的形状：目标秩不小于源秩，逐轴相等或源轴长为 1，否则 `None`。
    ///
    /// 被扩展的轴步长为 0——同一个元素被多个逻辑位置读到，零拷贝。步长 0 是转置与置换表达不出来的
    /// 布局，布局代数因此严格宽于"只有重排"。
    pub fn broadcast(self, dims: impl Into<R>) -> Option<Self> {
        let dims = dims.into();
        let lead = dims.as_slice().len().checked_sub(self.rank())?;
        let steps = dims
            .as_slice()
            .iter()
            .enumerate()
            .map(|(axis, &want)| -> Option<usize> {
                let Some(source) = axis.checked_sub(lead) else {
                    return Some(0); // 前导新轴
                };
                let have = self.dims.as_slice()[source];
                match (have, want) {
                    (have, want) if have == want => Some(self.steps.as_slice()[source]),
                    (1, _) => Some(0),
                    _ => None,
                }
            })
            .collect::<Option<Vec<usize>>>()?;
        let steps = R::from_slice(&steps)?;
        Some(assemble(dims, steps, self.buf))
    }

    /// 等元素数地改形状：仅连续布局可用（转置过的张量可以先转置回去，那一步仍是零拷贝）。
    pub fn reshape<R2: Rank>(self, dims: impl Into<R2>) -> Option<Tensor<B, R2, K>> {
        if !self.is_contiguous() {
            return None;
        }
        let dims = dims.into();
        if product(dims.as_slice()) != self.numel() {
            return None;
        }
        let steps = R2::contiguous(dims.as_slice())?;
        Some(assemble(dims, steps, self.buf))
    }

    /// 物化成连续缓冲：步长 0 的广播轴、非满长的窄视图都摊进一块紧凑内存，于是可以接着
    /// [`reshape`](Self::reshape)。这是本模块唯一真正搬数据的地方，O(numel)。
    #[must_use]
    pub fn to_contiguous(&self) -> Tensor<Owned<K>, R, K> {
        let shape = self.dims.as_slice();
        let source = self.buf.slice();
        let steps = R::contiguous(shape).unwrap_or_else(|| self.dims.clone());
        if self.is_contiguous() {
            return assemble(self.dims.clone(), steps, Owned::from(source.to_vec()));
        }
        let mut buf = Vec::with_capacity(crate::product(shape));
        let mut idx = vec![0usize; shape.len()];
        let mut at = 0usize;
        let axis_steps = self.steps.as_slice();
        for _ in 0..crate::product(shape) {
            // 构造时的 check_reach 已证明任一合法下标都落在缓冲内，故不走 `Option`。
            buf.push(source[at]);
            // 行主序进位，偏移跟着增减，省掉每次重算点积。
            for axis in (0..idx.len()).rev() {
                idx[axis] += 1;
                at += axis_steps[axis];
                if idx[axis] < shape[axis] {
                    break;
                }
                idx[axis] = 0;
                at -= axis_steps[axis] * shape[axis];
            }
        }
        assemble(self.dims.clone(), steps, Owned::from(buf))
    }
}

impl<'a, R, K> Tensor<&'a [K], R, K>
where
    R: Rank,
    K: Element,
{
    /// 沿 `axis` 零拷贝取 `[start, start + len)` 一段，返回借用同一缓冲的视图。
    ///
    /// 只对借来的存储开放：持有 [`Owned`] 的张量要分段就得复制，那不是"视图"。返回的视图按同一组
    /// 步长重新做一遍可达性校验——起点前移后仍落在调用方缓冲内。
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Option<View<'a, K>> {
        let dims = self.dims.as_slice();
        let steps = self.steps.as_slice().to_vec();
        if start.checked_add(len)? > *dims.get(axis)? {
            return None;
        }
        let skip = start.checked_mul(*steps.get(axis)?)?;
        let mut dims = dims.to_vec();
        dims[axis] = len;
        // 零元素口径与别处一致：空段借空缓冲，不借调用方的非空缓冲。
        let data = match len {
            0 => self.buf.get(..0)?,
            _ => self.buf.get(skip..)?,
        };
        Tensor::strided(dims, steps, data).ok()
    }
}

impl<R, K> Tensor<Owned<K>, R, K>
where
    R: Rank,
    K: Element,
{
    /// 按形状分配并清零。
    pub fn zeros(dims: impl Into<R>) -> Result<Self> {
        let dims = dims.into();
        let buf = Owned::zeros(product(dims.as_slice()))?;
        Self::build(dims, buf, "tensor::Tensor::zeros")
    }

    /// 可写元素。
    pub fn as_slice_mut(&mut self) -> &mut [K] {
        self.buf.slice_mut()
    }
}

/// 六处构造与重排共用同一个组装动作：元数据 + 存储 + 零尺寸类型标记。
fn assemble<B, R, K>(dims: R, steps: R, buf: B) -> Tensor<B, R, K> {
    Tensor {
        dims,
        steps,
        buf,
        marker: PhantomData,
    }
}

/// `order` 是否是 `0..rank` 的一个排列（每个值恰好出现一次）。
fn is_permutation(order: &[usize], rank: usize) -> bool {
    order.len() == rank && (0..rank).all(|value| order.contains(&value))
}
