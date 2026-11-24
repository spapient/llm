//! 逻辑张量：`Tensor<B, R, K>`——存储、秩、元素类型各占一个泛型位。
//!
//! | 位置 | 约束 | 管什么 | 两种形态 |
//! |---|---|---|---|
//! | `B` | [`Store`] | 元素存在哪 | 借来的 `&'a [K]` / 自己持有的 [`Owned<K>`](Owned) |
//! | `R` | [`Rank`] | 秩从哪来 | `[usize; D]` 编译期定长 / `Vec<usize>` 运行期长度 |
//! | `K` | [`Element`] | 元素是什么 | `f32` / `i8` / `u8`，精度钉在类型上 |
//!
//! 与 Burn `Tensor<B, const D, K>` 只差两处，都是刻意的：`B` 是存储而不是后端算子集（算子由
//! `nn` 数据驱动注册表给出，泛型派发会带来 `FloatTensorOps` 那种百方法接口，见 ADR 0053），
//! `R` 是 `[usize; D]` 的超集（运行期 batch / segment 几何也要进同一个类型，ADR 0046）。
//! 其余全部继承：连续与非连续布局、stride、transpose、permute、reshape 都是零拷贝的元数据
//! 重排；形状、元素数、可达性与溢出在构造期一次拦下。
//!
//! 本文件只放类型定义、构造、只读访问器与组装动作；布局/形状算子在 [`shape`](shape)，可写访问
//! 与自持分配、窗口写在 [`owned`](owned)。

use core::marker::PhantomData;
use diagnostic::Result;

use crate::invalid;

use crate::element::Element;
use crate::layout;
use crate::rank::Rank;
use crate::shape::element_bytes;
use crate::store::{Owned, Store};
use crate::{DType, product};

mod owned;
mod shape;

/// 逻辑张量：元数据 + 元素存储。
///
/// 两个默认参数给出运行期秩 + `f32`：`Tensor<&[f32]>` 是 2 维以内的任意秩 f32 借来视图，
/// 而 `Tensor<&[f32], [usize; 2], f32>` 把秩 2 钉进类型。
#[derive(Debug, Clone)]
pub struct Tensor<B, R = Vec<usize>, K = f32> {
    pub(crate) dims: R,
    pub(crate) steps: R,
    /// 起点偏移：下标 0 对应的缓冲位置。子视图不改缓冲、只把它前移。
    pub(crate) base: usize,
    pub(crate) buf: B,
    /// 零尺寸，把 `K` 钉在类型里；`fn() -> K` 位置避免自动 trait 泄漏到 `Tensor`。
    marker: PhantomData<fn() -> K>,
}

/// 借用调用方缓冲的只读张量：运行期秩 + 借来的元素。
pub type View<'a, K> = Tensor<&'a [K], Vec<usize>, K>;

/// 借用调用方缓冲的可写张量：运行期秩 + 借来的可写元素，用作算子输出操作数。
pub type ViewMut<'a, K> = Tensor<&'a mut [K], Vec<usize>, K>;

/// 擦除后的自持张量：存储自持、秩运行期——后端进出的统一形态（对标 Burn 的 `BridgeTensor`）。
///
/// 编译期秩与借用存储在装配边界 downgrade 成这一形态；设备后端只认这一张量类型进出，
/// 不再区分视图来自调用方缓冲还是自持分配。同步取出数据走 [`Tensor::to_vec`] /
/// `into_vec`；异步出口在执行缝侧接流，本层不持有设备句柄。
pub type DynTensor<K> = Tensor<Owned<K>, Vec<usize>, K>;

impl<'a, K: Element> View<'a, K> {
    /// 一维只读视图：唯一轴长即缓冲长度，行主序连续。
    ///
    /// 逐元素 / 归约这类单操作数内核，输入本来就是一段连续切片；几何只有一个长度，由缓冲自身
    /// 给出，构造期不可能失败，因此不走 [`Tensor::new`](Self::new) 的 `Result`。多维几何必须
    /// 显式传入形状，仍走 `new` 在校验拦下轴数与元素数不符。
    pub fn one(slice: &'a [K]) -> Self {
        assemble(vec![slice.len()], vec![1], 0, slice)
    }

    /// 二维只读视图：`rows × cols` 行主序连续。
    ///
    /// 矩阵族内核的输入本来就是一段连续二维切片，但两条轴长都要由调用方显式给出（不像一维
    /// [`View::one`] 能从缓冲长度反推唯一轴），因此仍走 [`Tensor::new`](Self::new) 的 `Result`，
    /// 在校验拦下元素数与 `rows·cols` 不符。
    pub fn matrix(rows: usize, cols: usize, slice: &'a [K]) -> Result<Self> {
        Self::new(vec![rows, cols], slice)
    }

    /// 三条同宽二维只读视图：`q/k/v` 都是 `rows×cols`，注意力族 prefill/decode 输入的公共装配。
    pub fn matrix_triple(
        rows: usize,
        cols: usize,
        q: &'a [K],
        k: &'a [K],
        v: &'a [K],
    ) -> Result<(Self, Self, Self)> {
        Ok((
            Self::matrix(rows, cols, q)?,
            Self::matrix(rows, cols, k)?,
            Self::matrix(rows, cols, v)?,
        ))
    }

    /// 拆出内部借用切片，保留构造视图时的借用生命周期 `'a`。
    ///
    /// [`as_slice`](Self::as_slice) 的生命周期绑在 `&self` 上；执行缝要把视图再交回调用方缓冲
    /// 时需要保留到原始借用 `'a`，走本入口。内部引用是 `Copy`，不会挪走视图本身。
    pub fn borrowed_slice(&self) -> &'a [K] {
        self.buf
    }
}

impl<'a, K: Element> ViewMut<'a, K> {
    /// 一维可写视图：唯一轴长即缓冲长度，行主序连续。口径同 [`View::one`]，只是存储可写。
    pub fn one(slice: &'a mut [K]) -> Self {
        assemble(vec![slice.len()], vec![1], 0, slice)
    }

    /// 二维可写视图：`rows × cols` 行主序连续。口径同 [`View::matrix`]，只是存储可写，
    /// 用作矩阵族算子的输出操作数。
    pub fn matrix(rows: usize, cols: usize, slice: &'a mut [K]) -> Result<Self> {
        Self::new(vec![rows, cols], slice)
    }

    /// 消费可写视图，取出内部可写借用切片；形状与步长元数据随之丢弃。
    ///
    /// 入口内核只按连续切片写结果时，用本入口把视图扁平化：视图是一次性输出操作数，
    /// 消费它取切片不留下第二个别名，也不需要再经 [`as_slice_mut`](Self::as_slice_mut)
    /// 走可变借用绑定。
    pub fn into_slice_mut(self) -> &'a mut [K] {
        self.buf
    }
}

impl<B, R, K> PartialEq for Tensor<B, R, K>
where
    R: Rank,
    K: Element + PartialEq,
    B: Store<K>,
{
    /// 形状逐轴一致，且逻辑序元素逐一相等；非连续布局也按 [`to_vec`](Self::to_vec) 的逻辑序比较。
    fn eq(&self, other: &Self) -> bool {
        self.dims() == other.dims() && self.to_vec() == other.to_vec()
    }
}

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
        Ok(assemble(dims, steps, 0, buf))
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
        Ok(assemble(dims, steps, 0, buf))
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// 各轴步长。
    pub fn strides(&self) -> &[usize] {
        self.steps.as_slice()
    }

    /// 起点偏移：下标 0 在存储里的位置。换行主序视图对象共享同一块缓冲时靠它区分。
    pub fn base(&self) -> usize {
        self.base
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
    ///
    /// 这是**存储**而不是"本张量的元素"：子视图只覆盖其中 `base .. base + 最高偏移` 一段。
    /// 要本张量的元素依次读出，用 [`to_vec`](Self::to_vec)；要一块可接着 [`reshape`](Self::reshape) 的存储，用 [`to_contiguous`](Self::to_contiguous)。
    pub fn as_slice(&self) -> &[K] {
        self.buf.slice()
    }

    /// 是否为行主序连续布局：与连续步长逐轴核对。长度为 1 的轴不参与核对（它的步长不贡献偏移）。
    pub fn is_contiguous(&self) -> bool {
        layout::is_contiguous(self.dims.as_slice(), self.steps.as_slice())
    }

    /// 按下标取值；秩不符或越界返回 `None`（不 panic）。
    pub fn at(&self, idx: impl AsRef<[usize]>) -> Option<K> {
        let offset = layout::offset(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.base,
            idx.as_ref(),
        )?;
        self.buf.slice().get(offset).copied()
    }

    /// 二维张量的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
    pub fn row(&self, row: usize) -> Option<&[K]> {
        layout::row(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.base,
            self.buf.slice(),
            row,
        )
    }

    /// 形状照抄、步长重算、起点归零，配上一块新的连续缓冲：摊平与翻转共用这一处收尾。
    pub(crate) fn respell(&self, buf: Vec<K>) -> Tensor<Owned<K>, R, K> {
        let steps = R::contiguous(self.dims.as_slice()).unwrap_or_else(|| self.dims.clone());
        assemble(self.dims.clone(), steps, 0, Owned::from(buf))
    }
}

/// 六处构造与重排共用同一个组装动作：元数据 + 起点 + 存储 + 零尺寸类型标记。
/// 把布局装进秩见证，存储原样带上：`respine`（吃掉存储）与借用版算子共用这一处。
pub(crate) fn assemble<B, R, K>(dims: R, steps: R, base: usize, buf: B) -> Tensor<B, R, K> {
    Tensor {
        dims,
        steps,
        base,
        buf,
        marker: PhantomData,
    }
}
