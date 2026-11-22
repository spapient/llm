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
//! # 形状算子面
//!
//! 元数据是 `(起点偏移, 各轴长度, 各轴步长)` 三元组，所以整套形状算子都是 O(轴数) 的整数运算，
//! 一个元素都不搬，且**与存储形态无关**——借来的视图与自持的张量一视同仁：
//!
//! | 算子 | 秩 | 语义 |
//! |---|---|---|
//! | [`broadcast`](Self::broadcast) | 不变 | 目标秩不小于源秩，逐轴相等或源轴长为 1 |
//! | [`narrow`](Self::narrow) / [`slice`](Self::slice) | 不变 | 沿一轴 / 多轴取连续段 |
//! | [`select`](Self::select) | -1 | 固定某轴下标，该轴消失 |
//! | [`squeeze`](Self::squeeze) / [`unsqueeze`](Self::unsqueeze) | ∓1 | 去掉长为 1 的轴 / 插入长为 1 的轴 |
//! | [`flatten`](Self::flatten) | 变短 | 把行主序紧凑的一段轴压成一个 |
//! | [`transpose`](Self::transpose) / [`permute`](Self::permute) | 不变 | 换轴序 |
//!
//! 改秩的四个算子要指定目标秩见证（`t.select::<[usize; 1]>(0, 3)`），因为稳定 Rust 算不出
//! `D - 1`；Burn 在这里要求 `.squeeze::<2>()`，同一手法，只是那边只能是定长数组，这边
//! `Vec<usize>` 也可以。不改秩的算子（`narrow` / `slice` / `broadcast`）没有这个摩擦。
//!
//! 偏移、可达性、连续性与取行只有 [`crate::layout`] 一份实现——两种秩、两种存储共用它。

use core::marker::PhantomData;
use core::ops::Range;

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
    /// 起点偏移：下标 0 对应的缓冲位置。子视图不改缓冲、只把它前移。
    base: usize,
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

    /// 交换两轴：零拷贝的元数据重排，与存储形态无关。
    pub fn transpose(self, a: usize, b: usize) -> Option<Self> {
        Some(assemble(
            self.dims.swapped(a, b)?,
            self.steps.swapped(a, b)?,
            self.base,
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
            self.base,
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
            .map(|(axis, &want)| {
                broadcast_step(
                    (lead, self.dims.as_slice(), self.steps.as_slice()),
                    (axis, want),
                )
            })
            .collect::<Option<Vec<usize>>>()?;
        let steps = R::from_slice(&steps)?;
        Some(assemble(dims, steps, self.base, self.buf))
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
        Some(assemble(dims, steps, self.base, self.buf))
    }

    /// 沿 `axis` 取 `[start, start + len)` 一段：秩不变，只把起点前移、该轴收窄。
    ///
    /// 越界返回 `None`。缓冲不动，所以自持张量也能切（不需要复制，也不必是借来的存储）。
    pub fn narrow(mut self, axis: usize, start: usize, len: usize) -> Option<Self> {
        self.shift(axis, start, len)?;
        Some(self)
    }

    /// 把一轴的 `[start, start + len)` 落到元数据上：起点前移、该轴收窄。
    fn shift(&mut self, axis: usize, start: usize, len: usize) -> Option<()> {
        let (base, dim) = layout::narrow(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.base,
            axis,
            start,
            len,
        )?;
        self.base = base;
        self.dims.as_slice_mut()[axis] = dim;
        Some(())
    }

    /// 多轴同时取段：`ranges[i]` 作用于第 `i` 轴，没给出的尾部轴原样保留。
    ///
    /// 比逐轴 [`narrow`](Self::narrow) 少一次重新校验和几趟搬动：一趟扫完所有轴。
    pub fn slice(mut self, ranges: impl AsRef<[Range<usize>]>) -> Option<Self> {
        let ranges = ranges.as_ref();
        if ranges.len() > self.rank() {
            return None;
        }
        for (axis, range) in ranges.iter().enumerate() {
            self.shift(axis, range.start, range.end.checked_sub(range.start)?)?;
        }
        Some(self)
    }

    /// 固定第 `axis` 轴为第 `idx` 个：起点前移、该轴消失，所以秩减一。
    pub fn select<R2: Rank>(self, axis: usize, idx: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::select(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.base,
            axis,
            idx,
        )?;
        self.respine(out)
    }

    /// 去掉长为 1 的第 `axis` 轴；该轴不是单例返回 `None`（不做静默通过）。
    pub fn squeeze<R2: Rank>(self, axis: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::squeeze(self.dims.as_slice(), self.steps.as_slice(), self.base, axis)?;
        self.respine(out)
    }

    /// 在第 `axis` 位插入一个长为 1 的轴：纯元数据，零拷贝。
    pub fn unsqueeze<R2: Rank>(self, axis: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::unsqueeze(self.dims.as_slice(), self.steps.as_slice(), self.base, axis)?;
        self.respine(out)
    }

    /// 把 `[start, end)` 各轴压成一个轴：仅当这段在存储里就是行主序紧凑时成立，否则 `None`。
    ///
    /// 比 [`reshape`](Self::reshape) 严格地窄一格也严格地安全一格：`reshape` 要求**整个**张量连续，
    /// 这里只要求这一段连续，于是转置过的张量也能先压平能压的那几轴。
    pub fn flatten<R2: Rank>(self, start: usize, end: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::flatten(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.base,
            start,
            end,
        )?;
        self.respine(out)
    }

    /// 换成本张量的语序：把 [`layout`] 算好的布局装进新的秩见证，存储原样带走。
    fn respine<R2: Rank>(self, out: layout::Layout) -> Option<Tensor<B, R2, K>> {
        Some(assemble(
            R2::from_slice(out.dims())?,
            R2::from_slice(out.steps())?,
            out.base(),
            self.buf,
        ))
    }

    /// 按逻辑下标顺序摊平：转置、广播、窄视图都给出 `numel` 个元素的新缓冲。
    ///
    /// 连续布局走一次整段拷贝（起点可能不是 0，仍是一段），其余布局按步长走一遍；`O(numel)`。
    pub fn to_vec(&self) -> Vec<K> {
        let shape = self.dims.as_slice();
        let source = self.buf.slice();
        if self.is_contiguous() {
            return gather(source, self.base, product(shape));
        }
        gather_strided(source, shape, self.steps.as_slice(), self.base)
    }

    /// 物化成连续缓冲：步长 0 的广播轴、非满长的窄视图都摊进一块紧凑内存，于是可以接着
    /// [`reshape`](Self::reshape)。这是本模块唯一真正搬数据的地方。
    #[must_use]
    pub fn to_contiguous(&self) -> Tensor<Owned<K>, R, K> {
        let steps = R::contiguous(self.dims.as_slice()).unwrap_or_else(|| self.dims.clone());
        assemble(self.dims.clone(), steps, 0, Owned::from(self.to_vec()))
    }
}

/// 广播时第 `axis` 轴的步长：前导新轴为 0；同长沿用源步长；源轴长为 1 则恒读同一元素。
///
/// 秩、轴长对不上都不是广播，返回 `None`。
fn broadcast_step(
    (lead, dims, steps): (usize, &[usize], &[usize]),
    (axis, want): (usize, usize),
) -> Option<usize> {
    let Some(source) = axis.checked_sub(lead) else {
        return Some(0);
    };
    let have = *dims.get(source)?;
    if have == 1 {
        return Some(0);
    }
    if have != want {
        return None;
    }
    steps.get(source).copied()
}

/// 按步长走一遍全部逻辑位置，把读到的元素压成连续缓冲：O(numel)。
///
/// 构造时的可达性校验已证明任一合法下标都落在缓冲内，故不走 `Option`。
fn gather_strided<T: Copy>(source: &[T], shape: &[usize], steps: &[usize], base: usize) -> Vec<T> {
    let numel = product(shape);
    let mut buf = Vec::with_capacity(numel);
    let mut idx = vec![0usize; shape.len()];
    let mut at = base;
    for _ in 0..numel {
        buf.push(source[at]);
        // 行主序进位：偏移跟着增减，省掉每次重算点积。
        for axis in (0..idx.len()).rev() {
            idx[axis] += 1;
            at += steps[axis];
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
            at -= steps[axis] * shape[axis];
        }
    }
    buf
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

/// 从 `start` 起拷 `len` 个元素；越界时按实际可得的截断（构造期已保证不会发生）。
fn gather<K: Copy>(source: &[K], start: usize, len: usize) -> Vec<K> {
    source
        .get(start..start.saturating_add(len))
        .unwrap_or_default()
        .to_vec()
}

/// 六处构造与重排共用同一个组装动作：元数据 + 起点 + 存储 + 零尺寸类型标记。
fn assemble<B, R, K>(dims: R, steps: R, base: usize, buf: B) -> Tensor<B, R, K> {
    Tensor {
        dims,
        steps,
        base,
        buf,
        marker: PhantomData,
    }
}

/// `order` 是否是 `0..rank` 的一个排列（每个值恰好出现一次）。
fn is_permutation(order: &[usize], rank: usize) -> bool {
    order.len() == rank && (0..rank).all(|value| order.contains(&value))
}
