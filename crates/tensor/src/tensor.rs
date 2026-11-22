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
//! 且**与存储形态无关**——借来的视图与自持的张量一视同仁。除 `flip` 外一个元素都不搬：
//!
//! | 算子 | 秩 | 语义 |
//! |---|---|---|
//! | [`broadcast`](Self::broadcast) | 不变 | 目标秩不小于源秩，逐轴相等或源轴长为 1 |
//! | [`narrow`](Self::narrow) / [`slice`](Self::slice) | 不变 | 沿一轴 / 多轴取段，段可带步长（降采样视图） |
//! | [`chunks`](Self::chunks) / [`split`](Self::split) | 不变 | 沿一轴等分 / 按段长切分，各段借用同一存储 |
//! | [`select`](Self::select) | -1 | 固定某轴下标，该轴消失 |
//! | [`unbind`](Self::unbind) | -1 | 沿一轴逐个固定，一次借出 N 个视图（`select` 的批量版） |
//! | [`squeeze`](Self::squeeze) / [`unsqueeze`](Self::unsqueeze) | ∓1 | 去掉长为 1 的轴 / 插入长为 1 的轴 |
//! | [`diagonal`](Self::diagonal) | -1 | 两轴折成一轴（长度取小、步长相加），对角即视图 |
//! | [`flatten`](Self::flatten) | 变短 | 把行主序紧凑的一段轴压成一个 |
//! | [`transpose`](Self::transpose) / [`permute`](Self::permute) / [`movedim`](Self::movedim) | 不变 | 换轴序 / 搬一轴（其余轴次序不动） |
//! | [`slice_fill`](Self::slice_fill) / [`slice_assign`](Self::slice_assign) | 不变 | 就地写同一个窗口（仅自持存储） |
//! | [`flip`](Self::flip) | 不变 | 沿指定轴反向读出——步长无符号，只有它要搬数据 |
//!
//! 改秩的六个算子要指定目标秩见证（`t.select::<[usize; 1]>(0, 3)`），因为稳定 Rust 算不出
//! `D - 1`；Burn 在这里要求 `.squeeze::<2>()`，同一手法，只是那边只能是定长数组，这边
//! `Vec<usize>` 也可以。不改秩的算子（`narrow` / `slice` / `broadcast`）没有这个摩擦。
//!
//! 偏移、可达性、连续性与取行只有 [`crate::layout`] 一份实现——两种秩、两种存储共用它。

use core::marker::PhantomData;
use diagnostic::Result;

use crate::invalid;

use crate::element::Element;
use crate::layout;
use crate::rank::Rank;
use crate::shape::{Slice, element_bytes};
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
    #[doc(alias = "swap_dims")]
    pub fn transpose(self, a: usize, b: usize) -> Option<Self> {
        self.reaxis(|rank| rank.swapped(a, b))
    }

    /// 把一轴搬到另一位置：其余轴的相对次序保持不动（`transpose` 只交换两轴，做不到这一点）。
    ///
    /// 与 [`transpose`](Self::transpose) / [`permute`](Self::permute) 同一笔账：只重排轴长与步长，
    /// 零拷贝、与存储形态无关。`from == to` 是恒等；越界返回 `None`。
    #[doc(alias = "moveaxis")]
    pub fn movedim(self, from: usize, to: usize) -> Option<Self> {
        let rank = self.rank();
        if from >= rank || to >= rank {
            return None;
        }
        let mut order: Vec<usize> = (0..rank).filter(|&axis| axis != from).collect();
        order.insert(to, from);
        self.permute(order)
    }

    /// 按 `order` 重排各轴：置换是零拷贝的元数据重排。
    pub fn permute(self, order: impl AsRef<[usize]>) -> Option<Self> {
        let order = order.as_ref();
        if !is_permutation(order, self.rank()) {
            return None;
        }
        self.reaxis(|rank| rank.permuted(order))
    }

    /// 把同一套轴重排同时施加到轴长与步长：两处分头写一遍就是漂移源。
    fn reaxis(self, permute: impl Fn(&R) -> Option<R>) -> Option<Self> {
        Some(assemble(
            permute(&self.dims)?,
            permute(&self.steps)?,
            self.base,
            self.buf,
        ))
    }

    /// 广播到更大的形状：目标秩不小于源秩，逐轴相等或源轴长为 1，否则 `None`。
    ///
    /// 被扩展的轴步长为 0——同一个元素被多个逻辑位置读到，零拷贝。步长 0 是转置与置换表达不出来的
    /// 布局，布局代数因此严格宽于"只有重排"。
    #[doc(alias = "expand")]
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
        Some(self)
    }

    /// 多轴同时取段：`ranges[i]` 作用于第 `i` 轴，没给出的尾部轴原样保留。
    ///
    /// 一段可以是连续区间（`0..4`），也可以是 [`Slice::stepped`] 这种跳着读的段——步长只
    /// 乘进该轴元数据，视图照样零拷贝。窗口在各处只算一份：`layout::window` 一次给出全部轴，
    /// 读用 [`slice`](Self::slice)，就地写用 [`slice_fill`](Self::slice_fill) /
    /// [`slice_assign`](Self::slice_assign)，三者的窗口语义不可能分叉。
    pub fn slice<S: Into<Slice> + Clone>(self, ranges: impl AsRef<[S]>) -> Option<Self> {
        let out = self.window(ranges.as_ref())?;
        self.respine(out)
    }

    /// `ranges` 圈出的那一窗的布局：逐轴换成 [`Slice`] 落在栈上算好带走，于是随后可以
    /// `buf.slice_mut()` 而不与 `self` 冲突。轴数超出栈上容量返回 `None`。
    fn window<S: Into<Slice> + Clone>(&self, ranges: &[S]) -> Option<layout::Layout> {
        let mut stack = [Slice::new(0, 0); layout::MAX_RANK];
        let slots = stack.get_mut(..ranges.len())?;
        for (slot, range) in slots.iter_mut().zip(ranges) {
            *slot = range.clone().into();
        }
        layout::window(
            self.dims.as_slice(),
            self.steps.as_slice(),
            self.base,
            slots,
        )
    }

    /// 沿 `axis` 等分成至多 `pieces` 段，每段都借用同一存储——不搬数据、不复制。
    ///
    /// 段长 `ceil(轴长 / pieces)`，末段可能更短；这就是 [`split`](Self::split) 的等长特例，
    /// 末段被夹到剩余长度这件事只有 `split` 一处实现。
    pub fn chunks(&self, axis: usize, pieces: usize) -> impl Iterator<Item = Tensor<&[K], R, K>> {
        let dim = self.dims.as_slice().get(axis).copied().unwrap_or(0);
        let size = dim.div_ceil(pieces.max(1));
        self.split(axis, (0..pieces).map(move |_| size)).into_iter()
    }

    /// 按给定段长沿 `axis` 依次切分，各段借用同一存储（Burn 的 `split`）。
    ///
    /// 段长是**请求**：越过轴长的那一段被夹到剩余长度，总段长恰好覆盖整轴——任何正段长序列都
    /// 给出一次无重叠、无缝隙的划分，且不 panic。Burn 在这里要求段长之和等于轴长，否则 panic；
    /// 夹紧是同一语义的全函数版本。切段与 [`narrow`](Self::narrow) 共用 `layout::narrow`。
    pub fn split(
        &self,
        axis: usize,
        sizes: impl IntoIterator<Item = usize>,
    ) -> Vec<Tensor<&[K], R, K>> {
        let dim = self.dims.as_slice().get(axis).copied().unwrap_or(0);
        let mut start = 0usize;
        let mut parts = Vec::new();
        for len in sizes {
            let len = len.min(dim.saturating_sub(start));
            if len == 0 {
                break;
            }
            let Some(piece) = self.segment(axis, start, len) else {
                break;
            };
            parts.push(piece);
            start += len;
        }
        parts
    }

    /// `[start, start + len)` 的借用视图：造一份元数据再走 [`narrow`](Self::narrow)。
    fn segment(&self, axis: usize, start: usize, len: usize) -> Option<Tensor<&[K], R, K>> {
        let view = assemble(
            self.dims.clone(),
            self.steps.clone(),
            self.base,
            self.buf.slice(),
        );
        view.narrow(axis, start, len)
    }

    /// 沿 `axis` 逐个位置固定，给出 `dims[axis]` 个秩减一的借用视图（Burn 的 `unbind`）。
    ///
    /// 与 [`select`](Self::select) 共用 `layout::select`，差别只在 `select` 吃掉 `self` 交出一份，
    /// 这里借出全部 `N` 份、原张量不动——于是可以在循环里同时向源张量写入。
    pub fn unbind<R2: Rank>(&self, axis: usize) -> Vec<Tensor<&[K], R2, K>> {
        let dim = self.dims.as_slice().get(axis).copied().unwrap_or(0);
        (0..dim)
            .filter_map(|idx| {
                let (dims, steps) = (self.dims.as_slice(), self.steps.as_slice());
                seat(
                    layout::select(dims, steps, self.base, axis, idx)?,
                    self.buf.slice(),
                )
            })
            .collect()
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

    /// 轴 `a` 与轴 `b` 的对角：两轴折成一轴（长度取较小者、步长相加），秩减一，零拷贝。
    ///
    /// 对角元素全落在 `base + i·(step_a + step_b)` 上，所以这只是把两轴步长相加——`diag` 与
    /// `trace` 因此不必搬数据。`a == b` 或轴越界返回 `None`（同一轴"对自己取对角"没有几何含义）。
    #[doc(alias = "diag")]
    pub fn diagonal<R2: Rank>(self, a: usize, b: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::diagonal(self.dims.as_slice(), self.steps.as_slice(), self.base, a, b)?;
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

    /// 沿 `axes` 各轴反向读出：元素顺序倒过来，缓冲重排（`O(numel)`），返回自持张量。
    ///
    /// 这不是布局能表达的动作——步长是无符号量，反向读要么负步长、要么搬数据，本仓库取后者，
    /// 于是借来的视图也能翻转（先摊成连续缓冲）。轴号超出秩的忽略，重复给出的轴只翻一次。
    #[must_use]
    pub fn flip(&self, axes: impl AsRef<[usize]>) -> Tensor<Owned<K>, R, K> {
        let shape = self.dims.as_slice();
        let mut marked = [false; layout::MAX_RANK];
        let Some(turn) = marked.get_mut(..shape.len()) else {
            return self.to_contiguous();
        };
        for &axis in axes.as_ref() {
            if let Some(slot) = turn.get_mut(axis) {
                *slot = true;
            }
        }
        let source = self.buf.slice();
        let mut buf = Vec::with_capacity(product(shape));
        layout::walk(shape, self.steps.as_slice(), self.base, turn, &mut |at| {
            buf.push(source[at]);
        });
        self.respell(buf)
    }

    /// 换成本张量的语序：把 [`layout`] 算好的布局装进新的秩见证，存储原样带走。
    fn respine<R2: Rank>(self, out: layout::Layout) -> Option<Tensor<B, R2, K>> {
        seat(out, self.buf)
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
        self.respell(self.to_vec())
    }

    /// 形状照抄、步长重算、起点归零，配上一块新的连续缓冲：摊平与翻转共用这一处收尾。
    fn respell(&self, buf: Vec<K>) -> Tensor<Owned<K>, R, K> {
        let steps = R::contiguous(self.dims.as_slice()).unwrap_or_else(|| self.dims.clone());
        assemble(self.dims.clone(), steps, 0, Owned::from(buf))
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
fn gather_strided<T: Copy>(source: &[T], shape: &[usize], steps: &[usize], base: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(product(shape));
    layout::walk(shape, steps, base, &[], &mut |at| buf.push(source[at]));
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

    /// 就地写 `ranges` 那一窗：窗口内每个位置都写成 `value`，窗口外原样。
    ///
    /// 窗口与 [`slice`](Self::slice) 完全一致（同一个 `layout::window`），只是方向反过来。这类
    /// 原地更新只出现在自持张量上——借来的存储只读。窗口越界返回 `None`。
    pub fn slice_fill<S: Into<Slice> + Clone>(
        &mut self,
        ranges: impl AsRef<[S]>,
        value: K,
    ) -> Option<()> {
        let out = self.window(ranges.as_ref())?;
        self.paint(out, || Some(value))
    }

    /// 就地写 `ranges` 那一窗：`values` 按行主序逻辑顺序逐位对应窗口，长度必须正好等于窗口元素数。
    ///
    /// 源取 `&[K]` 而不是张量，于是任何形态的源都能进来：[`to_vec`](Self::to_vec) 给的正是逻辑
    /// 顺序，非连续张量直接可用。长度不符返回 `None`——不静默截断，也不补零。
    pub fn slice_assign<S: Into<Slice> + Clone>(
        &mut self,
        ranges: impl AsRef<[S]>,
        values: impl AsRef<[K]>,
    ) -> Option<()> {
        let out = self.window(ranges.as_ref())?;
        let values = values.as_ref();
        if values.len() != product(out.dims()) {
            return None;
        }
        let mut source = values.iter();
        self.paint(out, || source.next().copied())
    }

    /// 往算好的窗口里逐位写：`next` 给出下一个值；取不到就停手并让整次写失败（长度已在调用处校验，
    /// 这条兜底只为「绝不留半写状态」）。
    fn paint(&mut self, out: layout::Layout, mut next: impl FnMut() -> Option<K>) -> Option<()> {
        let buf = self.buf.slice_mut();
        let mut complete = true;
        layout::walk(
            out.dims(),
            out.steps(),
            out.base(),
            &[],
            &mut |at| match next() {
                Some(value) => buf[at] = value,
                None => complete = false,
            },
        );
        complete.then_some(())
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
/// 把布局装进秩见证，存储原样带上：`respine`（吃掉存储）与借用版算子共用这一处。
fn seat<B, R2: Rank, K>(out: layout::Layout, buf: B) -> Option<Tensor<B, R2, K>> {
    Some(assemble(
        R2::from_slice(out.dims())?,
        R2::from_slice(out.steps())?,
        out.base(),
        buf,
    ))
}

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
