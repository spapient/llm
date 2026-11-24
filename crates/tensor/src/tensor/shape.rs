//! 张量布局与形状算子：轴重排、广播、切片、分段、改秩、物化与视图降级。
//!
//! 全部是 O(轴数) 的整数元数据运算，与存储形态无关（借来视图与自持张量一视同仁）；除
//! [`flip`](Tensor::flip) 外一个元素都不搬。构造、只读访问器与组装动作在 [`super`](crate::tensor)。

use crate::Tensor;
use crate::element::Element;
use crate::layout;
use crate::product;
use crate::rank::Rank;
use crate::shape::Slice;
use crate::store::{Owned, Store};

use super::{DynTensor, View, assemble};

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
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
    pub(crate) fn window<S: Into<Slice> + Clone>(&self, ranges: &[S]) -> Option<layout::Layout> {
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

    /// 借出现有存储，得到运行期秩的只读视图；元数据照抄，不复制元素。
    pub fn as_view(&self) -> View<'_, K> {
        assemble(
            self.dims.as_slice().to_vec(),
            self.steps.as_slice().to_vec(),
            self.base,
            self.buf.slice(),
        )
    }

    /// 物化成自持连续张量；已是连续布局时保留形状，非连续先摊平到紧凑内存。
    #[must_use]
    pub fn into_owned(self) -> Tensor<Owned<K>, R, K> {
        self.to_contiguous()
    }

    /// 擦除秩与存储：降级为运行期秩的自持张量，作为后端进出的统一形态。
    pub fn into_dyn(self) -> DynTensor<K> {
        let owned = self.to_contiguous();
        assemble(
            owned.dims.as_slice().to_vec(),
            owned.steps.as_slice().to_vec(),
            owned.base,
            owned.buf,
        )
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

/// `order` 是否是 `0..rank` 的一个排列（每个值恰好出现一次）。
fn is_permutation(order: &[usize], rank: usize) -> bool {
    order.len() == rank && (0..rank).all(|value| order.contains(&value))
}
