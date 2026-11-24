//! 可写访问与自持张量：可写视图通道、按形状分配清零、就地窗口写。
//!
//! 可写边界只出现在自持存储与借来的可写视图（[`ViewMut`](super::ViewMut)）上；借来的只读存储
//! 在这里调不到写方法。窗口写与 [`slice`](super::Tensor::slice) 共用同一份 `layout::window`。

use diagnostic::Result;

use crate::Tensor;
use crate::element::Element;
use crate::layout;
use crate::product;
use crate::rank::Rank;
use crate::shape::Slice;
use crate::store::{Owned, StoreMut};

use super::{ViewMut, assemble};

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: StoreMut<K>,
{
    /// 可写元素：自持张量与借来的可写视图（[`ViewMut`]）共用这一处访问器。
    pub fn as_slice_mut(&mut self) -> &mut [K] {
        self.buf.slice_mut()
    }

    /// 借出可写视图（自持存储或 `&mut [K]` 借入）；元数据照抄，不复制元素。
    pub fn as_view_mut(&mut self) -> ViewMut<'_, K> {
        assemble(
            self.dims.as_slice().to_vec(),
            self.steps.as_slice().to_vec(),
            self.base,
            self.buf.slice_mut(),
        )
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
