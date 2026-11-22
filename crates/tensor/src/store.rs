//! 元素存储：`Tensor` 的第一个泛型位。
//!
//! `B` 只管"元素存在哪"，不管怎么算。Burn 把第一个泛型位给了后端算子集（`B: Backend`，
//! `FloatTensorOps` 一条就 130 个方法）；本仓库的算子来自数据驱动注册表（ADR 0053），不需要
//! 泛型派发，于是这个位置让给存储——同一个 [`Tensor`](crate::Tensor) 既能借用调用方的 `&[K]`，
//! 也能持有自己的 `Vec<K>`，全套方法两处通用，不再有"字节视图 / 元素视图 / 拥有视图"三份实现。

use diagnostic::Result;

/// 只读元素存储。
pub trait Store<E> {
    /// 全部元素。
    fn slice(&self) -> &[E];
}

/// 可写元素存储。
pub trait StoreMut<E>: Store<E> {
    /// 全部元素（可写）。
    fn slice_mut(&mut self) -> &mut [E];
}

/// 借来的存储就是切片本身：`B = &'a [K]`，构造时不必再包一层。
impl<E> Store<E> for &[E] {
    fn slice(&self) -> &[E] {
        self
    }
}

/// 自己持有的存储。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owned<E>(Vec<E>);

impl<E> Owned<E> {
    /// 接管一个已有向量。
    pub fn new(elems: Vec<E>) -> Self {
        Self(elems)
    }

    /// 交出内部向量。
    pub fn into_vec(self) -> Vec<E> {
        self.0
    }

    /// 按元素个数分配并清零。
    ///
    /// 地址空间装不下时报 `Invalid` 而不是 abort：形状来自文件头，恶意或损坏的头不该成为
    /// 进程终止原语。
    pub fn zeros(len: usize) -> Result<Self>
    where
        E: Copy + Default,
    {
        let mut elems = Vec::new();
        elems.try_reserve_exact(len).map_err(|_| {
            crate::invalid(
                format!("{len} 个元素超出可用内存或地址空间"),
                "tensor::Owned::zeros",
            )
            .with("tensor::Owned::zeros")
        })?;
        elems.resize(len, E::default());
        Ok(Self(elems))
    }
}

impl<E> Store<E> for Owned<E> {
    fn slice(&self) -> &[E] {
        &self.0
    }
}

impl<E> StoreMut<E> for Owned<E> {
    fn slice_mut(&mut self) -> &mut [E] {
        &mut self.0
    }
}
