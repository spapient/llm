//! 字节表：[`Footprint`] 把变量编号映射成字节数，[`Sizes`] 是现成的形状表实现。

use tensor::{DType, Shape};

use crate::Var;

/// 一个变量的值占多少字节。
///
/// 形状不在反向图里（[`crate::Var`] 只是编号），所以字节数由调用方给：`nn` 用层的形状表，
/// `train` 用真实缓冲的容量。泛型而不是 `dyn`——`Plan::of` 每个变量只查一次，没有动态分派。
pub trait Footprint {
    /// 字节数；`0` 表示不需要为它留显存（标量、或者调用方还没登记形状）。
    fn bytes(&self, var: Var) -> u64;
}

impl Footprint for [u64] {
    fn bytes(&self, var: Var) -> u64 {
        self.get(var.id() as usize).copied().unwrap_or(0)
    }
}

impl Footprint for Vec<u64> {
    fn bytes(&self, var: Var) -> u64 {
        self.as_slice().bytes(var)
    }
}

impl<F: Footprint + ?Sized> Footprint for &F {
    fn bytes(&self, var: Var) -> u64 {
        (**self).bytes(var)
    }
}

/// 变量形状表：记录前向时顺手登记，`Plan` 直接换算成字节；只存形状与元素类型，不复制张量。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sizes {
    entries: Vec<Option<(Shape, DType)>>,
}

impl Sizes {
    /// 空表，容量按变量数预分配。
    pub fn new(vars: u32) -> Self {
        Self {
            entries: vec![None; vars as usize],
        }
    }

    /// 登记一个变量的形状；编号可以跳跃（后创建的变量先登记也不会错位）。
    pub fn set(&mut self, var: Var, shape: Shape, dtype: DType) {
        let index = var.id() as usize;
        if index >= self.entries.len() {
            self.entries.resize(index + 1, None);
        }
        self.entries[index] = Some((shape, dtype));
    }

    /// 一个变量登记的形状与元素类型；没登记时为 `None`。
    pub fn get(&self, var: Var) -> Option<(&Shape, DType)> {
        self.entries
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map(|(shape, dtype)| (shape, *dtype))
    }

    /// 表的长度：覆盖到的变量数（含还没登记的槽位），与 `Tape::vars()` 同口径。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否还没有任何变量被登记过（表的长度为 0）。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Footprint for Sizes {
    fn bytes(&self, var: Var) -> u64 {
        self.entries
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map_or(0, |(shape, dtype)| shape.element_bytes(*dtype) as u64)
    }
}
