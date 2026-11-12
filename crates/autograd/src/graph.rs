//! 图的静态部分：[`Requirement`]（谁在图上）、[`Var`]（变量的身份）、[`Node`]（一次算子调用）。
//!
//! 三者都是**数据**：不含数值、不含回调、不含设备知识。`Tape` 用稠密编号索引它们，
//! `Schedule` 只读它们，`Account` 只数它们的入边。

use std::sync::Arc;

/// 一个变量对反向传播的要求。
///
/// `None` 与另外两种是**图内/图外**的分界：不进图的变量既不产生反向步骤，也不接收梯度，
/// 因此它的"梯度为 0"不是算出来的结果，而是根本没有这个量。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Requirement {
    /// 不参与反向：常量、输入 token、detach 之后的输出。
    #[default]
    None,
    /// 只为反传存在：它的梯度要传给输入，但它本身不是梯度终点。
    Backward,
    /// 梯度终点：反向结束后调用方要取走它的梯度（参数、显式 retain 的变量）。
    Grad,
}

impl Requirement {
    /// 是否在图上（参与反向）。
    pub fn requires(self) -> bool {
        !self.is_none()
    }

    /// 是否完全在反向之外。
    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    /// 是否是梯度终点。
    pub fn is_endpoint(self) -> bool {
        matches!(self, Self::Grad)
    }

    /// 合并两个需求：都不需要梯度才是 `None`，否则输出只能是中间节点。
    ///
    /// 用"输出"而不是"两边的较大者"：两个梯度终点的和仍然不是梯度终点——终点由调用方
    /// 显式声明（`record_with(.., Requirement::Grad)`），不从输入继承。
    pub fn merged(self, other: Self) -> Self {
        match self.is_none() && other.is_none() {
            true => Self::None,
            false => Self::Backward,
        }
    }

    /// 诊断里的短名，与变体一一对应。
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Backward => "Backward",
            Self::Grad => "Grad",
        }
    }
}

/// 一个可微变量（只存编号，数值在运行时缓冲里）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    id: u32,
}

impl Var {
    /// 编号。编号按创建顺序单调递增，因此"编号大"就是"更晚创建"。
    pub fn id(&self) -> u32 {
        self.id
    }

    /// 由编号还原；只有本 crate 的记账需要，外部拿到 [`Var`] 的唯一途径是 [`crate::Tape`]。
    pub(crate) fn at(id: u32) -> Self {
        Self { id }
    }
}

/// 反向图的一个节点：一次算子调用产生的变量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    output: Var,
    op: Arc<str>,
    inputs: Arc<[Var]>,
    depth: u32,
    requirement: Requirement,
}

impl Node {
    /// 由 [`crate::Tape`] 在记录时构造；`inputs` 与 `op` 都用 `Arc`，
    /// 于是克隆一个节点只加两次引用计数，`Schedule` 不必复制字符串与边表。
    pub(crate) fn new(
        output: Var,
        op: Arc<str>,
        inputs: Arc<[Var]>,
        depth: u32,
        requirement: Requirement,
    ) -> Self {
        Self {
            output,
            op,
            inputs,
            depth,
            requirement,
        }
    }

    /// 产出的变量。
    pub fn output(&self) -> Var {
        self.output
    }

    /// 算子名（与 `op::Op::name` 一致，便于把反向公式与声明对上）。
    pub fn op(&self) -> &str {
        &self.op
    }

    /// 输入变量。
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }

    /// 深度：输入的最大深度加一，叶子为 0。反向顺序按它降序排。
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// 产出变量的需求。
    pub fn requirement(&self) -> Requirement {
        self.requirement
    }
}

#[cfg(test)]
mod tests {
    use super::Requirement::{Backward, Grad, None};

    #[test]
    fn only_two_none_inputs_merge_to_none() {
        let cases = [
            (None, None, None),
            (None, Backward, Backward),
            (None, Grad, Backward),
            (Backward, None, Backward),
            (Backward, Backward, Backward),
            (Backward, Grad, Backward),
            (Grad, None, Backward),
            (Grad, Backward, Backward),
            (Grad, Grad, Backward),
        ];
        for (left, right, expected) in cases {
            assert_eq!(left.merged(right), expected, "{left:?} 与 {right:?}");
        }
    }

    #[test]
    fn merging_is_commutative_and_idempotent() {
        for left in [None, Backward, Grad] {
            for right in [None, Backward, Grad] {
                assert_eq!(left.merged(right), right.merged(left));
            }
            assert_eq!(
                left.merged(left),
                if left.is_none() { None } else { Backward }
            );
        }
    }

    #[test]
    fn the_predicates_partition_the_three_variants() {
        assert!(None.is_none() && !None.requires() && !None.is_endpoint());
        assert!(!Backward.is_none() && Backward.requires() && !Backward.is_endpoint());
        assert!(!Grad.is_none() && Grad.requires() && Grad.is_endpoint());
        assert_eq!(
            [None.name(), Backward.name(), Grad.name()],
            ["None", "Backward", "Grad"]
        );
    }
}
