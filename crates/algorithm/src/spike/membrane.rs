use super::*;

/// 一层的状态：每通道一条 [`Lif`]、一段复用的整数累加缓冲，以及本步发放前的膜电位。
///
/// 缓冲在状态里而不是每步新分配：`drive` 的稀疏性体现在「少做乘加」，不该被一次 `vec!` 抹平。
/// `pre` 是反传的唯一入口：发放一提交，膜电位就被改写成 `reset`，那一刻的值只能靠前向顺手记下。
#[derive(Debug, Clone, PartialEq)]
pub struct Membrane {
    pub(super) outputs: usize,
    pub(super) lifs: Vec<Lif<f32>>,
    pub(super) acc: Vec<i32>,
    pub(super) pre: Vec<f32>,
}

impl Membrane {
    /// 输出通道数（构造它的层的宽度）。
    #[must_use]
    pub const fn outputs(&self) -> usize {
        self.outputs
    }

    /// 各通道当前膜电位（观测用；发放后等于 `reset`）。
    #[must_use]
    pub fn potential(&self) -> Vec<f32> {
        self.lifs.iter().map(Lif::mem).collect()
    }

    /// 各通道**本步发放前**的膜电位（反传判窗用；一次前向都还没走过时全零）。
    #[must_use]
    pub fn pre(&self) -> &[f32] {
        &self.pre
    }
}
