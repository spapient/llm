//! 参数角色闭集：一个参数槽归哪个优化器组（D11 / ADR 0037）。
//!
//! 角色由层构造器写进 [`crate::ContractSpec`]；权重衰减豁免与可训练判定都从角色派生，优化器按
//! 角色取组，不按字段名或层名前缀猜。闭集之外的角色在编译期不存在——新增一类参数必须先扩这个
//! 闭集，而不是在优化器侧再写一份名字表。

/// 参数角色：优化器分组的唯一事实源。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ParamRole {
    /// 普通可训练权重（投影矩阵、稠密权重）：参与权重衰减。
    #[default]
    Trainable,
    /// 偏置向量：可训练，豁免权重衰减。
    Bias,
    /// 归一化系数（RMSNorm γ）：可训练，豁免权重衰减。
    Norm,
    /// 词表嵌入表：可训练，豁免权重衰减。
    Embedding,
}

impl ParamRole {
    /// 该角色是否进入可训练参数表：四个闭集角色都是参数槽。
    pub const fn trainable(self) -> bool {
        true
    }

    /// 是否豁免权重衰减：偏置、归一化系数与嵌入表不衰减。
    pub const fn wd_exempt(self) -> bool {
        matches!(self, Self::Bias | Self::Norm | Self::Embedding)
    }

    /// 角色短名：诊断、快照与报告里的稳定标签，与变体一一对应。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trainable => "trainable",
            Self::Bias => "bias",
            Self::Norm => "norm",
            Self::Embedding => "embedding",
        }
    }
}

#[cfg(test)]
#[path = "../tests/param_role_tests.rs"]
mod tests;
