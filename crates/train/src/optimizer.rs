//! 优化器：对象安全 trait、矩状态按参数编号自持、按 [`ParamRole`](llm_nn::ParamRole) 取组。
//!
//! **拥有**：[`Optimizer`] trait、[`Group`] 分组闭集、[`Slot`] 槽位视图与 [`Sgd`] / [`Adam`] /
//! [`Muon`] 三种实现。**不拥有**：学习率调度（[`crate::Schedule`]）、梯度缓冲
//! （[`crate::Buffers`] 扩展）、快照格式（[`snapshot`]）。**依赖**：`nn`（角色闭集）、
//! `diagnostic`。
//!
//! 三条收口口径：
//!
//! - 组归属只从角色派生：[`Group::from_role`] 是角色 → 组的唯一映射，优化器不按字段名、层名
//!   前缀猜组；权重衰减豁免同样从 [`ParamRole::wd_exempt`] 派生。
//! - 矩状态由优化器自己按槽位编号自持：autograd 只出 tape / plan / Replay，不持梯度存储；
//!   优化器实例就是矩状态的唯一存放处，快照在 T1-4 从这里导出。
//! - 矩阵形状随槽一起走（[`Slot::shape`]）：`Sgd` / `Adam` 逐元素更新、用不上形状，但 [`Muon`]
//!   要在矩阵上做正交化——形状不在槽里，就只能靠调用方和优化器对暗号，而暗号对错了没人报错。
//!   非矩阵槽的 `shape` 是 `None`，矩阵槽的 `rows × cols` 必须等于参数长度。

use diagnostic::{Code, Error, Result};
use nn::ParamRole;

/// 本模块诊断站点：出错时标明"是优化器拒绝了这次步进"。
pub(crate) const SITE: &str = "train::Optimizer";

/// 优化器参数组：角色派生的闭集，与 [`ParamRole`] 一一对应，不引入第二份名字表。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// 普通投影权重：参与权重衰减。
    Trainable,
    /// 偏置：豁免权重衰减。
    Bias,
    /// 归一化系数：豁免权重衰减。
    Norm,
    /// 嵌入表：豁免权重衰减。
    Embedding,
}

impl Group {
    /// 角色 → 组的唯一映射；优化器分组走这里，不猜字段名。
    pub fn from_role(role: ParamRole) -> Self {
        match role {
            ParamRole::Trainable => Self::Trainable,
            ParamRole::Bias => Self::Bias,
            ParamRole::Norm => Self::Norm,
            ParamRole::Embedding => Self::Embedding,
        }
    }

    /// 组 → 角色的回映：标签与豁免判定都从 [`ParamRole`] 取，组侧不抄第二份名字表。
    const fn role(self) -> ParamRole {
        match self {
            Self::Trainable => ParamRole::Trainable,
            Self::Bias => ParamRole::Bias,
            Self::Norm => ParamRole::Norm,
            Self::Embedding => ParamRole::Embedding,
        }
    }

    /// 组短名：诊断与快照标签，字符串事实源在 [`ParamRole::as_str`]。
    pub const fn as_str(self) -> &'static str {
        self.role().as_str()
    }
}

/// 一个待更新参数槽的视图：参数值就地更新，梯度为该槽当前已累积/裁剪口径的结果。
///
/// 生命周期借用调用方的参数与梯度缓冲；优化器不拥有参数存储，只持有自己的矩状态。
/// 梯度在分布式同步阶段需要就地改写（归约/平均），因此这里是可变借用；优化器步进仍只读。
pub struct Slot<'a> {
    /// 参数值（就地写回）。
    pub params: &'a mut [f32],
    /// 当前梯度（同步阶段可就地归约/平均，步进阶段只读）。
    pub grads: &'a mut [f32],
    /// 参数角色：分组与衰减豁免的派生依据。
    pub role: ParamRole,
    /// 矩阵形状 `(rows, cols)`：参数在图上登记为二阶张量时给出，否则 `None`。
    ///
    /// `Sgd` / `Adam` 用不上它（逐元素更新），[`Muon`] 的正交化只在矩阵上有定义——形状因此是
    /// 更新口径的一部分，而不是调用方的私事。
    pub shape: Option<(usize, usize)>,
}

impl Slot<'_> {
    /// 参数与梯度长度必须一致，且矩阵槽的行列积必须等于参数长度。
    ///
    /// 不一致是调用方接线错误（[`Code::Invalid`]）：长度错了的梯度会整体错位，而不会有任何一处报错。
    fn check_len(&self) -> Result<()> {
        if self.params.len() != self.grads.len() {
            let message = format!(
                "参数槽长度 {} 与梯度长度 {} 不一致：角色 {} 的参数表与梯度表必须逐项对齐",
                self.params.len(),
                self.grads.len(),
                self.role.as_str()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        if let Some((rows, cols)) = self.shape {
            let expected = rows.checked_mul(cols);
            if expected != Some(self.params.len()) {
                let message = format!(
                    "参数槽声明为 [{rows}, {cols}] 矩阵，但长度是 {}：行列积与元素数不符",
                    self.params.len()
                );
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
        }
        Ok(())
    }
}

/// 视图与梯度向量的配对：梯度向量在会话侧取走后随视图一起存活，步进结束整块归还缓冲。
pub struct OwnedSlot {
    /// 参数值就地写回（会话槽参数的快照副本，步进后镜像回缓冲）。
    pub params: Vec<f32>,
    /// 从缓冲取走的梯度向量，同步/步进结束归还。
    pub grads: Vec<f32>,
    /// 参数角色。
    pub role: ParamRole,
    /// 矩阵形状；非矩阵槽为 `None`（口径见 [`Slot::shape`]）。
    pub shape: Option<(usize, usize)>,
}

impl OwnedSlot {
    /// 借成优化器/同步用的可变视图。
    pub fn view(&mut self) -> Slot<'_> {
        Slot {
            params: &mut self.params,
            grads: &mut self.grads,
            role: self.role,
            shape: self.shape,
        }
    }
}

/// 对象安全优化器 trait：实现自持矩状态，按槽位编号（`slots` 下标）存取。
///
/// 方法无泛型、无 `Self` 返回，因此可装成 `&mut dyn Optimizer`；新增优化器只实现本 trait，
/// 不另起一套参数组映射。
pub trait Optimizer {
    /// 一步更新全部槽位；`lr` 是调度之后的当前有效学习率。
    ///
    /// # Errors
    ///
    /// 任一槽位参数/梯度长度不一致（[`Code::Invalid`]）时返回说明，不做半截更新。
    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()>;

    /// 从快照回载矩状态与优化器步数；无矩优化器只接受空表。
    ///
    /// # Errors
    ///
    /// 两张表槽位数或每槽长度不符（[`Code::Invalid`]），或无矩优化器收到非空矩表。
    fn restore(
        &mut self,
        momentum: Vec<Vec<f32>>,
        variance: Vec<Vec<f32>>,
        step: u64,
    ) -> Result<()>;
}

/// 全局梯度范数（L2）：跨全部槽位汇总，裁剪按这个范数统一缩比。
fn global_norm(slots: &[Slot<'_>]) -> f32 {
    let squares = slots
        .iter()
        .flat_map(|slot| slot.grads.iter())
        .map(|g| g * g)
        .sum::<f32>();
    squares.sqrt()
}

/// 该角色参与的解耦衰减系数：豁免组为 `0`，其余取配置系数；豁免判定只从 [`ParamRole`] 派生。
fn decay_rate(role: ParamRole, weight_decay: f32) -> f32 {
    if role.wd_exempt() { 0.0 } else { weight_decay }
}

/// 裁剪系数：范数未超阈值为 `1.0`；超阈值按 `max_norm / norm` 等比缩小。
fn clip_scale(norm: f32, max_norm: Option<f32>) -> f32 {
    match max_norm {
        Some(max) if max > 0.0 && norm > max => max / norm,
        _ => 1.0,
    }
}

/// 校验全部槽位长度；校验失败不进入更新路径。
fn check_slots(slots: &[Slot<'_>]) -> Result<()> {
    slots.iter().try_for_each(Slot::check_len)
}

/// 普通 SGD（可选解耦权重衰减 + 全局范数裁剪）。
///
/// Sgd 与 Adam 共用的解耦衰减 / 范数裁剪构造器：同一份口径，不各写一遍。
macro_rules! decay_clip {
    ($T:ty) => {
        impl $T {
            /// 设置解耦权重衰减系数；角色豁免组不受此值影响。
            pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
                self.weight_decay = weight_decay;
                self
            }

            /// 设置全局梯度范数上限；`None` 表示不裁剪。
            pub fn with_clip_norm(mut self, clip_norm: f32) -> Self {
                self.clip_norm = (clip_norm > 0.0).then_some(clip_norm);
                self
            }
        }
    };
}

mod adam;
mod muon;
mod sgd;

pub use adam::Adam;
pub use muon::Muon;
pub use sgd::Sgd;

#[cfg(test)]
pub(crate) use muon::orthogonalize;

#[cfg(test)]
#[path = "../tests/optimizer_tests.rs"]
mod tests;
