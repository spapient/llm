//! 优化器：对象安全 trait、矩状态按参数编号自持、按 [`ParamRole`](llm_nn::ParamRole) 取组。

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
pub struct Slot<'a> {
    /// 参数值（就地写回）。
    pub params: &'a mut [f32],
    /// 当前梯度（同步阶段可就地归约/平均，步进阶段只读）。
    pub grads: &'a mut [f32],
    /// 参数角色：分组与衰减豁免的派生依据。
    pub role: ParamRole,
    /// 矩阵形状 `(rows, cols)`：参数在图上登记为二阶张量时给出，否则 `None`。
    pub shape: Option<(usize, usize)>,
}

impl Slot<'_> {
    /// 参数与梯度长度必须一致，且矩阵槽的行列积必须等于参数长度。
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

pub use adam::Adam;
pub use muon::Muon;

#[cfg(test)]
pub(crate) use muon::orthogonalize;

#[cfg(test)]
#[path = "../tests/optimizer_tests.rs"]
mod tests;

/// 无矩状态；衰减按 AdamW 同款解耦口径：`param -= lr · (grad + wd · param)`，角色豁免组
/// 不加衰减项。
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd {
    weight_decay: f32,
    clip_norm: Option<f32>,
}

impl Sgd {
    /// 默认：不衰减、不裁剪。
    pub fn new() -> Self {
        Self::default()
    }
}

decay_clip!(Sgd);

impl Default for Sgd {
    fn default() -> Self {
        Self {
            weight_decay: 0.0,
            clip_norm: None,
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()> {
        check_slots(slots)?;
        let scale = clip_scale(global_norm(slots), self.clip_norm);
        for slot in slots {
            let decay = decay_rate(slot.role, self.weight_decay);
            // 逐元素串行写回：每个参数只依赖自己的梯度与现值，长度随槽位数据而定。
            for (param, grad) in slot.params.iter_mut().zip(slot.grads.iter()) {
                let update = scale * grad + decay * *param;
                *param -= lr * update;
            }
        }
        Ok(())
    }

    fn restore(
        &mut self,
        momentum: Vec<Vec<f32>>,
        variance: Vec<Vec<f32>>,
        _step: u64,
    ) -> Result<()> {
        if !momentum.is_empty() || !variance.is_empty() {
            let message = format!(
                "SGD 无矩状态：快照带回 {} 个一阶矩槽、{} 个二阶矩槽，无法回载",
                momentum.len(),
                variance.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        Ok(())
    }
}
