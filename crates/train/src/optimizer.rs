//! 优化器：对象安全 trait、矩状态按参数编号自持、按 [`ParamRole`](llm_nn::ParamRole) 取组。
//!
//! **拥有**：[`Optimizer`] trait、[`Group`] 分组闭集、[`Slot`] 槽位视图与 [`Sgd`] / [`Adam`]
//! 两种实现。**不拥有**：学习率调度（[`crate::Schedule`]）、梯度缓冲（[`crate::Buffers`] 扩展）、
//! 快照格式（[`crate::snapshot`]）。**依赖**：`nn`（角色闭集）、`diagnostic`。
//!
//! 两条收口口径：
//!
//! - 组归属只从角色派生：[`Group::from_role`] 是角色 → 组的唯一映射，优化器不按字段名、层名
//!   前缀猜组；权重衰减豁免同样从 [`ParamRole::wd_exempt`] 派生。
//! - 矩状态由优化器自己按槽位编号自持：autograd 只出 tape / plan / Replay，不持梯度存储；
//!   优化器实例就是矩状态的唯一存放处，快照在 T1-4 从这里导出。

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
}

impl Slot<'_> {
    /// 参数与梯度长度必须一致；不一致是调用方接线错误（[`Code::Invalid`]）。
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
}

impl OwnedSlot {
    /// 借成优化器/同步用的可变视图。
    pub fn view(&mut self) -> Slot<'_> {
        Slot {
            params: &mut self.params,
            grads: &mut self.grads,
            role: self.role,
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

/// Adam（AdamW 解耦衰减 + 偏差修正 + 全局范数裁剪）。
///
/// 一阶矩 `m`、二阶矩 `v` 按槽位编号自持：槽位编号即 `step` 调用里的下标，首次见到的槽位
/// 惰性分配零向量。偏差修正用优化器全局步数；矩状态可在 T1-4 快照导出/回载。
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    clip_norm: Option<f32>,
    step: u64,
    momentum: Vec<Vec<f32>>,
    variance: Vec<Vec<f32>>,
}

impl Adam {
    /// 默认超参：`β1 = 0.9`、`β2 = 0.999`、`ε = 1e-8`，不衰减、不裁剪。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置 `β1`（一阶矩衰减系数）。
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// 设置 `β2`（二阶矩衰减系数）。
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// 设置 `ε`（分母保护项）。
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// 确保槽位 `index` 的矩向量长度等于参数长度；首次见到的槽位分配零向量。
    fn moments(&mut self, index: usize, len: usize) -> (&mut Vec<f32>, &mut Vec<f32>) {
        if index >= self.momentum.len() {
            let missing = index - self.momentum.len() + 1;
            self.momentum
                .resize_with(self.momentum.len() + missing, Vec::new);
            self.variance
                .resize_with(self.variance.len() + missing, Vec::new);
        }
        let m = &mut self.momentum[index];
        if m.len() != len {
            m.resize(len, 0.0);
            self.variance[index].resize(len, 0.0);
        }
        (&mut self.momentum[index], &mut self.variance[index])
    }

    /// 一阶矩表（按槽位编号）：快照导出走这里，优化器之外不另存一份矩状态。
    pub fn momentum(&self) -> &[Vec<f32>] {
        &self.momentum
    }

    /// 二阶矩表（按槽位编号）。
    pub fn variance(&self) -> &[Vec<f32>] {
        &self.variance
    }

    /// 优化器全局步数（偏差修正分母）。
    pub fn step_count(&self) -> u64 {
        self.step
    }
}

decay_clip!(Adam);

impl Default for Adam {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clip_norm: None,
            step: 0,
            momentum: Vec::new(),
            variance: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()> {
        check_slots(slots)?;
        let scale = clip_scale(global_norm(slots), self.clip_norm);
        self.step = self.step.saturating_add(1);
        let t = self.step as f32;
        let bias1 = 1.0 - self.beta1.powf(t);
        let bias2 = 1.0 - self.beta2.powf(t);

        let (beta1, beta2, eps, weight_decay) =
            (self.beta1, self.beta2, self.eps, self.weight_decay);
        for (index, slot) in slots.iter_mut().enumerate() {
            let decay = decay_rate(slot.role, weight_decay);
            let (m, v) = self.moments(index, slot.params.len());
            // 逐元素串行更新：m/v 对每个元素有跨步依赖，长度随参数数据而定，不能向量化展开。
            for ((param, grad), (m_value, v_value)) in slot
                .params
                .iter_mut()
                .zip(slot.grads.iter())
                .zip(m.iter_mut().zip(v.iter_mut()))
            {
                let g = scale * grad;
                *m_value = beta1 * *m_value + (1.0 - beta1) * g;
                *v_value = beta2 * *v_value + (1.0 - beta2) * g * g;
                let m_hat = *m_value / bias1;
                let v_hat = *v_value / bias2;
                let update = m_hat / (v_hat.sqrt() + eps) + decay * *param;
                *param -= lr * update;
            }
        }
        Ok(())
    }

    fn restore(
        &mut self,
        momentum: Vec<Vec<f32>>,
        variance: Vec<Vec<f32>>,
        step: u64,
    ) -> Result<()> {
        if momentum.len() != variance.len() {
            let message = format!(
                "矩状态槽位数不一致：一阶矩 {} 槽，二阶矩 {} 槽",
                momentum.len(),
                variance.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        for (index, (m, v)) in momentum.iter().zip(&variance).enumerate() {
            if m.len() != v.len() {
                let message = format!(
                    "第 {index} 槽矩长度不一致：一阶矩 {} 元素，二阶矩 {} 元素",
                    m.len(),
                    v.len()
                );
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
        }
        self.momentum = momentum;
        self.variance = variance;
        self.step = step;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use diagnostic::Code;
    use nn::ParamRole;

    fn slots<'a>(
        params: &'a mut [Vec<f32>],
        grads: &'a mut [Vec<f32>],
        roles: &[ParamRole],
    ) -> Vec<Slot<'a>> {
        params
            .iter_mut()
            .zip(grads.iter_mut())
            .zip(roles.iter())
            .map(|((params, grads), role)| Slot {
                params,
                grads,
                role: *role,
            })
            .collect()
    }

    /// 角色 → 组的唯一映射：四个闭集角色逐项对拍，短名同步对齐。
    #[test]
    fn group_mapping_follows_the_role() {
        let cases = [
            (ParamRole::Trainable, Group::Trainable, "trainable"),
            (ParamRole::Bias, Group::Bias, "bias"),
            (ParamRole::Norm, Group::Norm, "norm"),
            (ParamRole::Embedding, Group::Embedding, "embedding"),
        ];
        for (role, group, name) in cases {
            assert_eq!(Group::from_role(role), group);
            assert_eq!(group.as_str(), name);
        }
    }

    /// SGD 一步：参数按 `-lr · grad` 更新，无衰减时数值可逐项断言。
    #[test]
    fn sgd_takes_one_direct_step() {
        let mut params = [vec![1.0_f32, 2.0], vec![0.5]];
        let mut grads = [vec![0.5_f32, -1.0], vec![0.25]];
        let roles = [ParamRole::Trainable, ParamRole::Trainable];
        let mut optimizer = Sgd::new();

        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("长度一致");

        assert_eq!(params[0], [0.95_f32, 2.1]);
        assert_eq!(params[1], [0.475]);
    }

    /// 权重衰减只作用于非豁免组：归一化系数不衰减，投影权重按解耦口径衰减。
    #[test]
    fn weight_decay_applies_only_outside_exempt_groups() {
        let mut params = [vec![1.0_f32], vec![1.0]];
        let mut grads = [vec![0.2_f32], vec![0.2]];
        let roles = [ParamRole::Norm, ParamRole::Trainable];
        let mut optimizer = Sgd::new().with_weight_decay(0.1);

        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("长度一致");

        // 豁免组：只减梯度项 0.1·0.2 = 0.02。
        assert!((params[0][0] - 0.98).abs() < 1e-7);
        // 衰减组：更新 = 0.2 + 0.1·1.0 = 0.3，减 0.1·0.3 = 0.03。
        assert!((params[1][0] - 0.97).abs() < 1e-7);
    }

    /// 裁剪按全局范数等比缩比：梯度极大时一步更新仍在阈值口径内。
    #[test]
    fn global_norm_clipping_scales_every_slot() {
        let mut params = [vec![1.0_f32]];
        let mut grads = [vec![100.0_f32]];
        let roles = [ParamRole::Trainable];
        let mut optimizer = Sgd::new().with_clip_norm(1.0);

        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 1.0)
            .expect("长度一致");

        // 缩比 0.01：有效梯度 1.0，参数减 1.0。
        assert!((params[0][0] - 0.0).abs() < 1e-7);
    }

    /// 参数与梯度长度不一致是接线错误，一步都不更新。
    #[test]
    fn a_length_mismatch_is_rejected_before_update() {
        let mut params = [vec![1.0_f32, 2.0]];
        let mut grads = [vec![0.5_f32]];
        let roles = [ParamRole::Trainable];
        let mut optimizer = Sgd::new();

        let error = optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect_err("长度不一致");
        assert_eq!(error.code(), Code::Invalid);
        assert_eq!(params[0], [1.0, 2.0], "拒绝前不改写参数");
    }

    /// Adam 一步后权重沿梯度方向下降；矩状态由优化器自持，第二步不再是裸梯度方向。
    #[test]
    fn adam_moves_the_param_with_momentum() {
        let mut params = [vec![1.0_f32]];
        let mut grads = [vec![0.2_f32]];
        let roles = [ParamRole::Trainable];
        let mut optimizer = Adam::new();

        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("长度一致");
        assert!(params[0][0] < 1.0, "第一步沿梯度方向下降");
        assert!(params[0][0] > 0.85, "偏差修正后步幅小于裸 SGD");

        let first = params[0][0];
        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("第二步");
        assert!(params[0][0] < first, "矩状态跨步保留，继续下降");
    }

    /// 矩状态可导出并回载进新优化器：快照搬运的是这一份表，优化器之外不另存副本。
    #[test]
    fn adam_moments_round_trip_through_restore() {
        let mut params = [vec![1.0_f32, 2.0]];
        let mut grads = [vec![0.2_f32, -0.1]];
        let roles = [ParamRole::Trainable];
        let mut optimizer = Adam::new();
        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("第一步");
        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("第二步");

        let momentum = optimizer.momentum().iter().map(Vec::clone).collect();
        let variance = optimizer.variance().iter().map(Vec::clone).collect();
        let step = optimizer.step_count();

        let mut restored = Adam::new();
        restored.restore(momentum, variance, step).expect("回载");
        assert_eq!(restored.step_count(), step);
        let bits = |table: &[Vec<f32>]| {
            table
                .iter()
                .map(|slot| slot.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(restored.momentum()),
            bits(optimizer.momentum()),
            "一阶矩逐位恢复"
        );
        assert_eq!(
            bits(restored.variance()),
            bits(optimizer.variance()),
            "二阶矩逐位恢复"
        );

        // 回载后矩状态延续：第三步的一阶矩含上一时刻的动量，不是从零起。
        let mut params_restored = [params[0].clone()];
        let mut params_source = [params[0].clone()];
        restored
            .step(&mut slots(&mut params_restored, &mut grads, &roles), 0.1)
            .expect("回载后第一步");
        optimizer
            .step(&mut slots(&mut params_source, &mut grads, &roles), 0.1)
            .expect("源优化器同一步");
        assert_eq!(params_restored[0], params_source[0], "回载后轨迹逐位对齐");
    }

    /// 回载的两张表槽位数或槽内长度不符：接线错误，先报错不改写矩状态。
    #[test]
    fn restore_rejects_mismatched_moment_tables() {
        let mut optimizer = Adam::new();
        let error = optimizer
            .restore(vec![vec![1.0]], vec![vec![1.0, 2.0]], 3)
            .expect_err("长度不符");
        assert_eq!(error.code(), Code::Invalid);
        let error = optimizer
            .restore(vec![vec![1.0]], vec![], 3)
            .expect_err("槽位数不符");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// 对象安全：优化器可装进 `&mut dyn Optimizer`。
    #[test]
    fn optimizer_is_object_safe() {
        let mut params = [vec![1.0_f32]];
        let mut grads = [vec![0.5_f32]];
        let roles = [ParamRole::Trainable];
        let mut optimizer: Box<dyn Optimizer> = Box::new(Sgd::new());

        optimizer
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect("长度一致");
        assert!((params[0][0] - 0.95).abs() < 1e-7);
    }
}
