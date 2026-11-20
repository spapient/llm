//! 优化器：对象安全 trait、矩状态按参数编号自持、按 [`ParamRole`](llm_nn::ParamRole) 取组。
//!
//! **拥有**：[`Optimizer`] trait、[`Group`] 分组闭集、[`Slot`] 槽位视图与 [`Sgd`] / [`Adam`] /
//! [`Muon`] 三种实现。**不拥有**：学习率调度（[`crate::Schedule`]）、梯度缓冲
//! （[`crate::Buffers`] 扩展）、快照格式（[`crate::snapshot`]）。**依赖**：`nn`（角色闭集）、
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

/// Muon（MomentUm Orthogonalized by Newton–Schulz）：矩阵参数的近似正交化更新。
///
/// 出处是 Keller Jordan 的 Muon（第 10 款标杆 K3 的训练侧，§5.8.2 把 muon 与 mHC 一起列为**共用**项）。
/// 它只对**矩阵**参数有定义：偏置与归一化增益不在它的作用域里，这类槽的 [`Slot::shape`] 是 `None`，
/// 本优化器**拒绝**它们而不是悄悄跳过——跳过是「这一步没更新」，跑一万步就是某个组从未被训练过，
/// 而训练曲线不会告诉你少了谁。矩阵/向量的分派在装配侧（真实配方是 Muon 管矩阵、AdamW 管其余），
/// `Session` 每步只调一种优化器。
///
/// 一步的口径：
///
/// 1. 动量缓冲 `u ← μ·u + g`；Nesterov 变体再取 `g + μ·u` 当方向。
/// 2. 把 `u / (‖u‖_F + ε)` 用五次 Newton–Schulz 迭代推向正交阵——奇异值全部落在 1 附近，梯度谱再烂
///    也不会让某几个方向独占步长。这是「保持更新不被少数方向垄断」的唯一手段，不是数值技巧。
/// 3. `W ← W − lr·0.2·√max(rows, cols)·O − lr·decay·W`；缩放因子把正交更新的 RMS 拉回与 AdamW
///    同量级（Muon 参考实现的 `match_rms_adamw` 口径），否则同一条学习率曲线在两族参数上不是一个意思。
///
/// 关键性质：更新量只由**方向**决定，与梯度的尺度无关（先按 Frobenius 范数归一）。梯度整体乘 1000，
/// 一步走多远不变——这条有单测钉着（`the_update_ignores_the_gradient_scale`）。
#[derive(Debug, Clone, PartialEq)]
pub struct Muon {
    momentum: f32,
    nesterov: bool,
    eps: f32,
    weight_decay: f32,
    clip_norm: Option<f32>,
    step: u64,
    velocity: Vec<Vec<f32>>,
}

/// Newton–Schulz 的迭代次数：Muon 参考实现取 5，够把 `[0.3, 3]` 的谱压到 1 附近。
const NS_STEPS: usize = 5;

/// 五次迭代的多项式系数 `(a, b, c)`：`X ← aX + b(XA) + c(XA²)`，`A` 是小边上的 Gram。
const NS_COEF: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// 步长缩放的系数：与参考实现一致，把正交更新的尺度对齐 AdamW。
const SCALE_COEF: f32 = 0.2;

impl Muon {
    /// 默认超参：`μ = 0.95`、Nesterov、`ε = 1e-7`，不衰减、不裁剪。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置动量系数 `μ`。
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// 关掉 Nesterov 前瞻（默认开）。
    pub fn without_nesterov(mut self) -> Self {
        self.nesterov = false;
        self
    }

    /// 设置归一化保护项 `ε`。
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// 确保槽位 `index` 的动量缓冲长度等于参数长度；首次见到的槽位分配零向量。
    fn velocity(&mut self, index: usize, len: usize) -> &mut Vec<f32> {
        if index >= self.velocity.len() {
            let missing = index - self.velocity.len() + 1;
            self.velocity
                .resize_with(self.velocity.len() + missing, Vec::new);
        }
        let buffer = &mut self.velocity[index];
        if buffer.len() != len {
            buffer.resize(len, 0.0);
        }
        buffer
    }

    /// 一个槽的推进：动量累加 →（可选 Nesterov）→ 正交化 → 写回。
    ///
    /// 标量先复制出来再取动量缓冲：`velocity` 借了整个 `self`，之后再读 `self.nesterov`
    /// 就是同一块可变借用的第二份访问。
    fn advance(
        &mut self,
        index: usize,
        slot: &mut Slot<'_>,
        (rows, cols): (usize, usize),
        scale: f32,
        lr: f32,
    ) {
        let (beta, nesterov, eps) = (self.momentum, self.nesterov, self.eps);
        let decay = decay_rate(slot.role, self.weight_decay);
        let velocity = self.velocity(index, rows * cols);
        for (tracked, &grad) in velocity.iter_mut().zip(slot.grads.iter()) {
            *tracked = beta * *tracked + scale * grad;
        }
        let direction = if nesterov {
            nesterov_direction(velocity, slot.grads, beta, scale)
        } else {
            velocity.clone()
        };
        let orthogonal = orthogonalize(&direction, rows, cols, eps);
        let factor = lr * scale_factor(rows, cols);
        for (param, &step) in slot.params.iter_mut().zip(&orthogonal) {
            *param -= factor * step;
            *param -= lr * decay * *param;
        }
    }

    /// 动量表（按槽位编号）：快照导出走这里，优化器之外不另存一份状态。
    pub fn momentum_table(&self) -> &[Vec<f32>] {
        &self.velocity
    }

    /// 优化器全局步数。
    pub fn step_count(&self) -> u64 {
        self.step
    }
}

decay_clip!(Muon);

impl Default for Muon {
    fn default() -> Self {
        Self {
            momentum: 0.95,
            nesterov: true,
            eps: 1e-7,
            weight_decay: 0.0,
            clip_norm: None,
            step: 0,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Muon {
    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()> {
        check_slots(slots)?;
        let shapes = shapes_of(slots)?;
        let scale = clip_scale(global_norm(slots), self.clip_norm);
        self.step = self.step.saturating_add(1);
        for (index, (slot, &shape)) in slots.iter_mut().zip(&shapes).enumerate() {
            self.advance(index, slot, shape, scale, lr);
        }
        Ok(())
    }

    fn restore(
        &mut self,
        momentum: Vec<Vec<f32>>,
        variance: Vec<Vec<f32>>,
        step: u64,
    ) -> Result<()> {
        if !variance.is_empty() {
            let message = format!(
                "Muon 只持一阶动量：快照带回 {} 个二阶矩槽，无法回载",
                variance.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        self.velocity = momentum;
        self.step = step;
        Ok(())
    }
}

/// 每个槽的矩阵形状，**全部**先查完再返回。
///
/// 不能边查边写：查到第 7 个槽才发现它不是矩阵时，前 6 个参数已经写回去了，调用方拿到错误却
/// 留下半步更新。
fn shapes_of(slots: &[Slot<'_>]) -> Result<Vec<(usize, usize)>> {
    let mut shapes = Vec::with_capacity(slots.len());
    for (index, slot) in slots.iter().enumerate() {
        shapes.push(matrix_of(index, slot)?);
    }
    Ok(shapes)
}

/// Nesterov 前瞻方向 `g + μ·u`：动量缓冲 `u` 已经含了本步的梯度，前瞻取的是「再多走一点」。
fn nesterov_direction(velocity: &[f32], grads: &[f32], beta: f32, scale: f32) -> Vec<f32> {
    velocity
        .iter()
        .zip(grads)
        .map(|(&tracked, &grad)| scale * grad + beta * tracked)
        .collect()
}

/// 一个槽的矩阵形状：非矩阵、零维或梯度非有限都在这里拒绝。
///
/// 形状错、长度错、`NaN` 都会让正交化给出一个"看起来正常"的方向，报错点却离原因很远。
fn matrix_of(index: usize, slot: &Slot<'_>) -> Result<(usize, usize)> {
    let Some((rows, cols)) = slot.shape else {
        let message = format!(
            "第 {index} 槽（角色 {}）不是矩阵：Muon 的正交化只在矩阵上有定义，\
             偏置/归一化增益按角色分派给 AdamW",
            slot.role.as_str()
        );
        return Err(Error::new(Code::Invalid, message).with(SITE));
    };
    if rows == 0 || cols == 0 {
        let message = format!("第 {index} 槽声明为 [{rows}, {cols}]：两边都必须非零");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    if let Some(bad) = slot.grads.iter().position(|grad| !grad.is_finite()) {
        let message = format!("第 {index} 槽梯度第 {bad} 项非有限：正交化会把它铺满整个方向");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok((rows, cols))
}

/// 步长缩放 `0.2·√max(rows, cols)`：正交后奇异值为 1，不缩放的话更新量随矩阵阶数增长。
fn scale_factor(rows: usize, cols: usize) -> f32 {
    SCALE_COEF * (rows.max(cols) as f32).sqrt()
}

/// 近似正交化：先按 Frobenius 范数归一，再迭代 [`NS_STEPS`] 次 Newton–Schulz。
///
/// 每次迭代用**小边**上的 Gram（`rows ≥ cols` 取 `XᵀX`，否则 `X Xᵀ`）：两条路径在数学上是同一个
/// 迭代（`X(XᵀX) = (XXᵀ)X`），差别只是乘在左边还是右边、以及临时矩阵的阶数。零矩阵归一后仍是零，
/// 于是「这一步没有梯度」自然给出零更新，不需要另开一个分支。
fn orthogonalize(grad: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let norm = grad.iter().fold(0.0_f32, |acc, v| acc + v * v).sqrt();
    let scale = 1.0 / (norm + eps);
    let mut x: Vec<f32> = grad.iter().map(|v| v * scale).collect();
    for _ in 0..NS_STEPS {
        let (gram, side) = if rows >= cols {
            (gram_of_columns(&x, rows, cols), Side::Right)
        } else {
            (gram_of_rows(&x, rows, cols), Side::Left)
        };
        let squared = square(&gram, side.order(rows, cols));
        x = advance(&x, rows, cols, &gram, &squared, side);
    }
    x
}

/// `XᵀX`（`cols × cols`）：行数不少于列数时的小边。
fn gram_of_columns(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0_f32; cols * cols];
    for row in 0..rows {
        let lane = &x[row * cols..(row + 1) * cols];
        for col in 0..cols {
            let left = lane[col];
            if left == 0.0 {
                continue;
            }
            for inner in 0..cols {
                gram[col * cols + inner] += left * lane[inner];
            }
        }
    }
    gram
}

/// `X Xᵀ`（`rows × rows`）：列数多于行数时的小边。
fn gram_of_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0_f32; rows * rows];
    for outer in 0..rows {
        for inner in 0..rows {
            let mut acc = 0.0_f32;
            for col in 0..cols {
                acc += x[outer * cols + col] * x[inner * cols + col];
            }
            gram[outer * rows + inner] = acc;
        }
    }
    gram
}

/// 方阵自乘 `M·M`（行主序）。Gram 对称且阶数是小边，直接写三重循环。
fn square(matrix: &[f32], order: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; order * order];
    for row in 0..order {
        for inner in 0..order {
            let left = matrix[row * order + inner];
            if left == 0.0 {
                continue;
            }
            for col in 0..order {
                out[row * order + col] += left * matrix[inner * order + col];
            }
        }
    }
    out
}

/// Gram 乘在 `X` 的哪一边：小边是列时乘在右边（`XA`），小边是行时乘在左边（`AX`）。
#[derive(Clone, Copy)]
enum Side {
    /// `A` 是 `rows × rows`（小边是行），乘在 `X` 左边。
    Left,
    /// `A` 是 `cols × cols`（小边是列），乘在 `X` 右边。
    Right,
}

impl Side {
    /// 小边的阶数：Gram 与 `A²` 都是这个阶数的方阵。
    fn order(self, rows: usize, cols: usize) -> usize {
        match self {
            Side::Left => rows,
            Side::Right => cols,
        }
    }
}

/// 一步 `X ← aX + b(AX) + c(A²X)`；`side` 为 [`Side::Right`] 时是 `X ← aX + b(XA) + c(XA²)`。
/// 两者遍历的是同一块 `X`，只是 `A` 乘在另一侧：沿哪条轴收缩、Gram 槽位怎么排跟着 `side` 变。
fn advance(
    x: &[f32],
    rows: usize,
    cols: usize,
    gram: &[f32],
    squared: &[f32],
    side: Side,
) -> Vec<f32> {
    let (a, b, c) = NS_COEF;
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = a * x[row * cols + col];
            for inner in 0..side.order(rows, cols) {
                let (value, slot) = match side {
                    Side::Left => (x[inner * cols + col], row * rows + inner),
                    Side::Right => (x[row * cols + inner], inner * cols + col),
                };
                acc += b * value * gram[slot] + c * value * squared[slot];
            }
            out[row * cols + col] = acc;
        }
    }
    out
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
                shape: None,
            })
            .collect()
    }

    /// 带矩阵形状的槽视图：Muon 的入口要求形状随槽一起给出。
    fn matrix_slots<'a>(
        params: &'a mut [Vec<f32>],
        grads: &'a mut [Vec<f32>],
        roles: &[ParamRole],
        shape: (usize, usize),
    ) -> Vec<Slot<'a>> {
        params
            .iter_mut()
            .zip(grads.iter_mut())
            .zip(roles.iter())
            .map(|((params, grads), role)| Slot {
                params,
                grads,
                role: *role,
                shape: Some(shape),
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

    /// 正交化把谱压窄：`3 : 2 : 1` 的传播被拉到窄带里，方向不再被几个分量独占。
    ///
    /// 五次迭代给的是**近似**极分解（收敛比约 0.72，值在 `[0.65, 1.15]` 一带振荡），所以断言的是
    /// “压缩比”而不是“精确等于 1”：后者要几十次迭代，而优化器只需要方向不失衡。
    #[test]
    fn the_orthogonalization_compresses_the_spectrum() {
        // 对角谱 1 / 2 / 3：不处理的话，更新量会按 3 : 2 : 1 分配。
        let grad = vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let out = orthogonalize(&grad, 3, 3, 1e-7);
        for (index, value) in out.iter().enumerate() {
            assert!(value.is_finite(), "第 {index} 项不是有限值");
        }
        let spread = out[0] / out[8];
        assert!(
            (0.5..=1.8).contains(&spread),
            "谱比从 3.0 被压到 {spread}，该落进 0.5..=1.8 的窄带"
        );
        // 对角输入只受多项式作用，非对角项仍是零：正交化不动特征基。
        assert!(
            out[1].abs() < 1e-6 && out[3].abs() < 1e-6,
            "非对角项该保持为零"
        );
    }

    /// 更新量与梯度尺度无关：整体乘 1000，一步走多远不变。
    #[test]
    fn the_update_ignores_the_gradient_scale() {
        let base = vec![1.0_f32, -2.0, 0.5, 3.0, 0.0, -1.0];
        let mut params_small = [vec![1.0_f32; 6]];
        let mut params_large = [vec![1.0_f32; 6]];
        let mut grads_small = [base.clone()];
        let mut grads_large = [base.iter().map(|value| value * 1000.0).collect()];
        let roles = [ParamRole::Trainable];

        Muon::new()
            .without_nesterov()
            .step(
                &mut matrix_slots(&mut params_small, &mut grads_small, &roles, (2, 3)),
                0.1,
            )
            .expect("几何合法");
        Muon::new()
            .without_nesterov()
            .step(
                &mut matrix_slots(&mut params_large, &mut grads_large, &roles, (2, 3)),
                0.1,
            )
            .expect("几何合法");

        for (index, (small, large)) in params_small[0].iter().zip(&params_large[0]).enumerate() {
            assert!(
                (small - large).abs() < 1e-3,
                "第 {index} 项：小梯度 {small} 与大梯度 {large} 该给出同一步"
            );
            assert!((small - 1.0).abs() > 1e-9, "第 {index} 项根本没被更新");
        }
    }

    /// 非矩阵槽被拒且一步都不写回：跳过它等于让某个组永远不被训练。
    #[test]
    fn a_flat_slot_is_refused_before_any_update() {
        let mut params = [vec![1.0_f32, 2.0, 3.0]];
        let mut grads = [vec![0.5_f32, 0.5, 0.5]];
        let roles = [ParamRole::Bias];
        let error = Muon::new()
            .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
            .expect_err("偏置不是矩阵");
        assert_eq!(error.code(), Code::Invalid);
        assert_eq!(params[0], [1.0, 2.0, 3.0], "拒绝前不改写参数");
    }

    /// 声明的行列积与参数长度不符：接线错误，在槽校验处就失败。
    #[test]
    fn a_shape_that_does_not_match_the_length_is_refused() {
        let mut params = [vec![1.0_f32, 2.0, 3.0, 4.0]];
        let mut grads = [vec![0.5_f32; 4]];
        let roles = [ParamRole::Trainable];
        let error = Muon::new()
            .step(
                &mut matrix_slots(&mut params, &mut grads, &roles, (3, 3)),
                0.1,
            )
            .expect_err("3 × 3 不是 4 个元素");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// 非有限梯度在入口被拒：正交化会把 `NaN` 铺满整个方向。
    #[test]
    fn a_non_finite_gradient_is_refused() {
        let mut params = [vec![1.0_f32; 4]];
        let mut grads = [vec![1.0_f32, f32::NAN, 0.0, 0.0]];
        let roles = [ParamRole::Trainable];
        let error = Muon::new()
            .step(
                &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
                0.1,
            )
            .expect_err("梯度含 NaN");
        assert_eq!(error.code(), Code::Invalid);
        assert_eq!(params[0], [1.0, 1.0, 1.0, 1.0]);
    }

    /// 零梯度给出零更新：没有方向就没有正交化的东西，也不需要另开分支。
    #[test]
    fn a_zero_gradient_leaves_the_matrix_alone() {
        let mut params = [vec![1.0_f32, -2.0, 3.0, 0.5]];
        let mut grads = [vec![0.0_f32; 4]];
        let roles = [ParamRole::Trainable];
        Muon::new()
            .step(
                &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
                0.1,
            )
            .expect("几何合法");
        assert_eq!(params[0], [1.0, -2.0, 3.0, 0.5], "零方向不该动参数");
    }

    /// 解耦衰减跳过豁免角色：偏置原样保留，可训练权重按 `lr·decay` 缩。
    #[test]
    fn the_decay_still_follows_the_role() {
        let mut params = [vec![1.0_f32; 4], vec![1.0_f32; 4]];
        let mut grads = [vec![0.0_f32; 4], vec![0.0_f32; 4]];
        let roles = [ParamRole::Trainable, ParamRole::Bias];
        Muon::new()
            .with_weight_decay(0.1)
            .step(
                &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
                0.5,
            )
            .expect("几何合法");
        assert!(
            (params[0][0] - (1.0 - 0.5 * 0.1)).abs() < 1e-7,
            "可训练权重按衰减缩"
        );
        assert_eq!(params[1][0], 1.0, "偏置豁免权重衰减");
    }

    /// 动量跨步累积：两步同梯度后缓冲为 `μ(μ·g + g) + g`。
    #[test]
    fn the_velocity_accumulates_across_steps() {
        let mut params = [vec![1.0_f32; 4]];
        let mut grads = [vec![0.0_f32, 1.0, 0.0, 0.0]];
        let roles = [ParamRole::Trainable];
        let mut optimizer = Muon::new().with_momentum(0.5);
        for _ in 0..2 {
            optimizer
                .step(
                    &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
                    0.1,
                )
                .expect("几何合法");
        }
        let velocity = &optimizer.momentum_table()[0][1];
        let want = (1.0 - 0.5_f32.powi(2)) / (1.0 - 0.5);
        assert!(
            (velocity - want).abs() < 1e-6,
            "缓冲是 {velocity}，手算是 {want}"
        );
        assert_eq!(optimizer.step_count(), 2);
    }

    /// 回载后轨迹逐位对齐；二阶矩非空时拒绝（Muon 只持一阶动量）。
    #[test]
    fn the_restore_round_trip_matches_the_source() {
        let mut params = [vec![1.0_f32; 4]];
        let mut grads = [vec![0.25_f32, -1.0, 2.0, 0.0]];
        let roles = [ParamRole::Trainable];
        let mut source = Muon::new();
        source
            .step(
                &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
                0.1,
            )
            .expect("第一步");
        let table = source.momentum_table().to_vec();

        let mut restored = Muon::new();
        restored.restore(table, Vec::new(), 1).expect("表一致");
        let mut params_restored = [params[0].clone()];
        let mut params_source = [params[0].clone()];
        restored
            .step(
                &mut matrix_slots(&mut params_restored, &mut grads, &roles, (2, 2)),
                0.1,
            )
            .expect("回载后第一步");
        source
            .step(
                &mut matrix_slots(&mut params_source, &mut grads, &roles, (2, 2)),
                0.1,
            )
            .expect("源优化器同一步");
        assert_eq!(params_restored[0], params_source[0], "回载后轨迹逐位对齐");

        let error = Muon::new()
            .restore(Vec::new(), vec![vec![1.0]], 1)
            .expect_err("Muon 没有二阶矩");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// 高瘦与矮宽走的是同一个迭代：转置后的方向就是转置。
    #[test]
    fn the_tall_and_wide_branches_are_the_same_iteration() {
        let tall = vec![0.5_f32, -1.0, 2.0, 0.25, 3.0, -0.5];
        let mut wide = Vec::with_capacity(tall.len());
        for col in 0..2 {
            for row in 0..3 {
                wide.push(tall[row * 2 + col]);
            }
        }
        let tall_out = orthogonalize(&tall, 3, 2, 1e-7);
        let wide_out = orthogonalize(&wide, 2, 3, 1e-7);
        for row in 0..3 {
            for col in 0..2 {
                let lhs = tall_out[row * 2 + col];
                let rhs = wide_out[col * 3 + row];
                assert!((lhs - rhs).abs() < 1e-6, "({row}, {col})：{lhs} 对 {rhs}");
            }
        }
    }
}
