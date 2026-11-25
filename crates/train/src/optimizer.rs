//! 参数更新：统一配置预检、全局裁剪、矩状态恢复与按角色衰减。
use crate::orthogonalize;
use diagnostic::{Code, Error, Result};
use nn::ParamRole;

pub(crate) const SITE: &str = "train::Optimizer";
type Table = Vec<Vec<f32>>;
type Shapes<'a> = &'a [Option<(usize, usize)>];

/// 参数角色对应的更新组。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// 可衰减权重。
    Trainable,
    /// 偏置。
    Bias,
    /// 归一化参数。
    Norm,
    /// 嵌入表。
    Embedding,
}
impl Group {
    /// 从 `nn` 的角色闭集派生，不维护第二份标签表。
    pub fn from_role(role: ParamRole) -> Self {
        match role {
            ParamRole::Trainable => Self::Trainable,
            ParamRole::Bias => Self::Bias,
            ParamRole::Norm => Self::Norm,
            ParamRole::Embedding => Self::Embedding,
        }
    }
    /// 诊断短名复用 `ParamRole`。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trainable => ParamRole::Trainable,
            Self::Bias => ParamRole::Bias,
            Self::Norm => ParamRole::Norm,
            Self::Embedding => ParamRole::Embedding,
        }
        .as_str()
    }
}

/// 参数槽的泛型数据：借用视图与拥有所有权共用同一布局。
pub struct SlotData<P, G> {
    /// 参数值。
    pub params: P,
    /// 梯度。
    pub grads: G,
    /// 参数角色。
    pub role: ParamRole,
    /// 矩阵形状，非矩阵为 `None`。
    pub shape: Option<(usize, usize)>,
}
/// 一个参数槽的可变更新视图。
pub type Slot<'a> = SlotData<&'a mut [f32], &'a mut [f32]>;
/// 会话取走的参数/梯度所有权，步进后按槽归还。
pub type OwnedSlot = SlotData<Vec<f32>, Vec<f32>>;

impl Slot<'_> {
    fn reborrow(&mut self) -> Slot<'_> {
        Slot {
            params: self.params,
            grads: self.grads,
            role: self.role,
            shape: self.shape,
        }
    }
}
impl OwnedSlot {
    /// 借成短生命周期更新视图。
    pub fn view(&mut self) -> Slot<'_> {
        Slot {
            params: &mut self.params,
            grads: &mut self.grads,
            role: self.role,
            shape: self.shape,
        }
    }
}

/// 对象安全优化器；所有实现必须先用同一预检再修改状态。
pub trait Optimizer {
    /// 更新全部槽位；非法输入不得留下半次更新。
    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()>;
    /// 设置全局梯度范数上限。
    fn set_clip_norm(&mut self, clip_norm: Option<f32>);
    /// 校验静态超参；会话装配与直接步进均调用。
    fn validate(&self) -> Result<()> {
        Ok(())
    }
    /// 回载矩状态与优化器步数。
    fn restore(&mut self, momentum: Table, variance: Table, step: u64) -> Result<()>;
    /// 预检恢复输入；实现必须保证同一输入随后可原子提交。
    fn validate_restore(
        &self,
        momentum: &[Vec<f32>],
        variance: &[Vec<f32>],
        shapes: Shapes<'_>,
    ) -> Result<()> {
        let _ = shapes;
        paired(momentum, variance)
    }
    /// 带参数形状的恢复入口，混合路由据此分派子状态。
    fn restore_shaped(
        &mut self,
        momentum: Table,
        variance: Table,
        step: u64,
        shapes: Shapes<'_>,
    ) -> Result<()> {
        let _ = shapes;
        self.restore(momentum, variance, step)
    }
    /// 写入影响更新轨迹的静态配置指纹。
    fn write_config(&self, sink: &mut dyn FnMut(u64)) {
        let _ = sink;
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct DecayClip {
    decay: f32,
    clip: Option<f32>,
}
impl DecayClip {
    fn validate(&self) -> Result<()> {
        validation_rules!(SITE;
            self.decay.is_finite() && self.decay >= 0.0 => "权重衰减必须是有限非负数",
            self.clip.is_none_or(|v| v.is_finite() && v > 0.0) => "裁剪范数必须是有限正数",
        );
        Ok(())
    }
    fn scale(&self, norm: f32) -> f32 {
        match self.clip {
            Some(max) if norm > max => max / norm,
            _ => 1.0,
        }
    }
    fn write(&self, sink: &mut dyn FnMut(u64)) {
        sink(self.decay.to_bits().into());
        sink(self.clip.is_some().into());
        if let Some(value) = self.clip {
            sink(value.to_bits().into());
        }
    }
}

macro_rules! stock {
    ($ty:ident, $doc:literal, {$($field:ident:$ftype:ty = $default:expr),*},
     $valid:ident, $check:ident, $restore:ident, $config:ident) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty {
            tuning: DecayClip,
            $($field: $ftype),*
        }
        impl Default for $ty {
            fn default() -> Self {
                Self { tuning: DecayClip::default(), $($field: $default),* }
            }
        }
        impl $ty {
            /// 构造默认优化器。
            pub fn new() -> Self { Self::default() }
            /// 设置解耦权重衰减；合法性在装配/步进边界校验。
            pub fn with_weight_decay(mut self, value: f32) -> Self {
                self.tuning.decay = value;
                self
            }
            /// 设置全局裁剪阈值；非法值不静默改成不裁剪。
            pub fn with_clip_norm(mut self, value: f32) -> Self {
                self.tuning.clip = Some(value);
                self
            }
        }
        impl Optimizer for $ty {
            fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()> {
                self.validate()?;
                let norm = check_slots(slots, lr, self.tuning.clip.is_some())?;
                self.advance(slots, lr, norm)
            }
            fn set_clip_norm(&mut self, value: Option<f32>) { self.tuning.clip = value; }
            fn validate(&self) -> Result<()> {
                self.tuning.validate()?;
                $valid(self)
            }
            fn validate_restore(&self, m: &[Vec<f32>], v: &[Vec<f32>], _: Shapes<'_>) -> Result<()> {
                $check(m, v)
            }
            fn restore(&mut self, m: Table, v: Table, step: u64) -> Result<()> {
                $restore(self, m, v, step)
            }
            fn write_config(&self, sink: &mut dyn FnMut(u64)) {
                $config(self, sink);
                self.tuning.write(sink);
            }
        }
    };
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with(SITE)
}
fn decay(role: ParamRole, value: f32) -> f32 {
    if role.wd_exempt() { 0.0 } else { value }
}
fn check_lr(lr: f32) -> Result<()> {
    validation_rules!(SITE; lr.is_finite() && lr >= 0.0 => "学习率必须是有限非负数");
    Ok(())
}
fn check_slots(slots: &[Slot<'_>], lr: f32, clip: bool) -> Result<f32> {
    check_lr(lr)?;
    slots
        .iter()
        .enumerate()
        .try_for_each(|(idx, slot)| check_slot(idx, slot))?;
    gradient_norm(slots, clip)
}

fn gradient_norm(slots: &[Slot<'_>], required: bool) -> Result<f32> {
    if !required {
        // IEEE-754 指数全 1 当且仅当 NaN/Inf；最大值归约无短路分支，可向量化扫描。
        let exponents = slots
            .iter()
            .flat_map(|slot| slot.grads.iter())
            .fold(0u32, |max, grad| max.max(grad.to_bits() & 0x7f80_0000));
        validation_rules!(SITE; exponents != 0x7f80_0000 => "梯度含 NaN / Infinity");
        return Ok(0.0);
    }
    let norm = slots
        .iter()
        .flat_map(|slot| slot.grads.iter())
        .map(|grad| grad * grad)
        .sum::<f32>()
        .sqrt();
    validation_rules!(SITE; norm.is_finite() => "全局梯度范数非有限（含 NaN / Infinity 或溢出）");
    Ok(norm)
}

fn check_slot(idx: usize, slot: &Slot<'_>) -> Result<()> {
    let shape_ok = slot
        .shape
        .is_none_or(|(r, c)| r > 0 && c > 0 && r.checked_mul(c) == Some(slot.params.len()));
    validation_rules!(SITE;
        slot.params.len() == slot.grads.len() => format!("第 {idx} 槽参数与梯度长度不符"),
        shape_ok => format!("第 {idx} 槽矩阵形状与参数长度不符或含零维"),
    );
    Ok(())
}
fn tables(m: &[Vec<f32>], v: &[Vec<f32>]) -> Result<()> {
    validation_rules!(SITE;
        m.len() == v.len() => "一阶矩与二阶矩槽位数不符",
        m.iter().chain(v).flatten().all(|x| x.is_finite()) => "矩状态含非有限值",
    );
    Ok(())
}
fn paired(m: &[Vec<f32>], v: &[Vec<f32>]) -> Result<()> {
    tables(m, v)?;
    validation_rules!(SITE;
        m.iter().zip(v).all(|(m, v)| m.len() == v.len()) => "一阶矩与二阶矩槽内长度不符",
    );
    Ok(())
}
fn first_only(m: &[Vec<f32>], v: &[Vec<f32>]) -> Result<()> {
    validation_rules!(SITE;
        v.is_empty() => "该优化器只接受一阶动量",
        m.iter().flatten().all(|x| x.is_finite()) => "动量含非有限值",
    );
    Ok(())
}
fn no_moments(m: &[Vec<f32>], v: &[Vec<f32>]) -> Result<()> {
    validation_rules!(SITE; m.is_empty() && v.is_empty() => "SGD 无矩状态");
    Ok(())
}
fn state_slot(state: &mut Table, idx: usize, len: usize) -> &mut Vec<f32> {
    state.resize_with(state.len().max(idx + 1), Vec::new);
    let slot = &mut state[idx];
    if slot.len() != len {
        slot.resize(len, 0.0);
    }
    slot
}
fn rescale(slots: &mut [Slot<'_>], scale: f32, divide: bool) {
    if scale == 1.0 {
        return;
    }
    for grad in slots.iter_mut().flat_map(|s| s.grads.iter_mut()) {
        if divide {
            *grad /= scale;
        } else {
            *grad *= scale;
        }
    }
}

impl Sgd {
    fn advance(&mut self, slots: &mut [Slot<'_>], lr: f32, norm: f32) -> Result<()> {
        let scale = self.tuning.scale(norm);
        for slot in slots {
            let decay = decay(slot.role, self.tuning.decay);
            for (p, g) in slot.params.iter_mut().zip(slot.grads.iter()) {
                *p -= lr * (scale * g + decay * *p);
            }
        }
        Ok(())
    }
}
fn valid_sgd(_: &Sgd) -> Result<()> {
    Ok(())
}
fn restore_sgd(this: &mut Sgd, m: Table, v: Table, _: u64) -> Result<()> {
    let _ = this;
    no_moments(&m, &v)
}
fn write_sgd(_: &Sgd, _: &mut dyn FnMut(u64)) {}
stock!(
    Sgd,
    "无矩随机梯度下降。",
    {},
    valid_sgd,
    no_moments,
    restore_sgd,
    write_sgd
);

impl Adam {
    builders! { /// 设置分母保护项。
        with_eps => eps: f32,
    }
    ref_getters! { /// 一阶矩表。
        momentum => momentum: &[Vec<f32>], /// 二阶矩表。
        variance => variance: &[Vec<f32>],
    }
    getters! { /// 已提交步数。
        step_count => step: u64,
    }
    fn advance(&mut self, slots: &mut [Slot<'_>], lr: f32, norm: f32) -> Result<()> {
        let scale = self.tuning.scale(norm);
        self.step = self.step.saturating_add(1);
        let t = self.step as f32;
        let (bias1, bias2) = (1.0 - self.beta1.powf(t), 1.0 - self.beta2.powf(t));
        for (idx, slot) in slots.iter_mut().enumerate() {
            self.advance_slot(idx, slot, lr, scale, bias1, bias2);
        }
        Ok(())
    }
    fn advance_slot(
        &mut self,
        idx: usize,
        slot: &mut Slot<'_>,
        lr: f32,
        scale: f32,
        bias1: f32,
        bias2: f32,
    ) {
        let decay = decay(slot.role, self.tuning.decay);
        let (m, v) = (
            state_slot(&mut self.momentum, idx, slot.params.len()),
            state_slot(&mut self.variance, idx, slot.params.len()),
        );
        for ((p, g), (m, v)) in slot
            .params
            .iter_mut()
            .zip(slot.grads.iter())
            .zip(m.iter_mut().zip(v.iter_mut()))
        {
            let g = scale * g;
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            *p -= lr * ((*m / bias1) / ((*v / bias2).sqrt() + self.eps) + decay * *p);
        }
    }
}
fn valid_adam(this: &Adam) -> Result<()> {
    validation_rules!(SITE;
        [this.beta1, this.beta2].into_iter().all(|v| v.is_finite() && (0.0..1.0).contains(&v)) => "Adam 的 beta 必须在 [0, 1)",
        this.eps.is_finite() && this.eps > 0.0 => "Adam 的 eps 必须是有限正数",
    );
    Ok(())
}
fn restore_adam(this: &mut Adam, m: Table, v: Table, step: u64) -> Result<()> {
    paired(&m, &v)?;
    this.momentum = m;
    this.variance = v;
    this.step = step;
    Ok(())
}
fn write_adam(this: &Adam, sink: &mut dyn FnMut(u64)) {
    for value in [this.beta1, this.beta2, this.eps] {
        sink(value.to_bits().into());
    }
}
stock!(Adam, "AdamW：偏差修正双矩与解耦衰减。", {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    step: u64 = 0,
    momentum: Table = Vec::new(),
    variance: Table = Vec::new()
}, valid_adam, paired, restore_adam, write_adam);

impl Muon {
    builders! { /// 设置动量系数。
        with_momentum => momentum: f32, /// 设置归一化保护项。
        with_eps => eps: f32,
    }
    /// 关闭 Nesterov 前瞻。
    pub fn without_nesterov(mut self) -> Self {
        self.nesterov = false;
        self
    }
    ref_getters! { /// 动量表。
        momentum_table => velocity: &[Vec<f32>],
    }
    getters! { /// 已提交步数。
        step_count => step: u64,
    }
    fn advance(&mut self, slots: &mut [Slot<'_>], lr: f32, norm: f32) -> Result<()> {
        let shapes: Vec<_> = slots
            .iter()
            .map(|s| s.shape.ok_or_else(|| invalid("Muon 只接受矩阵槽")))
            .collect::<Result<_>>()?;
        let scale = self.tuning.scale(norm);
        self.step = self.step.saturating_add(1);
        for (idx, (slot, shape)) in slots.iter_mut().zip(shapes).enumerate() {
            self.advance_slot(idx, slot, shape, scale, lr);
        }
        Ok(())
    }
    fn advance_slot(
        &mut self,
        idx: usize,
        slot: &mut Slot<'_>,
        (rows, cols): (usize, usize),
        scale: f32,
        lr: f32,
    ) {
        let velocity = state_slot(&mut self.velocity, idx, rows * cols);
        for (u, &g) in velocity.iter_mut().zip(slot.grads.iter()) {
            *u = self.momentum * *u + scale * g;
        }
        let direction = if self.nesterov {
            velocity
                .iter()
                .zip(slot.grads.iter())
                .map(|(&u, &g)| scale * g + self.momentum * u)
                .collect()
        } else {
            velocity.clone()
        };
        let orthogonal = orthogonalize(&direction, rows, cols, self.eps);
        let factor = lr * 0.2 * (rows.max(cols) as f32).sqrt();
        let decay = decay(slot.role, self.tuning.decay);
        for (p, update) in slot.params.iter_mut().zip(orthogonal) {
            *p -= factor * update + lr * decay * *p;
        }
    }
}
fn valid_muon(this: &Muon) -> Result<()> {
    validation_rules!(SITE;
        this.momentum.is_finite() && (0.0..1.0).contains(&this.momentum) => "Muon 的 momentum 必须在 [0, 1)",
        this.eps.is_finite() && this.eps > 0.0 => "Muon 的 eps 必须是有限正数",
    );
    Ok(())
}
fn restore_muon(this: &mut Muon, m: Table, v: Table, step: u64) -> Result<()> {
    first_only(&m, &v)?;
    this.velocity = m;
    this.step = step;
    Ok(())
}
fn write_muon(this: &Muon, sink: &mut dyn FnMut(u64)) {
    for value in [this.momentum, this.eps] {
        sink(value.to_bits().into());
    }
    sink(this.nesterov.into());
}
stock!(Muon, "Muon：动量方向经 Newton–Schulz 近似正交化。", {
    momentum: f32 = 0.95,
    nesterov: bool = true,
    eps: f32 = 1e-7,
    step: u64 = 0,
    velocity: Table = Vec::new()
}, valid_muon, first_only, restore_muon, write_muon);

#[derive(Default)]
struct MomentState {
    m: Table,
    v: Table,
}
impl MomentState {
    fn validate<O: Optimizer>(&self, optimizer: &O) -> Result<()> {
        let v = if self.v.iter().all(Vec::is_empty) {
            &[][..]
        } else {
            &self.v
        };
        optimizer.validate_restore(&self.m, v, &[])
    }
    fn restore<O: Optimizer>(self, optimizer: &mut O, step: u64) -> Result<()> {
        let v = if self.v.iter().all(Vec::is_empty) {
            Vec::new()
        } else {
            self.v
        };
        optimizer.restore(self.m, v, step)
    }
}
#[derive(Default)]
struct Partitioned {
    matrix: MomentState,
    other: MomentState,
}
impl Partitioned {
    fn validate<M: Optimizer, N: Optimizer>(&self, matrix: &M, other: &N) -> Result<()> {
        self.matrix.validate(matrix)?;
        self.other.validate(other)
    }
}
fn split_shaped(m: Table, v: Table, shapes: Shapes<'_>) -> Result<Partitioned> {
    validation_rules!(SITE; m.len() == shapes.len() || m.is_empty() => "矩状态与形状表长度不符");
    let mut out = Partitioned::default();
    for (idx, (m, v)) in m.into_iter().zip(v).enumerate() {
        let target = if shapes[idx].is_some() {
            &mut out.matrix
        } else {
            &mut out.other
        };
        target.m.push(m);
        target.v.push(v);
    }
    Ok(out)
}

/// 矩阵槽与非矩阵槽的泛型混合优化器。
#[derive(Debug, Clone, PartialEq)]
pub struct Hybrid<M, N> {
    matrix: M,
    other: N,
    clip_norm: Option<f32>,
}
impl<M: Optimizer, N: Optimizer> Hybrid<M, N> {
    /// 构造混合路由；全局裁剪只在本层执行一次。
    pub fn new(mut matrix: M, mut other: N) -> Self {
        matrix.set_clip_norm(None);
        other.set_clip_norm(None);
        Self {
            matrix,
            other,
            clip_norm: None,
        }
    }
    ref_getters! { /// 矩阵子优化器。
        matrix => matrix: &M, /// 非矩阵子优化器。
        other => other: &N,
    }
    fn split_slots<'a, 'b>(slots: &'a mut [Slot<'b>]) -> (Vec<Slot<'a>>, Vec<Slot<'a>>) {
        let (mut matrix, mut other) = (Vec::new(), Vec::new());
        for slot in slots.iter_mut() {
            if slot.shape.is_some() {
                matrix.push(slot.reborrow());
            } else {
                other.push(slot.reborrow());
            }
        }
        (matrix, other)
    }

    fn tuning(&self) -> DecayClip {
        DecayClip {
            decay: 0.0,
            clip: self.clip_norm,
        }
    }
    fn validate_state(&self, m: &[Vec<f32>], v: &[Vec<f32>], shapes: Shapes<'_>) -> Result<()> {
        tables(m, v)?;
        for (idx, (slot, shape)) in m.iter().zip(shapes).enumerate() {
            if let Some((r, c)) = shape {
                validation_rules!(SITE; r.checked_mul(*c) == Some(slot.len()) => format!("第 {idx} 槽矩状态与形状不符"));
            }
        }
        split_shaped(m.to_vec(), v.to_vec(), shapes)?.validate(&self.matrix, &self.other)
    }
}
impl<M: Optimizer, N: Optimizer> Optimizer for Hybrid<M, N> {
    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()> {
        self.validate()?;
        let norm = check_slots(slots, lr, self.clip_norm.is_some())?;
        let scale = self.tuning().scale(norm);
        rescale(slots, scale, false);
        let (mut matrix, mut other) = Self::split_slots(slots);
        let result = self
            .matrix
            .step(&mut matrix, lr)
            .and_then(|_| self.other.step(&mut other, lr));
        drop((matrix, other));
        rescale(slots, scale, true);
        result
    }
    fn set_clip_norm(&mut self, value: Option<f32>) {
        self.clip_norm = value;
        self.matrix.set_clip_norm(None);
        self.other.set_clip_norm(None);
    }
    fn validate(&self) -> Result<()> {
        self.tuning().validate()?;
        self.matrix.validate()?;
        self.other.validate()
    }
    fn validate_restore(&self, m: &[Vec<f32>], v: &[Vec<f32>], shapes: Shapes<'_>) -> Result<()> {
        self.validate_state(m, v, shapes)
    }
    fn restore(&mut self, m: Table, v: Table, step: u64) -> Result<()> {
        no_moments(&m, &v)?;
        self.matrix.restore(Vec::new(), Vec::new(), step)?;
        self.other.restore(Vec::new(), Vec::new(), step)
    }
    fn restore_shaped(&mut self, m: Table, v: Table, step: u64, shapes: Shapes<'_>) -> Result<()> {
        self.validate_state(&m, &v, shapes)?;
        let states = split_shaped(m, v, shapes)?;
        states.matrix.restore(&mut self.matrix, step)?;
        states.other.restore(&mut self.other, step)
    }
    fn write_config(&self, sink: &mut dyn FnMut(u64)) {
        self.matrix.write_config(sink);
        self.other.write_config(sink);
        self.tuning().write(sink);
    }
}

#[cfg(test)]
#[path = "../tests/prop_optimizer.rs"]
mod prop_optimizer;
#[cfg(test)]
#[path = "../tests/optimizer_tests.rs"]
mod tests;
