//! SpikingBrain 2.0 的两条执行臂：SSE 稀疏事件驱动与 INT8-Spiking 量化脉冲线性层。
//!
//! 这一款此前只有神经元原语（[`Lif`]）与事件编码（[`Encoder`]），缺的是「一层脉冲线性怎么算」：
//! 权重是 int8、激活是三元脉冲 `{−1, 0, +1}`、累加走整数、反量化之后才进入 LIF。本条把它补上，
//! 并让两条路径共用**同一条算术**——[`Spiking::step`] 吃稠密三元脉冲，[`Spiking::drive`] 吃
//! [`Events`]（SSE：只有发放位置占记录），后者取事件码的符号再走前者的同一次整数累加。
//! `i8 × {−1,0,+1}` 是整数加法，因此两条路径逐位相同，差别只在少做多少个乘加——这正是 E14
//! 稀疏度所度量的量。
//!
//! # 三个决定
//!
//! **参数与状态分开**：[`Spiking`] 只持参数（权重、逐行反量化尺度、LIF 动力学），[`Membrane`]
//! 只持状态（每通道膜电位 + 整数累加缓冲）。状态只能由 [`Spiking::membrane`] 构造，跨层误用
//! （输出通道数不同）在入口被拒，与 `kda::Memory`、`mamba::Carry` 同一口径。
//!
//! **整数累加不溢出是可证的**：`i8` 权重绝对值不超过 127，构造期就要求
//! `inputs ≤ i32::MAX / 127`，于是每行最多累加 `127 · inputs` 次、恒在 `i32` 内，不必逐项做
//! 饱和判断（那会让「结果对不对」取决于宽度）。
//!
//! **本模块只给脉冲臂**：STE 代理梯度训练不在里面（`train` 侧没有脉冲反传），能效（E13）也不
//! 由它出数——那需要目标设备上的实测卡。这里给的是「怎么算」以及可测的不变量。

use diagnostic::Result;

use crate::csa::finite;
use crate::mix::{Events, Lif};

/// 报错出处：几何、动力学与脉冲取值都在入口失败，不留到膜电位推进中途。
const SITE: &str = "algorithm::spike";

/// `i8` 权重的最大绝对值：整数累加上界由它与输入宽度算。
const WEIGHT_MAGNITUDE: usize = 127;

/// 三元脉冲判据：`−1` / `0` / `+1`。稠密与事件两条路径、以及反传，都只认这三个值。
fn ternary(spike: i8) -> bool {
    matches!(spike, -1..=1)
}

/// 输出侧梯度判据：长度与输出通道一致，取值有限。
fn check_gradient(d_out: &[f32], outputs: usize) -> Result<()> {
    if d_out.len() != outputs {
        return err!("输出梯度 {} 格与输出通道 {} 不符", d_out.len(), outputs);
    }
    match d_out.iter().find(|value| !finite(**value)) {
        Some(value) => err!("输出梯度必须是有限值，得到 {value:?}"),
        None => Ok(()),
    }
}

/// 学习率判据：有限为正（零学习率等于「什么都不学」，负学习率等于「朝梯度方向上升」）。
/// 浮点权重判据：`NaN` 经 `round().clamp()` 会落成某个看着正常的 `i8`，只能在量化之前挡住。
fn check_weights(weights: &[f32]) -> Result<()> {
    match weights.iter().find(|weight| !finite(**weight)) {
        Some(weight) => err!("浮点主权重必须是有限值，得到 {weight:?}"),
        None => Ok(()),
    }
}

fn check_learning_rate(lr: f32) -> Result<()> {
    if !finite(lr) || lr <= 0.0 {
        return err!("学习率必须有限为正，得到 {lr:?}");
    }
    Ok(())
}

/// 尺度的合格条件：有限且为正——零尺度把整行压成零，负尺度翻转符号。
///
/// 训练里「读进来的尺度」与「走一步之后的尺度」用的是同一条条件，只是报错时说的话不同，
/// 故条件单独一处，消息各自写。
fn bad_scale(scale: &[f32]) -> Option<&f32> {
    scale
        .iter()
        .find(|value| !finite(**value) || **value <= 0.0)
}

/// 尺度判据：反量化尺度必须是有限正数（零尺度会把整行压成零，负尺度翻转符号）。
fn check_scale(scale: &[f32]) -> Result<()> {
    match bad_scale(scale) {
        Some(value) => err!("反量化尺度必须是有限正数，得到 {value:?}"),
        None => Ok(()),
    }
}

/// 量化格点判据：权重按 `[outputs, inputs]` 行主序占 `inputs × outputs` 格，尺度一条对一行。
///
/// 元素类型不参与判断，只数格——浮点主权重（`Master`）与定点权重（`Spiking`）用的是同一套
/// 量化布局，所以这里是唯一一份长度核对。少一条尺度就有输出通道没有尺度可乘，多一条就是尺度
/// 覆盖了不存在的通道；两者都只能在这里挡住，因为乘的时候按 `[row]` 取，越界不会报错。
fn check_lattice<T>(weights: &[T], scale: &[f32], inputs: usize, outputs: usize) -> Result<()> {
    let want = inputs
        .checked_mul(outputs)
        .ok_or_else(|| invalid!("输入宽度与输出通道相乘溢出"))?;
    if weights.len() != want {
        return err!("权重 {} 格与 {inputs}×{outputs} 不符", weights.len());
    }
    if scale.len() != outputs {
        return err!("反量化尺度 {} 条与输出通道 {} 不符", scale.len(), outputs);
    }
    Ok(())
}

/// 动力学判据：门槛有限正、泄漏落在 `[0,1]`、重置有限且低于门槛。
fn check_dynamics(threshold: f32, leak: f32, reset: f32) -> Result<()> {
    check_threshold(threshold)?;
    check_leak(leak)?;
    check_reset(threshold, reset)
}

/// 门槛判据：有限且为正（零门槛会让每一步都发放）。
fn check_threshold(threshold: f32) -> Result<()> {
    if !finite(threshold) || threshold <= 0.0 {
        return err!("发放门槛必须有限为正，得到 {threshold:?}");
    }
    Ok(())
}

/// 泄漏判据：落在 `[0,1]`——`0` 是纯 IF，`1` 是每步清零。
fn check_leak(leak: f32) -> Result<()> {
    if !finite(leak) || !(..=1.0).contains(&leak) || leak < 0.0 {
        return err!("泄漏比例必须落在 [0,1]，得到 {leak:?}");
    }
    Ok(())
}

/// 重置判据：有限且低于门槛，否则下一拍会立刻再发放一次。
fn check_reset(threshold: f32, reset: f32) -> Result<()> {
    if !finite(reset) || reset >= threshold {
        return err!("重置值必须有限且低于门槛 {threshold}，得到 {reset:?}");
    }
    Ok(())
}

#[path = "cell.rs"]
mod cell;

/// 浮点主权重（QAT 口径）：训练留浮点，前向只看到它的 `i8` 量化视图。
#[derive(Clone, Debug, PartialEq)]
pub struct Master {
    pub(in crate::spike) weights: Vec<f32>,
    pub(in crate::spike) scale: Vec<f32>,
    pub(in crate::spike) inputs: usize,
    pub(in crate::spike) outputs: usize,
    pub(in crate::spike) threshold: f32,
    pub(in crate::spike) leak: f32,
    pub(in crate::spike) reset: f32,
}

impl Master {
    /// 从浮点权重构造：几何与动力学都按 [`Spiking::new`] 同一份判据在入口挡住。
    ///
    /// 权重必须是有限值——`NaN` 在量化时会被夹成一个「看起来正常」的 `i8`，事后从视图上完全
    /// 看不出它原本是坏的。
    ///
    /// # Errors
    ///
    /// 长度与 `inputs·outputs` / `outputs` 不符、权重或尺度非有限、尺度非正、动力学越界，
    /// 一律 `Code::Invalid`。
    pub fn new(
        weights: Vec<f32>,
        scale: Vec<f32>,
        inputs: usize,
        outputs: usize,
        threshold: f32,
        leak: f32,
        reset: f32,
    ) -> Result<Self> {
        let master = Self {
            weights,
            scale,
            inputs,
            outputs,
            threshold,
            leak,
            reset,
        };
        master.check_master()?;
        Ok(master)
    }

    /// 浮点主权重，行主序 `[outputs, inputs]`。
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// 逐行反量化尺度，一条对一行。
    #[must_use]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// 量化成前向层：四舍五入后夹到 `[−127, 127]`。
    ///
    /// 对称量化、零点不入码——与 [`Spiking::new`] 的 `|w| ≤ 127` 前提一致，所以「夹到 127」只会
    /// 发生在训练把某个权重推出量化范围的时候，而不是正常路径。
    ///
    /// # Errors
    ///
    /// 量化后的几何或动力学不合（[`Master`] 字段私有，正常不会发生）时透传 [`Spiking::new`]
    /// 的错误。
    pub fn view(&self) -> Result<Spiking> {
        let weights = self
            .weights
            .iter()
            .map(|weight| weight.round().clamp(-127.0, 127.0) as i8)
            .collect();
        Spiking::new(
            weights,
            self.scale.clone(),
            self.inputs,
            self.outputs,
            self.threshold,
            self.leak,
            self.reset,
        )
    }

    /// 一步 SGD：`w ← w − lr·g_w`、`scale ← scale − lr·g_scale`。
    ///
    /// 只吃权重与尺度两条梯度：电流梯度是**给上一层**的，不落在本层参数上。
    ///
    /// # Errors
    ///
    /// 梯度布局与本层不符、学习率非有限或非正、更新把某条尺度推成非正（说明步长把量化区间
    /// 走穿了）——都返回 `Code::Invalid`，且失败时一格参数都不写。
    pub fn descend(&mut self, grad: &Gradients, lr: f32) -> Result<()> {
        grad.layout(self.weights.len(), self.outputs)?;
        check_learning_rate(lr)?;
        // 尺度必须在写权重之前全部验完：留下「权重已降、尺度是负的」这种半截状态，要等到下一次
        // `view` 才报错，而那时离出错点已经很远了。
        let scale = self.next_scale(grad, lr)?;
        for (weight, slope) in self.weights.iter_mut().zip(grad.weights.iter()) {
            *weight -= lr * slope;
        }
        self.scale = scale;
        Ok(())
    }

    /// 走一步之后的尺度：先算齐、再验「有限为正」，都不合就一格参数也不写。
    fn next_scale(&self, grad: &Gradients, lr: f32) -> Result<Vec<f32>> {
        let scale: Vec<f32> = self
            .scale
            .iter()
            .zip(grad.scale.iter())
            .map(|(value, slope)| value - lr * slope)
            .collect();
        match bad_scale(&scale) {
            Some(value) => err!(
                "更新后的反量化尺度必须有限为正，得到 {value:?}（学习率 {lr} 把量化区间走穿了）"
            ),
            None => Ok(scale),
        }
    }

    /// 构造判据：两条长度、权重的有限性、尺度的有限正性、动力学范围。
    pub(in crate::spike) fn check_master(&self) -> Result<()> {
        check_lattice(&self.weights, &self.scale, self.inputs, self.outputs)?;
        self.check_values()
    }

    /// 取值判据：权重有限、尺度有限为正、动力学在范围内。
    fn check_values(&self) -> Result<()> {
        check_weights(&self.weights)?;
        check_scale(&self.scale)?;
        check_dynamics(self.threshold, self.leak, self.reset)
    }
}

/// 一层的状态：每通道一条 [`Lif`]、一段复用的整数累加缓冲，以及本步发放前的膜电位。
#[derive(Debug, Clone, PartialEq)]
pub struct Membrane {
    pub(in crate::spike) outputs: usize,
    pub(in crate::spike) lifs: Vec<Lif<f32>>,
    pub(in crate::spike) acc: Vec<i32>,
    pub(in crate::spike) pre: Vec<f32>,
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

/// 代理梯度（surrogate）：只定义「发放对膜电位」的导数，不参与前向。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surrogate {
    /// 矩形窗：`|v − θ| < width` 内恒为 `1/(2·width)`。
    Rect {
        /// 窗的半宽。
        width: f32,
    },
    /// 三角形：窗内从阈值处的 `1/width` 线性衰减到边界上的零。
    Triangle {
        /// 窗的半宽。
        width: f32,
    },
}

impl Surrogate {
    /// 窗的半宽。
    #[must_use]
    pub fn width(&self) -> f32 {
        match self {
            Self::Rect { width } | Self::Triangle { width } => *width,
        }
    }

    /// 发放对膜电位的导数：`potential` 与 `threshold` 的距离落在窗内才有值。
    #[must_use]
    pub fn derivative(&self, potential: f32, threshold: f32) -> f32 {
        let offset = (potential - threshold).abs();
        // 非有限偏移连 `NaN` 一起挡在窗外：窗宽由构造期校验为有限正数，而 `NaN` 与任何值的
        // `>=` 都为假，单写比较会让它走进窗内拿到一个假的导数。
        if !offset.is_finite() || offset >= self.width() {
            return 0.0;
        }
        match self {
            Self::Rect { width } => 1.0 / (2.0 * width),
            Self::Triangle { width } => (width - offset) / (width * width),
        }
    }

    /// 参数判据：半宽必须有限为正（零宽等于没有梯度，负宽会让窗内外颠倒）。
    pub(in crate::spike) fn check_window(&self) -> Result<()> {
        let width = self.width();
        if !finite(width) || width <= 0.0 {
            return err!("代理梯度的窗宽必须有限为正，得到 {width:?}");
        }
        Ok(())
    }
}

/// 一层脉冲线性的梯度，布局与本层参数逐项对齐。
#[derive(Debug, Clone, PartialEq)]
pub struct Gradients {
    pub(in crate::spike) weights: Vec<f32>,
    pub(in crate::spike) scale: Vec<f32>,
    pub(in crate::spike) current: Vec<f32>,
}

impl Gradients {
    /// 对权重的梯度，行主序 `[outputs, inputs]`（与本层的权重同一布局）。
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// 对逐行反量化尺度的梯度，一条对一行。
    #[must_use]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// 对本层输出**电流**的梯度，可直接交给上一层的反传。
    #[must_use]
    pub fn current(&self) -> &[f32] {
        &self.current
    }

    /// 缓冲判据：三条梯度必须与本层参数同形，长了短了都会静默写错行。
    pub(in crate::spike) fn check_shapes(&self, layer: &Spiking) -> Result<()> {
        self.layout(layer.weights.len(), layer.outputs)
    }

    /// 反传入口判据：缓冲与本层同形，且输出梯度与通道数同长。
    pub(in crate::spike) fn check_for(&self, layer: &Spiking, d_out: &[f32]) -> Result<()> {
        self.check_shapes(layer)?;
        check_gradient(d_out, layer.outputs)
    }

    /// 判据本体：只按「权重格数 + 输出通道数」核对，好让量化前的 [`Master`] 与量化后的
    /// [`Spiking`] 共用同一份判据。
    pub(in crate::spike) fn layout(&self, weights: usize, outputs: usize) -> Result<()> {
        let shape = [self.weights.len(), self.scale.len(), self.current.len()];
        let want = [weights, outputs, outputs];
        if shape != want {
            return err!("梯度缓冲 {shape:?} 与本层 {want:?} 不符");
        }
        Ok(())
    }
}

pub use cell::Spiking;

#[cfg(test)]
#[path = "../tests/spike_tests.rs"]
mod spike_tests;
