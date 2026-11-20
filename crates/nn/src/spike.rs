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

use diagnostic::{Code, Error, Result};

use crate::csa::finite;
use crate::mix::{Events, Lif};

/// 报错出处：几何、动力学与脉冲取值都在入口失败，不留到膜电位推进中途。
const SITE: &str = "nn::spike";

/// `i8` 权重的最大绝对值：整数累加上界由它与输入宽度算。
const WEIGHT_MAGNITUDE: usize = 127;

/// 一层 INT8-Spiking 线性：`i8` 权重 × 三元脉冲，整数累加后按行反量化再走 LIF。
///
/// 权重行主序 `[outputs, inputs]`，第 `i` 行配第 `i` 条反量化尺度；脉冲按输入列寻址，
/// 因此「只按发放位置累加」就是跳过 `spike == 0` 的列，不需要另建稀疏矩阵。
#[derive(Debug, Clone, PartialEq)]
pub struct Spiking {
    inputs: usize,
    outputs: usize,
    weights: Vec<i8>,
    scale: Vec<f32>,
    threshold: f32,
    leak: f32,
    reset: f32,
}

impl Spiking {
    /// 构造：`threshold` 是发放门槛、`leak ∈ [0,1]` 是每步泄漏比例、`reset` 是发放后的膜电位。
    ///
    /// # Errors
    ///
    /// 通道为零、权重/尺度长度与几何不符、输入宽度大到整数累加会越过 `i32`、尺度非有限或非正、
    /// 门槛非正、泄漏出 `[0,1]`、`reset ≥ threshold`（会让下一拍立刻再发放）——都返回
    /// [`Code::Invalid`]，且不留下半截对象。
    pub fn new(
        weights: Vec<i8>,
        scale: Vec<f32>,
        inputs: usize,
        outputs: usize,
        threshold: f32,
        leak: f32,
        reset: f32,
    ) -> Result<Self> {
        let layer = Self {
            inputs,
            outputs,
            weights,
            scale,
            threshold,
            leak,
            reset,
        };
        layer.check()?;
        Ok(layer)
    }

    /// 输入宽度（每行权重的列数）。
    #[must_use]
    pub const fn inputs(&self) -> usize {
        self.inputs
    }

    /// 输出通道数（权重行数）。
    #[must_use]
    pub const fn outputs(&self) -> usize {
        self.outputs
    }

    /// 一份全新的膜电位（各通道积分从零开始）。
    ///
    /// 状态自带动力学：`step` 按状态里的那份阈值/泄漏/重置推进，所以一张膜电位不会被另一组动力学
    /// 悄悄改写——与 [`Events`] 自带步长是同一条判据的两个方向。
    #[must_use]
    pub fn membrane(&self) -> Membrane {
        Membrane {
            outputs: self.outputs,
            lifs: (0..self.outputs)
                .map(|_| Lif::new(self.threshold, self.leak, self.reset))
                .collect(),
            acc: vec![0; self.outputs],
        }
    }

    /// 稠密脉冲一步：`spikes` 逐输入给 `{−1, 0, +1}`，返回本步发放的通道数。
    ///
    /// 零脉冲只跳过累加，不改变任何通道的膜电位；非三元取值（如 `2`、`−3`）是调用方塞错了数，
    /// 一律拒绝——放行会把它当 `±1` 用，输出看起来正常却是另一组算术。
    ///
    /// 两条路径的整数乘加**完全相同**（都不为零脉冲做乘加），差别只在索引：本函数按宽度逐个看，
    /// [`Spiking::drive`] 只走发放记录。因此「事件臂省了多少乘加」是 [`Events::sparsity`] 的事，
    /// 而「省了多少遍历」才是两者墙钟之差——两个数在探针里分开报。
    ///
    /// # Errors
    ///
    /// 状态来自别的输出维度、`spikes`/`out` 长度与几何不符、脉冲不在 `{−1,0,+1}`——都返回
    /// [`Code::Invalid`]；失败的那一步不推进任何膜电位（累加缓冲只是暂存）。
    pub fn step(&self, spikes: &[i8], state: &mut Membrane, out: &mut [i8]) -> Result<usize> {
        self.check_state(state)?;
        if spikes.len() != self.inputs {
            return Err(bad(format!(
                "脉冲长度 {} 与输入宽度 {} 不符",
                spikes.len(),
                self.inputs
            )));
        }
        self.fire(spikes.iter().copied().enumerate(), state, out)
    }

    /// SSE 一步：只按 [`Events`] 里**发放**的位置累加，返回本步发放的通道数。
    ///
    /// 事件码只取符号（`int8` 的幅度是编码步长里的量化细节，权重侧不接受它），因此
    /// 「`encode` 之后再 `drive`」与「`decode` 之后取符号再 `step`」是同一个结果。
    ///
    /// # Errors
    ///
    /// 状态来自别的输出维度、事件宽度与输入宽度不符、`out` 长度不符——都返回 [`Code::Invalid`]。
    pub fn drive(
        &self,
        events: &Events<f32>,
        state: &mut Membrane,
        out: &mut [i8],
    ) -> Result<usize> {
        self.check_state(state)?;
        if events.width() != self.inputs {
            return Err(bad(format!(
                "事件宽度 {} 与输入宽度 {} 不符",
                events.width(),
                self.inputs
            )));
        }
        self.fire(
            events
                .indices()
                .iter()
                .copied()
                .zip(events.codes().iter().map(|code| code.signum())),
            state,
            out,
        )
    }

    /// 唯一一次整数累加 + 反量化 + 发放：两条公开路径都落到这里。
    fn fire(
        &self,
        events: impl Iterator<Item = (usize, i8)>,
        state: &mut Membrane,
        out: &mut [i8],
    ) -> Result<usize> {
        self.check_out(out)?;
        state.acc.fill(0);
        self.harvest(events, state)?;
        Ok(self.settle(state, out))
    }

    /// 事件流 → 整数累加：`harvest` 只管遍历，逐条的判据在 [`Spiking::pulse`]。
    fn harvest(
        &self,
        events: impl Iterator<Item = (usize, i8)>,
        state: &mut Membrane,
    ) -> Result<()> {
        for (column, spike) in events {
            self.pulse(column, spike, state)?;
        }
        Ok(())
    }

    /// 一条事件的三件判据（位置在宽度内、取值三元、零脉冲不累加）与它的累加。
    ///
    /// 零脉冲在这里退化成「什么都不做」而不是错误：稠密路径的全零位置本来就该被跳过，事件路径
    /// 不该为同一个事实多一条分支。
    fn pulse(&self, column: usize, spike: i8, state: &mut Membrane) -> Result<()> {
        if column >= self.inputs {
            return Err(bad(format!(
                "脉冲位置 {column} 越界（宽度 {}）",
                self.inputs
            )));
        }
        if spike == 0 {
            return Ok(());
        }
        if spike != 1 && spike != -1 {
            return Err(bad(format!("脉冲取值必须是 −1/0/+1，得到 {spike}")));
        }
        self.accumulate(column, spike, state);
        Ok(())
    }

    /// 输出缓冲判据：一格对应一个输出通道，短了会少算行、长了会留下没人写过的格。
    fn check_out(&self, out: &[i8]) -> Result<()> {
        if out.len() != self.outputs {
            return Err(bad(format!(
                "输出缓冲 {} 格与输出通道 {} 不符",
                out.len(),
                self.outputs
            )));
        }
        Ok(())
    }

    /// 一个输入列对整列输出的整数贡献：`acc[i] += w[i][column] · spike`。
    fn accumulate(&self, column: usize, spike: i8, state: &mut Membrane) {
        let value = i32::from(spike);
        for (row, slot) in state.acc.iter_mut().enumerate() {
            *slot += i32::from(self.weights[row * self.inputs + column]) * value;
        }
    }

    /// 反量化 + 推进 LIF + 统计发放：发放判据只由 [`Lif`] 定义。
    fn settle(&self, state: &mut Membrane, out: &mut [i8]) -> usize {
        let mut fired = 0;
        for (row, spike) in out.iter_mut().enumerate() {
            let potential = state.acc[row] as f32 * self.scale[row];
            let value = if state.lifs[row].step(potential) == 1.0 {
                fired += 1;
                1
            } else {
                0
            };
            *spike = value;
        }
        fired
    }

    /// 参数判据：几何、尺度、动力学各查一处，互不代答。
    fn check(&self) -> Result<()> {
        self.check_geometry()?;
        check_scale(&self.scale)?;
        check_dynamics(self.threshold, self.leak, self.reset)
    }

    /// 几何判据：先判宽度上界，再判两条长度——上界不过时连权重都不该读。
    fn check_geometry(&self) -> Result<()> {
        self.check_width()?;
        self.check_lengths()
    }

    /// 宽度判据：通道非零，且输入宽到会让整数累加越过 `i32` 时直接拒绝。
    fn check_width(&self) -> Result<()> {
        if self.inputs == 0 || self.outputs == 0 {
            return Err(bad("输入与输出通道都必须为正"));
        }
        if self.inputs > i32::MAX as usize / WEIGHT_MAGNITUDE {
            return Err(bad(format!(
                "输入宽度 {} 过大：整数累加会越过 i32",
                self.inputs
            )));
        }
        Ok(())
    }

    /// 长度判据：权重按 `[outputs, inputs]` 行主序，尺度一条对一行。
    fn check_lengths(&self) -> Result<()> {
        let width = self
            .inputs
            .checked_mul(self.outputs)
            .ok_or_else(|| bad("权重形状溢出"))?;
        if self.weights.len() != width {
            return Err(bad(format!(
                "权重长度 {} 与 {}×{} 不符",
                self.weights.len(),
                self.outputs,
                self.inputs
            )));
        }
        if self.scale.len() != self.outputs {
            return Err(bad(format!(
                "反量化尺度 {} 条与输出通道 {} 不符",
                self.scale.len(),
                self.outputs
            )));
        }
        Ok(())
    }

    /// 状态判据：膜电位必须属于同宽度的层。
    fn check_state(&self, state: &Membrane) -> Result<()> {
        if state.outputs != self.outputs {
            return Err(bad(format!(
                "膜电位属于 {} 个通道的层，本层是 {}",
                state.outputs, self.outputs
            )));
        }
        Ok(())
    }
}

/// 一层的状态：每通道一条 [`Lif`] 与一段复用的整数累加缓冲。
///
/// 缓冲在状态里而不是每步新分配：`drive` 的稀疏性体现在「少做乘加」，不该被一次 `vec!` 抹平。
#[derive(Debug, Clone, PartialEq)]
pub struct Membrane {
    outputs: usize,
    lifs: Vec<Lif<f32>>,
    acc: Vec<i32>,
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
}

/// 尺度判据：反量化尺度必须是有限正数（零尺度会把整行压成零，负尺度翻转符号）。
fn check_scale(scale: &[f32]) -> Result<()> {
    match scale
        .iter()
        .find(|value| !finite(**value) || **value <= 0.0)
    {
        Some(value) => Err(bad(format!("反量化尺度必须是有限正数，得到 {value:?}"))),
        None => Ok(()),
    }
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
        return Err(bad(format!("发放门槛必须有限为正，得到 {threshold:?}")));
    }
    Ok(())
}

/// 泄漏判据：落在 `[0,1]`——`0` 是纯 IF，`1` 是每步清零。
fn check_leak(leak: f32) -> Result<()> {
    if !finite(leak) || !(..=1.0).contains(&leak) || leak < 0.0 {
        return Err(bad(format!("泄漏比例必须落在 [0,1]，得到 {leak:?}")));
    }
    Ok(())
}

/// 重置判据：有限且低于门槛，否则下一拍会立刻再发放一次。
fn check_reset(threshold: f32, reset: f32) -> Result<()> {
    if !finite(reset) || reset >= threshold {
        return Err(bad(format!(
            "重置值必须有限且低于门槛 {threshold}，得到 {reset:?}"
        )));
    }
    Ok(())
}

/// 入口失败。
fn bad(why: impl Into<String>) -> Error {
    Error::new(Code::Invalid, why).with(SITE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mix::Encoder;

    /// 门槛调到不可能达到、无泄漏：此时膜电位就是这一步的整数累加。
    fn quiet(weights: Vec<i8>, scale: Vec<f32>, inputs: usize, outputs: usize) -> Spiking {
        Spiking::new(weights, scale, inputs, outputs, 1.0e6, 0.0, 0.0).expect("几何合法")
    }

    #[test]
    fn integer_accumulation_is_exact_against_the_f32_dot() {
        let layer = quiet(vec![1, -2, 3, 4], vec![1.0, 1.0], 2, 2);
        let mut state = layer.membrane();
        let mut out = [0i8; 2];
        let fired = layer
            .step(&[1, -1], &mut state, &mut out)
            .expect("合法脉冲");
        assert_eq!(fired, 0);
        // 第 0 行：1·1 + (−2)·(−1) = 3；第 1 行：3·1 + 4·(−1) = −1。
        assert_eq!(state.potential()[0].to_bits(), 3.0f32.to_bits());
        assert_eq!(state.potential()[1].to_bits(), (-1.0f32).to_bits());
        assert_eq!(out, [0, 0]);
    }

    #[test]
    fn no_pulses_leave_every_membrane_untouched() {
        let layer = quiet(vec![5, 5], vec![2.0], 2, 1);
        let mut state = layer.membrane();
        let mut out = [9i8; 1];
        for _ in 0..3 {
            let fired = layer
                .step(&[0, 0], &mut state, &mut out)
                .expect("零脉冲合法");
            assert_eq!(fired, 0);
            assert_eq!(out, [0]);
        }
        assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn a_potential_over_the_threshold_fires_and_resets() {
        let layer = Spiking::new(vec![1], vec![1.0], 1, 1, 1.0, 0.0, -0.5).expect("几何合法");
        let mut state = layer.membrane();
        let mut out = [0i8; 1];
        assert_eq!(layer.step(&[1], &mut state, &mut out).expect("合法"), 1);
        assert_eq!(out, [1]);
        assert_eq!(state.potential()[0].to_bits(), (-0.5f32).to_bits());
    }

    #[test]
    fn leak_shapes_the_membrane_between_steps() {
        let layer = Spiking::new(vec![1], vec![1.0], 1, 1, 4.0, 0.5, 0.0).expect("几何合法");
        let mut state = layer.membrane();
        let mut out = [0i8; 1];
        layer.step(&[1], &mut state, &mut out).expect("合法");
        // 第二步：1·0.5 + 1 = 1.5（无泄漏时会是 2）。
        layer.step(&[1], &mut state, &mut out).expect("合法");
        assert_eq!(state.potential()[0].to_bits(), 1.5f32.to_bits());
    }

    #[test]
    fn the_event_path_and_the_dense_path_agree_bitwise() {
        let layer = quiet(vec![2, -1, 4, 3, -5, 1], vec![0.5, 2.0], 3, 2);
        let events = Encoder::new(0.5f32, 1)
            .expect("参数合法")
            .encode(&[0.0, 0.9, -1.4])
            .expect("合法输入");
        assert_eq!(events.indices(), &[1, 2]);
        // 事件码的符号：0.9/0.5 → +1，−1.4/0.5 → −2 → 符号 −1。
        let mut driven = layer.membrane();
        let mut sparse = [0i8; 2];
        let sparse_fired = layer
            .drive(&events, &mut driven, &mut sparse)
            .expect("宽度一致");

        let mut dense_state = layer.membrane();
        let mut dense = [0i8; 2];
        let dense_fired = layer
            .step(&[0, 1, -1], &mut dense_state, &mut dense)
            .expect("合法脉冲");

        assert_eq!(sparse_fired, dense_fired);
        assert_eq!(sparse, dense);
        assert_eq!(driven.potential(), dense_state.potential());
    }

    #[test]
    fn the_count_matches_the_emitted_spikes() {
        let layer = Spiking::new(vec![1, 1, -1], vec![1.0, 1.0, 1.0], 1, 3, 1.0, 0.0, 0.0)
            .expect("几何合法");
        let mut state = layer.membrane();
        let mut out = [0i8; 3];
        let fired = layer.step(&[-1], &mut state, &mut out).expect("合法脉冲");
        assert_eq!(out, [0, 0, 1]);
        assert_eq!(fired, 1);
        assert_eq!(fired, out.iter().filter(|&&s| s == 1).count());
    }

    #[test]
    fn events_of_another_width_are_rejected() {
        let layer = quiet(vec![1, 1], vec![1.0], 2, 1);
        let events = Encoder::new(1.0f32, 1)
            .expect("参数合法")
            .encode(&[1.0, 1.0, 1.0])
            .expect("合法输入");
        let mut state = layer.membrane();
        let mut out = [0i8; 1];
        assert!(layer.drive(&events, &mut state, &mut out).is_err());
        assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn geometry_failures_are_rejected_before_any_object_exists() {
        assert!(Spiking::new(vec![1], vec![1.0], 0, 1, 1.0, 0.0, 0.0).is_err());
        assert!(Spiking::new(vec![1], vec![1.0], 1, 0, 1.0, 0.0, 0.0).is_err());
        assert!(Spiking::new(vec![1], vec![1.0], 2, 1, 1.0, 0.0, 0.0).is_err());
        assert!(Spiking::new(vec![1, 1], vec![1.0, 2.0], 1, 1, 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn an_input_width_that_would_overflow_the_accumulator_is_rejected() {
        let wide = i32::MAX as usize / WEIGHT_MAGNITUDE + 1;
        assert!(Spiking::new(Vec::new(), vec![1.0], wide, 1, 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn dynamics_outside_the_lif_range_are_rejected() {
        let base = (vec![1i8], vec![1.0f32], 1usize, 1usize);
        let build = |threshold, leak, reset| {
            Spiking::new(base.0.clone(), base.1.clone(), 1, 1, threshold, leak, reset)
        };
        assert!(build(0.0, 0.0, 0.0).is_err());
        assert!(build(f32::NAN, 0.0, 0.0).is_err());
        assert!(build(1.0, 1.5, 0.0).is_err());
        assert!(build(1.0, 0.0, 1.0).is_err());
        assert!(build(1.0, 0.0, f32::INFINITY).is_err());
        assert!(Spiking::new(vec![1], vec![0.0], 1, 1, 1.0, 0.0, 0.0).is_err());
        assert!(Spiking::new(vec![1], vec![f32::NAN], 1, 1, 1.0, 0.0, 0.0).is_err());
        assert!(build(1.0, 0.0, -0.5).is_ok());
    }

    #[test]
    fn dense_pulses_outside_the_ternary_set_are_rejected() {
        let layer = quiet(vec![1, 1], vec![1.0], 2, 1);
        let mut state = layer.membrane();
        let mut out = [0i8; 1];
        assert!(layer.step(&[2, 0], &mut state, &mut out).is_err());
        assert!(layer.step(&[0, -3], &mut state, &mut out).is_err());
        assert!(layer.step(&[1], &mut state, &mut out).is_err());
        assert!(layer.step(&[1, 0], &mut state, &mut [0i8; 2]).is_err());
        assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
    }

    /// E14 稀疏度与「事件臂 vs 稠密臂」的等效乘加对照（`#[ignore]` 实测：
    /// `cargo test -p llm-nn --lib -- spike_arm_costs --release --ignored --nocapture`）。
    ///
    /// 几何冻结在 1024 → 1024，权重与激活都由固定种子的 xorshift 生成，`Encoder` 的码位门槛决定
    /// 发放比例——于是「事件臂做了多少次整数累加」与「稠密臂做了多少次」可由 `active` 直接比出来，
    /// 不必假设稀疏度。两条臂在同一份权重上用同一段脉冲，墙钟差才是「少做多少个乘加」的实测。
    ///
    /// 注意两份耗时里都含相同的一份固定开销（`outputs` 次反量化 + LIF 推进），因此加速比**低于**
    /// 乘加比；这条只报数，判决仍归 `tool::card` 的协议卡。
    #[test]
    #[ignore = "E14 事件臂实测：cargo test -p llm-nn --lib -- spike_arm_costs --release --ignored --nocapture"]
    fn spike_arm_costs() {
        use std::hint::black_box;
        use std::time::Instant;

        let width = 1024usize;
        let mut seed = 0x5eed_f00d_1234_5678u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let weights: Vec<i8> = (0..width * width)
            .map(|_| ((next() % 255) as i16 - 128) as i8)
            .collect();
        let values: Vec<f32> = (0..width)
            .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
            .collect();
        let events = Encoder::new(0.25f32, 3)
            .expect("参数合法")
            .encode(&values)
            .expect("合法输入");
        let dense: Vec<i8> = events
            .decode()
            .iter()
            .map(|value| value.signum() as i8)
            .collect();

        let layer = quiet(weights, vec![1.0; width], width, width);
        let mut state = layer.membrane();
        let mut out = vec![0i8; width];
        let reps = 100usize;

        let start = Instant::now();
        for _ in 0..reps {
            black_box(layer.step(&dense, &mut state, &mut out).expect("合法脉冲"));
        }
        let dense_ns = start.elapsed().as_secs_f64() / reps as f64;

        let start = Instant::now();
        for _ in 0..reps {
            black_box(
                layer
                    .drive(&events, &mut state, &mut out)
                    .expect("宽度一致"),
            );
        }
        let event_ns = start.elapsed().as_secs_f64() / reps as f64;

        // 乘加按「一个输出行要对多少个输入位置累加」计：密集实现是 `inputs`，事件实现是
        // `active`，两者都再乘 `outputs`——比因此等于 `1 − S`，这正是 E14 的定义。
        let active = events.active();
        let dense_macs = width * width;
        let event_macs = active * width;
        println!(
            "E14 稀疏度 S={:.4}（active={active}/{width}） 乘加 dense={dense_macs} event={event_macs} 比={:.4}",
            events.sparsity(),
            event_macs as f64 / dense_macs as f64
        );
        println!(
            "E13 索引开销对照（同一份整数算术）dense={:.1} ms  event={:.1} ms  墙钟比={:.2}×",
            dense_ns * 1e3,
            event_ns * 1e3,
            dense_ns / event_ns
        );
    }

    #[test]
    fn a_membrane_from_another_layer_is_rejected() {
        let wide = quiet(vec![1, 1, 1, 1], vec![1.0, 1.0], 2, 2);
        let narrow = quiet(vec![1, 1], vec![1.0], 2, 1);
        let mut state = wide.membrane();
        let mut out = [0i8; 1];
        assert!(narrow.step(&[1, 1], &mut state, &mut out).is_err());
        assert_eq!(state.potential()[0].to_bits(), 0.0f32.to_bits());
    }
}
