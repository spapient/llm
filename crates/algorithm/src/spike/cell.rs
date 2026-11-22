//! 职责：INT8 脉冲线性层：i8 权重乘三元脉冲，整数累加后反量化走 LIF。
use super::*;

/// 一层 INT8-Spiking 线性：`i8` 权重 × 三元脉冲，整数累加后按行反量化再走 LIF。
#[derive(Debug, Clone, PartialEq)]
pub struct Spiking {
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    pub(super) weights: Vec<i8>,
    pub(super) scale: Vec<f32>,
    pub(super) threshold: f32,
    pub(super) leak: f32,
    pub(super) reset: f32,
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
    #[must_use]
    pub fn membrane(&self) -> Membrane {
        Membrane {
            outputs: self.outputs,
            lifs: (0..self.outputs)
                .map(|_| Lif::new(self.threshold, self.leak, self.reset))
                .collect(),
            acc: vec![0; self.outputs],
            pre: vec![0.0; self.outputs],
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
                .idxs()
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
    fn pulse(&self, column: usize, spike: i8, state: &mut Membrane) -> Result<()> {
        if column >= self.inputs {
            return Err(bad(format!(
                "脉冲位置 {column} 越界（宽度 {}）",
                self.inputs
            )));
        }
        if !ternary(spike) {
            return Err(bad(format!("脉冲取值必须是 −1/0/+1，得到 {spike}")));
        }
        if spike == 0 {
            return Ok(());
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
            let current = state.acc[row] as f32 * self.scale[row];
            let potential = state.lifs[row].peek(current);
            state.pre[row] = potential;
            let value = if state.lifs[row].commit(potential) == 1.0 {
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
    pub(super) fn check(&self) -> Result<()> {
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

    /// 一份与本层同布局的零化梯度缓冲（反传每次写满每一格，所以只需在这里分配一次）。
    #[must_use]
    pub fn gradients(&self) -> Gradients {
        Gradients {
            weights: vec![0.0; self.weights.len()],
            scale: vec![0.0; self.outputs],
            current: vec![0.0; self.outputs],
        }
    }

    /// 反向一步：吃本步的输入脉冲、当前状态与输出侧梯度，写满 `grad` 的三条梯度。
    ///
    /// `state` 必须是**同一步** [`Spiking::step`] / [`Spiking::drive`] 之后的那份状态：整数累加与
    /// 发放前膜电位都只在那里留有记录，跨步调用会把上一拍的累加当成这一拍。
    ///
    /// 链式三项：`∂L/∂v = ∂L/∂s · surrogate(v, θ)`；膜电位在发放前只由本层电流加进来一次，所以
    /// `∂L/∂current` 就是 `∂L/∂v`；再往前是两条线性路径——`∂L/∂scale[i] = ∂L/∂v · acc[i]`、
    /// `∂L/∂w[i][j] = ∂L/∂v · scale[i] · spike[j]`。发放本身是阶跃，它的导数只由 surrogate 给，
    /// 前向不因此改变。
    ///
    /// 权重梯度落在 `f32` 上而不是 `i8`：训练留一份浮点主权重，`i8` 只是它的量化视图（QAT 口径），
    /// 所以「对权重的梯度」本来就该由浮点侧接。
    ///
    /// # Errors
    ///
    /// 状态来自别的输出维度、脉冲长度/取值不合、输出梯度长度不符或非有限、梯度缓冲布局不符、
    /// 代理梯度窗宽非正——都返回 [`Code::Invalid`]，且失败时 `grad` 一格都不写。
    pub fn backward(
        &self,
        spikes: &[i8],
        state: &Membrane,
        surrogate: Surrogate,
        d_out: &[f32],
        grad: &mut Gradients,
    ) -> Result<()> {
        self.check_backward_inputs(spikes, state, surrogate, d_out, grad)?;
        for row in 0..self.outputs {
            self.fill_row(row, state, spikes, surrogate, d_out, grad);
        }
        Ok(())
    }

    /// 反传入口判据：状态、窗、脉冲、输出梯度与缓冲，各自只答一个问题，全部在写入之前。
    fn check_backward_inputs(
        &self,
        spikes: &[i8],
        state: &Membrane,
        surrogate: Surrogate,
        d_out: &[f32],
        grad: &Gradients,
    ) -> Result<()> {
        self.check_state(state)?;
        surrogate.check()?;
        self.check_spikes(spikes)?;
        grad.check_for(self, d_out)
    }

    /// 一行的三条梯度：电流交给上一层，尺度与权重只由该行自己这一拍的发放决定。
    fn fill_row(
        &self,
        row: usize,
        state: &Membrane,
        spikes: &[i8],
        surrogate: Surrogate,
        d_out: &[f32],
        grad: &mut Gradients,
    ) {
        // 发放对电位的敏感度：窗外的通道拿零，等于「这一拍它没参与决策」。
        let sensitivity = d_out[row] * surrogate.derivative(state.pre[row], self.threshold);
        grad.current[row] = sensitivity;
        grad.scale[row] = sensitivity * state.acc[row] as f32;
        let upstream = sensitivity * self.scale[row];
        let codes = &mut grad.weights[row * self.inputs..(row + 1) * self.inputs];
        for (slot, spike) in codes.iter_mut().zip(spikes) {
            *slot = upstream * f32::from(*spike);
        }
    }

    /// 脉冲判据：长度与输入宽度一致、取值三元——与 [`Spiking::pulse`] 用的是同一个 [`ternary`]。
    fn check_spikes(&self, spikes: &[i8]) -> Result<()> {
        if spikes.len() != self.inputs {
            return Err(bad(format!(
                "脉冲长度 {} 与输入宽度 {} 不符",
                spikes.len(),
                self.inputs
            )));
        }
        match spikes.iter().find(|spike| !ternary(**spike)) {
            Some(spike) => Err(bad(format!("脉冲取值必须是 −1/0/+1，得到 {spike}"))),
            None => Ok(()),
        }
    }
}
