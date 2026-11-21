//! 职责：浮点主权重（QAT）：训练留 f32，前向只暴露 i8 量化视图。
use super::*;

/// 浮点主权重（QAT 口径）：训练留浮点，前向只看到它的 `i8` 量化视图。
#[derive(Clone, Debug, PartialEq)]
pub struct Master {
    pub(super) weights: Vec<f32>,
    pub(super) scale: Vec<f32>,
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    pub(super) threshold: f32,
    pub(super) leak: f32,
    pub(super) reset: f32,
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
    /// 一律 [`Code::Invalid`]。
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
        master.check()?;
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
    /// 走穿了）——都返回 [`Code::Invalid`]，且失败时一格参数都不写。
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
        match scale
            .iter()
            .find(|value| !finite(**value) || **value <= 0.0)
        {
            Some(value) => Err(bad(format!(
                "更新后的反量化尺度必须有限为正，得到 {value:?}（学习率 {lr} 把量化区间走穿了）"
            ))),
            None => Ok(scale),
        }
    }

    /// 构造判据：两条长度、权重的有限性、尺度的有限正性、动力学范围。
    pub(super) fn check(&self) -> Result<()> {
        self.check_shape()?;
        self.check_values()
    }

    /// 形状判据：相乘不溢出，两条长度都对。
    fn check_shape(&self) -> Result<()> {
        let want = self
            .inputs
            .checked_mul(self.outputs)
            .ok_or_else(|| bad("输入宽度与输出通道相乘溢出"))?;
        if self.weights.len() != want {
            return Err(bad(format!(
                "权重 {} 格与 {inputs}×{outputs} 不符",
                self.weights.len(),
                inputs = self.inputs,
                outputs = self.outputs
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

    /// 取值判据：权重有限、尺度有限为正、动力学在范围内。
    fn check_values(&self) -> Result<()> {
        check_weights(&self.weights)?;
        check_scale(&self.scale)?;
        check_dynamics(self.threshold, self.leak, self.reset)
    }
}
