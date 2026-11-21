//! Adam（AdamW 解耦衰减 + 偏差修正 + 全局范数裁剪）。

use super::{Optimizer, SITE, Slot, check_slots, clip_scale, decay_rate, global_norm};
use diagnostic::{Code, Error, Result};

/// Adam（AdamW 解耦衰减 + 偏差修正 + 全局范数裁剪）。
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
