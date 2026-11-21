//! 随机梯度下降（SGD）：逐元素更新 + 解耦权重衰减 + 全局范数裁剪。

use super::{Optimizer, SITE, Slot, check_slots, clip_scale, decay_rate, global_norm};
use diagnostic::{Code, Error, Result};

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
