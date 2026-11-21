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
const SITE: &str = "algorithm::spike";

/// `i8` 权重的最大绝对值：整数累加上界由它与输入宽度算。
const WEIGHT_MAGNITUDE: usize = 127;

/// 三元脉冲判据：`−1` / `0` / `+1`。稠密与事件两条路径、以及反传，都只认这三个值。
fn ternary(spike: i8) -> bool {
    matches!(spike, -1..=1)
}

/// 输出侧梯度判据：长度与输出通道一致，取值有限。
///
/// `NaN` 必须挡在入口：它进了乘法就一路污染到每条权重，事后从结果里看不出是哪一格带进来的。
fn check_gradient(d_out: &[f32], outputs: usize) -> Result<()> {
    if d_out.len() != outputs {
        return Err(bad(format!(
            "输出梯度 {} 格与输出通道 {} 不符",
            d_out.len(),
            outputs
        )));
    }
    match d_out.iter().find(|value| !finite(**value)) {
        Some(value) => Err(bad(format!("输出梯度必须是有限值，得到 {value:?}"))),
        None => Ok(()),
    }
}

/// 学习率判据：有限为正（零学习率等于「什么都不学」，负学习率等于「朝梯度方向上升」）。
/// 浮点权重判据：`NaN` 经 `round().clamp()` 会落成某个看着正常的 `i8`，只能在量化之前挡住。
fn check_weights(weights: &[f32]) -> Result<()> {
    match weights.iter().find(|weight| !finite(**weight)) {
        Some(weight) => Err(bad(format!("浮点主权重必须是有限值，得到 {weight:?}"))),
        None => Ok(()),
    }
}

fn check_learning_rate(lr: f32) -> Result<()> {
    if !finite(lr) || lr <= 0.0 {
        return Err(bad(format!("学习率必须有限为正，得到 {lr:?}")));
    }
    Ok(())
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

mod cell;
mod master;
mod membrane;
mod surrogate;

pub use cell::Spiking;
pub use master::Master;
pub use membrane::Membrane;
pub use surrogate::{Gradients, Surrogate};

#[cfg(test)]
#[path = "../tests/spike_tests.rs"]
mod spike_tests;
