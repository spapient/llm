//! `T_θ` 三通道合成律（§1.5，数学验证 31）。
//!
//! `y_cont = π·y_α + (1−π)·y_β`；`y_γ = Encode_event(y_cont)`；`y_t = (1−γ)·y_cont + γ·y_γ`。
//! 系数 `((1−γ)π, (1−γ)(1−π), γ)` 构成单位剖分（和为 1、各项 ≥ 0），故 `y_t` 落在三通道输出的
//! 凸包内。事件编码 `y_γ` 由调用方（[`crate::mix::pulse`]）算出，本模块只做纯合成，无状态。

use op::scan::Scalar;

/// α/β 凸组合：`y_cont = π·y_α + (1−π)·y_β`。
pub fn cont<S: Scalar>(ya: &[S], yb: &[S], pi: S) -> Vec<S> {
    debug_assert_eq!(ya.len(), yb.len());
    let om = S::ONE.sub(pi);
    ya.iter()
        .zip(yb)
        .map(|(&a, &b)| a.mul(pi).add(b.mul(om)))
        .collect()
}

/// 连续/事件门控：`y_t = (1−γ)·y_cont + γ·y_γ`。
pub fn gate<S: Scalar>(yc: &[S], yg: &[S], gamma: S) -> Vec<S> {
    debug_assert_eq!(yc.len(), yg.len());
    let om = S::ONE.sub(gamma);
    yc.iter()
        .zip(yg)
        .map(|(&c, &g)| c.mul(om).add(g.mul(gamma)))
        .collect()
}

/// 全合成：`y_t = (1−γ)(π·y_α + (1−π)·y_β) + γ·y_γ`。
///
/// 单趟融合而不是 `gate(&cont(..))`：省掉中间向量的一次分配与一次全量读写。运算次序与两步写法
/// 逐项相同，故结果逐位一致（由 `mix_tests` 的等价测试锁定）。
pub fn synthesize<S: Scalar>(ya: &[S], yb: &[S], yg: &[S], pi: S, gamma: S) -> Vec<S> {
    debug_assert_eq!(ya.len(), yb.len());
    debug_assert_eq!(ya.len(), yg.len());
    let om_pi = S::ONE.sub(pi);
    let om_g = S::ONE.sub(gamma);
    ya.iter()
        .zip(yb)
        .zip(yg)
        .map(|((&a, &b), &g)| a.mul(pi).add(b.mul(om_pi)).mul(om_g).add(g.mul(gamma)))
        .collect()
}
mod pulse;

pub use pulse::{Audit, Encoder, Events, Lif, accumulate, encode_id, steepest};

#[cfg(test)]
#[path = "../tests/mix_tests.rs"]
mod tests;
