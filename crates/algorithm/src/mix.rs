//! `T_θ` 三通道合成律（§1.5，数学验证 31）。

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
#[path = "pulse.rs"]
mod pulse;

pub use pulse::{Audit, Encoder, Events, Lif, accumulate, encode_id, steepest};

#[cfg(test)]
#[path = "../tests/mix_tests.rs"]
mod tests;
