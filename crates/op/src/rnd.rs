//! 零依赖确定性随机源：splitmix64。
//!
//! 经典层的 `dropout` 与 `init` 都要抽样，但 `op` 不引入外部 RNG crate（保持零依赖）。
//! 这里只提供"同一种子必得同一序列"的最小口径，便于真机对拍与复现。

/// splitmix64 状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// 由种子构造。
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// 下一个 64 位输出。
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `[0, 1)` 均匀 f32：取高 24 位，保 24 位有效精度。
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / 16_777_216.0)
    }

    /// 标准正态（Box–Muller），用于 `init` 的正态初始化。
    pub fn next_normal(&mut self) -> f32 {
        // u1 取 (0,1] 避开 ln(0)。
        let u1 = 1.0 - self.next_f32();
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = core::f32::consts::TAU * u2;
        r * theta.cos()
    }
}

#[cfg(test)]
#[path = "../tests/rnd_tests.rs"]
mod tests;
