//! 确定性抽样一族：随机源、dropout 掩码与权重初始化。

pub mod rnd {
    //! 零依赖确定性随机源：splitmix64。

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
    #[cfg(test)]
    mod tests {
        include!("../tests/spec/rnd_tests.rs");
    }
}

pub mod dropout {
    //! Dropout：训练期按 `p` 置零、按 `1/(1-p)` 缩放；推理期为恒等。

    use diagnostic::{Code, Error, Result};

    use crate::rnd::SplitMix64;

    /// 就地对 `x` 施加 dropout。
    ///
    /// `train == false` 时原样返回（推理期不缩放）；训练期要求 `0 ≤ p < 1`。
    ///
    /// # Errors
    ///
    /// `p` 越界时返回说明。
    pub fn dropout(x: &mut [f32], p: f32, seed: u64, train: bool) -> Result<()> {
        if !train || p == 0.0 {
            return Ok(());
        }
        if !(0.0..1.0).contains(&p) {
            return Err(Error::new(
                Code::Invalid,
                format!("dropout 概率 {p} 应在 [0,1)"),
            ));
        }
        apply_mask(x, p, seed);
        Ok(())
    }

    /// 用 [`SplitMix64`] 抽同一掩码：低于 `p` 置零，否则乘 `1/(1-p)` 保均值。
    fn apply_mask(x: &mut [f32], p: f32, seed: u64) {
        let keep = 1.0 / (1.0 - p);
        let mut rng = SplitMix64::new(seed);
        for v in x.iter_mut() {
            if rng.next_f32() < p {
                *v = 0.0;
            } else {
                *v *= keep;
            }
        }
    }

    #[cfg(test)]
    #[cfg(test)]
    mod tests {
        include!("../tests/spec/dropout_tests.rs");
    }
}

pub mod init {
    //! 权重初始化：Xavier、Kaiming 与正交初始化（口径对齐 PyTorch 的 `nn.init`）。

    use diagnostic::{Code, Error, Result};

    use crate::rnd::SplitMix64;

    /// Xavier 均匀：`U(-a, a)`，`a = sqrt(6 / (fan_in + fan_out))`。
    ///
    /// # Errors
    ///
    /// `fan_in`/`fan_out` 为 0 时返回说明。
    pub fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Result<Vec<f32>> {
        let bound = xavier_bound(fan_in, fan_out)?;
        Ok(uniform(fan_in * fan_out, bound, seed))
    }

    /// Xavier 正态：`N(0, σ)`，`σ = sqrt(2 / (fan_in + fan_out))`。
    ///
    /// Kaiming 均匀：`U(-a, a)`，`a = sqrt(3)·gain / sqrt(fan_in)`，`gain = sqrt(2/(1+leak²))`。
    ///
    /// # Errors
    ///
    /// `fan_in` 为 0 时返回说明。
    pub fn kaiming_uniform(
        fan_in: usize,
        fan_out: usize,
        leak: f32,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let std = kaiming_std(fan_in, leak)?;
        Ok(uniform(fan_in * fan_out, bound_of(std), seed))
    }

    /// Kaiming 正态：`N(0, σ)`，`σ = gain / sqrt(fan_in)`。
    ///
    /// # Errors
    ///
    /// `fan_in` 为 0 时返回说明。
    pub fn kaiming_normal(fan_in: usize, fan_out: usize, leak: f32, seed: u64) -> Result<Vec<f32>> {
        let std = kaiming_std(fan_in, leak)?;
        Ok(normal(fan_in * fan_out, std, seed))
    }

    /// 正交初始化：`rows ≥ cols` 时列正交，否则行正交；输出按 `[rows, cols]` 行主序，再乘 `gain`。
    ///
    /// 用改进 Gram–Schmidt 对高斯矩阵正交化，数值上比拼 Givens 更稳。
    ///
    /// # Errors
    ///
    /// 任一维为 0 时返回说明。
    pub fn orthogonal(rows: usize, cols: usize, gain: f32, seed: u64) -> Result<Vec<f32>> {
        if rows == 0 || cols == 0 {
            return Err(Error::new(Code::Invalid, "正交初始化不接受 0 维"));
        }
        let big = rows.max(cols);
        let small = rows.min(cols);
        let mat = orthonormalize(big, small, gain, seed);
        Ok(reshape_orthogonal(&mat, small, rows, cols))
    }

    /// 按 `[big, small]` 生成高斯列并用改进 Gram–Schmidt 正交化。
    fn orthonormalize(big: usize, small: usize, gain: f32, seed: u64) -> Vec<f32> {
        let mut mat = vec![0.0f32; big * small];
        let mut rng = SplitMix64::new(seed);
        for v in &mut mat {
            *v = rng.next_normal();
        }
        for col in 0..small {
            subtract_prior(&mut mat, big, small, col);
            normalize_column(&mut mat, big, small, col, gain);
        }
        mat
    }

    /// 把列 `col` 对前面所有列做一次 Gram–Schmidt 减法。
    fn subtract_prior(mat: &mut [f32], big: usize, small: usize, col: usize) {
        for prev in 0..col {
            let mut dot = 0.0f32;
            for r in 0..big {
                dot += mat[r * small + col] * mat[r * small + prev];
            }
            for r in 0..big {
                mat[r * small + col] -= dot * mat[r * small + prev];
            }
        }
    }

    /// 把列 `col` 归一化并乘 `gain`；零范数列保持为零。
    fn normalize_column(mat: &mut [f32], big: usize, small: usize, col: usize, gain: f32) {
        let mut norm = 0.0f32;
        for r in 0..big {
            norm += mat[r * small + col] * mat[r * small + col];
        }
        let inv = if norm > 0.0 { gain / norm.sqrt() } else { 0.0 };
        for r in 0..big {
            mat[r * small + col] *= inv;
        }
    }

    /// `[big, small]` → `[rows, cols]`：`rows ≥ cols` 时列正交可直接拷贝，否则转置成行正交。
    fn reshape_orthogonal(mat: &[f32], small: usize, rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        if rows >= cols {
            out.copy_from_slice(mat);
            return out;
        }
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] = mat[c * small + r];
            }
        }
        out
    }

    /// Xavier 的界 `sqrt(6/(fan_in+fan_out))`。
    fn xavier_bound(fan_in: usize, fan_out: usize) -> Result<f32> {
        if fan_in == 0 || fan_out == 0 {
            return Err(Error::new(Code::Invalid, "Xavier 不接受 0 维"));
        }
        Ok((6.0 / (fan_in + fan_out) as f32).sqrt())
    }

    /// Kaiming 的标准差 `gain/sqrt(fan_in)`。
    fn kaiming_std(fan_in: usize, leak: f32) -> Result<f32> {
        if fan_in == 0 {
            return Err(Error::new(Code::Invalid, "Kaiming 不接受 0 维"));
        }
        let gain = (2.0 / (1.0 + leak * leak)).sqrt();
        Ok(gain / (fan_in as f32).sqrt())
    }

    /// 均匀分布的标准差 `σ = bound/√3` ⇒ `bound = √3·σ`。
    fn bound_of(std: f32) -> f32 {
        3.0f32.sqrt() * std
    }

    /// `[−bound, bound)` 均匀抽样。
    fn uniform(len: usize, bound: f32, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..len)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
            .collect()
    }

    /// `N(0, std)` 抽样。
    fn normal(len: usize, std: f32, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..len).map(|_| rng.next_normal() * std).collect()
    }

    #[cfg(test)]
    #[cfg(test)]
    mod tests {
        include!("../tests/spec/init_tests.rs");
    }
}
