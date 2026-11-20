//! 归一化层：`BatchNorm2d`、`GroupNorm`、`InstanceNorm`、`LayerNorm`（f32，NHWC）。
//!
//! 四种归一化只差"统计在哪些轴上做"：BatchNorm 在 `(N,H,W)` 上按通道统计，GroupNorm 在
//! `(C/G,H,W)` 上按组统计，InstanceNorm 是 `G = C` 的 GroupNorm，LayerNorm 在最后一维统计。
//! 因此它们共用同一段"减均值、除标准差、再仿射"的骨架，统计范围由结构体参数决定。
//!
//! BatchNorm 的 `train` 路径用批内统计并更新滑动统计，`eval` 路径只用滑动统计——两条路径
//! 分开成两个函数，避免一个 `bool` 参数把"是否改状态"藏进签名里。

use diagnostic::{Code, Error, Result};

/// 二维批归一化：通道数、数值稳定项与滑动统计的动量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNorm2d {
    /// 通道数。
    pub channels: usize,
    /// `sqrt(var + eps)` 的稳定项。
    pub eps: f32,
    /// 滑动统计的动量：`new = (1−m)·old + m·batch`。
    pub momentum: f32,
}

/// 组归一化：把通道分成 `groups` 组，每组在自己的 `(C/G,H,W)` 内统计。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupNorm {
    /// 组数（必须整除通道数）。
    pub groups: usize,
    /// 通道数。
    pub channels: usize,
    /// 数值稳定项。
    pub eps: f32,
}

/// 实例归一化：`GroupNorm` 在 `groups == channels` 时的特例，单独列出是为了让调用方不必
/// 记住"传 C 个组"这件事。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceNorm {
    /// 通道数。
    pub channels: usize,
    /// 数值稳定项。
    pub eps: f32,
}

/// 层归一化：在最后一维（宽度 `width`）上统计，逐行独立。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNorm {
    /// 归一化宽度（最后一维）。
    pub width: usize,
    /// 数值稳定项。
    pub eps: f32,
}

impl BatchNorm2d {
    /// 训练期：用批内统计归一化，并把批统计按 `momentum` 混入滑动统计。
    ///
    /// # Errors
    ///
    /// 形状不符或 `n·h·w == 0` 时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        n: usize,
        h: usize,
        w: usize,
        running_mean: &mut [f32],
        running_var: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        let (c, count) = self.geometry(x, gamma, beta, n, h, w, out)?;
        let (mean, var) = batch_stats(x, c, count);
        for ci in 0..c {
            running_mean[ci] =
                (1.0 - self.momentum) * running_mean[ci] + self.momentum * mean[ci];
            // 方差用无偏口径进滑动统计（与 PyTorch 一致），归一化本身用有偏方差。
            let unbiased = if count > 1 {
                var[ci] * count as f32 / (count - 1) as f32
            } else {
                var[ci]
            };
            running_var[ci] = (1.0 - self.momentum) * running_var[ci] + self.momentum * unbiased;
        }
        affine(x, gamma, beta, &mean, &var, self.eps, c, count, out);
        Ok(())
    }

    /// 推理期：只用滑动统计，不改状态。
    ///
    /// # Errors
    ///
    /// 形状不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_eval(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        n: usize,
        h: usize,
        w: usize,
        running_mean: &[f32],
        running_var: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let (c, count) = self.geometry(x, gamma, beta, n, h, w, out)?;
        let mean = running_mean.to_vec();
        let var = running_var.to_vec();
        affine(x, gamma, beta, &mean, &var, self.eps, c, count, out);
        Ok(())
    }

    fn geometry(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        n: usize,
        h: usize,
        w: usize,
        out: &[f32],
    ) -> Result<(usize, usize)> {
        let c = self.channels;
        let count = n * h * w;
        check(x.len(), n * h * w * c, "x")?;
        check(gamma.len(), c, "gamma")?;
        check(beta.len(), c, "beta")?;
        check(out.len(), n * h * w * c, "out")?;
        if count == 0 {
            return Err(Error::new(Code::Invalid, "BatchNorm 不接受 0 个空间位置"));
        }
        Ok((c, count))
    }
}

impl GroupNorm {
    /// 前向：按组统计后归一化。
    ///
    /// # Errors
    ///
    /// `groups` 不整除通道数或形状不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        n: usize,
        spatial: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let c = self.channels;
        if self.groups == 0 || c % self.groups != 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("GroupNorm: {c} 个通道不能分成 {} 组", self.groups),
            ));
        }
        check(x.len(), n * spatial * c, "x")?;
        check(gamma.len(), c, "gamma")?;
        check(beta.len(), c, "beta")?;
        check(out.len(), n * spatial * c, "out")?;

        let per = c / self.groups;
        for bi in 0..n {
            for g in 0..self.groups {
                let c0 = g * per;
                let mut sum = 0.0f32;
                let mut sq = 0.0f32;
                let len = (per * spatial) as f32;
                for s in 0..spatial {
                    for ci in c0..c0 + per {
                        let v = x[(bi * spatial + s) * c + ci];
                        sum += v;
                        sq += v * v;
                    }
                }
                let mean = sum / len;
                let var = (sq / len - mean * mean).max(0.0);
                let inv = 1.0 / (var + self.eps).sqrt();
                for s in 0..spatial {
                    for ci in c0..c0 + per {
                        let idx = (bi * spatial + s) * c + ci;
                        out[idx] = (x[idx] - mean) * inv * gamma[ci] + beta[ci];
                    }
                }
            }
        }
        Ok(())
    }
}

impl InstanceNorm {
    /// 前向：等价于 `GroupNorm{groups: channels}`。
    ///
    /// # Errors
    ///
    /// 形状不符时返回说明。
    pub fn forward(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        n: usize,
        spatial: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let gn = GroupNorm {
            groups: self.channels,
            channels: self.channels,
            eps: self.eps,
        };
        gn.forward(x, gamma, beta, n, spatial, out)
    }
}

impl LayerNorm {
    /// 前向：逐行在最后一维上减均值除标准差。
    ///
    /// # Errors
    ///
    /// 缓冲长度不是 `width` 整数倍或仿射参数长度不符时返回说明。
    pub fn forward(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let width = self.width;
        if width == 0 || x.len() % width != 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("LayerNorm: {} 不是宽度 {width} 的整数倍", x.len()),
            ));
        }
        check(gamma.len(), width, "gamma")?;
        check(beta.len(), width, "beta")?;
        check(out.len(), x.len(), "out")?;
        for row in 0..x.len() / width {
            let src = &x[row * width..(row + 1) * width];
            let dst = &mut out[row * width..(row + 1) * width];
            // 方差用 f64 累加：f32 前缀和在宽度上千时会掉到 1e-4 级精度。
            let sum: f64 = src.iter().map(|v| f64::from(*v)).sum();
            let mean = (sum / width as f64) as f32;
            let sq: f64 = src
                .iter()
                .map(|v| {
                    let d = f64::from(*v) - f64::from(mean);
                    d * d
                })
                .sum();
            let inv = 1.0 / ((sq / width as f64) as f32 + self.eps).sqrt();
            for i in 0..width {
                dst[i] = (src[i] - mean) * inv * gamma[i] + beta[i];
            }
        }
        Ok(())
    }
}

/// 批内按通道的均值与（有偏）方差。
fn batch_stats(x: &[f32], c: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; c];
    let mut sq = vec![0.0f32; c];
    for p in 0..count {
        for ci in 0..c {
            let v = x[p * c + ci];
            mean[ci] += v;
            sq[ci] += v * v;
        }
    }
    for ci in 0..c {
        mean[ci] /= count as f32;
        let m2 = sq[ci] / count as f32 - mean[ci] * mean[ci];
        sq[ci] = m2.max(0.0);
    }
    (mean, sq)
}

/// 统一仿射：`(x−mean)/√(var+eps)·γ+β`，`mean`/`var` 按通道索引。
fn affine(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    c: usize,
    count: usize,
    out: &mut [f32],
) {
    let inv: Vec<f32> = var.iter().map(|v| 1.0 / (v + eps).sqrt()).collect();
    for p in 0..count {
        for ci in 0..c {
            let idx = p * c + ci;
            out[idx] = (x[idx] - mean[ci]) * inv[ci] * gamma[ci] + beta[ci];
        }
    }
}

/// 缓冲长度校验。
fn check(actual: usize, required: usize, what: &str) -> Result<()> {
    if actual < required {
        return Err(Error::new(
            Code::Invalid,
            format!("{what}: 缓冲 {actual}，需要 {required}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let z = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((z >> 8) % 2048) as f32 / 1024.0 - 1.0
            })
            .collect()
    }

    /// 逐通道的均值与方差（有偏），测试用独立口径，避免复用被测代码。
    fn stats(x: &[f32], c: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
        let mut mean = vec![0.0f64; c];
        let mut sq = vec![0.0f64; c];
        for p in 0..count {
            for ci in 0..c {
                let v = x[p * c + ci] as f64;
                mean[ci] += v;
                sq[ci] += v * v;
            }
        }
        let mean: Vec<f32> = mean.iter().map(|v| (v / count as f64) as f32).collect();
        let var: Vec<f32> = sq
            .iter()
            .enumerate()
            .map(|(ci, v)| ((v / count as f64) as f32 - mean[ci] * mean[ci]).max(0.0))
            .collect();
        (mean, var)
    }

    #[test]
    fn batchnorm_train_normalizes_per_channel() {
        let bn = BatchNorm2d { channels: 3, eps: 1e-5, momentum: 0.1 };
        let (n, h, w, c) = (2, 2, 2, 3);
        let count = n * h * w;
        let x = sample(count * c, 1);
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let mut rm = vec![0.0f32; c];
        let mut rv = vec![1.0f32; c];
        let mut out = vec![0.0f32; count * c];
        bn.forward_train(&x, &gamma, &beta, n, h, w, &mut rm, &mut rv, &mut out)
            .unwrap();

        let (mean, var) = stats(&x, c, count);
        for ci in 0..c {
            let want: Vec<f32> = (0..count)
                .map(|p| (x[p * c + ci] - mean[ci]) / (var[ci] + bn.eps).sqrt())
                .collect();
            let got: Vec<f32> = (0..count).map(|p| out[p * c + ci]).collect();
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-5, "通道 {ci}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn batchnorm_train_updates_running_stats() {
        let bn = BatchNorm2d { channels: 1, eps: 1e-5, momentum: 0.5 };
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let gamma = [1.0f32];
        let beta = [0.0f32];
        let mut rm = [0.0f32];
        let mut rv = [1.0f32];
        let mut out = [0.0f32; 4];
        bn.forward_train(&x, &gamma, &beta, 1, 1, 4, &mut rm, &mut rv, &mut out)
            .unwrap();
        // batch mean = 2.5，动量 0.5 → 滑动均值 1.25。
        assert!((rm[0] - 1.25).abs() < 1e-6, "running_mean {}", rm[0]);
        // 有偏方差 1.25，无偏 5/3≈1.6667 → 滑动方差 (1+1.6667)/2。
        assert!((rv[0] - 1.333_333).abs() < 1e-4, "running_var {}", rv[0]);
    }

    #[test]
    fn batchnorm_eval_uses_running_stats_only() {
        let bn = BatchNorm2d { channels: 2, eps: 1e-5, momentum: 0.1 };
        let x = sample(4 * 2, 2);
        let gamma = [1.0f32, 2.0];
        let beta = [0.5f32, -0.5];
        let rm = [0.25f32, -0.75];
        let rv = [4.0f32, 16.0];
        let mut out = vec![0.0f32; 8];
        bn.forward_eval(&x, &gamma, &beta, 1, 1, 4, &rm, &rv, &mut out)
            .unwrap();
        for p in 0..4 {
            for ci in 0..2 {
                let want = (x[p * 2 + ci] - rm[ci]) / (rv[ci] + bn.eps).sqrt() * gamma[ci] + beta[ci];
                assert!((out[p * 2 + ci] - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn groupnorm_yields_zero_mean_unit_variance() {
        let gn = GroupNorm { groups: 2, channels: 4, eps: 1e-5 };
        let (n, spatial, c) = (2, 5, 4);
        let x = sample(n * spatial * c, 3);
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let mut out = vec![0.0f32; n * spatial * c];
        gn.forward(&x, &gamma, &beta, n, spatial, &mut out).unwrap();

        let per = c / gn.groups;
        for bi in 0..n {
            for g in 0..gn.groups {
                let mut sum = 0.0f32;
                let mut sq = 0.0f32;
                let len = (per * spatial) as f32;
                for s in 0..spatial {
                    for ci in g * per..g * per + per {
                        let v = out[(bi * spatial + s) * c + ci];
                        sum += v;
                        sq += v * v;
                    }
                }
                assert!(sum.abs() / len < 1e-5, "组均值未归零");
                assert!((sq / len - 1.0).abs() < 1e-4, "组方差未归一");
            }
        }
    }

    #[test]
    fn instancenorm_equals_groupnorm_with_c_groups() {
        let (n, spatial, c) = (1, 4, 3);
        let x = sample(n * spatial * c, 4);
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let mut a = vec![0.0f32; n * spatial * c];
        let mut b = vec![0.0f32; n * spatial * c];
        InstanceNorm { channels: c, eps: 1e-5 }
            .forward(&x, &gamma, &beta, n, spatial, &mut a)
            .unwrap();
        GroupNorm { groups: c, channels: c, eps: 1e-5 }
            .forward(&x, &gamma, &beta, n, spatial, &mut b)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn layernorm_normalizes_each_row() {
        let ln = LayerNorm { width: 5, eps: 1e-5 };
        let x = sample(15, 5);
        let gamma = vec![1.0f32; 5];
        let beta = vec![0.0f32; 5];
        let mut out = vec![0.0f32; 15];
        ln.forward(&x, &gamma, &beta, &mut out).unwrap();
        for row in 0..3 {
            let src = &out[row * 5..(row + 1) * 5];
            let mean: f64 = src.iter().map(|v| f64::from(*v)).sum::<f64>() / 5.0;
            assert!(mean.abs() < 1e-6, "行 {row} 均值 {mean}");
            // 输入的方差为 var 时，归一化后方差是 var/(var+eps)——eps 会把它压到 1 以下。
            let xin = &x[row * 5..(row + 1) * 5];
            let im: f64 = xin.iter().map(|v| f64::from(*v)).sum::<f64>() / 5.0;
            let var: f64 = xin
                .iter()
                .map(|v| (f64::from(*v) - im).powi(2))
                .sum::<f64>()
                / 5.0;
            let want = (var / (var + f64::from(ln.eps))) as f32;
            let got: f32 = src
                .iter()
                .map(|v| (v - mean as f32).powi(2))
                .sum::<f32>()
                / 5.0;
            assert!((got - want).abs() < 1e-5, "行 {row} 方差 {got}，期望 {want}");
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let gn = GroupNorm { groups: 3, channels: 4, eps: 1e-5 };
        let x = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        assert!(gn.forward(&x, &[1.0; 4], &[0.0; 4], 1, 2, &mut out).is_err());
        let ln = LayerNorm { width: 0, eps: 1e-5 };
        assert!(ln.forward(&x, &[], &[], &mut out).is_err());
    }
}
