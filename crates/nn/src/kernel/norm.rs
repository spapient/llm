//! 归一化层：`BatchNorm2d`、`GroupNorm`、`InstanceNorm`、`LayerNorm`（NHWC，精度由 `S: Scalar` 决定）。

use crate::Scalar;
use crate::guard::check_all;
use diagnostic::{Code, Error, Result};

/// 二维批归一化：通道数、数值稳定项与滑动统计的动量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNorm2d<S: Scalar> {
    /// 通道数。
    pub channels: usize,
    /// `sqrt(var + eps)` 的稳定项。
    pub eps: S,
    /// 滑动统计的动量：`new = (1−m)·old + m·batch`。
    pub momentum: S,
}

/// 组归一化：把通道分成 `groups` 组，每组在自己的 `(C/G,H,W)` 内统计。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupNorm<S: Scalar> {
    /// 组数（必须整除通道数）。
    pub groups: usize,
    /// 通道数。
    pub channels: usize,
    /// 数值稳定项。
    pub eps: S,
}

/// 实例归一化：`GroupNorm` 在 `groups == channels` 时的特例，单独列出是为了让调用方不必
/// 记住"传 C 个组"这件事。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceNorm<S: Scalar> {
    /// 通道数。
    pub channels: usize,
    /// 数值稳定项。
    pub eps: S,
}

/// 层归一化：在最后一维（宽度 `width`）上统计，逐行独立。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNorm<S: Scalar> {
    /// 归一化宽度（最后一维）。
    pub width: usize,
    /// 数值稳定项。
    pub eps: S,
}

impl<S: Scalar> BatchNorm2d<S> {
    /// 训练期：用批内统计归一化，并把批统计按 `momentum` 混入滑动统计。
    ///
    /// # Errors
    ///
    /// 形状不符或 `n·h·w == 0` 时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train(
        &self,
        x: &[S],
        gamma: &[S],
        beta: &[S],
        n: usize,
        h: usize,
        w: usize,
        running_mean: &mut [S],
        running_var: &mut [S],
        out: &mut [S],
    ) -> Result<()> {
        let (c, count) = self.geometry(x, gamma, beta, n, h, w, out)?;
        let (mean, var) = batch_stats(x, c, count);
        for ci in 0..c {
            running_mean[ci] =
                (S::ONE - self.momentum) * running_mean[ci] + self.momentum * mean[ci];
            // 方差用无偏口径进滑动统计（与 PyTorch 一致），归一化本身用有偏方差。
            let unbiased = if count > 1 {
                var[ci] * S::from_usize(count) / S::from_usize(count - 1)
            } else {
                var[ci]
            };
            running_var[ci] = (S::ONE - self.momentum) * running_var[ci] + self.momentum * unbiased;
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
        x: &[S],
        gamma: &[S],
        beta: &[S],
        n: usize,
        h: usize,
        w: usize,
        running_mean: &[S],
        running_var: &[S],
        out: &mut [S],
    ) -> Result<()> {
        let (c, count) = self.geometry(x, gamma, beta, n, h, w, out)?;
        let mean = running_mean.to_vec();
        let var = running_var.to_vec();
        affine(x, gamma, beta, &mean, &var, self.eps, c, count, out);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn geometry(
        &self,
        x: &[S],
        gamma: &[S],
        beta: &[S],
        n: usize,
        h: usize,
        w: usize,
        out: &[S],
    ) -> Result<(usize, usize)> {
        let c = self.channels;
        let count = n * h * w;
        check_all(&[
            (x.len(), count * c, "x"),
            (gamma.len(), c, "gamma"),
            (beta.len(), c, "beta"),
            (out.len(), count * c, "out"),
        ])?;
        if count == 0 {
            return Err(Error::new(Code::Invalid, "BatchNorm 不接受 0 个空间位置"));
        }
        Ok((c, count))
    }
}

impl<S: Scalar> GroupNorm<S> {
    /// 前向：按组统计后归一化。
    ///
    /// # Errors
    ///
    /// `groups` 不整除通道数或形状不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[S],
        gamma: &[S],
        beta: &[S],
        n: usize,
        spatial: usize,
        out: &mut [S],
    ) -> Result<()> {
        let c = self.channels;
        if self.groups == 0 || !c.is_multiple_of(self.groups) {
            return Err(Error::new(
                Code::Invalid,
                format!("GroupNorm: {c} 个通道不能分成 {} 组", self.groups),
            ));
        }
        check_all(&[
            (x.len(), n * spatial * c, "x"),
            (gamma.len(), c, "gamma"),
            (beta.len(), c, "beta"),
            (out.len(), n * spatial * c, "out"),
        ])?;
        normalize_groups(
            x,
            gamma,
            beta,
            out,
            n,
            spatial,
            c,
            c / self.groups,
            self.groups,
            self.eps,
        );
        Ok(())
    }
}

impl<S: Scalar> InstanceNorm<S> {
    /// 前向：等价于 `GroupNorm{groups: channels}`。
    ///
    /// # Errors
    ///
    /// 形状不符时返回说明。
    pub fn forward(
        &self,
        x: &[S],
        gamma: &[S],
        beta: &[S],
        n: usize,
        spatial: usize,
        out: &mut [S],
    ) -> Result<()> {
        let gn = GroupNorm {
            groups: self.channels,
            channels: self.channels,
            eps: self.eps,
        };
        gn.forward(x, gamma, beta, n, spatial, out)
    }
}

impl<S: Scalar> LayerNorm<S> {
    /// 前向：逐行在最后一维上减均值除标准差。
    ///
    /// # Errors
    ///
    /// 缓冲长度不是 `width` 整数倍或仿射参数长度不符时返回说明。
    pub fn forward(&self, x: &[S], gamma: &[S], beta: &[S], out: &mut [S]) -> Result<()> {
        let width = self.width;
        if width == 0 || !x.len().is_multiple_of(width) {
            return Err(Error::new(
                Code::Invalid,
                format!("LayerNorm: {} 不是宽度 {width} 的整数倍", x.len()),
            ));
        }
        check_all(&[
            (gamma.len(), width, "gamma"),
            (beta.len(), width, "beta"),
            (out.len(), x.len(), "out"),
        ])?;
        for row in 0..x.len() / width {
            normalize_row(x, gamma, beta, out, width, row, self.eps);
        }
        Ok(())
    }
}

/// 批内按通道的均值与（有偏）方差。
fn batch_stats<S: Scalar>(x: &[S], c: usize, count: usize) -> (Vec<S>, Vec<S>) {
    let mut mean = vec![S::ZERO; c];
    let mut sq = vec![S::ZERO; c];
    for p in 0..count {
        for ci in 0..c {
            let v = x[p * c + ci];
            mean[ci] = mean[ci] + v;
            sq[ci] = sq[ci] + v * v;
        }
    }
    let count_s = S::from_usize(count);
    for ci in 0..c {
        mean[ci] = mean[ci] / count_s;
        let m2 = sq[ci] / count_s - mean[ci] * mean[ci];
        sq[ci] = m2.max(S::ZERO);
    }
    (mean, sq)
}

/// 统一仿射：`(x−mean)/√(var+eps)·γ+β`，`mean`/`var` 按通道索引。
#[allow(clippy::too_many_arguments)]
fn affine<S: Scalar>(
    x: &[S],
    gamma: &[S],
    beta: &[S],
    mean: &[S],
    var: &[S],
    eps: S,
    c: usize,
    count: usize,
    out: &mut [S],
) {
    let inv: Vec<S> = var.iter().map(|v| S::ONE / (*v + eps).sqrt()).collect();
    for p in 0..count {
        for ci in 0..c {
            let idx = p * c + ci;
            out[idx] = (x[idx] - mean[ci]) * inv[ci] * gamma[ci] + beta[ci];
        }
    }
}

/// 按组遍历并逐组归一化（`GroupNorm` 的骨架）。
#[allow(clippy::too_many_arguments)]
fn normalize_groups<S: Scalar>(
    x: &[S],
    gamma: &[S],
    beta: &[S],
    out: &mut [S],
    n: usize,
    spatial: usize,
    c: usize,
    per: usize,
    groups: usize,
    eps: S,
) {
    for bi in 0..n {
        for g in 0..groups {
            normalize_group(x, gamma, beta, out, spatial, c, per, bi, g * per, eps);
        }
    }
}

/// 单组 `(C/G,H,W)` 内的减均值、除标准差、仿射。
#[allow(clippy::too_many_arguments)]
fn normalize_group<S: Scalar>(
    x: &[S],
    gamma: &[S],
    beta: &[S],
    out: &mut [S],
    spatial: usize,
    c: usize,
    per: usize,
    bi: usize,
    c0: usize,
    eps: S,
) {
    let mut sum = S::ZERO;
    let mut sq = S::ZERO;
    let len = S::from_usize(per * spatial);
    for s in 0..spatial {
        for ci in c0..c0 + per {
            let v = x[(bi * spatial + s) * c + ci];
            sum = sum + v;
            sq = sq + v * v;
        }
    }
    let mean = sum / len;
    let var = (sq / len - mean * mean).max(S::ZERO);
    let inv = S::ONE / (var + eps).sqrt();
    for s in 0..spatial {
        for ci in c0..c0 + per {
            let idx = (bi * spatial + s) * c + ci;
            out[idx] = (x[idx] - mean) * inv * gamma[ci] + beta[ci];
        }
    }
}

/// 单行 LayerNorm；方差用 f64 累加以保 f32 在宽行上的精度。
fn normalize_row<S: Scalar>(
    x: &[S],
    gamma: &[S],
    beta: &[S],
    out: &mut [S],
    width: usize,
    row: usize,
    eps: S,
) {
    let src = &x[row * width..(row + 1) * width];
    let dst = &mut out[row * width..(row + 1) * width];
    let sum: f64 = src.iter().map(|v| v.to_f64()).sum();
    let mean = S::from_f64(sum / width as f64);
    let sq: f64 = src
        .iter()
        .map(|v| {
            let d = v.to_f64() - mean.to_f64();
            d * d
        })
        .sum();
    let inv = S::ONE / (S::from_f64(sq / width as f64) + eps).sqrt();
    for i in 0..width {
        dst[i] = (src[i] - mean) * inv * gamma[i] + beta[i];
    }
}

#[cfg(test)]
mod tests {
    include!("../../tests/kernel_norm_tests.rs");
}
