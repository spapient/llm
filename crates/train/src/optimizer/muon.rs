//! Muon：动量、矩阵形状校验与 Newton–Schulz 近似正交化。
//!
//! 公共优化器契约、裁剪和衰减助手由父模块统一拥有，本文件只实现矩阵更新路径。

use diagnostic::{Code, Error, Result};

use super::{
    Optimizer, SITE, Slot, check_slots, clip_scale, decay_rate, global_norm, write_decay_clip,
};

/// Muon（MomentUm Orthogonalized by Newton–Schulz）：矩阵参数的近似正交化更新。
#[derive(Debug, Clone, PartialEq)]
pub struct Muon {
    momentum: f32,
    nesterov: bool,
    eps: f32,
    weight_decay: f32,
    clip_norm: Option<f32>,
    step: u64,
    velocity: Vec<Vec<f32>>,
}

/// Newton–Schulz 的迭代次数：Muon 参考实现取 5，够把 `[0.3, 3]` 的谱压到 1 附近。
const NS_STEPS: usize = 5;

/// 五次迭代的多项式系数 `(a, b, c)`：`X ← aX + b(XA) + c(XA²)`，`A` 是小边上的 Gram。
const NS_COEF: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// 步长缩放的系数：与参考实现一致，把正交更新的尺度对齐 AdamW。
const SCALE_COEF: f32 = 0.2;

impl Muon {
    /// 默认超参：`μ = 0.95`、Nesterov、`ε = 1e-7`，不衰减、不裁剪。
    pub fn new() -> Self {
        Self::default()
    }

    builders! {
        /// 设置动量系数 `μ`。
        with_momentum => momentum: f32,
        /// 设置归一化保护项 `ε`。
        with_eps => eps: f32,
    }

    /// 关掉 Nesterov 前瞻（默认开）。
    pub fn without_nesterov(mut self) -> Self {
        self.nesterov = false;
        self
    }

    /// 确保槽位 `idx` 的动量缓冲长度等于参数长度；首次见到的槽位分配零向量。
    fn velocity(&mut self, idx: usize, len: usize) -> &mut Vec<f32> {
        if idx >= self.velocity.len() {
            let missing = idx - self.velocity.len() + 1;
            self.velocity
                .resize_with(self.velocity.len() + missing, Vec::new);
        }
        let buf = &mut self.velocity[idx];
        if buf.len() != len {
            buf.resize(len, 0.0);
        }
        buf
    }

    /// 一个槽的推进：动量累加 →（可选 Nesterov）→ 正交化 → 写回。
    fn advance(
        &mut self,
        idx: usize,
        slot: &mut Slot<'_>,
        (rows, cols): (usize, usize),
        scale: f32,
        lr: f32,
    ) {
        let (beta, nesterov, eps) = (self.momentum, self.nesterov, self.eps);
        let decay = decay_rate(slot.role, self.weight_decay);
        let velocity = self.velocity(idx, rows * cols);
        for (tracked, &grad) in velocity.iter_mut().zip(slot.grads.iter()) {
            *tracked = beta * *tracked + scale * grad;
        }
        let direction = if nesterov {
            nesterov_direction(velocity, slot.grads, beta, scale)
        } else {
            velocity.clone()
        };
        let orthogonal = orthogonalize(&direction, rows, cols, eps);
        let factor = lr * scale_factor(rows, cols);
        for (param, &step) in slot.params.iter_mut().zip(&orthogonal) {
            *param -= factor * step;
            *param -= lr * decay * *param;
        }
    }

    /// 动量表（按槽位编号）：快照导出走这里，优化器之外不另存一份状态。
    pub fn momentum_table(&self) -> &[Vec<f32>] {
        &self.velocity
    }

    /// 优化器全局步数。
    pub fn step_count(&self) -> u64 {
        self.step
    }
}

decay_clip!(Muon);

impl Default for Muon {
    fn default() -> Self {
        Self {
            momentum: 0.95,
            nesterov: true,
            eps: 1e-7,
            weight_decay: 0.0,
            clip_norm: None,
            step: 0,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Muon {
    fn set_clip_norm(&mut self, clip_norm: Option<f32>) {
        self.clip_norm = clip_norm;
    }

    fn step(&mut self, slots: &mut [Slot<'_>], lr: f32) -> Result<()> {
        check_slots(slots)?;
        let shapes = shapes_of(slots)?;
        let scale = clip_scale(global_norm(slots), self.clip_norm);
        self.step = self.step.saturating_add(1);
        for (idx, (slot, &shape)) in slots.iter_mut().zip(&shapes).enumerate() {
            self.advance(idx, slot, shape, scale, lr);
        }
        Ok(())
    }

    fn restore(
        &mut self,
        momentum: Vec<Vec<f32>>,
        variance: Vec<Vec<f32>>,
        step: u64,
    ) -> Result<()> {
        if !variance.is_empty() {
            let message = format!(
                "Muon 只持一阶动量：快照带回 {} 个二阶矩槽，无法回载",
                variance.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        self.velocity = momentum;
        self.step = step;
        Ok(())
    }

    fn write_config(&self, sink: &mut dyn FnMut(u64)) {
        for value in [self.momentum, self.eps] {
            sink(value.to_bits().into());
        }
        sink(self.nesterov.into());
        write_decay_clip(self.weight_decay, self.clip_norm, sink);
    }
}

/// 每个槽的矩阵形状，**全部**先查完再返回。
fn shapes_of(slots: &[Slot<'_>]) -> Result<Vec<(usize, usize)>> {
    let mut shapes = Vec::with_capacity(slots.len());
    for (idx, slot) in slots.iter().enumerate() {
        shapes.push(matrix_of(idx, slot)?);
    }
    Ok(shapes)
}

/// Nesterov 前瞻方向 `g + μ·u`：动量缓冲 `u` 已经含了本步的梯度，前瞻取的是「再多走一点」。
fn nesterov_direction(velocity: &[f32], grads: &[f32], beta: f32, scale: f32) -> Vec<f32> {
    velocity
        .iter()
        .zip(grads)
        .map(|(&tracked, &grad)| scale * grad + beta * tracked)
        .collect()
}

/// 一个槽的矩阵形状：非矩阵、零维或梯度非有限都在这里拒绝。
fn matrix_of(idx: usize, slot: &Slot<'_>) -> Result<(usize, usize)> {
    let Some((rows, cols)) = slot.shape else {
        let message = format!(
            "第 {idx} 槽（角色 {}）不是矩阵：Muon 的正交化只在矩阵上有定义，\
             偏置/归一化增益按角色分派给 AdamW",
            slot.role.as_str()
        );
        return Err(Error::new(Code::Invalid, message).with(SITE));
    };
    if rows == 0 || cols == 0 {
        let message = format!("第 {idx} 槽声明为 [{rows}, {cols}]：两边都必须非零");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    if let Some(bad) = slot.grads.iter().position(|grad| !grad.is_finite()) {
        let message = format!("第 {idx} 槽梯度第 {bad} 项非有限：正交化会把它铺满整个方向");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok((rows, cols))
}

/// 步长缩放 `0.2·√max(rows, cols)`：正交后奇异值为 1，不缩放的话更新量随矩阵阶数增长。
fn scale_factor(rows: usize, cols: usize) -> f32 {
    SCALE_COEF * (rows.max(cols) as f32).sqrt()
}

/// 近似正交化：先按 Frobenius 范数归一，再迭代 [`NS_STEPS`] 次 Newton–Schulz。
pub(crate) fn orthogonalize(grad: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let norm = grad.iter().fold(0.0_f32, |acc, v| acc + v * v).sqrt();
    let scale = 1.0 / (norm + eps);
    let mut x: Vec<f32> = grad.iter().map(|v| v * scale).collect();
    for _ in 0..NS_STEPS {
        let (gram, side) = if rows >= cols {
            (gram_of_columns(&x, rows, cols), Side::Right)
        } else {
            (gram_of_rows(&x, rows, cols), Side::Left)
        };
        let squared = square(&gram, side.order(rows, cols));
        x = advance(&x, rows, cols, &gram, &squared, side);
    }
    x
}

/// `XᵀX`（`cols × cols`）：行数不少于列数时的小边。
fn gram_of_columns(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0_f32; cols * cols];
    for row in 0..rows {
        let lane = &x[row * cols..(row + 1) * cols];
        for col in 0..cols {
            let left = lane[col];
            if left == 0.0 {
                continue;
            }
            for inner in 0..cols {
                gram[col * cols + inner] += left * lane[inner];
            }
        }
    }
    gram
}

/// `X Xᵀ`（`rows × rows`）：列数多于行数时的小边。
fn gram_of_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0_f32; rows * rows];
    for outer in 0..rows {
        for inner in 0..rows {
            let mut acc = 0.0_f32;
            for col in 0..cols {
                acc += x[outer * cols + col] * x[inner * cols + col];
            }
            gram[outer * rows + inner] = acc;
        }
    }
    gram
}

/// 方阵自乘 `M·M`（行主序）。Gram 对称且阶数是小边，直接写三重循环。
fn square(matrix: &[f32], order: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; order * order];
    for row in 0..order {
        for inner in 0..order {
            let left = matrix[row * order + inner];
            if left == 0.0 {
                continue;
            }
            for col in 0..order {
                out[row * order + col] += left * matrix[inner * order + col];
            }
        }
    }
    out
}

/// Gram 乘在 `X` 的哪一边：小边是列时乘在右边（`XA`），小边是行时乘在左边（`AX`）。
#[derive(Clone, Copy)]
enum Side {
    /// `A` 是 `rows × rows`（小边是行），乘在 `X` 左边。
    Left,
    /// `A` 是 `cols × cols`（小边是列），乘在 `X` 右边。
    Right,
}

impl Side {
    /// 小边的阶数：Gram 与 `A²` 都是这个阶数的方阵。
    fn order(self, rows: usize, cols: usize) -> usize {
        match self {
            Side::Left => rows,
            Side::Right => cols,
        }
    }
}

/// 一步 `X ← aX + b(AX) + c(A²X)`；`side` 为 [`Side::Right`] 时是 `X ← aX + b(XA) + c(XA²)`。
/// 两者遍历的是同一块 `X`，只是 `A` 乘在另一侧：沿哪条轴收缩、Gram 槽位怎么排跟着 `side` 变。
fn advance(
    x: &[f32],
    rows: usize,
    cols: usize,
    gram: &[f32],
    squared: &[f32],
    side: Side,
) -> Vec<f32> {
    let (a, b, c) = NS_COEF;
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = a * x[row * cols + col];
            for inner in 0..side.order(rows, cols) {
                let (value, slot) = match side {
                    Side::Left => (x[inner * cols + col], row * rows + inner),
                    Side::Right => (x[row * cols + inner], inner * cols + col),
                };
                acc += b * value * gram[slot] + c * value * squared[slot];
            }
            out[row * cols + col] = acc;
        }
    }
    out
}
