//! β 指数梯形扫描语义（§5.6，数学验证 5）：幺半群前缀扫描 + 对角 SSM 半群离散化。

pub use crate::Scalar;

/// 幺半群：`combine` 满足结合律，[`Monoid::IDENT`] 是幺元。
pub trait Monoid: Clone {
    /// 幺元：`IDENT.combine(x)` 与 `x.combine(IDENT)` 都等于 `x`。
    const IDENT: Self;
    /// 结合合成（⊕）。
    fn combine(self, next: Self) -> Self;
}

/// 串行前缀（金标，§5.10 逐位一致）：`out[i]` = 前 `i` 个元素的合成，`out[0] = IDENT`。
pub fn scan_serial<M: Monoid>(items: &[M]) -> Vec<M> {
    let mut out = Vec::with_capacity(items.len());
    let mut acc = M::IDENT;
    for item in items {
        out.push(acc.clone());
        acc = acc.combine(item.clone());
    }
    out
}

/// Blelloch 非交换独占扫描（数学验证 5b：≡ [`scan_serial`]）。
pub fn scan_blelloch<M: Monoid>(items: &[M]) -> Vec<M> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let m = n.next_power_of_two();
    let mut buf = vec![M::IDENT; m];
    buf[..n].clone_from_slice(items);
    let d = up_sweep(&mut buf, m);
    down_sweep(&mut buf, m, d);
    buf.truncate(n);
    buf
}

/// up-sweep：归并成二叉树，`buf[right] = buf[left] ⊕ buf[right]`（左在前，保持时序），返回层数。
fn up_sweep<M: Monoid>(buf: &mut [M], m: usize) -> usize {
    let mut d = 0usize;
    while (1usize << d) < m {
        let stride = 1usize << (d + 1);
        let mut k = 0usize;
        while k < m {
            let left = k + (1usize << d) - 1;
            let right = k + stride - 1;
            buf[right] = buf[left].clone().combine(buf[right].clone());
            k += stride;
        }
        d += 1;
    }
    d
}

/// down-sweep：把「本段之前的合成」从根向下传播。
fn down_sweep<M: Monoid>(buf: &mut [M], m: usize, mut d: usize) {
    buf[m - 1] = M::IDENT;
    while d > 0 {
        d -= 1;
        let stride = 1usize << (d + 1);
        let mut k = 0usize;
        while k < m {
            let left = k + (1usize << d) - 1;
            let right = k + stride - 1;
            let before = buf[right].clone();
            let left_val = buf[left].clone();
            buf[left] = before.clone();
            buf[right] = before.combine(left_val);
            k += stride;
        }
    }
}

/// 对角 SSM 半群步：状态推进 `s' = a·s + b`（逐元素），是 [`Monoid`] 的实例。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsmStep<S> {
    /// 对角衰减系数（逐维）。
    pub a: S,
    /// 加性偏置（逐维）。
    pub b: S,
}

impl<S: Scalar> SsmStep<S> {
    /// 幺元步（状态不变）。
    pub const fn identity() -> Self {
        Self {
            a: S::ONE,
            b: S::ZERO,
        }
    }

    /// 施加一步：`s' = a·s + b`。
    pub fn apply(self, state: S) -> S {
        self.a * state + self.b
    }
}

impl<S: Scalar> Monoid for SsmStep<S> {
    const IDENT: Self = SsmStep {
        a: S::ONE,
        b: S::ZERO,
    };

    fn combine(self, next: Self) -> Self {
        SsmStep {
            a: next.a * self.a,
            b: next.a * self.b + next.b,
        }
    }
}

/// 零阶保持离散化（对角 `A`，§5.6）：连续 `h' = λ h + b x`，步长 `dt`，步内输入取常数。
pub fn exp_zero_order<S: Scalar>(lam: S, b: S, dt: S) -> SsmStep<S> {
    let a = (lam * dt).exp();
    let b = if lam == S::ZERO {
        dt * b
    } else {
        (lam * dt).exp_m1() / lam * b
    };
    SsmStep { a, b }
}

/// 复值状态的 2×2 实值块：`s' = ā·s + b̄·(B·x)`，`s` 是相邻两个实值分量（实部、虚部）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneStep<S> {
    /// 复乘子 `ā`：`[实部, 虚部]`。
    pub a: [S; 2],
    /// 复输入系数 `b̄`：`[实部, 虚部]`。
    pub b: [S; 2],
}

impl<S: Scalar> PlaneStep<S> {
    /// 幺元步：`ā = 1`、`b̄ = 0`（状态不变）。
    pub const fn identity() -> Self {
        Self {
            a: [S::ONE, S::ZERO],
            b: [S::ZERO, S::ZERO],
        }
    }

    /// 施加一步：`(re', im') = ā·(re, im) + b̄·select`。
    pub fn apply(&self, state: [S; 2], select: S) -> [S; 2] {
        let [a_re, a_im] = self.a;
        let [b_re, b_im] = self.b;
        let [re, im] = state;
        [
            a_re * re - a_im * im + b_re * select,
            a_im * re + a_re * im + b_im * select,
        ]
    }
}

/// 双抽头指数梯形离散化（§5.6，Mamba-3）：`λ = λ_re + i·ω`，输入取 `x_{t−1}` 与 `x_t` 的平均。
pub fn exp_trapezoid<S: Scalar>(lam_re: S, omega: S, dt: S, prev: S, cur: S) -> PlaneStep<S> {
    let decay = lam_re * dt;
    let phase = omega * dt;
    let magnitude = decay.exp();
    let [sin, cos] = [phase.sin(), phase.cos()];
    let [a_re, a_im] = [magnitude * cos, magnitude * sin];
    // (dt/2)·(ā·x_{t−1} + x_t)：`ā·x_{t−1}` 是复值，`x_t` 只进实部。
    let half = dt / (S::ONE + S::ONE);
    let b = [half * (a_re * prev + cur), half * (a_im * prev)];
    PlaneStep { a: [a_re, a_im], b }
}

/// 沿半群步序列推进状态，返回各步状态（`out[t]` = 施加前 `t` 步后的状态，`out[0] = init`）。
pub fn apply_scan<S: Scalar>(steps: &[SsmStep<S>], init: S) -> Vec<S> {
    let mut out = Vec::with_capacity(steps.len() + 1);
    let mut state = init;
    out.push(state);
    for step in steps {
        state = step.apply(state);
        out.push(state);
    }
    out
}

#[cfg(test)]
mod tests {
    include!("../tests/conv_scan_tests.rs");
}

pub mod f32 {
    //! `scan_f32`：β 指数梯形扫描（§5.6）的离散递推 `s_t = a·s_{t-1} + b·x_t`。
    use crate::model::{Cost, Sample, ShapeCtx, cost_n};
    use diagnostic::{Code, Error};
    use tensor::{View, ViewMut};

    // 单轴定义与形状投影复用 unary 骨架：时间轴 `n` 既是输入轴也是输出轴。
    pub use crate::kernel::unary::{AXES, project_shape};

    /// 对角衰减系数 `ā = exp(λ·dt)`（模板常量 `{{a}}`）。
    const DECAY: f32 = 0.5;
    /// 输入系数 `b̄ = expm1(λ·dt)/λ · b`（模板常量 `{{b}}`）。
    const INPUT: f32 = 1.0;

    /// 标量代价：每步一次乘加（2 flop）、读一次 `x`、写一次 `y`。
    ///
    /// # Errors
    ///
    /// 形状缺轴 `n` 时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "scan", 2.0, 1)
    }

    /// 确定性样本：`x`、`y` 两个等长缓冲，标量 `N` 是时间步数。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample::f32_io(len, len, len, len, vec![("N", len as i64)])
    }

    /// 标量参考：`s = ā·s + b̄·x[t]`，初态 `s = 0`，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 样本缺 `x` 缓冲时返回说明。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let mut state = 0.0f32;
        Ok(x.iter()
            .map(|value| {
                state = DECAY * state + INPUT * value;
                state
            })
            .collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出不等长时返回说明。
    pub fn forward(x: View<'_, f32>, out: ViewMut<'_, f32>) -> diagnostic::Result<()> {
        let x = x.as_slice();
        let out = out.as_slice_mut();
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        let mut state = 0.0f32;
        for (out, value) in out.iter_mut().zip(x) {
            state = DECAY * state + INPUT * value;
            *out = state;
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Scan;

    /// β 指数梯形扫描的具体语义别名。
    #[annotation::model(name = "scan_f32", shape = [4096, 4096], dtype = "f32", skeleton = "scan")]
    pub type ScanF32 = Scan;

    crate::operator_here!(Scan, "scan_f32", 1, (4096, 4096), false, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/conv_scan_f32_tests.rs");
    }
}
