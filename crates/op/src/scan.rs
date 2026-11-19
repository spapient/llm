//! β 指数梯形扫描语义（§5.6，数学验证 5）：幺半群前缀扫描 + 对角 SSM 半群离散化。
//!
//! 与 [`crate::reduce`] 同属「算子语义/骨架」层，不注册成内核算子：`cutile` 的 `Skeleton::Scan`
//! 执行路径尚未接线（`execution.rs` 对 Scan 返回错误）。本模块提供扫描的**第二份实现**——半群
//! 结合律、串行金标、Blelloch 并行扫描与指数梯形离散化——作为未来 `scan_f32` 算子的标量参考。
//!
//! 分层：[`Monoid`] 只约定「结合律 + 幺元」，[`scan_serial`] / [`scan_blelloch`] 不认识任何 SSM
//! 语义；[`SsmStep`] 是那个幺半群的具体实例（对角 `A`，每步 `N` 次乘加）；[`exp_trapezoid`]
//! 把连续 `h'=λh+bx` 离散成 [`SsmStep`] 序列。复值 2×2 块旋转（Mamba-3 实值扩展）在后续阶段接入。

use core::fmt::Debug;

/// 浮点标量闭集：`f32` / `f64` 满足；`nn` 的脉冲 / 合成 / 决策头复用本 trait。
pub trait Scalar: Copy + PartialEq + PartialOrd + Debug {
    /// 加法幺元。
    const ZERO: Self;
    /// 乘法幺元。
    const ONE: Self;
    /// 负无穷（DSA top-k 掩码：`softmax(−∞)=0`，实现真稀疏注意力）。
    const NEG_INFINITY: Self;

    /// 加法。
    fn add(self, rhs: Self) -> Self;
    /// 减法。
    fn sub(self, rhs: Self) -> Self;
    /// 乘法。
    fn mul(self, rhs: Self) -> Self;
    /// 除法。
    fn div(self, rhs: Self) -> Self;
    /// 自然指数 `exp(x)`。
    fn exp(self) -> Self;
    /// `exp(x) - 1`（无 `x≈0` 时的灾难性消去）。
    fn exp_m1(self) -> Self;
    /// 平方根。
    fn sqrt(self) -> Self;
    /// 绝对值。
    fn abs(self) -> Self;
    /// 由 `f64` 字面量构造（仅用于常量，`f32` 下可能有损）。
    fn from_f64(v: f64) -> Self;
}

/// 一份算术骨架为 `f32` / `f64` 两个标量实现，能变的部分只有类型、负无穷与 `f64` 降级式；
/// 逐字写两份会把 9 个方法的加/减/乘/除复制成孪生（jscpd 的 15 行克隆源）。
macro_rules! impl_scalar {
    ($ty:ty, $neg_inf:expr, $from_f64:expr) => {
        impl Scalar for $ty {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const NEG_INFINITY: Self = $neg_inf;

            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            fn div(self, rhs: Self) -> Self {
                self / rhs
            }
            fn exp(self) -> Self {
                <$ty>::exp(self)
            }
            fn exp_m1(self) -> Self {
                <$ty>::exp_m1(self)
            }
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            fn from_f64(v: f64) -> Self {
                ($from_f64)(v)
            }
        }
    };
}

impl_scalar!(f32, f32::NEG_INFINITY, |v| v as f32);
impl_scalar!(f64, f64::NEG_INFINITY, |v| v);

/// 幺半群：`combine` 满足结合律，[`Monoid::IDENT`] 是幺元。
///
/// 时间序：`self.combine(next)` 表示「先施加 `self`、再施加 `next`」的合成变换。
pub trait Monoid: Clone {
    /// 幺元：`IDENT.combine(x)` 与 `x.combine(IDENT)` 都等于 `x`。
    const IDENT: Self;
    /// 结合合成（⊕）。
    fn combine(self, next: Self) -> Self;
}

/// 串行前缀（金标，§5.10 逐位一致）：`out[i]` = 前 `i` 个元素的合成，`out[0] = IDENT`。
///
/// 结果长度为 `items.len()`；用 `out[t].apply(init)` 得第 `t` 步状态。
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
///
/// 工作 `O(n)`、跨度 `O(log n)`；内部把长度补齐到 2 的幂，空位填 [`Monoid::IDENT`]。
/// 数学上与串行前缀一致；浮点下因合成顺序改写产生 ulp 级差异（§5.10 并行归约容差）。
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
///
/// 合成 `⊕`：`(a₁,b₁)⊕(a₂,b₂) = (a₂a₁, a₂b₁+b₂)`（数学验证 5a 结合律）。`IDENT = (1, 0)`；
/// `apply(s) = a·s + b`。对角 `A` 时 `a`、`b` 与状态 `s` 同长，逐维独立乘加（§5.6）。
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
        self.a.mul(state).add(self.b)
    }
}

impl<S: Scalar> Monoid for SsmStep<S> {
    const IDENT: Self = SsmStep {
        a: S::ONE,
        b: S::ZERO,
    };

    fn combine(self, next: Self) -> Self {
        SsmStep {
            a: next.a.mul(self.a),
            b: next.a.mul(self.b).add(next.b),
        }
    }
}

/// 指数梯形离散化（对角 `A`，§5.6）：连续 `h' = λ h + b x`，步长 `dt`，步内输入取常数（ZOH）。
///
/// 输出 [`SsmStep`]：`ā = exp(λ·dt)`，`b̄ = expm1(λ·dt)/λ · b`；`λ = 0` 时取极限 `b̄ = dt·b`。
/// 对 `d×N` 状态逐维调用即可得到整层一步的 `d·N` 个半群步。
pub fn exp_trapezoid<S: Scalar>(lam: S, b: S, dt: S) -> SsmStep<S> {
    let a = lam.mul(dt).exp();
    let b = if lam == S::ZERO {
        dt.mul(b)
    } else {
        lam.mul(dt).exp_m1().div(lam).mul(b)
    };
    SsmStep { a, b }
}

/// 沿半群步序列推进状态，返回各步状态（`out[t]` = 施加前 `t` 步后的状态，`out[0] = init`）。
///
/// 等价于对 [`scan_serial`] / [`scan_blelloch`] 的前缀逐一 [`SsmStep::apply`]；语义参考实现。
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
    use super::*;

    /// f64 下半群合成因浮点非结合律产生 ulp 级差异，用远高于 ulp、远低于有意义误差的容差比较。
    const TOL: f64 = 1e-9;

    fn assert_step_close(a: SsmStep<f64>, b: SsmStep<f64>) {
        assert!((a.a - b.a).abs() <= TOL, "a 偏差：{} vs {}", a.a, b.a);
        assert!((a.b - b.b).abs() <= TOL, "b 偏差：{} vs {}", a.b, b.b);
    }

    fn assert_seq_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "序列长度不一致");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= TOL, "第 {i} 项偏差：{x} vs {y}");
        }
    }

    /// 确定性 xorshift64（测试内无外部 RNG 依赖）。
    struct Xs(u64);

    impl Xs {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// 高 53 位归一化到 `[0,1)`。
        fn next_f64(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// 随机半群步：`a ∈ (0.01, 0.99)` 保证衰减稳定，`b ∈ [-1,1)`。
    fn step(xs: &mut Xs) -> SsmStep<f64> {
        SsmStep {
            a: 0.01 + 0.98 * xs.next_f64(),
            b: 2.0 * xs.next_f64() - 1.0,
        }
    }

    /// 数学验证 5a：半群合成满足结合律（f64 下允许 ulp 级差异）。
    #[test]
    fn combine_is_associative() {
        let mut xs = Xs(0x9e37_79b9_7f4a_7c15);
        for _ in 0..1000 {
            let (s1, s2, s3) = (step(&mut xs), step(&mut xs), step(&mut xs));
            let left = s1.combine(s2).combine(s3);
            let right = s1.combine(s2.combine(s3));
            assert_step_close(left, right);
        }
    }

    /// 幺元是合成中性元。
    #[test]
    fn identity_is_neutral() {
        let mut xs = Xs(1);
        for _ in 0..100 {
            let s = step(&mut xs);
            assert_eq!(s.combine(SsmStep::identity()), s, "右乘幺元不变");
            assert_eq!(SsmStep::identity().combine(s), s, "左乘幺元不变");
        }
    }

    /// 数学验证 5b：Blelloch 非交换独占扫描与串行前缀在 64 点全一致（§5.10 容差）。
    #[test]
    fn blelloch_matches_serial_64() {
        let mut xs = Xs(2);
        let steps: Vec<_> = (0..64).map(|_| step(&mut xs)).collect();
        let serial = scan_serial(&steps);
        let parallel = scan_blelloch(&steps);
        assert_eq!(serial.len(), parallel.len());
        for (a, b) in serial.into_iter().zip(parallel) {
            assert_step_close(a, b);
        }
    }

    /// 前缀逐一 `apply` 与逐步递推一致（f64 下允许 ulp 级差异）。
    #[test]
    fn apply_matches_recurrence() {
        let mut xs = Xs(3);
        let steps: Vec<_> = (0..37).map(|_| step(&mut xs)).collect();
        let init = xs.next_f64();
        let expected = apply_scan(&steps, init);
        // 独占前缀给前 n 个状态，末态用全量合成一次施加补齐。
        let mut got: Vec<_> = scan_blelloch(&steps)
            .iter()
            .map(|p| p.apply(init))
            .collect();
        let total = steps
            .iter()
            .fold(SsmStep::identity(), |acc, s| acc.combine(*s));
        got.push(total.apply(init));
        assert_seq_close(&expected, &got);
    }

    /// 半群全量合成一次施加 == 逐步递推的末态。
    #[test]
    fn scan_matches_semigroup_recurrence() {
        let mut xs = Xs(4);
        let steps: Vec<_> = (0..50).map(|_| step(&mut xs)).collect();
        let init = 0.7;
        let mut state = init;
        for s in &steps {
            state = s.apply(state);
        }
        let mut total = SsmStep::identity();
        for s in &steps {
            total = total.combine(*s);
        }
        assert!((total.apply(init) - state).abs() < 1e-12);
    }

    /// `λ = 0` 时退化为 `ā = 1`、`b̄ = dt·b`。
    #[test]
    fn exp_trapezoid_zero_lambda() {
        let s = exp_trapezoid(0.0f64, 2.0, 0.5);
        assert_eq!(s.a, 1.0);
        assert_eq!(s.b, 1.0);
    }

    /// 离散化与直接 `exp` / `exp_m1` 公式一致。
    #[test]
    fn exp_trapezoid_matches_direct() {
        let (lam, b, dt) = (-0.5f64, 3.0, 0.25);
        let s = exp_trapezoid(lam, b, dt);
        let expected_a = (lam * dt).exp();
        let expected_b = (lam * dt).exp_m1() / lam * b;
        assert!((s.a - expected_a).abs() < 1e-12);
        assert!((s.b - expected_b).abs() < 1e-12);
    }
}
