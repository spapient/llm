//! β 指数梯形扫描语义（§5.6，数学验证 5）：幺半群前缀扫描 + 对角 SSM 半群离散化。
//!
//! 与 [`crate::reduce`] 同属「算子语义/骨架」层，不注册成内核算子：`cutile` 的 `Skeleton::Scan`
//! 执行路径尚未接线（`execution.rs` 对 Scan 返回错误）。本模块提供扫描的**第二份实现**——半群
//! 结合律、串行金标、Blelloch 并行扫描与指数梯形离散化——作为未来 `scan_f32` 算子的标量参考。
//!
//! 分层：[`Monoid`] 只约定「结合律 + 幺元」，[`scan_serial`] / [`scan_blelloch`] 不认识任何 SSM
//! 语义；[`SsmStep`] 是那个幺半群的具体实例（对角 `A`，每步 `N` 次乘加）；[`exp_zero_order`]
//! 把连续 `h'=λh+bx` 按步内输入取常数离散成 [`SsmStep`] 序列。
//!
//! [`PlaneStep`] 是同一套算术的复值形态：`λ` 带虚部时状态不再是实标量，而是一对实值分量
//! （旋转 + 衰减的 2×2 块），离散化由 [`exp_trapezoid`] 按双抽头梯形给出——它和
//! [`exp_zero_order`] 是两条**不同**的式子（后者步内输入取常数，前者用 `x_{t−1}` 与 `x_t` 的平均），
//! 名字按各自的实际算式取，不互相借用。

use core::fmt::Debug;

/// 浮点标量闭集：`f32` / `f64` / `half::f16` / `half::bf16` 满足；`nn` 的脉冲 / 合成 / 决策头复用本 trait。
///
/// 运算符超 trait（`Add/Sub/Mul/Div`）让上层代码能写 `a + b`、`a * b` 而不再逐方法调用，
/// 否则「泛型化」只把 `as f32` 换成了 `S::mul`，读起来反而更绕。
pub trait Scalar:
    Copy
    + PartialEq
    + PartialOrd
    + Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    /// 加法幺元。
    const ZERO: Self;
    /// 乘法幺元。
    const ONE: Self;
    /// 负无穷（DSA top-k 掩码：`softmax(−∞)=0`，实现真稀疏注意力）。
    const NEG_INFINITY: Self;

    /// 自然指数 `exp(x)`。
    fn exp(self) -> Self;
    /// `exp(x) - 1`（无 `x≈0` 时的灾难性消去）。
    fn exp_m1(self) -> Self;
    /// 平方根。
    fn sqrt(self) -> Self;
    /// 绝对值。
    fn abs(self) -> Self;
    /// 取两数较大者。
    fn max(self, rhs: Self) -> Self;
    /// 取两数较小者。
    fn min(self, rhs: Self) -> Self;
    /// 向下取整（池化/插值的坐标下取整）。
    fn floor(self) -> Self;
    /// 自然对数（自信息 `−ln p` 与交叉熵的唯一来源，不另写近似）。
    fn ln(self) -> Self;
    /// 正弦（复值状态的相位，单位弧度）。
    fn sin(self) -> Self;
    /// 余弦（复值状态的相位，单位弧度）。
    fn cos(self) -> Self;
    /// 由 `f64` 字面量构造（仅用于常量，`f32` 下可能有损）。
    fn from_f64(v: f64) -> Self;
    /// 由 `usize` 构造：形状/索引数学（步长、核尺寸、面积）的通用入口，不绕道 `f64`。
    fn from_usize(v: usize) -> Self;
    /// 取回 `usize`（截断向零）：只用于形状/索引数学，调用方保证值非负且落在 `usize` 范围内。
    fn to_usize(self) -> usize;
    /// 就近取整，半分远离零（`f64::round` 的口径）。
    ///
    /// 量化到码位要用它：截断会把分布整体压向零，于是码表不是以零为中心的对称表，
    /// `code·scale` 还原出来的偏差带符号。
    fn round(self) -> Self;
    /// 取回 `f64`。
    ///
    /// 只有两类用途：**量化边界**（浮点值 → 整数码位）与**实测指标**（稀疏度、相对重构损失、
    /// 放大比）。前向算术里出现它就意味着精度被悄悄降级成 `f64`（`f32` 路径上尤其如此）。
    fn to_f64(self) -> f64;
}

/// 一份算术骨架为 `f32` / `f64` / `half::f16` / `half::bf16` 四个标量实现，能变的部分只有类型、
/// 零/一/负无穷常量与 `f64`/`usize` 的进出转换式；逐字写四份会把 14 个方法的加/减/乘/除与
/// 超越函数复制成四胞胎（jscpd 的克隆源）。数学函数统一走 [`num_traits::Float`]：`f32`/`f64`
/// 由该 trait 直接委托给原生实现，`half` 类型由 half 自带实现经 `f32` 往返得到，语义与精度由
/// half 侧保证，`op` 侧不复制任何一份算术。
macro_rules! impl_scalar {
    ($ty:ty, $zero:expr, $one:expr, $neg_inf:expr, $from_f64:expr, $from_usize:expr, $to_f64:expr, $to_usize:expr) => {
        impl Scalar for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const NEG_INFINITY: Self = $neg_inf;

            fn exp(self) -> Self {
                num_traits::Float::exp(self)
            }
            fn exp_m1(self) -> Self {
                num_traits::Float::exp_m1(self)
            }
            fn sqrt(self) -> Self {
                num_traits::Float::sqrt(self)
            }
            fn abs(self) -> Self {
                num_traits::Float::abs(self)
            }
            fn max(self, rhs: Self) -> Self {
                num_traits::Float::max(self, rhs)
            }
            fn min(self, rhs: Self) -> Self {
                num_traits::Float::min(self, rhs)
            }
            fn floor(self) -> Self {
                num_traits::Float::floor(self)
            }
            fn ln(self) -> Self {
                num_traits::Float::ln(self)
            }
            fn sin(self) -> Self {
                num_traits::Float::sin(self)
            }
            fn cos(self) -> Self {
                num_traits::Float::cos(self)
            }
            fn from_f64(v: f64) -> Self {
                ($from_f64)(v)
            }
            fn from_usize(v: usize) -> Self {
                ($from_usize)(v)
            }
            fn round(self) -> Self {
                num_traits::Float::round(self)
            }
            fn to_f64(self) -> f64 {
                ($to_f64)(self)
            }
            fn to_usize(self) -> usize {
                ($to_usize)(self)
            }
        }
    };
}

impl_scalar!(
    f32,
    0.0_f32,
    1.0_f32,
    f32::NEG_INFINITY,
    |v| v as f32,
    |v| v as f32,
    |v| v as f64,
    |v| v as usize
);
impl_scalar!(
    f64,
    0.0_f64,
    1.0_f64,
    f64::NEG_INFINITY,
    |v| v,
    |v| v as f64,
    |v| v,
    |v| v as usize
);
impl_scalar!(
    half::f16,
    half::f16::ZERO,
    half::f16::ONE,
    half::f16::NEG_INFINITY,
    half::f16::from_f64,
    |v| half::f16::from_f64(v as f64),
    half::f16::to_f64,
    |v| half::f16::to_f64(v) as usize
);
impl_scalar!(
    half::bf16,
    half::bf16::ZERO,
    half::bf16::ONE,
    half::bf16::NEG_INFINITY,
    half::bf16::from_f64,
    |v| half::bf16::from_f64(v as f64),
    half::bf16::to_f64,
    |v| half::bf16::to_f64(v) as usize
);

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
///
/// 输出 [`SsmStep`]：`ā = exp(λ·dt)`，`b̄ = expm1(λ·dt)/λ · b`；`λ = 0` 时取极限 `b̄ = dt·b`。
/// `expm1` 的写法只为消 `λ·dt ≈ 0` 的灾难性消去，公式本身仍是零阶保持；`h'` 的**双抽头**梯形是
/// 另一条式子，住 [`exp_trapezoid`]——两条离散化的名字按各自算式取，不互相借用。
///
/// 对 `d×N` 状态逐维调用即可得到整层一步的 `d·N` 个半群步。
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
///
/// `a` 是复乘子 `ā = |ā|·(cos φ + i·sin φ)`，`b` 是复输入系数；`select` 是实值选择量 `B·x`。
/// 复值状态不必引入复数类型：任何带虚部 `λ` 的 SSM 都能写成一族这样的 2×2 块，这也是 Mamba-3
/// 「状态宽减半、同 PPL」那条实证红利的实值实现形态（§5.6）。
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
///
/// 连续方程 `h' = λ h + B x` 在 `[t−1, t]` 上按梯形求积：`h_t = ā·h_{t−1} + (dt/2)(ā·B x_{t−1} + B x_t)`，
/// 其中 `ā = exp(λ·dt)`，于是输入系数是复值 `b̄ = (dt/2)·(ā·x_{t−1} + x_t)`——`B` 留在调用方
/// （它是逐状态维的实值选择量，不是这一层的时间离散化）。
///
/// `ω = 0` 时 `ā` 退化为实衰减、`b̄` 的虚部为零，`PlaneStep` 与 [`exp_zero_order`] 的实值口径同形；
/// 但两条式子**数值不同**（梯形用两端平均，零阶保持用左端常数），别把其中一个当成另一个的等价物。
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

    /// M1 log-log 斜率实测（`#[ignore]`：
    /// `cargo test -p llm-op --lib -- scan_loglog_slope --release --ignored --nocapture`）。
    ///
    /// 口径与 §10.10 M1 卡一致：n ∈ {4K..128K}，每点多测取最小值，对 (ln n, ln T) 做最小二乘，
    /// 报斜率 k 与其 95% CI。产品串行核 `scan_f32::forward` 与第二份实现 `scan_blelloch` 各测一条。
    #[test]
    #[ignore = "M1 复杂度基准：cargo test -p llm-op --lib -- scan_loglog_slope --release --ignored --nocapture"]
    fn scan_loglog_slope() {
        let ns = [4096usize, 8192, 16384, 32768, 65536, 131072];
        let reps = 30usize;

        // 确定性输入（xorshift64）。
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next_f32 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as u32 as f32 / (u32::MAX as f32)) * 2.0 - 1.0
        };

        // 产品串行核：s = 0.5·s + 1.0·x[t]。
        let mut fwd_points = Vec::new();
        for &n in &ns {
            let x: Vec<f32> = (0..n).map(|_| next_f32()).collect();
            let mut out = vec![0.0f32; n];
            for _ in 0..3 {
                super::scan_f32::forward(&x, &mut out).expect("等长");
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = std::time::Instant::now();
                super::scan_f32::forward(&x, &mut out).expect("等长");
                best = best.min(t.elapsed().as_secs_f64());
            }
            fwd_points.push((n as f64, best));
        }
        let (k, ci, r2) = loglog_fit(&fwd_points);
        println!(
            "M1 scan_f32::forward 斜率 k={k:.4} 95%CI=[{:.4},{:.4}] R²={r2:.6} 点={:?}",
            ci.0,
            ci.1,
            fwd_points
                .iter()
                .map(|(n, t)| format!("{n:.0}:{t:.3e}s"))
                .collect::<Vec<_>>()
        );

        // 半群 Blelloch 并行扫描（f32）。
        let mut ble_points = Vec::new();
        for &n in &ns {
            let steps: Vec<SsmStep<f32>> = (0..n)
                .map(|_| SsmStep {
                    a: 0.5,
                    b: next_f32(),
                })
                .collect();
            for _ in 0..3 {
                std::hint::black_box(scan_blelloch(&steps));
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = std::time::Instant::now();
                std::hint::black_box(scan_blelloch(&steps));
                best = best.min(t.elapsed().as_secs_f64());
            }
            ble_points.push((n as f64, best));
        }
        let (k, ci, r2) = loglog_fit(&ble_points);
        println!(
            "M1 scan_blelloch       斜率 k={k:.4} 95%CI=[{:.4},{:.4}] R²={r2:.6} 点={:?}",
            ci.0,
            ci.1,
            ble_points
                .iter()
                .map(|(n, t)| format!("{n:.0}:{t:.3e}s"))
                .collect::<Vec<_>>()
        );
    }

    /// (ln n, ln T) 最小二乘：返回 (斜率 k, 95% CI, R²)。
    fn loglog_fit(points: &[(f64, f64)]) -> (f64, (f64, f64), f64) {
        let xs: Vec<f64> = points.iter().map(|(n, _)| n.ln()).collect();
        let ys: Vec<f64> = points.iter().map(|(_, t)| t.ln()).collect();
        let n = xs.len() as f64;
        let (xbar, ybar) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let sxx = xs.iter().map(|x| (x - xbar) * (x - xbar)).sum::<f64>();
        let sxy = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - xbar) * (y - ybar))
            .sum::<f64>();
        let k = sxy / sxx;
        let intercept = ybar - k * xbar;
        let residuals: Vec<f64> = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| y - (intercept + k * x))
            .collect();
        let sse = residuals.iter().map(|r| r * r).sum::<f64>();
        let sst = ys.iter().map(|y| (y - ybar) * (y - ybar)).sum::<f64>();
        let r2 = 1.0 - sse / sst;
        // 斜率标准误；df = 点数 - 2，n=6 时 t(0.975,4)=2.776。
        let se = (sse / (n - 2.0) / sxx).sqrt();
        let tcrit = 2.776;
        (k, (k - tcrit * se, k + tcrit * se), r2)
    }

    /// `λ = 0` 时退化为 `ā = 1`、`b̄ = dt·b`。
    #[test]
    fn exp_zero_order_zero_lambda() {
        let s = exp_zero_order(0.0f64, 2.0, 0.5);
        assert_eq!(s.a, 1.0);
        assert_eq!(s.b, 1.0);
    }

    /// 零阶保持与直接 `exp` / `exp_m1` 公式一致。
    #[test]
    fn exp_zero_order_matches_direct() {
        let (lam, b, dt) = (-0.5f64, 3.0, 0.25);
        let s = exp_zero_order(lam, b, dt);
        let expected_a = (lam * dt).exp();
        let expected_b = (lam * dt).exp_m1() / lam * b;
        assert!((s.a - expected_a).abs() < 1e-12);
        assert!((s.b - expected_b).abs() < 1e-12);
    }

    /// 幺元块不改状态。
    #[test]
    fn plane_identity_leaves_state() {
        assert_eq!(PlaneStep::identity().apply([0.3, -0.8], 2.5), [0.3, -0.8]);
    }

    /// `apply` 就是复数乘加：`ā·s + b̄·select`。
    #[test]
    fn plane_apply_is_complex_arithmetic() {
        let (a_re, a_im, b_re, b_im) = (0.6f64, 0.8, -0.2, 0.5);
        let s = PlaneStep {
            a: [a_re, a_im],
            b: [b_re, b_im],
        };
        let (re, im) = (0.7f64, -1.1);
        let select = 2.0f64;
        let [got_re, got_im] = s.apply([re, im], select);
        assert!((got_re - (a_re * re - a_im * im + b_re * select)).abs() < 1e-12);
        assert!((got_im - (a_im * re + a_re * im + b_im * select)).abs() < 1e-12);
    }

    /// 旋转只改相位、不改模长：任意 `ω` 下 `|ā| = exp(λ_re·dt)`。
    #[test]
    fn rotation_preserves_magnitude() {
        let dt = 0.125f64;
        let expected = (-0.5 * dt).exp();
        for omega in [0.0, 0.3, 1.7, -2.9] {
            let [re, im] = exp_trapezoid(-0.5f64, omega, dt, 1.0, 1.0).a;
            assert!((re.hypot(im) - expected).abs() < 1e-12, "ω={omega}");
        }
    }

    /// `ω = 0` 时块退化为实衰减，输入系数的虚部为零。
    #[test]
    fn zero_frequency_is_real() {
        let s = exp_trapezoid(-0.4f64, 0.0, 0.25, 0.5, 1.0);
        assert_eq!(s.a[1], 0.0);
        assert_eq!(s.b[1], 0.0);
        let expected_b = 0.25 / 2.0 * ((-0.4 * 0.25f64).exp() * 0.5 + 1.0);
        assert!((s.b[0] - expected_b).abs() < 1e-12);
    }

    /// 双抽头梯形与零阶保持是两条式子：同样的输入在本步给出不同的状态。
    #[test]
    fn trapezoid_and_zero_order_differ() {
        let (lam, dt) = (-0.6f64, 0.3);
        let order = exp_zero_order(lam, 1.0, dt).apply(0.0);
        // 零阶保持把这一步的输入铺满整段步长，梯形只铺两端平均：从零状态起跳时差近一倍。
        let half = exp_trapezoid(lam, 0.0, dt, 0.0, 1.0).apply([0.0, 0.0], 1.0)[0];
        assert!(half < order * 0.6, "{half} vs {order}");
        // 两端相等时两条式子仍然不同——步内线性与常数保持只在 `dt → 0` 的极限重合。
        let flat = exp_trapezoid(lam, 0.0, dt, 1.0, 1.0).apply([0.0, 0.0], 1.0)[0];
        assert!((flat - order).abs() > 1e-4, "{flat} vs {order}");
    }

    /// 梯形真的用了上一个抽头：`x_{t−1}` 不同，本步状态就不同。
    #[test]
    fn trapezoid_uses_the_previous_tap() {
        let (lam, omega, dt) = (-0.5f64, 0.9, 0.2);
        let a = exp_trapezoid(lam, omega, dt, 0.0, 1.0).apply([0.0, 0.0], 1.0);
        let b = exp_trapezoid(lam, omega, dt, 1.0, 1.0).apply([0.0, 0.0], 1.0);
        assert!(a[0] != b[0] || a[1] != b[1]);
    }
}

pub mod scan_f32 {
    //! `scan_f32`：β 指数梯形扫描（§5.6）的离散递推 `s_t = a·s_{t-1} + b·x_t`。
    //!
    //! 连续动力学 `h' = λ h + b x` 经 [`crate::scan::exp_zero_order`] 离散成对角半群步 `(a, b)`；
    //! 本算子把这两个离散系数固定为模板常量，内核沿时间轴串行递推。扫描的**第二份实现**（幺半群
    //! 前缀、Blelloch 并行扫描、结合律与金标验证）住在 [`crate::scan`]，本模块只声明这份算子的
    //! 形状、代价、样本与标量参考，`#[annotation::model]` 装配成 `MODEL`。

    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
    use diagnostic::{Code, Error};

    // 单轴定义与形状投影复用 unary 骨架：时间轴 `n` 既是输入轴也是输出轴。
    pub use crate::unary::{AXES, project_shape};

    /// 对角衰减系数 `ā = exp(λ·dt)`（模板常量 `{{a}}`）。
    const DECAY: f32 = 0.5;
    /// 输入系数 `b̄ = expm1(λ·dt)/λ · b`（模板常量 `{{b}}`）。
    const INPUT: f32 = 1.0;

    /// 标量代价：每步一次乘加（2 flop）、读一次 `x`、写一次 `y`。
    ///
    /// # Errors
    ///
    /// 形状缺轴 `n` 时返回说明。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        cost_n(shape, "scan", 2.0, 1)
    }

    /// 确定性样本：`x`、`y` 两个等长缓冲，标量 `N` 是时间步数。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            buffers: vec![
                x_buffer(len),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
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
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
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
        use super::*;

        #[test]
        fn scan_matches_the_scalar_reference() {
            let sample = sample(128, 1);
            let expected = reference(&sample).expect("参考可计算");
            let mut out = vec![0.0; 128];
            forward(sample.f32("x").expect("x"), &mut out).expect("等长");
            assert_eq!(out, expected);
        }

        /// 输入与输出长度不一致时拒绝；非对齐长度不静默截断。
        #[test]
        fn scan_rejects_mismatched_output_length() {
            let mut out = vec![0.0; 3];
            assert!(
                forward(&[1.0, 2.0], &mut out).is_err(),
                "输出比输入长必须拒绝"
            );
            let mut short = vec![0.0; 1];
            assert!(
                forward(&[1.0, 2.0, 3.0], &mut short).is_err(),
                "输出比输入短必须拒绝"
            );
        }

        /// 递推初态为零：`y[t] = Σ_{k≤t} ā^(t-k)·b̄·x[k]`。取 `x = [1, 0, 0, …]` 得几何衰减
        /// `1, ā, ā², …`，逐位验证半群语义与内核公式一致。
        #[test]
        fn scan_is_geometric_decay_for_impulse() {
            let x = [1.0f32, 0.0, 0.0, 0.0, 0.0];
            let expected = [1.0f32, 0.5, 0.25, 0.125, 0.0625];
            let mut out = vec![0.0; x.len()];
            forward(&x, &mut out).expect("等长");
            for (index, (got, want)) in out.iter().zip(expected).enumerate() {
                assert!((got - want).abs() < 1e-6, "第 {index} 步：{got} vs {want}");
            }
        }

        /// 空序列放行：空输入得空输出，扫描没有无定义的中间态。
        #[test]
        fn scan_allows_empty_sequence() {
            let mut out = vec![];
            forward(&[], &mut out).expect("空序列放行");
            assert_eq!(reference(&sample(0, 1)).expect("空参考"), Vec::<f32>::new());
        }

        /// M11 因果前缀不变性：给同一条输入追加未来 token 后，已生成前缀 `1..t` 的输出必须逐位不变。
        ///
        /// 串行递推的算术对前缀是确定且相同的，因此这里是**严格 0 偏差**（不是容差比较），
        /// 对应 §10.10 M11 卡「未来 token 改变不得影响前缀输出」。
        #[test]
        fn m11_prefix_is_invariant_to_future_tokens() {
            let mut seed = 0xfeed_beefu64;
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as u32 as f32 / (u32::MAX as f32)) * 2.0 - 1.0
            };
            let full: Vec<f32> = (0..256).map(|_| next()).collect();
            for t in [1usize, 7, 31, 64, 127, 255] {
                let mut prefix_out = vec![0.0f32; t];
                forward(&full[..t], &mut prefix_out).expect("等长");
                let mut full_out = vec![0.0f32; full.len()];
                forward(&full, &mut full_out).expect("等长");
                assert_eq!(
                    &full_out[..t],
                    prefix_out.as_slice(),
                    "前缀 {t} 输出因未来 token 改变"
                );
            }
        }
    }
}
