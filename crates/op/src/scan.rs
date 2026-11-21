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
#[path = "../tests/scan_tests.rs"]
mod tests;

pub mod f32 {
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
        include!("../tests/scan_f32_tests.rs");
    }
}
