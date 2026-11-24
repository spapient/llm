//! Typed-IR pass framework: named transforms over a [`Kernel`], driven to a fixpoint.
//!
//! 一个 [`Pass`] 只做一件事：把内核变成另一个内核（LLVM 的 pass 模型，规模压到 `#[dsl]` 这一族）。
//! [`Pipeline::run`] 就是全部契约——进来先校验，把注册的 pass 扫到不动点，**每个 pass 之后**
//! 再校验一次；离开管理器的内核在任何构建下都合法，与注册了哪些 pass 无关。
//!
//! 为什么要扫不止一轮：管理器**不要求**调用方按依赖顺序注册。任一 pass 都可能在已经跑过的 pass
//! 左侧造出新的机会，只有下一轮才轮得到它，所以「注册顺序」不该是正确性的前提。轮次不用魔法数：
//! 上限取初始内核的 `cost`（形参数 + 表达式节点数），一个严格收缩的 pass 必然在此之前收敛；
//! 上限耗尽说明流水线在震荡，此时**报错**（MLIR 的 "did not converge"），而不是悄悄截断一段
//! 半成品内核。对自定义 pass 的要求因此只有一条：要么严格收缩，要么幂等。

use crate::kernel::{Arith, Expr, Kernel, VerificationError};

/// One named transform over a kernel.
///
/// A pass may simplify an expression (`Simplify`), shrink the signature
/// (`EliminateDeadParams`), or do anything else a caller needs. It returns the next kernel, or a
/// [`VerificationError`] when the kernel must be rejected.
///
/// 契约只有一条：**要么严格收缩，要么幂等**。管理器会反复扫描直到内核不再变化，每次调用都改
/// 一点的 pass 会被判为不收敛（见 [`Pipeline::run`]）。
pub trait Pass {
    /// Short, stable name used for diagnostics.
    fn name(&self) -> &'static str;

    /// Apply this pass, returning the next kernel or a verification error.
    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError>;
}

/// An ordered list of passes, swept left-to-right to a fixpoint.
#[derive(Default)]
pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl Pipeline {
    /// An empty pipeline; append passes with [`Pipeline::push`].
    pub fn new() -> Self {
        Self::default()
    }

    /// 标准通道：化简（代数定律 + 常量折叠）→ 去死形参。校验由 [`Pipeline::run`] 提供。
    pub fn standard() -> Self {
        Self::new().push(Simplify).push(EliminateDeadParams)
    }

    /// 收缩通道：化简 → 乘加收缩（`a * b + c` → 一条 FMA）→ 去死形参。收缩会改变舍入，因此
    /// 不进入位精确的 [`Pipeline::standard`]，只在作者显式要求融合指令时启用。
    pub fn contracted() -> Self {
        Self::new()
            .push(Simplify)
            .push(FuseFma)
            .push(EliminateDeadParams)
    }

    /// Append one pass and return the manager for chaining.
    pub fn push<P: Pass + 'static>(mut self, pass: P) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Verify the input, then sweep the passes to a fixpoint.
    ///
    /// 轮次上限是初始内核的 `cost`：严格收缩的 pass 每轮至少减一，收敛必然在此之前发生。上限
    /// 耗尽说明某个 pass 既不收缩也不幂等，此时返回错误，而不是把半成品当结果交出去。每个 pass
    /// 之后都会校验，所以「交出的内核合法」由构造保证，调用方不必自己再验一次。
    ///
    /// # Errors
    ///
    /// The kernel or any intermediate kernel fails verification, a pass rejects it, or the pipeline
    /// never settles.
    pub fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        kernel.verify()?;
        self.fixpoint(kernel)
    }

    /// Sweep until a round leaves the kernel unchanged.
    fn fixpoint(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let limit = cost(&kernel);
        let mut kernel = kernel;
        for _ in 0..=limit {
            let (next, changed) = self.sweep(kernel)?;
            if !changed {
                return Ok(next);
            }
            kernel = next;
        }
        let names = self.passes.iter().map(|pass| pass.name());
        let names: Vec<_> = names.collect();
        Err(VerificationError::new(format!(
            "pass pipeline [{}] did not settle within {limit} sweeps",
            names.join(", ")
        )))
    }

    /// One left-to-right round, verifying after every pass.
    ///
    /// 每个 pass 之后都校验，而不是只在两端校验：坏内核不会在发布构建里溜到发射器，报错还能
    /// 直接点名是哪条 pass。校验是线性的、内核又小，换来的是「管理器交出的内核一定合法」在
    /// **任何构建**下都成立——`run` 因此不必再补一次出站校验。
    fn sweep(&self, kernel: Kernel) -> Result<(Kernel, bool), VerificationError> {
        let start = kernel.clone();
        let mut kernel = kernel;
        for pass in &self.passes {
            kernel = pass.run(kernel)?;
            kernel.verify().map_err(|error| {
                VerificationError::new(format!(
                    "pass `{}` produced an invalid kernel: {error}",
                    pass.name()
                ))
            })?;
        }
        let changed = kernel != start;
        Ok((kernel, changed))
    }
}

/// Fixpoint measure: parameters plus expression nodes.
fn cost(kernel: &Kernel) -> usize {
    kernel.params.len() + nodes(kernel.write.expr())
}

/// 表达式节点数：本节点一算一，子节点递归求和。
pub(super) fn nodes(expression: &Expr) -> usize {
    match expression {
        Expr::Neg(value) | Expr::Unary { child: value, .. } => 1 + nodes(value),
        Expr::Binary { left, right, .. }
        | Expr::Min { left, right }
        | Expr::Max { left, right } => 1 + nodes(left) + nodes(right),
        Expr::Fma { a, b, c } => 1 + nodes(a) + nodes(b) + nodes(c),
        _ => 1,
    }
}

/// 把所有代数定律在每个节点上跑到不动点。
struct Simplify;

impl Pass for Simplify {
    fn name(&self) -> &'static str {
        "simplify"
    }

    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let mut kernel = kernel;
        let expression = kernel.write.expr().clone();
        *kernel.write.expr_mut() = normalize(expression);
        Ok(kernel)
    }
}

/// 一条定律：命中就返回严格更小的等价项，否则 `None`。
type Law = fn(&Expr) -> Option<Expr>;

/// 代数定律表：名称 + 定律。加一条律只加一行，`debug_assert` 与诊断都从这里取名字。
///
/// 只有**逐位成立**的恒等能进表：`x + 0.0 → x` 不成立（`-0.0 + 0.0 = +0.0`），所以没有它；
/// `x * 1.0 → x`、`x / 1.0 → x`、`x - 0.0 → x` 在默认舍入下成立（sNaN 的静默化除外，而核里
/// 不产生 sNaN）。非有限的折叠结果同样不折叠，留给目标端自己的 IEEE 语义。
///
/// 这些判断不是靠注释成立的：`tests/ir_tests.rs` 的解释执行用例枚举每条律的形状，在 `±0`、次正规、
/// `inf`、`NaN` 上逐位比较化简前后，塞进一条不成立的律就会红。
const RULES: [(&str, Law); 9] = [
    ("double-negation", double_negation),
    ("mul-one", mul_one),
    ("div-one", div_one),
    ("sub-zero", sub_zero),
    ("fold-negation", fold_negation),
    ("fold-binary", fold_binary),
    ("fold-unary", fold_unary),
    ("fold-minmax", fold_minmax),
    ("fold-fma", fold_fma),
];

/// 自底向上求范式：先把子节点各自规约到不动点，再在本节点上反复施加第一条命中的定律。
///
/// 定律只会把「一个节点 + 子项」换成严格更小的项，所以内层循环按子树大小终止；而它作用在已经
/// 规约好的子节点上，因此一次自底向上就得到范式——`x * (0.5 + 0.5)` 一趟就成 `x`，折叠暴露的
/// `* 1.0` 不会漏到下一轮。
///
/// 内层循环的终止不靠 `debug_assert`：预算取起始节点数，每次命中至少消掉一个节点，所以
/// **任何构建下**都只会改写有限次——一条写错的律最多让化简变慢，不会让编译器打转。
pub(super) fn normalize(expression: Expr) -> Expr {
    let mut expression = expression.map_children(|child| normalize(child.clone()));
    let mut budget = nodes(&expression);
    while budget > 0 {
        let Some((rule, next)) = first_match(&expression) else {
            break;
        };
        debug_assert!(
            nodes(&next) < nodes(&expression),
            "law `{rule}` did not shrink its term"
        );
        expression = next;
        budget -= 1;
    }
    expression
}

/// 第一条命中的定律；名称一起返回，只用于诊断。
fn first_match(expression: &Expr) -> Option<(&'static str, Expr)> {
    RULES
        .iter()
        .find_map(|(name, law)| law(expression).map(|next| (*name, next)))
}

fn double_negation(expression: &Expr) -> Option<Expr> {
    let Expr::Neg(inner) = expression else {
        return None;
    };
    let Expr::Neg(value) = inner.as_ref() else {
        return None;
    };
    Some(value.as_ref().clone())
}

fn mul_one(expression: &Expr) -> Option<Expr> {
    let one = 1.0f32.to_bits();
    identity(expression, Arith::Mul, Side::Right, one)
        .or_else(|| identity(expression, Arith::Mul, Side::Left, one))
}

fn div_one(expression: &Expr) -> Option<Expr> {
    identity(expression, Arith::Div, Side::Right, 1.0f32.to_bits())
}

fn sub_zero(expression: &Expr) -> Option<Expr> {
    identity(expression, Arith::Sub, Side::Right, 0.0f32.to_bits())
}

/// 字面量落在哪一侧。只有乘法两侧都看：它可交换，换边不改变任何一位。
#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

/// `x op bits → x`：`bits` 那一侧是字面量时返回另一侧。`x op 1 → x` 这类恒等的唯一判据。
fn identity(expression: &Expr, op: Arith, side: Side, bits: u32) -> Option<Expr> {
    let Expr::Binary {
        op: found,
        left,
        right,
    } = expression
    else {
        return None;
    };
    let (literal, kept) = match side {
        Side::Left => (left, right),
        Side::Right => (right, left),
    };
    (*found == op && **literal == Expr::F32(bits)).then(|| kept.as_ref().clone())
}

fn fold_negation(expression: &Expr) -> Option<Expr> {
    let Expr::Neg(value) = expression else {
        return None;
    };
    let Expr::F32(bits) = value.as_ref() else {
        return None;
    };
    finite(-f32::from_bits(*bits))
}

fn fold_binary(expression: &Expr) -> Option<Expr> {
    let Expr::Binary { op, left, right } = expression else {
        return None;
    };
    let (Expr::F32(left), Expr::F32(right)) = (left.as_ref(), right.as_ref()) else {
        return None;
    };
    finite(op.apply(f32::from_bits(*left), f32::from_bits(*right)))
}

/// 一元组合子字面量折叠：子项是字面量时按算子表求值；非有限结果留给目标端 IEEE 语义。
fn fold_unary(expression: &Expr) -> Option<Expr> {
    let Expr::Unary { op, child } = expression else {
        return None;
    };
    let Expr::F32(bits) = child.as_ref() else {
        return None;
    };
    finite(op.apply(f32::from_bits(*bits)))
}

fn fold_minmax(expression: &Expr) -> Option<Expr> {
    let (minimum, left, right) = match expression {
        Expr::Min { left, right } => (true, left, right),
        Expr::Max { left, right } => (false, left, right),
        _ => return None,
    };
    let (Expr::F32(left), Expr::F32(right)) = (left.as_ref(), right.as_ref()) else {
        return None;
    };
    let (left, right) = (f32::from_bits(*left), f32::from_bits(*right));
    finite(if minimum {
        left.min(right)
    } else {
        left.max(right)
    })
}

/// 三个字面量的 FMA 折叠到 `fmaf` 的正确舍入结果；与 [`fold_binary`] 一样，非有限结果不折叠。
fn fold_fma(expression: &Expr) -> Option<Expr> {
    let Expr::Fma { a, b, c } = expression else {
        return None;
    };
    let (Expr::F32(a), Expr::F32(b), Expr::F32(c)) = (a.as_ref(), b.as_ref(), c.as_ref()) else {
        return None;
    };
    finite(f32::from_bits(*a).mul_add(f32::from_bits(*b), f32::from_bits(*c)))
}

/// 非有限结果不折叠：`0.0 / 0.0` 留在 IR 里，发射出去仍是原来的语义。
fn finite(value: f32) -> Option<Expr> {
    value.is_finite().then(|| Expr::F32(value.to_bits()))
}

/// Drop parameters the kernel never reads: not the extent, not the output, not named in the write.
struct EliminateDeadParams;

impl Pass for EliminateDeadParams {
    fn name(&self) -> &'static str {
        "eliminate-dead-params"
    }

    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let mut kernel = kernel;
        let extent = kernel.extent.clone();
        let output = kernel.output.clone();
        let expression = kernel.write.expr().clone();
        kernel.params.retain(|parameter| {
            parameter.name == extent
                || parameter.name == output
                || expr_mentions(&expression, &parameter.name)
        });
        Ok(kernel)
    }
}

fn expr_mentions(expression: &Expr, name: &str) -> bool {
    expression
        .children()
        .iter()
        .any(|child| expr_mentions(child, name))
        || matches!(
            expression,
            Expr::Scalar(used) | Expr::Load(used) if used == name
        )
}

/// 乘加收缩：把 `a * b + c`（乘法可在任一侧）改写为一条 FMA。这是**舍入改变**的优化，等价于
/// CUDA 的 fast-math，因此只出现在 [`Pipeline::contracted`]，绝不到位精确的 [`RULES`] 表里。
struct FuseFma;

impl Pass for FuseFma {
    fn name(&self) -> &'static str {
        "fuse-fma"
    }

    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let mut kernel = kernel;
        let expression = kernel.write.expr().clone();
        *kernel.write.expr_mut() = contract(expression);
        Ok(kernel)
    }
}

/// 自底向上收缩：先缩子项，再在本节点上尝试 `(a * b) + c → fmaf(a, b, c)`。
fn contract(expression: Expr) -> Expr {
    let expression = expression.map_children(|child| contract(child.clone()));
    fuse_multiply_add(expression.clone()).unwrap_or(expression)
}

/// `(a * b) + c` 或 `c + (a * b)` 命中则收缩；否则 `None`。只认乘法的字面形状，不递归。
fn fuse_multiply_add(expression: Expr) -> Option<Expr> {
    let Expr::Binary {
        op: Arith::Add,
        left,
        right,
    } = expression
    else {
        return None;
    };
    if let Expr::Binary {
        op: Arith::Mul,
        left: a,
        right: b,
    } = left.as_ref()
    {
        return Some(Expr::Fma {
            a: a.clone(),
            b: b.clone(),
            c: right,
        });
    }
    if let Expr::Binary {
        op: Arith::Mul,
        left: a,
        right: b,
    } = right.as_ref()
    {
        return Some(Expr::Fma {
            a: a.clone(),
            b: b.clone(),
            c: left,
        });
    }
    None
}

#[cfg(test)]
#[path = "../tests/pass_tests.rs"]
mod pass_tests;
