//! Typed-IR pass framework: named transforms over a [`Kernel`], driven to a fixpoint.
//!
//! 一个 [`Pass`] 只做一件事：把内核变成另一个内核（LLVM 的 pass 模型，规模压到 `#[dsl]` 这一族）。
//! [`PassManager::run`] 就是全部契约——进来先校验，把注册的 pass 扫到不动点，出去再校验；离开
//! 管理器的内核按构造成立，与注册了哪些 pass 无关。
//!
//! 为什么要扫不止一轮：管理器**不要求**调用方按依赖顺序注册。任一 pass 都可能在已经跑过的 pass
//! 左侧造出新的机会，只有下一轮才轮得到它，所以「注册顺序」不该是正确性的前提。轮次不用魔法数：
//! 上限取初始内核的 `cost`（形参数 + 表达式节点数），一个严格收缩的 pass 必然在此之前收敛；
//! 上限耗尽说明流水线在震荡，此时**报错**（MLIR 的 "did not converge"），而不是悄悄截断一段
//! 半成品内核。对自定义 pass 的要求因此只有一条：要么严格收缩，要么幂等。

use super::{BinaryOp, Expr, Kernel, VerificationError};

/// One named transform over a kernel.
///
/// A pass may simplify an expression (`Simplify`), shrink the signature
/// (`EliminateDeadParams`), or do anything else a caller needs. It returns the next kernel, or a
/// [`VerificationError`] when the kernel must be rejected.
///
/// 契约只有一条：**要么严格收缩，要么幂等**。管理器会反复扫描直到内核不再变化，每次调用都改
/// 一点的 pass 会被判为不收敛（见 [`PassManager::run`]）。
pub trait Pass {
    /// Short, stable name used for diagnostics.
    fn name(&self) -> &'static str;

    /// Apply this pass, returning the next kernel or a verification error.
    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError>;
}

/// An ordered list of passes, swept left-to-right to a fixpoint.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// An empty pipeline; append passes with [`PassManager::push`].
    pub fn new() -> Self {
        Self::default()
    }

    /// 标准通道：化简（代数定律 + 常量折叠）→ 去死形参。校验由 [`PassManager::run`] 提供。
    pub fn standard() -> Self {
        Self::new().push(Simplify).push(EliminateDeadParams)
    }

    /// Append one pass and return the manager for chaining.
    pub fn push<P: Pass + 'static>(mut self, pass: P) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Verify, sweep the passes to a fixpoint, verify again.
    ///
    /// 轮次上限是初始内核的 `cost`：严格收缩的 pass 每轮至少减一，收敛必然在此之前发生。上限
    /// 耗尽说明某个 pass 既不收缩也不幂等，此时返回错误，而不是把半成品当结果交出去。
    ///
    /// # Errors
    ///
    /// The kernel or any intermediate kernel fails verification, a pass rejects it, or the pipeline
    /// never settles.
    pub fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        kernel.verify()?;
        let kernel = self.fixpoint(kernel)?;
        kernel.verify()?;
        Ok(kernel)
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
        Err(VerificationError::new(format!(
            "pass pipeline did not settle within {limit} sweeps"
        )))
    }

    /// One left-to-right round. A pass that breaks verification is a framework bug, not a caller
    /// error, so it is caught here in debug builds instead of surfacing as invalid emitted code.
    fn sweep(&self, kernel: Kernel) -> Result<(Kernel, bool), VerificationError> {
        let start = kernel.clone();
        let mut kernel = kernel;
        for pass in &self.passes {
            kernel = pass.run(kernel)?;
            debug_assert!(
                kernel.verify().is_ok(),
                "pass `{}` produced an invalid kernel",
                pass.name()
            );
        }
        let changed = kernel != start;
        Ok((kernel, changed))
    }
}

/// Fixpoint measure: parameters plus expression nodes.
fn cost(kernel: &Kernel) -> usize {
    kernel.params.len() + nodes(kernel.write.expr())
}

/// 表达式节点数：每个二元 / 取负节点各算一，叶子算一。
fn nodes(expression: &Expr) -> usize {
    match expression {
        Expr::Binary { left, right, .. } => 1 + nodes(left) + nodes(right),
        Expr::Neg(value) => 1 + nodes(value),
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
const RULES: [(&str, Law); 6] = [
    ("double-negation", double_negation),
    ("mul-one", mul_one),
    ("div-one", div_one),
    ("sub-zero", sub_zero),
    ("fold-negation", fold_negation),
    ("fold-binary", fold_binary),
];

/// 自底向上求范式：先把子节点各自规约到不动点，再在本节点上反复施加第一条命中的定律。
///
/// 定律只会把「一个节点 + 子项」换成严格更小的项，所以内层循环按子树大小终止；而它作用在已经
/// 规约好的子节点上，因此一次自底向上就得到范式——`x * (0.5 + 0.5)` 一趟就成 `x`，折叠暴露的
/// `* 1.0` 不会漏到下一轮。
fn normalize(expression: Expr) -> Expr {
    let mut expression = match expression {
        Expr::Neg(value) => Expr::Neg(Box::new(normalize(*value))),
        Expr::Binary { op, left, right } => Expr::Binary {
            op,
            left: Box::new(normalize(*left)),
            right: Box::new(normalize(*right)),
        },
        leaf => leaf,
    };
    while let Some((rule, next)) = first_match(&expression) {
        debug_assert!(
            nodes(&next) < nodes(&expression),
            "law `{rule}` did not shrink its term"
        );
        expression = next;
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
    let Expr::Binary {
        op: BinaryOp::Mul,
        left,
        right,
    } = expression
    else {
        return None;
    };
    let one = 1.0f32.to_bits();
    literal_identity(left, right, one).or_else(|| literal_identity(right, left, one))
}

fn div_one(expression: &Expr) -> Option<Expr> {
    let Expr::Binary {
        op: BinaryOp::Div,
        left,
        right,
    } = expression
    else {
        return None;
    };
    literal_identity(left, right, 1.0f32.to_bits())
}

fn sub_zero(expression: &Expr) -> Option<Expr> {
    let Expr::Binary {
        op: BinaryOp::Sub,
        left,
        right,
    } = expression
    else {
        return None;
    };
    literal_identity(left, right, 0.0f32.to_bits())
}

/// `left` 当且仅当 `right` 正是 `bits` 对应的字面量；`x op 1 → x` 这类恒等的唯一判据。
fn literal_identity(left: &Expr, right: &Expr, bits: u32) -> Option<Expr> {
    let Expr::F32(found) = right else {
        return None;
    };
    (*found == bits).then(|| left.clone())
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
    match expression {
        Expr::Scalar(used) | Expr::Load(used) => used == name,
        Expr::Neg(value) => expr_mentions(value, name),
        Expr::Binary { left, right, .. } => binary_mentions(left, right, name),
        Expr::F32(_) => false,
    }
}

fn binary_mentions(left: &Expr, right: &Expr, name: &str) -> bool {
    expr_mentions(left, name) || expr_mentions(right, name)
}

#[cfg(test)]
mod tests {
    use super::super::{Access, Parameter, ScalarType, ValueType};
    use super::{BinaryOp, EliminateDeadParams, Expr, Kernel, Pass, PassManager, normalize};

    fn literal(value: f32) -> Expr {
        Expr::F32(value.to_bits())
    }

    fn load(name: &str) -> Expr {
        Expr::Load(name.to_string())
    }

    fn param(name: &str, ty: ValueType) -> Parameter {
        Parameter {
            name: name.to_string(),
            ty,
        }
    }

    fn out_buf() -> Parameter {
        param(
            "out",
            ValueType::Buf {
                element: ScalarType::F32,
                access: Access::ReadWrite,
            },
        )
    }

    fn extent() -> Parameter {
        param("extent", ValueType::Scalar(ScalarType::Usize))
    }

    fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// 定律表逐条验证，并顺带钉住两条边界：`x + 0.0` 必须原样保留（`-0.0` 反例），
    /// 子项先规约好的结果必须在同一趟里被父节点用上。
    #[test]
    fn laws_normalize_bottom_up() {
        let add = binary(BinaryOp::Add, load("a"), literal(0.0));
        let cases = [
            (
                Expr::Neg(Box::new(Expr::Neg(Box::new(load("a"))))),
                load("a"),
            ),
            (binary(BinaryOp::Mul, load("a"), literal(1.0)), load("a")),
            (binary(BinaryOp::Mul, literal(1.0), load("a")), load("a")),
            (binary(BinaryOp::Div, load("a"), literal(1.0)), load("a")),
            (binary(BinaryOp::Sub, load("a"), literal(0.0)), load("a")),
            (binary(BinaryOp::Add, literal(0.5), literal(0.5)), literal(1.0)),
            (Expr::Neg(Box::new(literal(2.0))), literal(-2.0)),
            (
                binary(
                    BinaryOp::Mul,
                    load("a"),
                    binary(BinaryOp::Add, literal(0.5), literal(0.5)),
                ),
                load("a"),
            ),
            (add.clone(), add),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize(input), expected);
        }
    }

    #[test]
    fn eliminate_dead_params_drops_unused_scalars_only() {
        let kernel = Kernel::map(
            "keep",
            vec![
                out_buf(),
                extent(),
                param("unused", ValueType::Scalar(ScalarType::F32)),
                param("used", ValueType::Scalar(ScalarType::F32)),
            ],
            "i",
            "extent",
            "out",
            Expr::Scalar("used".to_string()),
        );
        let kernel = EliminateDeadParams.run(kernel).expect("eliminate");
        let names: Vec<&str> = kernel
            .params
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        assert_eq!(names, ["out", "extent", "used"]);
    }

    #[test]
    fn standard_pipeline_is_idempotent() {
        let kernel = Kernel::map(
            "clean",
            vec![
                out_buf(),
                extent(),
                param("unused", ValueType::Scalar(ScalarType::F32)),
            ],
            "i",
            "extent",
            "out",
            Expr::Neg(Box::new(Expr::Neg(Box::new(literal(2.0))))),
        );
        let once = PassManager::standard().run(kernel.clone()).expect("once");
        let twice = PassManager::standard().run(once.clone()).expect("twice");
        assert_eq!(once, twice);
    }

    #[test]
    fn run_rejects_invalid_kernels() {
        let kernel = Kernel::map(
            "bad",
            vec![
                out_buf(),
                param("extent", ValueType::Scalar(ScalarType::F32)),
            ],
            "i",
            "extent",
            "out",
            literal(0.0),
        );
        let error = PassManager::new()
            .run(kernel)
            .expect_err("non-usize extent is rejected");
        assert!(error.to_string().contains("loop extent"));
    }
}
