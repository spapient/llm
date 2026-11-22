//! Typed-IR pass framework: named, ordered transforms over a [`Kernel`].
//!
//! Each [`Pass`] is a transform from one kernel to another (LLVM's pass model, sized to the
//! `#[dsl]` kernel family). A [`PassManager`] runs its passes left-to-right; the standard pipeline
//! [`PassManager::standard`] verifies, canonicalizes, folds constants, drops dead parameters, then
//! verifies again, so a compiled kernel is valid by construction.

use super::{BinaryOp, Expr, Kernel, VerificationError};

/// One named transform over a kernel.
///
/// A pass may validate (`Verify`), simplify an expression (`Canonicalize`, `FoldConstants`), or
/// shrink the signature (`EliminateDeadParams`). It returns the next kernel or a
/// [`VerificationError`] when the kernel must be rejected.
pub trait Pass {
    /// Short, stable name used for diagnostics.
    fn name(&self) -> &'static str;

    /// Apply this pass, returning the next kernel or a verification error.
    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError>;
}

/// An ordered list of passes, run left-to-right over a kernel.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// An empty pipeline; append passes with [`PassManager::add`].
    pub fn new() -> Self {
        Self::default()
    }

    /// The standard pipeline: verify → canonicalize → fold constants → drop dead params → verify.
    pub fn standard() -> Self {
        Self::new()
            .add(Verify)
            .add(Canonicalize)
            .add(FoldConstants)
            .add(EliminateDeadParams)
            .add(Verify)
    }

    /// Append one pass and return the manager for chaining.
    pub fn add<P: Pass + 'static>(mut self, pass: P) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Run every pass in registration order over `kernel`.
    pub fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let mut kernel = kernel;
        for pass in &self.passes {
            kernel = pass.run(kernel)?;
        }
        Ok(kernel)
    }
}

/// Validate the kernel in place; the pipeline's opening and closing gate.
struct Verify;

impl Pass for Verify {
    fn name(&self) -> &'static str {
        "verify"
    }

    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        kernel.verify()?;
        Ok(kernel)
    }
}

/// Rewrite an expression to its canonical shape.
///
/// `-(-x) → x`, `x * 1.0 → x`, `x / 1.0 → x`, `x - 0.0 → x`; every rewrite is bit-preserving.
struct Canonicalize;

impl Pass for Canonicalize {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let mut kernel = kernel;
        let expression = kernel.write.expr().clone();
        *kernel.write.expr_mut() = canonical_expr(expression);
        Ok(kernel)
    }
}

/// Fold constant-only subexpressions, keeping operations whose result is non-finite.
struct FoldConstants;

impl Pass for FoldConstants {
    fn name(&self) -> &'static str {
        "fold-constants"
    }

    fn run(&self, kernel: Kernel) -> Result<Kernel, VerificationError> {
        let mut kernel = kernel;
        let expression = kernel.write.expr().clone();
        *kernel.write.expr_mut() = fold_expr(expression);
        Ok(kernel)
    }
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

fn canonical_expr(expression: Expr) -> Expr {
    match expression {
        Expr::Neg(value) => canonical_negation(canonical_expr(*value)),
        Expr::Binary { op, left, right } => canonical_binary(op, canonical_expr(*left), canonical_expr(*right)),
        leaf => leaf,
    }
}

fn canonical_negation(value: Expr) -> Expr {
    let Expr::Neg(inner) = value else {
        return Expr::Neg(Box::new(value));
    };
    *inner
}

fn canonical_binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    if let Some(identity) = multiply_identity(op, &left, &right) {
        return identity;
    }
    if let Some(identity) = divide_identity(op, &left, &right) {
        return identity;
    }
    if let Some(identity) = subtract_identity(op, &left, &right) {
        return identity;
    }
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn multiply_identity(op: BinaryOp, left: &Expr, right: &Expr) -> Option<Expr> {
    if op != BinaryOp::Mul {
        return None;
    }
    if is_one(right) {
        return Some(left.clone());
    }
    if is_one(left) {
        return Some(right.clone());
    }
    None
}

fn divide_identity(op: BinaryOp, left: &Expr, right: &Expr) -> Option<Expr> {
    if op == BinaryOp::Div && is_one(right) {
        return Some(left.clone());
    }
    None
}

fn subtract_identity(op: BinaryOp, left: &Expr, right: &Expr) -> Option<Expr> {
    if op == BinaryOp::Sub && is_zero(right) {
        return Some(left.clone());
    }
    None
}

fn is_one(expression: &Expr) -> bool {
    let Expr::F32(bits) = expression else {
        return false;
    };
    *bits == 1.0f32.to_bits()
}

fn is_zero(expression: &Expr) -> bool {
    let Expr::F32(bits) = expression else {
        return false;
    };
    *bits == 0.0f32.to_bits()
}

fn fold_expr(expression: Expr) -> Expr {
    match expression {
        Expr::Neg(value) => fold_negation(*value),
        Expr::Binary { op, left, right } => fold_binary(op, *left, *right),
        expression => expression,
    }
}

fn fold_negation(value: Expr) -> Expr {
    let value = fold_expr(value);
    let Expr::F32(bits) = value else {
        return Expr::Neg(Box::new(value));
    };
    let result = -f32::from_bits(bits);
    if result.is_finite() {
        return Expr::F32(result.to_bits());
    }
    Expr::Neg(Box::new(Expr::F32(bits)))
}

fn fold_binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    let left = fold_expr(left);
    let right = fold_expr(right);
    if let (Expr::F32(left), Expr::F32(right)) = (&left, &right) {
        let result = op.apply(f32::from_bits(*left), f32::from_bits(*right));
        if result.is_finite() {
            return Expr::F32(result.to_bits());
        }
    }
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
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
    use super::{
        canonical_expr, BinaryOp, EliminateDeadParams, Expr, Kernel, Pass, PassManager, Verify,
    };

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

    #[test]
    fn canonicalize_removes_double_negation_and_identities() {
        let cases = [
            (
                Expr::Neg(Box::new(Expr::Neg(Box::new(load("a"))))),
                load("a"),
            ),
            (binary(BinaryOp::Mul, load("a"), literal(1.0)), load("a")),
            (binary(BinaryOp::Div, load("a"), literal(1.0)), load("a")),
            (binary(BinaryOp::Sub, load("a"), literal(0.0)), load("a")),
        ];
        for (input, expected) in cases {
            assert_eq!(canonical_expr(input), expected);
        }
    }

    #[test]
    fn canonicalize_preserves_non_identities() {
        let expression = binary(BinaryOp::Add, load("a"), literal(1.0));
        assert_eq!(canonical_expr(expression.clone()), expression);
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
    fn verify_pass_rejects_invalid_kernels() {
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
        let error = Verify.run(kernel).expect_err("non-usize extent is rejected");
        assert!(error.to_string().contains("loop extent"));
    }
}
