//! Reduction IR: accumulate a value into a fixed output slot over a loop.
//!
//! 与 `Kernel` 只有写回形状不同：逐点写 `output[idx] = value`，归约写 `output[at] += value`。
//! 校验复用 `super` 的名字/形参/表达式六道闸，发射时只有守卫与循环体分道。

use core::fmt::Write as _;

use super::{
    Expr, Parameter, VerificationError, buf_length_name, optimize_expr, verify_core, write_expr,
    write_signature,
};

/// One verified reduction kernel: accumulate `value` into a fixed output slot over a loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reduce {
    /// Kernel name.
    pub name: String,
    /// Ordered scalar and buffer parameters.
    pub params: Vec<Parameter>,
    /// Reduction index variable.
    pub idx: String,
    /// Name of the `usize` parameter providing the loop bound.
    pub extent: String,
    /// Name of the mutable output buffer.
    pub output: String,
    /// Constant slot accumulated into on every iteration.
    pub at: u32,
    /// Value added to the accumulator on each iteration.
    pub value: Expr,
}

impl Reduce {
    /// Construct a reduction kernel IR node.
    pub fn new(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        at: u32,
        value: Expr,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            idx: idx.into(),
            extent: extent.into(),
            output: output.into(),
            at,
            value,
        }
    }

    /// Validate names, parameter types, memory access, and expression types.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_core(
            &self.name,
            &self.idx,
            &self.params,
            &self.extent,
            &self.output,
            &self.value,
        )
    }

    /// Fold constant-only subexpressions, retaining operations whose result is non-finite.
    pub fn optimize(&self) -> Result<Self, VerificationError> {
        self.verify()?;
        let mut optimized = self.clone();
        optimized.value = optimize_expr(optimized.value);
        optimized.verify()?;
        Ok(optimized)
    }

    /// Emit portable C: one reduction loop accumulating into `output[at]`.
    pub fn emit_c(&self) -> Result<String, VerificationError> {
        let reduce = self.optimize()?;
        let mut source = String::from("#include <stddef.h>\n");
        write_signature(&mut source, &reduce.name, &reduce.params)?;
        write_reduce_guards(&mut source, &reduce);
        write_reduce_loop(&mut source, &reduce);
        Ok(source)
    }
}

/// 归约只碰 `output[at]` 一个槽：输出守卫看 `at` 是否越界，其余缓冲仍看 `extent`。
fn write_reduce_guards(source: &mut String, reduce: &Reduce) {
    let mut bufs = 0;
    for parameter in &reduce.params {
        if !parameter.ty.is_buf() {
            continue;
        }
        if parameter.name == reduce.output {
            let _ = writeln!(
                source,
                "    if ({} >= {}) return 1;",
                reduce.at,
                buf_length_name(bufs)
            );
        } else {
            let _ = writeln!(
                source,
                "    if ({} > {}) return 1;",
                reduce.extent,
                buf_length_name(bufs)
            );
        }
        bufs += 1;
    }
}

fn write_reduce_loop(source: &mut String, reduce: &Reduce) {
    let _ = write!(
        source,
        "    for (size_t {0} = 0; {0} < {1}; ++{0}) {{\n        {2}[{3}] += ",
        reduce.idx, reduce.extent, reduce.output, reduce.at
    );
    write_expr(source, &reduce.value, &reduce.idx);
    source.push_str(";\n    }\n    return 0;\n}\n");
}
