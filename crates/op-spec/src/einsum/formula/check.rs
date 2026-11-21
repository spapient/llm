use super::*;

/// 同名缓冲只允许出现一次：否则同一块内存会被重复计费。
pub(super) fn unique(operands: &[Operand]) -> Result<(), Error> {
    let mut names: Vec<&str> = Vec::with_capacity(operands.len());
    for operand in operands {
        if names.contains(&operand.name()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "操作数 `{}` 出现了两次：同名缓冲会被重复计费",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
        names.push(operand.name());
    }
    Ok(())
}

/// 一次计算只有一个出口。
pub(super) fn single_write(operands: &[Operand]) -> Result<(), Error> {
    let writes = operands
        .iter()
        .filter(|operand| operand.access() == Access::Write)
        .count();
    if writes > 1 {
        return Err(
            Error::new(Code::Invalid, "有多个写入操作数：一次计算只有一个出口")
                .with("einsum::Formula::new"),
        );
    }
    Ok(())
}

/// 写入操作数只能写输出轴：写回中间量不是算子语义。
pub(super) fn writes_only_outputs(expr: &Expr, operands: &[Operand]) -> Result<(), Error> {
    for (operand, term) in operands.iter().zip(expr.terms()) {
        if operand.access() != Access::Write {
            continue;
        }
        if let Some(label) = term.iter().find(|label| !expr.output().contains(label)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "写入操作数 `{}` 的轴 `{label}` 不在输出里：写回中间量不是算子语义",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
    }
    Ok(())
}
