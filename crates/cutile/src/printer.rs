//! typed IR 的双目标文本发射：可移植 C 与设备 `.k` 正文。
//!
//! 两种目标的正文是同一段循环，只差入口签名、长度配对与守卫、循环变量类型、循环收尾；四处分叉收进
//! [`Target`]，正文只写一遍。类型与校验留在 [`crate::kernel`]，本模块只把一份合法 [`Kernel`]
//! 渲染成源文本。

use core::fmt::Write as _;

use crate::kernel::{Arith, Expr, Kernel, Parameter, ScalarType, ValueType, VerificationError};

/// 发射目标。两边的正文是同一段循环，只差四处——入口签名、长度配对与守卫、循环变量类型、
/// 循环收尾。把这四处收进一个枚举，正文就只写一遍；少一份副本就少一处「改了一边忘了另一边」。
#[derive(Clone, Copy)]
pub(crate) enum Target {
    /// 可移植 C：配对 `size_t` 长度、带守卫、`return 0` 收尾。
    C,
    /// 设备 `.k` 正文：无长度配对、无守卫，入口与返回交给 vendor 骨架。
    K,
}

/// 两种目标共用的骨架：先优化到范式，再拼签名、守卫、循环。
pub(crate) fn render(kernel: &Kernel, target: Target) -> Result<String, VerificationError> {
    let kernel = kernel.optimize()?;
    let mut source = String::new();
    target.prologue(&mut source);
    write_signature(&mut source, target, &kernel.name, &kernel.params)?;
    target.guards(&mut source, &kernel);
    write_loop(&mut source, target, &kernel);
    Ok(source)
}

impl Target {
    /// 循环变量的 C 拼写：C 用 `size_t` 对齐 `size_t` 长度；`.k` 用 `int` 对齐 `int` 长度标量，
    /// 规避有符号 / 无符号比较。
    fn int(self) -> &'static str {
        match self {
            Target::C => "size_t",
            Target::K => "int",
        }
    }

    /// 入口签名前段：C 用 `int` 原名，`.k` 用 `void` 加前缀名。
    fn entry(self, name: &str) -> String {
        match self {
            Target::C => format!("int {name}("),
            Target::K => format!("void cutile_{name}_kernel("),
        }
    }

    /// 形参的 C 拼写。`.k` 把 `usize` 长度标量收敛成 `int`（签名解析认 `int`），其余同
    /// [`ValueType::c_type`]。
    fn declaration(self, ty: ValueType) -> Option<String> {
        match (self, ty) {
            (Target::K, ValueType::Scalar(ScalarType::Usize)) => Some("int".to_string()),
            _ => ty.c_type(),
        }
    }

    /// 是否给每个缓冲补一个隐式长度参数。只有 C 需要：设备后端的长度就是标量形参本身。
    fn pairs_lengths(self) -> bool {
        matches!(self, Target::C)
    }

    /// 头文件。`.k` 不要——vendor 骨架已经 include 过。`<math.h>` 供一元数学 / 取整与 min/max/fma 库调用。
    fn prologue(self, source: &mut String) {
        if let Target::C = self {
            source.push_str("#include <stddef.h>\n#include <math.h>\n");
        }
    }

    /// 越界守卫。`.k` 不要——设备骨架接管入口与返回，没有 `return 1` 可写。
    fn guards(self, source: &mut String, kernel: &Kernel) {
        if let Target::C = self {
            write_guards(source, kernel);
        }
    }

    /// 循环收尾：C 的入口要自己的返回值，`.k` 的返回交给骨架。
    fn trailer(self) -> &'static str {
        match self {
            Target::C => ";\n    }\n    return 0;\n}\n",
            Target::K => ";\n    }\n}\n",
        }
    }
}

/// 形参声明，C 还给每个缓冲补一个隐式长度参数：`buffer` 与其长度成对出现，顺序与
/// `check_generated` 对齐。
fn write_signature(
    source: &mut String,
    target: Target,
    name: &str,
    params: &[Parameter],
) -> Result<(), VerificationError> {
    source.push_str(&target.entry(name));
    let mut bufs = 0;
    for (idx, parameter) in params.iter().enumerate() {
        if idx != 0 {
            source.push_str(", ");
        }
        bufs += write_parameter(source, target, parameter, bufs)?;
    }
    source.push_str(") {\n");
    Ok(())
}

fn write_parameter(
    source: &mut String,
    target: Target,
    parameter: &Parameter,
    bufs: usize,
) -> Result<usize, VerificationError> {
    let Some(declaration) = target.declaration(parameter.ty) else {
        return Err(VerificationError::new(format!(
            "unsupported buffer type for `{}`",
            parameter.name
        )));
    };
    let separator = if declaration.ends_with('*') { "" } else { " " };
    let _ = write!(source, "{declaration}{separator}{}", parameter.name);
    if !(target.pairs_lengths() && parameter.ty.is_buf()) {
        return Ok(0);
    }
    let _ = write!(source, ", size_t {}", buf_length_name(bufs));
    Ok(1)
}

/// 归约只碰 `output[at]` 一个槽：输出守卫看 `at` 是否越界，其余缓冲仍看 `extent`。
fn write_guards(source: &mut String, kernel: &Kernel) {
    let slot = kernel.write.slot();
    let bufs = kernel
        .params
        .iter()
        .filter(|parameter| parameter.ty.is_buf());
    for (idx, parameter) in bufs.enumerate() {
        let guard = match slot {
            Some(at) if parameter.name == kernel.output => format!("{at} >="),
            _ => format!("{} >", kernel.extent),
        };
        let _ = writeln!(
            source,
            "    if ({guard} {}) return 1;",
            buf_length_name(idx)
        );
    }
}

/// 循环正文：头、写回值、收尾。两种目标共用同一份正文，差异全在 `target` 里。
fn write_loop(source: &mut String, target: Target, kernel: &Kernel) {
    write_header(source, target, kernel, kernel.write.slot());
    write_expr(source, kernel.write.expr(), &kernel.idx);
    source.push_str(target.trailer());
}

/// 循环头加上写回语句的左半段：循环变量类型由 `target` 定，归约多一个常量槽。
fn write_header(source: &mut String, target: Target, kernel: &Kernel, at: Option<u32>) {
    let ty = target.int();
    let Kernel {
        idx,
        extent,
        output,
        ..
    } = kernel;
    let _ = write!(
        source,
        "    for ({ty} {idx} = 0; {idx} < {extent}; ++{idx}) {{\n        {output}["
    );
    match at {
        Some(at) => {
            let _ = write!(source, "{at}] += ");
        }
        None => {
            let _ = write!(source, "{idx}] = ");
        }
    }
}

pub(crate) fn buf_length_name(buf: usize) -> String {
    format!("cutile_len_{buf}")
}

fn write_expr(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::F32(bits) => {
            let value = f32::from_bits(*bits);
            let _ = write!(source, "{value:?}f");
        }
        Expr::Int(value) => {
            let _ = write!(source, "{value}");
        }
        Expr::Scalar(name) => source.push_str(name),
        Expr::Load(name) => {
            let _ = write!(source, "{name}[{idx}]");
        }
        other => write_compound(source, other, idx),
    }
}

fn write_compound(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::Neg(_) | Expr::Not(_) | Expr::BitNot(_) | Expr::Address(_) | Expr::Deref(_) => {
            write_prefix(source, expression, expression.children()[0], idx)
        }
        Expr::Unary { .. }
        | Expr::Fma { .. }
        | Expr::Binary { .. }
        | Expr::Min { .. }
        | Expr::Max { .. }
        | Expr::Compare { .. }
        | Expr::Logical { .. } => write_binary_expression(source, expression, idx),
        Expr::Index { .. } | Expr::Call { .. } | Expr::Cast { .. } => {
            write_access(source, expression, idx)
        }
        Expr::Select { .. } | Expr::Assign { .. } | Expr::Update { .. } => {
            write_control(source, expression, idx)
        }
        Expr::F32(_) | Expr::Int(_) | Expr::Scalar(_) | Expr::Load(_) => {}
    }
}

fn write_binary_expression(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::Binary { op, left, right } => write_binary(source, *op, left, right, idx),
        Expr::Unary { .. } | Expr::Fma { .. } | Expr::Min { .. } | Expr::Max { .. } => {
            write_call(source, call_name(expression), &expression.children(), idx)
        }
        Expr::Compare { op, left, right } => {
            write_operator(source, op.c_symbol(), left, right, idx)
        }
        Expr::Logical { op, left, right } => {
            write_operator(source, op.c_symbol(), left, right, idx)
        }
        _ => {}
    }
}

fn write_access(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::Index { base, idx: at } => {
            write_expr(source, base, idx);
            source.push('[');
            write_expr(source, at, idx);
            source.push(']');
        }
        Expr::Call { function, args } => {
            write_call(source, function, &args.iter().collect::<Vec<_>>(), idx)
        }
        Expr::Cast { ty, value } => {
            let _ = write!(source, "({})(", ty.c_name());
            write_expr(source, value, idx);
            source.push(')');
        }
        _ => {}
    }
}

fn write_control(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::Select {
            condition,
            when_true,
            when_false,
        } => {
            source.push('(');
            write_expr(source, condition, idx);
            source.push_str(" ? ");
            write_expr(source, when_true, idx);
            source.push_str(" : ");
            write_expr(source, when_false, idx);
            source.push(')');
        }
        Expr::Assign { target, op, value } => write_assignment(source, target, *op, value, idx),
        Expr::Update {
            target,
            delta,
            prefix,
        } => write_update(source, target, *delta, *prefix, idx),
        _ => {}
    }
}

fn write_assignment(
    source: &mut String,
    target: &Expr,
    op: Option<Arith>,
    value: &Expr,
    idx: &str,
) {
    write_expr(source, target, idx);
    let symbol = op.map_or_else(|| "=".to_string(), |op| format!("{}=", op.c_symbol()));
    let _ = write!(source, " {symbol} ");
    write_expr(source, value, idx);
}

fn write_update(source: &mut String, target: &Expr, delta: i8, prefix: bool, idx: &str) {
    let operator = if delta == 1 { "++" } else { "--" };
    if prefix {
        source.push_str(operator);
    }
    write_expr(source, target, idx);
    if !prefix {
        source.push_str(operator);
    }
}

fn write_prefix(source: &mut String, expression: &Expr, value: &Expr, idx: &str) {
    source.push_str(prefix_symbol(expression));
    source.push('(');
    write_expr(source, value, idx);
    source.push(')');
}

fn prefix_symbol(expression: &Expr) -> &'static str {
    match expression {
        Expr::Neg(_) | Expr::Not(_) | Expr::BitNot(_) => unary_prefix_symbol(expression),
        Expr::Address(_) | Expr::Deref(_) => address_prefix_symbol(expression),
        _ => "",
    }
}

fn unary_prefix_symbol(expression: &Expr) -> &'static str {
    match expression {
        Expr::Neg(_) => "-",
        Expr::Not(_) => "!",
        Expr::BitNot(_) => "~",
        _ => "",
    }
}

fn address_prefix_symbol(expression: &Expr) -> &'static str {
    match expression {
        Expr::Address(_) => "&",
        Expr::Deref(_) => "*",
        _ => "",
    }
}

/// 复合变体对应的 C 库函数名；非调用形态返回空串。
fn call_name(expression: &Expr) -> &'static str {
    match expression {
        Expr::Unary { op, .. } => op.c_name(),
        Expr::Fma { .. } => "fmaf",
        Expr::Min { .. } => "fminf",
        Expr::Max { .. } => "fmaxf",
        _ => "",
    }
}

fn write_operator(source: &mut String, op: &str, left: &Expr, right: &Expr, idx: &str) {
    source.push('(');
    write_expr(source, left, idx);
    let _ = write!(source, " {op} ");
    write_expr(source, right, idx);
    source.push(')');
}

fn write_binary(source: &mut String, op: Arith, left: &Expr, right: &Expr, idx: &str) {
    source.push('(');
    write_expr(source, left, idx);
    let _ = write!(source, " {} ", op.c_symbol());
    write_expr(source, right, idx);
    source.push(')');
}

/// Emit a C function call over already-rendered argument expressions, comma-separated.
fn write_call(source: &mut String, function: &str, args: &[&Expr], idx: &str) {
    source.push_str(function);
    source.push('(');
    for (position, argument) in args.iter().enumerate() {
        if position != 0 {
            source.push_str(", ");
        }
        write_expr(source, argument, idx);
    }
    source.push(')');
}

// ── 完整程序 printer：只接受 kernel::Program ─────────────────────────────

use crate::kernel::{AtomArg, LocalDecl, Program, Statement};
use crate::prof::Emit;

/// 校验、优化并发射完整 `.k` 程序。
pub(crate) fn render_program(program: &Program<'_>) -> Result<String, VerificationError> {
    let program = program.optimize()?;
    let mut source = String::new();
    write_program(&mut source, &program)?;
    Ok(source)
}

/// 可移植 C 的四车道浮点类型；仅最终组装层注入完整翻译单元。
pub(crate) const C_FLOAT4_TYPEDEF: &str =
    "typedef float float4 __attribute__((vector_size(16)));\n";

/// 按真实后端完成类型 lowering；C 的四车道浮点由编译器向量扩展提供。
pub(crate) fn render_program_target(
    program: &Program<'_>,
    emit: Emit,
) -> Result<String, VerificationError> {
    let source = render_program(program)?;
    if !has_float4(&source) {
        return Ok(source);
    }
    match emit {
        Emit::C | Emit::Opencl | Emit::Metal => Ok(source),
        other => Err(VerificationError::new(format!(
            "target `{other}` has no registered native float4 lowering"
        ))),
    }
}

/// 只识别完整的类型 token，不把 `my_float4_value` 误认为向量类型。
pub(crate) fn has_float4(source: &str) -> bool {
    source
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| token == "float4")
}

/// 优化后的结构编码；原子类型与正文均进入缓存身份，不包含指针地址。
pub(crate) fn encode_program(program: &Program<'_>) -> Result<String, VerificationError> {
    let program = program.optimize()?;
    let mut out = String::new();
    let _ = writeln!(out, "{} {:?}", program.name(), program.params());
    encode_aliases(&mut out, &program);
    encode_helpers(&mut out, &program)?;
    encode_atoms(&mut out, &program);
    write_statements(&mut out, &program, program.body(), 0)?;
    Ok(out)
}

fn encode_aliases(out: &mut String, program: &Program<'_>) {
    for alias in program.aliases() {
        let _ = writeln!(out, "alias {} {:?}", alias.name, alias.ty);
    }
}

fn encode_helpers(out: &mut String, program: &Program<'_>) -> Result<(), VerificationError> {
    for helper in program.helpers() {
        let _ = writeln!(
            out,
            "helper {} {:?} {:?}",
            helper.name, helper.result, helper.params
        );
        write_statements(out, program, &helper.body, 0)?;
    }
    Ok(())
}

fn encode_atoms(out: &mut String, program: &Program<'_>) {
    for atom in program.atoms().as_slice() {
        let _ = writeln!(
            out,
            "atom {} {} {:?} {:?}",
            atom.prim(),
            atom.impl_name(),
            atom.need,
            atom.ports
        );
    }
}

fn write_program(source: &mut String, program: &Program<'_>) -> Result<(), VerificationError> {
    write_aliases(source, program);
    write_functions(source, program)?;
    write_signature(source, Target::K, program.name(), program.params())?;
    write_statements(source, program, program.body(), 1)?;
    source.push_str(
        "}
",
    );
    Ok(())
}

fn write_aliases(source: &mut String, program: &Program<'_>) {
    for alias in program.aliases() {
        let _ = writeln!(source, "typedef {} {};", alias.ty.c_name(), alias.name);
    }
}

fn write_functions(source: &mut String, program: &Program<'_>) -> Result<(), VerificationError> {
    for helper in program.helpers() {
        write_function(source, program, helper)?;
    }
    Ok(())
}

fn write_function(
    source: &mut String,
    program: &Program<'_>,
    helper: &crate::kernel::Function,
) -> Result<(), VerificationError> {
    let _ = write!(source, "static {} {}(", helper.result.c_name(), helper.name);
    for (idx, parameter) in helper.params.iter().enumerate() {
        if idx != 0 {
            source.push_str(", ");
        }
        write_parameter(source, Target::K, parameter, 0)?;
    }
    source.push_str(
        ") {
",
    );
    write_statements(source, program, &helper.body, 1)?;
    source.push_str(
        "}
",
    );
    Ok(())
}

fn write_statements(
    source: &mut String,
    program: &Program<'_>,
    statements: &[Statement],
    depth: usize,
) -> Result<(), VerificationError> {
    for statement in statements {
        write_statement(source, program, statement, depth)?;
    }
    Ok(())
}

fn write_statement(
    source: &mut String,
    program: &Program<'_>,
    statement: &Statement,
    depth: usize,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Decl(_) | Statement::Expr(_) => write_simple_statement(source, statement, depth),
        Statement::Atom(call) => write_atom(source, program, call, depth),
        Statement::Continue | Statement::Break => {
            indent(source, depth);
            source.push_str(if matches!(statement, Statement::Continue) {
                "continue;
"
            } else {
                "break;
"
            });
            Ok(())
        }
        other => write_complex_statement(source, program, other, depth),
    }
}

fn write_simple_statement(
    source: &mut String,
    statement: &Statement,
    depth: usize,
) -> Result<(), VerificationError> {
    indent(source, depth);
    match statement {
        Statement::Decl(local) => write_local(source, local),
        Statement::Expr(value) => write_expr(source, value, ""),
        _ => return Err(VerificationError::new("invalid simple statement")),
    }
    source.push_str(
        ";
",
    );
    Ok(())
}

fn write_complex_statement(
    source: &mut String,
    program: &Program<'_>,
    statement: &Statement,
    depth: usize,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Block(_) | Statement::For { .. } | Statement::If { .. } => {
            write_control_statement(source, program, statement, depth)
        }
        Statement::Pragma { name, value } => write_pragma(source, name, *value, depth),
        Statement::Return(value) => write_return(source, value, depth),
        _ => Err(VerificationError::new("invalid statement family")),
    }
}

fn write_control_statement(
    source: &mut String,
    program: &Program<'_>,
    statement: &Statement,
    depth: usize,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Block(body) => write_block(source, program, body, depth),
        Statement::For {
            init,
            condition,
            step,
            body,
        } => write_for(source, program, init, condition, step, body, depth),
        Statement::If {
            condition,
            then_branch,
            else_branch,
        } => write_if(source, program, condition, then_branch, else_branch, depth),
        _ => Ok(()),
    }
}

fn write_for(
    source: &mut String,
    program: &Program<'_>,
    init: &Option<Box<Statement>>,
    condition: &Option<Expr>,
    step: &Option<Expr>,
    body: &[Statement],
    depth: usize,
) -> Result<(), VerificationError> {
    indent(source, depth);
    source.push_str("for (");
    if let Some(init) = init {
        write_for_init(source, init)?;
    }
    source.push_str("; ");
    write_optional(source, condition);
    source.push_str("; ");
    write_optional(source, step);
    source.push_str(
        ") {
",
    );
    write_statements(source, program, body, depth + 1)?;
    indent(source, depth);
    source.push_str(
        "}
",
    );
    Ok(())
}

fn write_if(
    source: &mut String,
    program: &Program<'_>,
    condition: &Expr,
    then_branch: &[Statement],
    else_branch: &[Statement],
    depth: usize,
) -> Result<(), VerificationError> {
    indent(source, depth);
    source.push_str("if (");
    write_expr(source, condition, "");
    source.push_str(
        ") {
",
    );
    write_statements(source, program, then_branch, depth + 1)?;
    indent(source, depth);
    source.push('}');
    if !else_branch.is_empty() {
        source.push_str(
            " else {
",
        );
        write_statements(source, program, else_branch, depth + 1)?;
        indent(source, depth);
        source.push('}');
    }
    source.push('\n');
    Ok(())
}

fn write_pragma(
    source: &mut String,
    name: &str,
    value: Option<u32>,
    depth: usize,
) -> Result<(), VerificationError> {
    indent(source, depth);
    let _ = write!(source, "#pragma {name}");
    if let Some(value) = value {
        let _ = write!(source, " {value}");
    }
    source.push('\n');
    Ok(())
}

fn write_return(
    source: &mut String,
    value: &Option<Expr>,
    depth: usize,
) -> Result<(), VerificationError> {
    indent(source, depth);
    source.push_str("return");
    if let Some(value) = value {
        source.push(' ');
        write_expr(source, value, "");
    }
    source.push_str(
        ";
",
    );
    Ok(())
}

fn write_block(
    source: &mut String,
    program: &Program<'_>,
    body: &[Statement],
    depth: usize,
) -> Result<(), VerificationError> {
    indent(source, depth);
    source.push_str("{\n");
    write_statements(source, program, body, depth + 1)?;
    indent(source, depth);
    source.push_str("}\n");
    Ok(())
}

fn write_for_init(source: &mut String, statement: &Statement) -> Result<(), VerificationError> {
    match statement {
        Statement::Decl(local) => write_local(source, local),
        Statement::Expr(value) => write_expr(source, value, ""),
        _ => {
            return Err(VerificationError::new(
                "for initializer must be a declaration or expression",
            ));
        }
    }
    Ok(())
}

fn write_optional(source: &mut String, value: &Option<Expr>) {
    if let Some(value) = value {
        write_expr(source, value, "");
    }
}

fn write_local(source: &mut String, local: &LocalDecl) {
    if local.ty.constant {
        source.push_str("const ");
    }
    source.push_str(local.ty.element.c_name());
    for _ in 0..local.ty.pointers {
        source.push('*');
    }
    source.push(' ');
    source.push_str(&local.name);
    if let Some(size) = local.ty.array {
        let _ = write!(source, "[{size}]");
    }
    if let Some(init) = &local.init {
        source.push_str(" = ");
        write_expr(source, init, "");
    }
}

fn write_atom(
    source: &mut String,
    program: &Program<'_>,
    call: &crate::kernel::AtomCall,
    depth: usize,
) -> Result<(), VerificationError> {
    let atom = program
        .atom(call)
        .ok_or_else(|| VerificationError::new("atom id is not in the program arena"))?;
    let Some(body) = atom.body.text() else {
        return Ok(());
    };
    let values = atom_values(call);
    let args: Vec<crate::substitute::Arg<'_>> = call
        .args
        .iter()
        .zip(&values)
        .map(|(arg, value)| atom_arg_view(arg, value))
        .collect();
    let body = crate::substitute::splice(body, &args, call.line)
        .map_err(|error| VerificationError::new(error.to_string()))?;
    indent(source, depth);
    crate::substitute::place(source, &body);
    terminate_atom(source, &body, depth);
    Ok(())
}

fn atom_values(call: &crate::kernel::AtomCall) -> Vec<Option<String>> {
    call.args
        .iter()
        .map(|arg| match arg {
            AtomArg::Expr(value) => Some(expression_text(value)),
            AtomArg::Named { .. } => None,
        })
        .collect()
}

fn atom_arg_view<'a>(arg: &'a AtomArg, value: &'a Option<String>) -> crate::substitute::Arg<'a> {
    match arg {
        AtomArg::Expr(_) => crate::substitute::Arg::Expr(value.as_deref().unwrap_or_default()),
        AtomArg::Named { key, value } => crate::substitute::Arg::Named { key, value },
    }
}

fn terminate_atom(source: &mut String, body: &str, depth: usize) {
    if body
        .trim_end()
        .lines()
        .last()
        .is_some_and(|line| line.trim_start().starts_with("//"))
    {
        source.push('\n');
        indent(source, depth);
    }
    // 原模板调用语句的 `;` 已由 parser 消费；printer 必须重新终结该语句。
    source.push_str(
        ";
",
    );
}

fn expression_text(expression: &Expr) -> String {
    let mut out = String::new();
    write_expr(&mut out, expression, "");
    out
}

fn indent(source: &mut String, depth: usize) {
    for _ in 0..depth {
        source.push_str("    ");
    }
}
