//! Validation for every [`crate::kernel::Kernel`], regardless of its [`crate::kernel::Write`].
//!
//! 六道闸按序排布：名字 → 形参 → 生成名 → extent → 输出 → 表达式。前一半决定 C 声明能否
//! 合法发射，后一半判定表达式到底算不算 `f32` 输出；逐点与归约只有写回形状不同，闸口完全一样。

use crate::kernel::{Access, Expr, Parameter, ScalarType, ValueType, VerificationError};
use crate::printer::buf_length_name;

/// 逐点与归约共用的六道校验：名字/形参/生成名决定 C 声明，之后才验表达式是否算 f32 输出。
pub(super) fn verify_core(
    name: &str,
    idx: &str,
    params: &[Parameter],
    extent: &str,
    output: &str,
    value: &Expr,
) -> Result<(), VerificationError> {
    check_shape(name, idx, params)?;
    check_semantics(params, extent, output, value)
}

/// 名字与形参决定 C 声明，必须是最前面的闸。
fn check_shape(name: &str, idx: &str, params: &[Parameter]) -> Result<(), VerificationError> {
    check_names(name, idx, params)?;
    check_params(params)?;
    check_generated(params)
}

/// 声明合法之后才谈得上「表达式到底算不算 f32 输出」。
fn check_semantics(
    params: &[Parameter],
    extent: &str,
    output: &str,
    value: &Expr,
) -> Result<(), VerificationError> {
    check_extent(params, extent)?;
    check_output(params, output)?;
    check_value(params, value)
}

fn check_names(name: &str, idx: &str, params: &[Parameter]) -> Result<(), VerificationError> {
    valid_identifier(name)?;
    valid_identifier(idx)?;
    if params.iter().any(|param| param.name == idx) {
        return Err(VerificationError::new(format!(
            "loop index `{idx}` shadows a kernel parameter"
        )));
    }
    Ok(())
}

/// 逐条检查参数：名字合法、不重名、缓冲元素是 `f32`。
fn check_params(params: &[Parameter]) -> Result<(), VerificationError> {
    for (idx, param) in params.iter().enumerate() {
        valid_identifier(&param.name)?;
        if params[..idx].iter().any(|prior| prior.name == param.name) {
            return Err(VerificationError::new(format!(
                "duplicate parameter `{}`",
                param.name
            )));
        }
        if !param.ty.element_ok() {
            return Err(VerificationError::new(format!(
                "buffer `{}` must contain f32 elements",
                param.name
            )));
        }
    }
    Ok(())
}

/// 生成的长度名不许和用户形参撞车，否则 C 签名里会出现两个同名参数。
fn check_generated(params: &[Parameter]) -> Result<(), VerificationError> {
    for (idx, _) in params.iter().filter(|param| param.ty.is_buf()).enumerate() {
        let generated = buf_length_name(idx);
        if params.iter().any(|other| other.name == generated) {
            return Err(VerificationError::new(format!(
                "parameter `{generated}` conflicts with a generated buffer length"
            )));
        }
    }
    Ok(())
}

fn check_extent(params: &[Parameter], extent: &str) -> Result<(), VerificationError> {
    if parameter(params, extent) != Some(ValueType::Scalar(ScalarType::Usize)) {
        return Err(VerificationError::new(format!(
            "loop extent `{extent}` must be a `usize` parameter"
        )));
    }
    Ok(())
}

fn check_output(params: &[Parameter], output: &str) -> Result<(), VerificationError> {
    let expected = ValueType::Buf {
        element: ScalarType::F32,
        access: Access::ReadWrite,
    };
    if parameter(params, output) != Some(expected) {
        return Err(VerificationError::new(format!(
            "output `{output}` must be a mutable f32 buffer"
        )));
    }
    Ok(())
}

fn check_value(params: &[Parameter], value: &Expr) -> Result<(), VerificationError> {
    if value_type(params, value)? != ScalarType::F32 {
        return Err(VerificationError::new(
            "kernel output expression must have type f32",
        ));
    }
    Ok(())
}

fn parameter(params: &[Parameter], name: &str) -> Option<ValueType> {
    params
        .iter()
        .find(|parameter| parameter.name == name)
        .map(|parameter| parameter.ty)
}

fn value_type(params: &[Parameter], expression: &Expr) -> Result<ScalarType, VerificationError> {
    match expression {
        Expr::F32(bits) => literal_type(*bits),
        Expr::Scalar(name) => scalar_type(params, name),
        Expr::Load(name) => load_type(params, name),
        other => compound_type(params, other),
    }
}

fn compound_type(params: &[Parameter], expression: &Expr) -> Result<ScalarType, VerificationError> {
    match expression {
        Expr::Neg(value) => unary_f32(params, value, "negation"),
        Expr::Unary { op, child } => unary_f32(params, child, op.c_name()),
        Expr::Binary { left, right, .. }
        | Expr::Min { left, right }
        | Expr::Max { left, right } => binary_type(params, left, right),
        Expr::Fma { a, b, c } => ternary_type(params, a, b, c),
        // 完整程序的扩展节点由 `verify_program` 按作用域与局部类型校验。
        _ => Err(VerificationError::new(
            "extended expression requires the program verifier",
        )),
    }
}

fn scalar_type(params: &[Parameter], name: &str) -> Result<ScalarType, VerificationError> {
    match parameter(params, name) {
        Some(ValueType::Scalar(ty)) => Ok(ty),
        Some(ValueType::Buf { .. }) => Err(VerificationError::new(format!(
            "buffer `{name}` must be loaded at the loop index"
        ))),
        None => Err(VerificationError::new(format!(
            "unknown scalar parameter `{name}`"
        ))),
    }
}

fn load_type(params: &[Parameter], name: &str) -> Result<ScalarType, VerificationError> {
    match parameter(params, name) {
        Some(ValueType::Buf { element, .. }) if element.buf_ok() => Ok(ScalarType::F32),
        Some(ValueType::Buf { .. }) => Err(VerificationError::new(format!(
            "buffer `{name}` does not contain f32 elements"
        ))),
        Some(ValueType::Scalar(_)) => Err(VerificationError::new(format!(
            "scalar `{name}` cannot be loaded as a buffer"
        ))),
        None => Err(VerificationError::new(format!("unknown buffer `{name}`"))),
    }
}

/// 一元 f32 组合子（取负、绝对值）：子表达式必须算 f32，结果仍是 f32。
fn unary_f32(
    params: &[Parameter],
    value: &Expr,
    what: &str,
) -> Result<ScalarType, VerificationError> {
    let ty = value_type(params, value)?;
    if ty == ScalarType::F32 {
        return Ok(ty);
    }
    Err(VerificationError::new(format!("unary {what} requires f32")))
}

fn binary_type(
    params: &[Parameter],
    left: &Expr,
    right: &Expr,
) -> Result<ScalarType, VerificationError> {
    let left = value_type(params, left)?;
    let right = value_type(params, right)?;
    if left == right && left == ScalarType::F32 {
        return Ok(left);
    }
    Err(VerificationError::new(
        "pointwise arithmetic operands must both be f32",
    ))
}

/// 三元 f32 组合子（融合乘加）：三个操作数都必须算 f32，结果仍是 f32。
fn ternary_type(
    params: &[Parameter],
    a: &Expr,
    b: &Expr,
    c: &Expr,
) -> Result<ScalarType, VerificationError> {
    let types = [
        value_type(params, a)?,
        value_type(params, b)?,
        value_type(params, c)?,
    ];
    types
        .iter()
        .all(|&ty| ty == ScalarType::F32)
        .then_some(ScalarType::F32)
        .ok_or_else(|| VerificationError::new("fused multiply-add operands must all be f32"))
}

fn literal_type(bits: u32) -> Result<ScalarType, VerificationError> {
    if f32::from_bits(bits).is_finite() {
        return Ok(ScalarType::F32);
    }
    Err(VerificationError::new(
        "f32 literals in typed IR must be finite",
    ))
}

fn valid_identifier(name: &str) -> Result<(), VerificationError> {
    if identifier_shape_ok(name) && !is_c_keyword(name) {
        return Ok(());
    }
    Err(VerificationError::new(format!(
        "`{name}` is not a portable C identifier"
    )))
}

fn identifier_shape_ok(name: &str) -> bool {
    let mut bytes = name.bytes();
    let start = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    start && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// C11 关键字闭集：被发射的名字要原样写进 C 片段，撞上它就不是合法标识符。
/// 写成空白分隔的单串比四十二行数组短，`split_ascii_whitespace` 不分配。
const C_KEYWORDS: &str = "auto break case char const continue default do double else enum extern \
     float for goto if inline int long register restrict return short signed sizeof static struct \
     switch typedef union unsigned void volatile while _Alignas _Atomic _Bool _Complex _Generic \
     _Imaginary _Noreturn _Static_assert _Thread_local";

fn is_c_keyword(name: &str) -> bool {
    C_KEYWORDS
        .split_ascii_whitespace()
        .any(|keyword| keyword == name)
}

// ── 完整模板程序 ─────────────────────────────────────────────────────────

use crate::kernel::{AtomArg, LocalType, Program, Statement};

struct Scope<'a> {
    params: &'a [Parameter],
    locals: Vec<(String, LocalType)>,
}

impl<'a> Scope<'a> {
    fn new(params: &'a [Parameter]) -> Self {
        Self {
            params,
            locals: Vec::new(),
        }
    }

    fn child(&self) -> Self {
        Self {
            params: self.params,
            locals: self.locals.clone(),
        }
    }

    fn known(&self, name: &str) -> bool {
        self.params.iter().any(|param| param.name == name)
            || self.locals.iter().any(|(local, _)| local == name)
    }

    fn local(&self, name: &str) -> Option<LocalType> {
        self.locals
            .iter()
            .find(|(local, _)| local == name)
            .map(|(_, ty)| *ty)
    }
}

pub(crate) fn verify_program(program: &Program<'_>) -> Result<(), VerificationError> {
    verify_signature(program.name(), program.params())?;
    verify_aliases(program)?;
    verify_helpers(program)?;
    verify_statements(program, &mut Scope::new(program.params()), program.body())
}

fn verify_signature(name: &str, params: &[Parameter]) -> Result<(), VerificationError> {
    valid_identifier(name)?;
    check_params(params)?;
    check_generated(params)
}

fn verify_aliases(program: &Program<'_>) -> Result<(), VerificationError> {
    for (idx, alias) in program.aliases().iter().enumerate() {
        valid_identifier(&alias.name)?;
        if program.aliases()[..idx]
            .iter()
            .any(|prior| prior.name == alias.name)
        {
            return Err(VerificationError::new(format!(
                "duplicate type alias `{}`",
                alias.name
            )));
        }
    }
    Ok(())
}

fn verify_helpers(program: &Program<'_>) -> Result<(), VerificationError> {
    for (idx, function) in program.helpers().iter().enumerate() {
        if program.helpers()[..idx]
            .iter()
            .any(|prior| prior.name == function.name)
        {
            return Err(VerificationError::new(format!(
                "duplicate helper `{}`",
                function.name
            )));
        }
        verify_function(program, function)?;
    }
    Ok(())
}

fn verify_function(
    program: &Program<'_>,
    function: &crate::kernel::Function,
) -> Result<(), VerificationError> {
    verify_signature(&function.name, &function.params)?;
    verify_statements(program, &mut Scope::new(&function.params), &function.body)
}

fn verify_statements(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    statements: &[Statement],
) -> Result<(), VerificationError> {
    for statement in statements {
        verify_statement(program, scope, statement)?;
    }
    Ok(())
}

fn verify_statement(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    statement: &Statement,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Decl(_) | Statement::Expr(_) | Statement::Atom(_) => {
            verify_simple_statement(program, scope, statement)
        }
        other => verify_control_statement(program, scope, other),
    }
}

fn verify_simple_statement(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    statement: &Statement,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Decl(local) => declare(scope, local),
        Statement::Expr(expression) => verify_expression(scope, expression),
        Statement::Atom(call) => verify_atom(program, scope, call),
        _ => Ok(()),
    }
}

fn verify_control_statement(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    statement: &Statement,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Block(_) | Statement::For { .. } | Statement::If { .. } => {
            verify_nested_control(program, scope, statement)
        }
        Statement::Pragma { name, value } => verify_pragma(name, *value),
        Statement::Return(value) => verify_optional(scope, value.as_ref()),
        Statement::Continue | Statement::Break => Ok(()),
        _ => Ok(()),
    }
}

fn verify_nested_control(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    statement: &Statement,
) -> Result<(), VerificationError> {
    match statement {
        Statement::Block(body) => verify_statements(program, &mut scope.child(), body),
        Statement::For {
            init,
            condition,
            step,
            body,
        } => verify_for(program, scope, init, condition, step, body),
        Statement::If {
            condition,
            then_branch,
            else_branch,
        } => verify_if(program, scope, condition, then_branch, else_branch),
        _ => Ok(()),
    }
}

fn verify_for(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    init: &Option<Box<Statement>>,
    condition: &Option<Expr>,
    step: &Option<Expr>,
    body: &[Statement],
) -> Result<(), VerificationError> {
    let mut nested = scope.child();
    verify_for_header(program, &mut nested, init, condition, step)?;
    verify_statements(program, &mut nested, body)
}

fn verify_for_header(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    init: &Option<Box<Statement>>,
    condition: &Option<Expr>,
    step: &Option<Expr>,
) -> Result<(), VerificationError> {
    if let Some(init) = init {
        verify_statement(program, scope, init)?;
    }
    verify_optional(scope, condition.as_ref())?;
    verify_optional(scope, step.as_ref())
}

fn verify_if(
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    condition: &Expr,
    then_branch: &[Statement],
    else_branch: &[Statement],
) -> Result<(), VerificationError> {
    verify_expression(scope, condition)?;
    verify_statements(program, &mut scope.child(), then_branch)?;
    verify_statements(program, &mut scope.child(), else_branch)
}

fn verify_pragma(name: &str, value: Option<u32>) -> Result<(), VerificationError> {
    if name != "unroll" || value == Some(0) {
        return Err(VerificationError::new(format!(
            "unsupported pragma `{name}`"
        )));
    }
    Ok(())
}

fn verify_optional(scope: &Scope<'_>, value: Option<&Expr>) -> Result<(), VerificationError> {
    value.map_or(Ok(()), |value| verify_expression(scope, value))
}

fn declare(
    scope: &mut Scope<'_>,
    local: &crate::kernel::LocalDecl,
) -> Result<(), VerificationError> {
    valid_identifier(&local.name)?;
    if scope.known(&local.name) {
        return Err(VerificationError::new(format!(
            "duplicate local `{}`",
            local.name
        )));
    }
    if local.ty.array == Some(0) {
        return Err(VerificationError::new("array length must be positive"));
    }
    verify_optional(scope, local.init.as_ref())?;
    scope.locals.push((local.name.clone(), local.ty));
    Ok(())
}

fn verify_atom(
    program: &Program<'_>,
    scope: &Scope<'_>,
    call: &crate::kernel::AtomCall,
) -> Result<(), VerificationError> {
    let atom = program
        .atom(call)
        .ok_or_else(|| VerificationError::new("atom id is not in the program arena"))?;
    verify_atom_args(scope, call)?;
    verify_atom_ports(atom, call)
}

fn verify_atom_args(
    scope: &Scope<'_>,
    call: &crate::kernel::AtomCall,
) -> Result<(), VerificationError> {
    for arg in &call.args {
        verify_atom_arg(scope, arg)?;
    }
    Ok(())
}

fn verify_atom_arg(scope: &Scope<'_>, arg: &AtomArg) -> Result<(), VerificationError> {
    match arg {
        AtomArg::Expr(value) => verify_expression(scope, value),
        AtomArg::Named { key, value } => verify_named_arg(key, value),
    }
}

fn verify_named_arg(key: &str, value: &str) -> Result<(), VerificationError> {
    valid_identifier(key)?;
    if !identifier_shape_ok(value) && value.parse::<i64>().is_err() {
        return Err(VerificationError::new(format!(
            "invalid atom value `{value}`"
        )));
    }
    Ok(())
}

fn verify_atom_ports(
    atom: &crate::cap::Atom<'_>,
    call: &crate::kernel::AtomCall,
) -> Result<(), VerificationError> {
    for port in &atom.ports {
        if !matches!(call.args.get(port.idx), Some(AtomArg::Expr(_))) {
            return Err(VerificationError::new(format!(
                "atom `{}` port {} requires a positional expression",
                atom.prim(),
                port.idx
            )));
        }
    }
    Ok(())
}

fn verify_expression(scope: &Scope<'_>, expression: &Expr) -> Result<(), VerificationError> {
    for child in expression.children() {
        verify_expression(scope, child)?;
    }
    verify_expression_kind(scope, expression)
}

fn verify_expression_kind(scope: &Scope<'_>, expression: &Expr) -> Result<(), VerificationError> {
    match expression {
        Expr::F32(_) | Expr::Scalar(_) | Expr::Load(_) | Expr::Call { .. } => {
            verify_literal_or_name(scope, expression)
        }
        Expr::Assign { target, .. } => verify_assignment(scope, target),
        Expr::Select {
            when_true,
            when_false,
            ..
        } => verify_selection(scope, when_true, when_false),
        _ => Ok(()),
    }
}

fn verify_literal_or_name(scope: &Scope<'_>, expression: &Expr) -> Result<(), VerificationError> {
    match expression {
        Expr::F32(bits) => literal_type(*bits).map(|_| ()),
        Expr::Scalar(name) | Expr::Load(name) if !scope.known(name) => {
            Err(VerificationError::new(format!("unknown name `{name}`")))
        }
        Expr::Call { function, .. } => valid_identifier(function),
        _ => Ok(()),
    }
}

fn verify_selection(
    scope: &Scope<'_>,
    when_true: &Expr,
    when_false: &Expr,
) -> Result<(), VerificationError> {
    let (left, right) = (
        expression_type(scope, when_true),
        expression_type(scope, when_false),
    );
    if left.is_some() && right.is_some() && left != right {
        return Err(VerificationError::new(
            "selection branches have different types",
        ));
    }
    Ok(())
}

fn verify_assignment(scope: &Scope<'_>, target: &Expr) -> Result<(), VerificationError> {
    if let Expr::Scalar(name) = target
        && scope.local(name).is_some_and(|local| local.constant)
    {
        return Err(VerificationError::new(format!(
            "cannot assign const local `{name}`"
        )));
    }
    Ok(())
}

fn expression_type(scope: &Scope<'_>, expression: &Expr) -> Option<ScalarType> {
    match expression {
        Expr::Scalar(_) | Expr::Load(_) | Expr::Index { .. } => {
            named_expression_type(scope, expression)
        }
        Expr::F32(_)
        | Expr::Int(_)
        | Expr::Cast { .. }
        | Expr::Compare { .. }
        | Expr::Logical { .. }
        | Expr::Not(_) => leaf_expression_type(expression),
        Expr::Neg(_)
        | Expr::Unary { .. }
        | Expr::Update { .. }
        | Expr::Fma { .. }
        | Expr::Select { .. }
        | Expr::Assign { .. } => nested_expression_type(scope, expression),
        Expr::Binary { left, right, .. }
        | Expr::Min { left, right }
        | Expr::Max { left, right } => same_expression_type(scope, left, right),
        Expr::Call { .. } | Expr::Address(_) | Expr::Deref(_) | Expr::BitNot(_) => None,
    }
}

fn named_expression_type(scope: &Scope<'_>, expression: &Expr) -> Option<ScalarType> {
    match expression {
        Expr::Scalar(name) | Expr::Load(name) => named_type(scope, name),
        Expr::Index { base, .. } => indexed_type(scope, base),
        _ => None,
    }
}

fn leaf_expression_type(expression: &Expr) -> Option<ScalarType> {
    match expression {
        Expr::F32(_) => Some(ScalarType::F32),
        Expr::Int(_) => Some(ScalarType::I32),
        Expr::Cast { ty, .. } => Some(*ty),
        Expr::Compare { .. } | Expr::Logical { .. } | Expr::Not(_) => Some(ScalarType::Bool),
        _ => None,
    }
}

fn nested_expression_type(scope: &Scope<'_>, expression: &Expr) -> Option<ScalarType> {
    match expression {
        Expr::Neg(value)
        | Expr::Unary { child: value, .. }
        | Expr::Update { target: value, .. } => expression_type(scope, value),
        Expr::Fma { a, .. } => expression_type(scope, a),
        Expr::Select { when_true, .. } => expression_type(scope, when_true),
        Expr::Assign { value, .. } => expression_type(scope, value),
        _ => None,
    }
}

fn same_expression_type(scope: &Scope<'_>, left: &Expr, right: &Expr) -> Option<ScalarType> {
    let left = expression_type(scope, left);
    if left == expression_type(scope, right) {
        left
    } else {
        None
    }
}

fn named_type(scope: &Scope<'_>, name: &str) -> Option<ScalarType> {
    if let Some(local) = scope.local(name) {
        return (local.pointers == 0 && local.array.is_none()).then_some(local.element);
    }
    scope
        .params
        .iter()
        .find(|param| param.name == name)
        .map(|param| match param.ty {
            ValueType::Scalar(ty) => ty,
            ValueType::Buf { element, .. } => element,
        })
}

fn indexed_type(scope: &Scope<'_>, base: &Expr) -> Option<ScalarType> {
    if let Expr::Scalar(name) = base {
        return named_type(scope, name);
    }
    expression_type(scope, base)
}
