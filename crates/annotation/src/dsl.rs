//! Rust function front end for the initial typed pointwise kernel IR.
//!
//! 这一族只接受「一个 `for idx in 0..extent` 循环 + 一次 `output[idx] = expr` 赋值」的形状。
//! 收窄输入比猜一个意思便宜：被拒绝的函数原样保留为 CPU 参考实现，只多一条编译期错误。

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    BinOp, Error, Expr, ExprAssign, ExprBinary, ExprForLoop, ExprLit, ExprPath, ExprRange,
    ExprUnary, Ident, ItemFn, Lit, Pat, PatIdent, Stmt, Type, TypePath, TypeReference, UnOp,
    spanned::Spanned,
};

/// 一个形参在 C 侧的身份：只读缓冲、可写缓冲、`f32` 标量、`usize` 长度。
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    BufRead,
    BufWrite,
    F32,
    Usize,
}

struct Param {
    name: String,
    kind: ParamKind,
}

/// 一次 lowering 的成品：循环变量、上界、输出缓冲，以及要写进每个元素的值。
struct Body {
    idx: String,
    extent: String,
    output: String,
    value: TokenStream2,
}

/// Parse, type-check, and lower the supported Rust function subset.
pub(crate) fn expand(attribute: TokenStream, item: TokenStream) -> TokenStream {
    match syn::parse::<ItemFn>(item) {
        Ok(function) => expand_function(&function, attribute.is_empty()).into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// 无论 lowering 成功还是失败，原函数都照原样再发一遍：失败时多挂一条 `compile_error!`。
fn expand_function(function: &ItemFn, bare: bool) -> TokenStream2 {
    match lowered(function, bare) {
        Ok(ir_fn) => quote!(#function #ir_fn),
        Err(error) => {
            let error = error.to_compile_error();
            quote!(#function #error)
        }
    }
}

fn lowered(function: &ItemFn, bare: bool) -> syn::Result<TokenStream2> {
    if !bare {
        return Err(Error::new(
            function.sig.span(),
            "`#[dsl]` takes no arguments",
        ));
    }
    lower(function)
}

/// 四步：签名 → 循环 → 赋值 → 装配。每步一步到位，失败即整个 lowering 失败。
fn lower(function: &ItemFn) -> syn::Result<TokenStream2> {
    let params = parse_parameters(&function.sig)?;
    let loop_stmt = single_loop(function)?;
    let body = parse_body(loop_stmt, &params)?;
    Ok(render(function, &params, &body))
}

fn parse_body(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<Body> {
    let (idx, extent) = parse_loop(loop_stmt, params)?;
    let assign = single_assignment(loop_stmt)?;
    let (output, value) = parse_assignment(assign, &idx, params)?;
    Ok(Body {
        idx,
        extent,
        output,
        value,
    })
}

/// 这一族只接受「非泛型、安全、同步、返回 `()`」的函数；八条修饰符各占一行而不是串成一条 `||` 链。
fn check_signature(signature: &syn::Signature) -> syn::Result<()> {
    let rejected = [
        signature.asyncness.is_some(),
        signature.constness.is_some(),
        signature.unsafety.is_some(),
        signature.abi.is_some(),
        signature.variadic.is_some(),
        signature.generics.where_clause.is_some(),
        !signature.generics.params.is_empty(),
        !matches!(signature.output, syn::ReturnType::Default),
    ];
    if rejected.into_iter().any(|rejected| rejected) {
        return Err(Error::new(
            signature.span(),
            "`#[dsl]` currently supports non-generic, safe, synchronous functions returning `()`",
        ));
    }
    Ok(())
}

fn parse_parameters(signature: &syn::Signature) -> syn::Result<Vec<Param>> {
    check_signature(signature)?;
    let params = signature
        .inputs
        .iter()
        .map(parse_parameter)
        .collect::<syn::Result<Vec<_>>>()?;
    unique_parameters(&params, signature)?;
    Ok(params)
}

fn parse_parameter(argument: &syn::FnArg) -> syn::Result<Param> {
    let syn::FnArg::Typed(argument) = argument else {
        return Err(Error::new(
            argument.span(),
            "`self` parameters are not supported",
        ));
    };
    let pattern = plain_ident(&argument.pat, "kernel parameters")?;
    Ok(Param {
        name: portable_ident(&pattern.ident)?,
        kind: parameter_kind(&argument.ty)?,
    })
}

/// 参数与循环变量都必须是「光秃秃的标识符」：带 `ref` / `mut` / 子模式的形状摊不成 C 的形参。
fn plain_ident<'a>(pattern: &'a Pat, subject: &str) -> syn::Result<&'a PatIdent> {
    let Pat::Ident(ident) = pattern else {
        return Err(Error::new(
            pattern.span(),
            format!("{subject} must use a simple identifier pattern"),
        ));
    };
    if ident.by_ref.is_some() || ident.mutability.is_some() || ident.subpat.is_some() {
        return Err(Error::new(
            ident.span(),
            format!("{subject} must be an unmodified identifier pattern"),
        ));
    }
    Ok(ident)
}

fn portable_ident(ident: &Ident) -> syn::Result<String> {
    let name = ident.to_string();
    if is_c_identifier(&name) {
        Ok(name)
    } else {
        Err(Error::new(
            ident.span(),
            "kernel identifiers must be portable non-keyword C identifiers",
        ))
    }
}

/// 标识符形状 + 不在 C 关键字表里：两件事分开数条件，复杂度各自留在个位数。
fn is_c_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    let start = bytes
        .next()
        .is_some_and(|byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'_'));
    let rest = bytes.all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_'));
    start && rest && !C_KEYWORDS.contains(&name)
}

/// C11 关键字闭集：这些名字会被写进 C 片段，撞上它就不是合法标识符。
const C_KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while", "_Alignas", "_Atomic", "_Bool", "_Complex",
    "_Generic", "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
];

/// 内核体被限定成「恰好一个 `for` 循环」，其余形状一律报错而不是猜一个意思。
fn single_loop(function: &ItemFn) -> syn::Result<&ExprForLoop> {
    let [Stmt::Expr(Expr::ForLoop(loop_stmt), _)] = function.block.stmts.as_slice() else {
        return Err(Error::new(
            function.block.span(),
            "kernel body must contain exactly one `for index in 0..extent` loop",
        ));
    };
    Ok(loop_stmt)
}

fn single_assignment(loop_stmt: &ExprForLoop) -> syn::Result<&ExprAssign> {
    let [Stmt::Expr(Expr::Assign(assign), _)] = loop_stmt.body.stmts.as_slice() else {
        return Err(Error::new(
            loop_stmt.body.span(),
            "kernel loop must contain exactly one indexed assignment",
        ));
    };
    Ok(assign)
}

fn parse_loop(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<(String, String)> {
    no_loop_decorations(loop_stmt)?;
    let idx = loop_index(loop_stmt, params)?;
    let extent = loop_extent(loop_stmt, params)?;
    Ok((idx, extent))
}

fn no_loop_decorations(loop_stmt: &ExprForLoop) -> syn::Result<()> {
    if loop_stmt.label.is_none() && loop_stmt.attrs.is_empty() {
        return Ok(());
    }
    Err(Error::new(
        loop_stmt.span(),
        "labeled loops and loop attributes are not supported",
    ))
}

fn loop_index(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<String> {
    let pattern = plain_ident(loop_stmt.pat.as_ref(), "loop index")?;
    let idx = portable_ident(&pattern.ident)?;
    if params.iter().any(|param| param.name == idx) {
        return Err(Error::new(
            pattern.span(),
            "loop index must not shadow a kernel parameter",
        ));
    }
    Ok(idx)
}

fn loop_extent(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<String> {
    let range = half_open_range(loop_stmt.expr.as_ref())?;
    zero_start(range)?;
    extent_name(range, params)
}

fn half_open_range(expression: &Expr) -> syn::Result<&ExprRange> {
    let Expr::Range(range) = expression else {
        return Err(Error::new(
            expression.span(),
            "loop range must be `0..extent`",
        ));
    };
    if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) {
        return Err(Error::new(
            range.limits.span(),
            "loop range must be half-open (`..`)",
        ));
    }
    Ok(range)
}

fn zero_start(range: &ExprRange) -> syn::Result<()> {
    let Some(start) = range.start.as_deref() else {
        return Err(Error::new(range.span(), "loop range must start at zero"));
    };
    let Expr::Lit(ExprLit {
        lit: Lit::Int(start),
        ..
    }) = start
    else {
        return Err(Error::new(start.span(), "loop range must start at zero"));
    };
    if start.base10_parse::<u64>()? != 0 {
        return Err(Error::new(start.span(), "loop range must start at zero"));
    }
    Ok(())
}

fn extent_name(range: &ExprRange, params: &[Param]) -> syn::Result<String> {
    let end = range
        .end
        .as_deref()
        .ok_or_else(|| Error::new(range.span(), "loop range requires an extent"))?;
    let name = path_ident(end, "loop extent must be a `usize` parameter")?;
    let name = portable_ident(name)?;
    if !params
        .iter()
        .any(|param| param.name == name && param.kind == ParamKind::Usize)
    {
        return Err(Error::new(
            end.span(),
            "loop extent must name a `usize` parameter",
        ));
    }
    Ok(name)
}

/// 把 `&Expr` 当成「一个光秃秃的名字」读；不是就报给定的那句话。
fn path_ident<'a>(expression: &'a Expr, message: &str) -> syn::Result<&'a Ident> {
    let Expr::Path(ExprPath { path, .. }) = expression else {
        return Err(Error::new(expression.span(), message));
    };
    path.get_ident()
        .ok_or_else(|| Error::new(path.span(), message))
}

/// 下标只能是循环变量本身：`output[idx + 1]`、`output[0]` 都会落到这里。
fn at_loop_index(expression: &Expr, idx: &str) -> syn::Result<()> {
    let at = path_ident(expression, "buffer index must be the loop index")?;
    if portable_ident(at)? == idx {
        return Ok(());
    }
    Err(Error::new(
        expression.span(),
        "buffer index must be the loop index",
    ))
}

fn parse_assignment(
    assign: &ExprAssign,
    idx: &str,
    params: &[Param],
) -> syn::Result<(String, TokenStream2)> {
    let output = output_name(assign, idx, params)?;
    Ok((output, parse_expr(assign.right.as_ref(), idx, params)?))
}

fn output_name(assign: &ExprAssign, idx: &str, params: &[Param]) -> syn::Result<String> {
    let Expr::Index(target) = assign.left.as_ref() else {
        return Err(Error::new(
            assign.left.span(),
            "kernel assignment target must be `output[index]`",
        ));
    };
    let buf = path_ident(target.expr.as_ref(), "output must be a named mutable buffer")?;
    let output = portable_ident(buf)?;
    at_loop_index(target.index.as_ref(), idx)?;
    if param_kind(params, &output) != Some(ParamKind::BufWrite) {
        return Err(Error::new(
            assign.left.span(),
            "assignment target must be an `&mut [f32]` parameter",
        ));
    }
    Ok(output)
}

fn param_kind(params: &[Param], name: &str) -> Option<ParamKind> {
    params
        .iter()
        .find(|param| param.name == name)
        .map(|param| param.kind)
}

fn is_buffer_param(params: &[Param], name: &str) -> bool {
    matches!(
        param_kind(params, name),
        Some(ParamKind::BufRead | ParamKind::BufWrite)
    )
}

/// 表达式分三层：括号剥壳 → 叶子（字面量/名字/加载）→ 组合子（二元、取负）。
fn parse_expr(expression: &Expr, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    match expression {
        Expr::Paren(paren) => parse_expr(paren.expr.as_ref(), idx, params),
        Expr::Group(group) => parse_expr(group.expr.as_ref(), idx, params),
        other => parse_leaf(other, idx, params),
    }
}

fn parse_leaf(expression: &Expr, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    match expression {
        Expr::Lit(lit) => float_literal(lit),
        Expr::Path(path) => scalar_use(path, idx, params),
        Expr::Index(target) => buffer_load(target.expr.as_ref(), target.index.as_ref(), idx, params),
        other => parse_compound(other, idx, params),
    }
}

fn parse_compound(expression: &Expr, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    match expression {
        Expr::Binary(binary) => binary_expr(binary, idx, params),
        Expr::Unary(unary) => negate(unary, idx, params),
        _ => Err(unsupported(expression)),
    }
}

fn unsupported(expression: &Expr) -> Error {
    Error::new(expression.span(), UNSUPPORTED)
}

/// 表达式这一族的统一拒绝语：把「能用什么」写在一处，免得三个分支各说一套。
const UNSUPPORTED: &str =
    "unsupported expression; use f32 parameters, buffer loads, finite float literals, and basic arithmetic";

fn float_literal(lit: &ExprLit) -> syn::Result<TokenStream2> {
    let Lit::Float(literal) = &lit.lit else {
        return Err(Error::new(lit.span(), UNSUPPORTED));
    };
    let value = literal.base10_parse::<f32>()?;
    if !value.is_finite() {
        return Err(Error::new(
            literal.span(),
            "kernel float literals must be finite",
        ));
    }
    let bits = value.to_bits();
    Ok(quote!(::cutile::ir::Expr::F32(#bits)))
}

fn scalar_use(path: &ExprPath, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let name = scalar_name(path, idx, params)?;
    Ok(quote!(::cutile::ir::Expr::Scalar(#name.to_string())))
}

fn scalar_name(path: &ExprPath, idx: &str, params: &[Param]) -> syn::Result<String> {
    let name = scalar_ident(path, idx)?;
    reject_non_scalar(path, params, &name)
}

fn scalar_ident(path: &ExprPath, idx: &str) -> syn::Result<String> {
    let ident = path
        .path
        .get_ident()
        .ok_or_else(|| Error::new(path.span(), "kernel expressions use simple names"))?;
    let name = portable_ident(ident)?;
    if name == idx {
        return Err(Error::new(
            path.span(),
            "loop index is only valid as a buffer index",
        ));
    }
    Ok(name)
}

fn reject_non_scalar(
    path: &ExprPath,
    params: &[Param],
    name: &str,
) -> syn::Result<String> {
    let kind = param_kind(params, name)
        .ok_or_else(|| Error::new(path.span(), format!("unknown kernel value `{name}`")))?;
    match kind {
        ParamKind::F32 => Ok(name.to_string()),
        ParamKind::BufRead | ParamKind::BufWrite => Err(Error::new(
            path.span(),
            "buffer values must be read with `buffer[loop_index]`",
        )),
        ParamKind::Usize => Err(Error::new(
            path.span(),
            "`usize` is not valid in an f32 expression",
        )),
    }
}

fn buffer_load(
    buf: &Expr,
    at: &Expr,
    idx: &str,
    params: &[Param],
) -> syn::Result<TokenStream2> {
    let name = loaded_name(buf, at, idx, params)?;
    Ok(quote!(::cutile::ir::Expr::Load(#name.to_string())))
}

fn loaded_name(buf: &Expr, at: &Expr, idx: &str, params: &[Param]) -> syn::Result<String> {
    let ident = path_ident(buf, "only named buffers can be loaded")?;
    let name = portable_ident(ident)?;
    at_loop_index(at, idx)?;
    if !is_buffer_param(params, &name) {
        return Err(Error::new(
            ident.span(),
            format!("`{name}` is not a buffer parameter"),
        ));
    }
    Ok(name)
}

fn negate(unary: &ExprUnary, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    if !matches!(unary.op, UnOp::Neg(_)) {
        return Err(Error::new(unary.span(), UNSUPPORTED));
    }
    let value = parse_expr(unary.expr.as_ref(), idx, params)?;
    Ok(quote!(::cutile::ir::Expr::Neg(Box::new(#value))))
}

fn binary_expr(binary: &ExprBinary, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let operator = operator(&binary.op)?;
    let left = parse_expr(binary.left.as_ref(), idx, params)?;
    let right = parse_expr(binary.right.as_ref(), idx, params)?;
    Ok(quote! {
        ::cutile::ir::Expr::Binary {
            op: ::cutile::ir::BinaryOp::#operator,
            left: Box::new(#left),
            right: Box::new(#right),
        }
    })
}

/// `+`/`-` 与 `*`/`/` 分开查表：一个 `match` 四条臂就顶到复杂度上限，拆成两半各自留余量。
fn operator(op: &BinOp) -> syn::Result<Ident> {
    let name = additive(op)
        .or_else(|| multiplicative(op))
        .ok_or_else(|| Error::new(op.span(), "supported arithmetic is `+`, `-`, `*`, and `/`"))?;
    Ok(Ident::new(name, op.span()))
}

fn additive(op: &BinOp) -> Option<&'static str> {
    match op {
        BinOp::Add(_) => Some("Add"),
        BinOp::Sub(_) => Some("Sub"),
        _ => None,
    }
}

fn multiplicative(op: &BinOp) -> Option<&'static str> {
    match op {
        BinOp::Mul(_) => Some("Mul"),
        BinOp::Div(_) => Some("Div"),
        _ => None,
    }
}

fn ir_value_type(kind: ParamKind) -> TokenStream2 {
    match kind {
        ParamKind::BufRead => quote! {
            ::cutile::ir::ValueType::Buffer {
                element: ::cutile::ir::ScalarType::F32,
                access: ::cutile::ir::Access::Read,
            }
        },
        ParamKind::BufWrite => quote! {
            ::cutile::ir::ValueType::Buffer {
                element: ::cutile::ir::ScalarType::F32,
                access: ::cutile::ir::Access::ReadWrite,
            }
        },
        ParamKind::F32 => quote! {
            ::cutile::ir::ValueType::Scalar(::cutile::ir::ScalarType::F32)
        },
        ParamKind::Usize => quote! {
            ::cutile::ir::ValueType::Scalar(::cutile::ir::ScalarType::Usize)
        },
    }
}

fn parameter_kind(ty: &Type) -> syn::Result<ParamKind> {
    match ty {
        Type::Reference(reference) => reference_kind(reference),
        Type::Path(path) if path.qself.is_none() => scalar_kind(path),
        _ => Err(Error::new(
            ty.span(),
            "kernel parameters must be `f32`, `usize`, `&[f32]`, or `&mut [f32]`",
        )),
    }
}

fn scalar_kind(path: &TypePath) -> syn::Result<ParamKind> {
    match path.path.get_ident().map(ToString::to_string).as_deref() {
        Some("f32") => Ok(ParamKind::F32),
        Some("usize") => Ok(ParamKind::Usize),
        _ => Err(Error::new(
            path.span(),
            "scalar kernel parameters must be `f32` or `usize`",
        )),
    }
}

fn reference_kind(reference: &TypeReference) -> syn::Result<ParamKind> {
    f32_element(slice_element(reference)?)?;
    Ok(if reference.mutability.is_some() {
        ParamKind::BufWrite
    } else {
        ParamKind::BufRead
    })
}

fn slice_element(reference: &TypeReference) -> syn::Result<&Type> {
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return Err(Error::new(
            reference.span(),
            "buffer parameters must be slices (`&[f32]` or `&mut [f32]`)",
        ));
    };
    Ok(slice.elem.as_ref())
}

fn f32_element(ty: &Type) -> syn::Result<()> {
    if is_float(ty) {
        return Ok(());
    }
    Err(Error::new(ty.span(), "buffer element type must be `f32`"))
}

fn is_float(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.qself.is_none() && path.path.is_ident("f32")
}

fn unique_parameters(params: &[Param], signature: &syn::Signature) -> syn::Result<()> {
    for (idx, parameter) in params.iter().enumerate() {
        if params[..idx]
            .iter()
            .any(|prior| prior.name == parameter.name)
        {
            return Err(Error::new(
                signature.span(),
                format!("duplicate kernel parameter `{}`", parameter.name),
            ));
        }
    }
    Ok(())
}

fn render(function: &ItemFn, params: &[Param], body: &Body) -> TokenStream2 {
    let ir_params = params.iter().map(|parameter| {
        let name = &parameter.name;
        let ty = ir_value_type(parameter.kind);
        quote! {
            ::cutile::ir::Parameter {
                name: #name.to_string(),
                ty: #ty,
            }
        }
    });
    let fn_name = &function.sig.ident;
    let generated_name = format_ident!("__dsl_ir_{}", fn_name);
    let visibility = &function.vis;
    let idx = &body.idx;
    let extent = &body.extent;
    let output = &body.output;
    let value = &body.value;
    quote! {
        #[doc(hidden)]
        #[allow(missing_docs)]
        #visibility fn #generated_name() -> ::cutile::ir::Kernel {
            ::cutile::ir::Kernel::new(
                #fn_name.to_string(),
                vec![#(#ir_params),*],
                #idx,
                #extent,
                #output,
                #value,
            )
        }
    }
}
