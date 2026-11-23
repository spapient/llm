//! Rust 函数前端：把一个 `for idx in 0..extent` 循环 + 一次写回翻译成 `cutile::kernel::Kernel`。
//!
//! 只接受逐点 `output[idx] = expr` 与归约 `output[at] += expr`；其余形状、类型与表达式在展开期各
//! 报一条带 span 的错，原函数原样保留为 CPU 参考实现。装配期再按 `Kernel::new` 的闭集摊平。

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    BinOp, Error, Expr, ExprAssign, ExprBinary, ExprForLoop, ExprLit, ExprMethodCall,
    ExprPath, ExprUnary, Ident, ItemFn, Lit, Pat, PatIdent, Stmt, Type, TypePath,
    TypeReference, UnOp, spanned::Spanned,
};

/// 一处拒绝语的构造：`Error::new` 直接塞进 `return` 会让每条错误路径都折成四行。
fn fail<T>(span: Span, message: &str) -> syn::Result<T> {
    Err(Error::new(span, message))
}

/// 多处复用的拒绝语：目标形状、缓冲名、可写性、下标，以及四条最长的形状诊断。
const LOOP_IDX: &str = "buffer index must be the loop index";
const NAMED_BUF: &str = "output must be a named mutable buffer";
const NOT_WRITE: &str = "assignment target must be an `&mut [f32]` parameter";
const TARGET: &str = "kernel assignment target must be `output[index]`";
const NO_ARGS: &str = "`#[dsl]` takes no arguments";
const SIGNATURE: &str = "`#[dsl]` supports non-generic, safe, synchronous functions returning `()`";
const LOOP_BODY: &str = "kernel body must contain exactly one `for index in 0..extent` loop";
const ONE_WRITE: &str = "kernel loop must contain exactly one assignment";

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

/// lowering 的成品：循环变量、上界、输出缓冲，以及已摊平的 `cutile::kernel::Write` 表达式。
struct Body {
    idx: String,
    extent: String,
    output: String,
    write: TokenStream2,
}

impl Body {
    fn new(idx: String, extent: String, output: String, write: TokenStream2) -> Self {
        Self {
            idx,
            extent,
            output,
            write,
        }
    }
}

/// 解析函数；无论 lowering 成功还是失败，原函数都照原样再发一遍，失败时多挂一条 `compile_error!`。
pub(crate) fn expand(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let function = match syn::parse::<ItemFn>(item) {
        Ok(function) => function,
        Err(error) => return error.to_compile_error().into(),
    };
    let lowered = if attribute.is_empty() {
        lower(&function)
    } else {
        fail(function.sig.span(), NO_ARGS)
    };
    match lowered.map_err(|error| error.to_compile_error()) {
        Ok(ir_fn) => quote!(#function #ir_fn).into(),
        Err(error) => quote!(#function #error).into(),
    }
}

/// 三步：签名与形状 → 写回 → 装配；每步一步到位，失败即整个 lowering 失败。
fn lower(function: &ItemFn) -> syn::Result<TokenStream2> {
    let params = parse_parameters(&function.sig)?;
    let body = parse_body(single_loop(function)?, &params)?;
    Ok(render(function, &params, &body))
}

fn parse_body(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<Body> {
    let (idx, extent) = parse_loop(loop_stmt, params)?;
    match single_write(loop_stmt)? {
        Write::Map(assign) => map_body(assign, idx, extent, params),
        Write::Reduce(assign) => reduce_body(assign, idx, extent, params),
    }
}

/// 逐点写：输出是「可写缓冲 + 循环下标」，值写进每个元素。
fn map_body(
    assign: &ExprAssign,
    idx: String,
    extent: String,
    params: &[Param],
) -> syn::Result<Body> {
    let output = output_name(assign, &idx, params)?;
    let value = parse_expr(assign.right.as_ref(), &idx, params)?;
    Ok(Body::new(
        idx,
        extent,
        output,
        quote!(::cutile::kernel::Write::Map(#value)),
    ))
}

/// 归约写：先认 `+=`，再取「可写缓冲 + 常量槽」，值累加进那个槽。
fn reduce_body(
    assign: &ExprBinary,
    idx: String,
    extent: String,
    params: &[Param],
) -> syn::Result<Body> {
    reduce_op(assign)?;
    let (output, at) = reduce_output(assign, params)?;
    let value = parse_expr(assign.right.as_ref(), &idx, params)?;
    let write = quote!(::cutile::kernel::Write::Reduce { at: #at, value: #value });
    Ok(Body::new(idx, extent, output, write))
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
    if !rejected.contains(&true) {
        return Ok(());
    }
    fail(signature.span(), SIGNATURE)
}

fn parse_parameters(signature: &syn::Signature) -> syn::Result<Vec<Param>> {
    check_signature(signature)?;
    let mut params: Vec<Param> = Vec::new();
    for argument in &signature.inputs {
        let parameter = parse_parameter(argument)?;
        if params.iter().any(|prior| prior.name == parameter.name) {
            let message = format!("duplicate kernel parameter `{}`", parameter.name);
            return fail(signature.span(), &message);
        }
        params.push(parameter);
    }
    Ok(params)
}

fn parse_parameter(argument: &syn::FnArg) -> syn::Result<Param> {
    let syn::FnArg::Typed(argument) = argument else {
        return fail(argument.span(), "`self` parameters are not supported");
    };
    let pattern = plain_ident(&argument.pat, "kernel parameters")?;
    Ok(Param {
        name: portable_ident(&pattern.ident)?,
        kind: parameter_kind(&argument.ty)?,
    })
}

/// 形参类型闭集：`f32`、`usize`、`&[f32]`、`&mut [f32]`；其余在展开期报一条带 span 的错。
fn parameter_kind(ty: &Type) -> syn::Result<ParamKind> {
    match ty {
        Type::Reference(reference) => reference_kind(reference),
        Type::Path(path) if path.qself.is_none() => scalar_kind(path),
        _ => fail(
            ty.span(),
            "kernel parameters must be `f32`, `usize`, `&[f32]`, or `&mut [f32]`",
        ),
    }
}

fn scalar_kind(path: &TypePath) -> syn::Result<ParamKind> {
    match path.path.get_ident().map(ToString::to_string).as_deref() {
        Some("f32") => Ok(ParamKind::F32),
        Some("usize") => Ok(ParamKind::Usize),
        _ => fail(
            path.span(),
            "scalar kernel parameters must be `f32` or `usize`",
        ),
    }
}

/// 缓冲形参只有一种形状：`&[f32]` 或 `&mut [f32]`；切片元素再单独查一次 `f32`。
fn reference_kind(reference: &TypeReference) -> syn::Result<ParamKind> {
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return fail(reference.span(), "buffers must be `&[f32]` or `&mut [f32]`");
    };
    if !is_f32_path(slice.elem.as_ref()) {
        return fail(slice.elem.span(), "buffer element type must be `f32`");
    }
    Ok(if reference.mutability.is_some() {
        ParamKind::BufWrite
    } else {
        ParamKind::BufRead
    })
}

fn is_f32_path(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32"))
}

/// 参数与循环变量都必须是「光秃秃的标识符」：带 `ref` / `mut` / 子模式的形状摊不成 C 的形参。
fn plain_ident<'a>(pattern: &'a Pat, subject: &str) -> syn::Result<&'a PatIdent> {
    let Pat::Ident(ident) = pattern else {
        return fail(
            pattern.span(),
            &format!("{subject} must use a simple identifier pattern"),
        );
    };
    if ident.by_ref.is_some() || ident.mutability.is_some() || ident.subpat.is_some() {
        return fail(
            ident.span(),
            &format!("{subject} must be an unmodified identifier pattern"),
        );
    }
    Ok(ident)
}

fn portable_ident(ident: &Ident) -> syn::Result<String> {
    let name = ident.to_string();
    if is_c_identifier(&name) {
        return Ok(name);
    }
    fail(ident.span(), "identifiers must be portable non-keyword C")
}

/// 标识符形状 + 不在 C11 关键字表里：两件事分开数条件，复杂度各自留在个位数。
fn is_c_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    let start = bytes
        .next()
        .is_some_and(|byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'_'));
    let rest = bytes.all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_'));
    let padded = format!(" {name} ");
    start && rest && !C_KEYWORDS.contains(&padded)
}

/// C11 关键字闭集：空白分隔、首尾各留一个空格，于是 `contains(" for ")` 就是一次子串查找。
const C_KEYWORDS: &str = " auto break case char const continue default do double else enum \
     extern float for goto if inline int long register restrict return short signed sizeof \
     static struct switch typedef union unsigned void volatile while _Alignas _Atomic _Bool \
     _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local ";

/// 内核体被限定成「恰好一个 `for` 循环」，其余形状一律报错而不是猜一个意思。
fn single_loop(function: &ItemFn) -> syn::Result<&ExprForLoop> {
    let [Stmt::Expr(Expr::ForLoop(loop_stmt), _)] = function.block.stmts.as_slice() else {
        return fail(function.block.span(), LOOP_BODY);
    };
    Ok(loop_stmt)
}

/// 循环体里唯一一条写回：逐点 `out[i] = value`，或归约 `out[at] += value`。
///
/// syn 2 把 `out[at] += value` 解析成 `Expr::Binary`（`op` 是 `AddAssign`），与算术二元同构，
/// 所以这里只按「`=` 还是别的二元」分派，`+=` 的确认留给 `reduce_op`。
enum Write<'a> {
    Map(&'a ExprAssign),
    Reduce(&'a ExprBinary),
}

fn single_write(loop_stmt: &ExprForLoop) -> syn::Result<Write<'_>> {
    let [stmt] = loop_stmt.body.stmts.as_slice() else {
        return fail(loop_stmt.body.span(), ONE_WRITE);
    };
    match stmt {
        Stmt::Expr(Expr::Assign(assign), _) => Ok(Write::Map(assign)),
        Stmt::Expr(Expr::Binary(binary), _) => Ok(Write::Reduce(binary)),
        _ => fail(
            stmt.span(),
            "kernel assignment must be `out[i] = value` or `out[at] += value`",
        ),
    }
}

fn parse_loop(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<(String, String)> {
    if loop_stmt.label.is_some() || !loop_stmt.attrs.is_empty() {
        return fail(
            loop_stmt.span(),
            "labeled loops and loop attributes are not supported",
        );
    }
    let idx = loop_idx(loop_stmt, params)?;
    let extent = loop_extent(loop_stmt, params)?;
    Ok((idx, extent))
}

fn loop_idx(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<String> {
    let pattern = plain_ident(loop_stmt.pat.as_ref(), "loop index")?;
    let idx = portable_ident(&pattern.ident)?;
    if params.iter().any(|param| param.name == idx) {
        return fail(
            pattern.span(),
            "loop index must not shadow a kernel parameter",
        );
    }
    Ok(idx)
}

/// `for idx in 0..extent`：范围必须半开、从零字面量起、终点是 `usize` 形参。
fn loop_extent(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<String> {
    let Expr::Range(range) = loop_stmt.expr.as_ref() else {
        return fail(loop_stmt.expr.span(), "loop range must be `0..extent`");
    };
    if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) {
        return fail(range.limits.span(), "loop range must be half-open (`..`)");
    }
    let (_, end) = range_bounds(range)?;
    extent_param(end, params)
}

fn range_bounds(range: &syn::ExprRange) -> syn::Result<(&Expr, &Expr)> {
    let Some(start) = range.start.as_deref() else {
        return fail(range.span(), "loop range must start at zero");
    };
    zero_start(start)?;
    let Some(end) = range.end.as_deref() else {
        return fail(range.span(), "loop range requires an extent");
    };
    Ok((start, end))
}

fn zero_start(start: &Expr) -> syn::Result<()> {
    let Expr::Lit(ExprLit {
        lit: Lit::Int(lit), ..
    }) = start
    else {
        return fail(start.span(), "loop range must start at zero");
    };
    if lit.base10_parse::<u64>()? == 0 {
        return Ok(());
    }
    fail(lit.span(), "loop range must start at zero")
}

fn extent_param(end: &Expr, params: &[Param]) -> syn::Result<String> {
    let name = portable_ident(path_ident(end, "loop extent must be a `usize` parameter")?)?;
    if param_kind(params, &name) == Some(ParamKind::Usize) {
        return Ok(name);
    }
    fail(end.span(), "loop extent must name a `usize` parameter")
}

/// 把 `&Expr` 当成「一个光秃秃的名字」读；不是就报给定的那句话。
fn path_ident<'a>(expression: &'a Expr, message: &str) -> syn::Result<&'a Ident> {
    let Expr::Path(ExprPath { path, .. }) = expression else {
        return fail(expression.span(), message);
    };
    path.get_ident()
        .ok_or_else(|| Error::new(path.span(), message))
}

/// 下标只能是循环变量本身：`output[idx + 1]`、`output[0]` 都会落到这里。
fn at_loop_idx(expression: &Expr, idx: &str) -> syn::Result<()> {
    let at = path_ident(expression, LOOP_IDX)?;
    if portable_ident(at)? == idx {
        return Ok(());
    }
    fail(expression.span(), LOOP_IDX)
}

/// 赋值目标是「可写缓冲 + 循环下标」两件事；任一件不成立都在目标处报错。
fn output_name(assign: &ExprAssign, idx: &str, params: &[Param]) -> syn::Result<String> {
    let Expr::Index(target) = assign.left.as_ref() else {
        return fail(assign.left.span(), TARGET);
    };
    at_loop_idx(target.index.as_ref(), idx)?;
    let buf = path_ident(target.expr.as_ref(), NAMED_BUF)?;
    require_write(assign.left.span(), params, &portable_ident(buf)?)
}

/// 可写性单独成闸：调用方已经带了 `?`，再叠一个分支就超复杂度上限。
fn require_write(span: Span, params: &[Param], output: &str) -> syn::Result<String> {
    if param_kind(params, output) == Some(ParamKind::BufWrite) {
        return Ok(output.to_string());
    }
    fail(span, NOT_WRITE)
}

/// 归约写回的槽位：`out[0]`、`out[7]` 这类常量下标，读成一个放得进 C 下标的 `u32`。
fn reduce_slot(buf_expr: &Expr, at: &Expr) -> syn::Result<(String, u32)> {
    let Expr::Lit(ExprLit {
        lit: Lit::Int(lit), ..
    }) = at
    else {
        return fail(at.span(), "reduction slot must be a constant integer");
    };
    let slot = lit
        .base10_parse::<u32>()
        .map_err(|_| Error::new(lit.span(), "reduction slot must fit in `u32`"))?;
    let buf = path_ident(buf_expr, NAMED_BUF)?;
    Ok((portable_ident(buf)?, slot))
}

/// 归约只认 `+=`：`-=`、`*=` 这类复合赋值在同一处拒绝，避免误判成逐点写。
fn reduce_op(assign: &ExprBinary) -> syn::Result<()> {
    if matches!(assign.op, BinOp::AddAssign(_)) {
        return Ok(());
    }
    fail(assign.op.span(), "reduction assignment must be `+=`")
}

/// 归约目标是「可写缓冲 + 常量槽」两件事；任一件不成立都在目标处报错。
fn reduce_output(assign: &ExprBinary, params: &[Param]) -> syn::Result<(String, u32)> {
    let Expr::Index(target) = assign.left.as_ref() else {
        return fail(assign.left.span(), TARGET);
    };
    let (output, at) = reduce_slot(target.expr.as_ref(), target.index.as_ref())?;
    Ok((require_write(assign.left.span(), params, &output)?, at))
}

fn param_kind(params: &[Param], name: &str) -> Option<ParamKind> {
    params
        .iter()
        .find(|param| param.name == name)
        .map(|param| param.kind)
}

/// 形参类型 → `cutile::kernel::ValueType`：`&[f32]` 只读、`&mut [f32]` 可写、其余是标量。
fn ir_value_type(kind: ParamKind) -> TokenStream2 {
    let element = quote!(::cutile::kernel::ScalarType::F32);
    match kind {
        ParamKind::BufRead => quote! {
            ::cutile::kernel::ValueType::Buf { element: #element, access: ::cutile::kernel::Access::Read }
        },
        ParamKind::BufWrite => quote! {
            ::cutile::kernel::ValueType::Buf {
                element: #element,
                access: ::cutile::kernel::Access::ReadWrite,
            }
        },
        ParamKind::F32 => quote!(::cutile::kernel::ValueType::Scalar(#element)),
        ParamKind::Usize => quote!(::cutile::kernel::ValueType::Scalar(
            ::cutile::kernel::ScalarType::Usize
        )),
    }
}

/// `#[dsl]` 的 IR 构造器生成：把已校验的形状摊平成 `Kernel::new` 与 `Write` 的 token。
fn render(function: &ItemFn, params: &[Param], body: &Body) -> TokenStream2 {
    let name = function.sig.ident.to_string();
    let generated = format_ident!("__dsl_ir_{name}");
    let visibility = &function.vis;
    let ir_params = params
        .iter()
        .map(|parameter| {
            let name = &parameter.name;
            let ty = ir_value_type(parameter.kind);
            quote! { ::cutile::kernel::Parameter { name: #name.to_string(), ty: #ty } }
        })
        .collect::<Vec<_>>();
    let (idx, extent, output, write) = (&body.idx, &body.extent, &body.output, &body.write);
    quote! {
        #[doc(hidden)]
        #[allow(missing_docs)]
        #visibility fn #generated() -> ::cutile::kernel::Kernel {
            ::cutile::kernel::Kernel::new(#name, vec![#(#ir_params),*], #idx, #extent, #output, #write)
        }
    }
}

/// 表达式这一族的统一拒绝语：把「能用什么」写在一处，免得三个分支各说一套。
const UNSUPPORTED: &str = "unsupported expression: f32 params, buf loads, finite literals, + - * /, abs, min, max";
const ARITHMETIC: &str = "supported arithmetic is `+`, `-`, `*`, and `/`";
const METHODS: [(&str, usize); 3] = [("abs", 0), ("min", 1), ("max", 1)];

/// 表达式三层：括号剥壳 → 叶子（字面量/标量/加载）→ 组合子（二元、取负）。
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
        Expr::Index(target) => buf_load(target.expr.as_ref(), target.index.as_ref(), idx, params),
        other => parse_compound(other, idx, params),
    }
}

fn parse_compound(expression: &Expr, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    match expression {
        Expr::Binary(binary) => binary_expr(binary, idx, params),
        Expr::Unary(unary) => negate(unary, idx, params),
        Expr::MethodCall(method) => method_call(method, idx, params),
        Expr::Call(call) => match call.func.as_ref() {
            Expr::MethodCall(method) => method_call(method, idx, params),
            other => fail(other.span(), UNSUPPORTED),
        },
        _ => fail(expression.span(), UNSUPPORTED),
    }
}

/// 方法组合子闭集：`abs()` 一元，`min(y)` / `max(y)` 二元；其余方法名在展开期打回。
fn method_call(method: &ExprMethodCall, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let name = method.method.to_string();
    let Some((_, arity)) = METHODS.iter().find(|(known, _)| *known == name.as_str()) else {
        return fail(method.method.span(), "unsupported kernel method");
    };
    if method.args.len() != *arity {
        return fail(
            method.method.span(),
            &format!("`{name}` takes {} argument(s)", arity),
        );
    }
    let receiver = parse_expr(&method.receiver, idx, params)?;
    match name.as_str() {
        "abs" => Ok(quote!(::cutile::kernel::Expr::Abs(Box::new(#receiver)))),
        "min" | "max" => {
            let argument = parse_expr(&method.args[0], idx, params)?;
            let variant = Ident::new(if name == "min" { "Min" } else { "Max" }, method.method.span());
            Ok(quote! {
                ::cutile::kernel::Expr::#variant {
                    left: Box::new(#receiver),
                    right: Box::new(#argument),
                }
            })
        }
        _ => fail(method.method.span(), "unsupported kernel method"),
    }
}

fn float_literal(lit: &ExprLit) -> syn::Result<TokenStream2> {
    let Lit::Float(literal) = &lit.lit else {
        return fail(lit.span(), UNSUPPORTED);
    };
    let value = literal.base10_parse::<f32>()?;
    if !value.is_finite() {
        return fail(literal.span(), "kernel float literals must be finite");
    }
    let bits = value.to_bits();
    Ok(quote!(::cutile::kernel::Expr::F32(#bits)))
}

/// 名字在表达式里要么是 `f32` 标量，要么是「必须带下标读取」的缓冲；循环变量只能当下标用。
fn scalar_use(path: &ExprPath, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let name = scalar_ident(path, idx)?;
    let kind = match param_kind(params, &name) {
        Some(kind) => kind,
        None => return fail(path.span(), &format!("unknown kernel value `{name}`")),
    };
    match kind {
        ParamKind::F32 => Ok(quote!(::cutile::kernel::Expr::Scalar(#name.to_string()))),
        ParamKind::BufRead | ParamKind::BufWrite => fail(
            path.span(),
            "buffer values must be read with `buffer[loop_idx]`",
        ),
        ParamKind::Usize => fail(path.span(), "`usize` is not valid in an f32 expression"),
    }
}

fn scalar_ident(path: &ExprPath, idx: &str) -> syn::Result<String> {
    let ident = path
        .path
        .get_ident()
        .ok_or_else(|| Error::new(path.span(), "kernel expressions use simple names"))?;
    let name = portable_ident(ident)?;
    if name == idx {
        return fail(path.span(), "loop index is only valid as a buffer index");
    }
    Ok(name)
}

/// 缓冲加载：名字必须是缓冲形参，下标必须是循环变量。
fn buf_load(buf: &Expr, at: &Expr, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let ident = path_ident(buf, "only named buffers can be loaded")?;
    let name = portable_ident(ident)?;
    at_loop_idx(at, idx)?;
    if matches!(
        param_kind(params, &name),
        Some(ParamKind::BufRead | ParamKind::BufWrite)
    ) {
        return Ok(quote!(::cutile::kernel::Expr::Load(#name.to_string())));
    }
    fail(ident.span(), &format!("`{name}` is not a buffer parameter"))
}

fn negate(unary: &ExprUnary, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    if !matches!(unary.op, UnOp::Neg(_)) {
        return fail(unary.span(), UNSUPPORTED);
    }
    let value = parse_expr(unary.expr.as_ref(), idx, params)?;
    Ok(quote!(::cutile::kernel::Expr::Neg(Box::new(#value))))
}

fn binary_expr(binary: &ExprBinary, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let operator = operator(&binary.op)?;
    let left = parse_expr(binary.left.as_ref(), idx, params)?;
    let right = parse_expr(binary.right.as_ref(), idx, params)?;
    Ok(quote! {
        ::cutile::kernel::Expr::Binary {
            op: ::cutile::kernel::Arith::#operator,
            left: Box::new(#left),
            right: Box::new(#right),
        }
    })
}

/// 四则运算符闭集：查表命中就照抄名字，其余在运算符处报错。
fn operator(op: &BinOp) -> syn::Result<Ident> {
    let name = match op {
        BinOp::Add(_) => "Add",
        BinOp::Sub(_) => "Sub",
        BinOp::Mul(_) => "Mul",
        BinOp::Div(_) => "Div",
        _ => return fail(op.span(), ARITHMETIC),
    };
    Ok(Ident::new(name, op.span()))
}
