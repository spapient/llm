//! Rust function front end for the initial typed pointwise kernel IR.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    BinOp, Error, Expr, ExprAssign, ExprBinary, ExprForLoop, ExprIndex, ExprLit, ExprPath, Ident,
    ItemFn, Lit, Pat, Stmt, Type, TypeReference, UnOp, spanned::Spanned,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    BufferRead,
    BufferWrite,
    F32,
    Usize,
}

struct Param {
    name: String,
    kind: ParamKind,
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

/// C11 关键字闭集（`kernel identifiers` 会被写进 C 片段，撞上它就不是合法标识符）。
const C_KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while", "_Alignas", "_Atomic", "_Bool", "_Complex",
    "_Generic", "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
];

/// Parse, type-check, and lower the supported Rust function subset.
pub(crate) fn expand(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let function = match syn::parse::<ItemFn>(item) {
        Ok(function) => function,
        Err(error) => return error.to_compile_error().into(),
    };

    match lower(&attribute, &function) {
        Ok(ir_fn) => quote!(#function #ir_fn).into(),
        Err(error) => {
            let error = error.to_compile_error();
            quote!(#function #error).into()
        }
    }
}

fn lower(attribute: &TokenStream, function: &ItemFn) -> syn::Result<TokenStream2> {
    if !attribute.is_empty() {
        return Err(Error::new(
            function.sig.span(),
            "`#[dsl]` takes no arguments",
        ));
    }
    check_signature(&function.sig)?;
    let params = parse_parameters(&function.sig)?;
    let loop_stmt = single_loop(function)?;
    let (index, extent) = parse_loop(loop_stmt, &params)?;
    let (output, value) = parse_assignment(single_assignment(loop_stmt)?, &index, &params)?;
    Ok(render(function, &params, Body {
        index: &index,
        extent: &extent,
        output: &output,
        value,
    }))
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
    let pattern = identifier_pattern(&argument.pat)?;
    Ok(Param {
        name: portable_ident(&pattern.ident)?,
        kind: parameter_kind(&argument.ty)?,
    })
}

/// 参数与循环变量都必须是「光秃秃的标识符」：带 `ref` / `mut` / 子模式的形状无法摊平成 C 的形参。
fn identifier_pattern(pattern: &Pat) -> syn::Result<&syn::PatIdent> {
    let Pat::Ident(pattern) = pattern else {
        return Err(Error::new(
            pattern.span(),
            "kernel parameters must use simple identifier patterns",
        ));
    };
    if pattern.by_ref.is_some() || pattern.mutability.is_some() || pattern.subpat.is_some() {
        return Err(Error::new(
            pattern.span(),
            "kernel parameters must use unmodified identifier patterns",
        ));
    }
    Ok(pattern)
}

/// 内核体被限定成「恰好一个 `for index in 0..extent` 循环」，其余形状一律报错而不是猜一个意思。
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

/// 这是一次 lowering 的成品：起点、终点与两侧的表达式，装配成 `Kernel::new` 的调用。
struct Body<'a> {
    index: &'a str,
    extent: &'a str,
    output: &'a str,
    value: TokenStream2,
}

fn render(function: &ItemFn, params: &[Param], body: Body<'_>) -> TokenStream2 {
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
    let index = body.index;
    let extent = body.extent;
    let output = body.output;
    let value = body.value;
    quote! {
        #[doc(hidden)]
        #[allow(missing_docs)]
        #visibility fn #generated_name() -> ::cutile::ir::Kernel {
            ::cutile::ir::Kernel::new(
                #fn_name.to_string(),
                vec![#(#ir_params),*],
                #index,
                #extent,
                #output,
                #value,
            )
        }
    }
}

fn ir_value_type(kind: ParamKind) -> TokenStream2 {
    match kind {
        ParamKind::BufferRead => quote! {
            ::cutile::ir::ValueType::Buffer {
                element: ::cutile::ir::ScalarType::F32,
                access: ::cutile::ir::Access::Read,
            }
        },
        ParamKind::BufferWrite => quote! {
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
        Type::Path(path) if path.qself.is_none() => {
            match path.path.get_ident().map(ToString::to_string).as_deref() {
                Some("f32") => Ok(ParamKind::F32),
                Some("usize") => Ok(ParamKind::Usize),
                _ => Err(Error::new(
                    ty.span(),
                    "scalar kernel parameters must be `f32` or `usize`",
                )),
            }
        }
        _ => Err(Error::new(
            ty.span(),
            "kernel parameters must be `f32`, `usize`, `&[f32]`, or `&mut [f32]`",
        )),
    }
}

fn reference_kind(reference: &TypeReference) -> syn::Result<ParamKind> {
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return Err(Error::new(
            reference.span(),
            "buffer parameters must be slices (`&[f32]` or `&mut [f32]`)",
        ));
    };
    let Type::Path(element) = slice.elem.as_ref() else {
        return Err(Error::new(
            slice.elem.span(),
            "buffer element type must be `f32`",
        ));
    };
    if element.qself.is_some()
        || element.path.get_ident().map(ToString::to_string).as_deref() != Some("f32")
    {
        return Err(Error::new(
            slice.elem.span(),
            "buffer element type must be `f32`",
        ));
    }
    Ok(if reference.mutability.is_some() {
        ParamKind::BufferWrite
    } else {
        ParamKind::BufferRead
    })
}

fn unique_parameters(params: &[Param], signature: &syn::Signature) -> syn::Result<()> {
    for (index, parameter) in params.iter().enumerate() {
        if params[..index]
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

fn parse_loop(loop_stmt: &ExprForLoop, params: &[Param]) -> syn::Result<(String, String)> {
    if loop_stmt.label.is_some() || !loop_stmt.attrs.is_empty() {
        return Err(Error::new(
            loop_stmt.span(),
            "labeled loops and loop attributes are not supported",
        ));
    }
    let Pat::Ident(index) = loop_stmt.pat.as_ref() else {
        return Err(Error::new(
            loop_stmt.pat.span(),
            "loop index must be a simple identifier",
        ));
    };
    if index.by_ref.is_some() || index.mutability.is_some() || index.subpat.is_some() {
        return Err(Error::new(
            index.span(),
            "loop index must be an unmodified identifier",
        ));
    }
    let Expr::Range(range) = loop_stmt.expr.as_ref() else {
        return Err(Error::new(
            loop_stmt.expr.span(),
            "loop range must be `0..extent`",
        ));
    };
    if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) {
        return Err(Error::new(
            range.limits.span(),
            "loop range must be half-open (`..`)",
        ));
    }
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
    let Some(end) = range.end.as_deref() else {
        return Err(Error::new(range.span(), "loop range requires an extent"));
    };
    let Expr::Path(ExprPath { path, .. }) = end else {
        return Err(Error::new(
            end.span(),
            "loop extent must be a `usize` parameter",
        ));
    };
    let extent = path
        .get_ident()
        .ok_or_else(|| Error::new(path.span(), "loop extent must be a `usize` parameter"))?;
    let extent = portable_ident(extent)?;
    if !params
        .iter()
        .any(|param| param.name == extent && param.kind == ParamKind::Usize)
    {
        return Err(Error::new(
            end.span(),
            "loop extent must name a `usize` parameter",
        ));
    }
    let index = portable_ident(&index.ident)?;
    if params.iter().any(|param| param.name == index) {
        return Err(Error::new(
            loop_stmt.pat.span(),
            "loop index must not shadow a kernel parameter",
        ));
    }
    Ok((index, extent))
}

fn parse_assignment(
    assign: &ExprAssign,
    index: &str,
    params: &[Param],
) -> syn::Result<(String, TokenStream2)> {
    let Expr::Index(ExprIndex {
        expr: buffer,
        index: at,
        ..
    }) = assign.left.as_ref()
    else {
        return Err(Error::new(
            assign.left.span(),
            "kernel assignment target must be `output[index]`",
        ));
    };
    let Expr::Path(buffer) = buffer.as_ref() else {
        return Err(Error::new(
            buffer.span(),
            "output must be a named mutable buffer",
        ));
    };
    let output = buffer
        .path
        .get_ident()
        .ok_or_else(|| Error::new(buffer.path.span(), "output must be a named mutable buffer"))?;
    let output = portable_ident(output)?;
    if !params
        .iter()
        .any(|param| param.name == output && param.kind == ParamKind::BufferWrite)
    {
        return Err(Error::new(
            assign.left.span(),
            "assignment target must be an `&mut [f32]` parameter",
        ));
    }
    let Expr::Path(at) = at.as_ref() else {
        return Err(Error::new(at.span(), "buffer index must be the loop index"));
    };
    if at
        .path
        .get_ident()
        .map(portable_ident)
        .transpose()?
        .as_deref()
        != Some(index)
    {
        return Err(Error::new(at.span(), "buffer index must be the loop index"));
    }
    Ok((output, parse_expr(assign.right.as_ref(), index, params)?))
}

fn parse_expr(expression: &Expr, index: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    match expression {
        Expr::Paren(paren) => parse_expr(paren.expr.as_ref(), index, params),
        Expr::Group(group) => parse_expr(group.expr.as_ref(), index, params),
        Expr::Lit(ExprLit {
            lit: Lit::Float(literal),
            ..
        }) => {
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
        Expr::Path(path) => {
            let name = path
                .path
                .get_ident()
                .ok_or_else(|| Error::new(path.span(), "kernel expressions use simple names"))?;
            let name = portable_ident(name)?;
            if name == index {
                return Err(Error::new(
                    path.span(),
                    "loop index is only valid as a buffer index",
                ));
            }
            match params.iter().find(|param| param.name == name) {
                Some(Param {
                    kind: ParamKind::F32,
                    ..
                }) => Ok(quote!(::cutile::ir::Expr::Scalar(#name.to_string()))),
                Some(Param {
                    kind: ParamKind::BufferRead | ParamKind::BufferWrite,
                    ..
                }) => Err(Error::new(
                    path.span(),
                    "buffer values must be read with `buffer[loop_index]`",
                )),
                Some(Param {
                    kind: ParamKind::Usize,
                    ..
                }) => Err(Error::new(
                    path.span(),
                    "`usize` is not valid in an f32 expression",
                )),
                None => Err(Error::new(
                    path.span(),
                    format!("unknown kernel value `{name}`"),
                )),
            }
        }
        Expr::Index(ExprIndex {
            expr: buffer,
            index: at,
            ..
        }) => {
            let Expr::Path(buffer) = buffer.as_ref() else {
                return Err(Error::new(
                    buffer.span(),
                    "only named buffers can be loaded",
                ));
            };
            let name = buffer.path.get_ident().ok_or_else(|| {
                Error::new(buffer.path.span(), "only named buffers can be loaded")
            })?;
            let name = portable_ident(name)?;
            let Expr::Path(at) = at.as_ref() else {
                return Err(Error::new(at.span(), "buffer index must be the loop index"));
            };
            if at
                .path
                .get_ident()
                .map(portable_ident)
                .transpose()?
                .as_deref()
                != Some(index)
            {
                return Err(Error::new(at.span(), "buffer index must be the loop index"));
            }
            if !params.iter().any(|param| {
                param.name == name
                    && matches!(param.kind, ParamKind::BufferRead | ParamKind::BufferWrite)
            }) {
                return Err(Error::new(
                    buffer.span(),
                    format!("`{name}` is not a buffer parameter"),
                ));
            }
            Ok(quote!(::cutile::ir::Expr::Load(#name.to_string())))
        }
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let left = parse_expr(left.as_ref(), index, params)?;
            let right = parse_expr(right.as_ref(), index, params)?;
            let operator = match op {
                BinOp::Add(_) => quote!(::cutile::ir::BinaryOp::Add),
                BinOp::Sub(_) => quote!(::cutile::ir::BinaryOp::Sub),
                BinOp::Mul(_) => quote!(::cutile::ir::BinaryOp::Mul),
                BinOp::Div(_) => quote!(::cutile::ir::BinaryOp::Div),
                _ => {
                    return Err(Error::new(
                        op.span(),
                        "supported arithmetic is `+`, `-`, `*`, and `/`",
                    ));
                }
            };
            Ok(quote! {
                ::cutile::ir::Expr::Binary {
                    op: #operator,
                    left: Box::new(#left),
                    right: Box::new(#right),
                }
            })
        }
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
            let value = parse_expr(unary.expr.as_ref(), index, params)?;
            Ok(quote! {
                ::cutile::ir::Expr::Neg(Box::new(#value))
            })
        }
        _ => Err(Error::new(
            expression.span(),
            "unsupported expression; use f32 parameters, buffer loads, finite float literals, and basic arithmetic",
        )),
    }
}
