//! Rust function front end for the initial typed pointwise kernel IR.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    BinOp, Error, Expr, ExprAssign, ExprBinary, ExprForLoop, ExprIndex, ExprLit, ExprPath, ItemFn,
    Lit, Pat, Stmt, Type, TypeReference, UnOp, spanned::Spanned,
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
    let signature = &function.sig;
    if signature.asyncness.is_some()
        || signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || !signature.generics.params.is_empty()
        || signature.variadic.is_some()
        || !matches!(signature.output, syn::ReturnType::Default)
    {
        return Err(Error::new(
            signature.span(),
            "`#[dsl]` currently supports non-generic, safe, synchronous functions returning `()`",
        ));
    }

    let mut params = Vec::with_capacity(signature.inputs.len());
    for argument in &signature.inputs {
        let syn::FnArg::Typed(argument) = argument else {
            return Err(Error::new(
                argument.span(),
                "`self` parameters are not supported",
            ));
        };
        let Pat::Ident(pattern) = argument.pat.as_ref() else {
            return Err(Error::new(
                argument.pat.span(),
                "kernel parameters must use simple identifier patterns",
            ));
        };
        let kind = parameter_kind(&argument.ty)?;
        params.push(Param {
            name: pattern.ident.to_string(),
            kind,
        });
    }
    unique_parameters(&params, signature)?;

    let [Stmt::Expr(Expr::ForLoop(loop_stmt), _)] = function.block.stmts.as_slice() else {
        return Err(Error::new(
            function.block.span(),
            "kernel body must contain exactly one `for index in 0..extent` loop",
        ));
    };
    let (index, extent) = parse_loop(loop_stmt, &params)?;
    let [Stmt::Expr(Expr::Assign(assign), _)] = loop_stmt.body.stmts.as_slice() else {
        return Err(Error::new(
            loop_stmt.body.span(),
            "kernel loop must contain exactly one indexed assignment",
        ));
    };
    let (output, value) = parse_assignment(assign, &index, &params)?;

    let ir_params = params.iter().map(|parameter| {
        let name = &parameter.name;
        let ty = match parameter.kind {
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
        };
        quote! {
            ::cutile::ir::Parameter {
                name: #name.to_string(),
                ty: #ty,
            }
        }
    });
    let fn_name = signature.ident.to_string();
    let generated_name = format_ident!("__dsl_ir_{}", signature.ident);
    let visibility = &function.vis;

    Ok(quote! {
        #[doc(hidden)]
        #[allow(missing_docs)]
        #visibility fn #generated_name() -> ::cutile::ir::Kernel {
            ::cutile::ir::Kernel::new(
                #fn_name,
                vec![#(#ir_params),*],
                #index,
                #extent,
                #output,
                #value,
            )
        }
    })
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
    let Pat::Ident(index) = loop_stmt.pat.as_ref() else {
        return Err(Error::new(
            loop_stmt.pat.span(),
            "loop index must be a simple identifier",
        ));
    };
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
        .ok_or_else(|| Error::new(path.span(), "loop extent must be a `usize` parameter"))?
        .to_string();
    if !params
        .iter()
        .any(|param| param.name == extent && param.kind == ParamKind::Usize)
    {
        return Err(Error::new(
            end.span(),
            "loop extent must name a `usize` parameter",
        ));
    }
    Ok((index.ident.to_string(), extent))
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
        .ok_or_else(|| Error::new(buffer.path.span(), "output must be a named mutable buffer"))?
        .to_string();
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
    if at.path.get_ident().map(ToString::to_string).as_deref() != Some(index) {
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
                .ok_or_else(|| Error::new(path.span(), "kernel expressions use simple names"))?
                .to_string();
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
            let name = buffer
                .path
                .get_ident()
                .ok_or_else(|| Error::new(buffer.path.span(), "only named buffers can be loaded"))?
                .to_string();
            let Expr::Path(at) = at.as_ref() else {
                return Err(Error::new(at.span(), "buffer index must be the loop index"));
            };
            if at.path.get_ident().map(ToString::to_string).as_deref() != Some(index) {
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
