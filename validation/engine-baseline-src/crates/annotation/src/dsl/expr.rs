//! 表达式族解析：括号剥壳、叶子（字面量 / 标量 / 加载）、组合子与方法闭集。

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    BinOp, Error, Expr, ExprBinary, ExprLit, ExprMethodCall, ExprPath, ExprUnary, Ident, Lit, UnOp,
    spanned::Spanned,
};

use super::{Param, ParamKind, at_loop_idx, fail, param_kind, path_ident, portable_ident};

/// 表达式这一族的统一拒绝语：把「能用什么」写在一处，免得三个分支各说一套。
const UNSUPPORTED: &str = "unsupported expression: f32 params, buf loads, finite literals, + - * /, \
     abs/sqrt/exp/ln/sin/cos/ceil/floor/round/trunc, min, max";
const ARITHMETIC: &str = "supported arithmetic is `+`, `-`, `*`, and `/`";

/// 方法组合子的 IR 形状：一元落 `Expr::Unary`，二元落 `Min`/`Max`。
enum MethodKind {
    /// 一元算子，括号内是 [`cutile::kernel::UnaryFn`] 变体名。
    Unary(&'static str),
    /// 二元逐点组合子，括号内是 `Expr` 变体名（`Min` / `Max`）。
    Binary(&'static str),
}

/// 方法组合子闭集：方法名 + 元数 + IR 形状。
struct Method {
    name: &'static str,
    arity: usize,
    kind: MethodKind,
}

const METHODS: [Method; 12] = [
    Method {
        name: "abs",
        arity: 0,
        kind: MethodKind::Unary("Abs"),
    },
    Method {
        name: "sqrt",
        arity: 0,
        kind: MethodKind::Unary("Sqrt"),
    },
    Method {
        name: "exp",
        arity: 0,
        kind: MethodKind::Unary("Exp"),
    },
    Method {
        name: "ln",
        arity: 0,
        kind: MethodKind::Unary("Ln"),
    },
    Method {
        name: "sin",
        arity: 0,
        kind: MethodKind::Unary("Sin"),
    },
    Method {
        name: "cos",
        arity: 0,
        kind: MethodKind::Unary("Cos"),
    },
    Method {
        name: "ceil",
        arity: 0,
        kind: MethodKind::Unary("Ceil"),
    },
    Method {
        name: "floor",
        arity: 0,
        kind: MethodKind::Unary("Floor"),
    },
    Method {
        name: "round",
        arity: 0,
        kind: MethodKind::Unary("Round"),
    },
    Method {
        name: "trunc",
        arity: 0,
        kind: MethodKind::Unary("Trunc"),
    },
    Method {
        name: "min",
        arity: 1,
        kind: MethodKind::Binary("Min"),
    },
    Method {
        name: "max",
        arity: 1,
        kind: MethodKind::Binary("Max"),
    },
];

/// 表达式三层：括号剥壳 → 叶子（字面量/标量/加载）→ 组合子（二元、取负）。
pub(super) fn parse_expr(
    expression: &Expr,
    idx: &str,
    params: &[Param],
) -> syn::Result<TokenStream2> {
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
        _ => fail(expression.span(), UNSUPPORTED),
    }
}

/// 按 [`METHODS`] 闭集展开方法组合子；未登记的方法名在展开期打回。变体构造另走 [`build_method`]。
fn method_call(method: &ExprMethodCall, idx: &str, params: &[Param]) -> syn::Result<TokenStream2> {
    let name = method.method.to_string();
    let Some(known) = METHODS
        .iter()
        .find(|candidate| candidate.name == name.as_str())
    else {
        return fail(method.method.span(), "unsupported kernel method");
    };
    if method.args.len() != known.arity {
        return fail(
            method.method.span(),
            &format!("`{name}` takes {} argument(s)", known.arity),
        );
    }
    let receiver = parse_expr(&method.receiver, idx, params)?;
    build_method(known, method, receiver, idx, params)
}

/// 方法变体的 IR 构造：一元落 `Unary`，`min`/`max` 落二元；与查找 / 校验分流，压低单函数复杂度。
fn build_method(
    known: &Method,
    method: &ExprMethodCall,
    receiver: TokenStream2,
    idx: &str,
    params: &[Param],
) -> syn::Result<TokenStream2> {
    match known.kind {
        MethodKind::Unary(op) => {
            let op = Ident::new(op, method.method.span());
            Ok(quote! {
                ::cutile::kernel::Expr::Unary {
                    op: ::cutile::kernel::UnaryFn::#op,
                    child: Box::new(#receiver),
                }
            })
        }
        MethodKind::Binary(variant) => {
            let argument = parse_expr(&method.args[0], idx, params)?;
            let variant = Ident::new(variant, method.method.span());
            Ok(quote! {
                ::cutile::kernel::Expr::#variant {
                    left: Box::new(#receiver),
                    right: Box::new(#argument),
                }
            })
        }
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
