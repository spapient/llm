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
        Expr::Scalar(name) => source.push_str(name),
        Expr::Load(name) => {
            let _ = write!(source, "{name}[{idx}]");
        }
        other => write_compound(source, other, idx),
    }
}

fn write_compound(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::Neg(value) => {
            source.push_str("-(");
            write_expr(source, value, idx);
            source.push(')');
        }
        Expr::Binary { op, left, right } => write_binary(source, *op, left, right, idx),
        Expr::Unary { .. } | Expr::Fma { .. } | Expr::Min { .. } | Expr::Max { .. } => {
            write_call(source, call_name(expression), &expression.children(), idx)
        }
        // 叶子由 [`write_expr`] 前置分流到本函数之外；落到这里即空体，不再发射。
        Expr::F32(_) | Expr::Scalar(_) | Expr::Load(_) => {}
    }
}

/// 复合变体对应的 C 库函数名；非调用形态返回空串，不 panic。
fn call_name(expression: &Expr) -> &'static str {
    match expression {
        Expr::Unary { op, .. } => op.c_name(),
        Expr::Fma { .. } => "fmaf",
        Expr::Min { .. } => "fminf",
        Expr::Max { .. } => "fmaxf",
        _ => "",
    }
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
