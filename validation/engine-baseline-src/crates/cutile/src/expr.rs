//! `Expr` 子节点遍历与重建：规约、引用扫描、替换和双目标发射共用这一份结构分派。

use crate::kernel::Expr;

impl Expr {
    /// 按求值顺序排列的直接子节点；叶子返回空表。
    pub(crate) fn children(&self) -> Vec<&Expr> {
        match self {
            Self::F32(_) | Self::Int(_) | Self::Scalar(_) | Self::Load(_) => Vec::new(),
            Self::Call { args, .. } => args.iter().collect(),
            other => compound_children(other),
        }
    }

    /// 对直接子项应用 `map` 并按原变体身份重建；闭包编译期单态化，无虚调。
    pub(crate) fn map_children<F>(&self, map: F) -> Expr
    where
        F: FnMut(&Expr) -> Expr,
    {
        if let Self::Call { function, args } = self {
            return Self::Call {
                function: function.clone(),
                args: args.iter().map(map).collect(),
            };
        }
        let children: Vec<Expr> = self.children().into_iter().map(map).collect();
        self.rebuild_children(children)
    }

    fn rebuild_children(&self, mut rebuilt: Vec<Expr>) -> Expr {
        match rebuilt.len() {
            0 => self.clone(),
            1 => self.rebuild_unary(rebuilt.remove(0)),
            2 => {
                let left = rebuilt.remove(0);
                let right = rebuilt.remove(0);
                self.rebuild_binary(left, right)
            }
            3 => {
                let a = rebuilt.remove(0);
                let b = rebuilt.remove(0);
                let c = rebuilt.remove(0);
                self.rebuild_ternary(a, b, c)
            }
            _ => self.clone(),
        }
    }

    fn rebuild_unary(&self, child: Expr) -> Expr {
        let child = Box::new(child);
        match self {
            Self::Neg(_) | Self::Not(_) | Self::BitNot(_) | Self::Address(_) | Self::Deref(_) => {
                rebuild_prefix(self, child)
            }
            Self::Unary { op, .. } => Self::Unary { op: *op, child },
            Self::Cast { ty, .. } => Self::Cast {
                ty: *ty,
                value: child,
            },
            Self::Update { delta, prefix, .. } => Self::Update {
                target: child,
                delta: *delta,
                prefix: *prefix,
            },
            _ => self.clone(),
        }
    }

    fn rebuild_binary(&self, left: Expr, right: Expr) -> Expr {
        let (left, right) = (Box::new(left), Box::new(right));
        match self {
            Self::Binary { op, .. } => Self::Binary {
                op: *op,
                left,
                right,
            },
            Self::Min { .. } | Self::Max { .. } => rebuild_extremum(self, left, right),
            Self::Compare { .. } | Self::Logical { .. } => rebuild_predicate(self, left, right),
            Self::Index { .. } | Self::Assign { .. } => rebuild_memory(self, left, right),
            _ => self.clone(),
        }
    }

    fn rebuild_ternary(&self, a: Expr, b: Expr, c: Expr) -> Expr {
        match self {
            Self::Fma { .. } => Self::Fma {
                a: Box::new(a),
                b: Box::new(b),
                c: Box::new(c),
            },
            Self::Select { .. } => Self::Select {
                condition: Box::new(a),
                when_true: Box::new(b),
                when_false: Box::new(c),
            },
            _ => self.clone(),
        }
    }
}

fn compound_children(expression: &Expr) -> Vec<&Expr> {
    match expression {
        Expr::Neg(value)
        | Expr::Not(value)
        | Expr::BitNot(value)
        | Expr::Address(value)
        | Expr::Deref(value)
        | Expr::Unary { child: value, .. }
        | Expr::Cast { value, .. }
        | Expr::Update { target: value, .. } => vec![value],
        Expr::Binary { left, right, .. }
        | Expr::Min { left, right }
        | Expr::Max { left, right }
        | Expr::Compare { left, right, .. }
        | Expr::Logical { left, right, .. }
        | Expr::Index {
            base: left,
            idx: right,
        }
        | Expr::Assign {
            target: left,
            value: right,
            ..
        } => vec![left, right],
        Expr::Fma { a, b, c } => vec![a, b, c],
        Expr::Select {
            condition,
            when_true,
            when_false,
        } => vec![condition, when_true, when_false],
        _ => Vec::new(),
    }
}

fn rebuild_prefix(expression: &Expr, child: Box<Expr>) -> Expr {
    match expression {
        Expr::Neg(_) | Expr::Not(_) | Expr::BitNot(_) => rebuild_unary_prefix(expression, child),
        Expr::Address(_) | Expr::Deref(_) => rebuild_address(expression, child),
        _ => expression.clone(),
    }
}

fn rebuild_unary_prefix(expression: &Expr, child: Box<Expr>) -> Expr {
    match expression {
        Expr::Neg(_) => Expr::Neg(child),
        Expr::Not(_) => Expr::Not(child),
        Expr::BitNot(_) => Expr::BitNot(child),
        _ => expression.clone(),
    }
}

fn rebuild_address(expression: &Expr, child: Box<Expr>) -> Expr {
    match expression {
        Expr::Address(_) => Expr::Address(child),
        Expr::Deref(_) => Expr::Deref(child),
        _ => expression.clone(),
    }
}

fn rebuild_extremum(expression: &Expr, left: Box<Expr>, right: Box<Expr>) -> Expr {
    match expression {
        Expr::Min { .. } => Expr::Min { left, right },
        Expr::Max { .. } => Expr::Max { left, right },
        _ => expression.clone(),
    }
}

fn rebuild_predicate(expression: &Expr, left: Box<Expr>, right: Box<Expr>) -> Expr {
    match expression {
        Expr::Compare { op, .. } => Expr::Compare {
            op: *op,
            left,
            right,
        },
        Expr::Logical { op, .. } => Expr::Logical {
            op: *op,
            left,
            right,
        },
        _ => expression.clone(),
    }
}

fn rebuild_memory(expression: &Expr, left: Box<Expr>, right: Box<Expr>) -> Expr {
    match expression {
        Expr::Index { .. } => Expr::Index {
            base: left,
            idx: right,
        },
        Expr::Assign { op, .. } => Expr::Assign {
            target: left,
            op: *op,
            value: right,
        },
        _ => expression.clone(),
    }
}

// ── 编译期整数常量求值 ───────────────────────────────────────────────────

use crate::kernel::{Arith, LogicalKind};

/// 求值条件编译使用的整数常量；非有限、非整数或含变量的表达式返回 `None`。
pub(crate) fn const_int(expression: &Expr) -> Option<i128> {
    match expression {
        Expr::Int(value) => Some(i128::from(*value)),
        Expr::F32(bits) => {
            let value = f32::from_bits(*bits);
            (value.is_finite() && value.fract() == 0.0).then_some(value as i128)
        }
        _ => const_compound(expression),
    }
}

fn const_compound(expression: &Expr) -> Option<i128> {
    match expression {
        Expr::Cast { value, .. } | Expr::Neg(value) | Expr::Not(value) => {
            const_prefix(expression, value)
        }
        Expr::Binary { left, right, .. } | Expr::Compare { left, right, .. } => {
            const_binary(expression, left, right)
        }
        Expr::Logical { op, left, right } => const_logical(*op, left, right),
        Expr::Select {
            condition,
            when_true,
            when_false,
        } => const_select(condition, when_true, when_false),
        _ => None,
    }
}

fn const_prefix(expression: &Expr, value: &Expr) -> Option<i128> {
    match expression {
        Expr::Cast { .. } => const_int(value),
        Expr::Neg(_) => const_int(value).and_then(|value| value.checked_neg()),
        Expr::Not(_) => const_int(value).map(|value| i128::from(value == 0)),
        _ => None,
    }
}

fn const_logical(op: LogicalKind, left: &Expr, right: &Expr) -> Option<i128> {
    const RULES: [fn(bool, bool) -> bool; 2] =
        [|left, right| left && right, |left, right| left || right];
    let left = const_int(left)? != 0;
    let right = const_int(right)? != 0;
    Some(i128::from(RULES[op as usize](left, right)))
}

fn const_select(condition: &Expr, when_true: &Expr, when_false: &Expr) -> Option<i128> {
    if const_int(condition)? != 0 {
        const_int(when_true)
    } else {
        const_int(when_false)
    }
}

fn const_binary(expression: &Expr, left: &Expr, right: &Expr) -> Option<i128> {
    let (left, right) = (const_int(left)?, const_int(right)?);
    match expression {
        Expr::Binary { op, .. } => const_arith(*op, left, right),
        Expr::Compare { op, .. } => {
            const COMPARE: [fn(i128, i128) -> bool; 6] = [
                |left, right| left == right,
                |left, right| left != right,
                |left, right| left < right,
                |left, right| left <= right,
                |left, right| left > right,
                |left, right| left >= right,
            ];
            Some(i128::from(COMPARE[*op as usize](left, right)))
        }
        _ => None,
    }
}

fn const_arith(op: Arith, left: i128, right: i128) -> Option<i128> {
    const RULES: [fn(i128, i128) -> Option<i128>; 10] = [
        |left, right| left.checked_add(right),
        |left, right| left.checked_sub(right),
        |left, right| left.checked_mul(right),
        |left, right| left.checked_div(right),
        |left, right| left.checked_rem(right),
        |left, right| left.checked_shl(u32::try_from(right).ok()?),
        |left, right| left.checked_shr(u32::try_from(right).ok()?),
        |left, right| Some(left & right),
        |left, right| Some(left | right),
        |left, right| Some(left ^ right),
    ];
    RULES[op as usize](left, right)
}

use crate::substitute::error;

pub(crate) fn parse_number(text: &str, line: usize) -> diagnostic::Result<Expr> {
    let float = text.contains(['.', 'e', 'E']) || text.ends_with(['f', 'F']);
    if float {
        let literal = text.trim_end_matches(['f', 'F']);
        let value = literal
            .parse::<f32>()
            .map_err(|_| error(line, format!("非法浮点字面量 `{text}`")))?;
        return Ok(Expr::F32(value.to_bits()));
    }
    let literal = text.trim_end_matches(['u', 'U', 'l', 'L']);
    let value = if let Some(hex) = literal.strip_prefix("0x") {
        i64::from_str_radix(hex, 16)
    } else {
        literal.parse::<i64>()
    };
    value
        .map(Expr::Int)
        .map_err(|_| error(line, format!("非法整数字面量 `{text}`")))
}

type PrefixRule = fn(Box<Expr>) -> Expr;

pub(crate) fn prefix_node(operator: &str, child: Box<Expr>) -> Expr {
    const RULES: &[(&str, PrefixRule)] = &[
        ("+", |child| *child),
        ("-", |child| Expr::Neg(child)),
        ("!", |child| Expr::Not(child)),
        ("~", |child| Expr::BitNot(child)),
        ("&", |child| Expr::Address(child)),
        ("*", |child| Expr::Deref(child)),
        ("++", |child| Expr::Update {
            target: child,
            delta: 1,
            prefix: true,
        }),
        ("--", |child| Expr::Update {
            target: child,
            delta: -1,
            prefix: true,
        }),
    ];
    if let Some((_, build)) = RULES.iter().find(|(name, _)| *name == operator) {
        return build(child);
    }
    *child
}

pub(crate) fn numeric_start(text: &str) -> bool {
    text.bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_digit())
        || text.starts_with('.')
}

pub(crate) fn identifier_start(text: &str) -> bool {
    let mut bytes = text.bytes();
    let start = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    start && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}
