//! `Expr` 子节点遍历与按元数重建：归一化、引用扫描、替换、发射共用这一处分派。

use crate::kernel::Expr;

impl Expr {
    /// 按发射顺序排列的直接子节点；叶子返回空表。只读遍历（计数、引用扫描）走这里，
    /// 与 [`map_children`] 的重建分派各管一面；新增复合变体只补这一处元数据。
    pub(crate) fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Neg(value) | Expr::Abs(value) => vec![value.as_ref()],
            Expr::Binary { left, right, .. }
            | Expr::Min { left, right }
            | Expr::Max { left, right } => vec![left.as_ref(), right.as_ref()],
            Expr::Fma { a, b, c } => vec![a.as_ref(), b.as_ref(), c.as_ref()],
            Expr::F32(_) | Expr::Scalar(_) | Expr::Load(_) => Vec::new(),
        }
    }

    /// 对子节点逐个应用 `map`，再按原变体元数重建；叶子原样返回。泛型闭包编译期单态展开，无虚调。
    pub(crate) fn map_children<F>(&self, map: F) -> Expr
    where
        F: FnMut(&Expr) -> Expr,
    {
        let mut rebuilt: Vec<Expr> = self.children().into_iter().map(map).collect();
        match rebuilt.len() {
            1 => self.rebuild_unary(rebuilt.remove(0)),
            2 => self.rebuild_binary(rebuilt.remove(0), rebuilt.remove(0)),
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
        match self {
            Expr::Neg(_) => Expr::Neg(Box::new(child)),
            Expr::Abs(_) => Expr::Abs(Box::new(child)),
            _ => self.clone(),
        }
    }

    fn rebuild_binary(&self, left: Expr, right: Expr) -> Expr {
        match self {
            Expr::Binary { op, .. } => Expr::Binary {
                op: *op,
                left: Box::new(left),
                right: Box::new(right),
            },
            Expr::Min { .. } => Expr::Min {
                left: Box::new(left),
                right: Box::new(right),
            },
            Expr::Max { .. } => Expr::Max {
                left: Box::new(left),
                right: Box::new(right),
            },
            _ => self.clone(),
        }
    }

    fn rebuild_ternary(&self, a: Expr, b: Expr, c: Expr) -> Expr {
        match self {
            Expr::Fma { .. } => Expr::Fma {
                a: Box::new(a),
                b: Box::new(b),
                c: Box::new(c),
            },
            _ => self.clone(),
        }
    }
}
