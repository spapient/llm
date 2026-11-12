//! 代价推导：表达式 + 轴长度 + 操作数 → [`Cost`]，以及中间的轴分解 [`Axes`]。

use crate::{Bits, Cost, Expr, Extents};
use diagnostic::{Code, Error};

/// 操作数的数据流向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// 只读：算子入口的输入。
    Read,
    /// 只写：算子出口的输出。写入同样计费，因为"至少写一遍输出"是逃不掉的流量。
    Write,
}

/// 一个操作数：名字、元素位宽与读/写方向。
///
/// 名字只是诊断用的标签（与内核模板的形参名一致），但必须唯一：同名缓冲出现在两个位置会被
/// 重复计费，而这在真实算子里从来不是本意。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    name: String,
    bits: Bits,
    access: Access,
}

impl Operand {
    /// 只读操作数。
    pub fn read(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    pub fn write(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Write,
        }
    }

    /// 按方向构造（方向来自数据时用这个）。
    pub fn new(name: impl Into<String>, bits: Bits, access: Access) -> Self {
        Self {
            name: name.into(),
            bits,
            access,
        }
    }

    /// 名字。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素位宽。
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// 流向。
    pub fn access(&self) -> Access {
        self.access
    }
}

/// 一条轴在这次计算里的角色。
///
/// 这个分类不是装饰：`Batch` 与 `Free` 决定 TileLang 式的批处理能不能合批，`Reduced` 决定
/// roofline 的 `flops` 口径。把它算出来，上层就不必再猜。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 批轴：出现在输出里，且被至少两个**只读**操作数共享。
    ///
    /// 只统计只读操作数：输出缓冲自己也带着同样的字母，把它算进去会让所有输出轴都变成批轴，
    /// 这个分类就失去了信息量。
    Batch,
    /// 自由轴：出现在输出里，且只被一个操作数拥有。
    Free,
    /// 规约轴：不出现在输出里，求和后被消掉。
    Reduced,
    /// 分组轴：由另一条轴按固定因子切出（[`crate::Extents::group`]）。
    ///
    /// 它只影响**流量**（`scales[n, g]` 确实只有 `n · m/组宽` 个元素），不计入运算次数：
    /// 遍历 `g` 与遍历 `m` 是同一批元素的两层看法，乘起来就是重复计数。
    Grouped,
}

/// 一条轴：字母、长度与角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    label: char,
    extent: usize,
    role: AxisRole,
}

impl Axis {
    /// 轴字母。
    pub fn label(&self) -> char {
        self.label
    }

    /// 轴长度（元素个数）。
    pub fn extent(&self) -> usize {
        self.extent
    }

    /// 轴角色。
    pub fn role(&self) -> AxisRole {
        self.role
    }
}

/// 表达式的轴分解：输出轴在前（按输出顺序），规约轴在后（按首次出现顺序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axes {
    axes: Vec<Axis>,
}

impl Axes {
    /// 全部轴，顺序稳定（输出轴优先）。
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// 输出轴。
    ///
    /// 只含批轴与自由轴：分组轴虽然不由求和消掉，但它是一条已被别的轴覆盖的派生看法，
    /// 算进输出形状就会把 `y[n]` 说成 `y[n, g]`。
    pub fn output(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| matches!(axis.role, AxisRole::Batch | AxisRole::Free))
    }

    /// 规约轴。
    pub fn reduced(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Reduced)
    }

    /// 分组轴（由其他轴按因子切出，只计流量）。
    pub fn grouped(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Grouped)
    }

    /// 是否有真正的规约轴（决定 `flops` 用哪种口径）。
    ///
    /// 分组轴不算：它不增加运算次数。
    pub fn reduces(&self) -> bool {
        self.reduced().next().is_some()
    }

    /// 输出形状（按输出轴的声明顺序）。
    pub fn output_shape(&self) -> Vec<usize> {
        self.output().map(Axis::extent).collect()
    }

    /// 输出元素个数；标量输出为 `1`。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn output_elements(&self) -> Result<u128, Error> {
        self.output()
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 参与运算的轴的乘积，即"展开后的标量运算次数"的分母。
    ///
    /// 分组轴被排除：它是另一条轴的分组看法，不是独立的循环维。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn all_elements(&self) -> Result<u128, Error> {
        self.axes
            .iter()
            .filter(|axis| axis.role != AxisRole::Grouped)
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }
}

/// 一份可估算的语义：表达式 + 每个操作数的位宽与方向。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    expr: Expr,
    operands: Vec<Operand>,
}

impl Formula {
    /// 把表达式与操作数配对。
    ///
    /// # Errors
    ///
    /// - 操作数个数与表达式的项数不一致；
    /// - 操作数重名（同一块缓冲会被重复计费）；
    /// - 写入操作数多于一个，或写入的轴不在输出里（写回中间量不是算子语义）。
    pub fn new(expr: Expr, operands: Vec<Operand>) -> Result<Self, Error> {
        if expr.arity() != operands.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表达式 `{expr}` 有 {} 项，但给了 {} 个操作数",
                    expr.arity(),
                    operands.len()
                ),
            )
            .with("einsum::Formula::new"));
        }

        let mut names: Vec<&str> = Vec::with_capacity(operands.len());
        let mut writes = 0usize;
        for operand in &operands {
            if names.contains(&operand.name()) {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "操作数 `{}` 出现了两次：同名缓冲会被重复计费",
                        operand.name()
                    ),
                )
                .with("einsum::Formula::new"));
            }
            names.push(operand.name());
            if operand.access() == Access::Write {
                writes += 1;
            }
        }
        if writes > 1 {
            return Err(
                Error::new(Code::Invalid, "有多个写入操作数：一次计算只有一个出口")
                    .with("einsum::Formula::new"),
            );
        }

        for (operand, term) in operands.iter().zip(expr.terms()) {
            if operand.access() != Access::Write {
                continue;
            }
            if let Some(label) = term.iter().find(|label| !expr.output().contains(label)) {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "写入操作数 `{}` 的轴 `{label}` 不在输出里：写回中间量不是算子语义",
                        operand.name()
                    ),
                )
                .with("einsum::Formula::new"));
            }
        }

        Ok(Self { expr, operands })
    }

    /// 表达式。
    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    /// 操作数（顺序与表达式的项一致）。
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// 轴分解：每条轴的长度与角色。
    ///
    /// # Errors
    ///
    /// 表达式里的字母没有给长度、给了长度的字母没出现在表达式里，或乘积超出 `u128`。
    pub fn axes(&self, extents: &Extents) -> Result<Axes, Error> {
        let mut axes: Vec<Axis> = Vec::with_capacity(extents.len());

        for &label in self.expr.output() {
            if let Some((base, divisor)) = extents.grouped(label) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("轴 `{label}` 是 `{base}` 按 {divisor} 分组的派生轴，不能出现在输出里"),
                )
                .with("einsum::Formula::axes"));
            }
            let extent = extent(extents, label)?;
            let shared = self
                .operands
                .iter()
                .zip(self.expr.terms())
                .filter(|(operand, term)| operand.access() == Access::Read && term.contains(&label))
                .count();
            let role = if shared > 1 {
                AxisRole::Batch
            } else {
                AxisRole::Free
            };
            axes.push(Axis {
                label,
                extent,
                role,
            });
        }

        for term in self.expr.terms() {
            for &label in term {
                if self.expr.output().contains(&label)
                    || axes.iter().any(|axis| axis.label == label)
                {
                    continue;
                }
                let role = if extents.grouped(label).is_some() {
                    AxisRole::Grouped
                } else {
                    AxisRole::Reduced
                };
                axes.push(Axis {
                    label,
                    extent: extent(extents, label)?,
                    role,
                });
            }
        }

        for label in extents.labels() {
            if !axes.iter().any(|axis| axis.label == label) {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "轴 `{label}` 有长度 {} 但没出现在表达式 `{}` 里：多半是笔误",
                        extent(extents, label)?,
                        self.expr
                    ),
                )
                .with("einsum::Formula::axes"));
            }
        }

        Ok(Axes { axes })
    }

    /// 计算代价。
    ///
    /// # Errors
    ///
    /// 轴分解失败（见 [`Formula::axes`]）、维度乘积溢出，或流量超出 `u64`。
    pub fn cost(&self, extents: &Extents) -> Result<Cost, Error> {
        let axes = self.axes(extents)?;
        let flops = if axes.reduces() {
            2.0 * axes.all_elements()? as f64
        } else {
            axes.output_elements()? as f64
        };

        // 两个方向各自累加：合成一个数之后，"读受限还是写受限"就再也推不出来了。
        let mut read: u128 = 0;
        let mut write: u128 = 0;
        for (operand, term) in self.operands.iter().zip(self.expr.terms()) {
            let elements = term.iter().try_fold(1u128, |total, label| {
                multiply(total, extent(extents, *label)?)
            })?;
            let bytes = operand.bits().bytes_for(elements);
            match operand.access() {
                Access::Read => read = read.checked_add(bytes).ok_or_else(overflow)?,
                Access::Write => write = write.checked_add(bytes).ok_or_else(overflow)?,
            }
        }

        let read = u64::try_from(read).map_err(|_| {
            Error::new(Code::Invalid, format!("读流量 {read} 字节超出 u64"))
                .with("einsum::Formula::cost")
        })?;
        let write = u64::try_from(write).map_err(|_| {
            Error::new(Code::Invalid, format!("写流量 {write} 字节超出 u64"))
                .with("einsum::Formula::cost")
        })?;
        Ok(Cost { flops, read, write })
    }
}

fn extent(extents: &Extents, label: char) -> Result<usize, Error> {
    extents.get(label).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("轴 `{label}` 没有长度：表达式里的每个字母都必须给出轴长度"),
        )
        .with("einsum::Formula::axes")
    })
}

fn multiply(total: u128, length: usize) -> Result<u128, Error> {
    total.checked_mul(length as u128).ok_or_else(overflow)
}

fn overflow() -> Error {
    Error::new(Code::Invalid, "维度乘积或流量超出 u128：形状不可能是真的").with("einsum::Formula")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extents(pairs: [(char, usize); 3]) -> Extents {
        Extents::new(pairs).expect("轴长度合法")
    }

    fn matmul() -> Formula {
        Formula::new(
            Expr::parse("ij,jk,ik->ik").expect("合法"),
            vec![
                Operand::read("a", Bits::ThirtyTwo),
                Operand::read("b", Bits::ThirtyTwo),
                Operand::write("c", Bits::ThirtyTwo),
            ],
        )
        .expect("三项对三个操作数")
    }

    #[test]
    fn a_matmul_costs_two_flops_per_reduced_element_and_every_operand() {
        let cost = matmul()
            .cost(&extents([('i', 2), ('j', 3), ('k', 4)]))
            .expect("可估算");
        assert_eq!(cost.flops, 2.0 * 2.0 * 3.0 * 4.0);
        // a: 2·3、b: 3·4、c: 2·4，全部 f32。
        assert_eq!(cost.read, (2 * 3 + 3 * 4) * 4);
        assert_eq!(cost.write, 2 * 4 * 4, "只有 c 是写方向");
        assert_eq!(cost.bytes(), (2 * 3 + 3 * 4 + 2 * 4) * 4, "合计不变");
    }

    #[test]
    fn the_same_formula_is_reusable_for_any_shape() {
        let formula = matmul();
        let small = formula
            .cost(&extents([('i', 1), ('j', 1), ('k', 1)]))
            .expect("可估算");
        let large = formula
            .cost(&extents([('i', 512), ('j', 512), ('k', 512)]))
            .expect("可估算");
        assert_eq!(small.flops, 2.0);
        assert_eq!(large.flops, 2.0 * 512.0 * 512.0 * 512.0);
    }

    #[test]
    fn an_elementwise_expression_counts_one_operation_per_output_element() {
        // 转置：没有规约轴，每个输出元素一次索引运算。
        let formula = Formula::new(
            Expr::parse("ij,ji->ji").expect("合法"),
            vec![
                Operand::read("x", Bits::ThirtyTwo),
                Operand::write("y", Bits::ThirtyTwo),
            ],
        )
        .expect("两项对两个操作数");
        let cost = formula
            .cost(&extents([('i', 3), ('j', 4), ('k', 1)]))
            .expect_err("多给了一个轴");
        assert!(cost.to_string().contains("`k`"), "{cost}");

        let extents = Extents::new([('i', 3), ('j', 4)]).expect("合法");
        let cost = formula.cost(&extents).expect("可估算");
        assert_eq!(cost.flops, 12.0);
        assert_eq!(cost.read, 12 * 4);
        assert_eq!(cost.write, 12 * 4);
    }

    #[test]
    fn axes_are_classified_as_batch_free_and_reduced() {
        let formula = Formula::new(
            Expr::parse("bij,bjk,bik->bik").expect("合法"),
            vec![
                Operand::read("a", Bits::ThirtyTwo),
                Operand::read("b", Bits::ThirtyTwo),
                Operand::write("c", Bits::ThirtyTwo),
            ],
        )
        .expect("配对");
        let extents = Extents::new([('b', 2), ('i', 3), ('j', 4), ('k', 5)]).expect("轴长度合法");
        let axes = formula.axes(&extents).expect("可分解");

        let labels: Vec<(char, AxisRole)> = axes
            .axes()
            .iter()
            .map(|axis| (axis.label(), axis.role()))
            .collect();
        assert_eq!(
            labels,
            [
                ('b', AxisRole::Batch),
                ('i', AxisRole::Free),
                ('k', AxisRole::Free),
                ('j', AxisRole::Reduced)
            ]
        );
        assert_eq!(axes.output_shape(), [2, 3, 5]);
        assert_eq!(axes.output_elements().expect("可算"), 2 * 3 * 5);
        assert_eq!(axes.all_elements().expect("可算"), 2 * 3 * 4 * 5);
        assert!(axes.reduces());
    }

    #[test]
    fn output_shape_follows_the_declared_output_order() {
        let formula = Formula::new(
            Expr::parse("ij,ji->ji").expect("合法"),
            vec![
                Operand::read("x", Bits::ThirtyTwo),
                Operand::write("y", Bits::ThirtyTwo),
            ],
        )
        .expect("配对");
        let axes = formula
            .axes(&Extents::new([('i', 3), ('j', 4)]).expect("合法"))
            .expect("可分解");
        assert_eq!(axes.output_shape(), [4, 3]);
    }

    #[test]
    fn packed_weights_are_charged_per_nibble() {
        // 3 行 × 5 列 int4 = 15 元素 = 7.5 字节 → 8 字节。
        let formula = Formula::new(
            Expr::parse("nm,m,n->n").expect("合法"),
            vec![
                Operand::read("w", Bits::Four),
                Operand::read("x", Bits::ThirtyTwo),
                Operand::write("y", Bits::ThirtyTwo),
            ],
        )
        .expect("配对");
        let cost = formula
            .cost(&Extents::new([('n', 3), ('m', 5)]).expect("合法"))
            .expect("可估算");
        assert_eq!(cost.flops, 2.0 * 15.0);
        assert_eq!(cost.read, 8 + 5 * 4, "int4 权重与输入都是读");
        assert_eq!(cost.write, 3 * 4);
    }

    #[test]
    fn a_grouped_axis_counts_traffic_but_not_arithmetic() {
        let formula = Formula::new(
            Expr::parse("nm,m,ng,n->n").expect("合法"),
            vec![
                Operand::read("w", Bits::Four),
                Operand::read("x", Bits::ThirtyTwo),
                Operand::read("scales", Bits::ThirtyTwo),
                Operand::write("y", Bits::ThirtyTwo),
            ],
        )
        .expect("配对");
        let extents = Extents::new([('n', 8), ('m', 64)])
            .expect("合法")
            .group('g', 'm', 32)
            .expect("64 能被 32 整除");

        let axes = formula.axes(&extents).expect("可分解");
        let reduced: Vec<char> = axes.reduced().map(|axis| axis.label()).collect();
        let grouped: Vec<char> = axes.grouped().map(|axis| axis.label()).collect();
        assert_eq!(reduced, ['m'], "只有 m 是真正的求和维");
        assert_eq!(grouped, ['g'], "g 是 m 的分组面");
        assert_eq!(axes.all_elements().expect("可算"), 8 * 64, "g 不进运算次数");

        let cost = formula.cost(&extents).expect("可估算");
        assert_eq!(cost.flops, 2.0 * 8.0 * 64.0);
        // 读流量仍然按 scales 自己的轴算：8·2 个 f32。
        assert_eq!(cost.read, 8 * 64 / 2 + 64 * 4 + 8 * 2 * 4);
        assert_eq!(cost.write, 8 * 4);
    }

    #[test]
    fn a_grouped_axis_cannot_be_an_output_axis() {
        let formula = Formula::new(
            Expr::parse("m,g->g").expect("合法"),
            vec![
                Operand::read("x", Bits::ThirtyTwo),
                Operand::write("y", Bits::ThirtyTwo),
            ],
        )
        .expect("配对");
        let extents = Extents::new([('m', 64)])
            .expect("合法")
            .group('g', 'm', 32)
            .expect("64 能被 32 整除");
        let error = formula.axes(&extents).expect_err("g 是派生轴");
        assert!(error.to_string().contains("不能出现在输出里"), "{error}");
    }

    #[test]
    fn a_scalar_expression_reduces_to_one_element() {
        let formula = Formula::new(
            Expr::parse("m,m->").expect("合法"),
            vec![
                Operand::read("a", Bits::ThirtyTwo),
                Operand::read("b", Bits::ThirtyTwo),
            ],
        )
        .expect("配对");
        let cost = formula
            .cost(&Extents::new([('m', 8)]).expect("合法"))
            .expect("可估算");
        assert_eq!(cost.flops, 2.0 * 8.0);
        assert_eq!(cost.read, 8 * 4 * 2);
        assert_eq!(cost.write, 0, "标量输出不在缓冲里");
    }

    #[test]
    fn a_missing_extent_is_reported_with_its_label() {
        let formula = matmul();
        let error = formula
            .cost(&Extents::new([('i', 2), ('j', 3)]).expect("合法"))
            .expect_err("缺 k");
        assert!(error.to_string().contains("轴 `k`"), "{error}");
    }

    #[test]
    fn an_extent_that_the_expression_never_mentions_is_rejected() {
        let formula = matmul();
        let error = formula
            .cost(&Extents::new([('i', 2), ('j', 3), ('k', 4), ('z', 5)]).expect("合法"))
            .expect_err("多余的轴");
        assert!(error.to_string().contains("`z`"), "{error}");
    }

    #[test]
    fn an_operand_count_mismatch_is_rejected() {
        let error = Formula::new(
            Expr::parse("ij,jk,ik->ik").expect("合法"),
            vec![Operand::read("a", Bits::ThirtyTwo)],
        )
        .expect_err("三项式子只给了一个操作数");
        assert!(error.to_string().contains("3 项"), "{error}");
    }

    #[test]
    fn two_writes_are_rejected() {
        let error = Formula::new(
            Expr::parse("ij,jk,ik->ik").expect("合法"),
            vec![
                Operand::write("a", Bits::ThirtyTwo),
                Operand::read("b", Bits::ThirtyTwo),
                Operand::write("c", Bits::ThirtyTwo),
            ],
        )
        .expect_err("两个出口");
        assert!(error.to_string().contains("多个写入操作数"), "{error}");
    }

    #[test]
    fn a_write_that_targets_a_reduced_axis_is_rejected() {
        let error = Formula::new(
            Expr::parse("ij,jk,ik->ik").expect("合法"),
            vec![
                Operand::read("a", Bits::ThirtyTwo),
                Operand::read("b", Bits::ThirtyTwo),
                Operand::write("c", Bits::ThirtyTwo),
            ],
        )
        .expect("正常配对");
        assert_eq!(error.operands().len(), 3);

        let error = Formula::new(
            Expr::parse("ij,jk,ik->ik").expect("合法"),
            vec![
                Operand::read("a", Bits::ThirtyTwo),
                Operand::write("b", Bits::ThirtyTwo),
                Operand::read("c", Bits::ThirtyTwo),
            ],
        )
        .expect_err("第二个操作数写的是规约轴 j");
        assert!(error.to_string().contains("轴 `j`"), "{error}");
    }

    #[test]
    fn a_repeated_operand_name_is_rejected() {
        let error = Formula::new(
            Expr::parse("ij,jk,ik->ik").expect("合法"),
            vec![
                Operand::read("a", Bits::ThirtyTwo),
                Operand::read("a", Bits::ThirtyTwo),
                Operand::write("c", Bits::ThirtyTwo),
            ],
        )
        .expect_err("同名缓冲");
        assert!(error.to_string().contains("两次"), "{error}");
    }

    #[test]
    fn the_operands_keep_their_declared_direction_and_width() {
        let operand = Operand::new("w", Bits::Four, Access::Write);
        assert_eq!(operand.name(), "w");
        assert_eq!(operand.bits(), Bits::Four);
        assert_eq!(operand.access(), Access::Write);
    }
}
