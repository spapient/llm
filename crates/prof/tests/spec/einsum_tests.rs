// einsum 三层的测试：表达式解析、轴长度表与代价推导。
//
// `einsum` 已按职责轴拆为 `src/einsum/` 目录模块，本文件与 `cost_tests.rs` 由
// `src/einsum/mod.rs` 的测试模块按顺序
// `include!` 进来，两段共用同一个测试模块；`use super::*` 因此解析到 `einsum` 模块本身，
// 轴表与代价段直接用 `crate::einsum` 的再导出取类型。Bits / Cost 自身的用例在同目录的
// `cost_tests.rs`。

use super::*;

use crate::einsum::{Access, AxisRole, Bits, Extents, Formula, Operand};

// ===== 表达式解析（expr） =====

fn error(text: &str) -> String {
    Expr::parse(text).expect_err("应当被拒绝").to_string()
}

#[test]
fn an_explicit_expression_keeps_terms_and_output_order() {
    let expr = Expr::parse("nm,m,ng,n->n").expect("合法");
    assert_eq!(expr.arity(), 4);
    assert_eq!(expr.output(), ['n']);
    let terms: Vec<&[char]> = expr.terms().collect();
    assert_eq!(terms, [['n', 'm'].as_slice(), &['m'], &['n', 'g'], &['n']]);
    assert_eq!(expr.text(), "nm,m,ng,n->n");
    assert_eq!(expr.to_string(), "nm,m,ng,n->n");
}

#[test]
fn whitespace_is_trimmed_but_not_interpreted() {
    let expr = Expr::parse("  i,j->ij \n").expect("合法");
    assert_eq!(expr.text(), "i,j->ij");
    assert_eq!(expr.arity(), 2);
}

#[test]
fn a_scalar_output_and_scalar_terms_are_allowed() {
    let expr = Expr::parse("m,->").expect("标量输出合法");
    assert_eq!(expr.arity(), 2);
    assert!(expr.output().is_empty());
    let terms: Vec<&[char]> = expr.terms().collect();
    assert_eq!(terms, [['m'].as_slice(), &[]]);
}

#[test]
fn an_empty_expression_is_rejected() {
    assert!(error("").contains("为空"));
    assert!(error("   ").contains("为空"));
}

#[test]
fn a_missing_arrow_is_rejected_with_an_example() {
    let message = error("nm,m");
    assert!(message.contains("缺少 `->`"), "{message}");
    assert!(message.contains("nm,m->n"), "{message}");
}

#[test]
fn a_second_arrow_is_rejected() {
    assert!(error("nm,m->n->n").contains("只能出现一次"));
}

#[test]
fn a_non_letter_is_rejected_with_its_position() {
    let message = error("n1,m->n");
    assert!(message.contains("第 0 项"), "{message}");
    assert!(message.contains("`1`"), "{message}");
}

#[test]
fn a_repeated_letter_inside_one_term_is_rejected() {
    let message = error("nn,m->n");
    assert!(message.contains("重复"), "{message}");
    assert!(
        message.contains("对角线") || message.contains("对角"),
        "{message}"
    );
}

#[test]
fn a_repeated_output_letter_is_rejected() {
    assert!(error("nm->nn").contains("输出轴必须唯一"));
}

#[test]
fn an_output_axis_that_no_input_mentions_is_rejected() {
    let message = error("nm->nk");
    assert!(message.contains("输出轴 `k`"), "{message}");
}

#[test]
fn parsing_through_from_str_is_the_same_path() {
    let parsed: Expr = "ij,jk->ik".parse().expect("合法");
    assert_eq!(parsed, Expr::parse("ij,jk->ik").expect("合法"));
}

// ===== 轴长度表（extents） =====

#[test]
fn lengths_are_looked_up_by_letter() {
    let extents = Extents::new([('n', 8), ('m', 64), ('g', 2)]).expect("合法");
    assert_eq!(extents.get('n'), Some(8));
    assert_eq!(extents.get('m'), Some(64));
    assert_eq!(extents.get('z'), None);
    assert_eq!(extents.len(), 3);
    assert!(!extents.is_empty());
}

#[test]
fn the_display_is_stable_and_sorted() {
    let extents = Extents::new([('n', 8), ('g', 2), ('m', 64)]).expect("合法");
    assert_eq!(extents.to_string(), "g=2 m=64 n=8");
}

#[test]
fn an_empty_table_is_allowed_because_a_scalar_has_no_axes() {
    let extents = Extents::new([]).expect("合法");
    assert!(extents.is_empty());
    assert_eq!(extents.to_string(), "");
}

#[test]
fn a_zero_length_axis_is_rejected() {
    let error = Extents::new([('n', 0)]).expect_err("空轴应当被拒绝");
    assert!(error.to_string().contains("长度为 0"), "{error}");
}

#[test]
fn a_duplicate_axis_is_rejected_instead_of_overwritten() {
    let error = Extents::new([('n', 8), ('n', 16)]).expect_err("重复轴应当被拒绝");
    assert!(error.to_string().contains("两次"), "{error}");
}

#[test]
fn a_non_letter_axis_is_rejected() {
    let error = Extents::new([('1', 8)]).expect_err("非字母应当被拒绝");
    assert!(error.to_string().contains("不是字母"), "{error}");
}

#[test]
fn a_grouped_axis_derives_its_length_and_remembers_its_base() {
    let extents = Extents::new([('m', 64)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect("64 能被 32 整除");
    assert_eq!(extents.get('g'), Some(2));
    assert_eq!(extents.grouped('g'), Some(('m', 32)));
    assert_eq!(extents.grouped('m'), None);
    assert_eq!(extents.len(), 2);
    assert_eq!(extents.to_string(), "g=2 m=64");
}

#[test]
fn grouping_rejects_a_base_that_cannot_be_divided_evenly() {
    let error = Extents::new([('m', 33)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect_err("33 不能被 32 整除");
    assert!(error.to_string().contains("整数倍"), "{error}");
}

#[test]
fn grouping_rejects_a_missing_base_and_a_zero_divisor() {
    let missing = Extents::new([('n', 8)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect_err("没有 m");
    assert!(missing.to_string().contains("没有长度"), "{missing}");

    let zero = Extents::new([('m', 64)])
        .expect("合法")
        .group('g', 'm', 0)
        .expect_err("因子不能为 0");
    assert!(zero.to_string().contains("不能为 0"), "{zero}");
}

#[test]
fn grouping_rejects_a_taken_label_and_a_self_base() {
    let taken = Extents::new([('m', 64), ('g', 3)])
        .expect("合法")
        .group('g', 'm', 32)
        .expect_err("g 已经存在");
    assert!(taken.to_string().contains("两次"), "{taken}");

    let itself = Extents::new([('m', 64)])
        .expect("合法")
        .group('m', 'm', 32)
        .expect_err("不能以自己为基准");
    assert!(itself.to_string().contains("自己"), "{itself}");
}

// ===== 代价推导（formula） =====

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
    let reduced: Vec<char> = axes.reduced().map(super::axis::Axis::label).collect();
    let grouped: Vec<char> = axes.grouped().map(super::axis::Axis::label).collect();
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
