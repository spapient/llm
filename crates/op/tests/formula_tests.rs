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
