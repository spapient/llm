use super::*;
use crate::builtin::Catalogue;

#[test]
fn shape_validation_matches_the_operator_constraints() {
    let group = u64::from(model_for("gemv_i4").expect("有模型").group);
    assert_eq!(parse_shape("4096x4096", group).unwrap(), (4096, 4096));
    assert!(parse_shape("4096", group).is_err());
    assert!(parse_shape("1000x8", group).is_err());
    assert!(parse_shape("0x8", group).is_err());
}

#[test]
fn the_shape_matrix_follows_the_operator_model() {
    let model = model_for("gemv_i4").expect("有模型");
    assert_eq!(verify_shapes(&model), vec![(32, 1), (64, 3), (4096, 4096)]);
    // 换一个生产形状，矩阵跟着变：审计里没有第二个方阵。
    let other = op::model::Model {
        shape: (128, 64),
        ..model
    };
    assert_eq!(verify_shapes(&other), vec![(32, 1), (64, 3), (128, 64)]);
}

#[test]
fn the_model_group_matches_the_declaration() {
    // 声明里的 `needs.quant.group` 是能力请求，模型里的 `group` 是样本与代价的取值；
    // 两者漂移会让 verify 在一组原语支撑不了的组大小上"通过"。
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let declared = catalogue
        .op("gemv_i4")
        .expect("内置算子")
        .needs
        .iter()
        .filter(|need| need.prim == prof::atom::Prim::Quant)
        .flat_map(|need| need.attrs.iter())
        .find(|attr| attr.key() == prof::atom::AttrKey::Group)
        .and_then(|attr| attr.values().first())
        .expect("声明里有 quant.group")
        .parse::<u32>()
        .expect("组大小是整数");
    assert_eq!(declared, model_for("gemv_i4").expect("有模型").group);
}
