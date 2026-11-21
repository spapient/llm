use crate::ParamRole;

/// 闭集四个角色都是参数槽：都进入可训练参数表。
#[test]
fn every_role_is_a_parameter_slot() {
    for role in [
        ParamRole::Trainable,
        ParamRole::Bias,
        ParamRole::Norm,
        ParamRole::Embedding,
    ] {
        assert!(role.trainable(), "{role:?} 应当是可训练参数槽");
    }
}

/// 权重衰减豁免只从角色派生：偏置、归一化系数与嵌入表不衰减，投影权重衰减。
#[test]
fn weight_decay_exemption_follows_the_role() {
    assert!(!ParamRole::Trainable.wd_exempt(), "投影权重参与衰减");
    for exempt in [ParamRole::Bias, ParamRole::Norm, ParamRole::Embedding] {
        assert!(exempt.wd_exempt(), "{exempt:?} 应豁免权重衰减");
    }
}

/// 短名与变体一一对应：诊断、快照与报告按这个名字对齐。
#[test]
fn short_names_match_the_variants() {
    assert_eq!(ParamRole::Trainable.as_str(), "trainable");
    assert_eq!(ParamRole::Bias.as_str(), "bias");
    assert_eq!(ParamRole::Norm.as_str(), "norm");
    assert_eq!(ParamRole::Embedding.as_str(), "embedding");
}
