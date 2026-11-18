use super::*;

use diagnostic::Code;
use nn::ParamRole;

fn slots<'a>(
    params: &'a mut [Vec<f32>],
    grads: &'a mut [Vec<f32>],
    roles: &[ParamRole],
) -> Vec<Slot<'a>> {
    params
        .iter_mut()
        .zip(grads.iter_mut())
        .zip(roles.iter())
        .map(|((params, grads), role)| Slot {
            params,
            grads,
            role: *role,
        })
        .collect()
}

/// 角色 → 组的唯一映射：四个闭集角色逐项对拍，短名同步对齐。
#[test]
fn group_mapping_follows_the_role() {
    let cases = [
        (ParamRole::Trainable, Group::Trainable, "trainable"),
        (ParamRole::Bias, Group::Bias, "bias"),
        (ParamRole::Norm, Group::Norm, "norm"),
        (ParamRole::Embedding, Group::Embedding, "embedding"),
    ];
    for (role, group, name) in cases {
        assert_eq!(Group::from_role(role), group);
        assert_eq!(group.as_str(), name);
    }
}

/// SGD 一步：参数按 `-lr · grad` 更新，无衰减时数值可逐项断言。
#[test]
fn sgd_takes_one_direct_step() {
    let mut params = [vec![1.0_f32, 2.0], vec![0.5]];
    let mut grads = [vec![0.5_f32, -1.0], vec![0.25]];
    let roles = [ParamRole::Trainable, ParamRole::Trainable];
    let mut optimizer = Sgd::new();

    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("长度一致");

    assert_eq!(params[0], [0.95_f32, 2.1]);
    assert_eq!(params[1], [0.475]);
}

/// 权重衰减只作用于非豁免组：归一化系数不衰减，投影权重按解耦口径衰减。
#[test]
fn weight_decay_applies_only_outside_exempt_groups() {
    let mut params = [vec![1.0_f32], vec![1.0]];
    let mut grads = [vec![0.2_f32], vec![0.2]];
    let roles = [ParamRole::Norm, ParamRole::Trainable];
    let mut optimizer = Sgd::new().with_weight_decay(0.1);

    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("长度一致");

    // 豁免组：只减梯度项 0.1·0.2 = 0.02。
    assert!((params[0][0] - 0.98).abs() < 1e-7);
    // 衰减组：更新 = 0.2 + 0.1·1.0 = 0.3，减 0.1·0.3 = 0.03。
    assert!((params[1][0] - 0.97).abs() < 1e-7);
}

/// 裁剪按全局范数等比缩比：梯度极大时一步更新仍在阈值口径内。
#[test]
fn global_norm_clipping_scales_every_slot() {
    let mut params = [vec![1.0_f32]];
    let mut grads = [vec![100.0_f32]];
    let roles = [ParamRole::Trainable];
    let mut optimizer = Sgd::new().with_clip_norm(1.0);

    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 1.0)
        .expect("长度一致");

    // 缩比 0.01：有效梯度 1.0，参数减 1.0。
    assert!((params[0][0] - 0.0).abs() < 1e-7);
}

/// 参数与梯度长度不一致是接线错误，一步都不更新。
#[test]
fn a_length_mismatch_is_rejected_before_update() {
    let mut params = [vec![1.0_f32, 2.0]];
    let mut grads = [vec![0.5_f32]];
    let roles = [ParamRole::Trainable];
    let mut optimizer = Sgd::new();

    let error = optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect_err("长度不一致");
    assert_eq!(error.code(), Code::Invalid);
    assert_eq!(params[0], [1.0, 2.0], "拒绝前不改写参数");
}

/// Adam 一步后权重沿梯度方向下降；矩状态由优化器自持，第二步不再是裸梯度方向。
#[test]
fn adam_moves_the_param_with_momentum() {
    let mut params = [vec![1.0_f32]];
    let mut grads = [vec![0.2_f32]];
    let roles = [ParamRole::Trainable];
    let mut optimizer = Adam::new();

    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("长度一致");
    assert!(params[0][0] < 1.0, "第一步沿梯度方向下降");
    assert!(params[0][0] > 0.85, "偏差修正后步幅小于裸 SGD");

    let first = params[0][0];
    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("第二步");
    assert!(params[0][0] < first, "矩状态跨步保留，继续下降");
}

/// 矩状态可导出并回载进新优化器：快照搬运的是这一份表，优化器之外不另存副本。
#[test]
fn adam_moments_round_trip_through_restore() {
    let mut params = [vec![1.0_f32, 2.0]];
    let mut grads = [vec![0.2_f32, -0.1]];
    let roles = [ParamRole::Trainable];
    let mut optimizer = Adam::new();
    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("第一步");
    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("第二步");

    let momentum = optimizer.momentum().iter().map(Vec::clone).collect();
    let variance = optimizer.variance().iter().map(Vec::clone).collect();
    let step = optimizer.step_count();

    let mut restored = Adam::new();
    restored.restore(momentum, variance, step).expect("回载");
    assert_eq!(restored.step_count(), step);
    let bits = |table: &[Vec<f32>]| {
        table
            .iter()
            .map(|slot| slot.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(restored.momentum()),
        bits(optimizer.momentum()),
        "一阶矩逐位恢复"
    );
    assert_eq!(
        bits(restored.variance()),
        bits(optimizer.variance()),
        "二阶矩逐位恢复"
    );

    // 回载后矩状态延续：第三步的一阶矩含上一时刻的动量，不是从零起。
    let mut params_restored = [params[0].clone()];
    let mut params_source = [params[0].clone()];
    restored
        .step(&mut slots(&mut params_restored, &mut grads, &roles), 0.1)
        .expect("回载后第一步");
    optimizer
        .step(&mut slots(&mut params_source, &mut grads, &roles), 0.1)
        .expect("源优化器同一步");
    assert_eq!(params_restored[0], params_source[0], "回载后轨迹逐位对齐");
}

/// 回载的两张表槽位数或槽内长度不符：接线错误，先报错不改写矩状态。
#[test]
fn restore_rejects_mismatched_moment_tables() {
    let mut optimizer = Adam::new();
    let error = optimizer
        .restore(vec![vec![1.0]], vec![vec![1.0, 2.0]], 3)
        .expect_err("长度不符");
    assert_eq!(error.code(), Code::Invalid);
    let error = optimizer
        .restore(vec![vec![1.0]], vec![], 3)
        .expect_err("槽位数不符");
    assert_eq!(error.code(), Code::Invalid);
}

/// 对象安全：优化器可装进 `&mut dyn Optimizer`。
#[test]
fn optimizer_is_object_safe() {
    let mut params = [vec![1.0_f32]];
    let mut grads = [vec![0.5_f32]];
    let roles = [ParamRole::Trainable];
    let mut optimizer: Box<dyn Optimizer> = Box::new(Sgd::new());

    optimizer
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect("长度一致");
    assert!((params[0][0] - 0.95).abs() < 1e-7);
}
