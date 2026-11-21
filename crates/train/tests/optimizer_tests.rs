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
            shape: None,
        })
        .collect()
}

/// 带矩阵形状的槽视图：Muon 的入口要求形状随槽一起给出。
fn matrix_slots<'a>(
    params: &'a mut [Vec<f32>],
    grads: &'a mut [Vec<f32>],
    roles: &[ParamRole],
    shape: (usize, usize),
) -> Vec<Slot<'a>> {
    slots(params, grads, roles)
        .into_iter()
        .map(|slot| Slot {
            shape: Some(shape),
            ..slot
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

/// 正交化把谱压窄：`3 : 2 : 1` 的传播被拉到窄带里，方向不再被几个分量独占。
///
/// 五次迭代给的是**近似**极分解（收敛比约 0.72，值在 `[0.65, 1.15]` 一带振荡），所以断言的是
/// “压缩比”而不是“精确等于 1”：后者要几十次迭代，而优化器只需要方向不失衡。
#[test]
fn the_orthogonalization_compresses_the_spectrum() {
    // 对角谱 1 / 2 / 3：不处理的话，更新量会按 3 : 2 : 1 分配。
    let grad = vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let out = orthogonalize(&grad, 3, 3, 1e-7);
    for (index, value) in out.iter().enumerate() {
        assert!(value.is_finite(), "第 {index} 项不是有限值");
    }
    let spread = out[0] / out[8];
    assert!(
        (0.5..=1.8).contains(&spread),
        "谱比从 3.0 被压到 {spread}，该落进 0.5..=1.8 的窄带"
    );
    // 对角输入只受多项式作用，非对角项仍是零：正交化不动特征基。
    assert!(
        out[1].abs() < 1e-6 && out[3].abs() < 1e-6,
        "非对角项该保持为零"
    );
}

/// 更新量与梯度尺度无关：整体乘 1000，一步走多远不变。
#[test]
fn the_update_ignores_the_gradient_scale() {
    let base = vec![1.0_f32, -2.0, 0.5, 3.0, 0.0, -1.0];
    let mut params_small = [vec![1.0_f32; 6]];
    let mut params_large = [vec![1.0_f32; 6]];
    let mut grads_small = [base.clone()];
    let mut grads_large = [base.iter().map(|value| value * 1000.0).collect()];
    let roles = [ParamRole::Trainable];

    Muon::new()
        .without_nesterov()
        .step(
            &mut matrix_slots(&mut params_small, &mut grads_small, &roles, (2, 3)),
            0.1,
        )
        .expect("几何合法");
    Muon::new()
        .without_nesterov()
        .step(
            &mut matrix_slots(&mut params_large, &mut grads_large, &roles, (2, 3)),
            0.1,
        )
        .expect("几何合法");

    for (index, (small, large)) in params_small[0].iter().zip(&params_large[0]).enumerate() {
        assert!(
            (small - large).abs() < 1e-3,
            "第 {index} 项：小梯度 {small} 与大梯度 {large} 该给出同一步"
        );
        assert!((small - 1.0).abs() > 1e-9, "第 {index} 项根本没被更新");
    }
}

/// 非矩阵槽被拒且一步都不写回：跳过它等于让某个组永远不被训练。
#[test]
fn a_flat_slot_is_refused_before_any_update() {
    let mut params = [vec![1.0_f32, 2.0, 3.0]];
    let mut grads = [vec![0.5_f32, 0.5, 0.5]];
    let roles = [ParamRole::Bias];
    let error = Muon::new()
        .step(&mut slots(&mut params, &mut grads, &roles), 0.1)
        .expect_err("偏置不是矩阵");
    assert_eq!(error.code(), Code::Invalid);
    assert_eq!(params[0], [1.0, 2.0, 3.0], "拒绝前不改写参数");
}

/// 声明的行列积与参数长度不符：接线错误，在槽校验处就失败。
#[test]
fn a_shape_that_does_not_match_the_length_is_refused() {
    let mut params = [vec![1.0_f32, 2.0, 3.0, 4.0]];
    let mut grads = [vec![0.5_f32; 4]];
    let roles = [ParamRole::Trainable];
    let error = Muon::new()
        .step(
            &mut matrix_slots(&mut params, &mut grads, &roles, (3, 3)),
            0.1,
        )
        .expect_err("3 × 3 不是 4 个元素");
    assert_eq!(error.code(), Code::Invalid);
}

/// 非有限梯度在入口被拒：正交化会把 `NaN` 铺满整个方向。
#[test]
fn a_non_finite_gradient_is_refused() {
    let mut params = [vec![1.0_f32; 4]];
    let mut grads = [vec![1.0_f32, f32::NAN, 0.0, 0.0]];
    let roles = [ParamRole::Trainable];
    let error = Muon::new()
        .step(
            &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
            0.1,
        )
        .expect_err("梯度含 NaN");
    assert_eq!(error.code(), Code::Invalid);
    assert_eq!(params[0], [1.0, 1.0, 1.0, 1.0]);
}

/// 零梯度给出零更新：没有方向就没有正交化的东西，也不需要另开分支。
#[test]
fn a_zero_gradient_leaves_the_matrix_alone() {
    let mut params = [vec![1.0_f32, -2.0, 3.0, 0.5]];
    let mut grads = [vec![0.0_f32; 4]];
    let roles = [ParamRole::Trainable];
    Muon::new()
        .step(
            &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
            0.1,
        )
        .expect("几何合法");
    assert_eq!(params[0], [1.0, -2.0, 3.0, 0.5], "零方向不该动参数");
}

/// 解耦衰减跳过豁免角色：偏置原样保留，可训练权重按 `lr·decay` 缩。
#[test]
fn the_decay_still_follows_the_role() {
    let mut params = [vec![1.0_f32; 4], vec![1.0_f32; 4]];
    let mut grads = [vec![0.0_f32; 4], vec![0.0_f32; 4]];
    let roles = [ParamRole::Trainable, ParamRole::Bias];
    Muon::new()
        .with_weight_decay(0.1)
        .step(
            &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
            0.5,
        )
        .expect("几何合法");
    assert!(
        (params[0][0] - (1.0 - 0.5 * 0.1)).abs() < 1e-7,
        "可训练权重按衰减缩"
    );
    assert_eq!(params[1][0], 1.0, "偏置豁免权重衰减");
}

/// 动量跨步累积：两步同梯度后缓冲为 `μ(μ·g + g) + g`。
#[test]
fn the_velocity_accumulates_across_steps() {
    let mut params = [vec![1.0_f32; 4]];
    let mut grads = [vec![0.0_f32, 1.0, 0.0, 0.0]];
    let roles = [ParamRole::Trainable];
    let mut optimizer = Muon::new().with_momentum(0.5);
    for _ in 0..2 {
        optimizer
            .step(
                &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
                0.1,
            )
            .expect("几何合法");
    }
    let velocity = &optimizer.momentum_table()[0][1];
    let want = (1.0 - 0.5_f32.powi(2)) / (1.0 - 0.5);
    assert!(
        (velocity - want).abs() < 1e-6,
        "缓冲是 {velocity}，手算是 {want}"
    );
    assert_eq!(optimizer.step_count(), 2);
}

/// 回载后轨迹逐位对齐；二阶矩非空时拒绝（Muon 只持一阶动量）。
#[test]
fn the_restore_round_trip_matches_the_source() {
    let mut params = [vec![1.0_f32; 4]];
    let mut grads = [vec![0.25_f32, -1.0, 2.0, 0.0]];
    let roles = [ParamRole::Trainable];
    let mut source = Muon::new();
    source
        .step(
            &mut matrix_slots(&mut params, &mut grads, &roles, (2, 2)),
            0.1,
        )
        .expect("第一步");
    let table = source.momentum_table().to_vec();

    let mut restored = Muon::new();
    restored.restore(table, Vec::new(), 1).expect("表一致");
    let mut params_restored = [params[0].clone()];
    let mut params_source = [params[0].clone()];
    restored
        .step(
            &mut matrix_slots(&mut params_restored, &mut grads, &roles, (2, 2)),
            0.1,
        )
        .expect("回载后第一步");
    source
        .step(
            &mut matrix_slots(&mut params_source, &mut grads, &roles, (2, 2)),
            0.1,
        )
        .expect("源优化器同一步");
    assert_eq!(params_restored[0], params_source[0], "回载后轨迹逐位对齐");

    let error = Muon::new()
        .restore(Vec::new(), vec![vec![1.0]], 1)
        .expect_err("Muon 没有二阶矩");
    assert_eq!(error.code(), Code::Invalid);
}

/// 高瘦与矮宽走的是同一个迭代：转置后的方向就是转置。
#[test]
fn the_tall_and_wide_branches_are_the_same_iteration() {
    let tall = vec![0.5_f32, -1.0, 2.0, 0.25, 3.0, -0.5];
    let mut wide = Vec::with_capacity(tall.len());
    for col in 0..2 {
        for row in 0..3 {
            wide.push(tall[row * 2 + col]);
        }
    }
    let tall_out = orthogonalize(&tall, 3, 2, 1e-7);
    let wide_out = orthogonalize(&wide, 2, 3, 1e-7);
    for row in 0..3 {
        for col in 0..2 {
            let lhs = tall_out[row * 2 + col];
            let rhs = wide_out[col * 3 + row];
            assert!((lhs - rhs).abs() < 1e-6, "({row}, {col})：{lhs} 对 {rhs}");
        }
    }
}
