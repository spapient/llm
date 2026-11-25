//! 优化器的代数性质：闭式解可断言的地方就断言等号，不留近似余地。
//!
//! 这里最有分量的一条是 Adam 的**第一步**：偏差修正把一阶、二阶矩都归约成同一个
//! `|g|`，于是每个坐标恰好走 `lr`，与梯度幅值无关。少了任一处 `/(1-βᵗ)`
//! 这个等号立刻塌成"位移与梯度成正比"，最小反例是任意一组非零梯度。

use super::*;

use nn::ParamRole;
use proptest::prelude::*;
use testkit::prop;

/// 一条 `Trainable` 通道，参数与梯度等长。
fn step(optimizer: &mut impl Optimizer, params: &mut [f32], grads: &mut [f32], lr: f32) {
    let mut slots = [Slot {
        params,
        grads,
        role: ParamRole::Trainable,
        shape: None,
    }];
    optimizer.step(&mut slots, lr).expect("参数与梯度等长");
}

/// 非零梯度：十进数量级 `1e-3..=1e3` 取符号——避开 `eps` 主导的邻域，
/// 那里的闭式解会退化成"几乎不动"，等于什么都没测。
fn gradient() -> impl Strategy<Value = f32> {
    (-3.0f32..=3.0, any::<bool>()).prop_map(|(exponent, negative)| {
        let magnitude = 10.0f32.powf(exponent);
        if negative { -magnitude } else { magnitude }
    })
}

/// 长度固定的任意梯度，覆盖正负两侧。
fn vector(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-2.0f32..=2.0, len)
}

proptest! {
    #![proptest_config(prop::config())]

    #[test]
    fn the_first_adam_step_moves_every_coordinate_by_the_learning_rate(
        grad in gradient(),
        lr in 1e-4f32..=1e-1,
    ) {
        let (mut params, mut grads) = (vec![0.5f32], vec![grad]);
        step(&mut Adam::new(), &mut params, &mut grads, lr);
        let moved = params[0] - 0.5;
        prop_assert!(
            (moved.abs() - lr).abs() <= lr * 1e-3,
            "位移 {} 应等于 lr={}（梯度 {}）",
            moved.abs(),
            lr,
            grad,
        );
        prop_assert_eq!(
            moved.is_sign_negative(),
            grad.is_sign_positive(),
            "位移方向必须与梯度相反",
        );
    }

    /// 零梯度是 SGD 与 Adam 的**同一个**不动点：SGD 不动是因为没有方向，
    /// Adam 不动是因为 `m/(√v+ε)` 是 `0/(0+ε)`——若把 `ε` 挪到根号外，这里立刻变成 `NaN`。
    #[test]
    fn a_zero_gradient_is_a_fixed_point_of_both_optimizers(n in 1usize..=8, lr in 1e-4f32..=1e-1) {
        let base: Vec<f32> = (0..n).map(|i| 0.25 + i as f32).collect();
        let mut zeros = vec![0.0f32; n];
        let (mut sgd_params, mut adam_params) = (base.clone(), base.clone());
        step(&mut Sgd::new(), &mut sgd_params, &mut zeros.clone(), lr);
        step(&mut Adam::new(), &mut adam_params, &mut zeros, lr);
        prop::assert_exact("SGD 零梯度", &sgd_params, &base);
        prop::assert_exact("Adam 零梯度", &adam_params, &base);
    }

    /// 无衰减、无裁剪时 SGD 就是 `p - lr·g` 本身——这条是后一条的基准。
    #[test]
    fn sgd_is_exactly_the_learning_rate_times_the_gradient(
        grads in vector(4),
        lr in 1e-4f32..=1e-1,
    ) {
        let base = vec![0.75f32; grads.len()];
        let (mut params, mut moved) = (base.clone(), grads.clone());
        step(&mut Sgd::new(), &mut params, &mut moved, lr);
        let want: Vec<f32> = base.iter().zip(&grads).map(|(p, g)| p - lr * g).collect();
        prop::assert_within_rel(&params, &want, 1e-6, "SGD 单步");
    }

    /// 裁剪只缩比、不转向：位移模长恰好落在 `lr·max_norm` 的球面上，方向仍与梯度相反。
    /// 若把 `scale` 乘进参数而不是梯度，半径会随参数模长漂移，这里立刻对不上。
    #[test]
    fn clipping_lands_the_update_on_the_radius(
        raw in vector(4),
        max_norm in 1e-3f32..=1e-1,
        excess in 1.5f32..=4.0,
        lr in 1e-3f32..=1e-1,
    ) {
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        let unit = if norm > 1e-3 { raw } else { vec![1.0; 4] };
        let norm = unit.iter().map(|v| v * v).sum::<f32>().sqrt();
        // 梯度模长是阈值的 `excess` 倍，于是裁剪系数恰好是 `1/excess`。
        let grad: Vec<f32> = unit.iter().map(|v| v * (max_norm * excess / norm)).collect();
        let (mut params, mut moved) = (vec![0.0f32; 4], grad.clone());
        step(&mut Sgd::new().with_clip_norm(max_norm), &mut params, &mut moved, lr);
        let want: Vec<f32> = grad.iter().map(|g| -lr * g / excess).collect();
        prop::assert_within_rel(&params, &want, 1e-4, "裁剪后单步");
        let radius = want.iter().map(|v| v * v).sum::<f32>().sqrt();
        prop_assert!(
            (radius - lr * max_norm).abs() <= lr * max_norm * 1e-3,
            "位移模长 {} 应落在 lr·max_norm={}",
            radius,
            lr * max_norm,
        );
    }
}
