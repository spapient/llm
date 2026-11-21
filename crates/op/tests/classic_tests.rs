use super::*;
use half::{bf16, f16};

/// 每个算子的样本 -> 参考必须自洽（参考不能报错，长度与 `y` 缓冲一致）。
fn check(name: &str) {
    let model = crate::model(name).expect("已注册");
    let sample = (model.sample)(model.shape.0, model.shape.1);
    let values = (model.reference)(&sample).expect("参考可计算");
    let y = sample.f32("y").expect("y 存在");
    assert_eq!(values.len(), y.len(), "{name}: 参考长度必须等于 y 缓冲");
}

#[test]
fn every_classic_sample_matches_its_reference() {
    for name in [
        "max_pool1d_f32",
        "max_pool2d_f32",
        "avg_pool2d_f32",
        "adaptive_pool_f32",
        "interpolate_f32",
        "batch_norm_f32",
        "group_norm_f32",
        "instance_norm_f32",
        "layer_norm_f32",
    ] {
        check(name);
    }
}

#[test]
fn norm_references_are_statistically_sane() {
    // LayerNorm 参考：每行均值 0、方差 ≈ 1（eps 让方差略小于 1）。
    let sample = layer_norm::sample(1, 1);
    let y = layer_norm::reference(&sample).expect("参考可计算");
    for row in 0..3 {
        let part = &y[row * 4..(row + 1) * 4];
        let mean: f32 = part.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "行 {row} 均值 {mean}");
        let var: f32 = part.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "行 {row} 方差 {var}");
    }
    // BatchNorm 参考：批内两样本 × 空间 2 × 通道 4，输出必须全部有限。
    let sample = batch_norm::sample(1, 1);
    let y = batch_norm::reference(&sample).expect("参考可计算");
    assert_eq!(y.len(), 16);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn pool_references_pick_exact_window_extremes() {
    let sample = max_pool1d::sample(1, 1);
    let x = sample.f32("x").expect("x");
    let y = max_pool1d::reference(&sample).expect("参考可计算");
    for (o, value) in y.iter().enumerate() {
        let want = x[2 * o].max(x[2 * o + 1]);
        assert_eq!(*value, want, "位置 {o}");
    }
}

/// 把 `f32` 切片转成标量 `S` 的切片（精度对拍前的搬运用，不参与算术）。
fn cast<S: Scalar>(x: &[f32]) -> Vec<S> {
    x.iter().map(|&v| S::from_f64(v as f64)).collect()
}

/// 把标量 `S` 的切片转回 `f32`（对拍读数，只做展示级转换）。
fn back<S: Scalar>(x: &[S]) -> Vec<f32> {
    x.iter().map(|&v| v.to_f64() as f32).collect()
}

/// 逐位对拍：`a` 与 `b` 同长，逐元素差不超过 `tol`。
fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: 长度不一致");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{what}[{i}]: {x} != {y}（容差 {tol}）"
        );
    }
}

/// 窗口族精度对拍：给定 `f32` 参考，跑 `run::<S>` 并与参考逐位比较。
fn window_parity<S: Scalar>(
    sample: &Sample,
    want: &[f32],
    out_len: usize,
    run: impl Fn(&[S], &mut [S]) -> diagnostic::Result<()>,
    tol: f32,
    what: &str,
) {
    let x = cast::<S>(sample.f32("x").expect("x 缓冲"));
    let mut out = vec![S::ZERO; out_len];
    run(&x, &mut out).expect("run 应成功");
    close(want, &back(&out), tol, what);
}

/// 归一化族精度对拍：给定 `f32` 参考，跑 `run::<S>`（多带 gamma/beta）并逐位比较。
fn norm_parity<S: Scalar>(
    sample: &Sample,
    want: &[f32],
    out_len: usize,
    run: impl Fn(&[S], &[S], &[S], &mut [S]) -> diagnostic::Result<()>,
    tol: f32,
    what: &str,
) {
    let x = cast::<S>(sample.f32("x").expect("x 缓冲"));
    let gamma = cast::<S>(sample.f32("gamma").expect("gamma 缓冲"));
    let beta = cast::<S>(sample.f32("beta").expect("beta 缓冲"));
    let mut out = vec![S::ZERO; out_len];
    run(&x, &gamma, &beta, &mut out).expect("run 应成功");
    close(want, &back(&out), tol, what);
}

/// 窗口算子 `run` 在 f16 / bf16 / f64 下与 f32 参考一致：同一份泛型算术，四种精度一条路径。
macro_rules! window_precisions {
    ($mod:ident, $out_len:expr) => {
        let sample = $mod::sample(1, 1);
        let want = $mod::reference(&sample).expect("f32 参考");
        window_parity::<f16>(
            &sample,
            &want,
            $out_len,
            $mod::run::<f16>,
            1e-2,
            concat!(stringify!($mod), " f16"),
        );
        window_parity::<bf16>(
            &sample,
            &want,
            $out_len,
            $mod::run::<bf16>,
            5e-2,
            concat!(stringify!($mod), " bf16"),
        );
        window_parity::<f64>(
            &sample,
            &want,
            $out_len,
            $mod::run::<f64>,
            1e-5,
            concat!(stringify!($mod), " f64"),
        );
    };
}

#[test]
fn window_run_is_precision_consistent() {
    window_precisions!(max_pool1d, max_pool1d::OL);
    window_precisions!(max_pool2d, max_pool2d::H * max_pool2d::W);
    window_precisions!(avg_pool2d, avg_pool2d::H * avg_pool2d::W);
    window_precisions!(adaptive_pool, adaptive_pool::OH * adaptive_pool::OW);
    window_precisions!(interpolate, interpolate::OH * interpolate::OW);
}

/// 归一化算子 `run` 在 f16 / bf16 / f64 下与 f32 参考一致。
macro_rules! norm_precisions {
    ($mod:ident, $out_len:expr) => {
        let sample = $mod::sample(1, 1);
        let want = $mod::reference(&sample).expect("f32 参考");
        norm_parity::<f16>(
            &sample,
            &want,
            $out_len,
            $mod::run::<f16>,
            2e-2,
            concat!(stringify!($mod), " f16"),
        );
        norm_parity::<bf16>(
            &sample,
            &want,
            $out_len,
            $mod::run::<bf16>,
            1e-1,
            concat!(stringify!($mod), " bf16"),
        );
        norm_parity::<f64>(
            &sample,
            &want,
            $out_len,
            $mod::run::<f64>,
            1e-5,
            concat!(stringify!($mod), " f64"),
        );
    };
}

#[test]
fn norm_run_is_precision_consistent() {
    norm_precisions!(batch_norm, batch_norm::N * batch_norm::S * batch_norm::C);
    norm_precisions!(group_norm, group_norm::N * group_norm::S * group_norm::C);
    norm_precisions!(
        instance_norm,
        instance_norm::N * instance_norm::S * instance_norm::C
    );
    norm_precisions!(layer_norm, layer_norm::R * layer_norm::W);
}
