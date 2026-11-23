use super::*;

/// 确定性采样：伪随机但无外部依赖，同一个 `(len, salt)` 永远同一个序列。
fn sample(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((x >> 8) as f32) / 256.0 - 32.0
        })
        .collect()
}

fn assert_tiled_matches(m: usize, n: usize, k: usize) {
    let a = sample(m * k, 1);
    let b = sample(k * n, 2);
    let mut expected = vec![0.0f32; m * n];
    let mut actual = vec![0.0f32; m * n];
    forward_scalar(&a, &b, m, n, k, &mut expected).expect("参考可计算");
    forward_tiled(&a, &b, m, n, k, &mut actual).expect("tile 可计算");
    // K=4096 级 f32 累加的顺序误差在 1e-3 量级，rtol 取 1e-3、atol 取 1e-3。
    close_enough(&actual, &expected, 1e-3, 1e-3).expect("tile 与标量参考在容差内一致");
}

#[test]
fn tiled_matches_scalar_on_standard_shape() {
    assert_tiled_matches(256, 256, 4096);
}

#[test]
fn tiled_matches_scalar_on_tile_multiples() {
    assert_tiled_matches(128, 64, 96);
}

#[test]
fn tiled_matches_scalar_on_ragged_edges() {
    // 非 tile 整数倍：末块带守卫（TILE_M/N/K 都切出残边）。
    assert_tiled_matches(100, 70, 4093);
}

#[test]
fn tiled_matches_scalar_on_trivial_shapes() {
    assert_tiled_matches(1, 1, 1);
    assert_tiled_matches(1, 8, 3);
    assert_tiled_matches(8, 1, 3);
}

/// 线性层布局对拍：同一份输入与权重，标量参考与 tile 在 IEEE 容差内一致。
fn assert_tn_matches(rows: usize, n: usize, k: usize) {
    let x = sample(rows * k, 3);
    let w = sample(n * k, 4);
    let mut expected = vec![0.0f32; rows * n];
    let mut actual = vec![0.0f32; rows * n];
    forward_tn_scalar(&x, &w, rows, n, k, &mut expected).expect("参考可计算");
    forward_tn_tiled(&x, &w, rows, n, k, &mut actual).expect("tile 可计算");
    close_enough(&actual, &expected, 1e-3, 1e-3).expect("线性层 tile 与标量参考一致");
}

#[test]
fn tn_tiled_matches_scalar_on_prefill_shape() {
    // 行批 256、输出 128、收缩轴 1024：prefill 投影的代表形状。
    assert_tn_matches(256, 128, 1024);
}

#[test]
fn tn_tiled_matches_scalar_on_tile_multiples() {
    assert_tn_matches(TN_ROWS, TN_COLS, TN_K);
}

#[test]
fn tn_tiled_matches_scalar_on_ragged_edges() {
    // 非 tile 整数倍：行块、列块、收缩块三处都切出残边。
    assert_tn_matches(TN_ROWS + 3, TN_COLS + 5, TN_K + 7);
}

#[test]
fn tn_tiled_matches_scalar_on_single_row() {
    // decode 口径：一行输入也必须与 `gemv_f32` 同一算式（同一份点积顺序）。
    assert_tn_matches(1, 8, 13);
}

#[test]
fn tn_empty_contraction_yields_zeros() {
    let mut out = vec![1.0f32; 6];
    forward_tn_tiled(&[], &[], 2, 3, 0, &mut out).expect("k=0 合法");
    assert_eq!(out, vec![0.0f32; 6], "空收缩轴输出必须全零");
}

#[test]
fn tn_forward_rejects_mismatched_bufs() {
    let x = vec![0.0f32; 8];
    let w = vec![0.0f32; 8];
    let mut out = vec![0.0f32; 8];
    assert!(
        forward_tn_tiled(&x, &w, 3, 3, 3, &mut out).is_err(),
        "尺寸不符必须报错"
    );
    assert!(
        forward_tn_scalar(&x, &w, 2, 2, 2, &mut out).is_err(),
        "标量入口也必须报错"
    );
}

#[test]
fn empty_contraction_yields_zeros() {
    let mut out = vec![1.0f32; 4];
    forward_tiled::<f32>(&[], &[], 2, 2, 0, &mut out).expect("k=0 合法");
    assert_eq!(out, vec![0.0f32; 4], "空收缩轴输出必须全零");
}

#[test]
fn forward_rejects_mismatched_bufs() {
    let a = vec![0.0f32; 8];
    let b = vec![0.0f32; 8];
    let mut out = vec![0.0f32; 8];
    assert!(
        forward_scalar(&a, &b, 3, 3, 3, &mut out).is_err(),
        "尺寸不符必须报错"
    );
}

/// E10 计算可预测性（`#[ignore]`：
/// `cargo test -p llm-nn --lib -- e10_predictability --release --ignored --nocapture`）。
///
/// 口径与 §10.10 E10 卡一致：预热后对固定窗口（20 次 512×512×512 tile 矩阵乘）计 120 个
/// 窗口墙钟，`CV_t = σ_t/μ_t`，并报 P50/P95/P99。热节流项本机无传感器轨迹，单列不混入。
#[test]
#[ignore = "E10 可预测性基准：cargo test -p llm-nn --lib -- e10_predictability --release --ignored --nocapture"]
fn e10_predictability() {
    let (m, n, k) = (512usize, 512usize, 512usize);
    let a = sample(m * k, 11);
    let b = sample(k * n, 12);
    let mut out = vec![0.0f32; m * n];

    for _ in 0..20 {
        matmul_tiled(&a, &b, m, n, k, &mut out);
    }

    let iters_per_window = 20usize;
    let windows = 120usize;
    let mut times = Vec::with_capacity(windows);
    for _ in 0..windows {
        let t = std::time::Instant::now();
        for _ in 0..iters_per_window {
            matmul_tiled(&a, &b, m, n, k, &mut out);
            std::hint::black_box(&out);
        }
        times.push(t.elapsed().as_secs_f64());
    }

    let count = times.len() as f64;
    let mean = times.iter().sum::<f64>() / count;
    let var = times.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / (count - 1.0);
    let std = var.sqrt();
    let cv = std / mean;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| -> f64 { sorted[((p * (sorted.len() as f64 - 1.0)).round()) as usize] };
    println!(
        "E10 CV={:.4}%（mean={:.3} ms/window std={:.3} ms P50={:.3} P95={:.3} P99={:.3} windows={windows} iters/window={iters_per_window} shape={m}x{n}x{k}）",
        cv * 100.0,
        mean * 1e3,
        std * 1e3,
        q(0.50) * 1e3,
        q(0.95) * 1e3,
        q(0.99) * 1e3,
    );
}
