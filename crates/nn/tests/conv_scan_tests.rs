use super::*;

/// f64 下半群合成因浮点非结合律产生 ulp 级差异，用远高于 ulp、远低于有意义误差的容差比较。
const TOL: f64 = 1e-9;

fn assert_step_close(a: SsmStep<f64>, b: SsmStep<f64>) {
    assert!((a.a - b.a).abs() <= TOL, "a 偏差：{} vs {}", a.a, b.a);
    assert!((a.b - b.b).abs() <= TOL, "b 偏差：{} vs {}", a.b, b.b);
}

fn assert_seq_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "序列长度不一致");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= TOL, "第 {i} 项偏差：{x} vs {y}");
    }
}

/// 确定性 xorshift64（测试内无外部 RNG 依赖）。
struct Xs(u64);

impl Xs {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// 高 53 位归一化到 `[0,1)`。
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// 随机半群步：`a ∈ (0.01, 0.99)` 保证衰减稳定，`b ∈ [-1,1)`。
fn step(xs: &mut Xs) -> SsmStep<f64> {
    SsmStep {
        a: 0.01 + 0.98 * xs.next_f64(),
        b: 2.0 * xs.next_f64() - 1.0,
    }
}

/// 数学验证 5a：半群合成满足结合律（f64 下允许 ulp 级差异）。
#[test]
fn combine_is_associative() {
    let mut xs = Xs(0x9e37_79b9_7f4a_7c15);
    for _ in 0..1000 {
        let (s1, s2, s3) = (step(&mut xs), step(&mut xs), step(&mut xs));
        let left = s1.combine(s2).combine(s3);
        let right = s1.combine(s2.combine(s3));
        assert_step_close(left, right);
    }
}

/// 幺元是合成中性元。
#[test]
fn identity_is_neutral() {
    let mut xs = Xs(1);
    for _ in 0..100 {
        let s = step(&mut xs);
        assert_eq!(s.combine(SsmStep::identity()), s, "右乘幺元不变");
        assert_eq!(SsmStep::identity().combine(s), s, "左乘幺元不变");
    }
}

/// 数学验证 5b：Blelloch 非交换独占扫描与串行前缀在 64 点全一致（§5.10 容差）。
#[test]
fn blelloch_matches_serial_64() {
    let mut xs = Xs(2);
    let steps: Vec<_> = (0..64).map(|_| step(&mut xs)).collect();
    let serial = scan_serial(&steps);
    let parallel = scan_blelloch(&steps);
    assert_eq!(serial.len(), parallel.len());
    for (a, b) in serial.into_iter().zip(parallel) {
        assert_step_close(a, b);
    }
}

/// 前缀逐一 `apply` 与逐步递推一致（f64 下允许 ulp 级差异）。
#[test]
fn apply_matches_recurrence() {
    let mut xs = Xs(3);
    let steps: Vec<_> = (0..37).map(|_| step(&mut xs)).collect();
    let init = xs.next_f64();
    let expected = apply_scan(&steps, init);
    // 独占前缀给前 n 个状态，末态用全量合成一次施加补齐。
    let mut got: Vec<_> = scan_blelloch(&steps)
        .iter()
        .map(|p| p.apply(init))
        .collect();
    let total = steps
        .iter()
        .fold(SsmStep::identity(), |acc, s| acc.combine(*s));
    got.push(total.apply(init));
    assert_seq_close(&expected, &got);
}

/// 半群全量合成一次施加 == 逐步递推的末态。
#[test]
fn scan_matches_semigroup_recurrence() {
    let mut xs = Xs(4);
    let steps: Vec<_> = (0..50).map(|_| step(&mut xs)).collect();
    let init = 0.7;
    let mut state = init;
    for s in &steps {
        state = s.apply(state);
    }
    let mut total = SsmStep::identity();
    for s in &steps {
        total = total.combine(*s);
    }
    assert!((total.apply(init) - state).abs() < 1e-12);
}

/// M1 log-log 斜率实测（`#[ignore]`：
/// `cargo test -p llm-nn --lib -- scan_loglog_slope --release --ignored --nocapture`）。
///
/// 口径与 §10.10 M1 卡一致：n ∈ {4K..128K}，每点多测取最小值，对 (ln n, ln T) 做最小二乘，
/// 报斜率 k 与其 95% CI。产品串行核 `scan_f32::forward` 与第二份实现 `scan_blelloch` 各测一条。
#[test]
#[ignore = "M1 复杂度基准：cargo test -p llm-nn --lib -- scan_loglog_slope --release --ignored --nocapture"]
fn scan_loglog_slope() {
    let ns = [4096usize, 8192, 16384, 32768, 65536, 131072];
    let reps = 30usize;

    // 确定性输入（xorshift64）。
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next_f32 = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as u32 as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    };

    // 产品串行核：s = 0.5·s + 1.0·x[t]。
    let mut fwd_points = Vec::new();
    for &n in &ns {
        let x: Vec<f32> = (0..n).map(|_| next_f32()).collect();
        let mut out = vec![0.0f32; n];
        for _ in 0..3 {
            crate::conv::scan::f32::forward(&x, &mut out).expect("等长");
        }
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            crate::conv::scan::f32::forward(&x, &mut out).expect("等长");
            best = best.min(t.elapsed().as_secs_f64());
        }
        fwd_points.push((n as f64, best));
    }
    let (k, ci, r2) = loglog_fit(&fwd_points);
    println!(
        "M1 scan_f32::forward 斜率 k={k:.4} 95%CI=[{:.4},{:.4}] R²={r2:.6} 点={:?}",
        ci.0,
        ci.1,
        fwd_points
            .iter()
            .map(|(n, t)| format!("{n:.0}:{t:.3e}s"))
            .collect::<Vec<_>>()
    );

    // 半群 Blelloch 并行扫描（f32）。
    let mut ble_points = Vec::new();
    for &n in &ns {
        let steps: Vec<SsmStep<f32>> = (0..n)
            .map(|_| SsmStep {
                a: 0.5,
                b: next_f32(),
            })
            .collect();
        for _ in 0..3 {
            std::hint::black_box(scan_blelloch(&steps));
        }
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            std::hint::black_box(scan_blelloch(&steps));
            best = best.min(t.elapsed().as_secs_f64());
        }
        ble_points.push((n as f64, best));
    }
    let (k, ci, r2) = loglog_fit(&ble_points);
    println!(
        "M1 scan_blelloch       斜率 k={k:.4} 95%CI=[{:.4},{:.4}] R²={r2:.6} 点={:?}",
        ci.0,
        ci.1,
        ble_points
            .iter()
            .map(|(n, t)| format!("{n:.0}:{t:.3e}s"))
            .collect::<Vec<_>>()
    );
}

/// (ln n, ln T) 最小二乘：返回 (斜率 k, 95% CI, R²)。
fn loglog_fit(points: &[(f64, f64)]) -> (f64, (f64, f64), f64) {
    let xs: Vec<f64> = points.iter().map(|(n, _)| n.ln()).collect();
    let ys: Vec<f64> = points.iter().map(|(_, t)| t.ln()).collect();
    let n = xs.len() as f64;
    let (xbar, ybar) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxx = xs.iter().map(|x| (x - xbar) * (x - xbar)).sum::<f64>();
    let sxy = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - xbar) * (y - ybar))
        .sum::<f64>();
    let k = sxy / sxx;
    let intercept = ybar - k * xbar;
    let residuals: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| y - (intercept + k * x))
        .collect();
    let sse = residuals.iter().map(|r| r * r).sum::<f64>();
    let sst = ys.iter().map(|y| (y - ybar) * (y - ybar)).sum::<f64>();
    let r2 = 1.0 - sse / sst;
    // 斜率标准误；df = 点数 - 2，n=6 时 t(0.975,4)=2.776。
    let se = (sse / (n - 2.0) / sxx).sqrt();
    let tcrit = 2.776;
    (k, (k - tcrit * se, k + tcrit * se), r2)
}

/// `λ = 0` 时退化为 `ā = 1`、`b̄ = dt·b`。
#[test]
fn exp_zero_order_zero_lambda() {
    let s = exp_zero_order(0.0f64, 2.0, 0.5);
    assert_eq!(s.a, 1.0);
    assert_eq!(s.b, 1.0);
}

/// 零阶保持与直接 `exp` / `exp_m1` 公式一致。
#[test]
fn exp_zero_order_matches_direct() {
    let (lam, b, dt) = (-0.5f64, 3.0, 0.25);
    let s = exp_zero_order(lam, b, dt);
    let expected_a = (lam * dt).exp();
    let expected_b = (lam * dt).exp_m1() / lam * b;
    assert!((s.a - expected_a).abs() < 1e-12);
    assert!((s.b - expected_b).abs() < 1e-12);
}

/// 幺元块不改状态。
#[test]
fn plane_identity_leaves_state() {
    assert_eq!(PlaneStep::identity().apply([0.3, -0.8], 2.5), [0.3, -0.8]);
}

/// `apply` 就是复数乘加：`ā·s + b̄·select`。
#[test]
fn plane_apply_is_complex_arithmetic() {
    let (a_re, a_im, b_re, b_im) = (0.6f64, 0.8, -0.2, 0.5);
    let s = PlaneStep {
        a: [a_re, a_im],
        b: [b_re, b_im],
    };
    let (re, im) = (0.7f64, -1.1);
    let select = 2.0f64;
    let [got_re, got_im] = s.apply([re, im], select);
    assert!((got_re - (a_re * re - a_im * im + b_re * select)).abs() < 1e-12);
    assert!((got_im - (a_im * re + a_re * im + b_im * select)).abs() < 1e-12);
}

/// 旋转只改相位、不改模长：任意 `ω` 下 `|ā| = exp(λ_re·dt)`。
#[test]
fn rotation_preserves_magnitude() {
    let dt = 0.125f64;
    let expected = (-0.5 * dt).exp();
    for omega in [0.0, 0.3, 1.7, -2.9] {
        let [re, im] = exp_trapezoid(-0.5f64, omega, dt, 1.0, 1.0).a;
        assert!((re.hypot(im) - expected).abs() < 1e-12, "ω={omega}");
    }
}

/// `ω = 0` 时块退化为实衰减，输入系数的虚部为零。
#[test]
fn zero_frequency_is_real() {
    let s = exp_trapezoid(-0.4f64, 0.0, 0.25, 0.5, 1.0);
    assert_eq!(s.a[1], 0.0);
    assert_eq!(s.b[1], 0.0);
    let expected_b = 0.25 / 2.0 * ((-0.4 * 0.25f64).exp() * 0.5 + 1.0);
    assert!((s.b[0] - expected_b).abs() < 1e-12);
}

/// 双抽头梯形与零阶保持是两条式子：同样的输入在本步给出不同的状态。
#[test]
fn trapezoid_and_zero_order_differ() {
    let (lam, dt) = (-0.6f64, 0.3);
    let order = exp_zero_order(lam, 1.0, dt).apply(0.0);
    // 零阶保持把这一步的输入铺满整段步长，梯形只铺两端平均：从零状态起跳时差近一倍。
    let half = exp_trapezoid(lam, 0.0, dt, 0.0, 1.0).apply([0.0, 0.0], 1.0)[0];
    assert!(half < order * 0.6, "{half} vs {order}");
    // 两端相等时两条式子仍然不同——步内线性与常数保持只在 `dt → 0` 的极限重合。
    let flat = exp_trapezoid(lam, 0.0, dt, 1.0, 1.0).apply([0.0, 0.0], 1.0)[0];
    assert!((flat - order).abs() > 1e-4, "{flat} vs {order}");
}

/// 梯形真的用了上一个抽头：`x_{t−1}` 不同，本步状态就不同。
#[test]
fn trapezoid_uses_the_previous_tap() {
    let (lam, omega, dt) = (-0.5f64, 0.9, 0.2);
    let a = exp_trapezoid(lam, omega, dt, 0.0, 1.0).apply([0.0, 0.0], 1.0);
    let b = exp_trapezoid(lam, omega, dt, 1.0, 1.0).apply([0.0, 0.0], 1.0);
    assert!(a[0] != b[0] || a[1] != b[1]);
}
