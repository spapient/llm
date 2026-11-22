//! 职责：大形状张量核 GEMM 的抽样点对拍判据。

use super::*;

/// 大形状的抽样校验：`2048×2048×4096`（`M·N·K = 1.7e10 > 1024³`）在夹具里只打印 `误差 NaN`，
/// 即**吞吐跑得再多也没有任何正确性证据**。这里取 512 个散点、每点 `O(K)` 的 f64 参考，
/// 用与张量核小形状同一判据（[`TENSOR_TOLERANCE`]）判定，把盲区换成实数。
///
/// 抽样而非全量：全量 f64 参考是 `1.7e10` 次乘法（分钟级），而「布局错位、尾块漏算、半个 tile
/// 没写」这类缺陷必然在 512 个散点上暴露——它不是精度认证，是**覆盖认证**。
#[test]
fn tensor_gemm_large_shape_is_verified_on_sampled_elements() {
    const M: usize = 2048;
    const N: usize = 2048;
    const K: usize = 4096;
    const SAMPLES: usize = 512;

    let Some((device, queue)) = bench_device("大形状抽样校验") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-large-sample");

    // 满尾数输入：夹具的 `matmul_values` 是 0.25 的整数倍，进 half 无损，误差恒为 `0.00e0`，
    // 于是这条用例只能验覆盖、验不了舍入。换成必然要舍入的取值，报告出来的数字才有含义。
    let (a, b) = full_mantissa_values(M, N, K);
    let mut stage = tensor_stage_from(&device, &queue, (M, N, K), (a.clone(), b.clone()), false);

    let geometry = crate::tiled::tensor_geometry(M, N, K);
    let msl = program_sample(
        &catalogue,
        &dirs,
        "matmul_f32",
        Metal,
        Mode::tensor(geometry),
        &gemm_sample(M, N, K),
    )
    .and_then(|stages| stages.one("张量核"))
    .expect("张量核 MSL 应能生成");
    assert!(
        msl.contains("mpp::tensor_ops::matmul2d"),
        "Gemm 骨架应走张量核 lowering"
    );
    // `tensor_pipe` 内部即完成派发并断言成功，这里只需要它的副作用（输出已落在 `stage.y`）。
    let _pipe = tensor_pipe(&device, &queue, &stage, (M, N), geometry, &msl);

    let got = read_back(stage.y.as_mut_bytes());
    assert_eq!(got.len(), M * N, "输出元素数应与 M·N 一致");
    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for sample in 0..SAMPLES {
        // 互质步长散点，覆盖全部 tile 行/列（含 2048 与 4096 的边界附近）。
        let i = (sample * 7919) % M;
        let j = (sample * 104_729) % N;
        let mut want = 0.0f64;
        for p in 0..K {
            // 设备侧 A 与 B 都按 half 精度进入张量核，参考按同一精度舍入后再在 f64 里累加。
            want +=
                f64::from(half_precision(a[i * K + p])) * f64::from(half_precision(b[p * N + j]));
        }
        let value = got[i * N + j];
        assert!(
            value.is_finite(),
            "抽样元素 ({i},{j}) 不是有限值：张量核输出有 NaN/inf"
        );
        let error = (f64::from(value) - want).abs() / want.abs().max(1.0);
        if error > worst {
            worst = error;
            worst_at = (i, j);
        }
    }
    assert!(
        worst < TENSOR_TOLERANCE,
        "大形状抽样误差 {worst:.2e}（元素 {worst_at:?}）超过 {TENSOR_TOLERANCE:.0e}"
    );
    println!(
        "张量核 {M}×{N}×{K}（M·N·K={:.1e}）：{SAMPLES} 个抽样点最差相对误差 {worst:.2e} @ {worst_at:?}",
        (M * N * K) as f64
    );
}
