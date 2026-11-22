//! 职责：f32 GEMM 的前向误差界、残尾回落、写满与可复现判据。

use super::*;

/// f32 GEMM 的逐元素前向误差界：96×80×96，输入满尾数、非退化，标量网格与 threadgroup 分块
/// 两条路径都必须落在 `γ_{2k}·Σ|a·b|` 内。
///
/// 为什么不用夹具的 [`matmul_inputs`]：那套取值是 `0.25`/`0.5` 的整数倍，乘积在 f32 里**精确**，
/// 最佳累加顺序下误差恒为 `0.00e0`，累加顺序错了也测不出来。这里换成满尾数输入，判据也从
/// 定值阈值换成可推导的误差界。
#[test]
fn gemm_stays_within_forward_error_bound_on_full_mantissa_inputs() {
    const M: usize = 96;
    const N: usize = 80;
    const K: usize = 96;

    let Some((device, queue)) = bench_device("前向误差界 GEMM") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-bound");

    let (a, b, a_buf, b_buf, mut y) = gemm_fixture(&device, (M, N, K), 0x5eed_1234_abcd_0001);
    let grid = [M as u32, N as u32, 1];

    // 路径一：标量网格（每线程算一个输出元素）。
    let sample = gemm_sample(M, N, K);
    let grid_pipe = grid_pipeline_sample(
        &device,
        &catalogue,
        &dirs,
        "matmul_f32",
        &sample,
        "标量网格",
    );
    assert!(
        queue.execute(&grid_pipe, &[&a_buf, &b_buf, &y], grid, [1, 1, 1]),
        "标量网格派发应成功"
    );
    assert_within_bound(
        "标量网格 96×80×96",
        &read_back(y.as_mut_bytes()),
        &a,
        &b,
        M,
        N,
        K,
    );

    // 路径二：threadgroup 分块（16×16，M/N/K 都是 16 的整数倍，前置条件成立）。
    let tiled = program_sample(
        &catalogue,
        &dirs,
        "matmul_f32",
        Metal,
        Mode::tiled(16, 1),
        &sample,
    )
    .and_then(|stages| stages.one("分块"))
    .expect("分块 MSL 应能生成");
    assert!(
        tiled.contains("threadgroup float As["),
        "96/80/96 都是 16 的倍数，应走分块 lowering"
    );
    let tiled_pipe = device.compile(&tiled).expect("分块 MSL 应能编译");
    assert!(
        queue.execute(&tiled_pipe, &[&a_buf, &b_buf, &y], grid, [16, 16, 1]),
        "分块派发应成功"
    );
    assert_within_bound(
        "分块 16×16 96×80×96",
        &read_back(y.as_mut_bytes()),
        &a,
        &b,
        M,
        N,
        K,
    );
}

/// 边界形状：`1×1×1`、`3×5×7`、`17×33×65`、`129×7×3` 都要与 f64 参考落在误差界内；
/// 同时验证 `Mode::tiled` 在不整除时**必须回落**，不能算出一半分块就交付。
///
/// `65 = 4·16 + 1`、`33 = 2·16 + 1` 专打尾块：分块内核最容易错的地方是最后一块不满时的守卫，
/// 而整除形状（`64`、`80`）永远走不到那条分支。
#[test]
fn gemm_handles_ragged_tail_shapes_and_refuses_to_pretend_it_tiled() {
    let Some((device, queue)) = bench_device("边界形状 GEMM") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-ragged");

    for (m, n, k) in [(1, 1, 1), (3, 5, 7), (17, 33, 65), (129, 7, 3)] {
        let seed = 0x5eed_0000_0000_0002 ^ ((m * 1000 + n * 10 + k) as u64);
        let (a, b, a_buf, b_buf, mut y) = gemm_fixture(&device, (m, n, k), seed);
        let label = format!("{m}×{n}×{k}");

        let sample = gemm_sample(m, n, k);
        let grid_pipe = grid_pipeline_sample(
            &device,
            &catalogue,
            &dirs,
            "matmul_f32",
            &sample,
            "标量网格",
        );
        assert!(
            queue.execute(
                &grid_pipe,
                &[&a_buf, &b_buf, &y],
                [m as u32, n as u32, 1],
                [1, 1, 1]
            ),
            "{label} 标量网格派发应成功"
        );
        assert_within_bound(&label, &read_back(y.as_mut_bytes()), &a, &b, m, n, k);

        // 不整除时 `Mode::tiled` 的前置条件不成立：必须回落到标量路径。
        let fallback = program_sample(
            &catalogue,
            &dirs,
            "matmul_f32",
            Metal,
            Mode::tiled(16, 1),
            &sample,
        )
        .and_then(|stages| stages.one("分块"))
        .expect("不整除时应能回落到标量路径");
        assert!(
            !fallback.contains("threadgroup float As["),
            "{label} 不被 16 整除，不得走分块 lowering"
        );
    }
}

/// 写满 + 可复现：输出缓冲先填 NaN，派发后**每个元素都必须变成有限值**，且同一输入重复派发要
/// 逐位相同（含张量核双缓冲路径）。
///
/// 这条用例抓的是两类只在特定初值下露头的缺陷：漏写（尾块/越界守卫写少了）、竞争（共享内存
/// 少一次 `threadgroup_barrier`，或 ping-pong 两片被复用得过早）。二者在「输出缓冲本就是 0」
/// 的夹具里都不会显形。
#[test]
fn gemm_writes_every_output_and_repeats_bit_exactly() {
    const M: usize = 96;
    const N: usize = 80;
    const K: usize = 96;

    let Some((device, queue)) = bench_device("写满与可复现") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-repeat");

    let (a, b, a_buf, b_buf, mut y) = gemm_fixture(&device, (M, N, K), 0x5eed_0000_0000_0003);
    let grid = [M as u32, N as u32, 1];
    let block = [16, 16, 1];

    let sample = gemm_sample(M, N, K);
    let tiled = program_sample(
        &catalogue,
        &dirs,
        "matmul_f32",
        Metal,
        Mode::tiled(16, 1),
        &sample,
    )
    .and_then(|stages| stages.one("分块"))
    .expect("分块 MSL 应能生成");
    let tiled_pipe = device.compile(&tiled).expect("分块 MSL 应能编译");

    fill_nan(&mut y);
    assert!(
        queue.execute(&tiled_pipe, &[&a_buf, &b_buf, &y], grid, block),
        "分块派发应成功"
    );
    let first = snapshot(&mut y);
    assert!(
        read_back(&first).iter().all(|value| value.is_finite()),
        "分块内核必须写满每个输出元素，NaN 残留即漏写"
    );
    assert_within_bound("分块写满 96×80×96", &read_back(&first), &a, &b, M, N, K);

    // 再填一遍 NaN 再派发：初值不影响结果，两次必须逐位相同。
    fill_nan(&mut y);
    assert!(
        queue.execute(&tiled_pipe, &[&a_buf, &b_buf, &y], grid, block),
        "分块重复派发应成功"
    );
    assert_eq!(first, snapshot(&mut y), "同一输入的两次派发应逐位相同");

    // 张量核双缓冲：ping-pong 两片缓冲与「一次 barrier」的软件流水最容易出竞争。
    use nn::sig::Signature;

    let geometry = TensorGeometry::new(128, 64, 96, 8);
    let mut stage = tensor_stage(&device, &queue, (256, 256, 1024), true);
    let signature = Signature::parse(
        "bench",
        "void cutile_bench_kernel(const float *a, const float *b, float *y, int M, int N, int K) {}",
    )
    .expect("自建签名应能解析");
    let msl = crate::tiled::metal_tensor_gemm(
        "bench",
        &signature,
        &tensor_sample(256, 256, 1024),
        geometry,
    )
    .expect("张量核 MSL 应能生成");
    assert!(
        msl.contains("threadgroup half sb[2]"),
        "两片 B 片装得下 32 KiB 时应走双缓冲"
    );
    let pipe = tensor_pipe(&device, &queue, &stage, (256, 256), geometry, &msl);

    fill_nan(&mut stage.y);
    assert!(
        queue.execute(
            &pipe.pipe,
            &[&stage.a, &stage.bt, &stage.y],
            pipe.grid,
            pipe.block
        ),
        "双缓冲派发应成功"
    );
    let first = snapshot(&mut stage.y);
    assert!(
        read_back(&first).iter().all(|value| value.is_finite()),
        "双缓冲内核必须写满每个输出元素，NaN 残留即漏写"
    );
    assert_eq!(
        first,
        snapshot(&mut stage.y),
        "双缓冲的两次派发应逐位相同（ping-pong 复用过早会读出半成品）"
    );
    let error = stage.error("双缓冲写满 256×256×1024");
    assert!(
        error < TENSOR_TOLERANCE,
        "双缓冲的误差 {error:.2e} 超过 {TENSOR_TOLERANCE:.0e}"
    );
}
