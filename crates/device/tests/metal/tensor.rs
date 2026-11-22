//! 职责：张量核路径的 GEMM/双缓冲吞吐基准。

use super::*;

/// 张量核 GEMM 打通验证 + 吞吐测量：先跑一次 B 的布局前置（`K×N` f32 → `N×K` half），
/// 再对主内核连跑 3 轮取中位数。
///
/// 计时口径与 `llm.u` 的 `measure_kernel` 一致——区间里只有 GEMM 派发本身，不含上传、下载和
/// 布局前置；`GF/s = 2·M·N·K / 中位延迟`。四档形状对标 `llm.u/docs/perf.md` 的
/// `1024³ ≈ 5.2–8.5`、`2048³ ≈ 18.3`、`4096³ ≈ 24.5` TF/s（同机 M5 Pro、20 核）。
///
/// 正确性只在 `M·N·K ≤ 1024³` 的形状上对拍：更大的 f64 参考要跑上分钟，而吞吐测点只需要输入。
///
/// 运行：`cargo test --release -p llm-engine --lib -- metal_tensor_gemm --ignored --nocapture`
#[test]
#[ignore = "Metal 张量核 GEMM：cargo test --release -p llm-engine --lib -- metal_tensor_gemm --ignored --nocapture"]
fn metal_tensor_gemm() {
    let Some((device, queue)) = bench_device("张量核 GEMM") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-tensor");

    // 形状只放开 M/N：`matmul_f32` 的规约轴由算子语义固定为 4096。
    let shapes = [
        (512, 512, 4096),
        (1024, 1024, 4096),
        (2048, 2048, 4096),
        (4096, 4096, 4096),
    ];
    for (m, n, k) in shapes {
        let mut stage = tensor_stage(&device, &queue, (m, n, k), m * n * k <= 1024 * 1024 * 1024);
        // 首项是 `tensor_geometry` 的自适应选型，另两档留作对照（大块几何 / 小方块几何）。
        let geometries = [
            crate::tiled::tensor_geometry(m, n, k),
            TensorGeometry::new(128, 128, 96, 8),
            TensorGeometry::new(64, 128, 64, 8),
        ];
        let mut pipes = Vec::new();
        for geometry in geometries {
            let msl = program(
                &catalogue,
                &dirs,
                "matmul_f32",
                Metal,
                Mode::tensor(geometry),
                m,
                n,
            )
            .and_then(|stages| stages.one("张量核"))
            .expect("张量核 MSL 应能生成");
            assert!(
                msl.contains("mpp::tensor_ops::matmul2d"),
                "Gemm 骨架应走张量核 lowering"
            );
            let pipe = tensor_pipe(&device, &queue, &stage, (m, n), geometry, &msl);
            let label = format!("张量核 {m}×{n}×{k} {}", pipe.label);
            println!("{label} 误差 {:.2e}", stage.error(&label));
            pipes.push(pipe);
        }
        sweep(
            &queue,
            &stage,
            &pipes,
            (m, n, k),
            &format!("张量核 {m}×{n}×{k}"),
        );
    }
}

/// 张量核 GEMM **内核**微基准：立方形状（K=N=M），与 llm.u `docs/perf.md` 的 8409 GF/s 同口径。
///
/// 生产算子 `matmul_f32` 把规约轴固定为 4096，立方形状无法由算子样本表达；这里自建签名与样本直接
/// 调内核生成器——测的是内核本身，不是算子装配。运行：
/// `cargo test --release -p llm-engine --lib -- metal_tensor_kernel_gemm --ignored --nocapture`
#[test]
#[ignore = "Metal 张量核立方 GEMM：cargo test --release -p llm-engine --lib -- metal_tensor_kernel_gemm --ignored --nocapture"]
fn metal_tensor_kernel_gemm() {
    use op::sig::Signature;

    let Some((device, queue)) = bench_device("张量核立方 GEMM") else {
        return;
    };
    let declaration = "void cutile_bench_kernel(const float *a, const float *b, float *y, int M, int N, int K) {}";
    let signature = Signature::parse("bench", declaration).expect("自建签名应能解析");
    for (m, n, k) in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)] {
        let mut stage = tensor_stage(&device, &queue, (m, n, k), m <= 1024);
        // 首项是 `tensor_geometry` 的自适应选型；另两档为对照（大块几何 / 与 llm.u 同构的档位）。
        let geometries = [
            crate::tiled::tensor_geometry(m, n, k),
            TensorGeometry::new(128, 128, 96, 8),
            TensorGeometry::new(64, 128, 96, 8),
        ];
        let sample = tensor_sample(m, n, k);
        let mut pipes = Vec::new();
        for geometry in geometries {
            let msl = crate::tiled::metal_tensor_gemm("bench", &signature, &sample, geometry)
                .expect("张量核 MSL 应能生成");
            let pipe = tensor_pipe(&device, &queue, &stage, (m, n), geometry, &msl);
            let label = format!("张量核立方 {m}³ {}", pipe.label);
            println!("{label} 误差 {:.2e}", stage.error(&label));
            pipes.push(pipe);
        }
        sweep(
            &queue,
            &stage,
            &pipes,
            (m, n, k),
            &format!("张量核立方 {m}³"),
        );
    }
}

/// 张量核**双缓冲**对拍：选一个两片 B 片装得进 32 KiB 的几何（软件流水生效），
/// 证明「预取下一片 + 乘加当前片重叠」的 ping-pong 缓冲与 CPU 参考一致。
///
/// `(128, 64, 96, 8)` 的单缓冲 B 片是 12.8 KiB、双缓冲 25.6 KiB，都塞得进 32 KiB，
/// 因此走双缓冲分支；`K=960` 被 96 整除走满块路径，`K=1024` 走尾块路径，两条都覆盖。
/// 运行：`cargo test --release -p llm-device --lib -- metal_tensor_double_buffer --ignored --nocapture`
#[test]
#[ignore = "Metal 张量核双缓冲对拍：cargo test --release -p llm-device --lib -- metal_tensor_double_buffer --ignored --nocapture"]
fn metal_tensor_double_buffer() {
    use op::model::Sample;
    use op::sig::Signature;

    let Some((device, queue)) = bench_device("张量核双缓冲对拍") else {
        return;
    };
    let declaration = "void cutile_bench_kernel(const float *a, const float *b, float *y, int M, int N, int K) {}";
    let signature = Signature::parse("bench", declaration).expect("自建签名应能解析");
    let geometry = TensorGeometry::new(128, 64, 96, 8);
    for (m, n, k) in [(256, 256, 960), (256, 256, 1024)] {
        let mut stage = tensor_stage(&device, &queue, (m, n, k), true);
        let sample = Sample {
            m,
            n,
            scalars: vec![("M", m as i64), ("N", n as i64), ("K", k as i64)],
            buffers: Vec::new(),
        };
        let msl = crate::tiled::metal_tensor_gemm("bench", &signature, &sample, geometry)
            .expect("张量核 MSL 应能生成");
        assert!(
            msl.contains("threadgroup half sb[2]"),
            "两片 B 片装得下时应走双缓冲，实际 MSL：\n{msl}"
        );
        let pipe = tensor_pipe(&device, &queue, &stage, (m, n), geometry, &msl);
        let label = format!("双缓冲对拍 {m}×{n}×{k} {}", pipe.label);
        let error = stage.error(&label);
        println!("{label} 误差 {error:.2e}");
    }
}

/// 张量核**双缓冲 vs 单缓冲**吞吐对照：同一立方形状下，把 tile_n=64 的几何（两片 B 塞得进
/// 32 KiB → 双缓冲软件流水）与 tile_n=128 的大块几何（只能单片）放到同一次重扫里取中位。
///
/// 双缓冲把每轮两次 barrier 减为一次并让 B 装载延迟与乘加重叠；对照用同 k_tile 的大块几何
/// 隔离「缓冲策略」与「块大小」两个变量。运行：
/// `cargo test --release -p llm-device --lib -- metal_tensor_double_buffer_sweep --ignored --nocapture`
#[test]
#[ignore = "Metal 张量核双缓冲吞吐：cargo test --release -p llm-device --lib -- metal_tensor_double_buffer_sweep --ignored --nocapture"]
fn metal_tensor_double_buffer_sweep() {
    use op::sig::Signature;

    let Some((device, queue)) = bench_device("张量核双缓冲吞吐") else {
        return;
    };
    let declaration = "void cutile_bench_kernel(const float *a, const float *b, float *y, int M, int N, int K) {}";
    let signature = Signature::parse("bench", declaration).expect("自建签名应能解析");
    for (m, n, k) in [(2048, 2048, 2048), (4096, 4096, 4096)] {
        let stage = tensor_stage(&device, &queue, (m, n, k), false);
        // tile_n=64 的双缓冲两档（k_tile 96/64）对比 tile_n=128 的同 k_tile 单缓冲大块，
        // 隔离「缓冲策略」与「块大小」两个变量；大块双缓冲（k_tile=48）实测更慢，不入测点。
        let geometries = [
            TensorGeometry::new(128, 64, 96, 8),
            TensorGeometry::new(128, 64, 64, 8),
            TensorGeometry::new(128, 128, 96, 8),
        ];
        let sample = tensor_sample(m, n, k);
        let mut pipes = Vec::new();
        for geometry in geometries {
            let msl = crate::tiled::metal_tensor_gemm("bench", &signature, &sample, geometry)
                .expect("张量核 MSL 应能生成");
            let double = msl.contains("threadgroup half sb[2]");
            let pipe = tensor_pipe(&device, &queue, &stage, (m, n), geometry, &msl);
            let label = format!(
                "双缓冲吞吐 {m}³ {} {}",
                pipe.label,
                if double { "双片" } else { "单片" }
            );
            println!("{label}");
            pipes.push(pipe);
        }
        sweep(
            &queue,
            &stage,
            &pipes,
            (m, n, k),
            &format!("双缓冲吞吐 {m}³"),
        );
    }
}
