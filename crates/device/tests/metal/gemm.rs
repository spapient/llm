//! 职责：f32 与 int8 GEMM 的真机吞吐与对拍基准。

use super::*;

/// 基准的共同开工：系统默认设备 + 命令队列。
///
/// 无 Metal 的本机打印 `hint` 的跳过原因并返回 `None`，用例据此走
/// `let Some(..) = .. else { return; }` 静默跳过；跳过路径不 panic。
/// simdgroup 矩阵指令探测：编译并派发 8×8×8 矩阵乘（`simdgroup_float8x8` /
/// `simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`），与 CPU 参考对拍。
/// 探测失败只报原因，后续路线退回寄存器分块。
#[test]
#[ignore = "Metal simdgroup 探测：cargo test -p llm-engine --lib -- metal_simdgroup --release --ignored --nocapture"]
fn metal_simdgroup_probe() {
    let Some((device, queue)) = bench_device("simdgroup 探测") else {
        return;
    };
    let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device const float* A [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float* C [[buffer(2)]],
                        uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float zeros[64];
    for (uint i = lid.x; i < 64; i += 32) zeros[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 a, b, acc;
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_load(acc, zeros, 8);
    simdgroup_multiply_accumulate(acc, a, b, acc);
    simdgroup_store(acc, C, 8);
}
"#;
    let Some(pipeline) = device.compile(source) else {
        eprintln!("simdgroup 探测：MSL 编译失败，本机走寄存器分块路线");
        return;
    };
    let (a, b, reference) = matmul_inputs(8, 8, 8);
    let a_buf = upload(&device, &a, "A");
    let b_buf = upload(&device, &b, "B");
    let mut c_buf = output_buffer(&device, 64, "C");
    let ok = queue.execute(&pipeline, &[&a_buf, &b_buf, &c_buf], [32, 1, 1], [32, 1, 1]);
    assert!(ok, "simdgroup 探测派发失败");
    assert_matmul_matches("simdgroup-8x8", &reference, c_buf.as_mut_bytes());
    eprintln!("simdgroup 探测通过：8×8×8 矩阵乘 32 线程单线程组，输出与 CPU 参考一致");
}

/// Metal 真机 matmul 前向基准：同一份 `M×N×K` 稠密 f32 在两条路径上跑同一算术，
/// 对拍 GPU 与 CPU 参考、并报 naive（每线程一输出）与 tiled（threadgroup 共享内存分块）
/// 的真实端到端延迟与 GFLOP/s。
///
/// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_matmul_forward --release --ignored --nocapture`。
/// 无 Metal 设备时静默跳过；产品级 E1/E3（7B / 100K / NVIDIA A800·3090·T600）不在本机、保持 OPEN。
#[test]
#[ignore = "Metal matmul 前向基准：cargo test -p llm-engine --lib -- metal_matmul_forward --release --ignored --nocapture"]
fn metal_matmul_forward() {
    let Some((device, queue)) = bench_device("matmul 前向基准") else {
        return;
    };

    const M: usize = 512;
    const N: usize = 512;
    const K: usize = 512;

    let (a, b, reference) = matmul_inputs(M, N, K);
    let a_buf = upload(&device, &a, "A");
    let b_buf = upload(&device, &b, "B");

    let naive = device
        .compile(&matmul_msl(false, M, N, K))
        .expect("naive matmul 应能编译");
    let tiled = device
        .compile(&matmul_msl(true, M, N, K))
        .expect("tiled matmul 应能编译");

    let mut c_buf = output_buffer(&device, M * N, "C");
    let grid = [N as u32, M as u32, 1];

    // 先各跑一次对拍，再进入计时轮。
    queue.execute(&naive, &[&a_buf, &b_buf, &c_buf], grid, [1, 1, 1]);
    assert_matmul_matches("naive", &reference, c_buf.as_mut_bytes());
    queue.execute(&tiled, &[&a_buf, &b_buf, &c_buf], grid, [16, 16, 1]);
    assert_matmul_matches("tiled", &reference, c_buf.as_mut_bytes());

    let flop = 2.0 * (M * N * K) as f64;
    let naive_ns = measure(&queue, &naive, &[&a_buf, &b_buf, &c_buf], grid, [1, 1, 1]);
    let tiled_ns = measure(&queue, &tiled, &[&a_buf, &b_buf, &c_buf], grid, [16, 16, 1]);

    eprintln!(
        "Metal matmul 512×512×512：naive={naive_ns:.1} ns（{:.2} GFLOP/s） tiled={tiled_ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
        flop / naive_ns,
        flop / tiled_ns,
        naive_ns / tiled_ns,
    );
}

/// 生产态分块 GEMM 对照：同一份 `matmul_f32`（K=4096）分别经
/// [`program`]（标量网格基线）与
/// `Mode::tiled`（共享内存分块）生成 MSL，在真机 GPU 上跑
/// 同一组输入、对拍 CPU 参考并报延迟与 GFLOP/s。
///
/// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_tiled_gemm_k4096 --release --ignored --nocapture`。
/// 无 Metal 设备时静默跳过；产品级 E1/E3 不在本机、保持 OPEN。
#[test]
#[ignore = "Metal 生产态分块 GEMM：cargo test -p llm-engine --lib -- metal_tiled_gemm_k4096 --release --ignored --nocapture"]
fn metal_tiled_gemm_k4096() {
    let Some((device, queue)) = bench_device("分块 GEMM 基准") else {
        return;
    };
    crate::backends::install();

    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 4096;

    let (a, b, reference) = matmul_inputs(M, N, K);
    let a_buf = upload(&device, &a, "A");
    let b_buf = upload(&device, &b, "B");
    let mut y_buf = output_buffer(&device, M * N, "Y");

    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-matmul-k4096");
    let grid_pipe = grid_pipeline(&device, &catalogue, &dirs, "matmul_f32", M, N, "标量网格");

    let grid = [M as u32, N as u32, 1];
    queue.execute(&grid_pipe, &[&a_buf, &b_buf, &y_buf], grid, [1, 1, 1]);
    assert_matmul_matches("grid", &reference, y_buf.as_mut_bytes());

    let flop = 2.0 * (M * N * K) as f64;
    let grid_ns = measure(
        &queue,
        &grid_pipe,
        &[&a_buf, &b_buf, &y_buf],
        grid,
        [1, 1, 1],
    );
    eprintln!(
        "Metal 生产态 GEMM 256×256×4096：网格标量={grid_ns:.1} ns（{:.2} GFLOP/s）",
        flop / grid_ns,
    );
    for (tile, per_thread) in [(8usize, 1usize), (16, 1), (32, 1), (32, 4)] {
        let msl = program(
            &catalogue,
            &dirs,
            "matmul_f32",
            Metal,
            Mode::tiled(tile, per_thread),
            M,
            N,
        )
        .and_then(|stages| stages.one("分块"))
        .expect("分块 MSL 应能生成");
        let pipe = device.compile(&msl).expect("分块 MSL 应能编译");
        let threads = tile / per_thread;
        let (launch_grid, block) = if per_thread == 1 {
            (grid, [tile as u32, tile as u32, 1])
        } else {
            (
                [(M / per_thread) as u32, (N / per_thread) as u32, 1],
                [threads as u32, threads as u32, 1],
            )
        };
        queue.execute(&pipe, &[&a_buf, &b_buf, &y_buf], launch_grid, block);
        assert_matmul_matches("tiled", &reference, y_buf.as_mut_bytes());
        let ns = measure(&queue, &pipe, &[&a_buf, &b_buf, &y_buf], launch_grid, block);
        eprintln!(
            "  分块 tile={tile} pt={per_thread}：{ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
            flop / ns,
            grid_ns / ns,
        );
    }
    for (row_frags, col_frags) in [(1usize, 1usize), (2, 2), (4, 4)] {
        let (pipe, simd_grid) =
            simdgroup_gemm_pipeline(&device, &catalogue, &dirs, M, N, row_frags, col_frags);
        let block = [32, 1, 1];
        queue.execute(&pipe, &[&a_buf, &b_buf, &y_buf], simd_grid, block);
        assert_matmul_matches("simdgroup", &reference, y_buf.as_mut_bytes());
        let ns = measure(&queue, &pipe, &[&a_buf, &b_buf, &y_buf], simd_grid, block);
        eprintln!(
            "  simdgroup {row_frags}×{col_frags} 片段：{ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
            flop / ns,
            grid_ns / ns,
        );
    }
}

/// simdgroup GEMM 放大形状基准：M=N=512、K=4096，验证 8×8 矩阵指令在更大的输出块上
/// 仍与 CPU 参考一致，并报端到端 GFLOP/s（256×256 的小形状受启动开销影响大）。
///
/// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_simdgroup_gemm_512 --release --ignored --nocapture`。
#[test]
#[ignore = "Metal simdgroup GEMM 512×512×4096：cargo test -p llm-engine --lib -- metal_simdgroup_gemm_512 --release --ignored --nocapture"]
fn metal_simdgroup_gemm_512() {
    let Some((device, queue)) = bench_device("simdgroup GEMM 基准") else {
        return;
    };
    crate::backends::install();

    const M: usize = 512;
    const N: usize = 512;
    const K: usize = 4096;

    let (a, b, reference) = matmul_inputs(M, N, K);
    let a_buf = upload(&device, &a, "A");
    let b_buf = upload(&device, &b, "B");
    let mut y_buf = output_buffer(&device, M * N, "Y");

    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-simd-512");
    let grid_pipe = grid_pipeline(&device, &catalogue, &dirs, "matmul_f32", M, N, "标量网格");
    let grid = [M as u32, N as u32, 1];
    queue.execute(&grid_pipe, &[&a_buf, &b_buf, &y_buf], grid, [1, 1, 1]);
    assert_matmul_matches("grid-512", &reference, y_buf.as_mut_bytes());
    let flop = 2.0 * (M * N * K) as f64;
    let grid_ns = measure(
        &queue,
        &grid_pipe,
        &[&a_buf, &b_buf, &y_buf],
        grid,
        [1, 1, 1],
    );
    eprintln!(
        "Metal simdgroup GEMM 512×512×4096：网格标量={grid_ns:.1} ns（{:.2} GFLOP/s）",
        flop / grid_ns,
    );

    for (row_frags, col_frags) in [(1usize, 1usize), (4, 4)] {
        let (pipe, simd_grid) =
            simdgroup_gemm_pipeline(&device, &catalogue, &dirs, M, N, row_frags, col_frags);
        let block = [32, 1, 1];
        queue.execute(&pipe, &[&a_buf, &b_buf, &y_buf], simd_grid, block);
        assert_matmul_matches("simdgroup-512", &reference, y_buf.as_mut_bytes());
        let ns = measure(&queue, &pipe, &[&a_buf, &b_buf, &y_buf], simd_grid, block);
        eprintln!(
            "  simdgroup {row_frags}×{col_frags} 片段：{ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
            flop / ns,
            grid_ns / ns,
        );
    }
}

/// int8 分块 GEMM 真机基准：64×256×4096，`i8` 输入、`i32` 精确累加、`f32` 输出。
///
/// 同一输入先跑 CPU 参考（`op::matmul_i8::forward_tiled`）逐元素对拍，再对 tile=8/16/32
/// 报端到端延迟；这是 `matmul_i8` 的设备侧落地，不是 f32 GEMM 的近似。
///
/// 本用例只出测量：`cargo test --release -p llm-engine --lib -- metal_int8_gemm_k4096 --ignored --nocapture`。
#[test]
#[ignore = "Metal int8 GEMM：cargo test --release -p llm-engine --lib -- metal_int8_gemm_k4096 --ignored --nocapture"]
fn metal_int8_gemm_k4096() {
    let Some((device, queue)) = bench_device("int8 GEMM 基准") else {
        return;
    };

    let mut bench = int8_gemm_bench(&device);
    let (m, n, k) = (bench.m, bench.n, bench.k);

    for tile in [8usize, 16, 32] {
        let msl =
            crate::tiled::metal_int8_gemm_program(m, n, k, tile).expect("int8 GEMM MSL 应能生成");
        let pipe = device.compile(&msl).expect("int8 GEMM MSL 应能编译");
        let block = [tile as u32, tile as u32, 1];
        queue.execute(&pipe, &[&bench.a, &bench.b, &bench.y], bench.grid, block);
        assert_matmul_matches("int8", &bench.reference, bench.y.as_mut_bytes());
        let ns = measure(
            &queue,
            &pipe,
            &[&bench.a, &bench.b, &bench.y],
            bench.grid,
            block,
        );
        eprintln!(
            "Metal int8 GEMM 64×256×4096 tile={tile}：{ns:.1} ns（{:.2} G int-op/s）",
            (m * n * k) as f64 / ns
        );
    }
}

/// 端到端 int8 GEMM：cutile 生产入口（`resolve` 档案协商 → `Lower::Int8Gemm` → engine lowering）
/// 生成 MSL，在真机 GPU 上与 CPU i32 参考逐元素对拍并报延迟。
///
/// 同时验证两条回落契约：非 int8 签名、`K` 不被 `tile` 整除时都必须回到网格/串行路径，
/// 不允许把「没走 int8 分块」冒充成已落地。
///
/// 本用例只出测量：`cargo test --release -p llm-engine --lib -- metal_int8_gemm_cutile_k4096 --ignored --nocapture`。
#[test]
#[ignore = "Metal int8 GEMM cutile 端到端：cargo test --release -p llm-engine --lib -- metal_int8_gemm_cutile_k4096 --ignored --nocapture"]
fn metal_int8_gemm_cutile_k4096() {
    let Some((device, queue)) = bench_device("int8 GEMM cutile 端到端") else {
        return;
    };
    crate::backends::install();

    let mut bench = int8_gemm_bench(&device);
    let (m, n, k) = (bench.m, bench.n, bench.k);
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-int8-cutile");
    let fallback = program(&catalogue, &dirs, "matmul_f32", Metal, Mode::int8(16), m, n)
        .and_then(|stages| stages.one("int8"))
        .expect("非 int8 算子应回落到网格/串行路径");
    assert!(
        !fallback.contains("threadgroup int As"),
        "非 int8 签名不得走 int8 分块 lowering"
    );
    let ragged = program(&catalogue, &dirs, "matmul_i8", Metal, Mode::int8(24), m, n)
        .and_then(|stages| stages.one("int8"))
        .expect("K 不整除时应回落到网格/串行路径");
    assert!(
        !ragged.contains("threadgroup int As"),
        "K 不被 tile 整除时不得走 int8 分块 lowering"
    );

    for tile in [8usize, 16, 32] {
        let msl = program(
            &catalogue,
            &dirs,
            "matmul_i8",
            Metal,
            Mode::int8(tile),
            m,
            n,
        )
        .and_then(|stages| stages.one("int8"))
        .unwrap_or_else(|error| panic!("tile={tile} 的 MSL 应能生成：{error}"));
        assert!(
            msl.contains("threadgroup int As"),
            "tile={tile} 应走 int8 分块 lowering"
        );
        let pipe = device
            .compile(&msl)
            .unwrap_or_else(|| panic!("tile={tile} 的 MSL 应能编译"));
        let block = [tile as u32, tile as u32, 1];
        queue.execute(&pipe, &[&bench.a, &bench.b, &bench.y], bench.grid, block);
        assert_matmul_matches("int8-cutile", &bench.reference, bench.y.as_mut_bytes());
        let ns = measure(
            &queue,
            &pipe,
            &[&bench.a, &bench.b, &bench.y],
            bench.grid,
            block,
        );
        eprintln!(
            "cutile→Metal int8 GEMM 64×256×4096 tile={tile}：{ns:.1} ns（{:.2} G int-op/s）",
            (m * n * k) as f64 / ns
        );
    }
}

/// 生产态分块注意力对照：同一份 `attention_f32`（S=512、D=64）分别经
/// [`program`]（每查询行一线程）与
/// `Mode::tiled_attention`（每行一个 threadgroup）生成 MSL，
/// 在真机 GPU 上跑同一组 Q/K/V、对拍 CPU 参考并报延迟与吞吐。
///
/// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_tiled_attention_s512 --release --ignored --nocapture`。
#[test]
#[ignore = "Metal 生产态分块注意力：cargo test -p llm-engine --lib -- metal_tiled_attention_s512 --release --ignored --nocapture"]
fn metal_tiled_attention_s512() {
    let Some((device, queue)) = bench_device("分块注意力基准") else {
        return;
    };
    crate::backends::install();

    const S: usize = 512;
    const D: usize = 64;
    const THREADS: usize = 64;

    let AttentionBench(q_buf, k_buf, v_buf, mut y_buf, reference) = attention_bench(&device, S, D);

    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-attn-s512");
    let grid_pipe = grid_pipeline(
        &device,
        &catalogue,
        &dirs,
        "attention_f32",
        S,
        D,
        "网格注意力",
    );
    let tiled_msl = program(
        &catalogue,
        &dirs,
        "attention_f32",
        Metal,
        Mode::tiled_attention(THREADS),
        S,
        D,
    )
    .and_then(|stages| stages.one("分块注意力"))
    .expect("分块注意力 MSL 应能生成");
    let tiled_pipe = device.compile(&tiled_msl).expect("分块注意力 MSL 应能编译");

    queue.execute(
        &grid_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        [S as u32, 1, 1],
        [1, 1, 1],
    );
    assert_matmul_matches("attn-grid", &reference, y_buf.as_mut_bytes());
    let tiled_grid = [(S * THREADS) as u32, 1, 1];
    queue.execute(
        &tiled_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        tiled_grid,
        [THREADS as u32, 1, 1],
    );
    assert_matmul_matches("attn-tiled", &reference, y_buf.as_mut_bytes());

    let grid_ns = measure(
        &queue,
        &grid_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        [S as u32, 1, 1],
        [1, 1, 1],
    );
    let tiled_ns = measure(
        &queue,
        &tiled_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        tiled_grid,
        [THREADS as u32, 1, 1],
    );
    eprintln!(
        "Metal 生产态 attention 512×64：网格行线程={grid_ns:.1} ns 分块线程组={tiled_ns:.1} ns（每行 {THREADS} 线程） 加速={:.2}×",
        grid_ns / tiled_ns,
    );
    for block in [512usize, 64, 128, 256] {
        let flash_msl = program(
            &catalogue,
            &dirs,
            "attention_f32",
            Metal,
            Mode::flash(THREADS, block),
            S,
            D,
        )
        .and_then(|stages| stages.one("flash"))
        .expect("flash attention MSL 应能生成");
        let flash_pipe = device
            .compile(&flash_msl)
            .expect("flash attention MSL 应能编译");
        queue.execute(
            &flash_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            tiled_grid,
            [THREADS as u32, 1, 1],
        );
        assert_matmul_matches("attn-flash", &reference, y_buf.as_mut_bytes());
        let flash_ns = measure(
            &queue,
            &flash_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            tiled_grid,
            [THREADS as u32, 1, 1],
        );
        eprintln!(
            "  flash block={block}：{flash_ns:.1} ns（每行 {THREADS} 线程） 对网格={:.2}× 对分块={:.2}×",
            grid_ns / flash_ns,
            tiled_ns / flash_ns,
        );
    }
}
