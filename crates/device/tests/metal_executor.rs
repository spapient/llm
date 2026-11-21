use super::harness::*;
use super::*;
use llm_optim::backend::TensorGeometry;
use llm_optim::execution::Backend::Metal;
use llm_optim::execution::{Mode, program};

/// 设备 → 命令队列 的进程内对象模型打通：能建、能释放，反复建放不崩溃、不重复释放。
#[test]
fn device_and_queue_round_trip_without_crashing() {
    for _ in 0..2 {
        let Some(device) = MetalDevice::system_default() else {
            // 无 Metal 的本机（如 CI Linux）跳过而非失败：本测试只证明"有设备时路径成立"。
            return;
        };
        let queue = device.new_command_queue();
        assert!(queue.is_some(), "有设备必有命令队列");
    }
}

/// 共享缓冲分配 + CPU 读写回读：写进去的字节原样读回，证明设备内存与 CPU 视图是同一块内存。
#[test]
fn shared_buffer_round_trip_reads_written_bytes() {
    let Some(device) = MetalDevice::system_default() else {
        return;
    };
    let Some(mut buf) = device.new_buffer(16) else {
        panic!("分配共享缓冲失败");
    };
    let expect: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    buf.as_mut_bytes().copy_from_slice(&expect);
    assert_eq!(buf.as_mut_bytes(), &expect);
}

/// 一段合法 MSL 源码在进程内编译出计算管线：不依赖 xcrun 子进程，直接 `newLibraryWithSource`。
#[test]
fn compiles_a_trivial_kernel_in_process() {
    let Some(device) = MetalDevice::system_default() else {
        return;
    };
    let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = 42.0f;
}
"#;
    let pipeline = device.compile(source);
    assert!(pipeline.is_some(), "合法 MSL 应能编译出管线");
}

/// 派发一个写常量的内核并回读：4 个线程各写 `out[id] = 42.0`，`waitUntilCompleted` 后 CPU 读到 4 个 42.0。
#[test]
fn dispatches_a_kernel_and_reads_back() {
    let Some(device) = MetalDevice::system_default() else {
        return;
    };
    let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = 42.0f;
}
"#;
    let Some(pipeline) = device.compile(source) else {
        panic!("合法 MSL 应能编译出管线");
    };
    let Some(queue) = device.new_command_queue() else {
        panic!("有设备必有命令队列");
    };
    // 4 个 f32 = 16 字节。
    let mut out = device.new_buffer(16).expect("分配输出缓冲失败");
    assert!(
        queue.execute(&pipeline, &[&out], [4, 1, 1], [1, 1, 1]),
        "派发应成功"
    );

    let bytes = out.as_mut_bytes();
    assert_eq!(bytes.len(), 16);
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(got, vec![42.0f32; 4], "GPU 应把 42.0 写进每个元素");
}

/// 执行器把一张图的节点派发到 Metal 并回读：注册 `kernel_main` 双倍内核，绑定输入/输出，
/// `submit` → `sync` 后输出 = 输入的 2 倍，证明「提交 → 派发 → 同步」整条 [`Executor`] 契约在真机成立。
#[test]
fn executor_runs_a_registered_kernel_through_the_graph() {
    use engine::exec::Launch;

    let Some(mut exec) = MetalExecutor::new("metal", "apple") else {
        return;
    };
    exec.register(
        "double",
        r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f;
}
"#,
    )
    .expect("登记内核失败");
    exec.bind_bytes(0, 16).expect("绑定输入缓冲失败");
    exec.bind_bytes(1, 16).expect("绑定输出缓冲失败");

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_ne_bytes()).collect();
    exec.buffer_mut(0)
        .expect("输入缓冲应已绑定")
        .as_mut_bytes()
        .copy_from_slice(&input_bytes);

    let mut graph = Graph::new();
    let mut node = Node::new("double", Launch::new([4, 1, 1], [1, 1, 1]));
    node.bind_input(0).bind_output(1);
    graph.push(node);

    exec.submit(&graph, Stream::new(0)).expect("提交失败");
    exec.sync(Stream::new(0)).expect("同步失败");

    let (chunks, _) = exec
        .buffer_mut(1)
        .expect("输出缓冲应已绑定")
        .as_mut_bytes()
        .as_chunks::<4>();
    let output: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
    assert_eq!(output, vec![2.0f32, 4.0, 6.0, 8.0], "GPU 应把输入翻倍写回");
}

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

/// 长序列 flash attention：S=2048、D=64，验证 K/V 分块流式扫描不保留完整分数向量，
/// 与「整行分数留在 threadgroup 内存」的分块路径对拍并报延迟。
///
/// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_flash_attention_long --release --ignored --nocapture`。
#[test]
#[ignore = "Metal 长序列 flash attention：cargo test -p llm-engine --lib -- metal_flash_attention_long --release --ignored --nocapture"]
fn metal_flash_attention_long() {
    let Some((device, queue)) = bench_device("长序列 flash attention 基准") else {
        return;
    };
    crate::backends::install();

    const S: usize = 2048;
    const D: usize = 64;
    const THREADS: usize = 64;

    let AttentionBench(q_buf, k_buf, v_buf, mut y_buf, reference) = attention_bench(&device, S, D);

    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-flash-long");
    let grid_pipe = grid_pipeline(
        &device,
        &catalogue,
        &dirs,
        "attention_f32",
        S,
        D,
        "网格注意力",
    );
    let launch = [(S * THREADS) as u32, 1, 1];
    let block = [THREADS as u32, 1, 1];
    queue.execute(
        &grid_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        [S as u32, 1, 1],
        [1, 1, 1],
    );
    assert_matmul_matches("attn-long-grid", &reference, y_buf.as_mut_bytes());
    let grid_ns = measure(
        &queue,
        &grid_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        [S as u32, 1, 1],
        [1, 1, 1],
    );
    eprintln!("Metal 长序列 attention {S}×{D}：网格行线程={grid_ns:.1} ns");

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
        &tiled_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        launch,
        block,
    );
    assert_matmul_matches("attn-long-tiled", &reference, y_buf.as_mut_bytes());
    let tiled_ns = measure(
        &queue,
        &tiled_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        launch,
        block,
    );
    eprintln!(
        "  整行分数分块线程组={tiled_ns:.1} ns 对网格={:.2}×",
        grid_ns / tiled_ns
    );

    for key_block in [64usize, 128, 256] {
        let msl = program(
            &catalogue,
            &dirs,
            "attention_f32",
            Metal,
            Mode::flash(THREADS, key_block),
            S,
            D,
        )
        .and_then(|stages| stages.one("flash"))
        .expect("flash attention MSL 应能生成");
        let pipe = device.compile(&msl).expect("flash attention MSL 应能编译");
        queue.execute(&pipe, &[&q_buf, &k_buf, &v_buf, &y_buf], launch, block);
        assert_matmul_matches("attn-long-flash", &reference, y_buf.as_mut_bytes());
        let ns = measure(
            &queue,
            &pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            launch,
            block,
        );
        eprintln!(
            "  flash block={key_block}：{ns:.1} ns 对网格={:.2}× 对分块={:.2}×",
            grid_ns / ns,
            tiled_ns / ns,
        );
    }
}

/// 注意力 CPU 参考：分数 f64 累加、行内 max/exp 归一化、输出 f64 累加。
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
    use op::model::Sample;
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
        let sample = Sample {
            m,
            n,
            scalars: vec![("M", m as i64), ("N", n as i64), ("K", k as i64)],
            buffers: Vec::new(),
        };
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
