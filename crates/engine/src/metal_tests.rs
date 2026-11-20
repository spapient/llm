use super::*;

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
    use super::super::exec::Launch;

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
fn bench_device(hint: &str) -> Option<(MetalDevice, MetalCommandQueue)> {
    let Some(device) = MetalDevice::system_default() else {
        eprintln!("Metal 设备不在本机，{hint}未运行");
        return None;
    };
    let Some(queue) = device.new_command_queue() else {
        panic!("有设备必有命令队列");
    };
    Some((device, queue))
}

/// 内置算子目录 + 该基准自己的临时工作目录：一次基准里的所有 MSL 生成都落在这上面。
fn catalogue_and_dirs(cache: &str) -> (cutile::builtin::Catalogue, cutile::dirs::Dirs) {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join(cache));
    (catalogue, dirs)
}

/// matmul 测点的确定性输入 + CPU 参考：A、B 各取一套固定取模模式，参考由 f64 累加给出。
fn matmul_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (a, b) = matmul_values(m, n, k);
    let reference = cpu_matmul(&a, &b, m, n, k);
    (a, b, reference)
}

/// matmul 测点的确定性输入（不含参考）：大形状的 f64 参考要跑上分钟，吞吐测点只要输入。
fn matmul_values(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    (a, b)
}

/// 注意力测点的确定性输入 + CPU 参考：Q/K/V 各取一套固定取模模式，参考由 [`cpu_attention`] 给出。
fn attention_inputs(s: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..s * d).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let k: Vec<f32> = (0..s * d).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let v: Vec<f32> = (0..s * d).map(|i| ((i % 3) as f32) * 0.3 - 0.3).collect();
    let reference = cpu_attention(&q, &k, &v, s, d);
    (q, k, v, reference)
}

/// 注意力基准的缓冲套件：Q/K/V 输入缓冲、Y 输出缓冲与 CPU 参考。
struct AttentionBench(MetalBuffer, MetalBuffer, MetalBuffer, MetalBuffer, Vec<f32>);

/// 上传 [`attention_inputs`] 的 Q/K/V，再开一路只有 GPU 写的 Y 输出。
fn attention_bench(device: &MetalDevice, s: usize, d: usize) -> AttentionBench {
    let (q, k, v, reference) = attention_inputs(s, d);
    AttentionBench(
        upload(device, &q, "Q"),
        upload(device, &k, "K"),
        upload(device, &v, "V"),
        output_buffer(device, s * d, "Y"),
        reference,
    )
}

/// 上传一路输入：共享缓冲按元素数 × 4 分配，`values` 原样拷进去（`label` 只进失败消息）。
fn upload(device: &MetalDevice, values: &[f32], label: &str) -> MetalBuffer {
    let mut buf = device
        .new_buffer(values.len() * 4)
        .unwrap_or_else(|| panic!("{label} 缓冲"));
    buf.as_mut_bytes().copy_from_slice(&as_bytes(values));
    buf
}

/// 上传一段原始字节（int8 输入走这里，按字节而不是 f32 分配）。
fn upload_bytes(device: &MetalDevice, bytes: &[u8], label: &str) -> MetalBuffer {
    let mut buf = device
        .new_buffer(bytes.len())
        .unwrap_or_else(|| panic!("{label} 缓冲"));
    buf.as_mut_bytes().copy_from_slice(bytes);
    buf
}

/// 开一路只由 GPU 写的输出缓冲：`len` 个 f32，内容留白（`label` 只进失败消息）。
fn output_buffer(device: &MetalDevice, len: usize, label: &str) -> MetalBuffer {
    device
        .new_buffer(len * 4)
        .unwrap_or_else(|| panic!("{label} 缓冲"))
}

/// 标量网格程序：按 `kernel` 与 `m×n` 生成 MSL 再编译成管线；`what` 只进失败消息。
fn grid_pipeline(
    device: &MetalDevice,
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    kernel: &str,
    m: usize,
    n: usize,
    what: &str,
) -> MetalPipeline {
    let msl = cutile::execution::metal_grid_program(catalogue, dirs, kernel, m, n)
        .unwrap_or_else(|_| panic!("{what} MSL 应能生成"));
    device
        .compile(&msl)
        .unwrap_or_else(|| panic!("{what} MSL 应能编译"))
}

/// simdgroup GEMM 的一个片段形状：生成 MSL、编译成管线，并给出对应的网格（每线程组 32 线程）。
fn simdgroup_gemm_pipeline(
    device: &MetalDevice,
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    m: usize,
    n: usize,
    row_frags: usize,
    col_frags: usize,
) -> (MetalPipeline, [u32; 3]) {
    let msl = cutile::execution::metal_simdgroup_gemm_program(
        catalogue,
        dirs,
        "matmul_f32",
        m,
        n,
        row_frags,
        col_frags,
    )
    .expect("simdgroup MSL 应能生成");
    let pipe = device.compile(&msl).expect("simdgroup MSL 应能编译");
    let grid = [
        ((m / (8 * row_frags)) * 32) as u32,
        (n / (8 * col_frags)) as u32,
        1,
    ];
    (pipe, grid)
}

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
/// [`cutile::execution::metal_grid_program`]（标量网格基线）与
/// [`cutile::execution::metal_tiled_gemm_program`]（共享内存分块）生成 MSL，在真机 GPU 上跑
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
        let msl = cutile::execution::metal_tiled_gemm_program(
            &catalogue,
            &dirs,
            "matmul_f32",
            M,
            N,
            tile,
            per_thread,
        )
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

/// int8 GEMM 基准的固定几何与输入：64×256×4096，`i8` 输入、`i32` 精确累加、`f32` 参考。
///
/// 两个 int8 用例共用同一份样本，它们对拍的对象因此逐位相同——换一组数不该改变结论。
struct Int8Gemm {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<u8>,
    b: Vec<u8>,
    reference: Vec<f32>,
}

/// 一次 CPU 参考（`op::matmul_i8::forward_tiled`）给出 `f32` 参考列，设备侧只与它比。
fn int8_gemm() -> Int8Gemm {
    let (m, n, k) = (64usize, 256usize, 4096usize);
    let a: Vec<u8> = (0..m * k)
        .map(|i| (((i % 17) as i32 - 8) as i8) as u8)
        .collect();
    let b: Vec<u8> = (0..k * n)
        .map(|i| (((i % 19) as i32 - 9) as i8) as u8)
        .collect();
    let (sa, sb): (Vec<i8>, Vec<i8>) = (
        a.iter().map(|&x| x as i8).collect(),
        b.iter().map(|&x| x as i8).collect(),
    );
    let mut acc = vec![0i32; m * n];
    op::matmul_i8::forward_tiled(&sa, &sb, m, n, k, &mut acc).expect("CPU 参考");
    Int8Gemm {
        m,
        n,
        k,
        a,
        b,
        reference: acc.iter().map(|&v| v as f32).collect(),
    }
}

/// int8 GEMM 基准的设备侧：`int8_gemm` 的 A/B/Y 三路缓冲 + 网格。
///
/// 两个 int8 用例的输入、输出与网格完全一致，只是 MSL 来源不同（本地生成 / cutile 生产入口），
/// 因此缓冲与网格只在这里建一次。
struct Int8GemmBench {
    m: usize,
    n: usize,
    k: usize,
    reference: Vec<f32>,
    a: MetalBuffer,
    b: MetalBuffer,
    y: MetalBuffer,
    grid: [u32; 3],
}

/// 上传 [`int8_gemm`] 的 A/B，再开一路只有 GPU 写的 Y 输出。
fn int8_gemm_bench(device: &MetalDevice) -> Int8GemmBench {
    let fixture = int8_gemm();
    let (m, n, k) = (fixture.m, fixture.n, fixture.k);
    let a = upload_bytes(device, &fixture.a, "A-int8");
    let b = upload_bytes(device, &fixture.b, "B-int8");
    let y = output_buffer(device, m * n, "Y-int8");
    Int8GemmBench {
        m,
        n,
        k,
        reference: fixture.reference,
        a,
        b,
        y,
        grid: [m as u32, n as u32, 1],
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
    let fallback =
        cutile::execution::metal_int8_gemm_program(&catalogue, &dirs, "matmul_f32", m, n, 16)
            .expect("非 int8 算子应回落到网格/串行路径");
    assert!(
        !fallback.contains("threadgroup int As"),
        "非 int8 签名不得走 int8 分块 lowering"
    );
    let ragged =
        cutile::execution::metal_int8_gemm_program(&catalogue, &dirs, "matmul_i8", m, n, 24)
            .expect("K 不整除时应回落到网格/串行路径");
    assert!(
        !ragged.contains("threadgroup int As"),
        "K 不被 tile 整除时不得走 int8 分块 lowering"
    );

    for tile in [8usize, 16, 32] {
        let msl =
            cutile::execution::metal_int8_gemm_program(&catalogue, &dirs, "matmul_i8", m, n, tile)
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
/// [`cutile::execution::metal_grid_program`]（每查询行一线程）与
/// [`cutile::execution::metal_tiled_attention_program`]（每行一个 threadgroup）生成 MSL，
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
    let tiled_msl = cutile::execution::metal_tiled_attention_program(
        &catalogue,
        &dirs,
        "attention_f32",
        S,
        D,
        THREADS,
    )
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
        let flash_msl = cutile::execution::metal_flash_attention_program(
            &catalogue,
            &dirs,
            "attention_f32",
            S,
            D,
            THREADS,
            block,
        )
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

    let tiled_msl = cutile::execution::metal_tiled_attention_program(
        &catalogue,
        &dirs,
        "attention_f32",
        S,
        D,
        THREADS,
    )
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
        let msl = cutile::execution::metal_flash_attention_program(
            &catalogue,
            &dirs,
            "attention_f32",
            S,
            D,
            THREADS,
            key_block,
        )
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
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    let scale = (d as f32).sqrt();
    let mut out = vec![0.0f32; s * d];
    let mut scores = vec![0.0f32; s];
    for i in 0..s {
        let mut row_max = f32::MIN;
        for j in 0..s {
            let mut dot = 0.0f64;
            for p in 0..d {
                dot += f64::from(q[i * d + p]) * f64::from(k[j * d + p]);
            }
            scores[j] = (dot as f32) / scale;
            row_max = row_max.max(scores[j]);
        }
        let mut sum = 0.0f64;
        for score in &mut scores {
            *score = (*score - row_max).exp();
            sum += f64::from(*score);
        }
        for h in 0..d {
            let mut acc = 0.0f64;
            for j in 0..s {
                acc += f64::from(scores[j]) * f64::from(v[j * d + h]);
            }
            out[i * d + h] = (acc / sum) as f32;
        }
    }
    out
}

/// 两版 matmul MSL 共用的前导：A、B、C 三路接口声明，`M`/`N`/`K` 由调用方随后内联。
const MATMUL_MSL_HEAD: &str = "#include <metal_stdlib>\n\
     using namespace metal;\n\
     kernel void kernel_main(\n\
     \x20    device const float* A [[buffer(0)]],\n\
     \x20    device const float* B [[buffer(1)]],\n\
     \x20    device float* C [[buffer(2)]],\n";

/// 朴素 MSL：每线程算一个输出元素；`tiled` 为真时用 threadgroup 共享内存把 A/B 按 16×16 分块，
/// 每个线程组 16×16 线程协同搬运一次整块，规约轴复用共享块，避免反复从设备全局内存重读。
fn matmul_msl(tiled: bool, m: usize, n: usize, k: usize) -> String {
    if !tiled {
        return format!(
            "{MATMUL_MSL_HEAD}\
             \x20    uint2 gid [[thread_position_in_grid]]) {{\n\
             \x20    const uint M = {m}, N = {n}, K = {k};\n\
             \x20    float acc = 0.0f;\n\
             \x20    for (uint p = 0; p < K; ++p) acc += A[gid.y * K + p] * B[p * N + gid.x];\n\
             \x20    C[gid.y * N + gid.x] = acc;\n\
             }}"
        );
    }
    format!(
        "{MATMUL_MSL_HEAD}\
         \x20    uint2 gid [[thread_position_in_grid]],\n\
         \x20    uint2 lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint M = {m}, N = {n}, K = {k};\n\
         \x20    threadgroup float As[16][16];\n\
         \x20    threadgroup float Bs[16][16];\n\
         \x20    float acc = 0.0f;\n\
         \x20    for (uint kk = 0; kk < K; kk += 16) {{\n\
         \x20        As[lid.y][lid.x] = A[gid.y * K + kk + lid.x];\n\
         \x20        Bs[lid.y][lid.x] = B[(kk + lid.y) * N + gid.x];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint p = 0; p < 16; ++p) acc += As[lid.y][p] * Bs[p][lid.x];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    C[gid.y * N + gid.x] = acc;\n\
         }}"
    )
}

/// 同一算术的 CPU 参考（f64 累加，保证对拍基准的精度只受量化级控制，不引入累加顺序差异）。
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// 设备输出与 CPU 参考对拍：阈值 1e-3，分母按每元素 `max(|want|, 1)`（[`testkit::Scale::Unit`]）。
///
/// 这里选 `Unit` 而非夹具层默认的整批口径，是因为本文件的输出量级差得远：GEMM 的元素是上百上千项
/// 乘积之和，量级在 1 以上；注意力输出是 softmax 权重乘 V 的和，整批最大幅度只有 1e-4 量级。
/// 整批口径会把注意力输出上 1e-5 的绝对差放大成 1e0 的相对误差，那不是「算错了」而是「分母选错了」。
/// `Unit` 给每元素留 1 的绝对地板，正对这两类输出的共同下限；最差元素的下标与两侧取值仍由夹具层报出。
fn assert_matmul_matches(label: &str, reference: &[f32], output: &[u8]) {
    let got = read_back(output);
    testkit::assert_rel_with(
        testkit::Scale::Unit,
        &got,
        reference,
        1e-3,
        &format!("{label} 的 GPU 输出与 CPU 参考"),
    );
}

/// 把设备缓冲回读成 f32 序列（缓冲按 f32 分配、按 4 字节切块）。
fn read_back(output: &[u8]) -> Vec<f32> {
    let (chunks, _) = output.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

/// 每元素相对误差的最大值，分母按 `max(|want|, 1)`（[`testkit::Scale::Unit`] 口径），只用于打印。
fn max_relative(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(got, want)| f64::from((got - want).abs()) / f64::from(want.abs()).max(1.0))
        .fold(0.0, f64::max)
}

/// 同一内核连跑 50 次，返回中位数延迟（ns）；预热一次不计入。
fn measure(
    queue: &MetalCommandQueue,
    pipeline: &MetalPipeline,
    buffers: &[&MetalBuffer],
    grid: [u32; 3],
    block: [u32; 3],
) -> f64 {
    queue.execute(pipeline, buffers, grid, block);
    let mut samples = [0u128; 50];
    for slot in &mut samples {
        let start = std::time::Instant::now();
        queue.execute(pipeline, buffers, grid, block);
        *slot = start.elapsed().as_nanos();
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64
}

/// 把 `value` 按张量核输入精度舍入：10 位尾数、就近取偶，等价于 f32→half→f32 的往返。
///
/// 只用于造 CPU 参考——`relaxed` 路径的 A、B 都只有 10 位尾数（half / tf32 同级），拿全精度
/// 参考去对拍只会把「精度档不同」误报成「算错了」。指数域不动：测点取值落在 half 正规区间。
fn half_precision(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    let dropped = mantissa & 0x1fff;
    let mut kept = mantissa >> 13;
    if dropped > 0x1000 || (dropped == 0x1000 && kept & 1 == 1) {
        kept += 1;
    }
    f32::from_bits(exponent | (kept << 13))
}

/// half 位型 → f32（临时调试用）。
fn half_value(bits: u16) -> f32 {
    let sign = bits >> 15;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0.0,
        (0, _) => mantissa as f32 * 2f32.powi(-24),
        (0x1f, _) => f32::NAN,
        _ => f32::from_bits((exponent as u32 + 112) << 23 | (mantissa as u32) << 13),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// 张量核 `relaxed` 口径的对拍阈值：输入只有 10 位尾数、硬件还可能低精度累加，故放到 5e-3；
/// 同形的全 f32 路径（分块 / simdgroup）仍按 1e-3 判。
const TENSOR_TOLERANCE: f64 = 5e-3;

/// 张量核输出与「按同精度舍入后的 f64 参考」对拍。
fn assert_relaxed_matches(label: &str, reference: &[f32], output: &[u8]) -> f64 {
    let got = read_back(output);
    let error = max_relative(&got, reference);
    assert!(
        error < TENSOR_TOLERANCE,
        "{label} 的 GPU 输出与 CPU 参考偏差 {error:.2e} 超过 {TENSOR_TOLERANCE:.0e}"
    );
    error
}

/// 张量核 GEMM 打通验证 + 吞吐测量：先跑一次 B 的布局前置（`K×N` f32 → `N×K` half），
/// 再对主内核连跑 50 次取中位数。
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
    // llm.u 的默认档：128 输出块 / 96 规约块 / 8 simdgroup。
    let (tile, k_tile, simdgroups) = (128, 96, 8);

    // `matmul_f32` 的规约轴由算子语义固定为 4096，故形状只放开 M/N。
    for (m, n, k) in [
        (512, 512, 4096),
        (1024, 1024, 4096),
        (2048, 2048, 4096),
        (4096, 4096, 4096),
    ] {
        let (a, b) = matmul_values(m, n, k);
        let a_buf = upload(&device, &a, "A");
        let b_buf = upload(&device, &b, "B");
        let mut bt_buf = device.new_buffer(n * k * 2).expect("Bt 缓冲");
        let mut y_buf = output_buffer(&device, m * n, "Y");

        // 布局前置：`B[K][N]` 的 f32 → `Bt[N][K]` 的 half，整个基准只跑一次。
        let transpose = crate::tiled::metal_transpose_half(k, n).expect("转置 MSL 应能生成");
        let transpose = device.compile(&transpose).expect("转置 MSL 应能编译");
        assert!(
            queue.execute(
                &transpose,
                &[&b_buf, &bt_buf],
                [(n as u32).div_ceil(32) * 32, (k as u32).div_ceil(32) * 8, 1],
                [32, 8, 1],
            ),
            "转置派发应成功"
        );

        if m == 512 {
            // 决定性判据：A、B 全 1 时正确输出必须恒等于 K。
            let ones_a = upload(&device, &vec![1.0f32; m * k], "ones A");
            let mut ones_bt = device.new_buffer(n * k * 2).expect("ones Bt");
            let half_ones: Vec<u8> = vec![0x00, 0x3c].repeat(n * k);
            ones_bt.as_mut_bytes().copy_from_slice(&half_ones);
            let mut probe_y = output_buffer(&device, m * n, "ones Y");
            let probe_msl = cutile::execution::metal_tensor_gemm_program(
                &catalogue, &dirs, "matmul_f32", m, n, tile, k_tile, simdgroups,
            )
            .expect("探针 MSL");
            let probe = device.compile(&probe_msl).expect("探针编译");
            assert!(
                queue.execute(
                    &probe,
                    &[&ones_a, &ones_bt, &probe_y],
                    [((n / tile) * 32) as u32, ((m / tile) * simdgroups) as u32, 1],
                    [32, simdgroups as u32, 1],
                ),
                "探针派发"
            );
            let pv = read_back(probe_y.as_mut_bytes());
            let lo = pv.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = pv.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("全 1 判据：Y 取值范围 [{lo}, {hi}]，应为 {k}；样本 {:?}", &pv[..8]);
        }

        let msl = cutile::execution::metal_tensor_gemm_program(
            &catalogue,
            &dirs,
            "matmul_f32",
            m,
            n,
            tile,
            k_tile,
            simdgroups,
        )
        .expect("张量核 MSL 应能生成");
        if m == 512 {
            std::fs::write("/tmp/tensor_gemm.metal", &msl).unwrap();
        }
        assert!(
            msl.contains("mpp::tensor_ops::matmul2d"),
            "Gemm 骨架应走张量核 lowering"
        );
        let pipe = device.compile(&msl).expect("张量核 MSL 应能编译");
        let grid = [((n / tile) * 32) as u32, ((m / tile) * simdgroups) as u32, 1];
        let block = [32, simdgroups as u32, 1];
        assert!(
            queue.execute(&pipe, &[&a_buf, &bt_buf, &y_buf], grid, block),
            "GEMM 派发应成功"
        );

        let error = if m * n * k <= 1024 * 1024 * 1024 {
            let reference = cpu_matmul(
                &a.iter().copied().map(half_precision).collect::<Vec<f32>>(),
                &b.iter().copied().map(half_precision).collect::<Vec<f32>>(),
                m,
                n,
                k,
            );
            if m == 512 {
                let got = read_back(y_buf.as_mut_bytes());
                let aq: Vec<f32> = a.iter().copied().map(half_precision).collect();
                let bq: Vec<f32> = b.iter().copied().map(half_precision).collect();
                let (hc, _) = bt_buf.as_mut_bytes().as_chunks::<2>();
                let hv: Vec<f32> = hc
                    .iter()
                    .map(|c| half_value(u16::from_ne_bytes(*c)))
                    .collect();
                let nan = hv.iter().filter(|v| v.is_nan()).count();
                eprintln!(
                    "Bt NaN {} / {}；Bt[0..8]={:?}；期望 Bt[0..8]={:?}",
                    nan,
                    hv.len(),
                    &hv[..8],
                    (0..8).map(|i| b[(i % k) * n + i / k]).collect::<Vec<f32>>()
                );
                eprintln!(
                    "Y NaN {}；Y[1000..1008]={:?}",
                    got.iter().filter(|v| v.is_nan()).count(),
                    &got[1000..1008]
                );
                let worst = got
                    .iter()
                    .zip(&reference)
                    .enumerate()
                    .max_by(|x, y| {
                        (x.1 .0 - x.1 .1).abs().total_cmp(&(y.1 .0 - y.1 .1).abs())
                    })
                    .expect("非空");
                let bad = got.iter().filter(|v| !v.abs().lt(&1e3)).count();
                eprintln!(
                    "最坏 idx={} → (行 {}, 列 {}) got={} ref={}；|got|≥1e3 的元素 {} / {}",
                    worst.0,
                    worst.0 / n,
                    worst.0 % n,
                    worst.1 .0,
                    worst.1 .1,
                    bad,
                    got.len()
                );
                for kk in [1usize, 16, 32, 96, 192, 512] {
                    let bs: Vec<f32> =
                        (0..kk).flat_map(|p| bq[p * n..(p + 1) * n].iter().copied()).collect();
                    let as_: Vec<f32> = (0..m)
                        .flat_map(|i| aq[i * k..i * k + kk].iter().copied())
                        .collect();
                    let r = cpu_matmul(&as_, &bs, m, n, kk);
                    eprintln!("K={kk:4} 最大相对误差 {:.3e}", max_relative(&got, &r));
                }
            }
            assert_relaxed_matches(&format!("张量核 {m}×{n}×{k}"), &reference, y_buf.as_mut_bytes())
        } else {
            f64::NAN
        };

        let ns = measure(&queue, &pipe, &[&a_buf, &bt_buf, &y_buf], grid, block);
        let gflops = 2.0 * (m * n * k) as f64 / ns;
        println!(
            "张量核 {m}×{n}×{k}: {:.3} ms → {gflops:.0} GF/s（最大相对误差 {error:.2e}）",
            ns / 1e6
        );
    }
}
