//! 真机对拍：Metal / CUDA 上跑生成的核，与标量参考逐点比较。
//!
//! 本模块是 [`super`] 的子模块：装配样板（目录、语义表、样本、声明、绑缓冲、发射回读）都在
//! 父模块里，这里只放"在真机上跑一遍并断言"的用例；设备不在场时静默跳过，因为没有设备
//! 不是失败。

use super::*;
#[cfg(target_os = "windows")]
use llm_optim::execution::Backend::Cuda;
#[cfg(target_os = "macos")]
use llm_optim::execution::Backend::Metal;
use llm_optim::execution::{Mode, program};

/// Metal 执行器（真机不在场时 `None`；没有设备不是失败）。
#[cfg(target_os = "macos")]
fn metal_executor() -> Option<::device::metal::executor::MetalExecutor> {
    ::device::metal::executor::MetalExecutor::new("metal", "arm64")
}

/// 生成 `Mode::declared()` 的单通道 Metal 源并注册成 `op_name`。
#[cfg(target_os = "macos")]
fn declared_metal(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> Option<::device::metal::executor::MetalExecutor> {
    let msl = program(catalogue, dirs, op_name, Metal, Mode::declared(), m, n)
        .and_then(|s| s.one("降源"))
        .expect("内核应能生成");
    let mut executor = metal_executor()?;
    executor.register(op_name, &msl).expect("注册内核");
    Some(executor)
}

/// 生成 `Mode::segmented()` 的两段 Metal 源；注册名由调用方按语义决定。
#[cfg(target_os = "macos")]
fn segmented_metal(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> Option<(::device::metal::executor::MetalExecutor, String, String)> {
    let (first, second) = program(catalogue, dirs, op_name, Metal, Mode::segmented(), m, n)
        .and_then(|s| s.two("分段"))
        .expect("分段内核应能生成");
    metal_executor().map(|e| (e, first, second))
}

/// 首个真机对拍闭环：`silu_f32` 走 nn::registry::REGISTRY → cutile emit → MetalExecutor 进程内编译 →
/// 网格派发 → 回读，与标量参考对拍。证明「模板 → 拼接正文 → 网格包装 → Metal 执行」整条
/// 生产链路在真机上产出正确数值，而不只是编译通过。
#[cfg(target_os = "macos")]
#[test]
fn a_pointwise_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("silu_f32");
}

/// 分块归约真机对拍：`reduce_sum_f32` 声明 `[chunk]` 后走分块 lowering——网格 `ceil(N/256)`、
/// 每线程归约一块写部分和 `y[g]`，主机折叠部分和与 f64 参考对拍（f32 串行参考对 4096 个递增
/// 值的误差已到 ~1e-4，会淹没分块本身的正确性）。
#[cfg(target_os = "macos")]
#[test]
fn a_reduction_op_runs_on_metal_and_matches_the_reference() {
    op_chunked_reduce_matches_reference("reduce_sum_f32");
}

/// 分块归约真机对拍：`reduce_max_f32` 与 sum 同一条分块 lowering，只换折叠运算（max 是结合且
/// 顺序无关的，主机取最大与标量参考逐位一致）。
#[cfg(target_os = "macos")]
#[test]
fn a_max_reduction_op_runs_on_metal_and_matches_the_reference() {
    op_chunked_reduce_matches_reference("reduce_max_f32");
}

/// 分块扫描真机对拍：`scan_f32` 走「局部扫描 + 主机携带前缀 + 校正」三阶段，衰减 `a` 来自
/// `[consts]`，主机折叠携带前缀后回注校正，与标量参考对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_scan_op_runs_on_metal_and_matches_the_reference() {
    op_chunked_scan_matches_reference("scan_f32");
}

/// 两趟（带归约）真机对拍：`rmsnorm_f32` 声明 `[chunk]` 后走「部分平方和 + 主机折叠 +
/// 归一化」三段（两段内核 + 一次折叠），在 Metal 上算出均方根归一化并回读，与标量参考对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_two_pass_op_runs_on_metal_and_matches_the_reference() {
    op_chunked_rmsnorm_matches_reference("rmsnorm_f32");
}

/// Gemm 行网格真机对拍：`gemv_f32` 走 `[grid]` 声明映射——grid `[n,1,1]`、输出 `+ n`、
/// 权重 `+ n * m`（行跨步）、输入共享，在 Metal 上算 `y = dot(w[n,:], x)` 并回读，与标量
/// 参考对拍。用小形状（m=64, n=4）避开生产形状 4096×4096 的 64MB 权重样本。
#[cfg(target_os = "macos")]
#[test]
fn a_gemv_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference_at("gemv_f32", Some((64, 4)));
}

/// 真机对拍通用辅助：任一条走串行兜底或矩阵乘网格的算子，按签名序绑缓冲，派发网格由 `[grid]`
/// 声明推导（缺省 `[1,1,1]` 单线程串行），回读唯一可写输出，与标量参考对拍。缓冲按签名类型解释
/// （f32 四字节、u8 一字节）。`shape` 为 `None` 用算子声明形状；`gemv` 生产形状 4096×4096
/// 过大，传 `Some((64, 4))` 避开 64MB 样本。
#[cfg(target_os = "macos")]
fn op_matches_reference_at(op_name: &str, shape: Option<(usize, usize)>) {
    let Some((catalogue, dirs)) = metal_harness("serial") else {
        return;
    };

    let model = nn::registry::model(op_name).expect("语义表");
    let (m, n) = shape.unwrap_or(model.shape);
    let sample = (model.sample)(m, n);
    let expected = (model.reference)(&sample).expect("标量参考");
    let decl = catalogue.op(op_name).expect("算子声明");

    let Some(mut executor) = declared_metal(&catalogue, &dirs, op_name, m, n) else {
        return;
    };

    // 派发网格：`[grid]` 声明的各维标量取样本值；逐点算子（无归约/点积、单标量）走逐点网格
    // 快路径，线程下标即元素下标、网格覆盖整行（`N`）；其余缺省单线程串行。
    let mut grid = [1u32, 1, 1];
    if let Some(decl_grid) = decl.grid.as_ref() {
        for (dim, (_, scalar)) in decl_grid.dims.iter().enumerate() {
            grid[dim] = sample
                .scalar(scalar)
                .expect("网格维标量")
                .try_into()
                .expect("网格维落 u32");
        }
    } else if decl.pointwise() {
        let scalar = decl
            .signature
            .params()
            .iter()
            .find(|param| !param.ctype.is_buf())
            .expect("逐点单标量")
            .name
            .clone();
        grid[0] = sample
            .scalar(&scalar)
            .expect("逐点长度标量")
            .try_into()
            .expect("逐点长度落 u32");
    }
    let launch = engine::exec::Launch::new(grid, [1, 1, 1]);
    let mut node = engine::exec::Node::new(op_name, launch);
    bind_decl_bufs(&mut executor, &mut node, decl, &sample, expected.len());

    let output = submit_read(&mut executor, node, decl, expected.len());

    let max_rel_err = cutile::reference::max_rel_err(&output, &expected);
    let tolerance = cutile::reference::tolerance(model.dtype);
    assert!(
        max_rel_err <= tolerance,
        "{op_name} 串行对拍失败：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"
    );
}

/// 用算子声明形状真机对拍（缺省入口）。
#[cfg(target_os = "macos")]
fn op_matches_reference(op_name: &str) {
    op_matches_reference_at(op_name, None);
}

/// 分块归约真机对拍：`reduce` 骨架声明 `[chunk]` 后，内核按 `ceil(N/chunk)` 网格每线程归约
/// 一块、写部分结果 `y[g]`，主机折叠这些部分结果再与参考对拍（sum 求和、max 取最大）。
///
/// 对拍口径与串行辅助不同：分块求和是 pairwise（对数精度），f32 串行参考对 4096 个递增值
/// 的误差已到 ~1e-4，会淹没分块本身的正确性——所以用 f64 折叠输入做对照，分块结果的误差应
/// 落在 ~log2(N)·eps 量级，仍在 f32 阈值（1e-5）内。max 结合且顺序无关，主机取最大与参考
/// 逐位一致。
#[cfg(target_os = "macos")]
fn op_chunked_reduce_matches_reference(op_name: &str) {
    let Some(setup) = chunk_setup(op_name, "chunk", scalar_len) else {
        return;
    };
    let ChunkSetup {
        catalogue,
        dirs,
        model,
        sample,
        expected: _,
        decl,
        m,
        n,
        len: _,
        chunk: _,
        grid_size,
    } = setup;

    let Some(mut executor) = declared_metal(&catalogue, &dirs, op_name, m, n) else {
        return;
    };

    let launch = engine::exec::Launch::new([grid_size, 1, 1], [1, 1, 1]);
    let mut node = engine::exec::Node::new(op_name, launch);
    bind_decl_bufs(&mut executor, &mut node, &decl, &sample, grid_size as usize);

    let partials = submit_read(&mut executor, node, &decl, grid_size as usize);

    // 主机折叠部分结果（sum / max）；参考用 f64 折叠输入，理由见函数注释。
    let input = sample.f32("a").expect("归约输入");
    let (output, want) = fold_reduce(op_name, &partials, input);
    assert_close("分块归约", op_name, &output, &want, model.dtype);
}

/// CUDA 逐点网格真机对拍：纯逐点算子走 [`llm_optim::execution::program`]
/// （`kernel_main<<<ceil(N/256), 256>>>` 逐点网格，块内 256 线程、末块由守卫挡住），
/// nvcc 编译 + GPU 派发 + 回读，与标量参考对拍。
/// 与 Metal 侧的 [`op_matches_reference`] 同口径，只是发射/回读经 `compile_run_cuda`。
#[cfg(target_os = "windows")]
fn op_matches_reference_cuda(op_name: &str) {
    let Some(CudaSetup {
        catalogue,
        dirs,
        profile,
        model,
        sample,
        decl,
        m,
        n,
    }) = cuda_setup(op_name, "grid", None)
    else {
        return;
    };

    let device = program(&catalogue, &dirs, op_name, Cuda, Mode::pointwise(), m, n)
        .and_then(|s| s.one("降源"))
        .expect("内核应能生成");
    let threads: usize = 256;
    let blocks = (scalar_len(&decl, &sample) as usize).div_ceil(threads);
    let driver =
        ::device::cuda::cuda_point_driver(&device, &decl.signature, &sample, blocks, threads)
            .expect("host 驱动应能生成");
    let (_counts, bits) =
        llm_optim::witness::run_cuda(&profile, &dirs, &driver, op_name).expect("CUDA 真机执行");

    let expected = (model.reference)(&sample).expect("参考值");
    assert_close("CUDA 逐点", op_name, &bits, &expected, model.dtype);
}

/// CUDA 分块归约真机对拍：`reduce` 骨架声明 `[chunk]` 后按 `ceil(N/chunk)` 网格每线程归约一块、
/// 写部分结果 `y[g]`，主机折叠部分结果再与参考对拍。折叠用 f64 输入做对照（与 Metal 侧
/// [`op_chunked_reduce_matches_reference`] 同一口径：f32 串行参考对 4096 递增值误差已 ~1e-4）。
#[cfg(target_os = "windows")]
fn op_chunked_reduce_matches_reference_cuda(op_name: &str) {
    let Some(CudaSetup {
        catalogue,
        dirs,
        profile,
        model,
        sample,
        decl,
        m,
        n,
    }) = cuda_setup(op_name, "reduce", None)
    else {
        return;
    };

    let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size as u32;
    let grid = scalar_len(&decl, &sample).div_ceil(chunk) as usize;

    let device = program(&catalogue, &dirs, op_name, Cuda, Mode::segmented(), m, n)
        .and_then(|s| s.one("降源"))
        .expect("内核应能生成");
    let driver = ::device::cuda::cuda_grid_driver(&device, &decl.signature, &sample, grid, grid)
        .expect("host 驱动应能生成");
    let (_counts, bits) =
        llm_optim::witness::run_cuda(&profile, &dirs, &driver, op_name).expect("CUDA 真机执行");

    let input = sample.f32("a").expect("归约输入");
    let (output, want) = fold_reduce(op_name, &bits, input);
    assert_close("CUDA 分块归约", op_name, &output, &want, model.dtype);
}

/// 分块扫描真机对拍：`scan_f32` 声明 `[chunk]` 后走「局部扫描 + 主机携带前缀 + 校正」三阶段。
///
/// 1. 局部扫描：网格 `ceil(N/256)`，每线程 g 对块做基 0 扫描写 `y`，块末态即 `y[start+len-1]`；
/// 2. 主机携带前缀：`C_0=0`、`C_{g+1}=a^len·C_g + y[start+len-1]`（f64 折叠，衰减 `a` 来自 `[consts]`）；
/// 3. 校正：每线程 g 把 `C_g·a^(j+1)` 加回 `y`（就地累加）。
///
/// 对拍用 f64 折叠的携带前缀 + 校正结果，与 f32 标量参考比较；分块扫描的误差来自两段内核的
/// 浮点舍入，仍落在 f32 阈值（1e-5）内。
#[cfg(target_os = "macos")]
fn op_chunked_scan_matches_reference(op_name: &str) {
    use engine::exec::Executor as _;

    let Some(setup) = chunk_setup(op_name, "scan", scalar_len) else {
        return;
    };
    let ChunkSetup {
        catalogue,
        dirs,
        model,
        sample,
        expected,
        decl,
        m,
        n,
        len,
        chunk,
        grid_size,
    } = setup;
    let decay = decl
        .consts
        .iter()
        .find(|(key, _)| key == "a")
        .and_then(|(_, value)| value.parse::<f64>().ok())
        .expect("扫描衰减 a");

    let Some((mut executor, local_msl, correct_msl)) =
        segmented_metal(&catalogue, &dirs, op_name, m, n)
    else {
        return;
    };
    executor
        .register(op_name, &local_msl)
        .expect("注册局部扫描内核");
    let correct_name = format!("{op_name}_correct");
    executor
        .register(&correct_name, &correct_msl)
        .expect("注册校正内核");

    // 槽位约定（同 buf_decls 的签名序）：0 = 只读输入 x，1 = 可写输出 y。
    let input = sample.f32("x").expect("扫描输入");
    bind_f32(&mut executor, 0, input);
    executor
        .bind_bytes(1, expected.len() * 4)
        .expect("输出缓冲");

    let stream = engine::exec::Stream::new(0);
    let launch = engine::exec::Launch::new([grid_size, 1, 1], [1, 1, 1]);

    // 阶段 1：局部扫描（x → y）。
    let mut node = engine::exec::Node::new(op_name, launch);
    node.bind_input(0);
    node.bind_output(1);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交局部扫描");
    executor.sync(stream).expect("同步局部扫描");

    // 阶段 2（主机）：从局部扫描输出读块末态，做块级携带前缀。
    let local = {
        let bytes = executor.buf_mut(1).expect("局部扫描可读").as_mut_bytes();
        read_f32(bytes, expected.len())
    };
    let mut carry = vec![0.0f64; grid_size as usize];
    for g in 1..grid_size as usize {
        let prev_start = (g as u32 - 1) * chunk;
        let prev_len = chunk.min(len - prev_start);
        let tail = local[(prev_start + prev_len - 1) as usize] as f64;
        carry[g] = decay.powf(prev_len as f64) * carry[g - 1] + tail;
    }

    // 阶段 3：校正（c → y 就地累加）。槽 0 复用为携带态缓冲。
    bind_f32(
        &mut executor,
        0,
        &carry.iter().map(|&value| value as f32).collect::<Vec<_>>(),
    );
    let mut node = engine::exec::Node::new(&correct_name, launch);
    node.bind_input(0);
    node.bind_output(1);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交校正");
    executor.sync(stream).expect("同步校正");

    let output = {
        let bytes = executor.buf_mut(1).expect("输出可读").as_mut_bytes();
        read_f32(bytes, expected.len())
    };
    assert_close("分块扫描", op_name, &output, &expected, model.dtype);
}

/// 分块 RMSNorm 真机对拍：`rmsnorm_f32` 声明 `[chunk]` 后走「部分平方和 + 主机折叠 + 归一化」。
/// 1. 部分平方和：每线程 g 归约 x² 一块写 `y[g]`；2. 主机折叠成 `rms = sqrt(Σx²/N + eps)`；
/// 3. 归一化：每线程 g 算 `y = gamma·x/rms`。折叠用 f64，分块误差仍在 f32 阈值内。
#[cfg(target_os = "macos")]
fn op_chunked_rmsnorm_matches_reference(op_name: &str) {
    use engine::exec::Executor as _;

    let Some(setup) = chunk_setup(op_name, "rmsnorm", |_, sample| {
        sample.scalar("N").expect("长度标量") as u32
    }) else {
        return;
    };
    let ChunkSetup {
        catalogue,
        dirs,
        model,
        sample,
        expected,
        m,
        n,
        len,
        grid_size,
        ..
    } = setup;

    let Some((mut executor, partial_msl, map_msl)) =
        segmented_metal(&catalogue, &dirs, op_name, m, n)
    else {
        return;
    };
    executor
        .register(op_name, &partial_msl)
        .expect("注册部分平方和内核");
    let map_name = format!("{op_name}_map");
    executor
        .register(&map_name, &map_msl)
        .expect("注册归一化内核");

    // 槽位（同 buf_decls 签名序）：0 = x、1 = gamma、2 = y（部分和临时占 y 前 G 个槽）。
    let x = sample.f32("x").expect("rmsnorm 输入 x");
    let gamma = sample.f32("gamma").expect("rmsnorm 权重 gamma");
    bind_f32(&mut executor, 0, x);
    bind_f32(&mut executor, 1, gamma);
    executor.bind_bytes(2, expected.len() * 4).expect("y 缓冲");

    let stream = engine::exec::Stream::new(0);
    let launch = engine::exec::Launch::new([grid_size, 1, 1], [1, 1, 1]);

    let mut node = engine::exec::Node::new(op_name, launch);
    node.bind_input(0).bind_input(1).bind_output(2);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交部分平方和");
    executor.sync(stream).expect("同步部分平方和");

    let partials = {
        let bytes = executor.buf_mut(2).expect("部分和可读").as_mut_bytes();
        read_f32(bytes, grid_size as usize)
    };
    let sum_sq = partials
        .iter()
        .fold(0.0f64, |acc, &value| acc + f64::from(value));
    let rms = ((sum_sq / f64::from(len)) + f64::from(1e-6f32)).sqrt() as f32;

    // 槽 3 = rms（主机注入的 1 元素标量缓冲）。
    bind_scalar(&mut executor, 3, rms);

    let mut node = engine::exec::Node::new(&map_name, launch);
    node.bind_input(3)
        .bind_input(0)
        .bind_input(1)
        .bind_output(2);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交归一化");
    executor.sync(stream).expect("同步归一化");

    let output = {
        let bytes = executor.buf_mut(2).expect("输出可读").as_mut_bytes();
        read_f32(bytes, expected.len())
    };
    assert_close("分块归一化", op_name, &output, &expected, model.dtype);
}

/// 分块 Softmax 真机对拍：`softmax_f32` 声明 `[chunk]` 后走「部分最大 + 折叠 + 部分指数和 +
/// 折叠 + 归一化」五阶段（三段内核 + 两次主机折叠），与标量参考对拍。
#[cfg(target_os = "macos")]
fn op_chunked_softmax_matches_reference(op_name: &str) {
    use engine::exec::Executor as _;

    let Some(setup) = chunk_setup(op_name, "softmax", |_, sample| {
        sample.scalar("N").expect("长度标量") as u32
    }) else {
        return;
    };
    let ChunkSetup {
        catalogue,
        dirs,
        model,
        sample,
        expected,
        m,
        n,
        grid_size,
        ..
    } = setup;

    let (max_msl, expsum_msl, map_msl) =
        program(&catalogue, &dirs, op_name, Metal, Mode::segmented(), m, n)
            .and_then(|s| s.three("softmax"))
            .expect("分段内核应能生成");
    let Some(mut executor) = ::device::metal::executor::MetalExecutor::new("metal", "arm64") else {
        return;
    };
    executor
        .register(op_name, &max_msl)
        .expect("注册部分最大内核");
    let expsum_name = format!("{op_name}_expsum");
    let map_name = format!("{op_name}_map");
    executor
        .register(&expsum_name, &expsum_msl)
        .expect("注册指数和内核");
    executor
        .register(&map_name, &map_msl)
        .expect("注册归一化内核");

    // 槽位：0 = x、1 = y（部分结果临时占 y 前 G 个槽）、2 = max、3 = sum。
    let x = sample.f32("x").expect("softmax 输入");
    bind_f32(&mut executor, 0, x);
    executor.bind_bytes(1, expected.len() * 4).expect("y 缓冲");

    let stream = engine::exec::Stream::new(0);
    let launch = engine::exec::Launch::new([grid_size, 1, 1], [1, 1, 1]);
    let read_partials = |executor: &mut ::device::metal::executor::MetalExecutor| {
        let bytes = executor.buf_mut(1).expect("部分结果可读").as_mut_bytes();
        read_f32(bytes, grid_size as usize)
    };

    // 阶段 1：部分最大。
    let mut node = engine::exec::Node::new(op_name, launch);
    node.bind_input(0).bind_output(1);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交部分最大");
    executor.sync(stream).expect("同步部分最大");
    let max = read_partials(&mut executor)
        .into_iter()
        .fold(f32::MIN, f32::max);

    // 阶段 2：部分指数和（max 注入槽 2）。
    bind_scalar(&mut executor, 2, max);
    let mut node = engine::exec::Node::new(&expsum_name, launch);
    node.bind_input(2).bind_input(0).bind_output(1);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交部分指数和");
    executor.sync(stream).expect("同步部分指数和");
    let sum = read_partials(&mut executor)
        .into_iter()
        .fold(0.0f64, |acc, value| acc + f64::from(value));

    // 阶段 3：归一化（sum 注入槽 3）。
    bind_scalar(&mut executor, 3, sum as f32);
    let mut node = engine::exec::Node::new(&map_name, launch);
    node.bind_input(2)
        .bind_input(3)
        .bind_input(0)
        .bind_output(1);
    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    executor.submit(&graph, stream).expect("提交归一化");
    executor.sync(stream).expect("同步归一化");

    let output = {
        let bytes = executor.buf_mut(1).expect("输出可读").as_mut_bytes();
        read_f32(bytes, expected.len())
    };
    assert_close("分块 softmax", op_name, &output, &expected, model.dtype);
}

/// 三趟 softmax（max 归约 + exp 求和 + 归一化）声明 `[chunk]` 后走「部分最大 + 折叠 +
/// 部分指数和 + 折叠 + 归一化」五阶段，真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_softmax_op_runs_on_metal_and_matches_the_reference() {
    op_chunked_softmax_matches_reference("softmax_f32");
}

/// 矩阵乘（三重循环）走 2D 网格快路径，真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_gemm_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("gemm_f32");
}

/// 打包 int2 解码 GEMV 走行网格快路径（经 `base = "gemv_i4"` 继承同一 `[grid]` 映射），真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_gemv_i2_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("gemv_i2");
}

/// 打包 int4 解码 GEMV 走行网格快路径（`@quant` 解包 + 按组缩放留在正文），真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_gemv_i4_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("gemv_i4");
}

/// 对称 int8 解码 GEMV 走行网格快路径（内联 `(signed char)`、不占 `quant` 原语），真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_gemv_i8_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("gemv_i8");
}

/// 朴素三阶段注意力按查询行拆线程（`[grid] dims=["i:S"]`），行内软归一化留在每线程串行，真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn an_attention_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("attention_f32");
}

/// online softmax 分块注意力按查询行拆线程，真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_flash_attention_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("flash_attention_f32");
}

/// 单条 query 解码注意力（Q_ROWS=1 退化为单线程网格），真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_decode_attention_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("decode_attention_f32");
}

/// online softmax 单条 query 解码注意力（Q_ROWS=1 退化为单线程网格），真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn a_decode_flash_attention_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("decode_flash_attention_f32");
}

/// 行 gather（`table[idx[n],:]`，数据依赖下标）走串行兜底，真机对拍。
#[cfg(target_os = "macos")]
#[test]
fn an_embed_op_runs_on_metal_and_matches_the_reference() {
    op_matches_reference("embed_f32");
}

/// silu 逐点激活走 CUDA 逐点网格（`kernel_main<<<N,1>>>`），nvcc 编译 + T600 派发 + 回读，
/// 与标量参考对拍。
#[cfg(target_os = "windows")]
#[test]
fn a_silu_op_runs_on_cuda_and_matches_the_reference() {
    op_matches_reference_cuda("silu_f32");
}

/// 求和归约声明 `[chunk]` 后走 CUDA 分块归约（每线程归约一块写部分和 `y[g]`，主机折叠），真机对拍。
#[cfg(target_os = "windows")]
#[test]
fn a_reduce_sum_op_runs_on_cuda_and_matches_the_reference() {
    op_chunked_reduce_matches_reference_cuda("reduce_sum_f32");
}

/// CUDA 分块归一化（RMSNorm 两段 / Softmax 三段）真机对拍：设备段来自公开流水线
/// （`program(.., Cuda, Mode::segmented(), ..)` 的分段源），host 串联驱动在 C 里折叠部分和，
/// 一次编译执行整条链路，输出与标量参考逐元素对拍。
#[cfg(target_os = "windows")]
fn op_normalized_matches_reference_cuda(op_name: &str) {
    let Some(CudaSetup {
        catalogue,
        dirs,
        profile,
        model,
        sample,
        decl,
        m,
        n,
    }) = cuda_setup(op_name, "norm", None)
    else {
        return;
    };

    let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size;

    let driver = match op_name {
        "rmsnorm_f32" => {
            let (partial, norm) =
                program(&catalogue, &dirs, op_name, Cuda, Mode::segmented(), m, n)
                    .and_then(|s| s.two("分段"))
                    .expect("分段内核应能生成");
            ::device::norm::cuda_rmsnorm_driver(&[partial, norm], &decl.signature, &sample, chunk)
                .expect("host 串联驱动应能生成")
        }
        "softmax_f32" => {
            let (max, expsum, norm) =
                program(&catalogue, &dirs, op_name, Cuda, Mode::segmented(), m, n)
                    .and_then(|s| s.three("softmax"))
                    .expect("分段内核应能生成");
            ::device::norm::cuda_softmax_driver(
                &[max, expsum, norm],
                &decl.signature,
                &sample,
                chunk,
            )
            .expect("host 串联驱动应能生成")
        }
        other => panic!("`{other}` 不是归一化算子"),
    };

    let (_counts, bits) =
        llm_optim::witness::run_cuda(&profile, &dirs, &driver, op_name).expect("CUDA 真机执行");
    let expected = (model.reference)(&sample).expect("参考值");
    assert_close("CUDA 分块归一化", op_name, &bits, &expected, model.dtype);
}

/// CUDA 循环注入网格（`Gemm` 骨架：gemv 一维 / gemm 二维）真机对拍：设备段来自公开流水线
/// （`program(.., Cuda, Mode::loop_injected(), ..)` 返回设备源 + 各维网格），host 驱动用
/// `cuda_loop_driver` 按 `dim3` 发射，
/// 输出与标量参考逐元素对拍。小形状（64×64）避免样本内嵌体积膨胀。
#[cfg(target_os = "windows")]
fn op_looped_matches_reference_cuda(op_name: &str) {
    let Some(CudaSetup {
        catalogue,
        dirs,
        profile,
        model,
        sample,
        decl,
        m,
        n,
    }) = cuda_setup(op_name, "looped", Some((64, 64)))
    else {
        return;
    };

    let stages = program(
        &catalogue,
        &dirs,
        op_name,
        Cuda,
        Mode::loop_injected(),
        m,
        n,
    )
    .expect("循环注入内核应能生成");
    let dims = stages.dims().expect("循环注入应回网格各维").to_vec();
    let device = stages.one("循环注入").expect("循环注入设备源");
    let driver = ::device::cuda::cuda_loop_driver(&device, &decl.signature, &sample, &dims)
        .expect("host 驱动应能生成");
    let (_counts, bits) =
        llm_optim::witness::run_cuda(&profile, &dirs, &driver, op_name).expect("CUDA 真机执行");

    let expected = (model.reference)(&sample).expect("参考值");
    assert_close("CUDA 循环注入", op_name, &bits, &expected, model.dtype);
}

/// gemv 走 CUDA 一维循环注入网格，真机对拍。
#[cfg(target_os = "windows")]
#[test]
fn a_gemv_op_runs_on_cuda_and_matches_the_reference() {
    op_looped_matches_reference_cuda("gemv_f32");
}

/// gemm 走 CUDA 二维循环注入网格，真机对拍。
#[cfg(target_os = "windows")]
#[test]
fn a_gemm_op_runs_on_cuda_and_matches_the_reference() {
    op_looped_matches_reference_cuda("gemm_f32");
}

/// RMSNorm 走 CUDA 两段（部分平方和 → 主机折叠 rms → 归一化），真机对拍。
#[cfg(target_os = "windows")]
#[test]
fn a_rmsnorm_op_runs_on_cuda_and_matches_the_reference() {
    op_normalized_matches_reference_cuda("rmsnorm_f32");
}

/// Softmax 走 CUDA 三段（部分最大 → 部分指数和 → 归一化），真机对拍。
#[cfg(target_os = "windows")]
#[test]
fn a_softmax_op_runs_on_cuda_and_matches_the_reference() {
    op_normalized_matches_reference_cuda("softmax_f32");
}
