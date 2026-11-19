use super::*;

/// 装配 Metal / CUDA 后端（幂等，进程级只装一次）：cutile 的机制层按 `emit` 从注册表取后端，
/// 不注入的话 `device_available` 会恒为假，真机对拍就静默跳过。
fn ensure_backends() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(engine::backends::install);
}

/// 写一份两层稠密权重（`[out, in]` 行主序），取值落在 2 的负幂网格上：乘积与累加在 f32
/// 里都是精确的，因此手算的结果可以被逐位钉住。
fn write_dense_model(name: &str, layers: &[(&str, usize, usize)]) -> std::path::PathBuf {
    let mut entries: Vec<(String, Vec<f32>)> = Vec::new();
    for (entry, out, input) in layers {
        let values = (0..out * input)
            .map(|index| ((index * 5) % 17) as f32 * 0.25 - 1.0)
            .collect();
        entries.push((format!("{entry}{}", nn::WEIGHT_SUFFIX), values));
    }
    let mut json = String::from("{");
    let mut offset = 0usize;
    for (index, (name, values)) in entries.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let end = offset + values.len() * 4;
        let (_, out, input) = layers[index];
        json.push_str(&format!(
                "\"{name}\": {{\"dtype\": \"F32\", \"shape\": [{out}, {input}], \"data_offsets\": [{offset}, {end}]}}"
            ));
        offset = end;
    }
    json.push('}');
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    for (_, values) in &entries {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let path = std::env::temp_dir().join(format!("{}-{name}", std::process::id()));
    std::fs::write(&path, &bytes).expect("临时文件可写");
    path
}

/// 主机折叠分块归约的部分结果（sum / max）：max 结合且顺序无关，sum 用 f64 折叠输入做对照
/// （f32 串行求和对 4096 递增值的误差已 ~1e-4，会淹没分块本身的正确性）。
fn fold_reduce(op_name: &str, partials: &[f32], input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    if op_name.contains("_max_") {
        let out = partials.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want = input
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(f64::from(x)));
        (vec![out], vec![want as f32])
    } else {
        let out = partials.iter().sum::<f32>();
        let want = input.iter().fold(0.0f64, |acc, &x| acc + f64::from(x));
        (vec![out], vec![want as f32])
    }
}

/// 对拍断言：相对误差须落在 dtype 容差内。
fn assert_close(
    backend: &str,
    op_name: &str,
    output: &[f32],
    want: &[f32],
    dtype: op::model::DType,
) {
    let max_rel_err = cutile::reference::max_rel_err(output, want);
    let tolerance = cutile::reference::tolerance(dtype);
    assert!(
        max_rel_err <= tolerance,
        "{op_name} {backend} 对拍失败：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"
    );
}

/// 从设备缓冲读回 `len` 个 f32（小端位模式）。
fn read_f32(bytes: &[u8], len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let word: [u8; 4] = bytes[index * 4..index * 4 + 4]
                .try_into()
                .expect("四个字节一个 f32");
            f32::from_le_bytes(word)
        })
        .collect()
}

/// 把 f32 切片按 LE 字节序写进执行器缓冲（绑槽 + 落位）。
fn bind_f32(executor: &mut engine::metal::MetalExecutor, slot: u32, values: &[f32]) {
    executor
        .bind_bytes(slot, values.len() * 4)
        .expect("f32 缓冲");
    let bytes = executor.buffer_mut(slot).expect("f32 可写").as_mut_bytes();
    for (index, value) in values.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// 把 u8 切片写进执行器缓冲（绑槽 + 拷贝）。
fn bind_u8(executor: &mut engine::metal::MetalExecutor, slot: u32, values: &[u8]) {
    executor.bind_bytes(slot, values.len()).expect("u8 缓冲");
    let bytes = executor.buffer_mut(slot).expect("u8 可写").as_mut_bytes();
    bytes.copy_from_slice(values);
}

/// 把单个 f32 标量写进执行器缓冲（4 字节）。
fn bind_scalar(executor: &mut engine::metal::MetalExecutor, slot: u32, value: f32) {
    executor.bind_bytes(slot, 4).expect("标量缓冲");
    let bytes = executor.buffer_mut(slot).expect("标量可写").as_mut_bytes();
    bytes.copy_from_slice(&value.to_le_bytes());
}

/// 按声明签名把样本缓冲绑进执行器：可写参数绑 `output_len` 字节输出槽，只读参数按 C 类型
/// （f32/u8）逐元素写输入字节。逐点与分块对拍共用这一份绑缓冲样板。
fn bind_decl_buffers(
    executor: &mut engine::metal::MetalExecutor,
    node: &mut engine::exec::Node,
    decl: &op::decl::OpDecl,
    sample: &op::model::Sample,
    output_len: usize,
) {
    use op::sig::CType;

    for (slot, param) in decl
        .signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .enumerate()
    {
        let slot = slot as u32;
        if param.mutable {
            executor.bind_bytes(slot, output_len * 4).expect("输出缓冲");
            node.bind_output(slot);
        } else {
            node.bind_input(slot);
            match param.ctype {
                CType::F32 => {
                    let values = sample.f32(&param.name).expect("f32 输入");
                    bind_f32(executor, slot, values);
                }
                CType::U8 => {
                    let values = sample.u8(&param.name).expect("u8 输入");
                    bind_u8(executor, slot, values);
                }
                CType::Int => unreachable!("标量不是缓冲"),
            }
        }
    }
}

/// 提交单个节点并同步，回读 `len` 个 f32 输出。逐点与分块归约对拍的发射/回读样板。
fn submit_read(
    executor: &mut engine::metal::MetalExecutor,
    node: engine::exec::Node,
    decl: &op::decl::OpDecl,
    len: usize,
) -> Vec<f32> {
    use engine::exec::Executor as _;

    let mut graph = engine::exec::Graph::new();
    graph.push(node);
    let stream = engine::exec::Stream::new(0);
    executor.submit(&graph, stream).expect("提交");
    executor.sync(stream).expect("同步派发");

    let output_slot = decl
        .signature
        .params()
        .iter()
        .position(|param| param.mutable)
        .expect("唯一可写输出") as u32;
    let bytes = executor
        .buffer_mut(output_slot)
        .expect("输出可读")
        .as_mut_bytes();
    read_f32(bytes, len)
}

/// 装配 Metal 后端 + 内置目录 + 临时工作目录；设备不可用时返回 `None` 静默跳过（真机对拍
/// 在无 Metal 的环境里不失败）。
fn metal_harness(tag: &str) -> Option<(cutile::builtin::Catalogue, cutile::dirs::Dirs)> {
    ensure_backends();
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    if !cutile::witness::device_available(catalogue.profile("metal").expect("metal 档案")) {
        return None;
    }
    let dirs =
        cutile::dirs::Dirs::under(std::env::temp_dir().join(format!("llm-metal-{tag}-check")));
    Some((catalogue, dirs))
}

/// 分块内核对拍的公共装备：语义表 + 样本 + 参考 + 声明 + 分块网格。`len_of` 给出分块轴长度
/// （scan 从签名里找非缓冲标量，rmsnorm / softmax 直接取 `N`），三种分块对拍共用同一份装配。
struct ChunkSetup {
    catalogue: cutile::builtin::Catalogue,
    dirs: cutile::dirs::Dirs,
    model: op::model::Model,
    sample: op::model::Sample,
    expected: Vec<f32>,
    decl: op::decl::OpDecl,
    m: usize,
    n: usize,
    len: u32,
    chunk: u32,
    grid_size: u32,
}

fn chunk_setup(
    op_name: &str,
    tag: &str,
    len_of: impl FnOnce(&op::decl::OpDecl, &op::model::Sample) -> u32,
) -> Option<ChunkSetup> {
    let (catalogue, dirs) = metal_harness(tag)?;
    let model = op::model(op_name).expect("语义表");
    let (m, n) = model.shape;
    let sample = (model.sample)(m, n);
    let expected = (model.reference)(&sample).expect("标量参考");
    let decl = catalogue.op(op_name).expect("算子声明").clone();
    let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size as u32;
    let len = len_of(&decl, &sample);
    let grid_size = len.div_ceil(chunk);
    Some(ChunkSetup {
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
    })
}

#[test]
fn a_real_weight_file_runs_through_generated_kernels() {
    if cutile::toolchain::detect().is_none() {
        // 没有 C 编译器就没有执行证据；这条用例在这里不成立，不是通过。
        return;
    }
    let path = write_dense_model(
        "llm-run-test.safetensors",
        &[("blk.0.attn_q", 4, 8), ("blk.0.attn_o", 4, 4)],
    );
    let input = "-0.5,-0.375,-0.25,-0.125,0,0.125,0.25,0.375";
    run(&path.display().to_string(), Some(input), Some("cpu")).expect("两层链应当跑通");
}

#[test]
fn the_first_layer_computes_the_hand_arithmetic_on_its_own_bytes() {
    if cutile::toolchain::detect().is_none() {
        return;
    }
    let path = write_dense_model("llm-run-arithmetic.safetensors", &[("blk.0.attn_q", 4, 8)]);
    let file = weights::safetensors::File::open(&path).expect("文件合法");
    let entry = file
        .header()
        .get(&format!("blk.0.attn_q{}", nn::WEIGHT_SUFFIX))
        .expect("条目存在");
    let w = weights::load_f32(entry, &file).expect("f32 条目");
    let x: Vec<f32> = (0..8)
        .map(|index| (index % 8) as f32 * 0.125 - 0.5)
        .collect();
    let model = op::model(nn::Linear::DENSE).expect("稠密语义");
    let sample = op::model::bind(&model, 8, 4, &x, &w).expect("形状自洽");

    let settings = crate::settings::Settings::load()
        .map_err(|error| error.with("配置"))
        .expect("默认设置");
    let dirs = settings.dirs();
    let catalogue = cutile::builtin::Catalogue::load(&dirs).expect("内置数据");
    let executed = cutile::execution::execute(&catalogue, &dirs, nn::Linear::DENSE, "cpu", &sample)
        .expect("执行");
    let want: Vec<f32> = (0..4)
        .map(|row| (0..8).map(|column| w[row * 8 + column] * x[column]).sum())
        .collect();
    assert_eq!(executed.output, want, "真实权重算出来的数必须等于手算");
    assert_eq!(executed.max_rel_err, 0.0, "同一份舍入顺序应当逐位相等");
}

/// 首个真机对拍闭环：`silu_f32` 走 op::REGISTRY → cutile emit → MetalExecutor 进程内编译 →
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

    let model = op::model(op_name).expect("语义表");
    let (m, n) = shape.unwrap_or(model.shape);
    let sample = (model.sample)(m, n);
    let expected = (model.reference)(&sample).expect("标量参考");
    let decl = catalogue.op(op_name).expect("算子声明");

    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, op_name, m, n)
        .expect("内核应能生成");

    let Some(mut executor) = engine::metal::MetalExecutor::new("metal", "arm64") else {
        return;
    };
    executor.register(op_name, &msl).expect("注册内核");

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
            .find(|param| !param.ctype.is_buffer())
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
    bind_decl_buffers(&mut executor, &mut node, decl, &sample, expected.len());

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
    let Some((catalogue, dirs)) = metal_harness("chunk") else {
        return;
    };

    let model = op::model(op_name).expect("语义表");
    let (m, n) = model.shape;
    let sample = (model.sample)(m, n);
    let decl = catalogue.op(op_name).expect("算子声明");
    let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size as u32;
    let len = decl
        .signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .and_then(|param| sample.scalar(&param.name))
        .expect("归约长度标量") as u32;
    let grid_size = len.div_ceil(chunk);

    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, op_name, m, n)
        .expect("内核应能生成");
    let Some(mut executor) = engine::metal::MetalExecutor::new("metal", "arm64") else {
        return;
    };
    executor.register(op_name, &msl).expect("注册内核");

    let launch = engine::exec::Launch::new([grid_size, 1, 1], [1, 1, 1]);
    let mut node = engine::exec::Node::new(op_name, launch);
    bind_decl_buffers(&mut executor, &mut node, decl, &sample, grid_size as usize);

    let partials = submit_read(&mut executor, node, decl, grid_size as usize);

    // 主机折叠部分结果（sum / max）；参考用 f64 折叠输入，理由见函数注释。
    let input = sample.f32("a").expect("归约输入");
    let (output, want) = fold_reduce(op_name, &partials, input);
    assert_close("分块归约", op_name, &output, &want, model.dtype);
}

/// CUDA 逐点网格真机对拍：纯逐点算子走 [`cutile::execution::cuda_grid_program`]
/// （`kernel_main<<<N,1>>>` 逐点网格），nvcc 编译 + GPU 派发 + 回读，与标量参考对拍。
/// 与 Metal 侧的 [`op_matches_reference`] 同口径，只是发射/回读经 `compile_run_cuda`。
#[cfg(target_os = "windows")]
fn op_matches_reference_cuda(op_name: &str) {
    ensure_backends();
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("nvidia").cloned().expect("nvidia 档案");
    if !cutile::witness::device_available(&profile) {
        return;
    }
    let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("llm-cuda-grid-check"));

    let model = op::model(op_name).expect("语义表");
    let (m, n) = model.shape;
    let sample = (model.sample)(m, n);
    let decl = catalogue.op(op_name).expect("算子声明");
    let output = decl.signature.output().expect("可写输出");
    let output_len = sample.f32(output).expect("f32 输出").len();
    let grid = output_len;

    let device = cutile::execution::cuda_grid_program(&catalogue, &dirs, op_name, m, n)
        .expect("内核应能生成");
    let program =
        engine::codegen::cuda_grid_driver(&device, &decl.signature, &sample, grid, output_len)
            .expect("host 驱动应能生成");
    let (_counts, bits) =
        cutile::witness::run_cuda(&profile, &dirs, &program, op_name).expect("CUDA 真机执行");

    let expected = (model.reference)(&sample).expect("参考值");
    assert_close("CUDA 逐点", op_name, &bits, &expected, model.dtype);
}

/// CUDA 分块归约真机对拍：`reduce` 骨架声明 `[chunk]` 后按 `ceil(N/chunk)` 网格每线程归约一块、
/// 写部分结果 `y[g]`，主机折叠部分结果再与参考对拍。折叠用 f64 输入做对照（与 Metal 侧
/// [`op_chunked_reduce_matches_reference`] 同一口径：f32 串行参考对 4096 递增值误差已 ~1e-4）。
#[cfg(target_os = "windows")]
fn op_chunked_reduce_matches_reference_cuda(op_name: &str) {
    ensure_backends();
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("nvidia").cloned().expect("nvidia 档案");
    if !cutile::witness::device_available(&profile) {
        return;
    }
    let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("llm-cuda-reduce-check"));

    let model = op::model(op_name).expect("语义表");
    let (m, n) = model.shape;
    let sample = (model.sample)(m, n);
    let decl = catalogue.op(op_name).expect("算子声明");
    let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size;
    let len = decl
        .signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .and_then(|param| sample.scalar(&param.name))
        .expect("归约长度标量") as usize;
    let grid = len.div_ceil(chunk);

    let device = cutile::execution::cuda_reduce_program(&catalogue, &dirs, op_name, m, n)
        .expect("内核应能生成");
    let program = engine::codegen::cuda_grid_driver(&device, &decl.signature, &sample, grid, grid)
        .expect("host 驱动应能生成");
    let (_counts, bits) =
        cutile::witness::run_cuda(&profile, &dirs, &program, op_name).expect("CUDA 真机执行");

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

    let Some(setup) = chunk_setup(op_name, "scan", |decl, sample| {
        decl.signature
            .params()
            .iter()
            .find(|param| !param.ctype.is_buffer())
            .and_then(|param| sample.scalar(&param.name))
            .expect("扫描长度标量") as u32
    }) else {
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

    let (local_msl, correct_msl) =
        cutile::execution::metal_scan_program(&catalogue, &dirs, op_name, m, n)
            .expect("分块扫描内核应能生成");
    let Some(mut executor) = engine::metal::MetalExecutor::new("metal", "arm64") else {
        return;
    };
    executor
        .register(op_name, &local_msl)
        .expect("注册局部扫描内核");
    let correct_name = format!("{op_name}_correct");
    executor
        .register(&correct_name, &correct_msl)
        .expect("注册校正内核");

    // 槽位约定（同 buffer_decls 的签名序）：0 = 只读输入 x，1 = 可写输出 y。
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
        let bytes = executor.buffer_mut(1).expect("局部扫描可读").as_mut_bytes();
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
        let bytes = executor.buffer_mut(1).expect("输出可读").as_mut_bytes();
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

    let (partial_msl, map_msl) =
        cutile::execution::metal_rmsnorm_program(&catalogue, &dirs, op_name, m, n)
            .expect("分块 rmsnorm 内核应能生成");
    let Some(mut executor) = engine::metal::MetalExecutor::new("metal", "arm64") else {
        return;
    };
    executor
        .register(op_name, &partial_msl)
        .expect("注册部分平方和内核");
    let map_name = format!("{op_name}_map");
    executor
        .register(&map_name, &map_msl)
        .expect("注册归一化内核");

    // 槽位（同 buffer_decls 签名序）：0 = x、1 = gamma、2 = y（部分和临时占 y 前 G 个槽）。
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
        let bytes = executor.buffer_mut(2).expect("部分和可读").as_mut_bytes();
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
        let bytes = executor.buffer_mut(2).expect("输出可读").as_mut_bytes();
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
        cutile::execution::metal_softmax_program(&catalogue, &dirs, op_name, m, n)
            .expect("分块 softmax 内核应能生成");
    let Some(mut executor) = engine::metal::MetalExecutor::new("metal", "arm64") else {
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
    let read_partials = |executor: &mut engine::metal::MetalExecutor| {
        let bytes = executor.buffer_mut(1).expect("部分结果可读").as_mut_bytes();
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
        let bytes = executor.buffer_mut(1).expect("输出可读").as_mut_bytes();
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
