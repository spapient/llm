use super::*;

/// 装配 Metal / CUDA 后端（幂等，进程级只装一次）：cutile 的机制层按 `emit` 从注册表取后端，
/// 不注入的话 `device_available` 会恒为假，真机对拍就静默跳过。
fn ensure_backends() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(::shard::backends::install);
}

/// 写一份两层稠密权重（`[out, in]` 行主序），取值落在 2 的负幂网格上：乘积与累加在 f32
/// 里都是精确的，因此手算的结果可以被逐位钉住。
fn write_dense_model(name: &str, layers: &[(&str, usize, usize)]) -> std::path::PathBuf {
    let mut entries: Vec<(String, Vec<f32>)> = Vec::new();
    for (entry, out, input) in layers {
        let values = (0..out * input)
            .map(|idx| ((idx * 5) % 17) as f32 * 0.25 - 1.0)
            .collect();
        entries.push((format!("{entry}{}", nn::WEIGHT_SUFFIX), values));
    }
    let mut json = String::from("{");
    let mut offset = 0usize;
    for (idx, (name, values)) in entries.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        let end = offset + values.len() * 4;
        let (_, out, input) = layers[idx];
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
    dtype: nn::model::DType,
) {
    let max_rel_err = cutile::reference::max_rel_err(output, want);
    let tolerance = cutile::reference::tolerance(dtype);
    assert!(
        max_rel_err <= tolerance,
        "{op_name} {backend} 对拍失败：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"
    );
}

/// 从设备缓冲读回 `len` 个 f32（小端位模式）。
#[cfg(target_os = "macos")]
fn read_f32(bytes: &[u8], len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let word: [u8; 4] = bytes[idx * 4..idx * 4 + 4]
                .try_into()
                .expect("四个字节一个 f32");
            f32::from_le_bytes(word)
        })
        .collect()
}

/// 把 f32 切片按 LE 字节序写进执行器缓冲（绑槽 + 落位）。
#[cfg(target_os = "macos")]
fn bind_f32(executor: &mut ::engine::metal::MetalExecutor, slot: u32, values: &[f32]) {
    executor
        .bind_bytes(slot, values.len() * 4)
        .expect("f32 缓冲");
    let bytes = executor.buf_mut(slot).expect("f32 可写").as_mut_bytes();
    for (idx, value) in values.iter().enumerate() {
        bytes[idx * 4..idx * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// 把 u8 切片写进执行器缓冲（绑槽 + 拷贝）。
#[cfg(target_os = "macos")]
fn bind_u8(executor: &mut ::engine::metal::MetalExecutor, slot: u32, values: &[u8]) {
    executor.bind_bytes(slot, values.len()).expect("u8 缓冲");
    let bytes = executor.buf_mut(slot).expect("u8 可写").as_mut_bytes();
    bytes.copy_from_slice(values);
}

/// 绑定 16 位浮点缓冲：逐元素取小端两位字节（f16/bf16 共用，位模式由调用方给）。
fn bind_half(
    executor: &mut ::engine::metal::MetalExecutor,
    slot: u32,
    len: usize,
    bits: impl Fn(usize) -> [u8; 2],
) {
    executor.bind_bytes(slot, len * 2).expect("half 缓冲");
    let bytes = executor.buf_mut(slot).expect("half 可写").as_mut_bytes();
    for idx in 0..len {
        bytes[idx * 2..idx * 2 + 2].copy_from_slice(&bits(idx));
    }
}

/// 把单个 f32 标量写进执行器缓冲（4 字节）。
#[cfg(target_os = "macos")]
fn bind_scalar(executor: &mut ::engine::metal::MetalExecutor, slot: u32, value: f32) {
    bind_f32(executor, slot, &[value]);
}

/// 按声明签名把样本缓冲绑进执行器：可写参数绑 `output_len` 字节输出槽，只读参数按 C 类型
/// （f32/u8）逐元素写输入字节。逐点与分块对拍共用这一份绑缓冲样板。
#[cfg(target_os = "macos")]
fn bind_decl_bufs(
    executor: &mut ::engine::metal::MetalExecutor,
    node: &mut engine::exec::Node,
    decl: &prof::decl::OpDecl,
    sample: &nn::model::Sample,
    output_len: usize,
) {
    use nn::sig::CType;

    for (slot, param) in decl
        .signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
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
                CType::F16 => {
                    let values = sample.f16(&param.name).expect("f16 输入");
                    bind_half(executor, slot, values.len(), |i| {
                        values[i].to_bits().to_le_bytes()
                    });
                }
                CType::Bf16 => {
                    let values = sample.bf16(&param.name).expect("bf16 输入");
                    bind_half(executor, slot, values.len(), |i| {
                        values[i].to_bits().to_le_bytes()
                    });
                }
                CType::Int => unreachable!("标量不是缓冲"),
            }
        }
    }
}

/// 提交单个节点并同步，回读 `len` 个 f32 输出。逐点与分块归约对拍的发射/回读样板。
#[cfg(target_os = "macos")]
fn submit_read(
    executor: &mut ::engine::metal::MetalExecutor,
    node: engine::exec::Node,
    decl: &prof::decl::OpDecl,
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
        .buf_mut(output_slot)
        .expect("输出可读")
        .as_mut_bytes();
    read_f32(bytes, len)
}

/// 装配 Metal 后端 + 内置目录 + 临时工作目录；设备不可用时返回 `None` 静默跳过（真机对拍
/// 在无 Metal 的环境里不失败）。
#[cfg(target_os = "macos")]
fn metal_harness(tag: &str) -> Option<(cutile::builtin::Catalogue, cutile::dirs::Dirs)> {
    ensure_backends();
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    if !shard::witness::device_available(catalogue.profile("metal").expect("metal 档案")) {
        return None;
    }
    let dirs =
        cutile::dirs::Dirs::under(std::env::temp_dir().join(format!("llm-metal-{tag}-check")));
    Some((catalogue, dirs))
}

/// CUDA 真机对拍装备：内置数据 + nvidia 档案可用性检查 + 隔离临时目录。
#[cfg(target_os = "windows")]
fn cuda_harness(tag: &str) -> Option<(cutile::builtin::Catalogue, cutile::dirs::Dirs)> {
    ensure_backends();
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("nvidia").cloned().expect("nvidia 档案");
    if !shard::witness::device_available(&profile) {
        return None;
    }
    let dirs =
        cutile::dirs::Dirs::under(std::env::temp_dir().join(format!("llm-cuda-{tag}-check")));
    Some((catalogue, dirs))
}

/// CUDA 真机对拍的公共装备：目录 + nvidia 档案 + 语义表 + 样本 + 声明；`shape` 缺省取算子声明
/// 形状。逐点 / 分块归约 / 分块归一化 / 循环注入四条对拍共用这份装配，只在发射与回读处分叉。
#[cfg(target_os = "windows")]
struct CudaSetup {
    catalogue: cutile::builtin::Catalogue,
    dirs: cutile::dirs::Dirs,
    profile: cutile::prof::Profile,
    model: nn::model::Model,
    sample: nn::model::Sample,
    decl: prof::decl::OpDecl,
    m: usize,
    n: usize,
}

#[cfg(target_os = "windows")]
fn cuda_setup(op_name: &str, tag: &str, shape: Option<(usize, usize)>) -> Option<CudaSetup> {
    let (catalogue, dirs) = cuda_harness(tag)?;
    let profile = catalogue.profile("nvidia").cloned().expect("nvidia 档案");
    let model = nn::registry::model(op_name).expect("语义表");
    let (m, n) = shape.unwrap_or(model.shape);
    let sample = (model.sample)(m, n);
    let decl = catalogue.op(op_name).expect("算子声明").clone();
    Some(CudaSetup {
        catalogue,
        dirs,
        profile,
        model,
        sample,
        decl,
        m,
        n,
    })
}

/// 分块内核对拍的公共装备：语义表 + 样本 + 参考 + 声明 + 分块网格。`len_of` 给出分块轴长度
/// （scan / reduce 从签名里找非缓冲标量，rmsnorm / softmax 直接取 `N`），分块对拍共用这份装配。
#[cfg(target_os = "macos")]
struct ChunkSetup {
    catalogue: cutile::builtin::Catalogue,
    dirs: cutile::dirs::Dirs,
    model: nn::model::Model,
    sample: nn::model::Sample,
    expected: Vec<f32>,
    decl: prof::decl::OpDecl,
    m: usize,
    n: usize,
    len: u32,
    chunk: u32,
    grid_size: u32,
}

/// 从签名里取唯一非缓冲标量作为分块轴长度（Metal 扫描与 CUDA 归约共用）。
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn scalar_len(decl: &prof::decl::OpDecl, sample: &nn::model::Sample) -> u32 {
    decl.signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buf())
        .and_then(|param| sample.scalar(&param.name))
        .expect("长度标量") as u32
}

#[cfg(target_os = "macos")]
fn chunk_setup(
    op_name: &str,
    tag: &str,
    len_of: impl FnOnce(&prof::decl::OpDecl, &nn::model::Sample) -> u32,
) -> Option<ChunkSetup> {
    let (catalogue, dirs) = metal_harness(tag)?;
    let model = nn::registry::model(op_name).expect("语义表");
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
    if shard::toolchain::detect().is_none() {
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
    if shard::toolchain::detect().is_none() {
        return;
    }
    let path = write_dense_model("llm-run-arithmetic.safetensors", &[("blk.0.attn_q", 4, 8)]);
    let file = weights::safetensors::File::open(&path).expect("文件合法");
    let entry = file
        .header()
        .get(&format!("blk.0.attn_q{}", nn::WEIGHT_SUFFIX))
        .expect("条目存在");
    let w = weights::load_f32(entry, &file).expect("f32 条目");
    let x: Vec<f32> = (0..8).map(|idx| (idx % 8) as f32 * 0.125 - 0.5).collect();
    let model = nn::registry::model(nn::Linear::DENSE).expect("稠密语义");
    let sample = nn::model::bind(&model, 8, 4, &x, &w).expect("形状自洽");

    let settings = crate::settings::Settings::load()
        .map_err(|error| error.with("配置"))
        .expect("默认设置");
    let dirs = settings.dirs();
    let catalogue = cutile::builtin::Catalogue::load(&dirs).expect("内置数据");
    let executed = shard::execution::execute(&catalogue, &dirs, nn::Linear::DENSE, "cpu", &sample)
        .expect("执行");
    let want: Vec<f32> = (0..4)
        .map(|row| (0..8).map(|column| w[row * 8 + column] * x[column]).sum())
        .collect();
    assert_eq!(executed.output, want, "真实权重算出来的数必须等于手算");
    assert_eq!(executed.max_rel_err, 0.0, "同一份舍入顺序应当逐位相等");
}

// 子模块名与 `llm-device` 的 extern 名同为 `device`：本模块内指设备 crate 时必须写
// `::device::`，裸 `device::` 会落到这个测试子模块上。
#[path = "device.rs"]
mod device;
