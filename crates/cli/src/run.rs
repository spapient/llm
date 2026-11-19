//! `run` 命令：用权重文件里的**真实数值**算一条前向链，每层一次真机内核（L7 bin 内模块）。
//!
//! 与 `status` 的分工：`status` 回答"这些层至少需要多少流量"（算术，不需要设备），本命令回答
//! "这些层真的算出来是多少"（协商 → 拼接 → emit → 编译 → 执行 → 与标量参考对拍）。两者用同一份
//! 层清单与同一张算子语义表，因此下界与结果不会各说一套。

use diagnostic::{Code, Error, Result};
use nn::Contract as _;
use tool::Report;

/// 打开权重文件；打开失败把命令帧补到错误链上。
fn open_file(weights: &str) -> Result<weights::safetensors::File> {
    weights::safetensors::File::open(weights).map_err(|error| error.with("llm run"))
}

/// 从栈里抽出全部线性层（按值持有，便于把装载错误收敛到一处）；没有投影条目直接失败。
fn collect_layers(stack: &nn::Stack) -> Result<Vec<nn::Linear>> {
    let layers: Vec<nn::Linear> = stack
        .layers()
        .iter()
        .filter_map(|layer| layer.linear().cloned())
        .collect();
    if layers.is_empty() {
        return Err(Error::new(
            Code::Missing,
            "权重文件里没有秩为 2 的投影条目，没有可执行的层",
        )
        .with("llm run"));
    }
    Ok(layers)
}

/// 打开文件、按头部推出层清单并抽出线性层；任一步失败都带着命令帧返回。
fn load_model(weights: &str) -> Result<(weights::safetensors::File, Vec<nn::Linear>)> {
    let file = open_file(weights)?;
    let stack = nn::Stack::from_header(file.header())?;
    let layers = collect_layers(&stack)?;
    Ok((file, layers))
}

/// 初始向量：给了输入就解析，没给就用确定性图案；宽度不符直接报错。
fn start_vector(input: Option<&str>, first: &nn::Linear) -> Result<Vec<f32>> {
    let x = match input {
        Some(text) => parse_input(text)?,
        // 确定性图案：不给输入时也必须可复现（cutile README §12.3 的同一条要求）。
        None => (0..first.in_features())
            .map(|index| (index % 8) as f32 * 0.125 - 0.5)
            .collect(),
    };
    if x.len() != first.in_features() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "输入向量长度 {} 与第一层的输入宽度 {} 不符",
                x.len(),
                first.in_features()
            ),
        )
        .with("llm run"));
    }
    Ok(x)
}

/// 单层执行的两种结局：链继续往下传新向量，或在某个边界停下并说明原因。
enum Step {
    Continue { output: Vec<f32>, row: String },
    Stopped(String),
}

/// 取层对应的权重条目；条目缺失是声明不合法，按 [`Code::Missing`] 报。
fn weight_entry<'a>(
    file: &'a weights::safetensors::File,
    name: &str,
) -> Result<&'a weights::Entry> {
    file.header()
        .get(&format!("{name}{}", nn::WEIGHT_SUFFIX))
        .ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!(
                    "层 `{name}` 的权重条目 `{name}{}` 不存在",
                    nn::WEIGHT_SUFFIX
                ),
            )
            .with("llm run")
        })
}

/// 算子语义表 + 形状样本：没有语义表的算子无法对拍。
fn build_sample(
    operator: &str,
    m: usize,
    n: usize,
    x: &[f32],
    w: &[f32],
) -> Result<op::model::Sample> {
    let model = op::model(operator).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("算子 `{operator}` 没有语义表，无法构造样本"),
        )
        .with("llm run")
    })?;
    op::model::bind(&model, m, n, x, w)
}

/// 跑一条内核并回收结果：协商 → 拼接 → emit → 编译 → 执行 → 对拍。
fn run_variant(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    operator: &str,
    device: &str,
    sample: &op::model::Sample,
) -> Result<cutile::execution::Execution> {
    cutile::execution::execute(catalogue, dirs, operator, device, sample)
}

/// 一层的报告行：形状、算子、变体、tile、误差、片段行数与缓存键。
fn variant_row(
    executed: &cutile::execution::Execution,
    m: usize,
    n: usize,
    operator: &str,
) -> String {
    format!(
        "{m} → {n} · {operator} · 变体 {} · tile {} · 误差 {:.1e} · 片段 {} 行 · key {}",
        executed.variant,
        executed.tile,
        executed.max_rel_err,
        executed.fragment.lines,
        executed.key
    )
}

/// 算子取值的两种结局：宽度或编码不成立就停下，否则给出算子名。
enum OpGate<'a> {
    Ready(&'a str),
    Stop(String),
}

/// 校验输入宽度并取算子名；不成立时给出可写进报告的停止原因。
fn layer_operator<'a>(layer: &'a nn::Linear, x: &[f32]) -> OpGate<'a> {
    let name = layer.name();
    let m = layer.in_features();
    if x.len() != m {
        return OpGate::Stop(format!(
            "`{name}` 的输入宽度是 {m}，当前向量是 {} 个：本版只跑一条连续链",
            x.len()
        ));
    }
    match layer.format().operator() {
        Some(operator) => OpGate::Ready(operator),
        None => OpGate::Stop(format!("`{name}`：{}", layer.format().unpriced_reason())),
    }
}

/// 权重装载结局：形状相符给权重向量，不符给实际元素个数。
enum WeightCheck {
    Fit(Vec<f32>),
    Mismatch(usize),
}

/// 读权重条目并核对形状；条目缺失是错误，形状不符是链上停止。
fn layer_weights(
    file: &weights::safetensors::File,
    layer: &nn::Linear,
    n: usize,
    m: usize,
) -> Result<WeightCheck> {
    let entry = weight_entry(file, layer.name())?;
    let w = weights::load_f32(entry, file)?;
    if w.len() != n * m {
        return Ok(WeightCheck::Mismatch(w.len()));
    }
    Ok(WeightCheck::Fit(w))
}

/// 构造样本并跑一条内核，回收新向量与报告行。
fn evaluate(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    operator: &str,
    device: &str,
    shape: (usize, usize),
    x: &[f32],
    w: &[f32],
) -> Result<(Vec<f32>, String)> {
    let (m, n) = shape;
    let sample = build_sample(operator, m, n, x, w)?;
    let executed = run_variant(catalogue, dirs, operator, device, &sample)?;
    let row = variant_row(&executed, m, n, operator);
    Ok((executed.output, row))
}

/// 前向链上走一层：宽度、编码、权重形状、语义表、内核执行任一不成立就返回停止原因。
fn advance(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    device: &str,
    file: &weights::safetensors::File,
    layer: &nn::Linear,
    x: &[f32],
) -> Result<Step> {
    let name = layer.name();
    let (m, n) = (layer.in_features(), layer.out_features());
    let operator = match layer_operator(layer, x) {
        OpGate::Ready(operator) => operator,
        OpGate::Stop(reason) => return Ok(Step::Stopped(reason)),
    };
    let weights = match layer_weights(file, layer, n, m)? {
        WeightCheck::Fit(weights) => weights,
        WeightCheck::Mismatch(got) => {
            return Ok(Step::Stopped(format!(
                "`{name}`：条目是 {got} 个元素，而形状 {n}×{m} 需要 {}",
                n * m
            )));
        }
    };
    let (output, row) = evaluate(catalogue, dirs, operator, device, (m, n), x, &weights)?;
    Ok(Step::Continue { output, row })
}

/// 输出向量一行：只列前 4 个，避免长向量把报告撑满。
fn output_row(x: &[f32]) -> String {
    format!(
        "{} 个 f32：前 4 个 {}",
        x.len(),
        x.iter()
            .take(4)
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join("、")
    )
}

/// 收尾：一层都没跑出来是错误；否则把输出向量与停止原因排成行并渲染。
fn finish(
    report: Report,
    x: &[f32],
    stopped: Option<String>,
    executed_layers: usize,
) -> Result<()> {
    if executed_layers == 0 {
        let reason = stopped.unwrap_or_else(|| "没有可执行的层".to_string());
        return Err(Error::new(Code::Missing, reason).with("llm run"));
    }
    let report = report.row("输出    向量", output_row(x));
    let report = match stopped {
        Some(reason) => report.row("停下    原因", reason),
        None => report,
    };
    print!("{}", report.render());
    Ok(())
}

/// 配置与档案目录：设置装载失败只补命令帧，目录由设置投影出来。
fn prepare() -> Result<(cutile::dirs::Dirs, cutile::builtin::Catalogue)> {
    let settings = crate::settings::Settings::load().map_err(|error| error.with("配置"))?;
    let dirs = settings.dirs();
    let catalogue = cutile::builtin::Catalogue::load(&dirs)?;
    Ok((dirs, catalogue))
}

/// 沿层清单跑前向链；返回停止原因与已执行层数，报告行就地追加进 `report`。
fn drive(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    device: &str,
    file: &weights::safetensors::File,
    layers: &[nn::Linear],
    x: &mut Vec<f32>,
    report: Report,
) -> Result<(Report, Option<String>, usize)> {
    let mut report = report;
    let mut stopped: Option<String> = None;
    let mut executed = 0usize;
    for layer in layers {
        match advance(catalogue, dirs, device, file, layer, x)? {
            Step::Continue { output, row } => {
                report = report.row(format!("层       {}", layer.name()), row);
                *x = output;
                executed += 1;
            }
            Step::Stopped(reason) => {
                stopped = Some(reason);
                break;
            }
        }
    }
    Ok((report, stopped, executed))
}

/// 三个刻意的边界，都在输出里如实写出来而不是悄悄降级：
///
/// - 只跑**连续的一条链**：某一层的输入宽度与当前向量不等时停下，不再猜"是不是该换一份输入"；
/// - 只跑有算子的编码：没有算子的格式（f16 / bf16 / i8 / 分组不明的 4 位打包）停下并说明缺什么；
/// - 权重按需读：每层只读它自己那条条目，不把整份文件搬进内存。
pub(crate) fn run(weights: &str, input: Option<&str>, device: Option<&str>) -> Result<()> {
    // 配置解析在命令行层、档案装载在核心里；本函数只决定"跑哪些层、用什么数据"。
    let device = device.unwrap_or("auto");
    let (dirs, catalogue) = prepare()?;
    let (file, layers) = load_model(weights)?;
    let mut x = start_vector(input, &layers[0])?;

    let report = Report::new("llm —— 用真实权重算一次（每层一条内核）")
        .row("输入    向量", format!("{} 个 f32（{}）", x.len(), device));
    let (report, stopped, executed_layers) =
        drive(&catalogue, &dirs, device, &file, &layers, &mut x, report)?;

    finish(report, &x, stopped, executed_layers)
}

/// `--input` 单段：裁剪后解析为有限 f32；空段由 [`parse_input`] 跳过。
fn parse_word(word: &str) -> Result<f32> {
    let value: f32 = word.parse().map_err(|_| {
        Error::new(Code::Invalid, format!("`--input` 里的 `{word}` 不是 f32")).with("llm run")
    })?;
    if !value.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("`--input` 里的 `{word}` 不是有限数"))
                .with("llm run"),
        );
    }
    Ok(value)
}

/// `--input a,b,c` → 向量；空段被跳过，段内空白被裁剪。
///
/// 解析放在组装点而不是 `cli`：文法层只搬运字符串，`f32` 的语法与报错措辞属于要使用它的地方。
fn parse_input(text: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for word in text.split(',') {
        let word = word.trim();
        if word.is_empty() {
            continue;
        }
        values.push(parse_word(word)?);
    }
    if values.is_empty() {
        return Err(Error::new(Code::Invalid, "`--input` 里没有数值").with("llm run"));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let executed =
            cutile::execution::execute(&catalogue, &dirs, nn::Linear::DENSE, "cpu", &sample)
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

    /// 两趟（带归约）真机对拍：`rmsnorm_f32` 走串行 lowering（单线程网格、`x`/`gamma`/`y` 三缓冲
    /// 原样透传），在 Metal 上算出均方根归一化并回读，与标量参考对拍。
    #[cfg(target_os = "macos")]
    #[test]
    fn a_two_pass_op_runs_on_metal_and_matches_the_reference() {
        op_matches_reference("rmsnorm_f32");
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
        use engine::exec::Executor as _;
        use op::sig::CType;

        let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
        let profile = catalogue.profile("metal").cloned().expect("metal 档案");
        if !cutile::witness::device_available(&profile) {
            return;
        }
        let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("llm-metal-serial-check"));

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

        // 派发网格：`[grid]` 声明的各维标量取样本值；缺省单线程串行。
        let mut grid = [1u32, 1, 1];
        if let Some(decl_grid) = decl.grid.as_ref() {
            for (dim, (_, scalar)) in decl_grid.dims.iter().enumerate() {
                grid[dim] = sample
                    .scalar(scalar)
                    .expect("网格维标量")
                    .try_into()
                    .expect("网格维落 u32");
            }
        }
        let launch = engine::exec::Launch::new(grid, [1, 1, 1]);
        let mut node = engine::exec::Node::new(op_name, launch);
        for (slot, param) in decl
            .signature
            .params()
            .iter()
            .filter(|param| param.ctype.is_buffer())
            .enumerate()
        {
            let slot = slot as u32;
            if param.mutable {
                executor
                    .bind_bytes(slot, expected.len() * 4)
                    .expect("输出缓冲");
                node.bind_output(slot);
            } else {
                node.bind_input(slot);
                match param.ctype {
                    CType::F32 => {
                        let values = sample.f32(&param.name).expect("f32 输入");
                        executor
                            .bind_bytes(slot, values.len() * 4)
                            .expect("输入缓冲");
                        let bytes = executor.buffer_mut(slot).expect("输入可写").as_mut_bytes();
                        for (index, value) in values.iter().enumerate() {
                            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
                        }
                    }
                    CType::U8 => {
                        let values = sample.u8(&param.name).expect("u8 输入");
                        executor.bind_bytes(slot, values.len()).expect("输入缓冲");
                        let bytes = executor.buffer_mut(slot).expect("输入可写").as_mut_bytes();
                        for (index, value) in values.iter().enumerate() {
                            bytes[index] = *value;
                        }
                    }
                    CType::Int => unreachable!("标量不是缓冲"),
                }
            }
        }

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
        let output: Vec<f32> = (0..expected.len())
            .map(|index| {
                let chunk: [u8; 4] = bytes[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("四个字节一个 f32");
                f32::from_le_bytes(chunk)
            })
            .collect();

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
        use engine::exec::Executor as _;

        let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
        let profile = catalogue.profile("metal").cloned().expect("metal 档案");
        if !cutile::witness::device_available(&profile) {
            return;
        }
        let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("llm-metal-chunk-check"));

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
        for (slot, param) in decl
            .signature
            .params()
            .iter()
            .filter(|param| param.ctype.is_buffer())
            .enumerate()
        {
            let slot = slot as u32;
            if param.mutable {
                executor
                    .bind_bytes(slot, grid_size as usize * 4)
                    .expect("输出缓冲");
                node.bind_output(slot);
            } else {
                node.bind_input(slot);
                let values = sample.f32(&param.name).expect("f32 输入");
                executor
                    .bind_bytes(slot, values.len() * 4)
                    .expect("输入缓冲");
                let bytes = executor.buffer_mut(slot).expect("输入可写").as_mut_bytes();
                for (index, value) in values.iter().enumerate() {
                    bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }

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
        let partials: Vec<f32> = (0..grid_size as usize)
            .map(|index| {
                let bytes: [u8; 4] = bytes[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("四个字节一个 f32");
                f32::from_le_bytes(bytes)
            })
            .collect();

        // 主机折叠部分结果（sum / max）；参考用 f64 折叠输入，理由见函数注释。
        let input = sample.f32("a").expect("归约输入");
        let (output, want) = if op_name.contains("_max_") {
            let out = partials.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let want = input
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| acc.max(f64::from(x)));
            (vec![out], vec![want as f32])
        } else {
            let out = partials.iter().sum::<f32>();
            let want = input.iter().fold(0.0f64, |acc, &x| acc + f64::from(x));
            (vec![out], vec![want as f32])
        };

        let max_rel_err = cutile::reference::max_rel_err(&output, &want);
        let tolerance = cutile::reference::tolerance(model.dtype);
        assert!(
            max_rel_err <= tolerance,
            "{op_name} 分块归约对拍失败：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"
        );
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

        let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
        let profile = catalogue.profile("metal").cloned().expect("metal 档案");
        if !cutile::witness::device_available(&profile) {
            return;
        }
        let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("llm-metal-scan-check"));

        let model = op::model(op_name).expect("语义表");
        let (m, n) = model.shape;
        let sample = (model.sample)(m, n);
        let expected = (model.reference)(&sample).expect("标量参考");
        let decl = catalogue.op(op_name).expect("算子声明");
        let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size as u32;
        let len = decl
            .signature
            .params()
            .iter()
            .find(|param| !param.ctype.is_buffer())
            .and_then(|param| sample.scalar(&param.name))
            .expect("扫描长度标量") as u32;
        let grid_size = len.div_ceil(chunk);
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
        executor.register(op_name, &local_msl).expect("注册局部扫描内核");
        let correct_name = format!("{op_name}_correct");
        executor
            .register(&correct_name, &correct_msl)
            .expect("注册校正内核");

        // 槽位约定（同 buffer_decls 的签名序）：0 = 只读输入 x，1 = 可写输出 y。
        let input = sample.f32("x").expect("扫描输入");
        executor.bind_bytes(0, input.len() * 4).expect("输入缓冲");
        executor.bind_bytes(1, expected.len() * 4).expect("输出缓冲");
        {
            let bytes = executor.buffer_mut(0).expect("输入可写").as_mut_bytes();
            for (index, value) in input.iter().enumerate() {
                bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }

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
            (0..expected.len())
                .map(|index| {
                    let bytes: [u8; 4] = bytes[index * 4..index * 4 + 4]
                        .try_into()
                        .expect("四个字节一个 f32");
                    f32::from_le_bytes(bytes)
                })
                .collect::<Vec<_>>()
        };
        let mut carry = vec![0.0f64; grid_size as usize];
        for g in 1..grid_size as usize {
            let prev_start = (g as u32 - 1) * chunk;
            let prev_len = chunk.min(len - prev_start);
            let tail = local[(prev_start + prev_len - 1) as usize] as f64;
            carry[g] = decay.powf(prev_len as f64) * carry[g - 1] + tail;
        }

        // 阶段 3：校正（c → y 就地累加）。槽 0 复用为携带态缓冲。
        executor
            .bind_bytes(0, grid_size as usize * 4)
            .expect("携带态缓冲");
        {
            let bytes = executor.buffer_mut(0).expect("携带态可写").as_mut_bytes();
            for (index, value) in carry.iter().enumerate() {
                bytes[index * 4..index * 4 + 4].copy_from_slice(&(*value as f32).to_le_bytes());
            }
        }
        let mut node = engine::exec::Node::new(&correct_name, launch);
        node.bind_input(0);
        node.bind_output(1);
        let mut graph = engine::exec::Graph::new();
        graph.push(node);
        executor.submit(&graph, stream).expect("提交校正");
        executor.sync(stream).expect("同步校正");

        let output = {
            let bytes = executor.buffer_mut(1).expect("输出可读").as_mut_bytes();
            (0..expected.len())
                .map(|index| {
                    let bytes: [u8; 4] = bytes[index * 4..index * 4 + 4]
                        .try_into()
                        .expect("四个字节一个 f32");
                    f32::from_le_bytes(bytes)
                })
                .collect::<Vec<_>>()
        };

        let max_rel_err = cutile::reference::max_rel_err(&output, &expected);
        let tolerance = cutile::reference::tolerance(model.dtype);
        assert!(
            max_rel_err <= tolerance,
            "{op_name} 分块扫描对拍失败：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"
        );
    }

    /// 三趟 softmax（max 归约 + exp 求和 + 归一化）走串行兜底，真机对拍。
    #[cfg(target_os = "macos")]
    #[test]
    fn a_softmax_op_runs_on_metal_and_matches_the_reference() {
        op_matches_reference("softmax_f32");
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
}
