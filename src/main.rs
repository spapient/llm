//! 组装点：把各层连起来，**不定义任何新类型**（`docs/ARCHITECTURE.md` 的 L7）。
//!
//! 它做两件事：`status` 构造每一层的最小实例并打印摘要（"依赖图真的能编译、各层公共 API 真的
//! 可用"的端到端证据），`run` 用权重文件里的真实数值把各层的算子真的执行一遍（协商 → 拼接 →
//! emit → 编译 → 执行 → 与标量参考对拍）——依赖方向只是评审约定，能不能跑只能靠这里真正组装。
//!
//! `serve` / `train` 仍需 P1 接入；在那之前，任何新类型都应该长在它所属的层里，而不是长在这里。

use diagnostic::{Code, Error, Result};
use nn::Contract as _;
use nn::LmHead;
use std::process::ExitCode;
use tokenizer::Tokenizer as _;
use tool::Report;
use weights::Loader as _;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let invocation = match cli::Invocation::parse(&args) {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("✗ {error}");
            eprint!("{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };
    dispatch(invocation)
}

/// 命令分派：每个命令只接一个薄壳函数，退出码汇总统一走 [`finish`]。
fn dispatch(invocation: cli::Invocation) -> ExitCode {
    match invocation.command() {
        cli::Command::Help => {
            print!("{}", cli::USAGE);
            ExitCode::SUCCESS
        }
        cli::Command::Status { weights } => finish(status(weights.as_deref())),
        cli::Command::Run {
            weights,
            input,
            device,
        } => finish(run(weights, input.as_deref(), device.as_deref())),
        cli::Command::Verify { .. } | cli::Command::Bench { .. } => audit_note(),
        cli::Command::Train {
            lr,
            warmup,
            steps,
            clip,
            accumulation,
            load,
        } => finish(train_run(
            lr.as_deref(),
            warmup.as_deref(),
            steps.as_deref(),
            clip.as_deref(),
            accumulation.as_deref(),
            load.as_deref(),
        )),
    }
}

/// 命令结果 → 退出码：成功走 0，失败按诊断码分级打印。
fn finish(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("✗ {error}");
            ExitCode::from(error.code().exit_code())
        }
    }
}

/// 组装点不复制审计实现：这两个命令今天由 `cutile` 二进制提供。
fn audit_note() -> ExitCode {
    eprintln!("✗ 该命令当前由 `cargo run --bin cutile -- ...` 提供（`llm` 只负责组装）");
    ExitCode::from(2)
}

/// 每台设备的设备侧代码这次构建带了什么：只有源码，还是连镜像一起带上了。
///
/// 这一行也是镜像真的进了二进制的证据：`device-image` 关着时它全是"源码"，打开后 `metal`
/// 变成 `metallib N B/M 个符号`（`M` 是编译器数出来的符号数）。
fn device_images() -> String {
    let parts: Vec<String> = prim::devices()
        .iter()
        .map(|device| match &device.image {
            Some(image) => format!(
                "{}={} {} B/{} 个符号",
                device.vendor,
                image.format(),
                image.len(),
                image.symbols()
            ),
            None => format!("{}=源码", device.vendor),
        })
        .collect();
    parts.join("、")
}

/// 逐层构造一个最小实例：任何一层不可用都会让这个函数失败。
///
/// `weights` 是 `--weights <文件>` 给的 safetensors 路径：给了就读真实条目，不给就只用声明层。
fn status(weights: Option<&str>) -> Result<()> {
    let ops = op::builtin();
    let profiles = prof::names();
    let mut report = Report::new("llm —— 组装点（各层最小实例）")
        .row(
            "声明层  op",
            format!(
                "{} 个算子（{}）",
                ops.len(),
                tool::join_names(ops.iter().map(op::Op::name))
            ),
        )
        .row(
            "档案层  prof",
            format!(
                "{} 份设备档案（{}）",
                profiles.len(),
                tool::join_names(profiles.iter().copied())
            ),
        )
        .row(
            "片段层  prim",
            format!(
                "{} 台设备（{}）；设备侧代码：{}",
                prim::devices().len(),
                tool::join_names(prim::devices().iter().map(|device| device.vendor)),
                device_images(),
            ),
        );

    report = report.row(
        "层定义  nn",
        format!(
            "{} → {:?}，算子 {}",
            LmHead.name(),
            LmHead.output().dims(),
            LmHead.spca().ops().join("、")
        ),
    );

    // 层的性能下界：由算子语义推出（不是手写常数），与内核实测对账时才谈得上"优化了多少"。
    let projection = nn::Linear::new("blk.0.attn_q", 4096, 4096)?;
    let floor = projection.bounds()?.ok_or_else(|| {
        diagnostic::Error::new(diagnostic::Code::Internal, "Linear 有形状就应当能计价")
            .with("llm status")
    })?;
    report = report.row(
        "下界    层",
        format!("{} 4096 → ≥ {}", projection.name(), floor_row(&floor)),
    );

    // 模型下界：层的清单来自真实权重文件（或声明层），总和由 `Bounds::plus` 相加——
    // 与层下界是同一份语义，因此这里不会多出第二套流量公式。
    let (model, weights_row) = load_model(weights)?;
    report = report.row("权重    weights", weights_row);
    // 无法计价的层不会被悄悄算成 0：它们与"未覆盖"分开列出来，读者才知道这个和覆盖了什么。
    let floor = model.floor();
    let mut summary = format!(
        "{} 层，每 token ≥ {}",
        model.layers().len(),
        floor_row(&floor.total())
    );
    if !floor.unpriced().is_empty() {
        summary.push_str(&format!("；无法计价 {} 层", floor.unpriced().len()));
    }
    if !model.uncovered().is_empty() {
        summary.push_str(&format!(
            "；未覆盖 {} 条：{}",
            model.uncovered().len(),
            tool::join_names(model.uncovered().iter().map(String::as_str))
        ));
    }
    report = report.row("下界    模型", summary);
    if !floor.unpriced().is_empty() {
        // 为什么算不出来，逐条写出来：只说"无法计价 2 层"读者无法判断是格式、形状还是别的。
        let reasons = floor
            .unpriced()
            .iter()
            .map(nn::Unpriced::to_string)
            .collect::<Vec<_>>()
            .join("；");
        report = report.row("无法计价 nn", reasons);
    }

    let tensor = tensor::Tensor::new(tensor::DType::F32, tensor::Shape::new([2, 2]), &[0u8; 16])?;
    report = report.row(
        "张量    tensor",
        format!(
            "{:?} {:?} = {} 元素，连续布局：{}",
            tensor.dtype(),
            tensor.shape().dims(),
            tensor.numel(),
            tensor.is_contiguous()
        ),
    );

    let buffer = vec![0.5f32, 0.25, 0.125, 0.0625];
    let view = tensor::View::new(tensor::Shape::new([2, 2]), &buffer)?;
    report = report.row(
        "视图    tensor::View",
        format!(
            "{:?} {:?}，第 1 行和 = {:.3}（借用调用方 {} 个元素，不复制）",
            view.dtype(),
            view.shape().dims(),
            view.row(1)
                .map(|row| row.iter().sum::<f32>())
                .unwrap_or(0.0),
            view.numel()
        ),
    );

    let schedule = train::Schedule::new(10, 100)?;
    report = report.row(
        "训练    train",
        format!(
            "warmup {}/{} 步，第 0 步 lr = {:.3}",
            schedule.warmup(),
            schedule.total(),
            schedule.lr(1.0, 0)
        ),
    );

    let route = serve::Route::parse("POST /v1/completions")?;
    let tokens = tokenizer::Bytes.encode("hi")?;
    report = report.row(
        "服务    serve",
        format!(
            "{} {}，提示 \"hi\" → {} 个 token",
            route.method(),
            route.path(),
            tokens.len()
        ),
    );

    let mut scheduler = engine::Scheduler::new(status_pager()?);
    let sequence = engine::Sequence::new(0, &[0; 8], 4, 16, engine::exec::Stream::new(0))?;
    let response = serve::admit(&mut scheduler, &sequence)?;
    report = report.row(
        "调度    engine",
        format!(
            "准入 {} 个序列，占用 {} 页（页宽 {}），响应 {}",
            scheduler.running(),
            scheduler.cache().in_use(),
            scheduler.cache().page_size(),
            response.status().code()
        ),
    );

    print!("{}", report.render());
    Ok(())
}

/// `status` 用的 KV 页池：这里没有模型，行宽取 1——页里的数不参与账目。
fn status_pager() -> Result<engine::kv::Pager> {
    engine::kv::Pager::new(engine::kv::Cache::new(4, 16)?, 1)
}

/// `run`：用权重文件里的**真实数值**算一条前向链，每层一次真机内核。
///
/// 与 `status` 的分工：`status` 回答"这些层至少需要多少流量"（算术，不需要设备），本命令回答
/// "这些层真的算出来是多少"（协商 → 拼接 → emit → 编译 → 执行 → 与标量参考对拍）。两者用同一份
/// 层清单与同一张算子语义表，因此下界与结果不会各说一套。
///
/// 内核只收三个目录：把解析好的设置投影成那份纯数据（`cutile` 不读环境变量，ADR 0023）。
fn kernel_dirs(settings: &cli::settings::Settings) -> cutile::dirs::Dirs {
    cutile::dirs::Dirs::new(
        settings.cache_dir(),
        settings.work_dir(),
        settings.data_root().map(std::path::Path::to_path_buf),
    )
}

/// 三个刻意的边界，都在输出里如实写出来而不是悄悄降级：
///
/// - 只跑**连续的一条链**：某一层的输入宽度与当前向量不等时停下，不再猜"是不是该换一份输入"；
/// - 只跑有算子的编码：没有算子的格式（f16 / bf16 / i8 / 分组不明的 4 位打包）停下并说明缺什么；
/// - 权重按需读：每层只读它自己那条条目，不把整份文件搬进内存。
fn run(weights: &str, input: Option<&str>, device: Option<&str>) -> Result<()> {
    // 配置解析在命令行层、档案装载在核心里；本函数只决定"跑哪些层、用什么数据"。
    let settings = cli::settings::Settings::load().map_err(|error| error.with("配置"))?;
    let dirs = kernel_dirs(&settings);
    let catalogue = cutile::builtin::Catalogue::load(&dirs)?;
    let device = device.unwrap_or("auto");
    let file = weights::safetensors::File::open(weights).map_err(|error| error.with("llm run"))?;
    let stack = nn::Stack::from_header(file.header())?;
    let layers: Vec<&nn::Linear> = stack
        .layers()
        .iter()
        .filter_map(|layer| layer.linear())
        .collect();
    let first = layers.first().ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("`{weights}` 里没有秩为 2 的投影条目，没有可执行的层"),
        )
        .with("llm run")
    })?;

    let mut x = match input {
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

    let mut report = Report::new("llm —— 用真实权重算一次（每层一条内核）")
        .row("输入    向量", format!("{} 个 f32（{}）", x.len(), device));
    let mut stopped: Option<String> = None;
    let mut executed_layers = 0usize;

    for layer in &layers {
        let name = layer.name();
        let (m, n) = (layer.in_features(), layer.out_features());
        if x.len() != m {
            stopped = Some(format!(
                "`{name}` 的输入宽度是 {m}，当前向量是 {} 个：本版只跑一条连续链",
                x.len()
            ));
            break;
        }
        let Some(operator) = layer.format().operator() else {
            stopped = Some(format!("`{name}`：{}", layer.format().unpriced_reason()));
            break;
        };
        let entry = file
            .header()
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
            })?;
        let w = weights::load_f32(entry, &file)?;
        if w.len() != n * m {
            stopped = Some(format!(
                "`{name}`：条目是 {} 个元素，而形状 {n}×{m} 需要 {}",
                w.len(),
                n * m
            ));
            break;
        }
        let model = op::model(operator).ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!("算子 `{operator}` 没有语义表，无法构造样本"),
            )
            .with("llm run")
        })?;
        let sample = op::model::bind(&model, m, n, &x, &w)?;
        let executed = cutile::execution::execute(&catalogue, &dirs, operator, device, &sample)?;
        report = report.row(
            format!("层       {name}"),
            format!(
                "{m} → {n} · {operator} · 变体 {} · tile {} · 误差 {:.1e} · 片段 {} 行 · key {}",
                executed.variant,
                executed.tile,
                executed.max_rel_err,
                executed.fragment.lines,
                executed.key
            ),
        );
        x = executed.output;
        executed_layers += 1;
    }

    if executed_layers == 0 {
        let reason = stopped.unwrap_or_else(|| "没有可执行的层".to_string());
        return Err(Error::new(Code::Missing, reason).with("llm run"));
    }
    report = report.row(
        "输出    向量",
        format!(
            "{} 个 f32：前 4 个 {}",
            x.len(),
            x.iter()
                .take(4)
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
                .join("、")
        ),
    );
    if let Some(reason) = stopped {
        // 停下来不是错误：前 N 层的结果是真的，后面的层只是这一版不支持。，
        report = report.row("停下    原因", reason);
    }
    print!("{}", report.render());
    Ok(())
}

/// `train` 分派：把 cli 搬下来的字符串解析成 [`train::TrainConfig`]，`--load` 从快照恢复事实与矩状态。
///
/// 前向/反传会话循环在 `train::Session`（夹具级闭环见 train 测试）；本组装点不复制模型装载，
/// 真实权重通路仍属未接入范围。
fn train_run(
    lr: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    clip: Option<&str>,
    accumulation: Option<&str>,
    load: Option<&str>,
) -> Result<()> {
    let config = build_train_config(lr, warmup, steps, clip, accumulation)?;
    match load {
        Some(path) => report_recovered(path),
        None => {
            report_train_config(&config);
            Ok(())
        }
    }
}

/// 把 cli 搬下来的字符串解析成 [`train::TrainConfig`]；数值口径归组装点。
fn build_train_config(
    lr: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    clip: Option<&str>,
    accumulation: Option<&str>,
) -> Result<train::TrainConfig> {
    let (learning_rate, clip_norm) = parse_lr_clip(lr, clip)?;
    let (warmup_steps, max_steps, accum) = parse_counts(warmup, steps, accumulation)?;
    train::TrainConfig::new(learning_rate, warmup_steps, max_steps, clip_norm, accum)
        .map_err(|error| error.with("llm train"))
}

/// 学习率与裁剪阈值：两个 f32 选项。
fn parse_lr_clip(lr: Option<&str>, clip: Option<&str>) -> Result<(f32, Option<f32>)> {
    let learning_rate = parse_optional_f32(lr, "--lr")?.unwrap_or(1e-3);
    let clip_norm = parse_optional_f32(clip, "--clip")?;
    Ok((learning_rate, clip_norm))
}

/// warmup、总步数与累积窗口：三个无符号整数选项。
fn parse_counts(
    warmup: Option<&str>,
    steps: Option<&str>,
    accumulation: Option<&str>,
) -> Result<(u64, u64, u32)> {
    let warmup_steps = parse_optional_uint(warmup, "--warmup")?.unwrap_or(0);
    let max_steps = parse_optional_uint(steps, "--steps")?.unwrap_or(1);
    let accum = parse_optional_uint(accumulation, "--accumulation")?.unwrap_or(1);
    Ok((warmup_steps, max_steps, accum))
}

/// `--load`：从快照读出恢复事实、条目表与矩状态并排版汇报。
fn report_recovered(path: &str) -> Result<()> {
    let snapshot = train::snapshot::Snapshot::read(std::path::Path::new(path))?;
    let facts = snapshot.facts();
    let moments = snapshot.moments();
    let report = Report::new("llm —— 从检查点恢复")
        .row("步数", format!("第 {} 步", facts.step()))
        .row(
            "批形状",
            format!("{} × {}", facts.batch_rows(), facts.batch_cols()),
        )
        .row(
            "层 / 条目",
            format!(
                "{} 层，{} 条权重",
                facts.layers(),
                snapshot.weights().entries().len()
            ),
        )
        .row(
            "矩状态",
            format!(
                "{} 槽，优化器第 {} 步",
                moments.slots().len(),
                moments.step()
            ),
        );
    print!("{}", report.render());
    Ok(())
}

/// 无 `--load`：排版配置；会话闭环由 `train::Session` 装配。
fn report_train_config(config: &train::TrainConfig) {
    let report = Report::new("llm —— 训练配置（会话闭环由 train::Session 装配）")
        .row("学习率", format!("{}", config.learning_rate))
        .row(
            "调度",
            format!("warmup {}/{} 步", config.warmup_steps, config.max_steps),
        )
        .row(
            "裁剪",
            config
                .clip_norm
                .map_or_else(|| "不裁剪".to_string(), |value| format!("{value}")),
        )
        .row("累积", format!("{} micro-batch", config.accumulation()));
    print!("{}", report.render());
}

/// 可选 f32：缺失返回 `None`；非法或非有限数是 [`Code::Invalid`]。
fn parse_optional_f32(text: Option<&str>, flag: &str) -> Result<Option<f32>> {
    let Some(text) = text else {
        return Ok(None);
    };
    let value: f32 = text.parse().map_err(|_| {
        Error::new(Code::Invalid, format!("`{flag}` 的 `{text}` 不是 f32")).with("llm train")
    })?;
    if !value.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("`{flag}` 的 `{text}` 不是有限数")).with("llm train"),
        );
    }
    Ok(Some(value))
}

/// 可选无符号整数：解析在组装点，文法层只搬字符串。
fn parse_optional_uint<T>(text: Option<&str>, flag: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(text) = text else {
        return Ok(None);
    };
    let value = text.parse().map_err(|error| {
        Error::new(
            Code::Invalid,
            format!("`{flag}` 的 `{text}` 不是整数：{error}"),
        )
        .with("llm train")
    })?;
    Ok(Some(value))
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
        let value: f32 = word.parse().map_err(|_| {
            Error::new(Code::Invalid, format!("`--input` 里的 `{word}` 不是 f32")).with("llm run")
        })?;
        if !value.is_finite() {
            return Err(
                Error::new(Code::Invalid, format!("`--input` 里的 `{word}` 不是有限数"))
                    .with("llm run"),
            );
        }
        values.push(value);
    }
    if values.is_empty() {
        return Err(Error::new(Code::Invalid, "`--input` 里没有数值").with("llm run"));
    }
    Ok(values)
}

/// 一条下界的统一写法：读、写分开给，合计与强度作为派生量。
///
/// 两个方向分开是 [`nn::Bounds`] 的口径：把"读受限还是写受限"折叠成一个总量之后，任何关于
/// 这一层卡在哪一侧的判断都失去了依据。
fn floor_row(bounds: &nn::Bounds) -> String {
    format!(
        "{:.2} MB 读 / {:.2} MB 写（合计 {:.2} MB）/ {:.3} MFLOP，强度 {:.2} flop/byte",
        bounds.read() as f64 / 1e6,
        bounds.write() as f64 / 1e6,
        bounds.bytes() as f64 / 1e6,
        bounds.flops() / 1e6,
        bounds.intensity()
    )
}

/// 模型：给了权重文件就读真实条目，没给就用声明的一对投影。
///
/// 两条路都返回 [`nn::Stack`]，所以"层下界之和"只有一份算法；区别只在层的清单从哪来。返回的
/// 第二项是权重那一行的文字——把"读到了什么"与"算出了什么"分开，读者才能判断下界覆盖了什么。
fn load_model(weights: Option<&str>) -> Result<(nn::Stack, String)> {
    let Some(path) = weights else {
        let model = nn::Stack::new(vec![
            nn::Linear::new("blk.0.attn_q", 4096, 4096)?,
            nn::Linear::new("blk.0.attn_o", 4096, 4096)?,
        ])?;
        return Ok((
            model,
            "未提供权重文件：只用声明层（`--weights <file>` 读 safetensors）".to_string(),
        ));
    };
    let archive =
        weights::safetensors::File::open(path).map_err(|error| error.with("llm status"))?;
    let model = nn::Stack::from_header(archive.header())?;
    let row = format!(
        "safetensors：{} 条条目，数据段 {} 字节，只读头部 {} / {} 字节（{path}）；{}",
        archive.header().entries().len(),
        archive.header().data_bytes(),
        archive.header_len(),
        archive.file_len(),
        read_sample(&archive)?
    );
    Ok((model, row))
}

/// 一条样例最多读多少字节：样例只为证明"头部之外真能取出字节"，不替调用方读张量。
const SAMPLE_LIMIT: usize = 1 << 20;

/// 按需读最小的那条条目：把"头部说合法"与"字节能取出来"这两件事分开验。
///
/// 挑最小的条目而不是第一条，是因为真实权重里第一条往往是嵌入矩阵——读它等于把整份文件搬进
/// 内存，而这恰恰是 [`weights::safetensors::File`] 要避开的。超过 [`SAMPLE_LIMIT`] 时如实报
/// "未读"，不为了让输出好看而把上限偷偷抬高。
fn read_sample(archive: &weights::safetensors::File) -> Result<String> {
    let Some(entry) = archive
        .header()
        .entries()
        .iter()
        .min_by_key(|entry| entry.len())
    else {
        return Ok("空文件：没有条目可读".to_string());
    };
    if entry.len() > SAMPLE_LIMIT {
        return Ok(format!(
            "最小条目 `{}` 也有 {} 字节，超过样例上限 {SAMPLE_LIMIT}，未读",
            entry.name(),
            entry.len()
        ));
    }
    let bytes = archive.load(entry)?;
    Ok(format!(
        "按需读 `{}` {} 字节成功（偏移 {}），数据段未整份读入",
        entry.name(),
        bytes.len(),
        entry.offset()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 写一个最小的 safetensors：两个矩阵条目 + 一个范数条目（故意留一条未覆盖）。
    fn write_safetensors(name: &str) -> std::path::PathBuf {
        let json = r#"{"blk.0.attn_q.weight": {"dtype": "F32", "shape": [64, 32], "data_offsets": [0, 8192]},
                       "blk.0.attn_o.weight": {"dtype": "F32", "shape": [64, 64], "data_offsets": [8192, 24576]},
                       "blk.0.norm.weight": {"dtype": "F32", "shape": [64], "data_offsets": [24576, 24832]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(bytes.len() + 24832, 0);
        let path = std::env::temp_dir().join(format!("{}-{name}", std::process::id()));
        std::fs::write(&path, &bytes).expect("临时文件可写");
        // 读回确认：某些环境里临时目录被并行进程清理过，空文件会让断言变得难以解释。
        let written = std::fs::read(&path).expect("临时文件可读");
        assert_eq!(written, bytes, "落盘内容必须与待写内容一致");
        path
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

        let settings = cli::settings::Settings::load()
            .map_err(|error| error.with("配置"))
            .expect("默认设置");
        let dirs = kernel_dirs(&settings);
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

    #[test]
    fn without_a_weights_file_the_model_is_the_declared_stack() {
        let (model, row) = load_model(None).expect("声明层合法");
        assert_eq!(model.layers().len(), 2);
        assert!(model.uncovered().is_empty());
        assert!(row.contains("只用声明层"), "{row}");
        let floor = model.floor();
        assert!(floor.total().read() > 2 * 8 * 1024 * 1024);
        assert!(floor.unpriced().is_empty(), "声明的 int4 层都能计价");
    }

    #[test]
    fn a_real_file_supplies_the_layers_and_reports_what_it_could_not_use() {
        let path = write_safetensors("llm-status-test.safetensors");
        let (model, row) = load_model(Some(&path.display().to_string())).expect("文件合法");
        let names: Vec<&str> = model.layers().iter().map(|layer| layer.name()).collect();
        assert_eq!(names, ["blk.0.attn_q", "blk.0.attn_o"], "条目的偏移顺序");
        assert_eq!(model.uncovered(), ["blk.0.norm.weight"]);
        // 条目写的是 F32：层被列出来，而且由稠密算子计价——读的是权重本身的字节数，
        // 不是按 int4 折算出来的零头。
        let floor = model.floor();
        assert!(floor.unpriced().is_empty(), "{:?}", floor.unpriced());
        assert_eq!(
            floor.total(),
            nn::Bounds::of_op(nn::Linear::DENSE, 32, 64)
                .expect("稠密语义")
                .plus(nn::Bounds::of_op(nn::Linear::DENSE, 64, 64).expect("稠密语义")),
            "栈的总下界 = 两层稠密投影之和"
        );
        assert_eq!(
            floor.total().read(),
            ((64 * 32 + 32) + (64 * 64 + 64)) as u64 * 4,
            "读侧 = 权重 + 输入向量"
        );
        assert_eq!(
            floor.total().write(),
            (64 + 64) as u64 * 4,
            "写侧 = 两段输出"
        );
        assert!(row.contains("3 条条目"), "{row}");
        assert!(row.contains("24832 字节"), "{row}");
        assert!(
            row.contains("按需读 `blk.0.norm.weight` 256 字节成功"),
            "样例读的是最小的那条：{row}"
        );
        let header = row
            .split("只读头部 ")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .expect("行里要有头部读取量");
        let header: usize = header.parse().expect("头部读取量是数字");
        assert!(
            header < 1024,
            "数据段 24832 字节，但只读了 {header} 字节头部：{row}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_file_is_an_io_error_not_a_panic() {
        let error = load_model(Some("/nonexistent/llm.safetensors")).expect_err("读不到");
        assert_eq!(error.code(), diagnostic::Code::Io);
        assert!(error.to_string().contains("打开文件"), "{error}");
    }

    #[test]
    fn a_truncated_file_fails_before_any_layer_is_used() {
        let path =
            std::env::temp_dir().join(format!("{}-truncated.safetensors", std::process::id()));
        let json = r#"{"w.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(bytes.len() + 8, 0); // 头部说了 16 字节，只给了 8 字节
        std::fs::write(&path, &bytes).expect("临时文件可写");

        let error = load_model(Some(&path.display().to_string())).expect_err("被截断");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
        assert!(error.to_string().contains("文件被截断"), "{error}");
        let _ = std::fs::remove_file(path);
    }
}
