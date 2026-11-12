//! 组装点：把各层连起来，**不定义任何新类型**（`docs/ARCHITECTURE.md` 的 L7）。
//!
//! 现在它只做一件事：构造每一层的最小实例并打印一份摘要。这不是演示，而是"依赖图真的能编译、
//! 各层公共 API 真的可用"的端到端证据——依赖方向只是评审约定，能不能跑只能靠这里真正组装。
//!
//! 真正的子命令（`run` / `serve` / `train`）在 P1 接入；在那之前，任何新类型都应该长在
//! 它所属的层里，而不是长在这里。

use diagnostic::Result;
use nn::Layer as _;
use std::process::ExitCode;
use tokenizer::Tokenizer as _;
use tool::Report;
use weights::Loader as _;

/// 由注解声明的固定形状层：形状与算子只写一次，`impl` 与校验测试由宏生成。
#[nn::layer(name = "lm_head", out = [4096], ops = ["gemv_i4"])]
struct LmHead;

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
    match invocation.command() {
        cli::Command::Help => {
            print!("{}", cli::USAGE);
            ExitCode::SUCCESS
        }
        cli::Command::Status { weights } => match status(weights.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("✗ {error}");
                ExitCode::from(error.code().exit_code())
            }
        },
        cli::Command::Verify { .. } | cli::Command::Bench { .. } => {
            // 组装点只分派：这两个审计命令今天由 `cutile` 二进制提供，`llm` 不复制实现。
            eprintln!("✗ 该命令当前由 `cargo run --bin cutile -- ...` 提供（`llm` 只负责组装）");
            ExitCode::from(2)
        }
    }
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
    let profiles = prof::builtin();
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
                tool::join_names(profiles.iter().map(prof::Archive::name))
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
            LmHead.spec().ops().join("、")
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

    let mut scheduler = engine::Scheduler::new(kv::Cache::new(4, 16)?);
    let sequence = engine::Sequence::new(0, 8, 4, 16, exec::Stream::new(0))?;
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
