//! `status` 命令的组装与汇报（L7 bin 内模块）：逐层构造最小实例，把读到的事实与算出的下界排成一行。
//!
//! 判定与数值都来自各层库，本模块只装配、选词与分行；不定义新类型。

use diagnostic::{Code, Error, Result};
use nn::Contract as _;
use nn::LmHead;
use tokenizer::Tokenizer as _;
use tool::Report;
use weights::Loader as _;

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
pub(crate) fn status(weights: Option<&str>) -> Result<()> {
    let report = build_report(weights)?;
    print!("{}", report.render());
    Ok(())
}

/// 算子清单一行：数量与名字。
fn ops_row() -> String {
    let ops = op::builtin();
    format!(
        "{} 个算子（{}）",
        ops.len(),
        tool::join_names(ops.iter().map(op::Op::name))
    )
}

/// 设备档案一行：数量与名字。
fn prof_row() -> String {
    let profiles = prof::names();
    format!(
        "{} 份设备档案（{}）",
        profiles.len(),
        tool::join_names(profiles.iter().copied())
    )
}

/// 片段设备一行：数量、厂商与设备侧代码。
fn prim_row() -> String {
    format!(
        "{} 台设备（{}）；设备侧代码：{}",
        prim::devices().len(),
        tool::join_names(prim::devices().iter().map(|device| device.vendor)),
        device_images()
    )
}

/// 层定义一行：LmHead 形状与算子。
fn lmhead_row() -> String {
    format!(
        "{} → {:?}，算子 {}",
        LmHead.name(),
        LmHead.output().dims(),
        LmHead.spca().ops().join("、")
    )
}

/// 单层下界一行：构造一个 4096 投影并计价。
fn projection_row() -> Result<String> {
    let projection = nn::Linear::new("blk.0.attn_q", 4096, 4096)?;
    let floor = projection.bounds()?.ok_or_else(|| {
        Error::new(Code::Internal, "Linear 有形状就应当能计价").with("llm status")
    })?;
    Ok(format!(
        "{} 4096 → ≥ {}",
        projection.name(),
        floor_row(&floor)
    ))
}

/// 模型下界汇总：逐层求和，另把无法计价的层单独列一行。
fn model_floor(model: &nn::Stack) -> (String, Option<String>) {
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
    let unpriced_row = if !floor.unpriced().is_empty() {
        Some(
            floor
                .unpriced()
                .iter()
                .map(nn::Unpriced::to_string)
                .collect::<Vec<_>>()
                .join("；"),
        )
    } else {
        None
    };
    (summary, unpriced_row)
}

/// 权重与模型下界两行：读文件（或声明层），再排汇总与未计价理由。
fn model_rows(report: Report, weights: Option<&str>) -> Result<Report> {
    let (model, weights_row) = load_model(weights)?;
    let report = report.row("权重    weights", weights_row);
    let (model_row, unpriced_row) = model_floor(&model);
    let report = report.row("下界    模型", model_row);
    Ok(match unpriced_row {
        Some(reason) => report.row("无法计价 nn", reason),
        None => report,
    })
}

/// 张量与视图两行：最小实例布局。
fn data_rows(report: Report) -> Result<Report> {
    let tensor = tensor::Tensor::new(tensor::DType::F32, tensor::Shape::new([2, 2]), &[0u8; 16])?;
    let report = report.row(
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
    Ok(report.row(
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
    ))
}

/// 训练行：起一个最小训练计划实例。
fn train_row(report: Report) -> Result<Report> {
    let schedule = train::Schedule::new(10, 100)?;
    Ok(report.row(
        "训练    train",
        format!(
            "warmup {}/{} 步，第 0 步 lr = {:.3}",
            schedule.warmup(),
            schedule.total(),
            schedule.lr(1.0, 0)
        ),
    ))
}

/// 服务行：解析一条路由并编最小提示词。
fn serve_row(report: Report) -> Result<Report> {
    let route = serve::Route::parse("POST /v1/completions")?;
    let tokens = tokenizer::Bytes.encode("hi")?;
    Ok(report.row(
        "服务    serve",
        format!(
            "{} {}，提示 \"hi\" → {} 个 token",
            route.method(),
            route.path(),
            tokens.len()
        ),
    ))
}

/// 调度行：起页池与一条序列，走一次准入。
fn engine_row(report: Report) -> Result<Report> {
    let mut scheduler = engine::Scheduler::new(status_pager()?);
    let sequence = engine::Sequence::new(0, &[0; 8], 4, 16, engine::exec::Stream::new(0))?;
    let response = serve::admit(&mut scheduler, &sequence)?;
    Ok(report.row(
        "调度    engine",
        format!(
            "准入 {} 个序列，占用 {} 页（页宽 {}），响应 {}",
            scheduler.running(),
            scheduler.cache().in_use(),
            scheduler.cache().page_size(),
            response.status().code()
        ),
    ))
}

/// 训练、服务、调度三行：各自起一个最小实例。
fn runtime_rows(report: Report) -> Result<Report> {
    let report = train_row(report)?;
    let report = serve_row(report)?;
    engine_row(report)
}

/// 组装整份状态报告：各领域一行一段，错误传播收敛到各自助手。
fn build_report(weights: Option<&str>) -> Result<Report> {
    let report = Report::new("llm —— 组装点（各层最小实例）")
        .row("声明层  op", ops_row())
        .row("档案层  prof", prof_row())
        .row("片段层  prim", prim_row())
        .row("层定义  nn", lmhead_row())
        .row("下界    层", projection_row()?);
    let report = model_rows(report, weights)?;
    let report = data_rows(report)?;
    runtime_rows(report)
}

/// `status` 用的 KV 页池：这里没有模型，行宽取 1——页里的数不参与账目。
fn status_pager() -> Result<engine::kv::Pager> {
    engine::kv::Pager::new(engine::kv::Cache::new(4, 16)?, 1)
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

/// 没给权重文件：用声明的一对投影搭栈，附一行说明。
fn declared_stack() -> Result<(nn::Stack, String)> {
    let model = nn::Stack::new(vec![
        nn::Linear::new("blk.0.attn_q", 4096, 4096)?,
        nn::Linear::new("blk.0.attn_o", 4096, 4096)?,
    ])?;
    Ok((
        model,
        "未提供权重文件：只用声明层（`--weights <file>` 读 safetensors）".to_string(),
    ))
}

/// 真实档案一行：条目数、数据段与头部读取量，外加一条按需样例。
fn archive_row(archive: &weights::safetensors::File, path: &str) -> Result<String> {
    Ok(format!(
        "safetensors：{} 条条目，数据段 {} 字节，只读头部 {} / {} 字节（{path}）；{}",
        archive.header().entries().len(),
        archive.header().data_bytes(),
        archive.header_len(),
        archive.file_len(),
        read_sample(archive)?
    ))
}

/// 模型：给了权重文件就读真实条目，没给就用声明的一对投影。
///
/// 两条路都返回 [`nn::Stack`]，所以"层下界之和"只有一份算法；区别只在层的清单从哪来。返回的
/// 第二项是权重那一行的文字——把"读到了什么"与"算出了什么"分开，读者才能判断下界覆盖了什么。
fn load_model(weights: Option<&str>) -> Result<(nn::Stack, String)> {
    let Some(path) = weights else {
        return declared_stack();
    };
    let archive =
        weights::safetensors::File::open(path).map_err(|error| error.with("llm status"))?;
    let model = nn::Stack::from_header(archive.header())?;
    let row = archive_row(&archive, path)?;
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
