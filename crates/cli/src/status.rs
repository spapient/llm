//! `status` 命令的组装与汇报（L7 bin 内模块）：逐层构造最小实例，把读到的事实与算出的下界排成一行。

use diagnostic::render::Panel;
use diagnostic::{Code, Error, Result};
use nn::Contract as _;
use nn::LmHead;
use std::fmt::Write as _;
use tokenizer::Tokenizer as _;
use weights::Loader as _;

/// 每台设备的设备侧代码这次构建带了什么：只有源码，还是连镜像一起带上了。
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
pub(crate) fn status(weights: Option<&str>) -> Result<()> {
    let panel = build_report(weights)?;
    print!("{}", panel.render());
    Ok(())
}

/// 算子清单一行：数量与名字。
fn ops_row() -> String {
    let ops = op::builtin();
    format!(
        "{} 个算子（{}）",
        ops.len(),
        diagnostic::render::join_names(ops.iter().map(op::Op::name))
    )
}

/// 设备档案一行：数量与名字。
fn prof_row() -> String {
    let profiles = prof::names();
    format!(
        "{} 份设备档案（{}）",
        profiles.len(),
        diagnostic::render::join_names(profiles.iter().copied())
    )
}

/// 片段设备一行：数量、厂商与设备侧代码。
fn prim_row() -> String {
    format!(
        "{} 台设备（{}）；设备侧代码：{}",
        prim::devices().len(),
        diagnostic::render::join_names(prim::devices().iter().map(|device| device.vendor)),
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
        let _ = write!(summary, "；无法计价 {} 层", floor.unpriced().len());
    }
    if !model.uncovered().is_empty() {
        let _ = write!(
            summary,
            "；未覆盖 {} 条：{}",
            model.uncovered().len(),
            diagnostic::render::join_names(model.uncovered().iter().map(String::as_str))
        );
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
fn model_rows(panel: Panel, weights: Option<&str>) -> Result<Panel> {
    let (model, weights_row) = load_model(weights)?;
    let panel = panel.row("权重    weights", weights_row);
    let (model_row, unpriced_row) = model_floor(&model);
    let panel = panel.row("下界    模型", model_row);
    Ok(match unpriced_row {
        Some(reason) => panel.row("无法计价 nn", reason),
        None => panel,
    })
}

/// 张量与视图两行：最小实例布局。
fn data_rows(panel: Panel) -> Result<Panel> {
    let buf = vec![0.5f32, 0.25, 0.125, 0.0625];
    let view: tensor::View<'_, f32> =
        tensor::Tensor::new(tensor::Shape::new([2, 2]), buf.as_slice())?;
    let profile = format!(
        "{:?} {:?}，第 1 行和 = {:.3}（借用调用方 {} 个元素，不复制）",
        view.dtype(),
        view.dims(),
        view.row(1)
            .map(|row| row.iter().sum::<f32>())
            .unwrap_or(0.0),
        view.numel()
    );
    // 先零拷贝切出末行，再广播成 4 行：步长 0 复用同一份元素，转置与置换表达不出来。
    let last = view
        .narrow(0, 1, 1)
        .ok_or_else(|| Error::new(Code::Internal, "2×2 视图切末行必成立").with("llm status"))?;
    let wide = last.broadcast(vec![4usize, 2]);
    let panel = panel.row(
        "张量    tensor::Tensor",
        match wide {
            Some(wide) => {
                // 摊平按逻辑下标顺序读一遍：步长 0 的广播轴在这里展开成真实元素，缓冲仍只有一份。
                let flat = wide.to_vec();
                format!(
                    "{:?} {:?} 步长 {:?}，连续：{}，摊平 {} 个 = {:?}",
                    wide.dtype(),
                    wide.dims(),
                    wide.strides(),
                    wide.is_contiguous(),
                    flat.len(),
                    flat
                )
            }
            None => "广播失败".to_owned(),
        },
    );
    Ok(panel.row("视图    tensor::View", profile))
}

/// 训练行：起一个最小训练计划实例。
fn train_row(panel: Panel) -> Result<Panel> {
    let schedule = train::Schedule::new(10, 100)?;
    Ok(panel.row(
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
fn serve_row(panel: Panel) -> Result<Panel> {
    let route = serve::Route::parse("POST /v1/completions")?;
    let tokens = tokenizer::Bytes.encode("hi")?;
    Ok(panel.row(
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
fn engine_row(panel: Panel) -> Result<Panel> {
    let mut scheduler = engine::Scheduler::new(status_pager()?);
    let sequence = engine::Sequence::new(0, &[0; 8], 4, 16, engine::exec::Stream::new(0))?;
    let response = serve::admit(&mut scheduler, &sequence)?;
    Ok(panel.row(
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
fn runtime_rows(panel: Panel) -> Result<Panel> {
    let panel = train_row(panel)?;
    let panel = serve_row(panel)?;
    engine_row(panel)
}

/// 组装整份状态报告：各领域一行一段，错误传播收敛到各自助手。
fn build_report(weights: Option<&str>) -> Result<Panel> {
    let panel = Panel::new("llm —— 组装点（各层最小实例）")
        .row("声明层  op", ops_row())
        .row("档案层  prof", prof_row())
        .row("片段层  prim", prim_row())
        .row("层定义  nn", lmhead_row())
        .row("下界    层", projection_row()?);
    let panel = model_rows(panel, weights)?;
    let panel = data_rows(panel)?;
    runtime_rows(panel)
}

/// `status` 用的 KV 页池：这里没有模型，行宽取 1——页里的数不参与账目。
fn status_pager() -> Result<engine::kv::Pager> {
    engine::kv::Pager::new(engine::kv::Cache::new(4, 16)?, 1)
}

/// 一条下界的统一写法：读、写分开给，合计与强度作为派生量。
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

/// 装配与汇报的单元测试。
#[cfg(test)]
#[path = "../tests/status.rs"]
mod tests;
