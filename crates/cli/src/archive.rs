//! `import` / `inspect` / `check` 三条 `.u` 容器命令的组装与汇报（L7 bin 内模块）。
//!
//! 分工与既有编排模块一致：本模块只装配库层调用、取值解析与人可读排版，不定义新领域类型；
//! 条目流、编码、解码与度量口径都在 `weights` 层，这里不复制第二份规则。
//!
//! - `import`：源文件（`.safetensors` / `.gguf` / `.bin` / `.pt`）→ 中性段流 → [`Writer`]
//!   逐段装入 → 流式落盘 `write_to_path(6, out)`；不支持的 wire / 块量化由 weights 层显式
//!   `Unsupported`，本层不静默转换。
//! - `inspect`：整份读进内存 → [`Archive::parse`] → `metrics()`；度量是打开后现场现算，
//!   不读文件内保存的度量字段，也不出 Roofline / 设备数字。
//! - `check`：`parse` 已在打开期校验段级 / 全局 CRC；逐段 `load` 触发解码与逐位还原，
//!   输出解码条目数。坏文件按既有 [`diagnostic::Code`] 口径落成非零退出码。

use std::path::Path;

use diagnostic::{Code, Error, Result};
use tool::Report;
use weights::Archive;

/// 容器对齐指数：6 = 64 B，与既有落盘口径一致。
const ALIGN_LOG: u8 = 6;

/// `import`：按扩展名分派源格式，中性段流装入写者并流式写出 `.u` 容器。
///
/// 条目字段不自洽、wire 不在映射闭集内或源是未映射块量化时，weights 层显式报错，
/// 本命令只把命令帧补到错误链上，不把未映射类型转成别的存储表示。
pub(crate) fn import(input: &str, output: &str) -> Result<()> {
    let segments = weights::import::segments_from_path(Path::new(input))
        .map_err(|error| error.with("llm import"))?;
    let mut writer = weights::Writer::new();
    for segment in segments {
        writer
            .push(segment)
            .map_err(|error| error.with("llm import"))?;
    }
    writer
        .write_to_path(ALIGN_LOG, Path::new(output))
        .map_err(|error| error.with("llm import"))?;
    let report = Report::new("llm —— safetensors 导入 `.u` 容器")
        .row("源", input)
        .row("输出", output)
        .row("条目", format!("{} 段", writer.len()));
    print!("{}", report.render());
    Ok(())
}

/// `inspect`：读字节 → 解析 → 现场现算度量，条目清单与度量排成报告。
pub(crate) fn inspect(input: &str) -> Result<()> {
    let bytes = std::fs::read(input).map_err(|error| {
        Error::new(Code::Io, format!("读取 {input} 失败：{error}")).with("llm inspect")
    })?;
    let archive = Archive::parse(&bytes).map_err(|error| error.with("llm inspect"))?;
    let metrics = archive
        .metrics()
        .map_err(|error| error.with("llm inspect"))?;

    let mut report = Report::new("llm —— `.u` 容器清单与度量")
        .row("条目数", format!("{}", metrics.entries()))
        .row("参数总数", format!("{}", metrics.params()))
        .row("V_artifact", tool::human_bytes(metrics.v_artifact() as u64))
        .row("V_FP16", tool::human_bytes(metrics.v_fp16()))
        .row("ratio", format!("{:.3}", metrics.ratio()))
        .row(
            "lossless",
            if metrics.lossless() { "true" } else { "false" },
        )
        .row(
            "REF 命中",
            format!(
                "{}（节省 {}）",
                metrics.ref_hits(),
                tool::human_bytes(metrics.ref_saved_bytes())
            ),
        );
    for segment in archive.segments() {
        report = report.row("条目", entry_row(segment));
    }
    print!("{}", report.render());
    Ok(())
}

/// `check`：打开期 CRC 已校验；逐段 `load` 触发解码与逐位还原，按解码条目数汇报。
pub(crate) fn check(input: &str) -> Result<()> {
    let bytes = std::fs::read(input).map_err(|error| {
        Error::new(Code::Io, format!("读取 {input} 失败：{error}")).with("llm check")
    })?;
    let archive = Archive::parse(&bytes).map_err(|error| error.with("llm check"))?;
    let mut decoded = 0usize;
    for segment in archive.segments() {
        archive
            .load(segment)
            .map_err(|error| error.with("llm check"))?;
        decoded += 1;
    }
    println!("{decoded} tensors decoded");
    Ok(())
}

/// 一条条目的清单行：名字 / dtype / 形状 / codec；形状轴长用 `×` 连接，空形状记 `[]`。
fn entry_row(segment: &weights::Stored) -> String {
    let shape = segment
        .shape()
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("×");
    let shape = if shape.is_empty() {
        "[]".to_string()
    } else {
        shape
    };
    format!(
        "{} · {} · [{}] · {:?}",
        segment.name(),
        segment.dtype().name(),
        shape,
        segment.codec()
    )
}
