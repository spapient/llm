//! `import` / `inspect` / `check` / `deploy` / `settle` / `audit` 六条 `.u` 容器命令的组装与汇报（L7 bin 内模块）。

use std::path::Path;

use diagnostic::render::Panel;
use diagnostic::{Code, Error, Result};
use weights::{Archive, Settlement};

/// 容器对齐指数：6 = 64 B，与既有落盘口径一致。
const ALIGN_LOG: u8 = 6;

/// `import`：按扩展名分派源格式，中性段流装入写者并流式写出 `.u` 容器。
pub(crate) fn import(input: &str, output: &str) -> Result<()> {
    let writer = write_container(input, output)?;
    // 导入自证：把刚写下的字节重新读一遍现算账本，“写成功”之外还有一句“写进去的是这个
    // 体积、有损与否”。口径与 `settle` 同源（`ledger_of`），两处不会各说一套。
    let ledger = ledger_of(output, "llm import")?;
    let report = import_report(input, output, writer.len(), &ledger);
    print!("{}", report.render());
    Ok(())
}

/// 源文件 → 中性段流 → 流式写出 `.u` 容器；返回写者本身（报告要用它的条目数）。
fn write_container(input: &str, output: &str) -> Result<weights::Writer> {
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
    Ok(writer)
}

/// 导入报告：条目数出自写者，体积比与无损出自**重读一遍**的账本。
fn import_report(input: &str, output: &str, entries: usize, ledger: &Settlement) -> Panel {
    Panel::new("llm —— safetensors 导入 `.u` 容器")
        .row("源", input)
        .row("输出", output)
        .row("条目", format!("{entries} 段"))
        .row("体积比", format!("{:.3}", ledger.total_ratio()))
        .row("lossless", lossless_label(ledger.lossless()))
}

/// 落盘字节 → 账本：`import` 的自证与 `settle` 都走这一入口，口径不可能分家。
fn ledger_of(path: &str, site: &str) -> Result<Settlement> {
    let bytes = read_container(path, site)?;
    Settlement::settle(&bytes).map_err(|error| error.with(site))
}

/// 读容器字节：IO 失败按 `Code::Io` 补命令站点诊断。
fn read_container(input: &str, site: &str) -> Result<Vec<u8>> {
    std::fs::read(input)
        .map_err(|error| Error::new(Code::Io, format!("读取 {input} 失败：{error}")).with(site))
}

/// `inspect`：读字节 → 解析 → 现场现算度量，条目清单与度量排成报告。
pub(crate) fn inspect(input: &str) -> Result<()> {
    let bytes = read_container(input, "llm inspect")?;
    let archive = Archive::parse(&bytes).map_err(|error| error.with("llm inspect"))?;
    let metrics = archive
        .metrics()
        .map_err(|error| error.with("llm inspect"))?;
    let report = metrics_report(&archive, &metrics);
    print!("{}", report.render());
    Ok(())
}

/// 度量行 + 条目清单行；度量是打开后现场现算，不读文件内保存字段。
fn metrics_report(archive: &Archive<'_>, metrics: &weights::Metrics) -> Panel {
    let report = metric_rows(Panel::new("llm —— `.u` 容器清单与度量"), metrics);
    archive.segments().iter().fold(report, |panel, segment| {
        panel.row("条目", entry_row(segment))
    })
}

/// 度量行：`inspect` 与 `deploy` 共用同一份，免得两处各写一遍、迟早在某一行上对不上——
/// 「清单说 ratio_param=X、裁决说 Y」是无法自证的坏账。
fn metric_rows(report: Panel, metrics: &weights::Metrics) -> Panel {
    let bytes = diagnostic::render::human_bytes;
    report
        .row("条目数", format!("{}", metrics.entries()))
        .row("参数总数", format!("{}", metrics.params()))
        .row("V_artifact", bytes(metrics.v_artifact() as u64))
        .row("V_FP16", bytes(metrics.v_fp16()))
        .row("ratio", format!("{:.3}", metrics.ratio()))
        .row("ratio_param", format!("{:.6}", metrics.ratio_param()))
        .row("lossless", lossless_label(metrics.lossless()))
        .row(
            "REF 命中",
            format!(
                "{}（节省 {}）",
                metrics.ref_hits(),
                bytes(metrics.ref_saved_bytes())
            ),
        )
}

/// 无损标志的人可读取值。
fn lossless_label(lossless: bool) -> &'static str {
    if lossless { "true" } else { "false" }
}

/// `settle`：读字节 → 逐张量账本；`out` 给出就落成 JSON，否则把同一份文本打到 stdout。
pub(crate) fn settle(input: &str, output: Option<&str>) -> Result<()> {
    let ledger = ledger_of(input, "llm settle")?;
    let text = ledger.render();
    let Some(path) = output else {
        print!("{text}");
        return Ok(());
    };
    write_ledger(Path::new(path), &text)?;
    let report = Panel::new("llm —— 沉降账本")
        .row("容器", input)
        .row("账本", path)
        .row("条目数", format!("{}", ledger.tensors().len()))
        .row("体积比", format!("{:.3}", ledger.total_ratio()))
        .row("lossless", lossless_label(ledger.lossless()));
    print!("{}", report.render());
    Ok(())
}

/// 账本落盘：失败按 `Code::Io` 补命令站点诊断，不静默地当成“已写出”。
fn write_ledger(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|error| {
        Error::new(Code::Io, format!("写出 {} 失败：{error}", path.display())).with("llm settle")
    })
}

/// `audit`：落盘账本 × 同一份字节 → 每格复核。
pub(crate) fn audit(input: &str, ledger: &str) -> Result<()> {
    let bytes = read_container(input, "llm audit")?;
    let text = read_ledger(ledger)?;
    let checked = Settlement::verify(&text, &bytes).map_err(|error| error.with("llm audit"))?;
    let report = Panel::new("llm —— 账本复核")
        .row("容器", input)
        .row("账本", ledger)
        .row("条目数", format!("{} 段", checked.tensors().len()))
        .row("体积比", format!("{:.3}", checked.total_ratio()))
        .row("lossless", lossless_label(checked.lossless()))
        .row("结论", "一致：每一格都由这份字节现算得到");
    print!("{}", report.render());
    Ok(())
}

/// 读账本文本：读不到或不是合法 UTF-8 都按 `Code::Io`——那是“读不出来”，不是“账本写错了”。
fn read_ledger(path: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|error| {
        Error::new(Code::Io, format!("读取账本 {path} 失败：{error}")).with("llm audit")
    })
}

/// `check`：打开期 CRC 已校验；逐段 `load` 触发解码与逐位还原，按解码条目数汇报。
pub(crate) fn check_archive(input: &str) -> Result<()> {
    let bytes = read_container(input, "llm check")?;
    let archive = Archive::parse(&bytes).map_err(|error| error.with("llm check"))?;
    let decoded = decode_all(&archive, "llm check")?;
    println!("{decoded} tensors decoded");
    Ok(())
}

/// `deploy`：E8 部署裁决（LLD §E8）。
pub(crate) fn deploy(input: &str) -> Result<()> {
    let bytes = read_container(input, "llm deploy")?;
    let archive = Archive::parse(&bytes).map_err(|error| error.with("llm deploy"))?;
    let decoded = decode_all(&archive, "llm deploy")?;
    let metrics = archive
        .metrics()
        .map_err(|error| error.with("llm deploy"))?;
    let pass = metrics.e8_pass();
    let rows = metric_rows(
        Panel::new("llm —— E8 部署裁决（24× 无损）").row("容器", input),
        &metrics,
    );
    let report = rows
        .row("解码通过", format!("{decoded}/{}", metrics.entries()))
        .row(
            "门",
            format!("ratio_param ≤ 1/24 = {:.6}", weights::E8_RATIO),
        )
        .row("e8_pass", lossless_label(pass));
    // 裁决不通过时整份报告走 stderr：失败命令不该污染 stdout（管道下游读到的是"通过"的
    // 内容），而 stderr 无缓冲，报告与其后的错误行天然保持先后。
    announce(
        &report,
        &Ruling {
            pass,
            ratio: metrics.ratio_param(),
            lossless: metrics.lossless(),
        },
    )
}

/// 裁决 → 输出路由 → 退出码，三件事是一件事：通过走 stdout 与 `Ok`，不通过走 stderr 与
/// [`Code::Invalid`]。拆开写就会出现"报告说通过、退出码说失败"的两份真相。
fn announce(report: &Panel, ruling: &Ruling) -> Result<()> {
    if ruling.pass {
        print!("{}", report.render());
        return Ok(());
    }
    eprint!("{}", report.render());
    Err(Error::new(
        Code::Invalid,
        format!(
            "E8 裁决未通过：ratio_param={:.6}（门 {:.6}），lossless={}",
            ruling.ratio,
            weights::E8_RATIO,
            lossless_label(ruling.lossless)
        ),
    )
    .with("llm deploy"))
}

/// 一份 E8 裁决的全部数：报告要展示的与退出码要判定的同一份，不各自算。
struct Ruling {
    pass: bool,
    ratio: f64,
    lossless: bool,
}

/// 逐段解码还原全表，返回解码条目数；任一段解码失败按命令站点补诊断。
fn decode_all(archive: &Archive<'_>, site: &str) -> Result<usize> {
    let mut decoded = 0usize;
    for segment in archive.segments() {
        archive.load(segment).map_err(|error| error.with(site))?;
        decoded += 1;
    }
    Ok(decoded)
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
