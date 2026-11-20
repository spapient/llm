//! `train` 命令：把 cli 搬下来的字符串解析成 [`train::TrainConfig`]，`--load` 从快照恢复事实、
//! 矩状态、累积窗口与权重绑定表。
//!
//! 前向/反传会话循环在 `train::Session`（夹具级闭环见 train 测试）；本组装点不复制模型装载，
//! 真实权重通路仍属未接入范围。本模块只解析到 [`Option`]，不复制第二份默认值。

use diagnostic::{Code, Error, Result};
use tool::Report;

/// `train` 分派：解析配置、按 `--load` 决定汇报恢复事实还是配置。
pub(crate) fn train_run(
    lr: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    clip: Option<&str>,
    accumulation: Option<&str>,
    load: Option<&str>,
) -> Result<()> {
    let config = build_train_config(lr, warmup, steps, clip, accumulation)?;
    match load {
        Some(path) => report_recovered(path, &config),
        None => {
            report_train_config(&config);
            Ok(())
        }
    }
}

/// 把 cli 搬下来的字符串解析成 [`train::TrainConfig`]；缺省常数归
/// [`train::TrainConfig::default`]，本函数只解析到 [`Option`]，不复制第二份默认值。
fn build_train_config(
    lr: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    clip: Option<&str>,
    accumulation: Option<&str>,
) -> Result<train::TrainConfig> {
    let learning_rate = parse_optional_f32(lr, "--lr")?;
    let clip_norm = parse_optional_f32(clip, "--clip")?;
    let (warmup_steps, max_steps, accumulation) = parse_train_uints(warmup, steps, accumulation)?;
    train::TrainConfig::with_overrides(
        learning_rate,
        warmup_steps,
        max_steps,
        clip_norm,
        accumulation,
    )
    .map_err(|error| error.with("llm train"))
}

/// 无符号档位：warmup/步数是 `u64`，累积窗是 `u32`；类型随快照事实口径固定。
fn parse_train_uints(
    warmup: Option<&str>,
    steps: Option<&str>,
    accumulation: Option<&str>,
) -> Result<(Option<u64>, Option<u64>, Option<u32>)> {
    Ok((
        parse_optional_uint::<u64>(warmup, "--warmup")?,
        parse_optional_uint::<u64>(steps, "--steps")?,
        parse_optional_uint::<u32>(accumulation, "--accumulation")?,
    ))
}

/// `--load`：从快照读出恢复事实、条目表、矩状态、累积窗口与权重绑定表并排版汇报。
fn report_recovered(path: &str, config: &train::TrainConfig) -> Result<()> {
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
        )
        .row(
            "累积窗口",
            format!(
                "{} / {} micro-batch",
                snapshot.accum().count(),
                config.accumulation()
            ),
        )
        .row("权重绑定", format!("{} 对", snapshot.ties().pairs().len()));
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

pub mod archive {
    //! `import` / `inspect` / `check` / `settle` 四条 `.u` 容器命令的组装与汇报（L7 bin 内模块）。
    //!
    //! 分工与既有编排模块一致：本模块只装配库层调用、取值解析与人可读排版，不定义新领域类型；
    //! 条目流、编码、解码与度量口径都在 `weights` 层，这里不复制第二份规则。
    //!
    //! - `import`：源文件（`.safetensors` / `.gguf` / `.bin` / `.pt`）→ 中性段流 → [`Writer`]
    //!   逐段装入 → 流式落盘 `write_to_path(6, out)`；不支持的 wire / 块量化由 weights 层显式
    //!   `Unsupported`，本层不静默转换。落盘后再把字节读一遍现算账本，报告里的体积比与无损
    //!   因此是「读出来的」而不是「自以为写进去的」。
    //! - `inspect`：整份读进内存 → [`Archive::parse`] → `metrics()`；度量是打开后现场现算，
    //!   不读文件内保存的度量字段，也不出 Roofline / 设备数字。
    //! - `check`：`parse` 已在打开期校验段级 / 全局 CRC；逐段 `load` 触发解码与逐位还原，
    //!   输出解码条目数。坏文件按既有 [`diagnostic::Code`] 口径落成非零退出码。
    //! - `settle`：同一份字节 → 逐张量体积 / 熵 / 量化档位 / 无损与原因（`weights::settle`
    //!   现算）。账本 JSON 的文本只有一个来源，写盘与 stdout 用的是同一份。
    //! - `audit`：把落盘的账本读回，与同一份字节再算一遍的账逐格核对——账本唯一的消费方式。

    use std::path::Path;

    use diagnostic::{Code, Error, Result};
    use tool::Report;
    use weights::{Archive, Settlement};

    /// 容器对齐指数：6 = 64 B，与既有落盘口径一致。
    const ALIGN_LOG: u8 = 6;

    /// `import`：按扩展名分派源格式，中性段流装入写者并流式写出 `.u` 容器。
    ///
    /// 条目字段不自洽、wire 不在映射闭集内或源是未映射块量化时，weights 层显式报错，
    /// 本命令只把命令帧补到错误链上，不把未映射类型转成别的存储表示。
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
    fn import_report(input: &str, output: &str, entries: usize, ledger: &Settlement) -> Report {
        Report::new("llm —— safetensors 导入 `.u` 容器")
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
    fn metrics_report(archive: &Archive<'_>, metrics: &weights::Metrics) -> Report {
        let mut report = Report::new("llm —— `.u` 容器清单与度量")
            .row("条目数", format!("{}", metrics.entries()))
            .row("参数总数", format!("{}", metrics.params()))
            .row("V_artifact", tool::human_bytes(metrics.v_artifact() as u64))
            .row("V_FP16", tool::human_bytes(metrics.v_fp16()))
            .row("ratio", format!("{:.3}", metrics.ratio()))
            .row("lossless", lossless_label(metrics.lossless()))
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
        report
    }

    /// 无损标志的人可读取值。
    fn lossless_label(lossless: bool) -> &'static str {
        if lossless { "true" } else { "false" }
    }

    /// `settle`：读字节 → 逐张量账本；`out` 给出就落成 JSON，否则把同一份文本打到 stdout。
    ///
    /// 账本从落盘字节现算（往返 + 描述符），不读文件里存的度量字段，也不向调用方猜一个产物
    /// 目录旁的默认文件名：没给 `--out` 就是“打到 stdout”，重定向的活留给 shell。
    pub(crate) fn settle(input: &str, output: Option<&str>) -> Result<()> {
        let ledger = ledger_of(input, "llm settle")?;
        let text = ledger.render();
        let Some(path) = output else {
            print!("{text}");
            return Ok(());
        };
        write_ledger(Path::new(path), &text)?;
        let report = Report::new("llm —— 沉降账本")
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
            Error::new(Code::Io, format!("写出 {} 失败：{error}", path.display()))
                .with("llm settle")
        })
    }

    /// `audit`：落盘账本 × 同一份字节 → 每格复核。
    ///
    /// 报告里的数一律取自**读回并通过复核**的那份账（`Settlement::verify` 返回它），不是重新生成
    /// 一份“看起来一样”的账：核账与记账因此不可能是两条各自算路径。
    pub(crate) fn audit(input: &str, ledger: &str) -> Result<()> {
        let bytes = read_container(input, "llm audit")?;
        let text = read_ledger(ledger)?;
        let checked = Settlement::verify(&text, &bytes).map_err(|error| error.with("llm audit"))?;
        let report = Report::new("llm —— 账本复核")
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
    pub(crate) fn check(input: &str) -> Result<()> {
        let bytes = read_container(input, "llm check")?;
        let archive = Archive::parse(&bytes).map_err(|error| error.with("llm check"))?;
        let decoded = decode_all(&archive)?;
        println!("{decoded} tensors decoded");
        Ok(())
    }

    /// 逐段解码还原全表，返回解码条目数；任一段解码失败按命令站点补诊断。
    fn decode_all(archive: &Archive<'_>) -> Result<usize> {
        let mut decoded = 0usize;
        for segment in archive.segments() {
            archive
                .load(segment)
                .map_err(|error| error.with("llm check"))?;
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
}
