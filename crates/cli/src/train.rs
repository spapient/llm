//! `train` 命令：配置文件、环境变量与显式选项共同投影成 [`train::TrainSpec`]；
//! `--load` 从快照恢复事实、矩状态、累积窗口与权重绑定表。

use diagnostic::render::Panel;
use diagnostic::{Code, Error, Result};

use crate::settings::Settings;

/// `train` 分派：解析配置、按 `--load` 决定汇报恢复事实还是配置。
pub(crate) fn train_run(
    settings: &Settings,
    lr: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    clip: Option<&str>,
    accumulation: Option<&str>,
    load: Option<&str>,
) -> Result<()> {
    let spec = build_train_spec(settings, lr, warmup, steps, clip, accumulation)?;
    if let Some(path) = load {
        report_recovered(path, &spec)
    } else {
        report_train_config(&spec);
        Ok(())
    }
}

/// 先取四层配置投影，再应用优先级相同的显式训练选项；默认值只由 train 类型提供。
fn build_train_spec(
    settings: &Settings,
    lr: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    clip: Option<&str>,
    accumulation: Option<&str>,
) -> Result<train::TrainSpec> {
    let spec = settings.train_spec()?;
    let (learning_rate, clip_norm, warmup_steps, max_steps, accumulation) =
        parse_overrides(lr, clip, warmup, steps, accumulation)?;
    let session = spec.session.overridden(
        learning_rate,
        warmup_steps,
        max_steps,
        clip_norm,
        accumulation,
    )?;
    train::TrainSpec::new(session, spec.loader).map_err(|error| error.with("llm train"))
}

/// 显式训练选项按类型解析：学习率 / 裁剪为 `f32`，warmup / 步数为 `u64`，累积窗为 `u32`。
fn parse_overrides(
    lr: Option<&str>,
    clip: Option<&str>,
    warmup: Option<&str>,
    steps: Option<&str>,
    accumulation: Option<&str>,
) -> Result<(Option<f32>, Option<f32>, Option<u64>, Option<u64>, Option<u32>)> {
    let learning_rate = parse_optional_f32(lr, "--lr")?;
    let clip_norm = parse_optional_f32(clip, "--clip")?;
    let (warmup_steps, max_steps, accumulation) =
        parse_train_uints(warmup, steps, accumulation)?;
    Ok((learning_rate, clip_norm, warmup_steps, max_steps, accumulation))
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
fn report_recovered(path: &str, spec: &train::TrainSpec) -> Result<()> {
    let snapshot = weights::Snapshot::read(std::path::Path::new(path))?;
    let facts = snapshot.facts();
    let moments = snapshot.moments();
    let report = Panel::new("llm —— 从检查点恢复")
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
                spec.session.accumulation()
            ),
        )
        .row("权重绑定", format!("{} 对", snapshot.ties().pairs().len()));
    print!("{}", report.render());
    Ok(())
}

/// 无 `--load`：排版完整装配配置；会话闭环由 `train::Session` 消费其中的会话配置。
fn report_train_config(spec: &train::TrainSpec) {
    let config = &spec.session;
    let report = Panel::new("llm —— 训练配置（会话闭环由 train::Session 装配）")
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
        .row("累积", format!("{} micro-batch", config.accumulation()))
        .row(
            "批形状",
            format!("{} × {}", spec.loader.rows, spec.loader.cols),
        );
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
