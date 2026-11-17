//! `run` 命令：用权重文件里的**真实数值**算一条前向链，每层一次真机内核（L7 bin 内模块）。
//!
//! 与 `status` 的分工：`status` 回答"这些层至少需要多少流量"（算术，不需要设备），本命令回答
//! "这些层真的算出来是多少"（协商 → 拼接 → emit → 编译 → 执行 → 与标量参考对拍）。两者用同一份
//! 层清单与同一张算子语义表，因此下界与结果不会各说一套。

use diagnostic::{Code, Error, Result};
use nn::Contract as _;
use tool::Report;

/// 内核只收三个目录：把解析好的设置投影成那份纯数据（`cutile` 不读环境变量，ADR 0023）。
fn kernel_dirs(settings: &crate::settings::Settings) -> cutile::dirs::Dirs {
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
pub(crate) fn run(weights: &str, input: Option<&str>, device: Option<&str>) -> Result<()> {
    // 配置解析在命令行层、档案装载在核心里；本函数只决定"跑哪些层、用什么数据"。
    let settings = crate::settings::Settings::load().map_err(|error| error.with("配置"))?;
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
        // 停下来不是错误：前 N 层的结果是真的，后面的层只是这一版不支持。
        report = report.row("停下    原因", reason);
    }
    print!("{}", report.render());
    Ok(())
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
}
