//! `run` 命令：用权重文件里的**真实数值**算一条前向链，每层一次真机内核（L7 bin 内模块）。
//!
//! 与 `status` 的分工：`status` 回答"这些层至少需要多少流量"（算术，不需要设备），本命令回答
//! "这些层真的算出来是多少"（协商 → 拼接 → emit → 编译 → 执行 → 与标量参考对拍）。两者用同一份
//! 层清单与同一张算子语义表，因此下界与结果不会各说一套。

use diagnostic::render::Panel;
use diagnostic::{Code, Error, Result};
use nn::Contract as _;

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
fn finish(panel: Panel, x: &[f32], stopped: Option<String>, executed_layers: usize) -> Result<()> {
    if executed_layers == 0 {
        let reason = stopped.unwrap_or_else(|| "没有可执行的层".to_string());
        return Err(Error::new(Code::Missing, reason).with("llm run"));
    }
    let panel = panel.row("输出    向量", output_row(x));
    let panel = match stopped {
        Some(reason) => panel.row("停下    原因", reason),
        None => panel,
    };
    print!("{}", panel.render());
    Ok(())
}

/// 配置与档案目录：设置装载失败只补命令帧，目录由设置投影出来。
fn prepare() -> Result<(cutile::dirs::Dirs, cutile::builtin::Catalogue)> {
    let settings = crate::settings::Settings::load().map_err(|error| error.with("配置"))?;
    let dirs = settings.dirs();
    let catalogue = cutile::builtin::Catalogue::load(&dirs)?;
    Ok((dirs, catalogue))
}

/// 沿层清单跑前向链；返回停止原因与已执行层数，报告行就地追加进 `panel`。
fn drive(
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    device: &str,
    file: &weights::safetensors::File,
    layers: &[nn::Linear],
    x: &mut Vec<f32>,
    panel: Panel,
) -> Result<(Panel, Option<String>, usize)> {
    let mut panel = panel;
    let mut stopped: Option<String> = None;
    let mut executed = 0usize;
    for layer in layers {
        match advance(catalogue, dirs, device, file, layer, x)? {
            Step::Continue { output, row } => {
                panel = panel.row(format!("层       {}", layer.name()), row);
                *x = output;
                executed += 1;
            }
            Step::Stopped(reason) => {
                stopped = Some(reason);
                break;
            }
        }
    }
    Ok((panel, stopped, executed))
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

    let panel = Panel::new("llm —— 用真实权重算一次（每层一条内核）")
        .row("输入    向量", format!("{} 个 f32（{}）", x.len(), device));
    let (panel, stopped, executed_layers) =
        drive(&catalogue, &dirs, device, &file, &layers, &mut x, panel)?;

    finish(panel, &x, stopped, executed_layers)
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
#[path = "../tests/run_tests.rs"]
mod tests;
