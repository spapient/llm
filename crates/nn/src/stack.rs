//! 层栈：一个模型就是一串层，本模块给出这串层以及它们的**总下界**。
//!
//! 层下界（[`Bounds`]）回答"这一层至少要搬多少、算多少"；模型下界回答同一个问题在整串层上的和。
//! 这个和不是新公式：它就是 [`Bounds::plus`]，所以新增一层不会多出第二份流量口径。
//!
//! 层的清单有两个来源：
//!
//! - 声明：[`Stack::new`] 直接给一串 [`Linear`]，[`Stack::of`] 收任意 [`Contract`]（含注解层）；
//! - 权重：[`Stack::from_header`] 按权重文件的条目推出来——秩为 2 的条目按 `[out, in]` 视为一次
//!   投影，名字去掉 `.weight` 后缀，**编码取自条目自己的 `dtype`**；**其余条目进
//!   [`Stack::uncovered`]，绝不静默丢弃**。
//!
//! 「未覆盖」与「无法计价」是两件事，分别记录：
//!
//! | 清单 | 含义 |
//! |---|---|
//! | [`Stack::uncovered`] | 根本不是矩阵（范数、偏置、位置编码……），不构成一次投影 |
//! | [`Floor::unpriced`] | 是投影，但推不出下界：形状没有合法内核，或这种权重编码还没有算子 |
//!
//! 两者都**不会**让整张模型的下界变成错误。一个 LoRA 适配器（`[4096, 16]`，16 不是量化分组的
//! 整数倍）或一张 f32 权重（还没有浮点投影算子）只让自己无法计价，其余层照常相加——这正是
//! "允许某个组件没有下界"的含义。
//!
//! 维度约定与 [`Linear`] 一致：`shape = [out_features, in_features]`，即 safetensors 里的
//! `weight` 按 PyTorch 的行主序形状书写。
//!
//! # 例子
//!
//! ```
//! use llm_nn::{Linear, Stack};
//!
//! let model = Stack::new(vec![Linear::new("blk.0.attn_q", 4096, 4096).expect("合法")]).expect("名字唯一");
//! let floor = model.floor();
//! assert!(floor.total().read() > 8 * 1024 * 1024, "int4 权重本身就有 8 MiB");
//! assert!(floor.unpriced().is_empty());
//! assert!(model.uncovered().is_empty());
//! ```

use std::collections::BTreeMap;
use std::fmt;

use diagnostic::{Code, Error, Result};
use tensor::Shape;
use weights::{Entry, Header};

use crate::recurrent::{Cell, Recurrent, RecurrentTensor};
use crate::{Bounds, Contract, Linear, WeightFormat};

/// 权重条目名里表示"这是一个权重矩阵"的后缀；层名去掉它，才能与其它按名查找的地方对齐。
pub const WEIGHT_SUFFIX: &str = ".weight";

/// 一条"是投影、但推不出下界"的层：名字 + 原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unpriced {
    name: String,
    reason: String,
}

impl Unpriced {
    fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// 层名。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 为什么没有下界。
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for Unpriced {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}（{}）", self.name, self.reason)
    }
}

/// 一次模型计价的结果：能加的那部分 + 加不进去的那部分。
#[derive(Debug, Clone, PartialEq)]
pub struct Floor {
    total: Bounds,
    unpriced: Vec<Unpriced>,
}

impl Floor {
    /// 各层下界之和。无法计价的层不出现在这个数里。
    pub fn total(&self) -> Bounds {
        self.total
    }

    /// 无法计价的层，保持构造顺序。
    pub fn unpriced(&self) -> &[Unpriced] {
        &self.unpriced
    }
}

/// 一串层及其总下界。
pub struct Stack {
    layers: Vec<Box<dyn Contract>>,
    uncovered: Vec<String>,
}

impl fmt::Debug for Stack {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Stack")
            .field(
                "layers",
                &self
                    .layers
                    .iter()
                    .map(|layer| layer.name())
                    .collect::<Vec<_>>(),
            )
            .field("uncovered", &self.uncovered)
            .finish()
    }
}

impl Stack {
    /// 由一串声明好的线性层构造。
    ///
    /// # Errors
    ///
    /// 层名重复时返回 [`Code::Invalid`]：后面的遍历（参数表、切分、下界归因）都按名字对齐，
    /// 重名会让"到底算的是哪一层"变成猜测。
    pub fn new(layers: Vec<Linear>) -> Result<Self> {
        Self::of(
            layers
                .into_iter()
                .map(|layer| Box::new(layer) as Box<dyn Contract>)
                .collect(),
        )
    }

    /// 由任意一串层构造（可以把注解层与手写层混在一起）。
    ///
    /// 收 `Box<dyn Contract>` 而不是泛型 `L: Contract`：一个栈里既要放 [`Linear`] 这种按模型配置
    /// 构造的层，也要放 `#[spec]` 注解出来的零大小类型，两者没有共同的静态类型。
    ///
    /// # Errors
    ///
    /// 层名重复时返回 [`Code::Invalid`]（理由同 [`Stack::new`]）。
    pub fn of(layers: Vec<Box<dyn Contract>>) -> Result<Self> {
        unique_names(layers.iter().map(|layer| layer.name()))?;
        Ok(Self {
            layers,
            uncovered: Vec::new(),
        })
    }

    /// 按权重文件的条目推出层清单。
    ///
    /// 秩为 2 的条目 → 一次 [`Linear`]（`[out, in]`），**权重编码由条目自己的 `dtype` 决定**
    /// （[`WeightFormat::of_dtype`]）；秩不是 2 的条目 → 未覆盖清单。
    ///
    /// dtype 只说明"元素怎么存"，不说明"多少个元素共享一个缩放因子"。因此 4 位打包的条目只会
    /// 得到 [`WeightFormat::Packed4`]：层照样被列出来，但 [`Stack::floor`] 会明确报"分组未声明、
    /// 无法计价"。缺什么就说什么，不拿一个默认分组去凑。
    ///
    /// # Errors
    ///
    /// - 条目形状为 `[0, n]` / `[n, 0]`：宽度为零的投影没有意义（[`Code::Invalid`]）；
    /// - 去掉 `.weight` 后层名重复（[`Code::Invalid`]）。
    pub fn from_header(header: &Header) -> Result<Self> {
        let scales = header.scales_names()?;
        let mut layers: Vec<Box<dyn Contract>> = Vec::new();
        let mut uncovered = Vec::new();
        classify(header, &scales, &mut layers, &mut uncovered)?;
        unique_names(layers.iter().map(|layer| layer.name()))?;
        Ok(Self { layers, uncovered })
    }

    /// 全部层，保持权重文件里的偏移顺序（构造顺序）。
    pub fn layers(&self) -> &[Box<dyn Contract>] {
        &self.layers
    }

    /// 无法归因为投影的条目名（范数、偏置、位置编码……）。
    pub fn uncovered(&self) -> &[String] {
        &self.uncovered
    }

    /// 总下界：各层下界之和，加上无法计价的层的清单。
    pub fn floor(&self) -> Floor {
        let mut total = Bounds::ZERO;
        let mut unpriced = Vec::new();
        for layer in &self.layers {
            match layer.bounds() {
                Ok(Some(bounds)) => total = total.plus(bounds),
                Ok(None) => unpriced.push(Unpriced::new(
                    layer.name(),
                    "没有可计价的形状（缺输入宽度）",
                )),
                Err(error) => unpriced.push(Unpriced::new(layer.name(), error.to_string())),
            }
        }
        Floor { total, unpriced }
    }
}

/// 条目归到两侧：能认成循环层的按层归组，其余逐条判投影。
fn classify(
    header: &Header,
    scales: &[&str],
    layers: &mut Vec<Box<dyn Contract>>,
    uncovered: &mut Vec<String>,
) -> Result<()> {
    let (groups, rest) = partition(header, scales);
    let mut found = layers_of(groups, uncovered);
    found.extend(projections_of(header, rest, uncovered)?);
    // 认出来的层与投影按条目在文件里的偏移排回去：下界归因与切分都按这个顺序读。
    found.sort_by_key(|(at, _)| *at);
    layers.extend(found.into_iter().map(|(_, layer)| layer));
    Ok(())
}

/// 分组：一次投影的标度条目跳过，循环件名按层名归组，其余留着逐条判投影。
fn partition<'a>(
    header: &'a Header,
    scales: &[&str],
) -> (BTreeMap<&'a str, (usize, Vec<&'a Entry>)>, Vec<(usize, &'a Entry)>) {
    let mut groups: BTreeMap<&str, (usize, Vec<&Entry>)> = BTreeMap::new();
    let mut rest: Vec<(usize, &Entry)> = Vec::new();
    for (at, entry) in header.entries().iter().enumerate() {
        if scales.contains(&entry.name()) {
            continue;
        }
        match recurrent_parts(entry.name()) {
            Some((block, ..)) => groups
                .entry(block)
                .or_insert_with(|| (at, Vec::new()))
                .1
                .push(entry),
            None => rest.push((at, entry)),
        }
    }
    (groups, rest)
}

/// 认出来的循环层；认不出就一件都不认——半认会把一半权重算进下界、另一半丢进清单。
fn layers_of(
    groups: BTreeMap<&str, (usize, Vec<&Entry>)>,
    uncovered: &mut Vec<String>,
) -> Vec<(usize, Box<dyn Contract>)> {
    groups
        .into_iter()
        .filter_map(|(block, (at, group))| match recurrent_layer(block, &group) {
            Some(layer) => Some((at, Box::new(layer) as Box<dyn Contract>)),
            None => {
                uncovered.extend(group.iter().map(|entry| entry.name().to_string()));
                None
            }
        })
        .collect()
}

/// 其余条目逐条判投影：认得出的进层清单，认不出的进未覆盖清单。
fn projections_of(
    header: &Header,
    rest: Vec<(usize, &Entry)>,
    uncovered: &mut Vec<String>,
) -> Result<Vec<(usize, Box<dyn Contract>)>> {
    let mut found: Vec<(usize, Box<dyn Contract>)> = Vec::new();
    for (at, entry) in rest {
        match projection(header, entry)? {
            Some(layer) => found.push((at, Box::new(layer))),
            None => uncovered.push(entry.name().to_string()),
        }
    }
    Ok(found)
}

/// 循环权重条目的拆解：把 `blk.lstm.weight_ih_l0_reverse` 拆成 `(层名, 件, 层号, 方向)`。
///
/// 从右往左拆，而不是拿固定后缀反复 `ends_with`：层号与方向都是变长的，右拆一次就拿到；
/// 后缀不落在四件之内（比如 `blk.0.mix_l2`）就返回 `None`，不会被误认成循环权重。
fn recurrent_parts(entry: &str) -> Option<(&str, RecurrentTensor, usize, usize)> {
    let (stem, dir) = split_dir(entry);
    let (stem, level) = stem.rsplit_once("_l")?;
    let (block, suffix) = stem.rsplit_once('.')?;
    let tensor = RecurrentTensor::from_suffix(suffix)?;
    let level: usize = level.parse().ok()?;
    (!block.is_empty()).then_some((block, tensor, level, dir))
}

/// 方向：`_reverse` 后缀是反向（1），其余是正向（0）。
fn split_dir(entry: &str) -> (&str, usize) {
    match entry.strip_suffix("_reverse") {
        Some(stem) => (stem, 1),
        None => (entry, 0),
    }
}

/// 由一组同前缀的条目认出一次循环层；认不出返回 `None`。
///
/// 判据不是「名字像」，而是**清单能重新生成**：构造出来之后按 [`Recurrent::weight_entries`]
/// 把名字与形状重新生成一遍，与检查点里的一条不差才认。认的是「这份权重就是这一层」，
/// 不是「它大概像这一层」。
fn recurrent_layer(block: &str, entries: &[&Entry]) -> Option<Recurrent> {
    let layer = build_layer(block, entries)?;
    let mut wanted = layer.weight_entries();
    let mut actual = recovered(block, entries);
    // 按名字排序后逐条比对：条目名在候选里唯一，名字序也就唯一。
    wanted.sort_by(|a, b| a.0.cmp(&b.0));
    actual.sort_by(|a, b| a.0.cmp(&b.0));
    (wanted == actual).then_some(layer)
}

/// 由同前缀条目构造一层：`weight_hh` 的行列比定门数、它的第二维定隐宽、层号与方向的
/// 最大值定规模，`bias_ih` 在不在定有没有偏置。
fn build_layer(block: &str, entries: &[&Entry]) -> Option<Recurrent> {
    let (rows, hidden) = matrix_of(block, entries, RecurrentTensor::WeightHh)?;
    let (_, input) = matrix_of(block, entries, RecurrentTensor::WeightIh)?;
    let gates = gates_of(rows, hidden)?;
    let (layers, dirs) = extent_of(block, entries);
    let bias = shape_of(block, entries, RecurrentTensor::BiasIh, 0, 0).is_some();
    Recurrent::new(
        block,
        Cell::of_gates(gates)?,
        input,
        hidden,
        layers,
        dirs,
        bias,
    )
    .ok()
}

/// 某一件某层某方向的二维形状；缺这一件或不是二维就返回 `None`。
fn matrix_of(
    block: &str,
    entries: &[&Entry],
    tensor: RecurrentTensor,
) -> Option<(usize, usize)> {
    match shape_of(block, entries, tensor, 0, 0)?.as_slice() {
        [rows, cols] => Some((*rows, *cols)),
        _ => None,
    }
}

/// 某一件某层某方向的形状。
fn shape_of(
    block: &str,
    entries: &[&Entry],
    tensor: RecurrentTensor,
    level: usize,
    dir: usize,
) -> Option<Vec<usize>> {
    entries.iter().find_map(|entry| {
        let (prefix, kind, at, side) = recurrent_parts(entry.name())?;
        (prefix == block && kind == tensor && at == level && side == dir)
            .then(|| entry.shape().dims().to_vec())
    })
}

/// 门数：`weight_hh` 的行数必须是隐宽的整数倍，否则不是一次门堆叠。
fn gates_of(rows: usize, hidden: usize) -> Option<usize> {
    (hidden != 0)
        .then(|| rows / hidden)
        .filter(|gates| gates * hidden == rows)
}

/// 层数与方向数：条目里出现的 `(层号, 方向)` 最大值加一。
fn extent_of(block: &str, entries: &[&Entry]) -> (usize, usize) {
    entries
        .iter()
        .filter_map(|entry| recurrent_parts(entry.name()))
        .filter(|(prefix, ..)| *prefix == block)
        .fold((0, 0), |(layers, dirs), (_, _, level, dir)| {
            (layers.max(level + 1), dirs.max(dir + 1))
        })
}

/// 条目里属于这一块的那些 `(名字, 形状)`。
fn recovered(block: &str, entries: &[&Entry]) -> Vec<(String, Shape)> {
    entries
        .iter()
        .filter(|entry| {
            matches!(recurrent_parts(entry.name()), Some((prefix, ..)) if prefix == block)
        })
        .map(|entry| (entry.name().to_string(), entry.shape().clone()))
        .collect()
}

/// 一个条目能否被视为一次投影：能则给出层，不能则返回 `None`（进未覆盖清单）。
fn projection(header: &Header, entry: &Entry) -> Result<Option<Linear>> {
    let name = layer_name(entry.name());
    // 量化条目先问标度：宽度、分组与位宽都从那一处来；这是本函数里唯一需要"先问"的理由。
    if let Some(quantized) = header.quantization(entry.name())? {
        let format = quantized_format(quantized.group(), quantized.bits());
        return Linear::with_format(
            name,
            format,
            quantized.in_features(),
            quantized.out_features(),
        )
        .map(Some)
        .map_err(|error| error.with(format!("权重条目 `{}`", entry.name())));
    }
    let [out_features, in_features] = entry.shape().dims() else {
        return Ok(None);
    };
    let Some(format) = WeightFormat::of_dtype(entry.dtype()) else {
        return Ok(None);
    };
    Linear::with_format(name, format, *in_features, *out_features)
        .map(Some)
        .map_err(|error| error.with(format!("权重条目 `{}`", entry.name())))
}

/// 量化条目按位宽选编码：逐元素 int8（8 位）走 `gemv_i8`，打包 nibble（4 位）走 `gemv_i4`。
fn quantized_format(group: usize, bits: u8) -> WeightFormat {
    if bits == 8 {
        WeightFormat::int8(group)
    } else {
        WeightFormat::int4(group)
    }
}

/// 层名：去掉 [`WEIGHT_SUFFIX`]，其它名字原样保留。
fn layer_name(entry: &str) -> &str {
    entry
        .strip_suffix(WEIGHT_SUFFIX)
        .filter(|name| !name.is_empty())
        .unwrap_or(entry)
}

fn unique_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(Error::new(
                Code::Invalid,
                format!("层名 `{name}` 重复：下界与参数表都按名字对齐，重名无法归因"),
            )
            .with("nn::Stack"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/stack_tests.rs"]
mod tests;
