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

use std::fmt;

use diagnostic::{Code, Error, Result};
use weights::{Entry, Header};

use crate::{Bounds, Contract, Linear, WeightFormat};

/// 权重条目名里表示"这是一个权重矩阵"的后缀；层名去掉它，才能与其它按名查找的地方对齐。
pub const WEIGHT_SUFFIX: &str = ".weight";

/// 一条"是投影、但推不出下界"的层：名字 + 原因。
///
/// 原因不是装饰：`floor()` 的返回值要和 `uncovered` 一起被打印出来，读者才能判断"模型下界偏低"
/// 是因为多了一张 LoRA 表、一张 f32 权重，还是因为某个形状确实没有合法内核。
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
///
/// 没有 `Result`：无法计价不是错误，而是**必须被如实报告**的事实。真正的错误（权重文件损坏、
/// 层重名、零宽投影）在构造 [`Stack`] 时就已经报了。
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
    ///
    /// 单层的 `Err`（形状没有合法内核）或 `None`（没有可计价的形状）只让**那一层**进清单，
    /// 不会让整张模型算不出来：这样一张 LoRA 表不会把其余 300 层的和一起清零。
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

/// 条目归到两侧：一次投影进层清单，其余进未覆盖清单。
///
/// `scales` 里的标度条目不是层：它的分组已经算进某条权重那一层的流量（算子语义里 `ng` 那条
/// 派生轴），单独列进"未覆盖"只会让读者以为漏了东西，所以它两边都不进。
fn classify(
    header: &Header,
    scales: &[&str],
    layers: &mut Vec<Box<dyn Contract>>,
    uncovered: &mut Vec<String>,
) -> Result<()> {
    for entry in header.entries() {
        if scales.contains(&entry.name()) {
            continue;
        }
        match projection(header, entry)? {
            Some(layer) => layers.push(Box::new(layer)),
            None => uncovered.push(entry.name().to_string()),
        }
    }
    Ok(())
}

/// 一个条目能否被视为一次投影：能则给出层，不能则返回 `None`（进未覆盖清单）。
///
/// 编码的判定分两步，顺序不能反：
///
/// 1. 文件里有没有这条权重的**标度伙伴**？有，分组就从标度的形状读出来（[`Header::quantization`]），
///    于是 `WeightFormat::int4 { group }` 是实测值而不是默认值；
/// 2. 没有伙伴，才退回按元素类型判断（f32 / f16 / i8 / 无分组的 4 位打包）。
///
/// 反过来的顺序（先按 dtype 定格式、再尝试修正）会让"默认 32 个一组"混进真实分组不同的模型里，
/// 得到的下界看起来有数、其实算错了流量。
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
