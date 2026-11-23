//! 语义层：一条爱因斯坦求和表达式 + 轴长度 + 操作数位宽 → [`Cost`]（flops 与最少字节）。
//!
//! 这一层回答的是**与设备、与 tile、与片段都无关**的问题：这个算子至少要算多少次浮点、至少
//! 要搬多少字节。它不认识 `cutile`、设备档案与片段，因此可以在任何机器上离线复现——roofline
//! 的分子从此只有一个来源，而不是每个算子各写一遍公式。
//!
//! **拥有**：[`Expr`]、[`Extents`]、[`Axis`] / [`Axes`]、[`Operand`] / [`OperandSpec`]、[`Bits`]、
//! [`Cost`]，以及四条骨架共用的表达式缓存 [`FormulaCache`]。
//! **不拥有**：设备峰值（[`crate::prof`]）、roofline 的编排（`cutile::plan`）、算子声明与样本（[`crate`]）。
//! **依赖**：`diagnostic`（全仓唯一的错误类型）。**被依赖**：`op` 的算子语义（用本模块推导代价）。
//!
//! 它原是一个独立成员（`llm-einsum`），唯一消费者是 `op`；并入后 `Cost` 与 `OpDecl` 同处一个
//! crate，算子文件不必再为"代价来自哪里"跨越 crate 边界，`op::model` 的再导出也不再是补丁。
//!
//! 本模块原先按职责拆成 `einsum/` 下的 `expr` / `extents` / `formula` / `formula/axes` / `cache`
//! 五个文件，但它们共用同一条表达式与同一张轴表，彼此互相 import；现在平铺为单个文件，`mod`、
//! `pub use` 与跨文件 `use` 都不再需要。
//!
//! # 表达式
//!
//! 文法与 `numpy.einsum` 的**显式**模式一致，只保留能穷举的那部分：
//!
//! ```text
//! expr   = term { "," term } "->" output
//! term   = { label }                # 空项表示标量缓冲
//! output = { label }
//! label  = "a".."z" | "A".."Z"
//! ```
//!
//! 不支持的三种写法都是刻意的：
//!
//! - **省略号（广播）**：广播让同一个字母在不同操作数里有不同长度，代价模型就必须开始猜形状；
//! - **同一项里重复字母（对角提取）**：那是索引变换而不是算子语义，应当由调用方先 reshape；
//! - **省略 `->`（隐式模式）**：输出轴顺序变成"输入出现顺序的排序规则"，读者与机器都会各猜一套。
//!
//! 三件都能用两条显式式子改写出来，而支持它们会让这份文档变成唯一事实源之外的第二个解释器。
//!
//! **输出缓冲同样是左侧的一项**：`ij,jk,ik->ik` 表示 `C = A·B`，第三项就是 `C` 的轴。这样每个
//! 操作数的位宽与读写方向都有出处，输出不必在表达式之外再声明一次形状。
//!
//! # 代价口径
//!
//! | 量 | 口径 |
//! |---|---|
//! | `flops` | 有规约轴时 `2 × Π(参与运算的轴)`（每个规约项一次乘加）；没有规约轴时 `Π(输出轴)`（每个输出元素一次运算） |
//! | `read` | 只读操作数各自的 `ceil(Π(自己的轴) × 位宽 / 8)` 之和 |
//! | `write` | 只写操作数各自的 `ceil(Π(自己的轴) × 位宽 / 8)` 之和（至少写一遍输出） |
//!
//! 读与写**分开给**而不是合成一个 `bytes`：它们的上限不是同一个（只流式读权重的 GEMV 可以
//! 跑过设备的拷贝上限），合成之后"这层是读受限还是写受限"就再也判断不了。`Cost::bytes()` 仍然
//! 提供合计值，但它是**派生**量，不是口径本身。
//!
//! 位宽用 [`Bits`] 闭集表达，因此 `i4` 权重按 nibble 计费，而不是被当成一个 `f32` 或一个 `u8`。
//!
//! 有一类轴必须被排除在运算次数之外：**分组轴**。量化 GEMV 的 `scales[n, g]` 里，`g` 是 `m`
//! 每 `GROUP` 个元素切出来的一面，遍历 `g` 与遍历 `m` 是同一批元素的两层看法；把它乘进
//! `Π(轴)` 会让 `flops` 多出一个因子。这类轴用 [`Extents::group`] 声明为派生轴：长度由基准轴
//! 推出，只计流量、不计运算次数。
//!
//! # 例子
//!
//! ```
//! use llm_prof::einsum::{Bits, Expr, Extents, Formula, Operand};
//!
//! // int4 权重、按 32 个元素一组缩放的 GEMV：
//! let expr = Expr::parse("nm,m,ng,n->n").expect("表达式合法");
//! let formula = Formula::new(
//!     expr,
//!     vec![
//!         Operand::read("w", Bits::Four),
//!         Operand::read("x", Bits::ThirtyTwo),
//!         Operand::read("scales", Bits::ThirtyTwo),
//!         Operand::write("y", Bits::ThirtyTwo),
//!     ],
//! )
//! .expect("操作数与表达式一一对应");
//!
//! // `g` 是 `m` 每 32 个元素一组切出来的派生轴：长度由 `m` 推出，不参与运算次数。
//! let extents = Extents::new([('n', 8), ('m', 64)])
//!     .expect("轴长度")
//!     .group('g', 'm', 32)
//!     .expect("64 能被 32 整除");
//! let cost = formula.cost(&extents).expect("可估算");
//!
//! assert_eq!(cost.flops, 2.0 * 8.0 * 64.0);
//! // w: 8·64 元素 × 4 bit/8；x: 64 × 4；scales: 8·2 × 4。
//! assert_eq!(cost.read, 8 * 64 / 2 + 64 * 4 + 8 * 2 * 4);
//! // 只有一个输出缓冲，所以写方向只有它。
//! assert_eq!(cost.write, 8 * 4);
//! ```

use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

// ── 位宽 ────────────────────────────────────────────

/// 元素位宽闭集（bit）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bits {
    /// 4 位（打包的 int4 权重）。
    Four,
    /// 8 位。
    Eight,
    /// 16 位（f16 / bf16 / i16）。
    Sixteen,
    /// 32 位（f32 / i32）。
    ThirtyTwo,
    /// 64 位（f64 / i64）。
    SixtyFour,
    /// 2 位（打包的 int2 权重，每字节四个码）。
    Two,
}

/// 与 [`Bits::ALL`] 同序的名字表；下标即判别值，加变体时与 `ALL`、[`WIDTHS`] 三处对齐。
const NAMES: [&str; Bits::ALL.len()] = ["4", "8", "16", "32", "64", "2"];

/// 与 [`Bits::ALL`] 同序的位宽表；下标即判别值。
const WIDTHS: [u32; Bits::ALL.len()] = [4, 8, 16, 32, 64, 2];

/// `Bits` 的词汇表由 [`crate::atom::Closed`] 单点声明，`ALL` / `as_str` / `parse` 由
/// [`crate::inherent_names!`] 以同名固有项转发。
impl crate::atom::Closed for Bits {
    const ALL: &'static [Self] = &[
        Self::Four,
        Self::Eight,
        Self::Sixteen,
        Self::ThirtyTwo,
        Self::SixtyFour,
        Self::Two,
    ];

    fn as_str(self) -> &'static str {
        NAMES[self as usize]
    }

    fn idx(self) -> usize {
        self as usize
    }
}

crate::inherent_names!(Bits);

impl Bits {
    /// 每元素占多少位。
    pub const fn bits(self) -> u32 {
        WIDTHS[self as usize]
    }

    /// 给定元素个数需要多少字节，**向上取整**。
    pub const fn bytes_for(self, elements: u128) -> u128 {
        (elements * self.bits() as u128).div_ceil(8)
    }
}

// ── 代价 ────────────────────────────────────────────

/// 一次计算的标量代价，roofline 的分子（cutile README §12.1）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// 浮点运算次数。
    pub flops: f64,
    /// 最少读入字节：算子入口的输入。
    pub read: u64,
    /// 最少写出字节：算子出口的输出。
    pub write: u64,
}

impl Cost {
    /// 零代价：空的表达式或纯标量计算。
    pub const ZERO: Self = Self {
        flops: 0.0,
        read: 0,
        write: 0,
    };

    /// 由三个分量构造（便于测试与适配旧的调用点）。
    pub const fn new(flops: f64, read: u64, write: u64) -> Self {
        Self { flops, read, write }
    }

    /// 最少流量（字节）：读与写合计。
    pub fn bytes(&self) -> u64 {
        self.read.saturating_add(self.write)
    }

    /// 两笔代价相加：复合算子的总代价。
    pub fn plus(self, other: Self) -> Self {
        Self {
            flops: self.flops + other.flops,
            read: self.read.saturating_add(other.read),
            write: self.write.saturating_add(other.write),
        }
    }
}

// ── 表达式 ──────────────────────────────────────────

/// 解析后的爱因斯坦求和表达式：若干输入项 + 一个输出项。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    text: String,
    terms: Vec<Vec<char>>,
    output: Vec<char>,
}

impl Expr {
    /// 输出分隔符。
    pub const ARROW: &'static str = "->";

    /// 解析显式模式的表达式，例如 `"nm,m,ng,n->n"`。
    ///
    /// # Errors
    ///
    /// - 表达式为空、缺 `->` 或出现多个 `->`；
    /// - 某个字母不是 `a..z` / `A..Z`；
    /// - 同一项里出现重复字母（对角提取不在本层支持范围内）；
    /// - 输出项里有重复字母，或输出项里的字母没有出现在任何输入项里。
    pub fn parse(text: &str) -> Result<Self, Error> {
        let trimmed = text.trim();
        let (inputs, output) = sides(trimmed)?;
        let terms = inputs
            .split(',')
            .enumerate()
            .map(|(idx, term)| term_labels(term, idx))
            .collect::<Result<Vec<_>, _>>()?;
        let output = output_labels(output)?;
        check_output(&terms, &output, trimmed)?;

        Ok(Self {
            text: trimmed.to_string(),
            terms,
            output,
        })
    }

    /// 原文（去掉首尾空白），用于诊断。
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 输入项的个数，即操作数的个数。
    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    /// 逐个输入项的字母，顺序与操作数一致。
    pub fn terms(&self) -> impl Iterator<Item = &[char]> + '_ {
        self.terms.iter().map(Vec::as_slice)
    }

    /// 输出项的字母，顺序即输出的轴顺序。
    pub fn output(&self) -> &[char] {
        &self.output
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl FromStr for Expr {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

/// 轴字母的闭集：`a..z` / `A..Z`；[`Extents`] 也用这一条。
fn is_label(label: char) -> bool {
    label.is_ascii_alphabetic()
}

/// 不是合法轴字母时的公共措辞（调用点负责补上下文）。
const NOT_LABEL: &str = "不是字母（闭集是 `a..z` / `A..Z`）";

fn bad(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("einsum::Expr::parse")
}

/// `->` 的左右两侧：表达式非空，箭头恰好一个。
fn sides(trimmed: &str) -> Result<(&str, &str), Error> {
    if trimmed.is_empty() {
        return Err(bad("表达式为空"));
    }
    if trimmed.matches(Expr::ARROW).count() > 1 {
        return Err(bad(format!(
            "`->` 只能出现一次：`{trimmed}`（要拆成多个算子，请分别声明）"
        )));
    }
    trimmed.split_once(Expr::ARROW).ok_or_else(|| {
        bad(format!(
            "缺少 `->`：`{trimmed}`（显式模式必须写出输出轴，例如 `nm,m->n`）"
        ))
    })
}

fn term_labels(term: &str, idx: usize) -> Result<Vec<char>, Error> {
    let mut labels = Vec::with_capacity(term.len());
    for label in term.chars() {
        if !is_label(label) {
            return Err(bad(format!(
                "第 {idx} 项（`{term}`）：`{label}` {NOT_LABEL}"
            )));
        }
        if labels.contains(&label) {
            return Err(bad(format!(
                "第 {idx} 项（`{term}`）里字母 `{label}` 重复：对角提取请先 reshape，再写成两条式子"
            )));
        }
        labels.push(label);
    }
    Ok(labels)
}

fn output_labels(output: &str) -> Result<Vec<char>, Error> {
    let mut labels = Vec::with_capacity(output.len());
    for label in output.chars() {
        if !is_label(label) {
            return Err(bad(format!("输出项（`{output}`）：`{label}` {NOT_LABEL}")));
        }
        if labels.contains(&label) {
            return Err(bad(format!(
                "输出项（`{output}`）里字母 `{label}` 重复：输出轴必须唯一"
            )));
        }
        labels.push(label);
    }
    Ok(labels)
}

fn check_output(terms: &[Vec<char>], output: &[char], text: &str) -> Result<(), Error> {
    for label in output {
        if !terms.iter().any(|term| term.contains(label)) {
            return Err(bad(format!(
                "表达式 `{text}` 的输出轴 `{label}` 没有出现在任何输入项里：算不出来"
            )));
        }
    }
    Ok(())
}

// ── 轴长度表 ────────────────────────────────────────

/// 轴长度表。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Extents {
    entries: BTreeMap<char, Entry>,
}

/// 表中的一条轴：长度，以及（分组轴才有的）分组来历。长度与来历同住一格，声明分组轴时
/// 不再需要把「长度」和「这是什么分组」两处分别写入、再靠调用方保证它们同步。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    /// 元素个数。
    len: usize,
    /// 分组轴的分组来历：`(基准轴, 因子)`；普通轴为 `None`。
    group: Option<(char, usize)>,
}

/// 分组声明的静态判据：轴名合法、不以自己为基准、因子非零。
fn declared(label: char, base: char, divisor: usize) -> Result<(), Error> {
    if !is_label(label) {
        return Err(blame(format!("轴 `{label}` {NOT_LABEL}")));
    }
    if label == base {
        return Err(blame(format!("轴 `{label}` 不能以自己为分组基准")));
    }
    if divisor != 0 {
        return Ok(());
    }
    Err(blame(format!("轴 `{label}` 的分组因子不能为 0")))
}

/// 分组错误的统一边框：类别恒为 `invalid`，上下文固定到 [`Extents::group`]。
fn blame(message: String) -> Error {
    Error::new(Code::Invalid, message).with("einsum::Extents::group")
}

impl Extents {
    /// 由 `(字母, 长度)` 构造。
    ///
    /// # Errors
    ///
    /// - 字母不是 `a..z` / `A..Z`；
    /// - 长度为 `0`（空轴没有代价可言，通常是上游形状算错）；
    /// - 同一个字母给了两次（后者会静默覆盖前者，因此拒绝）。
    pub fn new(pairs: impl IntoIterator<Item = (char, usize)>) -> Result<Self, Error> {
        let mut extents = Self::default();
        for (label, length) in pairs {
            if !is_label(label) {
                return Err(
                    Error::new(Code::Invalid, format!("轴 `{label}` {NOT_LABEL}"))
                        .with("einsum::Extents::new"),
                );
            }
            if length == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("轴 `{label}` 的长度为 0：空轴没有可估算的代价"),
                )
                .with("einsum::Extents::new"));
            }
            let entry = Entry {
                len: length,
                group: None,
            };
            if extents.entries.insert(label, entry).is_some() {
                return Err(
                    Error::new(Code::Invalid, format!("轴 `{label}` 被声明了两次"))
                        .with("einsum::Extents::new"),
                );
            }
        }
        Ok(extents)
    }

    /// 取一个轴的长度。
    pub fn get(&self, label: char) -> Option<usize> {
        self.entries.get(&label).map(|entry| entry.len)
    }

    /// 声明一条**分组轴**：`label` 是 `base` 按 `divisor` 切出来的一面，长度由前者推出。
    ///
    /// 例如量化 GEMV 里 `scales[n, g]` 的 `g` 并不是独立的求和维，而是 `m` 每 `GROUP` 个元素
    /// 一组：`g` 只影响 `scales` 的**流量**（它确实只有 `n · m/GROUP` 个元素），不能参与运算
    /// 次数。用推导而不是手写长度，正是为了让这两件事不再被混为一谈。
    ///
    /// # Errors
    ///
    /// - `label` 不是字母、与 `base` 同名，或已经声明过；
    /// - `base` 没有长度；
    /// - `divisor` 为 `0`，或 `base` 的长度不是它的整数倍。
    pub fn group(mut self, label: char, base: char, divisor: usize) -> Result<Self, Error> {
        declared(label, base, divisor)?;
        let length = self.base_length(label, base, divisor)?;
        let entry = Entry {
            len: length / divisor,
            group: Some((base, divisor)),
        };
        if self.entries.insert(label, entry).is_some() {
            return Err(blame(format!("轴 `{label}` 被声明了两次")));
        }
        Ok(self)
    }

    /// 基准轴的分组长度：基准必须先声明，且长度能被因子整除。
    fn base_length(&self, label: char, base: char, divisor: usize) -> Result<usize, Error> {
        let Some(length) = self.get(base) else {
            return Err(blame(format!(
                "轴 `{label}` 以 `{base}` 为分组基准，但 `{base}` 没有长度（先声明基准轴）"
            )));
        };
        if length.is_multiple_of(divisor) {
            return Ok(length);
        }
        Err(blame(format!(
            "轴 `{base}` 的长度 {length} 不是分组因子 {divisor} 的整数倍"
        )))
    }

    /// 一条轴的分组来历；普通轴返回 `None`。
    pub fn grouped(&self, label: char) -> Option<(char, usize)> {
        self.entries.get(&label).and_then(|entry| entry.group)
    }

    /// 轴的数量（含分组轴）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否一个轴都没有（纯标量表达式）。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按字母序遍历所有轴（顺序稳定，供诊断使用）。
    pub fn labels(&self) -> impl Iterator<Item = char> + '_ {
        self.entries.keys().copied()
    }
}

impl fmt::Display for Extents {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, (label, entry)) in self.entries.iter().enumerate() {
            if idx > 0 {
                formatter.write_str(" ")?;
            }
            write!(formatter, "{label}={}", entry.len)?;
        }
        Ok(())
    }
}

// ── 操作数 ──────────────────────────────────────────

/// 操作数的数据流向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// 只读：算子入口的输入。
    Read,
    /// 只写：算子出口的输出。写入同样计费，因为"至少写一遍输出"是逃不掉的流量。
    Write,
}

/// 表达式里一个操作数的**编译期**形态：名字、元素位宽与读/写方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandSpec {
    /// 缓冲名（与 `.k` 的形参名一致，也是诊断里显示的名字）。
    pub name: &'static str,
    /// 元素位宽。
    pub bits: Bits,
    /// 读写方向。
    pub access: Access,
}

impl OperandSpec {
    /// 只读操作数。
    #[must_use]
    pub const fn read(name: &'static str, bits: Bits) -> Self {
        Self {
            name,
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    #[must_use]
    pub const fn write(name: &'static str, bits: Bits) -> Self {
        Self {
            name,
            bits,
            access: Access::Write,
        }
    }
}

/// f32 一读一写的标准操作数：`a` 读、`y` 写（一元与归约共用同一份）。
#[must_use]
pub const fn f32_io_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// f32 两读一写的标准操作数：`a`、`b` 读、`y` 写（二元逐元素）。
#[must_use]
pub const fn f32_binary_operands() -> [OperandSpec; 3] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::read("b", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// 一个操作数：名字、元素位宽与读/写方向。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    name: String,
    bits: Bits,
    access: Access,
}

impl Operand {
    /// 只读操作数。
    pub fn read(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    pub fn write(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Write,
        }
    }

    /// 按方向构造（方向来自数据时用这个）。
    pub fn new(name: impl Into<String>, bits: Bits, access: Access) -> Self {
        Self {
            name: name.into(),
            bits,
            access,
        }
    }

    /// 名字。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素位宽。
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// 流向。
    pub fn access(&self) -> Access {
        self.access
    }
}

// ── 轴分解 ──────────────────────────────────────────

/// 一条轴在这次计算里的角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 批轴：出现在输出里，且被至少两个**只读**操作数共享。
    Batch,
    /// 自由轴：出现在输出里，且只被一个操作数拥有。
    Free,
    /// 规约轴：不出现在输出里，求和后被消掉。
    Reduced,
    /// 分组轴：由另一条轴按固定因子切出（[`Extents::group`]）。
    Grouped,
}

/// 一条轴：字母、长度与角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    label: char,
    extent: usize,
    role: AxisRole,
}

impl Axis {
    /// 轴字母。
    pub fn label(&self) -> char {
        self.label
    }

    /// 轴长度（元素个数）。
    pub fn extent(&self) -> usize {
        self.extent
    }

    /// 轴角色。
    pub fn role(&self) -> AxisRole {
        self.role
    }
}

/// 表达式的轴分解：输出轴在前（按输出顺序），规约轴在后（按首次出现顺序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axes {
    axes: Vec<Axis>,
}

impl Axes {
    /// 全部轴，顺序稳定（输出轴优先）。
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// 输出轴。
    pub fn output(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| matches!(axis.role, AxisRole::Batch | AxisRole::Free))
    }

    /// 规约轴。
    pub fn reduced(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Reduced)
    }

    /// 分组轴（由其他轴按因子切出，只计流量）。
    pub fn grouped(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Grouped)
    }

    /// 是否有真正的规约轴（决定 `flops` 用哪种口径）。
    pub fn reduces(&self) -> bool {
        self.reduced().next().is_some()
    }

    /// 输出形状（按输出轴的声明顺序）。
    pub fn output_shape(&self) -> Vec<usize> {
        self.output().map(Axis::extent).collect()
    }

    /// 输出元素个数；标量输出为 `1`。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn output_elements(&self) -> Result<u128, Error> {
        self.output()
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 参与运算的轴的乘积，即"展开后的标量运算次数"的分母。
    ///
    /// 分组轴被排除：它是另一条轴的分组看法，不是独立的循环维。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn all_elements(&self) -> Result<u128, Error> {
        self.axes
            .iter()
            .filter(|axis| axis.role != AxisRole::Grouped)
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 运算次数：有规约轴时按"每个规约元素一次乘加（2 flops）"，否则只算输出元素。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn flops(&self) -> Result<f64, Error> {
        if self.reduces() {
            return Ok(2.0 * self.all_elements()? as f64);
        }
        Ok(self.output_elements()? as f64)
    }
}

// ── 代价推导 ────────────────────────────────────────

/// 一份可估算的语义：表达式 + 每个操作数的位宽与方向。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    expr: Expr,
    operands: Vec<Operand>,
}

impl Formula {
    /// 把表达式与操作数配对。
    ///
    /// # Errors
    ///
    /// - 操作数个数与表达式的项数不一致；
    /// - 操作数重名（同一块缓冲会被重复计费）；
    /// - 写入操作数多于一个，或写入的轴不在输出里（写回中间量不是算子语义）。
    pub fn new(expr: Expr, operands: Vec<Operand>) -> Result<Self, Error> {
        if expr.arity() != operands.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表达式 `{expr}` 有 {} 项，但给了 {} 个操作数",
                    expr.arity(),
                    operands.len()
                ),
            )
            .with("einsum::Formula::new"));
        }
        unique(&operands)?;
        single_write(&operands)?;
        writes_only_outputs(&expr, &operands)?;
        Ok(Self { expr, operands })
    }

    /// 表达式。
    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    /// 操作数（顺序与表达式的项一致）。
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// 轴分解：每条轴的长度与角色。
    ///
    /// # Errors
    ///
    /// 表达式里的字母没有给长度、给了长度的字母没出现在表达式里，或乘积超出 `u128`。
    pub fn axes(&self, extents: &Extents) -> Result<Axes, Error> {
        let mut axes = self.output_axes(extents)?;
        self.push_reduced(extents, &mut axes)?;
        self.check_used(extents, &axes)?;
        Ok(Axes { axes })
    }

    /// 输出轴：角色看"有几个读操作数共享它"（多于一个即 batch）。
    fn output_axes(&self, extents: &Extents) -> Result<Vec<Axis>, Error> {
        let mut axes: Vec<Axis> = Vec::with_capacity(extents.len());
        for &label in self.expr.output() {
            let Some((base, divisor)) = extents.grouped(label) else {
                let extent = extent(extents, label)?;
                axes.push(Axis {
                    label,
                    extent,
                    role: self.output_role(label),
                });
                continue;
            };
            return Err(Error::new(
                Code::Invalid,
                format!("轴 `{label}` 是 `{base}` 按 {divisor} 分组的派生轴，不能出现在输出里"),
            )
            .with("einsum::Formula::axes"));
        }
        Ok(axes)
    }

    /// 输出轴的角色：被多个读操作数共享的是 batch，否则是 free。
    fn output_role(&self, label: char) -> AxisRole {
        let shared = self
            .operands
            .iter()
            .zip(self.expr.terms())
            .filter(|(operand, term)| operand.access() == Access::Read && term.contains(&label))
            .count();
        if shared > 1 {
            AxisRole::Batch
        } else {
            AxisRole::Free
        }
    }

    /// 输入项里还没被收进来的轴：派生轴是 [`AxisRole::Grouped`]，其余是 [`AxisRole::Reduced`]。
    fn push_reduced(&self, extents: &Extents, axes: &mut Vec<Axis>) -> Result<(), Error> {
        for term in self.expr.terms() {
            for &label in term {
                if axes.iter().any(|axis| axis.label() == label) {
                    continue;
                }
                let role = extents
                    .grouped(label)
                    .map_or(AxisRole::Reduced, |_| AxisRole::Grouped);
                axes.push(Axis {
                    label,
                    extent: extent(extents, label)?,
                    role,
                });
            }
        }
        Ok(())
    }

    /// 给了长度却没出现在表达式里的轴：多半是笔误。
    fn check_used(&self, extents: &Extents, axes: &[Axis]) -> Result<(), Error> {
        for label in extents.labels() {
            if axes.iter().any(|axis| axis.label() == label) {
                continue;
            }
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "轴 `{label}` 有长度 {} 但没出现在表达式 `{}` 里：多半是笔误",
                    extent(extents, label)?,
                    self.expr
                ),
            )
            .with("einsum::Formula::axes"));
        }
        Ok(())
    }

    /// 计算代价。
    ///
    /// # Errors
    ///
    /// 轴分解失败（见 [`Formula::axes`]）、维度乘积溢出，或流量超出 `u64`。
    pub fn cost(&self, extents: &Extents) -> Result<Cost, Error> {
        let axes = self.axes(extents)?;
        self.estimate(extents, &axes)
    }

    /// 一次算完：flops 与两个方向的流量。
    fn estimate(&self, extents: &Extents, axes: &Axes) -> Result<Cost, Error> {
        let flops = axes.flops()?;
        let (read, write) = self.traffic(extents)?;
        Ok(Cost {
            flops,
            read: bytes(read, "读")?,
            write: bytes(write, "写")?,
        })
    }

    /// 流量：每个操作数按自己的项长 × 位宽计费，读与写分开累加。
    fn traffic(&self, extents: &Extents) -> Result<(u128, u128), Error> {
        let mut read: u128 = 0;
        let mut write: u128 = 0;
        for (operand, term) in self.operands.iter().zip(self.expr.terms()) {
            let elements = term_elements(extents, term)?;
            let bytes = operand.bits().bytes_for(elements);
            let total = match operand.access() {
                Access::Read => &mut read,
                Access::Write => &mut write,
            };
            accumulate(total, bytes)?;
        }
        Ok((read, write))
    }
}

// ── 校验 ────────────────────────────────────────────

/// 同名缓冲只允许出现一次：否则同一块内存会被重复计费。
fn unique(operands: &[Operand]) -> Result<(), Error> {
    let mut names: Vec<&str> = Vec::with_capacity(operands.len());
    for operand in operands {
        if names.contains(&operand.name()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "操作数 `{}` 出现了两次：同名缓冲会被重复计费",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
        names.push(operand.name());
    }
    Ok(())
}

/// 一次计算只有一个出口。
fn single_write(operands: &[Operand]) -> Result<(), Error> {
    let writes = operands
        .iter()
        .filter(|operand| operand.access() == Access::Write)
        .count();
    if writes > 1 {
        return Err(
            Error::new(Code::Invalid, "有多个写入操作数：一次计算只有一个出口")
                .with("einsum::Formula::new"),
        );
    }
    Ok(())
}

/// 写入操作数只能写输出轴：写回中间量不是算子语义。
fn writes_only_outputs(expr: &Expr, operands: &[Operand]) -> Result<(), Error> {
    for (operand, term) in operands.iter().zip(expr.terms()) {
        if operand.access() != Access::Write {
            continue;
        }
        if let Some(label) = term.iter().find(|label| !expr.output().contains(label)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "写入操作数 `{}` 的轴 `{label}` 不在输出里：写回中间量不是算子语义",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
    }
    Ok(())
}

// ── 项元素数算术 ──────────────────────────────────────

/// 一项的元素数：该项每条轴长度的乘积。
fn term_elements(extents: &Extents, term: &[char]) -> Result<u128, Error> {
    term.iter().try_fold(1u128, |total, label| {
        multiply(total, extent(extents, *label)?)
    })
}

/// 按方向累加：两个计数器各自溢出都要报。
fn accumulate(total: &mut u128, bytes: u128) -> Result<(), Error> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}

/// 流量收口到 `u64`：`Cost` 的字段是 `u64`，装不下说明形状不可能是真的。
fn bytes(value: u128, kind: &str) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| {
        Error::new(Code::Invalid, format!("{kind}流量 {value} 字节超出 u64"))
            .with("einsum::Formula::cost")
    })
}

fn extent(extents: &Extents, label: char) -> Result<usize, Error> {
    extents.get(label).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("轴 `{label}` 没有长度：表达式里的每个字母都必须给出轴长度"),
        )
        .with("einsum::Formula::axes")
    })
}

fn multiply(total: u128, length: usize) -> Result<u128, Error> {
    total.checked_mul(length as u128).ok_or_else(overflow)
}

fn overflow() -> Error {
    Error::new(Code::Invalid, "维度乘积或流量超出 u128：形状不可能是真的").with("einsum::Formula")
}

// ── 表达式缓存 ──────────────────────────────────────

/// 一份表达式缓存：构造一次，之后复用到任意形状。
pub struct FormulaCache(OnceLock<Result<Formula, Error>>);

impl FormulaCache {
    /// 空缓存（`static` 初始化用，必须是 `const fn`）。
    #[must_use]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// 取这条表达式：第一次调用时解析并校验，之后直接复用。
    ///
    /// `site` 是错误帧（如 `op::gemv::formula`），由**骨架**给出而不是由格式给出：同一条骨架的
    /// 所有精度共享同一个位置，重复写在每个实现里只会多出几份可以写歪的事实。
    ///
    /// # Errors
    ///
    /// 表达式或操作数表不自洽时返回第一次得到的说明（带 `site` 帧）。
    pub fn get(
        &'static self,
        expr: &'static str,
        operands: &'static [OperandSpec],
        site: &'static str,
    ) -> Result<&'static Formula, Error> {
        self.0
            .get_or_init(|| {
                let operands = operands
                    .iter()
                    .map(|spec| Operand::new(spec.name, spec.bits, spec.access))
                    .collect();
                Expr::parse(expr)
                    .and_then(|expr| Formula::new(expr, operands))
                    .map_err(|error| error.with(site))
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Default for FormulaCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    include!("../tests/spec/einsum_tests.rs");
    include!("../tests/spec/cost_tests.rs");
}
