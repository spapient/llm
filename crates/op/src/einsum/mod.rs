//! 语义层：一条爱因斯坦求和表达式 + 轴长度 + 操作数位宽 → [`Cost`]（flops 与最少字节）。
//!
//! 这一层回答的是**与设备、与 tile、与片段都无关**的问题：这个算子至少要算多少次浮点、至少
//! 要搬多少字节。它不认识 `cutile`、`prof`、`prim`，因此可以在任何机器上离线复现——roofline
//! 的分子从此只有一个来源，而不是每个算子各写一遍公式。
//!
//! **拥有**：[`Expr`]、[`Extents`]、[`Axis`] / [`Axes`]、[`Operand`] / [`OperandSpec`]、[`Bits`]、
//! [`Cost`]，以及四条骨架共用的表达式缓存 [`FormulaCache`]。
//! **不拥有**：设备峰值（[`prof`]）、roofline 的编排（`cutile::plan`）、算子声明与样本（[`crate`]）。
//! **依赖**：`diagnostic`（全仓唯一的错误类型）。**被依赖**：`op` 的算子语义（用本模块推导代价）。
//!
//! 它原是一个独立成员（`llm-einsum`），唯一消费者是 `op`；并入后 `Cost` 与 `OpDecl` 同处一个
//! crate，算子文件不必再为"代价来自哪里"跨越 crate 边界，`op::model` 的再导出也不再是补丁。
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
//! use llm_op::einsum::{Bits, Expr, Extents, Formula, Operand};
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

mod cache;
mod expr;
mod extents;
mod formula;

pub use cache::FormulaCache;
pub use expr::Expr;
pub use extents::Extents;
pub use formula::{Access, Axes, Axis, AxisRole, Formula, Operand, OperandSpec};

/// 元素位宽闭集（bit）。
///
/// 闭集而不是 `u32`：位宽决定流量怎么算（`i4` 是半个字节），随手写一个 `7` 不该通过。
/// 加一个变体要同步 [`Bits::ALL`] 与 [`Bits::bits`]——它们是这条口径的对外契约。
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
}

/// `Bits` 的词汇表由 [`prof::atom::Closed`] 单点声明，`ALL` / `as_str` / `parse` 由
/// [`prof::inherent_names!`] 以同名固有项转发。
///
/// 并入 `op` 之后不再保留自己的 `ALL` 数组：`op` 在 `prof` 下游，`tests/contracts.rs` 要求
/// 下游成员把名字表交给 `Closed`（`lints/vocabulary.toml` 的 `[closed]` 白名单不给 `op` 开口子）。
impl prof::atom::Closed for Bits {
    const ALL: &'static [Self] = &[
        Self::Four,
        Self::Eight,
        Self::Sixteen,
        Self::ThirtyTwo,
        Self::SixtyFour,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Four => "4",
            Self::Eight => "8",
            Self::Sixteen => "16",
            Self::ThirtyTwo => "32",
            Self::SixtyFour => "64",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

prof::inherent_names!(Bits);

impl Bits {
    /// 每元素占多少位。
    pub const fn bits(self) -> u32 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
            Self::ThirtyTwo => 32,
            Self::SixtyFour => 64,
        }
    }

    /// 给定元素个数需要多少字节，**向上取整**。
    ///
    /// 打包权重按 nibble 计费：`3` 个 int4 元素要 2 字节，而不是 1 字节。
    pub const fn bytes_for(self, elements: u128) -> u128 {
        (elements * self.bits() as u128).div_ceil(8)
    }
}

/// 一次计算的标量代价，roofline 的分子（cutile README §12.1）。
///
/// 流量按**方向**分账：读与写的上限不是同一个数，把它们加成一个总量之后，任何"这层到底卡在
/// 哪一侧"的结论都失去了依据，而"把读挪成写、总量不变"也会变得看起来等价（`nn::Bounds`
/// 的 `moves_within` 正是靠这个区分来发现融合是否真的更省）。
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
    ///
    /// 这是**派生**量，用来打印与对齐旧口径。判断 roofline 卡在哪一侧必须分别拿
    /// [`Cost::read`] 与 [`Cost::write`] 去比峰值，而不是拿这个和。
    pub fn bytes(&self) -> u64 {
        self.read.saturating_add(self.write)
    }

    /// 两笔代价相加：复合算子的总代价。
    ///
    /// 每个方向各自相加——这正是"复合组件的下界在**每个方向**上都不高于子组件之和"那句话
    /// 里的那个和。
    pub fn plus(self, other: Self) -> Self {
        Self {
            flops: self.flops + other.flops,
            read: self.read.saturating_add(other.read),
            write: self.write.saturating_add(other.write),
        }
    }
}

#[cfg(test)]
mod tests;
