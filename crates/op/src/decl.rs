//! 声明解析与轴展开（cutile README §4、§10.2 的 ①）。
//!
//! 文本来自声明层（磁盘与内嵌两种来源，见 [`crate::Op`]），本模块只解释内容、不决定数据放在哪。
//! 解析的结果是 [`OpDecl`]：机制层（`cutile`）从这一步之后只消费它，不再看 TOML。

use crate::layout::LayoutSpec;
use crate::sch::Schedule;
use core::fmt;
use diagnostic::{Code, Error};
use prof::atom::Need;

/// 变体数上限（cutile README §4.1）。
pub const MAX_VARIANTS: usize = 8;
/// 轴展开后的 tile 候选上限（cutile README §4.1）。
pub const MAX_TILES: usize = 8;

/// 骨架闭集（cutile README 附录 B）：决定用哪个模板基座。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Skeleton {
    /// 逐元素。
    Elementwise,
    /// 归约。
    Reduce,
    /// 矩阵乘。
    Gemm,
    /// 扫描。
    Scan,
}

impl prof::atom::Closed for Skeleton {
    /// 闭集全量。
    const ALL: &'static [Self] = &[Self::Elementwise, Self::Reduce, Self::Gemm, Self::Scan];

    /// 声明层写法。
    fn as_str(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Reduce => "reduce",
            Self::Gemm => "gemm",
            Self::Scan => "scan",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

prof::inherent_names!(Skeleton);

impl fmt::Display for Skeleton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 一个轴：有限枚举，展开后成为 tile 候选。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axis {
    /// 轴名（也是模板里的槽位名）。
    pub name: String,
    /// 候选取值。
    pub values: Vec<u32>,
}

/// Gemm 骨架的网格映射：声明哪些串行外层循环按网格维拆线程（N 维、每维一个 `线程:标量` 对）。
///
/// 这是「配置化优先」的落点：串行模板的外层循环 `for (int t = 0; t < S; ++t)` 由机制层把循环头
/// 绑定到线程下标（`t = tid.x`、循环只跑一次），正文里的跨步（`i*K`、`p*N`）原样保留——因此既能
/// 覆盖 gemv 的一维行网格，也能覆盖 gemm 的二维网格（右操作数的规约维跨步 `p*N` 不被误压缩）。
/// 线程名必须与模板里的循环变量同名，标量名是该循环的边界；两者都由声明层显式给出，机制层
/// （cutile）只做循环头替换、不索引分析模板。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    /// 网格维（按序）：每项 `(线程名, 边界标量名)`。线程名 = 模板循环变量名，网格大小 = 标量样本值。
    pub dims: Vec<(String, String)>,
}

/// 分块并行声明：把长轴切成 `size` 元素一块，网格 = `ceil(N / size)`，每线程处理一块。
///
/// 与 [`Grid`]（循环头绑定到线程下标、正文跨步原样保留）的分工是「规约轴交给谁」：归约/扫描
/// 的跨元素依赖不能靠逐元素拆线程（每线程会各算一整遍），只能切成连续块、每线程串行处理自己
/// 那一块，再把块级部分结果交给主机折叠（归约）或做 carry 前缀（扫描）。块长由声明层显式给出，
/// 机制层只做 `start = g*size` / `len = min(size, N-start)` 的切块，不索引分析模板。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// 块长（元素/线程）。
    pub size: usize,
}

/// 降级链上的一个变体。
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// 变体名。
    pub name: String,
    /// 优先级，小者优先。
    pub order: u32,
    /// 调度模板。
    pub schedule: Schedule,
    /// 非 MMA 占比上界。
    pub mma_share: f64,
    /// 变体级增量需求：只能**加强** `[needs]`，不能少写模板用到的原语。
    pub extra_needs: Vec<Need>,
}

impl Variant {
    /// 该变体的完整需求 = 算子级公共需求 + 变体级增量需求。
    pub fn needs<'a>(&'a self, op: &'a OpDecl) -> impl Iterator<Item = &'a Need> {
        op.needs.iter().chain(self.extra_needs.iter())
    }
}

/// 一个算子的声明加模板。
#[derive(Debug, Clone, PartialEq)]
pub struct OpDecl {
    /// 算子名。
    pub name: String,
    /// 骨架。
    pub skeleton: Skeleton,
    /// 一句话说明。
    pub summary: String,
    /// 轴。
    pub axes: Vec<Axis>,
    /// 算子级公共需求。
    pub needs: Vec<Need>,
    /// 布局契约：槽位 → 构造子。
    pub layouts: Vec<(String, LayoutSpec)>,
    /// Gemm 骨架的网格映射（仅 Gemm 算子声明；其余骨架不需要）。
    pub grid: Option<Grid>,
    /// 分块并行的块长（reduce/scan 骨架；`None` 表示串行兜底）。
    pub chunk: Option<Chunk>,
    /// 模板常量：`{{槽位}}` 的固定取值（非轴、非布局），按十进制渲染成字符串。
    pub consts: Vec<(String, String)>,
    /// 降级链，按 `order` 升序。
    pub variants: Vec<Variant>,
    /// 模板文本。
    pub template: String,
    /// 内核签名：从模板的入口函数解析，是 ABI 的唯一事实源。
    pub signature: crate::sig::Signature,
}

/// 轴展开出的一组具体值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileCombo {
    values: Vec<(String, u32)>,
}

impl TileCombo {
    /// 取一个轴的值。
    pub fn get(&self, name: &str) -> Option<u32> {
        self.values
            .iter()
            .find(|(axis, _)| axis == name)
            .map(|(_, value)| *value)
    }

    /// 全部取值。
    pub fn values(&self) -> &[(String, u32)] {
        &self.values
    }

    /// 渲染成 `GMAX=32` 形式。
    pub fn summary(&self) -> String {
        self.values
            .iter()
            .map(|(axis, value)| format!("{axis}={value}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl fmt::Display for TileCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// 字段级错误：`invalid` 类别，消息里带字段路径（声明层的一切错误都是"输入不合法"）。
pub(crate) fn field(path: &str, detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("`{path}` 非法：{}", detail.into()))
}

mod build;
mod expand;
mod parse;
mod variant;

pub use expand::{expand, tiles};

#[cfg(test)]
#[path = "../tests/decl_tests.rs"]
mod tests;
