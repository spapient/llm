//! 算子语义：代价模型、样本数据与标量参考。
//!
//! 声明（`ops/<name>.toml`）说"需要什么能力"，本模块说"算什么、拿什么对拍"。两者都是算子
//! 内容，住在同一个 crate；`cutile` 只消费它们，不认识任何一个算子名，因此新增算子不改
//! `cutile` 的任何一行（DESIGN §13.2）。
//!
//! 用函数指针而不是 trait：注册表可以是一张 `const` 表，新增算子只加一行，不需要 `dyn` 或
//! 工厂的样板代码。
//!
//! 代价本身不在这里计算：`flops` / `bytes` 由 [`einsum`] 从"一条求和表达式 + 每个操作数的
//! 位宽"推出，本模块只负责把算子的语义（表达式、样本、标量参考）挂到注册表上。`Cost` 从
//! `einsum` 再导出，是为了让 `cutile` 继续只依赖 `op` 就能拿到它（所有权仍在 `einsum`）。

use diagnostic::{Code, Error};
use prof::atom::DType;

#[doc(inline)]
pub use einsum::Cost;

/// 缓冲的取值：P0 只出现 f32 与打包字节。
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    /// 单精度浮点。
    F32(Vec<f32>),
    /// 打包字节（如 int4 权重）。
    U8(Vec<u8>),
}

impl Values {
    /// 元素个数。
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U8(values) => values.len(),
        }
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 一份具名缓冲；名字就是内核模板里的形参名。
#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    /// C 标识符（模板的形参名）。
    pub name: &'static str,
    /// 取值。
    pub values: Values,
}

/// 一次运行用的样本：形状 + 具名缓冲 + 具名标量。
///
/// 名字与类型由 `.k` 的签名界定（[`crate::sig::Signature::check`]）：缓冲顺序**不**参与 ABI，
/// 实参顺序由签名给出，样本只负责提供同名取值。
///
/// 样本必须确定性——同一形状永远得到同一组输入（DESIGN §12.3 的"可复现"）。
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// 规约维度。
    pub m: usize,
    /// 输出维度。
    pub n: usize,
    /// 非缓冲实参的取值：内核签名里的整数形参（如 `M` / `N` / `gsz`）。
    pub scalars: Vec<(&'static str, i64)>,
    /// 具名缓冲（输出缓冲也在其中）。
    pub buffers: Vec<Buffer>,
}

impl Sample {
    /// 按名字取缓冲。
    pub fn buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.iter().find(|buffer| buffer.name == name)
    }

    /// 按名字取标量实参。
    pub fn scalar(&self, name: &str) -> Option<i64> {
        self.scalars
            .iter()
            .find(|(scalar, _)| *scalar == name)
            .map(|(_, value)| *value)
    }

    /// 取 f32 缓冲。
    ///
    /// # Errors
    ///
    /// 缓冲不存在或不是 f32 时返回带名字的说明。
    pub fn f32(&self, name: &str) -> diagnostic::Result<&[f32]> {
        match self.buffer(name).map(|buffer| &buffer.values) {
            Some(Values::F32(values)) => Ok(values),
            Some(Values::U8(_)) => {
                Err(Error::new(Code::Invalid, format!("缓冲 `{name}` 不是 f32")))
            }
            None => Err(Error::new(
                Code::Missing,
                format!("样本里没有缓冲 `{name}`"),
            )),
        }
    }

    /// 取 u8 缓冲。
    ///
    /// # Errors
    ///
    /// 缓冲不存在或不是 u8 时返回带名字的说明。
    pub fn u8(&self, name: &str) -> diagnostic::Result<&[u8]> {
        match self.buffer(name).map(|buffer| &buffer.values) {
            Some(Values::U8(values)) => Ok(values),
            Some(Values::F32(_)) => {
                Err(Error::new(Code::Invalid, format!("缓冲 `{name}` 不是 u8")))
            }
            None => Err(Error::new(
                Code::Missing,
                format!("样本里没有缓冲 `{name}`"),
            )),
        }
    }
}

/// 一个算子的语义表。
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// 形状粒度：`m` 必须是它的整数倍（量化分组）。
    pub group: u32,
    /// 生产形状 `(m, n)`：`doctor` 用它做 roofline 估值，`verify` 把它当形状矩阵里的生产形状，
    /// `bench` 在没有 `--shape` 时用它。必须是本算子的合法输入（`m` 是 [`Model::group`] 的整数倍）。
    ///
    /// 它属于算子数据而不是命令策略：换一个算子，生产形状跟着它走，审计不写第二个方阵。
    pub shape: (usize, usize),
    /// 输出（也是 tile 中转）的元素类型：`verify` 的对拍阈值与缓冲区读回宽度由它推出，不写死 f32。
    pub dtype: DType,
    /// 给定 `(m, n)` 的标量代价。
    ///
    /// 可失败：表达式与形状不自洽（例如列数不是量化分组的整数倍）时返回带类别的错误，而不是
    /// 在代价里悄悄截断一个因子。
    pub cost: fn(usize, usize) -> Result<Cost, Error>,
    /// 给定 `(m, n)` 的确定性样本。
    pub sample: fn(usize, usize) -> Sample,
    /// 标量参考：不依赖任何设备，也不依赖片段。
    pub reference: fn(&Sample) -> diagnostic::Result<Vec<f32>>,
}
