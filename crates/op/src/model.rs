//! 算子语义：代价模型、样本数据与标量参考。
//!
//! 声明（`ops/<name>.toml`）说"需要什么能力"，本模块说"算什么、拿什么对拍"。两者都是算子
//! 内容，住在同一个 crate；`cutile` 只消费它们，不认识任何一个算子名，因此新增算子不改
//! `cutile` 的任何一行（cutile README §13.2）。
//!
//! 用函数指针而不是 trait：注册表可以是一张 `const` 表，新增算子只加一行，不需要 `dyn` 或
//! 工厂的样板代码。
//!
//! 代价本身不在这里计算：`flops` / `read` / `write` 由 [`crate::einsum`] 从"一条求和表达式 + 每个
//! 操作数的位宽与方向"推出，本模块只负责把算子的语义（表达式、样本、标量参考）挂到注册表上。
//! `Cost` 与 [`Extents`] 从 `einsum` 再导出，是为了让 `cutile` / `nn` 继续只依赖 `op` 就能拿到
//! 它们（所有权仍在 `einsum`）。
//!
//! # 形状是具名轴，不是一对数字
//!
//! 算子的 `cost` 收 [`ShapeContext`]（轴字母 → 长度），而不是 `(m, n)`：后者把"一个规约轴 +
//! 一个输出轴"写死进接口，RMSNorm 的 `[batch, seq, d_model]` 与 Attention 的 `seq²` 只能靠伪造
//! 一对 `(m, n)` 来表达，而伪造就意味着第二份公式。**派生轴**（量化分组 `g = m / GROUP`）由算子
//! 自己在 `cost` 里声明：分组属于算子语义，把它推给调用方等于让每个层都抄一遍 32。

use diagnostic::{Code, Error};

/// 输出类型从 `prof` 再导出：`#[annotation::model]` 生成的 `Model` 与审计侧用同一个 `DType`，
/// 不因为"谁先定义"而出现两条路径。
pub use prof::atom::DType;

#[doc(inline)]
pub use crate::einsum::Cost;
#[doc(inline)]
pub use crate::einsum::Extents;

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
/// 样本必须确定性——同一形状永远得到同一组输入（cutile README §12.3 的"可复现"）。
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

    /// 把一个缓冲换成调用方的取值。
    ///
    /// 名字是算子契约的一部分：模板的形参名、[`Model::reference`] 读的缓冲名、`@prim(...)` 的
    /// 实参名是同一个字符串。因此"换数据"只能按名字换，不能按顺序换——顺序属于签名，不属于样本。
    ///
    /// # Errors
    ///
    /// 缓冲不存在（[`Code::Missing`]）或长度与算子声明的形状不符（[`Code::Invalid`]）时返回说明。
    pub fn set(&mut self, name: &str, values: Values) -> diagnostic::Result<()> {
        let buffer = self
            .buffers
            .iter_mut()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| {
                Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`"))
                    .with("op::model::Sample::set")
            })?;
        let (want, got) = (buffer.values.len(), values.len());
        if want != got {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲 `{name}` 声明了 {want} 个元素，换入 {got} 个"),
            )
            .with("op::model::Sample::set"));
        }
        buffer.values = values;
        Ok(())
    }
}

/// 用真实数据构造一份可执行样本：形状由层给，`x` / `w` 换成权重文件里的值。
///
/// 这是"真实权重"进入内核的唯一入口，因此两条约束写在这里而不是散在调用方：
///
/// - 形状必须与算子的形状粒度自洽（`m` 是 [`Model::group`] 的整数倍），否则样本与内核算的
///   根本不是同一件事；
/// - 只有**稠密**（权重缓冲是 f32）的算子能这样注入——量化算子的 `w` 是打包字节，塞 f32 进去
///   会得到一份与 `.toml` 声明不符的样本（字节数对、含义不对），因此以 [`Code::Unsupported`]
///   拒绝，而不是让内核去解读半份字节。
///
/// `x` / `w` 之外的一切（标量 `M` / `N` / `gsz`、输出缓冲、其余输入）都来自
/// [`Model::sample`]：真实数据不该顺手改掉形状或 ABI。
///
/// # Errors
///
/// `m` 不是分组的整数倍、`x` 长度不是 `m`、`w` 长度不是 `n · m`，或该算子的权重缓冲不是 f32
/// 时返回说明。
pub fn bind(model: &Model, m: usize, n: usize, x: &[f32], w: &[f32]) -> diagnostic::Result<Sample> {
    shape(model, m, n)?;
    lengths(m, n, x, w)?;
    let mut sample = (model.sample)(m, n);
    dense(&sample)?;
    fill(&mut sample, x, w)?;
    Ok(sample)
}

/// 形状判据：`m` 必须是形状粒度 `group` 的正整数倍。
fn shape(model: &Model, m: usize, n: usize) -> diagnostic::Result<()> {
    let group = model.group.max(1) as usize;
    if m == 0 || n == 0 || !m.is_multiple_of(group) {
        return Err(Error::new(
            Code::Invalid,
            format!("形状 {m}x{n} 不自洽：m 必须是形状粒度 {group} 的正整数倍"),
        )
        .with("op::model::bind"));
    }
    Ok(())
}

/// 长度判据：`x` 覆盖输入宽度，`w` 覆盖整张权重；两条各报各的。
fn lengths(m: usize, n: usize, x: &[f32], w: &[f32]) -> diagnostic::Result<()> {
    if x.len() != m {
        return Err(Error::new(
            Code::Invalid,
            format!("输入向量长度 {} 与输入宽度 {m} 不符", x.len()),
        )
        .with("op::model::bind"));
    }
    if w.len() != n * m {
        return Err(Error::new(
            Code::Invalid,
            format!("权重元素数 {} 与 {n}×{m} 不符", w.len()),
        )
        .with("op::model::bind"));
    }
    Ok(())
}

/// 只有稠密（f32）权重能注入真实数据：打包字节里的 `w` 读出来是半份字节。
fn dense(sample: &Sample) -> diagnostic::Result<()> {
    match sample.buffer("w").map(|buffer| &buffer.values) {
        Some(Values::F32(_)) => Ok(()),
        Some(Values::U8(_)) => Err(Error::new(
            Code::Unsupported,
            "该算子的权重缓冲是打包字节（量化），真实 f32 权重不能注入：先按算子声明的分组量化",
        )
        .with("op::model::bind")),
        None => {
            Err(Error::new(Code::Missing, "算子样本里没有权重缓冲 `w`").with("op::model::bind"))
        }
    }
}

/// 用真实数据替掉样本里的 `x` / `w`。
fn fill(sample: &mut Sample, x: &[f32], w: &[f32]) -> diagnostic::Result<()> {
    sample.set("x", Values::F32(x.to_vec()))?;
    sample.set("w", Values::F32(w.to_vec()))
}

/// 一条基本轴的语义角色（闭集）。
///
/// 闭集而不是字符串：层按角色向算子要轴（"输入宽度填哪条轴"），写错角色应当是类型错误。
/// 一个新层类型需要新角色时在这里加一个变体，并同时定义"层该怎么填它"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 投影的规约轴：层用输入宽度填它。
    Input,
    /// 投影的输出轴：层用输出宽度填它。
    Output,
}

/// 算子的一条基本轴：角色 + 表达式里的字母。
///
/// 字母属于算子（同一条表达式 `nm,m,ng,n->n`），角色属于层（"输入宽度"）。两者分开之后，
/// `nn::Linear` 不必知道自己的规约轴叫 `m` 还是 `k`，换一份表达式也不会把层一起改掉。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisSpec {
    /// 语义角色。
    pub role: AxisRole,
    /// 表达式里的字母。
    pub label: char,
}

/// 一次调用的形状上下文：算子声明的**基本轴**及长度。
///
/// 调用方只填自己知道的轴（投影的输入/输出宽度、注意力的序列长度……）。派生轴不在这里：
/// 量化分组 `g = m / GROUP` 由算子在 `cost` 里用 [`ShapeContext::extents_with_group`] 声明——
/// 分组若由调用方填，每个层都要抄一遍 32，而且改分组时所有层都得跟着改。
///
/// 轴表的合法性（非零、不重复、字母在 `a..z` / `A..Z` 内）由 [`Extents`] 保证，本类型不重写
/// 一遍判断。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShapeContext {
    extents: Extents,
}

impl ShapeContext {
    /// 由 `(轴字母, 长度)` 构造。
    ///
    /// # Errors
    ///
    /// 字母非法、长度为 0、同一个字母给了两次（见 [`Extents::new`]）。
    pub fn new(pairs: impl IntoIterator<Item = (char, usize)>) -> Result<Self, Error> {
        Ok(Self {
            extents: Extents::new(pairs)?,
        })
    }

    /// 取一条轴的长度。
    pub fn get(&self, label: char) -> Option<usize> {
        self.extents.get(label)
    }

    /// 基底轴表（不含派生轴）。
    pub fn extents(&self) -> &Extents {
        &self.extents
    }

    /// 克隆一份并声明派生轴 `label = base / divisor`，返回可直接交给 `Formula::cost` 的轴表。
    ///
    /// 这是"派生轴属于算子"的落点：调用方只给 `m`，`g` 由算子按自己的 `GROUP` 切出来。
    ///
    /// # Errors
    ///
    /// `base` 不在上下文里、`divisor` 为 0，或 `base` 的长度不是它的整数倍。
    pub fn extents_with_group(
        &self,
        label: char,
        base: char,
        divisor: usize,
    ) -> Result<Extents, Error> {
        self.extents
            .clone()
            .group(label, base, divisor)
            .map_err(|error| error.with("op::model::ShapeContext"))
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
    /// 基本轴：层按角色取值，不必知道表达式里的字母。
    ///
    /// 投影算子声明 [`AxisRole::Input`] 与 [`AxisRole::Output`]；将来的 RMSNorm / Attention
    /// 可以只声明自己需要的角色（或先留空，由调用方用 `nn::Bounds` 那侧的形状上下文自行构造）。
    pub axes: &'static [AxisSpec],
    /// 把形状对 `(m, n)` 翻译成算子自己的具名轴上下文。
    ///
    /// 投影算子指向按 `m/n` 构造的函数，逐元素/归约/规范化等非投影算子各自按骨架的轴构造，
    /// 因此 `verify` 不需要假设每条算子都声明了投影的 Input/Output 轴。
    pub shape_ctx: fn(usize, usize) -> Result<ShapeContext, Error>,
    /// 给定形状上下文的标量代价。
    ///
    /// 可失败：表达式与形状不自洽（例如列数不是量化分组的整数倍）时返回带类别的错误，而不是
    /// 在代价里悄悄截断一个因子。
    pub cost: fn(&ShapeContext) -> Result<Cost, Error>,
    /// 给定 `(m, n)` 的确定性样本。
    pub sample: fn(usize, usize) -> Sample,
    /// 标量参考：不依赖任何设备，也不依赖片段。
    pub reference: fn(&Sample) -> diagnostic::Result<Vec<f32>>,
}

impl Model {
    /// 按角色取轴字母。
    pub fn axis(&self, role: AxisRole) -> Option<char> {
        self.axes
            .iter()
            .find(|axis| axis.role == role)
            .map(|axis| axis.label)
    }

    /// 把投影的 `(输入宽度, 输出宽度)` 放进本算子声明的具名轴。
    ///
    /// 这是"投影"这一**层**概念与"具名轴"这一**算子**概念之间唯一的翻译点。算子没有声明
    /// 投影轴时返回 [`Code::Unsupported`]，而不是退回一对 `(m, n)`：那会把接口重新钉回它
    /// 试图摆脱的形状，RMSNorm / Attention 也就永远只能伪造一对数字。
    ///
    /// # Errors
    ///
    /// 缺少 [`AxisRole::Input`] 或 [`AxisRole::Output`] 的声明；宽度为 0（见
    /// [`ShapeContext::new`]）。
    pub fn projection(
        &self,
        in_features: usize,
        out_features: usize,
    ) -> Result<ShapeContext, Error> {
        let input = self.axis(AxisRole::Input).ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                "算子没有声明输入轴（`AxisRole::Input`），不能按投影计价",
            )
            .with("op::model::Model::projection")
        })?;
        let output = self.axis(AxisRole::Output).ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                "算子没有声明输出轴（`AxisRole::Output`），不能按投影计价",
            )
            .with("op::model::Model::projection")
        })?;
        ShapeContext::new([(input, in_features), (output, out_features)])
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
