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
//! `Cost` 与 [`Extents`] 从 `einsum` 再导出，是为了让 `cutile` / `nn` 只依赖 `prof` 就能拿到
//! 它们（所有权仍在 `einsum`）。
//!
//! # 形状是具名轴，不是一对数字
//!
//! 算子的 `cost` 收 [`ShapeCtx`]（轴字母 → 长度），而不是 `(m, n)`：后者把"一个规约轴 +
//! 一个输出轴"写死进接口，RMSNorm 的 `[batch, seq, d_model]` 与 Attention 的 `seq²` 只能靠伪造
//! 一对 `(m, n)` 来表达，而伪造就意味着第二份公式。**派生轴**（量化分组 `g = m / GROUP`）由算子
//! 自己在 `cost` 里声明：分组属于算子语义，把它推给调用方等于让每个层都抄一遍 32。

use diagnostic::{Code, Error};

/// 输出类型从 `prof` 再导出：`#[annotation::model]` 生成的 `Model` 与审计侧用同一个 `DType`，
/// 不因为"谁先定义"而出现两条路径。
pub use crate::atom::DType;

#[doc(inline)]
pub use crate::einsum::Cost;
#[doc(inline)]
pub use crate::einsum::Extents;

/// 类型级定宽向量：每组恰有 `N` 个标量，表示层不存在未对齐的尾部。
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T, const N: usize> {
    values: Vec<[T; N]>,
}

impl<T, const N: usize> Vector<T, N> {
    /// 从扁平标量构造；`N = 0` 或标量数不整除时拒绝。
    pub fn new(values: Vec<T>) -> diagnostic::Result<Self>
    where
        T: Copy,
    {
        if N == 0 {
            return Err(Error::new(Code::Invalid, "向量车道数必须大于 0"));
        }
        if !values.len().is_multiple_of(N) {
            return Err(Error::new(
                Code::Invalid,
                format!("标量数 {} 不能整除车道数 {N}", values.len()),
            ));
        }
        let groups = (0..values.len() / N)
            .map(|group| std::array::from_fn(|lane| values[group * N + lane]))
            .collect();
        Ok(Self { values: groups })
    }

    /// 定宽分组切片，供 C 初始化器保留每车道边界。
    pub fn as_arrays(&self) -> &[[T; N]] {
        &self.values
    }

    /// 扁平标量视图，供后端打包与主机对拍零拷贝共用。
    pub fn as_slice(&self) -> &[T] {
        self.values.as_flattened()
    }

    /// 标量元素数（不是向量组数）。
    pub fn len(&self) -> usize {
        self.values.len() * N
    }

    /// 向量组数。
    pub fn groups(&self) -> usize {
        self.values.len()
    }

    /// 向量是否不含任何标量元素。
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 编译期车道数。
    pub const fn lanes(&self) -> usize {
        N
    }
}

impl<T> Vector<T, 4> {
    /// 按向量组数生成四车道取值，无需运行期整除校验。
    pub fn from_fn(groups: usize, mut generate: impl FnMut(usize) -> T) -> Self {
        let values = (0..groups)
            .map(|group| std::array::from_fn(|lane| generate(group * 4 + lane)))
            .collect();
        Self { values }
    }
}

/// 缓冲取值：标量档与显式四车道向量档共用一条对拍通道。
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    /// 单精度浮点。
    F32(Vec<f32>),
    /// 打包字节（如 int4 权重）。
    U8(Vec<u8>),
    /// 半精度浮点（IEEE binary16）。
    F16(Vec<half::f16>),
    /// brain 浮点（bfloat16）。
    Bf16(Vec<half::bf16>),
    /// 四车道 f32。
    F32x4(Vector<f32, 4>),
    /// 四车道 f16。
    F16x4(Vector<half::f16, 4>),
    /// 四车道 bf16。
    Bf16x4(Vector<half::bf16, 4>),
    /// 四车道 u8。
    U8x4(Vector<u8, 4>),
}

/// 变体元数据表：长度、车道数、诊断名与元素档只在一处登记，新增变体缺表项即编译失败。
macro_rules! values_by_variant {
    ($($variant:ident => ($describe:literal, $dtype:expr, $lanes:expr)),+ $(,)?) => {
        impl Values {
            /// 标量元素个数；向量档返回底层标量数。
            pub fn len(&self) -> usize {
                match self { $(Self::$variant(values) => values.len()),+ }
            }

            /// 每组的标量车道数；普通标量档为 1。
            pub fn lanes(&self) -> usize {
                match self { $(Self::$variant(_) => $lanes),+ }
            }

            /// 元素类型的唯一诊断名。
            pub fn describe(&self) -> &'static str {
                match self { $(Self::$variant(_) => $describe),+ }
            }

            /// 元素档，不包含车道数；车道由具体 `Vector<T, N>` 类型与枚举身份共同表达。
            pub fn dtype(&self) -> DType {
                match self { $(Self::$variant(_) => $dtype),+ }
            }
        }
    };
}

values_by_variant! {
    F32 => ("float", DType::F32, 1),
    U8 => ("unsigned char", DType::U8, 1),
    F16 => ("half", DType::F16, 1),
    Bf16 => ("bf16", DType::BF16, 1),
    F32x4 => ("float4", DType::F32, 4),
    F16x4 => ("half4", DType::F16, 4),
    Bf16x4 => ("bf16x4", DType::BF16, 4),
    U8x4 => ("uchar4", DType::U8, 4),
}

impl Values {
    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 缓冲的实际存储字节数；向量档按底层标量数计，不把车宽度重复乘入。
    pub fn byte_len(&self) -> u64 {
        self.dtype().bytes_for(self.len() as u64)
    }

    /// C 数组的存储项数；向量档按组计，对拍长度仍由 [`Self::len`] 给出。
    pub fn storage_len(&self) -> usize {
        self.len() / self.lanes()
    }

    /// 类型与车道数都相同才算同档；判别看完整枚举身份。
    pub fn same_kind(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// 取 f32 标量切片；四车道 f32 直接零拷贝返回，其余档显式拒绝。
    pub fn as_f32(&self) -> diagnostic::Result<&[f32]> {
        match self {
            Self::F32(values) => Ok(values),
            Self::F32x4(values) => Ok(values.as_slice()),
            other => Err(Error::new(
                Code::Invalid,
                format!("对拍通道仅支持 f32，当前档 `{}`", other.describe()),
            )),
        }
    }
}

/// 一份具名缓冲；名字就是内核模板里的形参名。
#[derive(Debug, Clone, PartialEq)]
pub struct Buf {
    /// C 标识符（模板的形参名）。
    pub name: &'static str,
    /// 取值。
    pub values: Values,
}

impl Buf {
    /// 构造一份具名 f32 缓冲，样本装配不重复展开 `Values` 变体。
    pub fn f32(name: &'static str, values: Vec<f32>) -> Self {
        Self {
            name,
            values: Values::F32(values),
        }
    }
}

/// 一次运行用的样本：形状 + 具名缓冲 + 具名标量。
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// 规约维度。
    pub m: usize,
    /// 输出维度。
    pub n: usize,
    /// 非缓冲实参的取值：内核签名里的整数形参（如 `M` / `N` / `gsz`）。
    pub scalars: Vec<(&'static str, i64)>,
    /// 具名缓冲（输出缓冲也在其中）。
    pub bufs: Vec<Buf>,
}

/// 生成一个类型化缓冲读取器：缺失与错档诊断保持同一措辞。
macro_rules! sample_getter {
    ($method:ident, $variant:ident, $ty:ty, $label:literal) => {
        #[doc = concat!("取 `", $label, "` 缓冲。\n\n# Errors\n\n缓冲不存在或不是 `", $label, "` 时返回带名字的说明。")]
        pub fn $method(&self, name: &str) -> diagnostic::Result<&[$ty]> {
            match self.buf(name).map(|buf| &buf.values) {
                Some(Values::$variant(values)) => Ok(values),
                Some(_) => Err(Error::new(Code::Invalid, format!("缓冲 `{name}` 不是 {}", $label))),
                None => Err(Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`"))),
            }
        }
    };
}

impl Sample {
    /// 构造 f32 输入/输出样本：`x` 使用确定性夹具，`y` 初始化为零。
    pub fn f32_io(
        m: usize,
        n: usize,
        input_len: usize,
        output_len: usize,
        scalars: Vec<(&'static str, i64)>,
    ) -> Self {
        Self {
            m,
            n,
            scalars,
            bufs: vec![
                x_buf(input_len),
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; output_len]),
                },
            ],
        }
    }

    /// 按名字取缓冲。
    pub fn buf(&self, name: &str) -> Option<&Buf> {
        self.bufs.iter().find(|buf| buf.name == name)
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
        match self.buf(name).map(|buf| &buf.values) {
            Some(values) => values.as_f32().map_err(|_| {
                Error::new(Code::Invalid, format!("缓冲 `{name}` 不是 f32 标量或向量"))
            }),
            None => Err(Error::new(
                Code::Missing,
                format!("样本里没有缓冲 `{name}`"),
            )),
        }
    }

    sample_getter!(f16, F16, half::f16, "f16");
    sample_getter!(bf16, Bf16, half::bf16, "bf16");
    sample_getter!(u8, U8, u8, "u8");

    /// 把一个缓冲换成调用方的取值。
    ///
    /// 名字是算子契约的一部分：模板的形参名、[`Model::reference`] 读的缓冲名、`@prim(...)` 的
    /// 实参名是同一个字符串。因此"换数据"只能按名字换，不能按顺序换——顺序属于签名，不属于样本。
    ///
    /// # Errors
    ///
    /// 缓冲不存在（[`Code::Missing`]）、元素类型与声明不符（[`Code::Invalid`]）或长度与算子
    /// 声明的形状不符（[`Code::Invalid`]）时返回说明。
    pub fn set(&mut self, name: &str, values: Values) -> diagnostic::Result<()> {
        let buf = self
            .bufs
            .iter_mut()
            .find(|buf| buf.name == name)
            .ok_or_else(|| {
                Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`"))
                    .with("prof::model::Sample::set")
            })?;
        if !buf.values.same_kind(&values) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲 `{name}` 是 `{}`，换入的是 `{}`",
                    buf.values.describe(),
                    values.describe()
                ),
            )
            .with("prof::model::Sample::set"));
        }
        let (want, got) = (buf.values.len(), values.len());
        if want != got {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲 `{name}` 声明了 {want} 个元素，换入 {got} 个"),
            )
            .with("prof::model::Sample::set"));
        }
        buf.values = values;
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
        .with("prof::model::bind"));
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
        .with("prof::model::bind"));
    }
    if w.len() != n * m {
        return Err(Error::new(
            Code::Invalid,
            format!("权重元素数 {} 与 {n}×{m} 不符", w.len()),
        )
        .with("prof::model::bind"));
    }
    Ok(())
}

/// 只有稠密（f32）权重能注入真实数据：打包字节里的 `w` 读出来是半份字节。
fn dense(sample: &Sample) -> diagnostic::Result<()> {
    match sample.buf("w").map(|buf| &buf.values) {
        Some(Values::F32(_)) => Ok(()),
        Some(_) => Err(Error::new(
            Code::Unsupported,
            "该算子的权重缓冲不是稠密标量 f32：先按声明转换成真实 f32 权重，不向量化档静默注入",
        )
        .with("prof::model::bind")),
        None => {
            Err(Error::new(Code::Missing, "算子样本里没有权重缓冲 `w`").with("prof::model::bind"))
        }
    }
}

/// 用真实数据替掉样本里的 `x` / `w`。
fn fill(sample: &mut Sample, x: &[f32], w: &[f32]) -> diagnostic::Result<()> {
    sample.set("x", Values::F32(x.to_vec()))?;
    sample.set("w", Values::F32(w.to_vec()))
}

/// 一条基本轴的语义角色（闭集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 投影的规约轴：层用输入宽度填它。
    Input,
    /// 投影的输出轴：层用输出宽度填它。
    Output,
}

/// 算子的一条基本轴：角色 + 表达式里的字母。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisSpec {
    /// 语义角色。
    pub role: AxisRole,
    /// 表达式里的字母。
    pub label: char,
}

/// 一次调用的形状上下文：算子声明的**基本轴**及长度。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShapeCtx {
    extents: Extents,
}

impl ShapeCtx {
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
            .map_err(|error| error.with("prof::model::ShapeCtx"))
    }
}

/// 一个算子的语义表。
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// 形状粒度：`m` 必须是它的整数倍（量化分组）。
    pub group: u32,
    /// 生产形状 `(m, n)`：`doctor` 用它做 roofline 估值，`verify` 把它当形状矩阵里的生产形状，
    /// `bench` 在没有 `--shape` 时用它。必须是本算子的合法输入（`m` 是 [`Model::group`] 的整数倍）。
    pub shape: (usize, usize),
    /// 输出（也是 tile 中转）的元素类型：`verify` 的对拍阈值与缓冲区读回宽度由它推出，不写死 f32。
    pub dtype: DType,
    /// 基本轴：层按角色取值，不必知道表达式里的字母。
    pub axes: &'static [AxisSpec],
    /// 把形状对 `(m, n)` 翻译成算子自己的具名轴上下文。
    pub shape_ctx: fn(usize, usize) -> Result<ShapeCtx, Error>,
    /// 给定形状上下文的标量代价。
    pub cost: fn(&ShapeCtx) -> Result<Cost, Error>,
    /// 给定 `(m, n)` 的确定性样本。
    pub sample: fn(usize, usize) -> Sample,
    /// 标量参考：不依赖任何设备，也不依赖片段；结果按元素档装进 [`Values`]，对拍侧按档解包。
    pub reference: fn(&Sample) -> diagnostic::Result<Values>,
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
    /// [`ShapeCtx::new`]）。
    pub fn projection(&self, in_features: usize, out_features: usize) -> Result<ShapeCtx, Error> {
        let input = self.axis(AxisRole::Input).ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                "算子没有声明输入轴（`AxisRole::Input`），不能按投影计价",
            )
            .with("prof::model::Model::projection")
        })?;
        let output = self.axis(AxisRole::Output).ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                "算子没有声明输出轴（`AxisRole::Output`），不能按投影计价",
            )
            .with("prof::model::Model::projection")
        })?;
        ShapeCtx::new([(input, in_features), (output, out_features)])
    }
}

/// 单轴 `n` 的 f32 标量代价：`bytes = n·4`，`flops = n·flops_per`，`read = bytes·reads`，`write = bytes`。
/// rmsnorm / rope / softmax / dequant 这类「一条轴、f32 元素」的算子共用这一份口径，不各写一份。
pub fn cost_n(shape: &ShapeCtx, op: &str, flops_per: f64, reads: u64) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `n`")))?;
    let bytes = (n as u64) * 4;
    Ok(Cost {
        flops: (n as f64) * flops_per,
        read: bytes * reads,
        write: bytes,
    })
}

/// 确定性 `x` 样本缓冲：`x[i] = 0.25i − 2`，有正有负、可复现，供 softmax / rmsnorm / rope 共用。
pub fn x_buf(len: usize) -> Buf {
    Buf {
        name: "x",
        values: Values::F32((0..len).map(|i| ((i as f32) * 0.25) - 2.0).collect()),
    }
}

/// 线性确定性夹具：`base + step·i`，同一长度取值可复现。
#[must_use]
pub fn linear(len: usize, step: f32, base: f32) -> Vec<f32> {
    (0..len).map(|i| base + i as f32 * step).collect()
}

pub mod quant {
    //! 量化格式的编译期配置表：位宽、打包密度、对称零点、对称码表与样本偏移单点收敛。
    use crate::einsum::Bits;

    /// 一档对称量化格式的静态配置。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Row {
        /// 元素位宽（闭集 [`Bits`]）。
        pub bits: Bits,
        /// 每字节多少个码：`8 / 位宽`；逐元素为 1。
        pub cpb: usize,
        /// 对称零点（无符号码域 `0..2^bits` 减去它得到有符号整数码）。
        pub zero: i32,
        /// 对称解包表：码点下标 → 有符号整数值；打包格式长度 `2^bits`。
        pub table: &'static [f32],
        /// 样本码值偏移：打包取样时按 `(byte · seed + offsets[code]) mod 2^width` 填位。
        pub offsets: &'static [u32],
    }

    /// 2 位打包：4 码/字节，零点 2，表 `{-2,-1,0,1}`。
    pub const I2: Row = Row {
        bits: Bits::Two,
        cpb: 4,
        zero: 2,
        table: &[-2.0, -1.0, 0.0, 1.0],
        offsets: &[1, 2, 3, 4],
    };

    /// 4 位打包：2 码/字节，零点 8，表 `{-8..=7}`。
    pub const I4: Row = Row {
        bits: Bits::Four,
        cpb: 2,
        zero: 8,
        table: &[
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ],
        offsets: &[1, 9],
    };

    /// 8 位逐元素：1 码/字节，零点 0；不打包，`table` 留空由骨架按 `i8` 解释。
    pub const I8: Row = Row {
        bits: Bits::Eight,
        cpb: 1,
        zero: 0,
        table: &[],
        offsets: &[],
    };

    impl Row {
        /// 码域大小：`2^bits`；零点必须落在 `0..domain` 内。
        pub const fn domain(self) -> u32 {
            1 << self.bits.bits()
        }

        /// 打包宽度（位）：`8 / cpb`；逐元素格式为 8。
        pub const fn width(self) -> usize {
            8 / self.cpb
        }

        /// 是否打包（每字节多于一个码）。
        pub const fn packed(self) -> bool {
            self.cpb > 1
        }
    }

    /// 由每字节码数 `cpb` 推位宽：`8 / cpb`。
    pub const fn width_cpb(cpb: usize) -> usize {
        8 / cpb
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/spec/quant_tests.rs");
    }
}
