//! 层要向装配侧声明的两件事：**形状**与**下界**。
//!
//! 前半是单轴族与归约族共享的规格内核与三张委托宏（形状口径、访问器、规格透传）；后半是由算子
//! 语义表推出的流量与运算下界。两者同居一个文件，用的是同一条判据：**层只声明，不写第二份常数**
//! ——形状由构造参数给出，代价由算子名去 `op` 里查，`nn` 不认识除算子名以外的领域知识。
//!
//! 层的性能下界：任何实现都不可能低于的流量与运算次数。
//!
//! 这些数**不是手写的**：它们由层声明的算子在 `op` 里的语义（一条求和表达式 + 每个操作数的
//! 位宽与方向）推出。因此新增算子不会多出第二条公式，`nn` 也不必认识除算子名以外的任何领域知识。
//!
//! ### 为什么要有它
//!
//! `cutile` 的 roofline 在 **kernel** 层回答"这条片段至少要搬多少、算多少"；本模块把同一件事
//! 抬到**层**：下界是声明，`bench` 的实测是事实，两者之差正是融合、量化与布局真正赚到的部分。
//! 没有层的下界，"这一层优化了多少"就只能靠人肉估算。
//!
//! ### 口径：按方向分账
//!
//! 流量分成 [`Bounds::read`] 与 [`Bounds::write`] 两个方向，而不是一个 `bytes`：设备的读、写
//! 上限不是同一个数（只流式读权重的 GEMV 可以跑过拷贝上限），合成一个总量之后"这层卡在哪一侧"
//! 就再也判断不了。[`Bounds::bytes`] 仍然给出合计，但它是**派生**量。
//!
//! 这个区分不是为了好看：[`Bounds::moves_within`] 靠它识别"总量不变、只是把读挪成写"——那不是
//! 融合省下的流量，只是换了一个瓶颈。
//!
//! ### 与设备无关
//!
//! 本模块不认识峰值带宽、厂商或存储层，也不做 roofline 编排：需要与设备比较时由调用方拿峰值
//! 去比（`cutile::plan` 就是这么做的）。这里只给数对，而且每个方向各给一份。

use std::fmt;

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

use crate::ParamRole;

/// 层下界的结果：[`Contract::bounds`] 的返回类型。
pub type BoundsResult = Result<Option<Bounds>>;

/// 已声明的算子名清单，仅用于报错文案（热路径不调用）。
fn declared_ops() -> String {
    let names: Vec<&str> = prof::catalogue::all()
        .iter()
        .map(prof::catalogue::Op::name)
        .collect();
    format!("已声明：{}", names.join("、"))
}

/// 逐个校验「必须非零」的维度，失败时点名装配位（`site`）。
///
/// `prefix` 自带分隔符（如 `"注意力层 "` / `"Embedding 的 "`），因此各层原有文案逐字保留。
pub(crate) fn nonzero_all(site: &str, prefix: &str, axes: &[(&str, usize)]) -> Result<()> {
    for (label, value) in axes {
        if *value == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{prefix}{label} 必须非零（{label} = {value}）"),
            )
            .with(site));
        }
    }
    Ok(())
}

/// 校验单个「必须非零」的维度，失败时点名层与字段：经典几何层族共用。
///
/// 与 [`nonzero_all`] 的分工是调用形态不同——这里逐字段校验并**返回**该值供构造器继续用，
/// `nonzero_all` 一次吃完一张表、只报第一个空轴。
pub(crate) fn nonzero(value: usize, field: &str, layer: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::new(Code::Invalid, format!("{layer}: `{field}` 必须非零")).with("nn"));
    }
    Ok(value)
}

/// 层定义产出的声明：名字、输出形状、算子列表、它导出的声明，以及**自己的下界**。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContractSpec {
    name: String,
    output: Shape,
    ops: Vec<String>,
    role: ParamRole,
}

impl ContractSpec {
    /// 校验并构造：`ops` 必须既存在于 [`prof::catalogue::all`]，又有 [`registry::model`] 的语义表。
    ///
    /// `ops` 收 `impl Into<String>` 的任意可迭代物：字面量数组 `["gemv_i4"]` 与已拼好的
    /// `Vec<String>` 都能直接传，调用侧不必为一条静态名字写 `.to_string()`。
    pub fn new(
        name: impl Into<String>,
        output: Shape,
        ops: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self> {
        Self::new_with_role(name, output, ops, ParamRole::Trainable)
    }

    /// 校验并构造，并显式声明参数角色；投影之外的层（归一化系数、偏置、嵌入表）必须经此入口
    /// 填角色——优化器分组从角色派生，猜字段名或层名前缀会把衰减组与豁免组放错。
    pub fn new_with_role(
        name: impl Into<String>,
        output: Shape,
        ops: impl IntoIterator<Item = impl Into<String>>,
        role: ParamRole,
    ) -> Result<Self> {
        // 热路径不建整表：每层只按名字查一次（`builtin_named` 无分配），全表清单只在
        // 报错时才拼——装配一个几百层的模型不该为每条算子名付一次全表 `Vec` 的分配。
        let mut checked = Vec::new();
        for wanted in ops {
            let wanted = wanted.into();
            if prof::catalogue::named(&wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有声明（{}）", declared_ops()),
                )
                .with("nn::ContractSpec::new"));
            }
            if crate::registry::model(&wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有语义表，层无法推出下界也无法对拍"),
                )
                .with("nn::ContractSpec::new"));
            }
            checked.push(wanted);
        }
        Ok(Self {
            name: name.into(),
            output,
            ops: checked,
            role,
        })
    }

    /// 层名（与权重条目的前缀一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 参数角色：优化器分组与权重衰减豁免的派生依据。
    pub fn role(&self) -> ParamRole {
        self.role
    }

    /// 输出形状。
    pub fn output(&self) -> &Shape {
        &self.output
    }

    /// 要用的算子名。空切片表示这一层的编码还没有算子，因此无法计价。
    pub fn ops(&self) -> &[String] {
        &self.ops
    }
}

/// 层契约：名字、输出形状、算子列表、它导出的声明，以及**自己的下界**。
pub trait Contract {
    /// 层名。
    fn name(&self) -> &str;

    /// 输出形状。
    fn output(&self) -> &Shape;

    /// 导出给 `cutile` 的声明。
    fn spca(&self) -> &ContractSpec;

    /// 本层的性能下界。
    fn bounds(&self) -> BoundsResult {
        Ok(None)
    }

    /// 若这一层是按形状构造的线性投影，返回它的类型化视图。
    fn linear(&self) -> Option<&Linear> {
        None
    }

    /// 若这一层是查表层，返回它的类型化视图。
    fn embedding(&self) -> Option<&crate::Embedding> {
        None
    }

    /// 若这一层是注意力组合块，返回它的类型化视图。
    fn attention_block(&self) -> Option<&crate::attention::AttentionBlock> {
        None
    }

    /// 若这一层是条件记忆层，返回它的类型化视图。
    fn memory(&self) -> Option<&crate::engram::MemoryBlock> {
        None
    }

    /// 若这一层是循环层，返回它的类型化视图。
    ///
    /// 与 [`Contract::linear`] 同一个用途：从检查点装配一侧只拿到 `dyn Contract`，要读回
    /// 「几层、几个方向、哪种单元」就得靠这条下行通道，而不是再猜一次名字。
    fn recurrent(&self) -> Option<&crate::recurrent::Recurrent> {
        None
    }
}

/// 线性层：`y = x · W`。用哪个算子由权重编码 [`WeightFormat`] 决定，而不是一个全局常量。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Linear {
    spec: ContractSpec,
    in_features: usize,
    out_features: usize,
    format: WeightFormat,
}

impl Linear {
    /// 默认编码（[`WeightFormat::INT4`]）对应的算子名。
    pub const OP: &str = "gemv_i4";

    /// 稠密 f32 权重对应的算子名。
    pub const DENSE: &str = "gemv_f32";

    /// 对称 int8 权重对应的算子名。
    pub const I8: &str = "gemv_i8";

    /// 对称 int2 权重对应的算子名。
    pub const I2: &str = "gemv_i2";

    /// 由输入/输出宽度构造，按 int4 权重（[`WeightFormat::INT4`]）计价；两个宽度都必须非零。
    pub fn new(name: impl Into<String>, in_features: usize, out_features: usize) -> Result<Self> {
        Self::with_format(name, WeightFormat::INT4, in_features, out_features)
    }

    /// 按给定的权重编码构造。
    ///
    /// 编码没有对应算子时（例如 f16 / bf16 / i8），层的 `ops` 为空：它仍然是一个合法的投影
    /// 声明，只是推不出下界。把它做成 `Err` 反而会逼调用方"随便挑一个格式"来让构造成功。
    ///
    /// # Errors
    ///
    /// 任一宽度为零（[`Code::Invalid`]）；算子名由编码给出却没有通过 [`ContractSpec::new`] 的校验
    /// （只可能发生在编码与 `op` 注册表不一致时）。
    pub fn with_format(
        name: impl Into<String>,
        format: WeightFormat,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("线性层宽度必须非零（in = {in_features}，out = {out_features}）"),
            )
            .with("nn::Linear::with_format"));
        }
        let spec = ContractSpec::new_with_role(
            name,
            Shape::new([out_features]),
            format.operator(),
            ParamRole::Trainable,
        )?;
        Ok(Self {
            spec,
            in_features,
            out_features,
            format,
        })
    }

    crate::accessors! {
        /// 输入宽度。
        in_features -> usize = in_features,
        /// 输出宽度。
        out_features -> usize = out_features,
        /// 权重编码。它决定了本层用哪个算子，也决定了"这一层的下界是否可算"。
        format -> WeightFormat = format,
    }

    /// 本层的性能下界，见 [`Contract::bounds`]。
    pub fn bounds(&self) -> BoundsResult {
        <Self as Contract>::bounds(self)
    }
}

impl Contract for Linear {
    crate::spec_accessors!();

    fn linear(&self) -> Option<&Linear> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        // 编码没有算子：不是"下界为零"，而是"这个格式还没有可计价的内核"。
        // 如实报出而不是拿另一个格式的算子凑一个数——那正是改造前 F32 权重按 int4
        // 报出 10 MiB 的原因。
        let Some(operator) = self.format.operator() else {
            // 理由由编码自己给：`Int4 { group }` 要说出"分组不同"，`Packed4` 要说出"缺标度条目"，
            // 否则读完只知道"没有算子"，不知道该改文件还是该换内核。
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "层 `{}` 的权重{}，无法推出下界",
                    self.spec.name(),
                    self.format.unpriced_reason()
                ),
            )
            .with("nn::Linear::bounds"));
        };
        Bounds::of_op(operator, self.in_features, self.out_features).map(Some)
    }
}

/// 一份已校验规格 + 元素数，被单轴族的具体层包裹。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AxisCore {
    spec: ContractSpec,
    len: usize,
}

impl AxisCore {
    /// 构造：元素数必须非零，输出恒为一条轴 `[len]`；算子名由层绑定，构造期经
    /// [`ContractSpec::new`] 校验。
    ///
    /// # Errors
    ///
    /// 元素数为零（[`Code::Invalid`]）。`kind` 只用于诊断文案点名族别。
    pub(crate) fn new(
        name: impl Into<String>,
        op: &'static str,
        len: usize,
        kind: &str,
        role: ParamRole,
    ) -> Result<Self> {
        if len == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{kind}元素数必须非零（len = {len}）"),
            )
            .with("nn::AxisCore::new"));
        }
        let spec = ContractSpec::new_with_role(name, Shape::new([len]), [op], role)?;
        Ok(Self { spec, len })
    }

    pub(crate) fn name(&self) -> &str {
        self.spec.name()
    }

    pub(crate) fn output(&self) -> &Shape {
        self.spec.output()
    }

    pub(crate) fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    /// 元素数（被处理的轴长度）。
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// 本层下界：单轴算子的第二形参被骨架忽略，补 `1` 只满足 `shape_ctx` 签名，不参与代价。
    pub(crate) fn bounds(&self) -> BoundsResult {
        let op = self
            .spec
            .ops()
            .first()
            .map(String::as_str)
            .unwrap_or_default();
        Bounds::of_ctx(op, self.len, 1).map(Some)
    }
}

/// 为一个已显式声明的层绑定算子名、构造器、访问器与 [`Contract`] 委托。
#[doc(hidden)]
#[macro_export]
macro_rules! layer_delegate {
    ($ty:ident, $op:literal, $kind:literal, $role:expr) => {
        impl $ty {
            /// 本层绑定的算子名。
            pub const OP: &str = $op;

            /// 由层名与元素数构造；元素数为零、或算子未在 `op` 注册时返回说明。
            pub fn new(name: impl Into<String>, len: usize) -> diagnostic::Result<Self> {
                Ok(Self {
                    inner: $crate::declare::AxisCore::new(name, Self::OP, len, $kind, $role)?,
                })
            }

            /// 元素数（被处理的轴长度）。
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            /// 是否为空：构造期已拒绝零长度，因此恒为 `false`。
            pub fn is_empty(&self) -> bool {
                self.inner.len() == 0
            }
        }

        impl $crate::Contract for $ty {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn output(&self) -> &$crate::Shape {
                self.inner.output()
            }

            fn spca(&self) -> &$crate::ContractSpec {
                self.inner.spca()
            }

            fn bounds(&self) -> $crate::BoundsResult {
                self.inner.bounds()
            }
        }
    };
}

/// 单轴层的整份声明：`struct { inner: AxisCore }`、`derive` 与 `layer_delegate!` 接线。
///
/// 单轴层只有「绑哪个算子、扮演什么参数角色」两个自由度，其余逐字相同；把两者写成一行，
/// 新增一层就与在算子表里登记一条同宽，也不会漏掉 derive 或契约接线。
#[macro_export]
macro_rules! axis {
    ($(#[$meta:meta])* $ty:ident, $op:literal) => {
        $crate::axis!($(#[$meta])* $ty, $op, Trainable);
    };
    ($(#[$meta:meta])* $ty:ident, $op:literal, $role:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $ty {
            inner: $crate::declare::AxisCore,
        }
        $crate::layer_delegate!($ty, $op, "单轴层", $crate::ParamRole::$role);
    };
}

/// 运行期配置化的单轴层：算子名与参数角色由**构造参数**给出，而不是编译期的类型。
///
/// 编译期单轴层（`RmsNorm`、`Relu` …）用类型表达「绑哪个算子」，一层一个类型；这一层把同样的
/// 能力收进一个类型，让装配侧在运行期从配置（算子名 + 长度 + 角色）拼出层，而契约、访问器与
/// 下界推导与编译期层**逐字相同**——都只是对 [`AxisCore`] 的委托。三者都不产生新类型，
/// `Vec<OpLayer>` 就是一张可直接迭代、可直接做键的配置表。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpLayer {
    inner: AxisCore,
    op: &'static str,
}

impl OpLayer {
    /// 由层名、算子名、元素数与参数角色构造。算子名取自登记过的名字（如 `"relu_f32"`）。
    ///
    /// # Errors
    ///
    /// 元素数为零，或算子名未在 `op` 注册 / 没有语义表（[`ContractSpec::new`] 上报）。
    pub fn new(
        name: impl Into<String>,
        op: &'static str,
        len: usize,
        role: ParamRole,
    ) -> Result<Self> {
        Ok(Self {
            inner: AxisCore::new(name, op, len, "配置化单轴层", role)?,
            op,
        })
    }

    /// 元素数（被处理的轴长度）。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 是否为空：构造期已拒绝零长度，因此恒为 `false`。
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// 本层绑定的算子名：就是构造时给的那一个。
    pub fn op(&self) -> &'static str {
        self.op
    }
}

impl crate::Contract for OpLayer {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn output(&self) -> &crate::Shape {
        self.inner.output()
    }

    fn spca(&self) -> &crate::ContractSpec {
        self.inner.spca()
    }

    fn bounds(&self) -> BoundsResult {
        self.inner.bounds()
    }
}

/// 对「持有 `spec: ContractSpec` 字段」的层，把层名/输出形状/规格三件委托收口一份；
/// 各层仍在自己的 `Contract` impl 里补类型化访问器与下界。
#[macro_export]
macro_rules! spec_accessors {
    () => {
        fn name(&self) -> &str {
            self.spec.name()
        }

        fn output(&self) -> &$crate::Shape {
            self.spec.output()
        }

        fn spca(&self) -> &$crate::ContractSpec {
            &self.spec
        }
    };
}

/// 下界只由「一次上下文算子：`numel / 末轴` 行 × 末轴宽」决定的层。
///
/// 这一族（卷积、反卷积、池化、插值、逐元）的输出形状已经把两件事写清楚了：末轴是通道，
/// 前面各轴合成外层行数。手写这些层就是把同一句 `of_ctx(op, 行, 通道)` 连同它的乘法抄 N 遍，
/// 而抄错的后果是编译器不会报的静默错价——下界实参与输出形状本来就该同源，本宏让它们同源。
///
/// 一行调用替掉五行，顺带保证不会漏掉 `spec_accessors!()`——漏了就不实现 `Contract`。
/// 下界口径不同（按投影一对宽度、按归约到标量、按格式计价）的层仍手写，本宏只收这一种。
///
/// 末轴取 `unwrap_or(1)` 只为标量形状不 panic、除法不除零：零轴层在构造期已被各层自己拒绝
/// （`nonzero_channels` / `nonzero`），这里不再伪造第二道校验。
#[macro_export]
macro_rules! priced {
    ($(#[$doc:meta])* $ty:ident, $op:expr) => {
        impl $crate::Contract for $ty {
            $crate::spec_accessors!();

            $(#[$doc])*
            fn bounds(&self) -> $crate::BoundsResult {
                let out = self.output();
                let width = out.dim(out.rank().saturating_sub(1)).unwrap_or(1);
                let rows = out.numel() / width.max(1);
                $crate::Bounds::of_ctx($op, rows, width).map(Some)
            }
        }
    };
}

/// 只读访问器表：`名字 -> 类型 = 字段`，字段必填（访问器返字段值，不做计算）。
///
/// 一行一条的写法与 [`layer_delegate!`] 同源：层数多、访问器都是"返字段"，手写会把
/// `pub fn x(&self) -> T { self.x }` 平铺成四行 × 每层数条。带计算的访问器（如 `bytes()`、
/// `intensity()`）不进表，仍手写——表只收"读一个字段"这一种语义。
#[doc(hidden)]
#[macro_export]
macro_rules! accessors {
    ($($(#[$attr:meta])* $name:ident -> $ty:ty = $field:ident),+ $(,)?) => {
        $(
            $(#[$attr])*
            pub fn $name(&self) -> $ty {
                self.$field
            }
        )+
    };
}

/// 一次调用的性能下界：读、写两个方向的最少流量与最少浮点运算次数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    read: u64,
    write: u64,
    flops: f64,
}

impl Bounds {
    /// 零：还没有声明任何算子。
    pub const ZERO: Self = Self {
        read: 0,
        write: 0,
        flops: 0.0,
    };

    /// 由算子语义推出下界。
    ///
    /// 两个宽度按 [`Model::projection`](crate::model::Model::projection) 的口径给 `(输入宽度,
    /// 输出宽度)`：算子自己的轴角色把它们翻成具名轴，层不必知道规约轴在表达式里叫 `m` 还是 `k`，
    /// 也不必知道量化分组的派生轴怎么切。
    ///
    /// # Errors
    ///
    /// - 算子没有语义表（[`Code::Missing`]）：声明了却不能对拍，也就无法定义它的下界；
    /// - `(输入宽度, 输出宽度)` 不是该算子的合法形状（[`Code::Invalid`]）：由算子自己的代价模型
    ///   报出，例如量化 GEMV 的规约轴必须是分组因子的整数倍。
    pub fn of_op(op: &str, in_features: usize, out_features: usize) -> Result<Self> {
        let model = model_of(op, "nn::Bounds::of_op")?;
        let shape = model.projection(in_features, out_features)?;
        Ok(Self::of_cost((model.cost)(&shape)?))
    }

    /// 通用形状入口：把 `(a, b)` 交给算子自己的 [`Model::shape_ctx`](crate::model::Model::shape_ctx)
    /// 翻成具名轴，再由 [`Model::cost`](crate::model::Model::cost) 推出下界。
    ///
    /// 与 [`Bounds::of_op`] 的分工：投影算子按轴角色（输入/输出）把宽度填进规约轴与输出轴；单轴逐元素、
    /// 归约、注意力这类形状由算子骨架自己定义的，走本入口。两条路径都只调用算子语义表的
    /// `shape_ctx` 与 `cost`，`nn` 里没有第二份流量常数。
    ///
    /// # Errors
    ///
    /// - 算子没有语义表（[`Code::Missing`]）：声明了却不能对拍，也就无法定义下界；
    /// - 形状上下文构造失败，或代价模型拒绝该形状（错误原样上报并点名算子）。
    pub fn of_ctx(op: &str, a: usize, b: usize) -> Result<Self> {
        let model = model_of(op, "nn::Bounds::of_ctx")?;
        let shape = (model.shape_ctx)(a, b).map_err(|error| error.with("nn::Bounds::of_ctx"))?;
        let cost = (model.cost)(&shape).map_err(|error| error.with("nn::Bounds::of_ctx"))?;
        Ok(Self::of_cost(cost))
    }

    /// 一串算子在**同一个形状**上的下界之和：组合层自己的下界。
    ///
    /// 空列表是 [`Bounds::ZERO`]——"没有可计价的算子"与"代价为零"在数值上一致，两者的区别由
    /// 调用方的清单（`Stack` 的未覆盖/无法计价条目）表达，而不是靠一个特殊值。
    ///
    /// 每个算子都用同一组 `(m, n)`：这只对"同一条投影上叠了几个算子"成立。轴映射不同的复合层
    /// （RMSNorm 的 `[batch, seq, d_model]`、Attention 的 `seq²`）需要先把形状上下文抬进算子
    /// 语义表，本函数不替它们伪造一对 `(m, n)`。
    ///
    /// # Errors
    ///
    /// 任一算子没有语义表或该形状对它不合法时，错误原样上报并点名该算子。
    pub fn of_ops(ops: &[String], in_features: usize, out_features: usize) -> Result<Self> {
        let mut total = Self::ZERO;
        for op in ops {
            total = total.plus(
                Self::of_op(op, in_features, out_features)
                    .map_err(|error| error.with(format!("算子 `{op}`")))?,
            );
        }
        Ok(total)
    }

    /// 由算子语义表给出的代价转成下界：字段一一对应，不做二次解释。
    fn of_cost(cost: crate::einsum::Cost) -> Self {
        Self {
            read: cost.read,
            write: cost.write,
            flops: cost.flops,
        }
    }

    /// 两笔下界相加：一串层（或一个复合算子）的总下界。
    pub fn plus(self, other: Self) -> Self {
        Self {
            read: self.read.saturating_add(other.read),
            write: self.write.saturating_add(other.write),
            flops: self.flops + other.flops,
        }
    }

    crate::accessors! {
        /// 最少读入字节。
        read -> u64 = read,
        /// 最少写出字节。
        write -> u64 = write,
    }

    /// 最少流量（字节）：读与写合计的**派生**量。
    pub fn bytes(&self) -> u64 {
        self.read.saturating_add(self.write)
    }

    /// 最少浮点运算次数。
    pub fn flops(&self) -> f64 {
        self.flops
    }

    /// 算术强度（flop/字节）：roofline 的横轴。用合计流量，因为横轴本来就只关心"每字节算几次"。
    pub fn intensity(&self) -> f64 {
        let bytes = self.bytes();
        if bytes == 0 {
            0.0
        } else {
            self.flops / bytes as f64
        }
    }

    /// 本下界是否落在那笔下界之内：**每个方向**都不更高。
    pub fn moves_within(self, other: Self) -> bool {
        self.read <= other.read && self.write <= other.write && self.flops <= other.flops
    }
}

/// 取算子的语义表：没有就报"声明了却不能对拍"，并点名是哪个入口要的。
///
/// 两个入口（投影 / 通用形状）的诊断只差这个点名，措辞收在这里一份。
fn model_of(op_name: &str, site: &'static str) -> Result<crate::model::Model> {
    crate::registry::model(op_name).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("算子 `{op_name}` 没有语义表，无法推出下界"),
        )
        .with(site)
    })
}

// ============================ 权重编码 ============================
//
// 权重编码是层的「存储事实」，以及它决定了哪个算子。它与上面的 `Bounds` 同居本文件，用的是同一条
// 判据：两者都是层要向装配侧声明的**事实**——形状由构造参数给，算子由编码给，代价由算子语义表给。
// 拆成两个文件时，`format` 只被 `declare` 的消费者用，且从无独立导出路径，故合并。

/// 权重编码。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// 32 位浮点。
    F32,
    /// 16 位浮点。
    F16,
    /// bfloat16。
    Bf16,
    /// 8 位整数（int8 / uint8 的语义由算子决定，本层不猜）。
    I8,
    /// 2 位打包整数，但**没有声明分组**：与 [`WeightFormat::Packed4`] 同一类"缺标度条目"的编码。
    Packed2,
    /// 4 位打包整数，但**没有声明分组**：元素类型本身不携带"多少个元素共享一个缩放因子"。
    Packed4,
    /// 2 位打包整数 + 每 `group` 个元素一个缩放因子。
    Int2 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
    /// 4 位打包整数 + 每 `group` 个元素一个缩放因子。
    Int4 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
    /// 8 位逐元素整数 + 每 `group` 个元素一个缩放因子（对称 int8，不打包）。
    Int8 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
}

/// 元素类型 ↔ 权重编码的唯一事实源：`of_dtype` 与 `dtype()` 是它的两个方向。
const FORMATS: &[(DType, WeightFormat)] = &[
    (DType::F32, WeightFormat::F32),
    (DType::F16, WeightFormat::F16),
    (DType::BF16, WeightFormat::Bf16),
    (DType::I8, WeightFormat::I8),
    (DType::U4, WeightFormat::Packed4),
    (DType::U2, WeightFormat::Packed2),
];

impl WeightFormat {
    /// 对称 int2 的默认格式：32 元素分组（与 int4 共用内核的 group 声明档），即 [`Linear::I2`] 的输入格式。
    pub const INT2: Self = Self::Int2 { group: 32 };

    /// 量化侧的默认格式：int4 + 32 元素分组，即 [`Linear::OP`] 的输入格式。
    pub const INT4: Self = Self::Int4 { group: 32 };

    /// 对称 int8 的默认格式：32 元素分组，即 [`Linear::I8`] 的输入格式。
    pub const INT8: Self = Self::Int8 { group: 32 };

    /// 由权重的元素类型推出格式：查 `FORMATS` 的那一个方向。
    #[must_use]
    pub fn of_dtype(dtype: DType) -> Option<Self> {
        FORMATS
            .iter()
            .find(|(element, _)| *element == dtype)
            .map(|(_, format)| *format)
    }

    /// 显式声明 4 位量化的分组。
    pub const fn int4(group: usize) -> Self {
        Self::Int4 { group }
    }

    /// 显式声明 2 位量化的分组。
    pub const fn int2(group: usize) -> Self {
        Self::Int2 { group }
    }

    /// 显式声明 8 位逐元素量化的分组。
    pub const fn int8(group: usize) -> Self {
        Self::Int8 { group }
    }

    /// 这种编码在内存里的元素类型：查 `FORMATS` 的另一个方向。
    pub fn dtype(self) -> DType {
        match self {
            Self::Int4 { .. } => DType::U4,
            Self::Int8 { .. } => DType::I8,
            Self::Int2 { .. } => DType::U2,
            other => FORMATS
                .iter()
                .find(|(_, format)| *format == other)
                .map_or(DType::U4, |(element, _)| *element),
        }
    }

    /// 本格式对应的已声明算子；`None` = 还没有这个格式的算子，因此这一层无法计价。
    pub fn operator(self) -> Option<&'static str> {
        match self {
            Self::F32 => Some(Linear::DENSE),
            other => other
                .quantized()
                .filter(|quant| Some(quant.group) == quant.declared)
                .map(|quant| quant.op),
        }
    }

    /// 为什么这种编码取不到算子：一句话说清"缺什么"，而不是只说"没有"。
    pub fn unpriced_reason(self) -> String {
        // 量化格式的说法只有一种：缺的是「分组对得上」而不是算子本身。
        if let Some(quant) = self.quantized() {
            return quantized_reason(quant);
        }
        match self {
            Self::Packed2 | Self::Packed4 => format!(
                "是 {} 位打包，但找不到它的标度条目，分组不明",
                self.dtype().bits()
            ),
            // 稠密算子目前只有 32 位那一条：16 位的元素宽度不同，套上去读下界会正好差两倍。
            Self::F16 | Self::Bf16 => format!("编码是 {self}，还没有 16 位元素的投影算子"),
            other => format!("编码是 {other}，还没有这个格式的投影算子"),
        }
    }

    /// 带分组的量化格式描述；非量化返回 `None`。
    fn quantized(self) -> Option<Quant> {
        match self {
            Self::Int2 { group } => Some(Quant {
                group,
                op: Linear::I2,
                declared: kernel_group(Linear::I2),
                label: "int2",
            }),
            Self::Int4 { group } => Some(Quant {
                group,
                op: Linear::OP,
                declared: kernel_group(Linear::OP),
                label: "int4",
            }),
            Self::Int8 { group } => Some(Quant {
                group,
                op: Linear::I8,
                declared: kernel_group(Linear::I8),
                label: "int8",
            }),
            _ => None,
        }
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Packed2 | Self::Packed4 => {
                write!(formatter, "{} 位打包（未声明分组）", self.dtype().bits())
            }
            Self::Int2 { group } => write!(formatter, "int2（分组 {group}）"),
            Self::Int4 { group } => write!(formatter, "int4（分组 {group}）"),
            Self::Int8 { group } => write!(formatter, "int8（分组 {group}）"),
            // 其余四种编码的名字就是元素类型的规范名，不在这里再写一遍字面量。
            element => formatter.write_str(element.dtype().name()),
        }
    }
}

/// 带分组的量化格式：文件声明的分组、对应算子、内核声明分组与诊断文案标签。
struct Quant {
    group: usize,
    op: &'static str,
    declared: Option<usize>,
    label: &'static str,
}

/// 量化分组对不上内核声明档时的诊断文案。
fn quantized_reason(quant: Quant) -> String {
    match quant.declared {
        Some(declared) => format!(
            "按 {} 个元素一组量化（{}），而内核声明的是 {declared} 个元素一组：分组不同不能共用同一条内核",
            quant.group, quant.label
        ),
        None => format!(
            "按 {} 个元素一组量化，但没有任何算子声明过 {} 的分组",
            quant.group, quant.label
        ),
    }
}

/// 算子在语义表里声明的分组；`None` = 算子未注册或未声明分组。
fn kernel_group(op: &str) -> Option<usize> {
    crate::registry::model(op).map(|model| model.group as usize)
}

#[cfg(test)]
#[path = "../tests/bounds_tests.rs"]
mod bounds_tests;

#[cfg(test)]
#[path = "../tests/format_tests.rs"]
mod format_tests;
