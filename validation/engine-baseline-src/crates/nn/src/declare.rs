//! 单轴族与归约族共享的规格骨架、委托宏与性能下界 [`Bounds`]。
//!
//! 前半是元素数驱动的 `AxisCore` 与四张装配宏（形状口径、访问器、规格透传、下界计价）；后半
//! 是由算子语义表推出的读/写下界。层只声明，不写第二份常数：形状由构造参数给出，代价由算子名
//! 去 [`registry`](crate::registry) 里查，`nn` 不认识除算子名以外的领域知识。
//!
//! 契约三件（[`Contract`] / [`ContractSpec`] / [`Linear`]）在 [`crate::contract`]，权重编码
//! 在 [`crate::weight`]，本模块经 crate 根路径引用它们。
//!
//! # 下界有三种状态，不是一个数
//!
//! [`crate::Contract::bounds`] 返回 [`crate::BoundsResult`]，三种情况刻意分开：
//!
//! | 返回值 | 含义 |
//! |---|---|
//! | `Ok(Some(bounds))` | 有下界，且它由算子语义推出（不是手写常数） |
//! | `Ok(None)` | 层已声明，但**没有可计价的形状**（例如注解层没给出输入宽度） |
//! | `Err(..)` | 有算子却算不出来：形状不合法（没有合法内核），或该格式还没有算子 |
//!
//! `Err` 不会被 [`crate::Stack::floor`] 放大成"整张模型算不出下界"：它进无法计价清单，其余层照常相加。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::ParamRole;

/// 一份已校验规格 + 元素数，被单轴族的具体层包裹。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AxisCore {
    spec: crate::ContractSpec,
    len: usize,
}

impl AxisCore {
    /// 构造：元素数必须非零，输出恒为一条轴 `[len]`；算子名由层绑定，构造期经
    /// [`crate::ContractSpec::new`] 校验。
    ///
    /// # Errors
    ///
    /// 元素数为零（[`Code::Invalid`]）。`kind` 只用于诊断文案点名族别。
    pub(crate) fn new(
        name: impl Into<String>,
        op: impl Into<String>,
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
        let spec = crate::ContractSpec::new_with_role(name, Shape::new([len]), [op], role)?;
        Ok(Self { spec, len })
    }

    pub(crate) fn name(&self) -> &str {
        self.spec.name()
    }

    pub(crate) fn output(&self) -> &Shape {
        self.spec.output()
    }

    pub(crate) fn spca(&self) -> &crate::ContractSpec {
        &self.spec
    }

    /// 元素数（被处理的轴长度）。
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// 本层下界：单轴算子的第二形参被骨架忽略，补 `1` 只满足 `shape_ctx` 签名，不参与代价。
    pub(crate) fn bounds(&self) -> crate::BoundsResult {
        let op = self.spec.ops().first().map_or_default(String::as_str);
        Bounds::of_geometry(op, LayerGeometry::axis(self.len)).map(Some)
    }
}

/// 为一个已显式声明的层绑定算子名、构造器、访问器与 [`crate::Contract`] 委托。
#[doc(hidden)]
#[macro_export]
macro_rules! layer_delegate {
    ($ty:ident, $op:literal, $kind:literal, $role:expr) => {
        impl $ty {
            /// 本层绑定的算子名。
            pub const OP: &str = $op;

            /// 由层名与元素数构造；元素数为零、或算子未在 `registry` 注册时返回说明。
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
    ($($(#[$meta:meta])* $ty:ident => $op:literal $(, $role:ident)?);* $(;)?) => {
        $($crate::axis!($(#[$meta])* $ty, $op $(, $role)?);)*
    };
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
/// 下界推导与编译期层**逐字相同**——都只是对 `AxisCore` 的委托。三者都不产生新类型，
/// `Vec<OpLayer>` 就是一张可直接迭代、可直接做键的配置表。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpLayer {
    inner: AxisCore,
}

impl OpLayer {
    /// 由层名、算子名、元素数与参数角色构造。算子名可直接来自运行时配置。
    ///
    /// # Errors
    ///
    /// 元素数为零，或算子名未在 `registry` 注册 / 没有语义表（[`crate::ContractSpec::new`] 上报）。
    pub fn new(
        name: impl Into<String>,
        op: impl Into<String>,
        len: usize,
        role: ParamRole,
    ) -> Result<Self> {
        Ok(Self {
            inner: AxisCore::new(name, op, len, "配置化单轴层", role)?,
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

    /// 本层绑定的算子名，直接读自已校验规格，不另存一份字符串。
    pub fn op(&self) -> &str {
        self.inner
            .spca()
            .ops()
            .first()
            .map_or_default(String::as_str)
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

    fn bounds(&self) -> crate::BoundsResult {
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
                $crate::Bounds::of_geometry($op, $crate::LayerGeometry::shape(rows, width)).map(Some)
            }
        }
    };
}

/// 只读访问器表：`名字 -> 类型 = 字段` 返回副本，`= &字段` 返回借用；两者都不做计算。
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
    ($($(#[$attr:meta])* $name:ident -> $ty:ty = &$field:ident),+ $(,)?) => {
        $(
            $(#[$attr])*
            pub fn $name(&self) -> $ty {
                &self.$field
            }
        )+
    };
}

/// 层交给算子的几何：既保留宽度，也保留算子应使用哪一种轴映射。
///
/// `Projection` 由输入／输出角色填轴；`Shape` 把 `(outer, width)` 交给算子自己的形状投影。
/// 两者不能只靠一对整数猜测，矩阵乘、Embedding 与投影正是因此需要不同入口。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerGeometry {
    /// 投影几何。
    Projection {
        /// 输入宽度。
        input: usize,
        /// 输出宽度。
        output: usize,
    },
    /// 算子自定义形状。
    Shape {
        /// 外层行数。
        outer: usize,
        /// 末轴宽度。
        width: usize,
    },
}

impl LayerGeometry {
    /// 构造一次投影几何。
    pub const fn projection(input: usize, output: usize) -> Self {
        Self::Projection { input, output }
    }

    /// 构造一次算子自定义形状。
    pub const fn shape(outer: usize, width: usize) -> Self {
        Self::Shape { outer, width }
    }

    /// 构造单轴几何；第二形参只是算子签名槽，不参与代价。
    pub const fn axis(len: usize) -> Self {
        Self::shape(len, 1)
    }
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

    /// 由统一层几何推出下界；旧 `of_op` / `of_ctx` 暂作兼容适配器。
    ///
    /// # Errors
    ///
    /// 算子缺少语义表，或几何不满足算子自己的形状与代价约束时返回诊断。
    pub fn of_geometry(op: &str, geometry: LayerGeometry) -> Result<Self> {
        match geometry {
            LayerGeometry::Projection { input, output } => Self::of_op(op, input, output),
            LayerGeometry::Shape { outer, width } => Self::of_ctx(op, outer, width),
        }
    }

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
                Self::of_geometry(op, LayerGeometry::projection(in_features, out_features))
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

#[cfg(test)]
#[path = "../tests/bounds_tests.rs"]
mod bounds_tests;
