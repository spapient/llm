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

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{BoundsResult, ContractSpec, ParamRole};

/// 一份已校验规格 + 元素数，被单轴族的具体层包裹。
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// 对"持有 `spec: ContractSpec` 字段"的层，把层名/输出形状/规格三件委托收口一份；
/// 各层仍在自己的 `Contract` impl 里补类型化访问器与下界。
#[doc(hidden)]
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
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $ty {
            inner: $crate::declare::AxisCore,
        }
        $crate::layer_delegate!($ty, $op, "单轴层", $crate::ParamRole::$role);
    };
}

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
    /// 两个宽度按 [`Model::projection`](op::model::Model::projection) 的口径给 `(输入宽度,
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

    /// 通用形状入口：把 `(a, b)` 交给算子自己的 [`Model::shape_ctx`](op::model::Model::shape_ctx)
    /// 翻成具名轴，再由 [`Model::cost`](op::model::Model::cost) 推出下界。
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
    fn of_cost(cost: op::einsum::Cost) -> Self {
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
fn model_of(op_name: &str, site: &'static str) -> Result<op::model::Model> {
    op::model(op_name).ok_or_else(|| {
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
