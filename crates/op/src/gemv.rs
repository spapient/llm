//! 一条 GEMV 骨架 + 权重格式契约。
//!
//! `gemv_i4` 与 `gemv_f32` 是同一副骨架的两种权重格式：求和表达式只差一条派生轴，样本只差
//! 权重的取值，参考只差内层循环。差异全部收进 [`Format`] 的关联常量与方法，骨架（表达式缓存、
//! 代价、样本、校验、遍历）只有一份——按格式**单态化**，因此 `Model` 里仍是三个 `fn` 指针，
//! 每个指针指向一条独立的机器码，而不是一个在热路径上查 `match` 的分发器。
//!
//! # 为什么是 trait + const 泛型，而不是 `enum`
//!
//! 格式在编译期就确定：量化格式的每个元素占半个字节、稠密格式没有标度轴、累加按固定块宽推进。
//! 写成运行时枚举会让 `forward` 在每行开始前多一次分支，也会让"分组必须是偶数""每行字节数
//! 折半"这类约束从编译期退化成运行期检查。
//!
//! 哪些东西进类型、哪些留在数据里，是按**内核自己怎么收**划的：
//!
//! | 事实 | 落在哪 | 理由 |
//! |---|---|---|
//! | 权重元素类型、位宽、解包方式 | [`Format::Elem`] / [`Format::OPERANDS`] / `forward` | 只有代码能表达，且必须进单态化 |
//! | 稠密格式的分块宽度 | [`Dense`] 的 const 参数 | 它出现在最内层循环的 `chunks` 上，是真正的热路径常数 |
//! | 量化格式的声明粒度 | [`Quantized`] 的 const 参数 | 它决定派生轴除数与样本里的标度个数（代价与样本，冷路径） |
//! | 实际分组大小 | 运行时数据（`scales` 长度 / `gsz` 实参） | 模板本身就是 `int gsz` 的运行时参数，钉进类型会让训练侧按 8 元素分组的调用编译不过 |
//!
//! 最后一行是刻意的：把"内核已经做成运行时参数"的东西改成类型参数，换来的是每次不同分组都要
//! 一份单态化副本与一次调用点上的匹配，而不是省下一条分支。

use crate::einsum::{Bits, Formula, FormulaCache, OperandSpec};
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use crate::quant::Row;
use diagnostic::{Code, Error};

/// 本骨架的基本轴：`m` 是规约轴（输入宽度），`n` 是输出轴。
///
/// 字母属于表达式，角色属于层：`nn::Linear` 按 [`AxisRole::Input`] / [`AxisRole::Output`]
/// 填宽度，不需要知道它们叫 `m` 和 `n`。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'm',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'n',
    },
];

/// 样本缓冲里能装的元素类型：闭集 `f32` / `u8`（[`Values`] 的两个变体）。
///
/// 这是一条**编译期**契约：`Format::Elem` 只能是这里的两个实现之一，因此
/// "从样本里取权重"不需要在 `reference` 里写 `match` 分发——`<F::Elem as Element>::borrow`
/// 在单态化时就只剩一个分支。
pub trait Element: Copy + PartialEq + std::fmt::Debug + 'static {
    /// 诊断里怎么称呼它。
    const NAME: &'static str;

    /// 从样本缓冲里借出同类元素；类型不符返回 `None`。
    fn borrow(values: &Values) -> Option<&[Self]>;

    /// 直接从 `u8` 切片借出本元素；`u8` 元素返回原样，`f32` 返回 `None`。
    ///
    /// 测试侧闭式重算按位流解包时用：打包格式的元素就是 `u8`，不必再绕一遍 `Values` 变体。
    fn borrow_u8(slice: &[Self]) -> Option<&[u8]>;

    /// 从打包字节构造本元素向量；`u8` 元素原样回收，`f32` 不可用。
    fn from_bytes(bytes: Vec<u8>) -> Vec<Self>;

    /// 装成同名缓冲。
    fn buffer(name: &'static str, values: Vec<Self>) -> Buffer;
}

impl Element for f32 {
    const NAME: &'static str = "f32";

    fn borrow(values: &Values) -> Option<&[Self]> {
        match values {
            Values::F32(values) => Some(values),
            Values::U8(_) => None,
        }
    }

    fn borrow_u8(_slice: &[Self]) -> Option<&[u8]> {
        None
    }

    fn from_bytes(_bytes: Vec<u8>) -> Vec<Self> {
        // f32 逐元素格式不参与打包取样：此路径在单态化时不可达，留空切片作安全占位。
        Vec::new()
    }

    fn buffer(name: &'static str, values: Vec<Self>) -> Buffer {
        Buffer {
            name,
            values: Values::F32(values),
        }
    }
}

impl Element for u8 {
    const NAME: &'static str = "u8";

    fn borrow(values: &Values) -> Option<&[Self]> {
        match values {
            Values::F32(_) => None,
            Values::U8(values) => Some(values),
        }
    }

    fn borrow_u8(slice: &[Self]) -> Option<&[u8]> {
        Some(slice)
    }

    fn from_bytes(bytes: Vec<u8>) -> Vec<Self> {
        bytes
    }

    fn buffer(name: &'static str, values: Vec<Self>) -> Buffer {
        Buffer {
            name,
            values: Values::U8(values),
        }
    }
}

/// 格式自证：派生轴的除数必须就是形状粒度。
///
/// 两处都拿 `GRAIN` 当分组：`cost` 用 [`Format::DERIVED`] 的除数造派生轴，`sample` 用
/// [`Format::GRAIN`] 算每行标度个数。写歪一个数不会让任何一步失败——代价算的是一个分组、
/// 样本给的是另一个分组，对拍比的是两种不同的语义。因此这条关系必须是**同一个常数**。
pub const fn derived_matches_grain(grain: usize, derived: Option<(char, char, usize)>) -> bool {
    match derived {
        Some((_, _, divisor)) => divisor == grain,
        None => true,
    }
}

/// 权重格式：一条 GEMV 里随精度变化的那部分。
///
/// 实现者只写"这份格式怎么表达、怎么解包"，骨架（[`cost`] / [`sample`] / [`reference()`]）由
/// 本模块提供。所有关联项都是编译期常量：没有一件需要运行期再判断一次。
pub trait Format {
    /// 权重在样本里的元素类型。
    type Elem: Element;

    /// 一条 einsum 求和表达式。
    const EXPR: &'static str;

    /// 表达式各项的位宽与方向，顺序与 [`Format::EXPR`] 的项一一对应。
    const OPERANDS: &'static [OperandSpec];

    /// 形状粒度：`m` 必须是它的正整数倍；`1` 表示任意宽度。
    ///
    /// 它就是 [`crate::model::Model::group`]，也是"这一份格式按多少个元素一组记账"的声明。
    const GRAIN: usize;

    /// 派生轴 `(派生轴, 基准轴, 除数)`：量化格式声明一条（`g = m / GRAIN`），稠密格式没有。
    ///
    /// 派生轴只计流量、不计运算次数，见 [`crate::einsum::Extents::group`]。
    const DERIVED: Option<(char, char, usize)>;

    /// 标度取值函数；`None` 表示这份格式没有标度缓冲（每个标度不带参数）。
    ///
    /// 一个常数同时回答"有没有标度"与"标度怎么取"，于是样本与参考不会各拿一半事实。
    const SCALES: Option<fn(usize) -> f32>;

    /// 模板每轮推进的元素数，也是样本里 `gsz` 的取值。
    const STEP: usize;

    /// 契约自证：本格式的声明必须自洽。
    ///
    /// 这是一个**编译期**判据，不是运行期检查：`#[annotation::model]` 把它抄进
    /// `const _: () = <名字 as Format>::CHECK;`，于是写错粒度或派生轴除数的格式别名直接编译不过。
    /// 放在 trait 里而不是宏里，是因为规则属于格式契约，注解只负责让它被求值。
    const CHECK: () = {
        assert!(Self::GRAIN > 0, "形状粒度必须大于 0");
        assert!(
            derived_matches_grain(Self::GRAIN, Self::DERIVED),
            "派生轴的除数必须等于形状粒度：代价里的分组与样本里的分组必须同源"
        );
    };

    /// 这份格式的缓存单元格（每个格式一份）。
    ///
    /// 实现者写成函数体里的 `static CACHE: FormulaCache = FormulaCache::new(); &CACHE`。缓存必须
    /// 按**格式**分开：写在骨架泛型里的 `static` 在 Rust 里只有一份，两种格式会共用同一个单元格，
    /// 先跑的那一份会让另一份拿到错误的轴表。
    fn cache() -> &'static FormulaCache;

    /// 这份格式的表达式：第一次调用时解析并校验，之后复用。
    ///
    /// # Errors
    ///
    /// 表达式与操作数表不自洽时返回说明（带 `op::gemv::formula` 帧）。
    fn formula() -> Result<&'static Formula, Error> {
        Self::cache().get(Self::EXPR, Self::OPERANDS, "op::gemv::formula")
    }

    /// 一行权重占多少个元素。
    fn row(m: usize) -> usize;

    /// 确定性权重：同一形状永远得到同一组取值（cutile README §12.3 的"可复现"要求）。
    fn weights(m: usize, n: usize) -> Vec<Self::Elem>;

    /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
    ///
    /// 与 [`reference()`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配：重放时每一步
    /// 都要算，稳定态不该再向分配器要一块新的 `Vec`。
    fn forward(
        x: &[f32],
        weights: &[Self::Elem],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()>;
}

/// 4 位打包配置标记：骨架测试按不同粒度复用这一份配置；生产 i4 lane 用同名 [`I4`](crate::gemv_i4::I4) 别名。
pub struct QuantI4;

impl PackedRow for QuantI4 {
    const ROW: Row = crate::quant::I4;

    fn cell() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }
}

/// 量化格式：`GRAIN` 个元素共用一条标度，每两个元素占一个字节（4 位一元素）。
///
/// 底层复用 [`Packed`] 的 4 位配置；`GRAIN` 是**声明粒度**（代价的派生轴除数、样本里每行的
/// 标度数），不是运行期分组：内核模板把分组当 `gsz` 实参收，所以 `forward` 仍按 `scales` 的
/// 长度推分组——把实际分组也钉进类型会让"同一份权重换一种分组"变成另一份单态化副本。
pub type Quantized<const GRAIN: usize> = Packed<QuantI4, GRAIN>;

/// 稠密格式：每 `BLOCK` 个元素一块累加（最后一块可以短），无打包、无标度。
///
/// `BLOCK` 出现在最内层循环的 `chunks` 上，是真正进热路径的常数：单态化之后编译器看到的是
/// `chunks(32)` 而不是 `chunks(gsz)`，块内可以整体展开与向量化。
pub struct Dense<const BLOCK: usize>;

/// 与 [`Quantized`] 的差别是不打包（每元素 1 字节、无 nibble 解包），与 [`Dense`] 的差别是
/// 有 per-group 标度。`u8` 承载 int8 的位模式，算术时按 `i8` 解释。
pub struct SymI8<const GRAIN: usize>;

/// 打包格式的静态配置提供者：把一档精度的 [`Row`] 挂到一个标记类型上。
///
/// i2 / i4 两条打包格式只在配置表 [`Row`] 的位宽、码数、零点、码表、偏移上不同；这些事实
/// 收敛到 [`Packed`] 的一份泛型 [`Format`] impl，新增一档打包精度只需写一条 `impl PackedRow`
/// （或直接复用配置表），不再平行抄表达式、操作数表、派生轴、缓存与行守卫。
pub trait PackedRow {
    /// 这档精度的静态配置（位宽 / 码数 / 零点 / 码表 / 偏移）。
    const ROW: Row;

    /// 这档精度自己的表达式缓存单元格。
    ///
    /// 缓存必须按**配置**分开：实现者写成函数体里的 `static CACHE: FormulaCache =
    /// FormulaCache::new(); &CACHE`；泛型 [`Packed`] 内的 `static` 不保证按配置单态化成独立
    /// 单元格，两种精度会共用同一个 OnceLock，先解析的那份 operands 被后跑的格式复用（流量
    /// 位宽串档）。与 [`FormulaCache`](crate::einsum::FormulaCache) 文档里"缓存留在各骨架
    /// 类型上"的口径一致。
    fn cell() -> &'static FormulaCache;
}

/// 配置驱动的打包权重格式：`GRAIN` 个元素共用一条标度，打包密度由 [`PackedRow::ROW`] 给出。
///
/// 类型参数 `P` 承载配置（编译期单态化），`G` 承载声明粒度；表达式、操作数表、派生轴除数、
/// 分组倍数守卫与行切片全部读配置，格式侧只决定"用哪一行配置"。`P` 只在关联常量上被读出，
/// 不占运行期字段，故用 [`PhantomData`] 标记这个编译期绑定。
pub struct Packed<P: PackedRow, const G: usize> {
    _config: core::marker::PhantomData<P>,
}

impl<P: PackedRow, const G: usize> Format for Packed<P, G> {
    type Elem = u8;

    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`G`] 个元素切出来的派生轴。
    const EXPR: &'static str = "nm,m,ng,n->n";

    /// 三个输入都是读，只有输出是写；权重位宽取配置表，其余为 32 位浮点。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", P::ROW.bits),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::read("scales", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    const GRAIN: usize = G;

    const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', G));

    const SCALES: Option<fn(usize) -> f32> = Some(scale_of);

    const STEP: usize = G;

    fn cache() -> &'static FormulaCache {
        // 缓存单元格由标记类型各自持有，避免泛型 static 被合并成跨配置共享的一份。
        P::cell()
    }

    fn row(m: usize) -> usize {
        // 粒度必须是打包码数的倍数，否则"每行 m/cpb 字节"没有意义；单态化时求值。
        const {
            assert!(
                G > 0 && G.is_multiple_of(P::ROW.cpb),
                "打包粒度必须是正的每字节码数倍数"
            )
        };
        m / P::ROW.cpb
    }

    fn weights(m: usize, n: usize) -> Vec<u8> {
        packed::weights(m, n, P::ROW.cpb)
    }

    fn forward(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        packed::forward_rows(x, weights, scales, out, P::ROW.cpb, P::ROW.table)
    }
}

/// 与权重格式无关的输入向量：两条算子共用同一份取值。
fn input(m: usize) -> Vec<f32> {
    (0..m)
        .map(|index| ((index * 7) % 13) as f32 * 0.25 - 1.0)
        .collect()
}

/// 量化格式共用的标度取值：`0.03125` 的 1..=4 倍，落在 2 的负幂网格，f32 精确。
pub(crate) fn scale_of(index: usize) -> f32 {
    0.031_25 * (1 + (index % 4)) as f32
}

/// 把形状对翻译成投影骨架的具名轴上下文：`m` 是规约轴、`n` 是输出轴。
///
/// 这是 [`crate::model::Model::shape_ctx`] 在投影骨架上的实例：`verify` 不再假设每条算子都能
/// 走 [`crate::model::Model::projection`]，非投影算子会提供各自的形状函数。
pub fn project_shape(m: usize, n: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('m', m), ('n', n)])
}

/// 标量代价：完全由 [`Format::EXPR`] 与操作数位宽推出，本函数不含任何流量常数。
///
/// 调用方只给基本轴；派生轴由格式自己声明——分组是算子语义，不是每个层都该抄一遍的常数。
///
/// # Errors
///
/// 形状与表达式不自洽（缺轴、长度为 0，或 `m` 不是 [`Format::GRAIN`] 的整数倍）时返回说明。
pub fn cost<F: Format>(shape: &ShapeContext) -> Result<Cost, Error> {
    let extents = match F::DERIVED {
        Some((label, base, divisor)) => shape.extents_with_group(label, base, divisor)?,
        None => shape.extents().clone(),
    };
    F::formula()?
        .cost(&extents)
        .map_err(|error| error.with("op::gemv::cost"))
}

/// 确定性样本：同一形状永远得到同一组输入。
///
/// 缓冲集合由格式决定（标度只在 [`Format::SCALES`] 给出时出现），标量 `gsz` 取
/// [`Format::STEP`]：模板按它分块，参考实现按同一个数取标度，两者不会各算一套。
pub fn sample<F: Format>(m: usize, n: usize) -> Sample {
    let mut buffers = vec![
        Buffer {
            name: "x",
            values: Values::F32(input(m)),
        },
        F::Elem::buffer("w", F::weights(m, n)),
    ];
    if let Some(scale) = F::SCALES {
        let groups = m / F::GRAIN.max(1);
        buffers.push(Buffer {
            name: "scales",
            values: Values::F32((0..n * groups).map(scale).collect()),
        });
    }
    buffers.push(Buffer {
        name: "y",
        values: Values::F32(vec![0.0; n]),
    });
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("N", n as i64), ("gsz", F::STEP as i64)],
        buffers,
    }
}

/// 从样本里取权重：名字固定是 `w`，元素类型由格式给出。
fn weights<F: Format>(sample: &Sample) -> diagnostic::Result<&[F::Elem]> {
    let buffer = sample
        .buffer("w")
        .ok_or_else(|| Error::new(Code::Missing, "样本里没有缓冲 `w`"))?;
    F::Elem::borrow(&buffer.values).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("缓冲 `w` 不是 {}：这份格式的权重元素类型", F::Elem::NAME),
        )
    })
}

/// 输出维度不能为 0：三条格式与参考实现共同的前置约束。
///
/// 写成返回值而不是就地 `if`：`?` 会在调用方的复杂度账上记一格，而这条判据的"为什么"
/// （空输出让后面的 `chunks(0)` 与 `scales.len() / n` 都失去意义）只需要说一次。
pub(crate) fn non_empty(n: usize) -> Result<(), Error> {
    if n == 0 {
        return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
    }
    Ok(())
}

/// 带标度格式的公共前置约束：`scales` 能按行均分、分组能整除输入宽度。
///
/// 返回 `(groups, group)`——每行多少组、每组多少元素。int4 与 int8 只在"一行权重多少个元素"
/// 与"分组必须是偶数"上不同，那两条留在各自的 `forward` 里：它们属于格式，不属于骨架。
pub(crate) fn scaled_groups(m: usize, n: usize, scales: usize) -> Result<(usize, usize), Error> {
    non_empty(n)?;
    if !scales.is_multiple_of(n) {
        return Err(Error::new(
            Code::Invalid,
            format!("scale 长度 {scales} 不是 {n} 的整数倍"),
        ));
    }
    let groups = scales / n;
    if groups == 0 || !m.is_multiple_of(groups) {
        return Err(Error::new(
            Code::Invalid,
            format!("scale 长度 {scales} 推不出能整除 {m} 的分组数"),
        ));
    }
    Ok((groups, m / groups))
}

/// 标度值守卫：每组分标度必须是有限值；非有限标度会把整组乘积带进 NaN/Inf，且与格式无关。
///
/// 标度个数已由 [`scaled_groups`] 推过；这里只逐条判有限，不重复推分组。
pub(crate) fn finite_scales(scales: &[f32], groups: usize) -> Result<(), Error> {
    let bad = scales.iter().any(|scale| !scale.is_finite());
    if bad {
        return Err(Error::new(
            Code::Invalid,
            format!("标度含非有限值（{groups} 组）"),
        ));
    }
    Ok(())
}

/// 打包权重（每字节 `cpb` 个码）的取样、分组校验与解包点积助手。
///
/// int2（4 码/字节）与 int4（2 码/字节）只在码宽、查表与粒度倍数上不同；打包位域的字节布局、
/// 分组推导与按行点积收口到这里，避免两条打包格式各抄一遍。int8 不打包，不走本模块。
pub mod packed {
    use super::{finite_scales, required_weights, scaled_groups};
    use diagnostic::{Code, Error};

    /// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节，同一形状取值可复现。
    pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
        (0..n * (m / cpb))
            .map(|index| ((index * 37) % 251) as u8)
            .collect()
    }

    /// 分组推导 + "分组必须是 `cpb` 的倍数" + 打包权重长度下界。
    ///
    /// 返回 `(groups, group)`：每行多少组、每组多少元素。
    pub fn groups(
        m: usize,
        n: usize,
        scales: usize,
        weights: usize,
        cpb: usize,
    ) -> Result<(usize, usize), Error> {
        let (groups, group) = scaled_groups(m, n, scales)?;
        if !group.is_multiple_of(cpb) {
            return Err(Error::new(
                Code::Invalid,
                format!("分组 {group} 必须是 {cpb} 的倍数"),
            ));
        }
        required_weights(weights, n, m / cpb)?;
        Ok((groups, group))
    }

    /// 一行的点积：按分组推进，组内解包 `cpb` 个码查 `dequant` 表，整组共用同一个标度。
    ///
    /// 累加顺序与朴素写法逐位一致（同一字节的低位码在前）；每组 `group` 为 `cpb` 倍数，
    /// 解包只读整组码，不触碰补齐位。
    pub fn dot_row(
        packed_row: &[u8],
        x: &[f32],
        s_row: &[f32],
        per_group: usize,
        group: usize,
        cpb: usize,
        dequant: &[f32],
    ) -> f32 {
        // 位宽走配置表派生：`width = 8 / cpb`，与 [`crate::quant::width_cpb`] 同一公式，不在这里第二份除法。
        let width = crate::quant::width_cpb(cpb);
        let mask = (1u8 << width) - 1;
        let mut acc = 0.0f32;
        for ((packed, scale), xs) in packed_row
            .chunks_exact(per_group)
            .zip(s_row)
            .zip(x.chunks_exact(group))
        {
            let mut dot = 0.0f32;
            for (byte, codes) in packed.iter().zip(xs.chunks(cpb)) {
                for (slot, &code_slot) in codes.iter().enumerate() {
                    let code = usize::from((byte >> (slot * width)) & mask);
                    dot += dequant[code] * code_slot;
                }
            }
            acc += dot * scale;
        }
        acc
    }

    /// 打包格式的行遍历：分组与长度守卫通过后，每行解包点积一次写回调用方缓冲。
    ///
    /// 行切片、分组推进与格式无关；格式只给每字节码数 [`cpb`] 与反量化表 [`dequant`]。
    /// 解包位域与累加顺序和 [`dot_row`] 同一套算术，参考与内核对拍逐位一致。
    pub fn forward_rows(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
        cpb: usize,
        dequant: &[f32],
    ) -> diagnostic::Result<()> {
        let m = x.len();
        let n = out.len();
        let (groups, group) = groups(m, n, scales.len(), weights.len(), cpb)?;
        finite_scales(scales, groups)?;
        let row_bytes = m / cpb;
        let per_group = group / cpb;
        for (row, value) in out.iter_mut().enumerate() {
            let w_row = &weights[row * row_bytes..][..row_bytes];
            let s_row = &scales[row * groups..][..groups];
            *value = dot_row(w_row, x, s_row, per_group, group, cpb, dequant);
        }
        Ok(())
    }

    /// 逐元素（不打包）格式的行遍历：守卫与行切片收口到骨架，每行点积算术由闭包给出。
    ///
    /// 与 [`forward_rows`] 对称，差别只是权重不按字节打包、一行即 `m` 个元素；格式只给
    /// 逐行点积函数，不在自己的 `forward` 里再抄一遍分组/标度/权重长度守卫。
    pub fn forward_plain(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
        row_dot: impl Fn(&[u8], &[f32], &[f32], usize) -> f32,
    ) -> diagnostic::Result<()> {
        let m = x.len();
        let n = out.len();
        let (groups, group) = scaled_groups(m, n, scales.len())?;
        finite_scales(scales, groups)?;
        required_weights(weights.len(), n, m)?;
        for (row, value) in out.iter_mut().enumerate() {
            let w_row = &weights[row * m..][..m];
            let s_row = &scales[row * groups..][..groups];
            *value = row_dot(w_row, x, s_row, group);
        }
        Ok(())
    }
}

/// 权重至少要够 `n` 行、每行 `row` 个元素，返回这个下界。
///
/// 三条格式与参考实现的说明逐字相同，判据也相同（`>=` 而不是 `==`：调用方可以带多余尾部），
/// 所以它只有一份。上界用饱和乘：`n · row` 溢出时直接判为"长度不足"，而不是回绕成一个轻松
/// 满足的小数。
pub(crate) fn required_weights(weights: usize, n: usize, row: usize) -> Result<usize, Error> {
    let want = n.saturating_mul(row);
    if weights < want {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {weights} < {want}"),
        ));
    }
    Ok(want)
}

/// 标量参考：校验由骨架给出，算术由格式给出。
///
/// 骨架负责的部分与格式无关——输出非空、输入够长、权重够长、结果写进一块新缓冲；格式负责
/// 自己的解包与分组约束。于是"再加一份精度"不需要重写这些检查。
///
/// # Errors
///
/// 缓冲缺失、元素类型不符、长度不足，或格式自己的约束（分组、标度个数）不成立时返回说明。
pub fn reference<F: Format>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let weights = weights::<F>(sample)?;
    let scales = scales::<F>(sample)?;
    let (m, n) = (sample.m, sample.n);
    gauge(x.len(), m, n)?;
    run::<F>(&x[..m], weights, scales, m, n)
}

/// 标度缓冲：只有格式声明了 [`Format::SCALES`] 才读；没有标度的格式拿空切片。
fn scales<F: Format>(sample: &Sample) -> diagnostic::Result<&[f32]> {
    match F::SCALES {
        Some(_) => sample.f32("scales"),
        None => Ok(&[]),
    }
}

/// 骨架侧的尺寸判据：输出非空、输入够长。只问答不答格式——权重够不够由 [`run`] 判。
fn gauge(x: usize, m: usize, n: usize) -> diagnostic::Result<()> {
    non_empty(n)?;
    if x < m {
        return Err(Error::new(Code::Invalid, format!("输入长度 {x} < {m}")));
    }
    Ok(())
}

/// 权重下界够了之后，把结果交给格式自己的算术（分配也在这里，格式只写缓冲）。
fn run<F: Format>(
    x: &[f32],
    weights: &[F::Elem],
    scales: &[f32],
    m: usize,
    n: usize,
) -> diagnostic::Result<Vec<f32>> {
    let want = required_weights(weights.len(), n, F::row(m))?;
    let mut y = vec![0.0f32; n];
    F::forward(x, &weights[..want], scales, &mut y)?;
    Ok(y)
}

#[cfg(test)]
#[path = "../tests/gemv_tests.rs"]
mod tests;
