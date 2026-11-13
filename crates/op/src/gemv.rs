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

use crate::einsum::{Access, Bits, Expr, Formula, Operand};
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};
use std::sync::OnceLock;

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

/// 表达式里一个操作数的名字、位宽与方向。
///
/// 用 `const` 数组而不是 `Vec<Operand>`：`einsum` 的 [`Operand`] 装着 `String`，构造不了常量；
/// 这里只放编译期就能定下的三件事，装载时再转成 [`Operand`]。
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

    /// 这份格式的表达式缓存。
    ///
    /// 实现者写成 `static CACHE: Cache = Cache::new(); CACHE.get::<Self>()`：缓存按**格式**
    /// 分开，`cost` 因此只解析一次表达式。写在骨架里的泛型 `static` 做不到这件事——Rust 的
    /// `static` 在泛型作用域里只有一份，两种格式会共用同一个单元格。
    fn formula() -> Result<&'static Formula, Error>;

    /// 一行权重占多少个元素。
    fn row(m: usize) -> usize;

    /// 确定性权重：同一形状永远得到同一组取值（DESIGN §12.3 的"可复现"要求）。
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

/// 量化格式：`GRAIN` 个元素共用一条标度，每两个元素占一个字节（4 位一元素）。
///
/// `GRAIN` 是**声明粒度**（代价的派生轴除数、样本里每行的标度数），不是运行期分组：
/// 内核模板把分组当 `gsz` 实参收，所以 `forward` 仍按 `scales` 的长度推分组——把实际分组
/// 也钉进类型会让"同一份权重换一种分组"变成另一份单态化副本，而模板根本没有这个自由度。
pub struct Quantized<const GRAIN: usize>;

/// 稠密格式：每 `BLOCK` 个元素一块累加（最后一块可以短），无打包、无标度。
///
/// `BLOCK` 出现在最内层循环的 `chunks` 上，是真正进热路径的常数：单态化之后编译器看到的是
/// `chunks(32)` 而不是 `chunks(gsz)`，块内可以整体展开与向量化。
pub struct Dense<const BLOCK: usize>;

/// 对称 8 位量化格式：`GRAIN` 个元素共用一条标度，每元素占一个字节（不打包、零点 0）。
///
/// 与 [`Quantized`] 的差别是不打包（每元素 1 字节、无 nibble 解包），与 [`Dense`] 的差别是
/// 有 per-group 标度。`u8` 承载 int8 的位模式，算术时按 `i8` 解释。
pub struct SymI8<const GRAIN: usize>;

/// 与权重格式无关的输入向量：两条算子共用同一份取值。
fn input(m: usize) -> Vec<f32> {
    (0..m)
        .map(|index| ((index * 7) % 13) as f32 * 0.25 - 1.0)
        .collect()
}

/// 一份表达式缓存：构造一次，之后复用到任意形状。
///
/// 缓存**由具体格式持有**（`static CACHE: Cache` 写在 impl 的 `formula()` 里）。写成泛型函数
/// 里的 `static` 会两种格式共用一份，先跑的那一份表达式会让另一份拿到错误的轴表——这正是
/// 一次实测里"`g` 不在 `nm,m,n->n` 里"的来源。`Cache::get::<F>` 仍是每格式单态化的，
/// 没有 `HashMap` 那样的运行期索引。
///
/// 初始化失败也一起缓存——同一条常量表达式不会第二次成功。
pub struct Cache(OnceLock<Result<Formula, Error>>);

impl Cache {
    /// 空缓存（`static` 初始化用，必须是 `const fn`）。
    #[must_use]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// 取这份格式的表达式：第一次调用时解析并校验，之后直接复用。
    ///
    /// # Errors
    ///
    /// 表达式或操作数表不自洽时返回第一次得到的说明（带 `op::gemv::formula` 帧）。
    pub fn get<F: Format>(&'static self) -> Result<&'static Formula, Error> {
        self.0
            .get_or_init(|| {
                let operands = F::OPERANDS
                    .iter()
                    .map(|spec| Operand::new(spec.name, spec.bits, spec.access))
                    .collect();
                Expr::parse(F::EXPR)
                    .and_then(|expr| Formula::new(expr, operands))
                    .map_err(|error| error.with("op::gemv::formula"))
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
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
    let scales: &[f32] = match F::SCALES {
        Some(_) => sample.f32("scales")?,
        None => &[],
    };
    let m = sample.m;
    let n = sample.n;

    if n == 0 {
        return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
    }
    if x.len() < m {
        return Err(Error::new(
            Code::Invalid,
            format!("输入长度 {} < {m}", x.len()),
        ));
    }
    let want = n * F::row(m);
    if weights.len() < want {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {} < {want}", weights.len()),
        ));
    }

    let mut y = vec![0.0f32; n];
    F::forward(&x[..m], &weights[..want], scales, &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_f32::F32;
    use crate::gemv_i4::I4;

    fn at<F: Format>(m: usize, n: usize) -> Result<Cost, Error> {
        cost::<F>(&ShapeContext::new([('m', m), ('n', n)])?)
    }

    /// 两条格式的表达式必须各自与操作数配对，且派生轴只出现在量化格式里。
    #[test]
    fn the_derived_axis_belongs_to_the_quantized_format_only() {
        let quantized = ShapeContext::new([('m', 64), ('n', 4)])
            .expect("轴")
            .extents_with_group('g', 'm', 32)
            .expect("可分组");
        let axes = <I4 as Format>::formula()
            .expect("配对")
            .axes(&quantized)
            .expect("可分解");
        let reduced: Vec<char> = axes.reduced().map(|axis| axis.label()).collect();
        let grouped: Vec<char> = axes.grouped().map(|axis| axis.label()).collect();
        assert_eq!(reduced, ['m'], "只有输入维参与求和");
        assert_eq!(grouped, ['g'], "量化分组是派生轴");
        assert_eq!(axes.output_shape(), [4]);

        let dense = ShapeContext::new([('m', 64), ('n', 4)]).expect("轴");
        let axes = <F32 as Format>::formula()
            .expect("配对")
            .axes(dense.extents())
            .expect("可分解");
        assert_eq!(axes.grouped().count(), 0, "稠密格式没有派生轴");
    }

    /// 同一形状下两种格式的流量差 8 倍以上：f32 权重是 int4 的 8 倍，标度只属于量化格式。
    #[test]
    fn the_two_formats_price_the_same_shape_differently() {
        let dense = at::<F32>(64, 4).expect("可估");
        let quantized = at::<I4>(64, 4).expect("可估");
        assert_eq!(dense.read, (64 * 4 + 64) * 4, "w 与 x 都是 f32");
        assert_eq!(quantized.read, 4 * 32 + 64 * 4 + 4 * 2 * 4);
        assert!(dense.read > quantized.read, "稠密权重的读下界更高");
    }

    /// 样本的缓冲集合由格式决定：量化格式有 `scales`，稠密格式没有，`gsz` 取各自的步长。
    #[test]
    fn the_sample_follows_the_format_instead_of_a_fixed_buffer_list() {
        let dense = sample::<F32>(64, 4);
        assert!(dense.buffer("scales").is_none(), "稠密格式没有标度");
        assert_eq!(dense.scalar("gsz"), Some(32));

        let quantized = sample::<I4>(64, 4);
        assert_eq!(quantized.f32("scales").expect("scales").len(), 4 * 2);
        assert_eq!(quantized.scalar("gsz"), Some(32));
        assert_eq!(quantized, sample::<I4>(64, 4), "样本必须可复现");
    }

    /// 权重元素类型不符时报的是"这份格式期望什么"，而不是笼统的"不是 u8"。
    #[test]
    fn a_weight_buffer_of_the_wrong_element_type_names_the_expected_one() {
        let mut dense = sample::<F32>(8, 2);
        dense
            .set("w", Values::U8(vec![0; 16]))
            .expect("长度相同即可换");
        let error = reference::<F32>(&dense).expect_err("u8 不是 f32 格式的权重");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("f32"), "{error}");
    }

    /// 测试用的小粒度实例：`Quantized<4>` 与生产实例共用同一份实现。
    #[test]
    fn a_smaller_grain_is_the_same_implementation() {
        let sample = sample::<Quantized<4>>(16, 2);
        assert_eq!(sample.f32("scales").expect("scales").len(), 2 * 4);
        assert_eq!(
            at::<Quantized<4>>(16, 2).expect("可估").flops,
            2.0 * 16.0 * 2.0
        );
        assert!(at::<Quantized<8>>(16, 2).is_ok());
        assert!(
            at::<Quantized<32>>(16, 2).is_err(),
            "16 不是 32 的整数倍：声明粒度决定形状合法性"
        );
    }

    /// 派生轴的除数只能是形状粒度：写歪一个数时，代价与样本算的是两种分组。
    #[test]
    fn the_derived_divisor_must_be_the_grain() {
        assert!(derived_matches_grain(32, None), "没有派生轴时不受约束");
        assert!(derived_matches_grain(32, Some(('g', 'm', 32))));
        assert!(
            !derived_matches_grain(32, Some(('g', 'm', 8))),
            "8 元素一组的代价配 32 元素一组的标度，是两份语义"
        );

        // 两条生产格式的自证真的被求值过：这两行编译得过，说明 `CHECK` 对它们成立。
        let _: () = <I4 as Format>::CHECK;
        let _: () = <F32 as Format>::CHECK;
    }

    /// 骨架的检查对两种格式都成立：输出为空与输入过短由同一段代码拒绝。
    #[test]
    fn the_shared_checks_apply_to_every_format() {
        for error in [
            reference::<F32>(&Sample {
                m: 4,
                n: 0,
                scalars: Vec::new(),
                buffers: vec![],
            })
            .expect_err("输出维度为 0"),
            reference::<I4>(&Sample {
                m: 4,
                n: 1,
                scalars: Vec::new(),
                buffers: vec![],
            })
            .expect_err("缺缓冲"),
        ] {
            assert!(
                matches!(error.code(), Code::Invalid | Code::Missing),
                "{error}"
            );
        }
        assert!(
            <Dense<16> as Format>::EXPR.contains('m'),
            "常量在类型上可取"
        );
        assert_eq!(<Quantized<8> as Format>::GRAIN, 8);
    }
}
