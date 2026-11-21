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
//! | 权重元素类型、位宽、解包方式 | [`Format::Elem`] / [`FormulaOwner::OPERANDS`] / `forward` | 只有代码能表达，且必须进单态化 |
//! | 稠密格式的分块宽度 | [`Dense`] 的 const 参数 | 它出现在最内层循环的 `chunks` 上，是真正的热路径常数 |
//! | 量化格式的声明粒度 | [`Quantized`] 的 const 参数 | 它决定派生轴除数与样本里的标度个数（代价与样本，冷路径） |
//! | 实际分组大小 | 运行时数据（`scales` 长度 / `gsz` 实参） | 模板本身就是 `int gsz` 的运行时参数，钉进类型会让训练侧按 8 元素分组的调用编译不过 |
//!
//! 最后一行是刻意的：把"内核已经做成运行时参数"的东西改成类型参数，换来的是每次不同分组都要
//! 一份单态化副本与一次调用点上的匹配，而不是省下一条分支。

use crate::model::{AxisRole, AxisSpec};

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
/// 带标度格式的操作数表：权重位宽取配置，其余三项恒为 32 位浮点。
///
/// 展开在 [`FormulaOwner`] 的实现里；[`Packed`] 与 [`SymI8`] 只差这一个数。
macro_rules! scaled_operands {
    ($bits:expr) => {
        /// 三个输入都是读、只有输出是写；权重位宽取配置，其余为 32 位浮点。
        const OPERANDS: &'static [$crate::einsum::OperandSpec] =
            &$crate::gemv::scaled_operands($bits);
    };
}

/// 带标度格式的量化几何：粒度、派生轴、标度取值与步长。
///
/// 展开在 [`Format`] 与 [`Plain`] 的实现里。四条都从粒度算出来，调用方只给粒度：抄两遍就会让
/// `DERIVED` 的除数与 `GRAIN` 各自漂移（判据见 [`derived_matches_grain`]）。
macro_rules! scaled_grain {
    ($grain:expr) => {
        const GRAIN: usize = $grain;

        const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', $grain));

        const SCALES: Option<fn(usize) -> f32> = Some($crate::gemv::scale_of);

        const STEP: usize = $grain;
    };
}

mod format {
    //! 权重格式：把「一行权重怎么摆」抽象成 trait，解包与分组约束都由格式自己负责。
    //!
    //! [`Element`] 是标量通道（`f32` 与 `u8` 两种解释），[`PackedRow`] 是打包布局（i2/i4/i8），
    //! [`Plain`] 是不打包的稠密布局（`f32` 与 `i8`）。`Format` 只组装，不做算术。
    use crate::einsum::{Bits, FormulaCache, OperandSpec};
    use crate::formula::FormulaOwner;
    use crate::model::{Buffer, Values};
    use crate::quant::Row;

    use super::forward::{forward_dense, forward_rows, scaled_dot};
    use super::packed;
    use super::shape::non_empty;

    /// 权重元素：进样本、读缓冲、拼 `_Generic` 三处都按它分派。
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

    /// 带分组标度的操作数表：权重按 `bits` 读，输入与标度都是 32 位浮点，输出写 32 位。
    ///
    /// 打包格式（[`Packed`]）与不打包的对称 int8 格式（[`SymI8`]）只差权重的位宽，其余三项逐字
    /// 相同；写成一条 [`const fn`] 让两份实现共用同一张表，位宽成为唯一参数。
    pub const fn scaled_operands(bits: Bits) -> [OperandSpec; 4] {
        [
            OperandSpec::read("w", bits),
            OperandSpec::read("x", Bits::ThirtyTwo),
            OperandSpec::read("scales", Bits::ThirtyTwo),
            OperandSpec::write("y", Bits::ThirtyTwo),
        ]
    }

    /// 权重格式：一条 GEMV 里随精度变化的那部分。
    ///
    /// 实现者只写"这份格式怎么表达、怎么解包"：表达式在 [`FormulaOwner`]，行切片与守卫由本 trait
    /// 声明，骨架（[`cost`] / [`sample`] / [`reference()`]）由本模块提供。所有关联项都是编译期
    /// 常量：没有一件需要运行期再判断一次。
    pub trait Format: FormulaOwner {
        /// 权重在样本里的元素类型。
        type Elem: Element;

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
        /// 这是一个**编译期**判据，不是运行期检查：注解不负责让它被求值（注解只生成 `MODEL`，
        /// 契约层不认识内核），强制求值的 `const _: () = <名字 as Format>::CHECK;` 写在本模块的
        /// 父级 [`mod@crate::gemv`] 里，每个格式别名一条，写错粒度或派生轴除数就编译不过。
        /// 规则放在 trait 里而不是宏里，是因为它属于格式几何，不该由生成器各写一份。
        const CHECK: () = {
            assert!(Self::GRAIN > 0, "形状粒度必须大于 0");
            assert!(
                derived_matches_grain(Self::GRAIN, Self::DERIVED),
                "派生轴的除数必须等于形状粒度：代价里的分组与样本里的分组必须同源"
            );
        };

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

        fn cache() -> &'static FormulaCache {
            crate::formula_cell!()
        }
    }

    /// 量化格式：`GRAIN` 个元素共用一条标度，每两个元素占一个字节（4 位一元素）。
    ///
    /// 底层复用 [`Packed`] 的 4 位配置；`GRAIN` 是**声明粒度**（代价的派生轴除数、样本里每行的
    /// 标度数），不是运行期分组：内核模板把分组当 `gsz` 实参收，所以 `forward` 仍按 `scales` 的
    /// 长度推分组——把实际分组也钉进类型会让"同一份权重换一种分组"变成另一份单态化副本。
    pub type Quantized<const GRAIN: usize> = Packed<QuantI4, GRAIN>;

    /// 稠密逐元素格式的配置标记：每 `BLOCK` 个元素一块累加（最后一块可以短），无打包、无标度。
    ///
    /// `BLOCK` 出现在最内层循环的 `chunks` 上，是真正进热路径的常数：单态化之后编译器看到的是
    /// `chunks(32)` 而不是 `chunks(gsz)`，块内可以整体展开与向量化。`impl Plain` 在 `gemv_f32`。
    pub struct Dense<const BLOCK: usize>;

    /// 逐元素对称 int8 格式的配置标记：不打包（每元素 1 字节、无 nibble 解包），有 per-group
    /// 标度。`u8` 承载 int8 的位模式，算术时按 `i8` 解释。`impl Plain` 在 `gemv_i8`。
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
        /// 位宽串档）。与 [`FormulaCache`] 文档里"缓存留在各骨架
        /// 类型上"的口径一致。
        fn cache() -> &'static FormulaCache;
    }

    /// 配置驱动的打包权重格式：`GRAIN` 个元素共用一条标度，打包密度由 [`PackedRow::ROW`] 给出。
    ///
    /// 类型参数 `P` 承载配置（编译期单态化），`G` 承载声明粒度；表达式、操作数表、派生轴除数、
    /// 分组倍数守卫与行切片全部读配置，格式侧只决定"用哪一行配置"。`P` 只在关联常量上被读出，
    /// 不占运行期字段，故用 [`core::marker::PhantomData`] 标记这个编译期绑定。
    pub struct Packed<P: PackedRow, const G: usize> {
        _config: core::marker::PhantomData<P>,
    }

    impl<P: PackedRow, const G: usize> FormulaOwner for Packed<P, G> {
        /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 `G` 个元素切出来的派生轴。
        const EXPR: &'static str = "nm,m,ng,n->n";

        scaled_operands!(P::ROW.bits);

        const FRAME: &'static str = "op::gemv::formula";

        fn cache() -> &'static FormulaCache {
            // 缓存单元格由标记类型各自持有，避免泛型 static 被合并成跨配置共享的一份。
            P::cache()
        }
    }

    impl<P: PackedRow, const G: usize> Format for Packed<P, G> {
        type Elem = u8;

        scaled_grain!(G);

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

    /// 逐元素（不打包）GEMV 的静态配置：元素类型、量化几何、样本与元素反量化。
    ///
    /// `i8`（`u8` 承载 int8、带分组标度）与 `f32`（稠密、无标度）只在元素类型、标度有无与
    /// 反量化上不同；表达式在 [`FormulaOwner`]，[`PlainFmt`] 的一份泛型 [`Format`] impl 负责
    /// 行切片、守卫与逐行点积遍历。新增一档逐元素精度加一条 `impl Plain` 即可。
    pub trait Plain: FormulaOwner {
        /// 权重元素类型。
        type Elem: Element;

        /// 形状粒度：`m` 必须是它的正整数倍；`1` 表示任意宽度。
        const GRAIN: usize;

        /// 派生轴 `(派生轴, 基准轴, 除数)`；无派生轴为 `None`。
        const DERIVED: Option<(char, char, usize)>;

        /// 标度取值；`None` 表示没有标度缓冲。
        const SCALES: Option<fn(usize) -> f32>;

        /// 模板每轮推进元素数（样本 `gsz`；稠密同时是累加分块）。
        const STEP: usize;

        /// 确定性权重。
        fn weights(m: usize, n: usize) -> Vec<Self::Elem>;

        /// 元素反量化到 f32：int8 按 `i8` 解释，f32 恒等。
        fn to_f32(elem: Self::Elem) -> f32;
    }

    /// 配置驱动的逐元素权重格式：把一条 [`Plain`] 配置挂进 [`Format`] 骨架。
    ///
    /// 与 [`Packed`] 对称：打包格式按 [`PackedRow`] 收敛，逐元素格式按 [`Plain`] 收敛，两边
    /// 都只写"格式事实"，缓存 / 行切片 / 守卫 / 逐行点积遍历各有一份。
    pub struct PlainFmt<C: Plain> {
        _config: core::marker::PhantomData<C>,
    }

    impl<C: Plain> FormulaOwner for PlainFmt<C> {
        const EXPR: &'static str = C::EXPR;

        const OPERANDS: &'static [OperandSpec] = C::OPERANDS;

        const FRAME: &'static str = C::FRAME;

        fn cache() -> &'static FormulaCache {
            // 缓存单元格由配置各自持有，避免泛型 static 被合并成跨配置共享的一份。
            C::cache()
        }
    }

    impl<C: Plain> Format for PlainFmt<C> {
        type Elem = C::Elem;

        const GRAIN: usize = C::GRAIN;

        const DERIVED: Option<(char, char, usize)> = C::DERIVED;

        const SCALES: Option<fn(usize) -> f32> = C::SCALES;

        const STEP: usize = C::STEP;

        fn row(m: usize) -> usize {
            // 逐元素不打包：一行权重就是 m 个元素。
            m
        }

        fn weights(m: usize, n: usize) -> Vec<Self::Elem> {
            C::weights(m, n)
        }

        fn forward(
            x: &[f32],
            weights: &[Self::Elem],
            scales: &[f32],
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            non_empty(out.len())?;
            if C::SCALES.is_some() {
                forward_rows(x, weights, scales, out, 1, scaled_dot::<C>)
            } else {
                forward_dense::<C>(x, weights, scales, out)
            }
        }
    }
}
mod forward {
    //! 前向内核：按行（解码）与按批（prefill）两条路径，点积本身分「带标度」与「稠密」两种。
    //!
    //! 两条路径都把「权重行 × 输入向量」叫成一行计算，差别只在输入是单向量还是批；点积的
    //! lane 数（[`LANES`]）按格式的元素宽度挑，保证每轮取满一条向量寄存器。
    use diagnostic::{Code, Error};

    use super::format::Plain;
    use super::shape::{finite_scales, required_weights, scaled_groups};

    /// 带标度行骨架：分组、粒度、权重与标度守卫连同行切片统一收口，逐行点积由 `row_dot` 给出。
    ///
    /// `cpb` 是「一个权重单位里有多少个元素」：打包格式给每字节码数（`2` / `4`），不打包的逐元素
    /// 格式给 `1`——那时行宽就是 `m` 个元素，粒度倍数守卫恒真。于是打包与不打包只差这一处行宽
    /// 折算和一份逐行算术，守卫、切片与遍历是同一份代码，按元素类型单态化。
    pub(super) fn forward_rows<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        out: &mut [f32],
        cpb: usize,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) -> diagnostic::Result<()> {
        let ctx = scaled_row_context(x.len(), out.len(), scales, weights.len(), cpb)?;
        forward_rows_scaled(x, weights, scales, out, ctx, row_dot);
        Ok(())
    }

    /// 逐行遍历一遍，算术由 `row_dot` 给：守卫已经过了（`ctx` 就是证据），这里只剩切片与调用。
    ///
    /// 单独一支是因为叶有时要把好几行算在一起（`gemv::simd` 的四列累加器），那时 `row_dot`
    /// 这一行一个值的接口不合适；叶拿着同一个 `ctx` 自己走遍历，且**不必重跑守卫**——
    /// [`scaled_row_context`] 里的标度有限性判定是 `O(n·groups)`，在 4096² 上重复一次就是内核
    /// 算术的两成。
    pub(crate) fn forward_rows_scaled<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        out: &mut [f32],
        ctx: ScaledRow,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) {
        for (index, value) in out.iter_mut().enumerate() {
            let w_row = &weights[index * ctx.row..][..ctx.row];
            let s_row = &scales[index * ctx.groups..][..ctx.groups];
            *value = row_dot(w_row, x, s_row, ctx.group);
        }
    }

    /// 行批骨架：输出行在外、输入行在里，一行权重只读一遍。
    ///
    /// 与 [`forward_rows`] 是同一个遍历的两种嵌套：那里输入是向量（一层），这里输入是
    /// `rows × m` 的行批（两层）。输入行放内层就是"权重复用"的全部内容——权重行只从内存读一遍、
    /// 解包一遍，而不是按行数重复；逐行调用单行入口会把同一份权重读 `rows` 遍。`cpb` 见
    /// [`forward_rows`]：打包与不打包共用这一副嵌套。
    fn forward_batch_rows<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        geo: Geometry,
        out: &mut [f32],
        cpb: usize,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) -> diagnostic::Result<()> {
        batch_geometry(x.len(), out.len(), geo)?;
        let ctx = scaled_row_context(geo.m, geo.n, scales, weights.len(), cpb)?;
        forward_batch_rows_scaled(x, weights, scales, geo, out, ctx, row_dot);
        Ok(())
    }

    /// 行批的逐行遍历，算术由 `row_dot` 给：与 [`forward_rows_scaled`] 同一个"守卫已过"约定。
    pub(crate) fn forward_batch_rows_scaled<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        geo: Geometry,
        out: &mut [f32],
        ctx: ScaledRow,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) {
        for j in 0..geo.n {
            let w_row = &weights[j * ctx.row..][..ctx.row];
            let s_row = &scales[j * ctx.groups..][..ctx.groups];
            for r in 0..geo.rows {
                out[r * geo.n + j] = row_dot(w_row, &x[r * geo.m..][..geo.m], s_row, ctx.group);
            }
        }
    }

    /// 带标度格式的行守卫：分组、粒度倍数、权重长度与标度有限四项一次判齐，返回 `(groups, group)`。
    ///
    /// 单行与行批两条入口共用这一处：批入口只是先判几何。先判结构（分组、倍数、长度）再判取值
    /// （标度有限），于是报错指向第一处真正不对的地方；`cpb = 1` 时不打包格式的倍数守卫恒真。
    pub(super) fn scaled_row_context(
        m: usize,
        n: usize,
        scales: &[f32],
        weights: usize,
        cpb: usize,
    ) -> diagnostic::Result<ScaledRow> {
        let (groups, group) = scaled_groups(m, n, scales.len())?;
        if !group.is_multiple_of(cpb) {
            return Err(Error::new(
                Code::Invalid,
                format!("分组 {group} 必须是 {cpb} 的倍数"),
            ));
        }
        required_weights(weights, n, m / cpb)?;
        finite_scales(scales, groups)?;
        Ok(ScaledRow {
            groups,
            group,
            row: m / cpb,
        })
    }

    /// 已过守卫的带标度行上下文：每行多少组、每组多少元素、每行多少个权重单位。
    ///
    /// 这是"守卫已经过了"的证据而不是三个裸 `usize`——裸数传进叶里，"它们一定对"就成了一句注释；
    /// 收成一个值之后，重跑守卫（尤其 `O(n·groups)` 的标度判定）就不再有理由，叶只需要读权重与
    /// 标度。构造只有 [`scaled_row_context`] 一条路。
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ScaledRow {
        /// 每行多少组。
        pub groups: usize,
        /// 每组多少元素（必须是 `cpb` 的倍数）。
        pub group: usize,
        /// 每行多少个权重单位（打包格式是 `m / cpb` 个字节）。
        pub row: usize,
    }

    /// 行批逐元素入口：按配置选择带标度 / 无标度两条骨架，与 [`Format::forward`] 同一分叉。
    pub fn forward_plain_batch<C: Plain>(
        x: &[f32],
        weights: &[C::Elem],
        scales: &[f32],
        geo: Geometry,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        if C::SCALES.is_some() {
            forward_batch_rows(x, weights, scales, geo, out, 1, scaled_dot::<C>)
        } else {
            if !scales.is_empty() {
                return Err(Error::new(
                    Code::Invalid,
                    format!("稠密格式没有标度，收到 {} 个", scales.len()),
                ));
            }
            forward_batch_rows(x, weights, scales, geo, out, 1, |w, x, _, _| {
                dense_dot::<C>(w, x)
            })
        }
    }

    /// 行批几何：`rows` 行输入、输入行宽 `m`、输出行宽 `n`。
    ///
    /// 三个数必须同时自洽（`x.len() == rows·m`、`out.len() == rows·n`），且都只对行批有意义；
    /// 分开传三个同型 `usize` 既容易在调用点换位，也让每个入口都多出两个参数。收成一个值后，
    /// 几何的校只在入口守卫一处，量化与逐元素两条骨架传同一个类型。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Geometry {
        /// 输入行数。
        pub rows: usize,
        /// 输入行宽（权重行宽）。
        pub m: usize,
        /// 输出行宽。
        pub n: usize,
    }

    impl Geometry {
        /// 由三个数直接构造；合法性留给入口守卫，构造本身不报错。
        pub const fn new(rows: usize, m: usize, n: usize) -> Self {
            Self { rows, m, n }
        }
    }

    /// 行批几何：行数非零，输入与输出的长度分别对上 `rows × m` 与 `rows × n`。
    ///
    /// 三处一起判而不分判：批次调用时三个数必须同时自洽，单独对上一个没有意义。
    pub(super) fn batch_geometry(x: usize, out: usize, geo: Geometry) -> diagnostic::Result<()> {
        if geo.rows == 0 {
            return Err(Error::new(Code::Invalid, "行批至少一行"));
        }
        let (want_x, want_out) = (
            geo.rows.saturating_mul(geo.m),
            geo.rows.saturating_mul(geo.n),
        );
        if x != want_x || out != want_out {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "行批尺寸不符：x={x}，out={out}，期望 {}×{}={want_x}、{}×{}={want_out}",
                    geo.rows, geo.m, geo.rows, geo.n
                ),
            ));
        }
        Ok(())
    }

    /// 无标度逐元素格式的行遍历：换错算子（给了标度）必须报错，而不是静默忽略。
    pub(super) fn forward_dense<C: Plain>(
        x: &[f32],
        weights: &[C::Elem],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        if !scales.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                format!("稠密格式没有标度，收到 {} 个", scales.len()),
            ));
        }
        let (m, n) = (x.len(), out.len());
        required_weights(weights.len(), n, m)?;
        for (row, value) in out.iter_mut().enumerate() {
            let w_row = &weights[row * m..][..m];
            *value = dense_dot::<C>(w_row, x);
        }
        Ok(())
    }

    /// 一行的带标度点积：按组推进，每组解包点积一次乘标度（[`dot`] 承担单元素反量化）。
    pub(super) fn scaled_dot<C: Plain>(
        w_row: &[C::Elem],
        x: &[f32],
        s_row: &[f32],
        group: usize,
    ) -> f32 {
        let mut acc = 0.0f32;
        for ((w_chunk, scale), x_chunk) in w_row.chunks(group).zip(s_row).zip(x.chunks(group)) {
            acc += dot::<C>(w_chunk, x_chunk) * scale;
        }
        acc
    }

    /// 累加链数：16 条 = 4 条 NEON 寄存器（x86 的 AVX2 上是 2 条 `ymm`）。
    ///
    /// 这个数不是凑出来的：`crates/benchmarks/src/kernel.rs` 的 `gemv_f32/decode_1x4096x4096`
    /// （1×4096×4096 decode 步，权重一遍 64 MiB）换链数实测 4 条链 3.8 ms、8 条 1.3 ms、12 条
    /// 3.7 ms、16 条 0.89 ms、24 条 2.4 ms、32 条 0.91 ms——
    /// 只有 2 的幂的条数能被展开成整条向量指令，16 条在多一个寄存器轮转之后就不再获益。
    const LANES: usize = 16;

    /// 一行的无标度点积：`LANES` 条独立累加链，内层编译成整条向量乘加。
    ///
    /// 与 [`dot`] 的 `sum` 差在归约的形状：`sum` 是严格的左折叠，f32 没有 fast-math，LLVM 只能一次
    /// 加一个——上面那组实测里它就是 2.85 ms 那一档，decode 的带宽全被它挡住。`LANES` 条链彼此独立，
    /// 同样的权重读一遍算出整条向量的宽度（实测 3.2×）；链内顺序与左折叠不同，只带来末位舍入差，
    /// 确定性样本的取值落在 2 的幂网格上，两者逐位相同。
    fn dense_dot<C: Plain>(w_row: &[C::Elem], x: &[f32]) -> f32 {
        let (w_chunks, w_tail) = w_row.as_chunks::<LANES>();
        let (x_chunks, x_tail) = x.as_chunks::<LANES>();
        let mut lanes = [0.0f32; LANES];
        for (w, x) in w_chunks.iter().zip(x_chunks) {
            let mut lane = 0;
            while lane < LANES {
                lanes[lane] += C::to_f32(w[lane]) * x[lane];
                lane += 1;
            }
        }
        let mut acc = lanes.iter().sum::<f32>();
        for (w, x) in w_tail.iter().zip(x_tail) {
            acc += C::to_f32(*w) * *x;
        }
        acc
    }

    /// 一段逐元素点积：`sum(to_f32(w) · x)`，按配置的单元素反量化。
    fn dot<C: Plain>(w: &[C::Elem], x: &[f32]) -> f32 {
        w.iter()
            .zip(x)
            .map(|(elem, xv)| C::to_f32(*elem) * xv)
            .sum()
    }
}

pub mod packed {
    //! 打包权重（每字节 `cpb` 个码）的取样与解包点积助手。
    //!
    //! int2（4 码/字节）与 int4（2 码/字节）只在码宽、查表与粒度倍数上不同；打包位域的字节布局
    //! 与按行点积收口到这里，避免两条打包格式各抄一遍。行遍历、守卫与行切片在骨架
    //! `gemv::forward_rows` 里只有一份——本模块只把“行宽按 `cpb` 折算”接上去。
    //! int8 不打包，不走本模块。
    /// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节，同一形状取值可复现。
    pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
        (0..n * (m / cpb))
            .map(|index| ((index * 37) % 251) as u8)
            .collect()
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

    /// 把骨架的逐行点积接口接到 [`dot_row`]：骨架只给分组元素数 `group`，解包点积还要每组字节数
    /// `group / cpb`、每字节码数与反量化表。
    fn row_dot(cpb: usize, dequant: &[f32]) -> impl Fn(&[u8], &[f32], &[f32], usize) -> f32 + '_ {
        move |w_row, x, s_row, group| dot_row(w_row, x, s_row, group / cpb, group, cpb, dequant)
    }

    /// 打包格式的单行入口：行宽按 `cpb` 折算，逐行点积走 `row_dot`。
    ///
    /// 守卫、行切片与遍历都在骨架 `gemv::forward_rows` 里；这里只接上“一行 `m / cpb` 个字节”
    /// 与解包算术，解包位域与累加顺序和 [`dot_row`] 同一套，参考与内核对拍逐位一致。
    pub fn forward_rows(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
        cpb: usize,
        dequant: &[f32],
    ) -> diagnostic::Result<()> {
        let ctx =
            super::forward::scaled_row_context(x.len(), out.len(), scales, weights.len(), cpb)?;
        if let Some(affine) = super::simd::admit(cpb, out.len(), ctx, dequant) {
            super::simd::forward(x, weights, scales, affine, out);
            return Ok(());
        }
        super::forward::forward_rows_scaled(x, weights, scales, out, ctx, row_dot(cpb, dequant));
        Ok(())
    }

    /// 打包格式的行批入口：与 [`forward_rows`] 同一套算术与同一份 `row_dot`，只换遍历嵌套。
    ///
    /// 输出行 `j` 在外（权重行 `j` 由此只解包一次），输入行 `r` 在内——量化权重在 prefill 下不会
    /// 被按行数重复读，而每个元素仍是同一累加顺序，批结果与逐行调用逐位相同。
    pub fn forward_batch(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        geo: super::forward::Geometry,
        out: &mut [f32],
        cpb: usize,
        dequant: &[f32],
    ) -> diagnostic::Result<()> {
        super::forward::batch_geometry(x.len(), out.len(), geo)?;
        let ctx = super::forward::scaled_row_context(geo.m, geo.n, scales, weights.len(), cpb)?;
        if let Some(affine) = super::simd::admit(cpb, geo.n, ctx, dequant) {
            let rows: Vec<&[f32]> = x.chunks(geo.m).collect();
            super::simd::forward_batch(&rows, weights, scales, affine, out);
            return Ok(());
        }
        super::forward::forward_batch_rows_scaled(
            x,
            weights,
            scales,
            geo,
            out,
            ctx,
            row_dot(cpb, dequant),
        );
        Ok(())
    }
}

mod shape {
    //! 形状上下文、确定性样本与标量参考：`#[annotation::model]` 要的三件产物。
    //!
    //! 参考实现只做「校验 + 交给格式」，与格式无关的检查（输出非空、输入/权重够长、标度有限）
    //! 都集中在这里，于是新增一份精度不需要重写它们。
    use diagnostic::{Code, Error};

    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};

    use super::format::{Element, Format};

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

    /// 标量代价：完全由 [`FormulaOwner::EXPR`] 与操作数位宽推出，本函数不含任何流量常数。
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
    pub(crate) fn scaled_groups(
        m: usize,
        n: usize,
        scales: usize,
    ) -> Result<(usize, usize), Error> {
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
}
mod simd {
    //! 解包点积的向量通道：`n·m` 点积、4 列累加器、4 行权重与一份 `x` 窗口在 L1 里轮转。
    //!
    //! 标量叶（[`super::packed::dot_row`]）每读一个码要做三件事：查反量化表、乘 `x`、加进**同一条**
    //! 串行累加链。后两件是延迟问题——一行的和必须按顺序进同一个 `f32`，后一项要等前一项舍入，
    //! 四个码才凑得满一次流水线；第一件是核对问题——查表把四个码的并行度换成了四次访存。本模块
    //! 把三件事一起换掉：
    //!
    //! * **查表换成整数码的直接换算。** 打包格式的反量化表恒为 `table[code] = code - zero`
    //!   （[`crate::quant::Row`] 的 `I2` 是 `{-2, -1, 0, 1}`、`I4` 是 `{-8..=7}`；[`affine_zero`]
    //!   逐条核对整张表，不是假设），于是每个码只剩"移位、掩码、换算成 `f32`、减零点"四步纯算术，
    //!   四次访存变成零次。零点按车道减（一条 `fsub` 摊在 `LANES` 个码上）；也可以按组修正
    //!   （`Σ (code − zero)·x = Σ code·x − zero·Σ x`，右半每组一次），但那样每组多一条标量链、
    //!   还要预先备好每组的 `Σ x`，实测不更快，所以码值就地减零点。
    //! * **串行累加换成一条向量链。** 一组的码宽刚好装满 `LANES = group / cpb` 个车道——`group = 32`
    //!   时 4 位码是 16 车道、2 位码是 8 车道；一条 `fmla` 同时算 `LANES` 个码，归约只在组尾做一次。
    //! * **4 列共享一次 `x` 载入。** 解码的输出宽度是 `n`（生产形状 4096），四行的权重落在同一批
    //!   16 字节窗口上，一份 `x` 供四条链用——这正是 `n·m` 里那个 `n` 的复用点。
    //! * **`x` 在入口过一遍。** 元素重排成"槽位优先"（`x[slot·row + byte]`），内层于是只有连续载入，
    //!   没有步长，也没有除槽位之外的索引算术。
    //!
    //! 标度留在原位：每组一条标度、每组一乘。这不是保守——把标度提前乘进 `x` 只有"整行共用一个
    //! 标度"时才成立，而量化权重的标度是逐行逐组的，提前乘就意味着每行重排一遍 `m` 个元素，
    //! `O(n·m)` 的入口开销会和内核本身一样大。所以入口只做与行无关的重排，标度乘法留在组尾。
    //!
    //! 同值性：`Σ(code·x)·s` 与 `Σ code·(x·s)` 数学上相等，这里取前者。舍入关联与标量叶不同
    //! （融合乘加还少一次舍入），所以本通道的判据是**"与标量叶在舍入量级内一致、且批与逐行逐位
    //! 一致"**；`packed_test_support` 里"内核与参考逐位相同"仍然成立，因为 `reference` 走的就是
    //! 同一个 `Format::forward`。取值可精确表示时（整数码 × 整数 `x` × 2 的幂标度）两条通道逐位
    //! 相同，`exactly_representable_values_are_bit_identical` 钉住的就是这一条。
    //!
    //! 落地：`std::simd` 是架构无关的源码，aarch64 上落成 NEON（`fmla` / `ucvtf` / `ldr q`），
    //! x86_64 上落成 SSE2 基线。两处编译的是同一份算术，于是两条验证链验的是同一段代码；整条通道
    //! 不用 `unsafe`（`unsafe_code = "deny"` 不动）。
    //!
    //! 实测（`crates/benchmarks/src/kernel.rs` 的 `gemv_i4` 组，生产形状 `1×4096×4096`、权重一遍
    //! 9.8 MB）：标量叶 2.73 ms / 3.58 GiB/s，本通道 1.87 ms / 5.21 GiB/s，**1.46×**。剩下的时间
    //! 几乎全在换算上——每 16 个码要 4 条 `ucvtf` 加 4 条拓宽，约 0.6 条指令/码，乘加只占
    //! 0.06 条；按这个账内核已跑到该形态约 85% 的吞吐上限。列数加密不再有收益：`COLS = 8` 因
    //! 16 条累加链溢出到栈帧，实测慢 47%；`COLS = 2` 落在噪声内。真要再上一个量级得靠 `sdot`
    //! 的整数点积（8 位码 × 8 位 `x`，一条指令 16 个乘加），那需要 `unsafe` 内建函数，与
    //! `unsafe_code = "deny"` 冲突，故本通道止于可移植的这一档。

    use std::simd::prelude::{SimdFloat, SimdUint};
    use std::simd::{Simd, StdFloat};

    use super::ScaledRow;

    /// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 载入摊到 4 行。
    ///
    /// 4 行权重按 16 字节窗口轮转：权重窗口 `4 × 16 B = 64 B`、一行的 `x` 窗口 `16 码 × 4 B = 64 B`
    /// 都远在 L1 之内。列数再加密不再有收益——`COLS = 8` 因 16 条累加链溢出到栈帧，实测慢 47%。
    const COLS: usize = 4;

    /// 反量化表的仿射零：`table[code] == code - zero` 时返回 `zero`，否则 `None`（走标量叶）。
    ///
    /// 逐条核对整张表而不是拿首项猜：`zero = -table[0]` 对任何表都算得出，但只有整张表都落在这条
    /// 仿射关系上，"查表"才等价于"码减零点"。表是格式常量（[`crate::quant::Row`]），核对一遍让
    /// 通道的选择依据落在表的内容上，而不是"打包格式一定是对称表"这句话上。
    pub(crate) fn affine_zero(dequant: &[f32], codes: usize) -> Option<f32> {
        if dequant.len() != codes {
            return None;
        }
        let zero = -dequant.first()?;
        let affine = dequant
            .iter()
            .enumerate()
            .all(|(code, &value)| value == code as f32 - zero);
        affine.then_some(zero)
    }

    /// 形状够不够走本通道：车道数落在 8 / 16 两档，且一组的字节数被它整除。
    ///
    /// 车道数是 `group / cpb`（一组里每个槽位多少码）；一组的每槽字节数是 `per_group = group / cpb`，
    /// 与车道数同一个数——所以"一组恰好是整数个向量"恒成立，组尾才只需要一次归约。恒等之外的形状
    /// （一组凑不满一条向量，或凑出来不是 8 / 16 车道）回退标量叶。
    fn fits(cpb: usize, lanes: usize) -> bool {
        matches!((cpb, lanes), (2 | 4, 8 | 16))
    }

    /// 通过校验之后的通道参数：形状（[`ScaledRow`]）、码宽、车道数、反量化表的仿射零点。
    ///
    /// 把"校验过了"做成一个**值**，而不是三个平行的 `usize` 加一个 `bool` 返回值：拿到了 `Affine`
    /// 就说明表是仿射的、形状凑得满整条向量、输出宽度够分 [`COLS`] 列，内核于是不必再判一次，
    /// 也不可能拿到一组互相矛盾的参数。
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Affine {
        ctx: ScaledRow,
        cpb: usize,
        lanes: usize,
        zero: f32,
    }

    /// 校验并构造 [`Affine`]：`n` 是输出宽度（权重行数），`ctx` 是入口已经量好的行形状。
    ///
    /// 三件事都在这里一次定下来——反量化表逐条核对是否 `table[code] == code - zero`
    /// （[`affine_zero`]）、一组的字节数是否落在车道数的整档上（[`fits`]）、输出宽度是否够一整组
    /// （[`COLS`]）。任一条不成立就返回 `None`，调用方回退标量叶：本通道是加速，不是唯一实现。
    pub(crate) fn admit(cpb: usize, n: usize, ctx: ScaledRow, dequant: &[f32]) -> Option<Affine> {
        let zero = affine_zero(dequant, codes_of(cpb))?;
        let lanes = ctx.group / cpb;
        (n >= COLS && fits(cpb, lanes)).then_some(Affine {
            ctx,
            cpb,
            lanes,
            zero,
        })
    }

    /// 单行入口：`out` 是 `n` 个输出。
    pub(crate) fn forward(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        affine: Affine,
        out: &mut [f32],
    ) {
        let input = [layout(x, affine)];
        run(
            &Plan {
                weights,
                inputs: &input,
                affine,
                scales,
            },
            out,
        );
    }

    /// 行批入口：与 [`forward`] 同一段算术、同一份列块遍历，只是一次带上所有输入行。
    ///
    /// 遍历嵌套与标量叶一致：列块在外、输入行在内——同一批权重窗口供每个输入行用一遍，权重只读
    /// 一遍；而逐列组的 `dot` 与输入行走在哪一层无关，于是批结果与逐行调用逐位相同。
    pub(crate) fn forward_batch(
        rows: &[&[f32]],
        weights: &[u8],
        scales: &[f32],
        affine: Affine,
        out: &mut [f32],
    ) {
        let input: Vec<Vec<f32>> = rows.iter().map(|row| layout(row, affine)).collect();
        run(
            &Plan {
                weights,
                inputs: &input,
                affine,
                scales,
            },
            out,
        );
    }

    /// 码表长度：`1 << width`，`width = 8 / cpb`（与 [`crate::quant::width_cpb`] 同一公式）。
    fn codes_of(cpb: usize) -> usize {
        1 << crate::quant::width_cpb(cpb)
    }

    /// 内核的全部输入：权重、排好版的输入行、通道参数、标度。
    struct Plan<'a> {
        weights: &'a [u8],
        inputs: &'a [Vec<f32>],
        affine: Affine,
        scales: &'a [f32],
    }

    /// 按 `cpb` 与车道数单态化：槽位循环与组尾归约的形态都由这两个常量决定，编译期摊平。
    fn run(plan: &Plan<'_>, out: &mut [f32]) {
        match (plan.affine.cpb, plan.affine.lanes) {
            (2, 8) => columns::<2, 8>(plan, out),
            (2, _) => columns::<2, 16>(plan, out),
            (_, 8) => columns::<4, 8>(plan, out),
            _ => columns::<4, 16>(plan, out),
        }
    }

    /// 列遍历：整块走 `R = COLS`，不足一块的尾部逐列走 `R = 1`（同一个 `dot`、同一副累加器
    /// 形态），所以 `n` 不必是 `COLS` 的倍数。
    fn columns<const CPB: usize, const LANES: usize>(plan: &Plan<'_>, out: &mut [f32]) {
        let n = out.len() / plan.inputs.len();
        let mut j = 0;
        while j + COLS <= n {
            strip::<CPB, LANES, COLS>(plan, out, n, j);
            j += COLS;
        }
        if j < n {
            strip::<CPB, LANES, 1>(plan, out, n, j);
        }
    }

    /// 一列块：`R` 行权重与同一段 `x` 窗口在 L1 里轮转；权重与标度切片与行无关，只在块首取一次。
    fn strip<const CPB: usize, const LANES: usize, const R: usize>(
        plan: &Plan<'_>,
        out: &mut [f32],
        n: usize,
        j: usize,
    ) {
        let (row, groups, group) = (
            plan.affine.ctx.row,
            plan.affine.ctx.groups,
            plan.affine.ctx.group,
        );
        let (w, s) = (
            &plan.weights[j * row..][..R * row],
            &plan.scales[j * groups..][..R * groups],
        );
        for (r, input) in plan.inputs.iter().enumerate() {
            let dest = &mut out[r * n + j..][..R];
            dest.copy_from_slice(&dot::<CPB, LANES, R>(
                w,
                row,
                s,
                input,
                group,
                plan.affine.zero,
            ));
        }
    }

    /// `R` 行的点积：一组一条 `LANES` 车道链，四行同时推进，**每行只在行尾归约一次**。
    ///
    /// 归约是水平求和——串且长（`LANES` 车道要 `log2` 步一路加到底），一组归约一次就意味着每组
    /// 一次 ~20 周期的等待，`groups` 个组串起来能把内核按在 L1 里空等。这里用线性关系把它摊开：
    /// `Σ_g reduce(acc_g)·s_g = reduce(Σ_g acc_g·s_g)`——每组的 `LANES` 车道先乘上该组标度加进一条
    /// **跨组**的累加链（一条 `fma`，`LANES` 车道互不相干），归约推迟到行尾做一次。于是组循环里
    /// 只剩一条无等待的向量链，而不是 `groups` 条。
    ///
    /// `zero` 折进码值（`Σ (code − zero)·x`），于是不需要每组再拿 `Σ x` 去修正——`x` 的和在两条
    /// 通道里都不再出现。
    fn dot<const CPB: usize, const LANES: usize, const R: usize>(
        w: &[u8],
        row: usize,
        s: &[f32],
        input: &[f32],
        group: usize,
        zero: f32,
    ) -> [f32; R] {
        let per_group = group / CPB;
        let groups = input.len() / group;
        assert!(groups > 0, "一组都凑不满的形状在 admit 就该被挡下");
        let shift = Simd::<f32, LANES>::splat(zero);
        let mut running = [Simd::<f32, LANES>::splat(0.0); R];
        for g in 0..groups {
            let at = g * per_group;
            let acc = group_acc::<CPB, LANES, R>(w, row, input, at, per_group, shift);
            for (r, (acc, running)) in acc.iter().zip(running.iter_mut()).enumerate() {
                *running = acc.mul_add(Simd::splat(s[r * groups + g]), *running);
            }
        }
        running.map(SimdFloat::reduce_sum)
    }

    /// 一组：`CPB` 个槽位 × `per_group / LANES` 段，`R` 条车道链各累加一份。
    #[inline]
    fn group_acc<const CPB: usize, const LANES: usize, const R: usize>(
        w: &[u8],
        row: usize,
        input: &[f32],
        at: usize,
        per_group: usize,
        shift: Simd<f32, LANES>,
    ) -> [Simd<f32, LANES>; R] {
        let mut acc = [Simd::<f32, LANES>::splat(0.0); R];
        for slot in 0..CPB {
            for chunk in (at..at + per_group).step_by(LANES) {
                let xs = Simd::<f32, LANES>::from_slice(&input[slot * row + chunk..][..LANES]);
                for (r, acc) in acc.iter_mut().enumerate() {
                    let bytes = Simd::<u8, LANES>::from_slice(&w[r * row + chunk..][..LANES]);
                    let value = codes::<CPB, LANES>(bytes, slot).cast::<f32>() - shift;
                    *acc = value.mul_add(xs, *acc);
                }
            }
        }
        acc
    }

    /// 第 `slot` 个码的 `LANES` 个无符号码值：低位槽在前，`mask = (1 << width) - 1`。
    fn codes<const CPB: usize, const LANES: usize>(
        bytes: Simd<u8, LANES>,
        slot: usize,
    ) -> Simd<u8, LANES> {
        let width = 8 / CPB as u8;
        (bytes >> Simd::splat(width * slot as u8)) & Simd::splat((1 << width) - 1)
    }

    /// 入口那一遍：把元素重排成"槽位优先"——`input[slot·row + byte]`。
    ///
    /// 元素 `e` 落在槽位 `e % cpb`、行内字节 `e / cpb`。重排与行无关（所以只做一遍），标度不进
    /// 这一步（每组标度逐行不同，见模块文档）。内层读 `x` 连续、写 `cpb` 个连续流。
    fn layout(x: &[f32], affine: Affine) -> Vec<f32> {
        let (row, cpb) = (affine.ctx.row, affine.cpb);
        let mut input = vec![0.0f32; x.len()];
        for byte in 0..row {
            for slot in 0..cpb {
                input[slot * row + byte] = x[byte * cpb + slot];
            }
        }
        input
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/simd_tests.rs");
    }
}

pub use format::{
    Dense, Element, Format, Packed, PackedRow, Plain, PlainFmt, QuantI4, Quantized, SymI8,
    derived_matches_grain, scaled_operands,
};
pub use forward::{Geometry, forward_plain_batch};
pub use shape::{cost, project_shape, reference, sample};

pub(crate) use forward::ScaledRow;
pub(crate) use shape::scale_of;

#[cfg(test)]
#[path = "../tests/packed_test_support.rs"]
pub(crate) mod packed_test_support;

/// 打包量化的分组长度：一行权重的元素个数是它的整数倍，标度按它分组记账。
pub const GROUP: usize = 32;

/// 生成一份「打包量化」GEMV 精度变体的完整公开面：`GROUP` 常量、`PackedRow` 标记、
/// [`annotation::model`] 别名、`forward` 入口与测试挂载。`i2`/`i4` 只差一行 [`Row`]；
/// 新增一档打包精度加一次调用即可。
#[macro_export]
macro_rules! gemv_packed {
    (
        $conf:ident, $alias:ident, $op:tt, $row:ident, $bits:expr,
        $w_bytes:expr, $sample_w:expr, $w_note:literal,
        $dequant_vals:expr, $dequant_expect:expr,
        $w_group:expr, $scales_group:expr, $ref_assert:expr,
        $w_sum:expr, $row0:expr, $row1:expr
    ) => {
        /// 量化分组（声明粒度）：唯一事实源在 [`GROUP`](crate::gemv::GROUP)。
        pub const GROUP: usize = $crate::gemv::GROUP;

        /// 打包配置标记：把一行 [`Row`](crate::quant::Row) 挂进 [`Packed`](crate::gemv::Packed)。
        pub struct $conf;

        impl $crate::gemv::PackedRow for $conf {
            const ROW: $crate::quant::Row = $row;

            fn cache() -> &'static $crate::einsum::FormulaCache {
                $crate::formula_cell!()
            }
        }

        // 基座契约（ADR 0050）：粒度与三条行为全部从格式契约转发。
        $crate::operator_via!(
            grained $crate::gemv::Packed<$conf, { $crate::gemv::GROUP }>,
            $op,
            3,
            (4096, 4096),
            false,
            super,
            $crate::gemv::AXES
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32")]
        #[doc = concat!("`", $op, "` 的语义类型：声明粒度 [`GROUP`]、每行按位宽打包。")]
        pub type $alias = $crate::gemv::Packed<$conf, GROUP>;

        /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
        pub fn forward(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            <$alias as $crate::gemv::Format>::forward(x, weights, scales, out)
        }

        /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只解包一次服务所有输入行。
        ///
        /// 与 [`forward`] 同一套算术与累加顺序，所以批结果与逐行调用逐位相同；差别只在权重的
        /// 读取次数：这里是 1 遍，逐行调用是 `rows` 遍。
        pub fn forward_batch(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            geo: $crate::gemv::Geometry,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            $crate::gemv::packed::forward_batch(x, weights, scales, geo, out, $row.cpb, $row.table)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use $crate::gemv::packed_test_support::{at, multi_group, pack, with};

            $crate::gemv::packed_test_support::packed_gemv_tests!(
                $alias, MODEL, GROUP, forward, $w_bytes, $sample_w
            );
            $crate::gemv::packed_test_support::packed_gemv_config_tests!(
                $alias,
                $op,
                $bits,
                $w_note,
                $dequant_vals,
                $dequant_expect
            );
            $crate::gemv::packed_test_support::packed_gemv_batch_tests!(
                forward,
                forward_batch,
                $row.cpb,
                GROUP
            );
            $crate::gemv::packed_test_support::packed_gemv_value_tests!(
                $alias,
                $bits,
                $w_group,
                $scales_group,
                $ref_assert,
                $w_sum,
                $row0,
                $row1
            );
        }
    };
}

pub mod f32 {
    //! `gemv_f32` 的算子语义：稠密 f32 权重投影。
    //!
    //! 骨架在 [`crate::gemv`]，本模块只有"稠密权重"这一份逐元素配置：表达式、操作数位宽、分块宽度
    //! 与元素反量化。模板在 `ops/gemv_f32.k`，本模块是它的第二份实现：片段只与这里对拍。
    //! 内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。
    use crate::einsum::{Bits, FormulaCache, OperandSpec};
    use crate::gemv::{self, Plain, PlainFmt};

    /// 分块宽度：模板每轮处理这么多元素，标量参考按同一个宽度分块累加。
    pub const BLOCK: usize = 32;

    /// 稠密逐元素的配置：`f32` 权重、无标度、任意输入宽度，按 [`BLOCK`] 分块累加。
    impl<const B: usize> crate::formula::FormulaOwner for gemv::Dense<B> {
        /// `w[nm] · x[m] → y[n]`：无派生轴，宽度不参与分组。
        const EXPR: &'static str = "nm,m,n->n";

        /// 三个操作数都是 f32：`w` 是 `m·n` 个元素、`x` 是 `m` 个、`y` 是 `n` 个。
        const OPERANDS: &'static [OperandSpec] = &[
            OperandSpec::read("w", Bits::ThirtyTwo),
            OperandSpec::read("x", Bits::ThirtyTwo),
            OperandSpec::write("y", Bits::ThirtyTwo),
        ];

        const FRAME: &'static str = "op::gemv::formula";

        fn cache() -> &'static FormulaCache {
            crate::formula_cell!()
        }
    }

    /// 稠密 f32 的样本与反量化（恒等）：骨架的一切算术都由 [`PlainFmt`] 那一份给出。
    impl<const B: usize> Plain for gemv::Dense<B> {
        type Elem = f32;

        /// 稠密算子的形状粒度是 1：一条只为对齐 nibble 的约束不该溢出到 f32 权重上。
        const GRAIN: usize = 1;

        const DERIVED: Option<(char, char, usize)> = None;

        const SCALES: Option<fn(usize) -> f32> = None;

        /// 模板每轮推进的元素数就是分块宽度（样本里的 `gsz` 与分块累加的宽度同源）。
        const STEP: usize = B;

        fn weights(m: usize, n: usize) -> Vec<f32> {
            (0..n * m)
                .map(|index| ((index * 37) % 251) as f32 * 0.125 - 8.0)
                .collect()
        }

        fn to_f32(elem: f32) -> f32 {
            elem
        }
    }

    /// 稠密格式的语义类型：分块宽度 [`BLOCK`]、任意输入宽度都合法。
    #[annotation::model(name = "gemv_f32", shape = [4096, 4096], dtype = "f32")]
    pub type F32 = PlainFmt<gemv::Dense<BLOCK>>;

    // 基座契约（ADR 0050）：粒度与三条行为从格式契约转发。
    crate::operator_via!(grained F32, "gemv_f32", 2, (4096, 4096), true, super, gemv::AXES);

    /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
    pub fn forward(x: &[f32], weights: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        <F32 as gemv::Format>::forward(x, weights, &[], out)
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/gemv.rs");
    }
}
pub mod i2 {
    //! `gemv_i2` 的算子语义：2 位打包权重 + 每 [`GROUP`] 个元素一条标度。
    //!
    //! 骨架在 [`crate::gemv`]，本模块只有"2 位打包"这一份配置：位宽、别名与测试挂载。
    //! 四个 2 位码共用一个字节（低 2 位在前），因此输入宽度 `m` 必须是 4 的倍数且整除分组，
    //! 非 4 倍数尾码在解包前拦下；解包与分组推进的算术都在 [`crate::gemv::Packed`]。
    use crate::quant::I2;

    crate::gemv_packed!(
        I2Conf,
        I2,
        "gemv_i2",
        I2,
        2,
        1,
        16,
        "权重 4 MiB + 输入 16 KiB + 缩放 2 MiB",
        [1, -1, 0, 1],
        0.5,
        [1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1],
        [0.25, 0.5, 1.0, -1.0],
        |y: &[f32]| {
            assert!(y[0].is_finite());
            assert!(y[1].is_finite());
        },
        [1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1],
        [1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0],
        [-2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0]
    );
}
pub mod i4 {
    //! `gemv_i4` 的算子语义：4 位打包权重 + 每 [`GROUP`] 个元素一条标度。
    //!
    //! 骨架在 [`crate::gemv`]，本模块只有"4 位打包"这一份配置：位宽、别名与测试挂载。
    //! 两个 4 位码共用一个字节（低 4 位在前），因此输入宽度 `m` 必须是 2 的倍数且整除分组，
    //! 奇数尾码在解包前拦下；解包与分组推进的算术都在 [`crate::gemv::Packed`]。
    use crate::quant::I4;

    crate::gemv_packed!(
        I4Conf,
        I4,
        "gemv_i4",
        I4,
        4,
        2,
        32,
        "权重 8 MiB + 输入 16 KiB + 缩放 2 MiB",
        [1, 2, 3, 4],
        5.0,
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2],
        [0.25, 0.5, 1.0, 1.0],
        |y: &[f32]| {
            assert!(y[0] > 0.0);
            assert!(y[1] > y[0], "第二行权重更大");
        },
        [1, 2, 3, 4, 5, 6, 7, -8, -1, 0, 1, 2, 3, -4, 5, -6],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0],
        [-1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0, -6.0]
    );
}
pub mod i8 {
    //! `gemv_i8` 的算子语义：对称 8 位权重（不打包、零点 0）+ 每 [`GROUP`] 个元素一条标度。
    //!
    //! 骨架在 [`crate::gemv`]，本模块只有"对称 int8"这一份逐元素配置：表达式、操作数位宽、声明粒度、
    //! 标度取值与元素反量化。模板在 `ops/gemv_i8.k`，片段只与这里对拍。`u8` 承载 int8 位模式，
    //! 算术时按 `i8` 解释。
    use crate::einsum::FormulaCache;
    use crate::gemv::{self, Plain, PlainFmt};
    use crate::quant::I8;

    /// 量化分组（声明粒度）：唯一事实源在 [`GROUP`](crate::gemv::GROUP)。
    pub const GROUP: usize = gemv::GROUP;

    /// 逐元素对称 int8 的配置：`u8` 承载位模式、按 `i8` 反量化、每 [`GROUP`] 个元素一条标度。
    impl<const G: usize> crate::formula::FormulaOwner for gemv::SymI8<G> {
        /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出的派生轴。
        const EXPR: &'static str = "nm,m,ng,n->n";

        // 权重是 8 位（每元素 1 字节），其余同打包格式：操作数表走同一条展开。
        scaled_operands!(I8.bits);

        const FRAME: &'static str = "op::gemv::formula";

        fn cache() -> &'static FormulaCache {
            crate::formula_cell!()
        }
    }

    /// 逐元素对称 int8 的配置：样本取对称小整数，反量化按 `i8` 解释位模式。
    impl<const G: usize> Plain for gemv::SymI8<G> {
        type Elem = u8;

        scaled_grain!(G);

        fn weights(m: usize, n: usize) -> Vec<u8> {
            // 取对称小整数（-8..7）的 int8 位模式，避免大整数在 f32 网格外引入舍入。
            (0..n * m)
                .map(|index| ((index * 37) % 16) as i8 - 8)
                .map(|value| value as u8)
                .collect()
        }

        fn to_f32(elem: u8) -> f32 {
            (elem as i8) as f32
        }
    }

    /// 对称 int8 格式的语义类型：声明粒度 [`GROUP`]、每行 `m` 个字节。
    #[annotation::model(name = "gemv_i8", shape = [4096, 4096], dtype = "f32")]
    pub type I8 = PlainFmt<gemv::SymI8<GROUP>>;

    // 基座契约（ADR 0050）：粒度与三条行为从格式契约转发；量化算子不可微。
    crate::operator_via!(grained I8, "gemv_i8", 3, (4096, 4096), false, super, gemv::AXES);

    /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
    pub fn forward(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        <I8 as gemv::Format>::forward(x, weights, scales, out)
    }

    /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只读一遍。
    ///
    /// 与 [`forward`] 同一份 `scaled_dot` 算术，批结果与逐行调用逐位相同；省掉的是权重行的重复读。
    pub fn forward_batch(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        geo: gemv::Geometry,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        gemv::forward_plain_batch::<gemv::SymI8<GROUP>>(x, weights, scales, geo, out)
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/gemv_i8.rs");
    }
}

/// 每个格式别名的契约自证都必须在**本 crate**里被求值一次。
///
/// [`Format::CHECK`] 是编译期判据，但它只有在被求值时才是判据：注解只负责生成 `MODEL`，
/// 不再引用内核类型（契约层不认识内核），所以强制求值的责任回到格式的归属地。
/// 这四个别名就是 `gemv_f32` / `gemv_i2` / `gemv_i4` / `gemv_i8`，写错粒度或派生轴除数直接编译不过。
const _: () = <f32::F32 as Format>::CHECK;
const _: () = <i2::I2 as Format>::CHECK;
const _: () = <i4::I4 as Format>::CHECK;
const _: () = <i8::I8 as Format>::CHECK;
