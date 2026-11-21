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
