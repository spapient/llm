//! 权重格式：把「一行权重怎么摆」抽象成 trait，解包与分组约束都由格式自己负责。
use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::formula::FormulaOwner;
use crate::model::{Buf, Values};
use crate::quant::Row;

use super::forward::{forward_dense, forward_rows, scaled_dot};
use super::packed;
use super::shape::non_empty;
use tensor::View;

/// 权重元素：进样本、读缓冲、拼 `_Generic` 三处都按它分派。
pub trait Element: Copy + PartialEq + std::fmt::Debug + 'static {
    /// 诊断里怎么称呼它。
    const NAME: &'static str;

    /// 从样本缓冲里借出同类元素；类型不符返回 `None`。
    fn borrow(values: &Values) -> Option<&[Self]>;

    /// 直接从 `u8` 切片借出本元素；`u8` 元素返回原样，`f32` 返回 `None`。
    fn borrow_u8(slice: &[Self]) -> Option<&[u8]>;

    /// 从打包字节构造本元素向量；`u8` 元素原样回收，`f32` 不可用。
    fn from_bytes(bytes: Vec<u8>) -> Vec<Self>;

    /// 装成同名缓冲。
    fn buf(name: &'static str, values: Vec<Self>) -> Buf;
}

impl Element for f32 {
    const NAME: &'static str = "f32";

    fn borrow(values: &Values) -> Option<&[Self]> {
        match values {
            Values::F32(values) => Some(values),
            Values::U8(_)
            | Values::F16(_)
            | Values::Bf16(_)
            | Values::F32x4(_)
            | Values::F16x4(_)
            | Values::Bf16x4(_)
            | Values::U8x4(_) => None,
        }
    }

    fn borrow_u8(_slice: &[Self]) -> Option<&[u8]> {
        None
    }

    fn from_bytes(_bytes: Vec<u8>) -> Vec<Self> {
        // f32 逐元素格式不参与打包取样：此路径在单态化时不可达，留空切片作安全占位。
        Vec::new()
    }

    fn buf(name: &'static str, values: Vec<Self>) -> Buf {
        Buf::f32(name, values)
    }
}

impl Element for u8 {
    const NAME: &'static str = "u8";

    fn borrow(values: &Values) -> Option<&[Self]> {
        match values {
            Values::F32(_)
            | Values::F16(_)
            | Values::Bf16(_)
            | Values::F32x4(_)
            | Values::F16x4(_)
            | Values::Bf16x4(_)
            | Values::U8x4(_) => None,
            Values::U8(values) => Some(values),
        }
    }

    fn borrow_u8(slice: &[Self]) -> Option<&[u8]> {
        Some(slice)
    }

    fn from_bytes(bytes: Vec<u8>) -> Vec<Self> {
        bytes
    }

    fn buf(name: &'static str, values: Vec<Self>) -> Buf {
        Buf {
            name,
            values: Values::U8(values),
        }
    }
}

/// 格式自证：派生轴的除数必须就是形状粒度。
pub const fn derived_matches_grain(grain: usize, derived: Option<(char, char, usize)>) -> bool {
    match derived {
        Some((_, _, divisor)) => divisor == grain,
        None => true,
    }
}

/// 带分组标度的操作数表：权重按 `bits` 读，输入与标度都是 32 位浮点，输出写 32 位。
pub const fn scaled_operands(bits: Bits) -> [OperandSpec; 4] {
    [
        OperandSpec::read("w", bits),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::read("scales", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// 权重格式：一条 GEMV 里随精度变化的那部分。
pub trait Format: FormulaOwner {
    /// 权重在样本里的元素类型。
    type Elem: Element;

    /// 形状粒度：`m` 必须是它的正整数倍；`1` 表示任意宽度。
    const GRAIN: usize;

    /// 派生轴 `(派生轴, 基准轴, 除数)`：量化格式声明一条（`g = m / GRAIN`），稠密格式没有。
    const DERIVED: Option<(char, char, usize)>;

    /// 标度取值函数；`None` 表示这份格式没有标度缓冲（每个标度不带参数）。
    const SCALES: Option<fn(usize) -> f32>;

    /// 模板每轮推进的元素数，也是样本里 `gsz` 的取值。
    const STEP: usize;

    /// 契约自证：本格式的声明必须自洽。
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
    /// 激活 `x` 与标度 `scales` 是普通 `f32` 元素，包成只读张量视图；权重按位宽打包或
    /// 位模式承载，保持裸切片（亚字节/重解释语义不进 tensor 元素集）。
    fn forward(
        x: View<'_, f32>,
        weights: &[Self::Elem],
        scales: View<'_, f32>,
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
pub type Quantized<const GRAIN: usize> = Packed<QuantI4, GRAIN>;

/// 稠密逐元素格式的配置标记：每 `BLOCK` 个元素一块累加（最后一块可以短），无打包、无标度。
pub struct Dense<const BLOCK: usize>;

/// 逐元素对称 int8 格式的配置标记：不打包（每元素 1 字节、无 nibble 解包），有 per-group
/// 标度。`u8` 承载 int8 的位模式，算术时按 `i8` 解释。`impl Plain` 在 `gemv_i8`。
pub struct SymI8<const GRAIN: usize>;

/// 打包格式的静态配置提供者：把一档精度的 [`Row`] 挂到一个标记类型上。
pub trait PackedRow {
    /// 这档精度的静态配置（位宽 / 码数 / 零点 / 码表 / 偏移）。
    const ROW: Row;

    /// 这档精度自己的表达式缓存单元格。
    fn cache() -> &'static FormulaCache;
}

/// 配置驱动的打包权重格式：`GRAIN` 个元素共用一条标度，打包密度由 [`PackedRow::ROW`] 给出。
pub struct Packed<P: PackedRow, const G: usize> {
    _config: core::marker::PhantomData<P>,
}

impl<P: PackedRow, const G: usize> FormulaOwner for Packed<P, G> {
    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 `G` 个元素切出来的派生轴。
    const EXPR: &'static str = "nm,m,ng,n->n";

    scaled_operands!(P::ROW.bits);

    const FRAME: &'static str = "nn::gemv::formula";

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
        x: View<'_, f32>,
        weights: &[u8],
        scales: View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        packed::forward_rows(
            x.as_slice(),
            weights,
            scales.as_slice(),
            out,
            P::ROW.cpb,
            P::ROW.table,
        )
    }
}

/// 逐元素（不打包）GEMV 的静态配置：元素类型、量化几何、样本与元素反量化。
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

    /// 一段权重与输入的逐元素点积：默认逐元素标量推进。
    ///
    /// `f32` 与对称 `int8` 都覆盖成 `simd::elementwise` 的向量通道——这里是唯一一处
    /// 配置能自己决定"用哪台机器算这一行"，其余算术全部锁在骨架里。
    fn dot(w_row: &[Self::Elem], x: &[f32]) -> f32 {
        w_row
            .iter()
            .zip(x)
            .map(|(elem, xv)| Self::to_f32(*elem) * xv)
            .sum()
    }

    /// `COLS` 行并行点积：`w` 是连续 `COLS` 条行、
    /// 每行 `m` 个元素，结果写进 `out` 的 `COLS` 个槽。
    ///
    /// 默认逐行转发 [`Plain::dot`]；稠密 `f32` 覆盖成 `simd::elementwise_four`，
    /// 一份 `x` 段摊到多行。行数由 `COLS` 编译期定死，没有运行期分支。
    fn dot_four(w: &[Self::Elem], x: &[f32], m: usize, out: &mut [f32]) {
        for (r, o) in out.iter_mut().enumerate() {
            *o = Self::dot(&w[r * m..][..m], x);
        }
    }
}

/// 配置驱动的逐元素权重格式：把一条 [`Plain`] 配置挂进 [`Format`] 骨架。
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
        x: View<'_, f32>,
        weights: &[Self::Elem],
        scales: View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let x = x.as_slice();
        let scales = scales.as_slice();
        non_empty(out.len())?;
        if C::SCALES.is_some() {
            forward_rows(x, weights, scales, out, 1, scaled_dot::<C>)
        } else {
            forward_dense::<C>(x, weights, scales, out)
        }
    }
}
