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
//! | 权重元素类型、位宽、解包方式 | [`Format::Elem`] / `FormulaOwner::OPERANDS` / `forward` | 只有代码能表达，且必须进单态化 |
//! | 稠密格式的分块宽度 | [`Dense`] 的 const 参数 | 它出现在最内层循环的 `chunks` 上，是真正的热路径常数 |
//! | 量化格式的声明粒度 | [`Quantized`] 的 const 参数 | 它决定派生轴除数与样本里的标度个数（代价与样本，冷路径） |
//! | 实际分组大小 | 运行时数据（`scales` 长度 / `gsz` 实参） | 模板本身就是 `int gsz` 的运行时参数，钉进类型会让训练侧按 8 元素分组的调用编译不过 |
//!
//! 最后一行是刻意的：把"内核已经做成运行时参数"的东西改成类型参数，换来的是每次不同分组都要
//! 一份单态化副本与一次调用点上的匹配，而不是省下一条分支。

use crate::model::{AxisRole, AxisSpec};

/// 本骨架的基本轴：`m` 是规约轴（输入宽度），`n` 是输出轴。
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
macro_rules! scaled_operands {
    ($bits:expr) => {
        /// 三个输入都是读、只有输出是写；权重位宽取配置，其余为 32 位浮点。
        const OPERANDS: &'static [$crate::einsum::OperandSpec] =
            &$crate::gemv::scaled_operands($bits);
    };
}

/// 带标度格式的量化几何：粒度、派生轴、标度取值与步长。
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
}
mod forward {
    //! 前向内核：按行（解码）与按批（prefill）两条路径，点积本身分「带标度」与「稠密」两种。
    use diagnostic::{Code, Error};

    use super::format::Plain;
    use super::shape::{finite_scales, required_weights, scaled_groups};
    use super::simd::COLS;

    /// 带标度行骨架：分组、粒度、权重与标度守卫连同行切片统一收口，逐行点积由 `row_dot` 给出。
    pub(super) fn forward_rows<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        out: &mut [f32],
        cpb: usize,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) -> diagnostic::Result<()> {
        let ctx = scaled_row_ctx(x.len(), out.len(), scales, weights.len(), cpb)?;
        forward_rows_scaled(x, weights, scales, out, ctx, row_dot);
        Ok(())
    }

    /// 逐行遍历一遍，算术由 `row_dot` 给：守卫已经过了（`ctx` 就是证据），这里只剩切片与调用。
    pub(crate) fn forward_rows_scaled<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        out: &mut [f32],
        ctx: ScaledRow,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) {
        for (idx, value) in out.iter_mut().enumerate() {
            let w_row = &weights[idx * ctx.row..][..ctx.row];
            let s_row = &scales[idx * ctx.groups..][..ctx.groups];
            *value = row_dot(w_row, x, s_row, ctx.group);
        }
    }

    /// 行批骨架：输出行在外、输入行在里，一行权重只读一遍。
    fn forward_batch_rows<E>(
        x: &[f32],
        weights: &[E],
        scales: &[f32],
        geo: Geometry,
        out: &mut [f32],
        cpb: usize,
        row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
    ) -> diagnostic::Result<()> {
        let ctx = scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
        forward_batch_rows_scaled(x, weights, scales, geo, out, ctx, row_dot);
        Ok(())
    }

    /// 稠密行批：无标度、输出行在外使一行权重只读一遍，守卫与 [`forward_dense`] 逐条对齐。
    fn forward_batch_dense<C: Plain>(
        x: &[f32],
        weights: &[C::Elem],
        scales: &[f32],
        geo: Geometry,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        dense_scales(scales)?;
        required_weights(weights.len(), geo.n, geo.m)?;
        for row in 0..geo.n {
            let w_row = &weights[row * geo.m..][..geo.m];
            for r in 0..geo.rows {
                out[r * geo.n + row] = C::dot(w_row, &x[r * geo.m..][..geo.m]);
            }
        }
        Ok(())
    }

    /// 稠密格式的标度守卫：稠密没有标度，收到任何一条都是调用方的形状错误。
    fn dense_scales(scales: &[f32]) -> diagnostic::Result<()> {
        if scales.is_empty() {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!("稠密格式没有标度，收到 {} 个", scales.len()),
        ))
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
    pub(super) fn scaled_row_ctx(
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
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ScaledRow {
        /// 每行多少组。
        pub groups: usize,
        /// 每组多少元素（必须是 `cpb` 的倍数）。
        pub group: usize,
        /// 每行多少个权重单位（打包格式是 `m / cpb` 个字节）。
        pub row: usize,
    }

    /// 行批逐元素入口：按配置选择带标度 / 无标度两条骨架，与 `Format::forward` 同一分叉。
    pub fn forward_plain_batch<C: Plain>(
        x: &[f32],
        weights: &[C::Elem],
        scales: &[f32],
        geo: Geometry,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        batch_geometry(x.len(), out.len(), geo)?;
        if C::SCALES.is_some() {
            forward_batch_rows(x, weights, scales, geo, out, 1, scaled_dot::<C>)
        } else {
            forward_batch_dense::<C>(x, weights, scales, geo, out)
        }
    }

    /// 行批几何：`rows` 行输入、输入行宽 `m`、输出行宽 `n`。
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
        dense_scales(scales)?;
        let (m, n) = (x.len(), out.len());
        required_weights(weights.len(), n, m)?;
        let full = n - n % COLS;
        for j in (0..full).step_by(COLS) {
            let block = &weights[j * m..][..COLS * m];
            C::dot_four(block, x, m, &mut out[j..][..COLS]);
        }
        for j in full..n {
            out[j] = C::dot(&weights[j * m..][..m], x);
        }
        Ok(())
    }

    /// 一行的带标度点积：按组推进，每组解包点积一次乘标度（[`Plain::dot`] 承担单组点积）。
    pub(super) fn scaled_dot<C: Plain>(
        w_row: &[C::Elem],
        x: &[f32],
        s_row: &[f32],
        group: usize,
    ) -> f32 {
        let mut acc = 0.0f32;
        for ((w_chunk, scale), x_chunk) in w_row.chunks(group).zip(s_row).zip(x.chunks(group)) {
            acc += C::dot(w_chunk, x_chunk) * scale;
        }
        acc
    }

    pub mod packed {
        //! 打包权重的确定性取样与解包点积助手。

        /// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节。
        pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
            (0..n * (m / cpb))
                .map(|idx| ((idx * 37) % 251) as u8)
                .collect()
        }

        /// 一行的点积：组内解包 `cpb` 个码查 `dequant` 表，整组共用同一个标度。
        pub fn dot_row(
            packed_row: &[u8],
            x: &[f32],
            s_row: &[f32],
            per_group: usize,
            group: usize,
            cpb: usize,
            dequant: &[f32],
        ) -> f32 {
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

        fn row_dot(
            cpb: usize,
            dequant: &[f32],
        ) -> impl Fn(&[u8], &[f32], &[f32], usize) -> f32 + '_ {
            move |w_row, x, s_row, group| dot_row(w_row, x, s_row, group / cpb, group, cpb, dequant)
        }

        /// 打包格式的单行入口：行宽按 `cpb` 折算，逐行点积走 `row_dot`。
        pub fn forward_rows(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            out: &mut [f32],
            cpb: usize,
            dequant: &[f32],
        ) -> diagnostic::Result<()> {
            let ctx = crate::gemv::forward::scaled_row_ctx(
                x.len(),
                out.len(),
                scales,
                weights.len(),
                cpb,
            )?;
            if let Some(affine) = crate::gemv::simd::admit(cpb, out.len(), ctx, dequant) {
                crate::gemv::simd::forward_batch(&[x], weights, scales, affine, out);
                return Ok(());
            }
            crate::gemv::forward::forward_rows_scaled(
                x,
                weights,
                scales,
                out,
                ctx,
                row_dot(cpb, dequant),
            );
            Ok(())
        }

        /// 打包格式的行批入口：与单行入口共用算术，只换遍历嵌套。
        pub fn forward_batch(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            geo: crate::gemv::forward::Geometry,
            out: &mut [f32],
            cpb: usize,
            dequant: &[f32],
        ) -> diagnostic::Result<()> {
            crate::gemv::forward::batch_geometry(x.len(), out.len(), geo)?;
            let ctx =
                crate::gemv::forward::scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
            if let Some(affine) = crate::gemv::simd::admit(cpb, geo.n, ctx, dequant) {
                let rows: Vec<&[f32]> = x.chunks(geo.m).collect();
                crate::gemv::simd::forward_batch(&rows, weights, scales, affine, out);
                return Ok(());
            }
            crate::gemv::forward::forward_batch_rows_scaled(
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

    pub(in crate::gemv) mod shape {
        //! 形状上下文、确定性样本与标量参考：`#[annotation::model]` 要的三件产物。
        use diagnostic::{Code, Error};

        use crate::model::{Buf, Cost, Sample, ShapeCtx};

        use crate::gemv::format::{Element, Format};

        /// 与权重格式无关的输入向量：两条算子共用同一份取值。
        fn input(m: usize) -> Vec<f32> {
            (0..m)
                .map(|idx| ((idx * 7) % 13) as f32 * 0.25 - 1.0)
                .collect()
        }

        /// 量化格式共用的标度取值：`0.03125` 的 1..=4 倍，落在 2 的负幂网格，f32 精确。
        pub(crate) fn scale_of(idx: usize) -> f32 {
            0.031_25 * (1 + (idx % 4)) as f32
        }

        /// 把形状对翻译成投影骨架的具名轴上下文：`m` 是规约轴、`n` 是输出轴。
        pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx, Error> {
            ShapeCtx::new([('m', m), ('n', n)])
        }

        /// 标量代价：完全由 `FormulaOwner::EXPR` 与操作数位宽推出，本函数不含任何流量常数。
        ///
        /// 调用方只给基本轴；派生轴由格式自己声明——分组是算子语义，不是每个层都该抄一遍的常数。
        ///
        /// # Errors
        ///
        /// 形状与表达式不自洽（缺轴、长度为 0，或 `m` 不是 [`Format::GRAIN`] 的整数倍）时返回说明。
        pub fn cost<F: Format>(shape: &ShapeCtx) -> Result<Cost, Error> {
            let extents = match F::DERIVED {
                Some((label, base, divisor)) => shape.extents_with_group(label, base, divisor)?,
                None => shape.extents().clone(),
            };
            F::formula()?
                .cost(&extents)
                .map_err(|error| error.with("nn::gemv::cost"))
        }

        /// 确定性样本：同一形状永远得到同一组输入。
        pub fn sample<F: Format>(m: usize, n: usize) -> Sample {
            let mut bufs = vec![Buf::f32("x", input(m)), F::Elem::buf("w", F::weights(m, n))];
            if let Some(scale) = F::SCALES {
                let groups = m / F::GRAIN.max(1);
                bufs.push(Buf::f32("scales", (0..n * groups).map(scale).collect()));
            }
            bufs.push(Buf::f32("y", vec![0.0; n]));
            Sample {
                m,
                n,
                scalars: vec![("M", m as i64), ("N", n as i64), ("gsz", F::STEP as i64)],
                bufs,
            }
        }

        /// 从样本里取权重：名字固定是 `w`，元素类型由格式给出。
        fn weights<F: Format>(sample: &Sample) -> diagnostic::Result<&[F::Elem]> {
            let buf = sample
                .buf("w")
                .ok_or_else(|| Error::new(Code::Missing, "样本里没有缓冲 `w`"))?;
            F::Elem::borrow(&buf.values).ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!("缓冲 `w` 不是 {}：这份格式的权重元素类型", F::Elem::NAME),
                )
            })
        }

        /// 输出维度不能为 0：三条格式与参考实现共同的前置约束。
        pub(crate) fn non_empty(n: usize) -> Result<(), Error> {
            if n == 0 {
                return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
            }
            Ok(())
        }

        /// 带标度格式的公共前置约束：`scales` 能按行均分、分组能整除输入宽度。
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
        pub(crate) fn required_weights(
            weights: usize,
            n: usize,
            row: usize,
        ) -> Result<usize, Error> {
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
            F::forward(
                tensor::View::one(x),
                &weights[..want],
                tensor::View::one(scales),
                &mut y,
            )?;
            Ok(y)
        }
    }
}
pub use forward::packed;
use forward::shape;
#[cfg(nightly)]
mod simd {
    //! 真 SIMD 点积，两条通道：
    //!
    //! * [`forward`]／[`forward_batch`]——打包格式（一字节 `CPB` 个码）。解包、偏移、乘加全在
    //!   `LANES` 宽的车道上，[`COLS`] 行权重并行，一份 `x` 块摊到 4 行；`x` 就地转置，不额外分配。
    //! * [`elementwise`]——逐元素格式（稠密 `f32`、对称 `int8`）。`LANES` 条车道 × [`CHAINS`] 条
    //!   独立累加链，权重按 [`Load`] 取成车道。
    //!
    //! 两条通道的行尾都只归约一次，于是舍入只发生在"每个元素乘加一次"这一层。

    use std::simd::prelude::{SimdFloat, SimdInt, SimdUint};
    use std::simd::{Simd, StdFloat};

    use super::ScaledRow;

    /// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 块摊到 4 行。
    pub(crate) const COLS: usize = 4;

    /// 宽档车道数：一条 AVX2 `ymm`、四条 NEON 寄存器。
    const WIDE: usize = 16;

    /// 窄档车道数：一组凑不满 [`WIDE`] 时用，半个 `ymm`、两条 NEON 寄存器。
    const NARROW: usize = 8;

    /// 逐元素通道的独立累加链数：一条车道链的乘加延迟约 4 拍，4 条刚好把流水填满。
    const CHAINS: usize = 4;

    /// 反量化表的仿射零：`table[code] == code - zero` 时返回 `zero`，否则 `None`（走标量叶）。
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

    /// 车道数：一组里每槽位的字节数被 [`WIDE`] 整除取宽档，被 [`NARROW`] 整除取窄档，都不行就不走本通道。
    ///
    /// 车道数整除一组是硬前提：真整除才能让索引算术完全无分支、无掩码、无零头。真实分组
    /// （[`GROUP`](super::GROUP) 的倍数配 2 / 4 个码一字节）永远落在宽档或窄档，其余形状走标量叶，
    /// 算术一模一样，只是不向量化。
    fn lanes_of(per_group: usize) -> Option<usize> {
        if per_group.is_multiple_of(WIDE) {
            Some(WIDE)
        } else if per_group.is_multiple_of(NARROW) {
            Some(NARROW)
        } else {
            None
        }
    }

    /// 通过校验之后的通道参数：形状（[`ScaledRow`]）、码宽、车道数、反量化表的仿射零点。
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Affine {
        ctx: ScaledRow,
        cpb: usize,
        lanes: usize,
        zero: f32,
    }

    /// 校验并构造 [`Affine`]：`n` 是输出宽度（权重行数），`ctx` 是入口已经量好的行形状。
    pub(crate) fn admit(cpb: usize, n: usize, ctx: ScaledRow, dequant: &[f32]) -> Option<Affine> {
        // 码表长度 `1 << width`，`width = 8 / cpb`（与 `quant::width_cpb` 同一公式）。
        let zero = affine_zero(dequant, 1 << crate::quant::width_cpb(cpb))?;
        let lanes = lanes_of(ctx.group / cpb)?;
        (n >= COLS).then_some(Affine {
            ctx,
            cpb,
            lanes,
            zero,
        })
    }

    /// 唯一入口：一次带上所有输入行，单行只是 `rows` 只有一个元素。
    pub(crate) fn forward_batch(
        rows: &[&[f32]],
        weights: &[u8],
        scales: &[f32],
        affine: Affine,
        out: &mut [f32],
    ) {
        run(
            &Plan {
                weights,
                inputs: rows,
                affine,
                scales,
            },
            out,
        );
    }

    /// 内核的全部输入：权重、输入行、通道参数、标度。
    struct Plan<'a> {
        weights: &'a [u8],
        inputs: &'a [&'a [f32]],
        affine: Affine,
        scales: &'a [f32],
    }

    /// 两个轴一起单态化：`cpb` 决定槽位循环与位移量，车道数决定向量宽度，编译期全摊平。
    fn run(plan: &Plan<'_>, out: &mut [f32]) {
        match (plan.affine.cpb, plan.affine.lanes == WIDE) {
            (2, true) => columns::<2, WIDE>(plan, out),
            (2, false) => columns::<2, NARROW>(plan, out),
            (_, true) => columns::<4, WIDE>(plan, out),
            (_, false) => columns::<4, NARROW>(plan, out),
        }
    }

    /// 列遍历：整块走 `R = COLS`，不足一块的尾部逐列走 `R = 1`（同一个 [`dot`]、同一副累加器
    /// 形态），所以 `n` 不必是 [`COLS`] 的倍数。
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
        let ctx = plan.affine.ctx;
        let (w, s) = (
            &plan.weights[j * ctx.row..][..R * ctx.row],
            &plan.scales[j * ctx.groups..][..R * ctx.groups],
        );
        for (r, input) in plan.inputs.iter().enumerate() {
            let dest = &mut out[r * n + j..][..R];
            dest.copy_from_slice(&dot::<CPB, LANES, R>(w, s, input, ctx, plan.affine.zero));
        }
    }

    /// `R` 行的点积：一组一条 `LANES` 车道链，`R` 行同时推进，**每行只在行尾归约一次**。
    ///
    /// `ctx.groups` 不是从 `input.len()` 反推出来的，而是入口 [`admit`] 量好的：形状守卫已经保证
    /// `input.len() == ctx.groups · ctx.group`，所以这里没有"一组都凑不满"这种需要断言的中间态。
    fn dot<const CPB: usize, const LANES: usize, const R: usize>(
        w: &[u8],
        s: &[f32],
        input: &[f32],
        ctx: ScaledRow,
        zero: f32,
    ) -> [f32; R] {
        let per_group = ctx.group / CPB;
        let shift = Simd::<f32, LANES>::splat(zero);
        let mut running = [Simd::<f32, LANES>::splat(0.0); R];
        for g in 0..ctx.groups {
            let at = g * per_group;
            let acc = group_acc::<CPB, LANES, R>(w, ctx.row, input, at, per_group, shift);
            for (r, (acc, running)) in acc.iter().zip(running.iter_mut()).enumerate() {
                *running = acc.mul_add(Simd::splat(s[r * ctx.groups + g]), *running);
            }
        }
        running.map(SimdFloat::reduce_sum)
    }

    /// 一组：`CPB` 个槽位 × `per_group / LANES` 段，`R` 条车道链各累加一份。
    ///
    /// `x` 在这里就地转置——每段只取一次连续的 `LANES · CPB` 个元素，摊成 `CPB` 条车道向量，
    /// 于是 `x` 每元素只读一遍，入口那一遍整行重排连同它的分配一起消失。
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
        for byte in (at..at + per_group).step_by(LANES) {
            let block = &input[byte * CPB..][..LANES * CPB];
            for slot in 0..CPB {
                let xs = slot_lanes::<CPB, LANES>(block, slot);
                for (r, acc) in acc.iter_mut().enumerate() {
                    let bytes = Simd::<u8, LANES>::from_slice(&w[r * row + byte..][..LANES]);
                    let value = codes::<CPB, LANES>(bytes, slot).cast::<f32>() - shift;
                    *acc = value.mul_add(xs, *acc);
                }
            }
        }
        acc
    }

    /// 一段输入的某个槽位，摊成 `LANES` 条车道：槽位 `s` 取第 `i · CPB + s` 个元素。
    #[inline]
    fn slot_lanes<const CPB: usize, const LANES: usize>(
        block: &[f32],
        slot: usize,
    ) -> Simd<f32, LANES> {
        let mut lane = [0.0f32; LANES];
        for (i, value) in lane.iter_mut().enumerate() {
            *value = block[i * CPB + slot];
        }
        Simd::from_array(lane)
    }

    /// 第 `slot` 个码的 `LANES` 个无符号码值：低位槽在前，`mask = (1 << width) - 1`。
    fn codes<const CPB: usize, const LANES: usize>(
        bytes: Simd<u8, LANES>,
        slot: usize,
    ) -> Simd<u8, LANES> {
        let width = 8 / CPB as u8;
        (bytes >> Simd::splat(width * slot as u8)) & Simd::splat((1 << width) - 1)
    }

    /// 逐元素通道的权重取法：稠密 `f32` 直取，对称 `int8` 按二补数重新解释。
    ///
    /// 一条通道只由权重元素类型选，[`elementwise`] 因此对两种格式是同一段代码。
    pub(crate) trait Load: Copy {
        /// [`WIDE`] 个元素一次取成 `f32` 车道。
        fn wide(block: &[Self]) -> Simd<f32, WIDE>;

        /// 单个元素：不足一条车道的零头用。
        fn one(elem: Self) -> f32;
    }

    impl Load for f32 {
        #[inline]
        fn wide(block: &[Self]) -> Simd<f32, WIDE> {
            Simd::from_slice(block)
        }

        #[inline]
        fn one(elem: Self) -> f32 {
            elem
        }
    }

    impl Load for u8 {
        /// `u8 → i8` 是位模式重解释，`i8 → f32` 是有符号展宽，两条都是单条指令。
        #[inline]
        fn wide(block: &[Self]) -> Simd<f32, WIDE> {
            Simd::<u8, WIDE>::from_slice(block)
                .cast::<i8>()
                .cast::<f32>()
        }

        #[inline]
        fn one(elem: Self) -> f32 {
            (elem as i8) as f32
        }
    }

    /// 连续点积骨架：`CHAINS` 条独立累加链轮转，权重按 [`Load`] 取成车道，行尾归约一次。
    ///
    /// 这是 `f32` 与对称 `int8` 共用的唯一一份点积——两种格式的差别只剩 [`Load`] 的取法。
    ///
    /// 链按 [`CHAINS`] 分段常量展开，`acc` 的下标因此是编译期常量、累加器整份留在向量寄存器里。
    /// 分段之后块 `g·CHAINS + c` 落在链 `c`，与"块下标取模"是同一张映射，故逐位不变——2026-11-22
    /// 实测两者同为 `gemv_f32/decode` 39.8–40.2 GiB/s（离散 0.5%），即常量下标只是把意图写明，
    /// 并没有换来速度：这条路径只在 `cfg(nightly)` 下编译，那台机器上的实测数字属于 stable 退路
    /// （本文件 `gemv::simd` 的 stable 分支），热路径结论必须在那条通道上取。
    pub(crate) fn elementwise<T: Load>(w: &[T], x: &[f32]) -> f32 {
        let (w_groups, w_tail) = w.as_chunks::<{ WIDE * CHAINS }>();
        let (x_groups, x_tail) = x.as_chunks::<{ WIDE * CHAINS }>();
        let mut acc = [Simd::<f32, WIDE>::splat(0.0); CHAINS];
        for (wg, xg) in w_groups.iter().zip(x_groups) {
            for c in 0..CHAINS {
                let (wc, xc) = (&wg[c * WIDE..][..WIDE], &xg[c * WIDE..][..WIDE]);
                acc[c] = T::wide(wc).mul_add(Simd::from_slice(xc), acc[c]);
            }
        }
        let mut total = acc.iter().copied().map(SimdFloat::reduce_sum).sum::<f32>();
        for (w, x) in w_tail.iter().zip(x_tail) {
            total += T::one(*w) * *x;
        }
        total
    }

    /// [`COLS`] 行并行点积：一份 `x` 段摊到 [`COLS`] 行，权重每 64 B 的取数从 8 次降到 5 次。
    ///
    /// 行内按 [`WIDE`] 整除的部分走向量，零头逐行标量收口——零头重读 `x` 的代价可忽略，换来行宽
    /// 不必是 [`WIDE`] 的倍数。每行只有一条累加链：链的独立性由"行"提供（[`COLS`] 条独立链），
    /// 而不是由链数提供，所以累加器刚好 `COLS` 份，全部留在寄存器里。
    pub(crate) fn elementwise_four<T: Load>(w: &[T], x: &[f32], m: usize, out: &mut [f32]) {
        let mut acc = [Simd::<f32, WIDE>::splat(0.0); COLS];
        let full = m - m % WIDE;
        let row = |r: usize| &w[r * m..][..m];
        let chunks: [&[[T; WIDE]]; COLS] = [0, 1, 2, 3].map(|r| row(r).as_chunks::<WIDE>().0);
        let (x_head, x_tail) = x.split_at(full);
        for (k, xv) in x_head.as_chunks::<WIDE>().0.iter().enumerate() {
            let xv = Simd::from_slice(xv);
            for r in 0..COLS {
                acc[r] = T::wide(&chunks[r][k]).mul_add(xv, acc[r]);
            }
        }
        for (r, o) in out.iter_mut().enumerate() {
            let mut total = acc[r].reduce_sum();
            for (wc, xc) in row(r)[full..].iter().zip(x_tail) {
                total += T::one(*wc) * *xc;
            }
            *o = total;
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_simd_tests.rs");
    }
}
#[cfg(not(nightly))]
mod simd {
    //! stable 退路：与 nightly 版的 `super::simd` 同名同签名，只是不做显式车道。
    //!
    //! 两边的差别只有「向量化没向量化」，语义一模一样：
    //!
    //! * [`Load::one`] + [`elementwise`] 保持与向量版**相同的分段形状**——`WIDE` 个元素一段、
    //!   `CHAINS` 条独立累加链轮转。段内逐条 `mul_add`，于是每条链累加的元素序列与向量版逐车道
    //!   一一对应，stable 上 LLVM 也能把这份源码自动向量化成近乎同一组指令；两份实现只差最后那
    //!   一步归约（`reduce_sum` 是树形，这里是顺序）。
    //! * 打包格式的通道（`admit` / `forward_batch`）在 stable 上恒为不可达：调用方
    //!   `gemv::packed` 本来就有标量叶 `forward::forward_rows_scaled` 兜底，那条路径一直都在。

    use super::ScaledRow;

    /// 宽档车道数，与向量版一致：决定 [`elementwise`] 的分段宽度。
    const WIDE: usize = 16;

    /// 独立累加链数，与向量版一致：一条链的乘加延迟约 4 拍，4 条刚好把流水填满。
    const CHAINS: usize = 4;

    /// 行并行度：一次算 [`COLS`] 行，一份 `x` 段摊到这 [`COLS`] 行上。
    ///
    /// 实测每多一条行流就多一条 DRAM 预取流，2026-11-22 扫描 2/4/8 得到 55.1／48.9／44.7 GiB/s，
    /// 故取 2：`x` 复用与流数的最优点，再往上就是流数占主导。
    pub(crate) const COLS: usize = 2;

    /// 段宽：一条累加链一次推进的元素数，与 [`WIDE`] 同义但可独立调（见 [`COLS`] 的扫描）。
    const SPAN: usize = 8;

    /// 每条行的独立累加链数：行并行度只有 [`COLS`]，链不够就喂不满 4 拍的乘加延迟。
    const DEPTH: usize = 2;

    /// 逐元素通道的权重取法：稠密 `f32` 直取，对称 `int8` 按二补数重新解释。
    ///
    /// 一条通道只由权重元素类型选，[`elementwise`] 因此对两种格式是同一段代码。
    pub(crate) trait Load: Copy {
        /// 单个元素取成 `f32`。
        fn one(elem: Self) -> f32;
    }

    impl Load for f32 {
        fn one(elem: Self) -> f32 {
            elem
        }
    }

    impl Load for u8 {
        /// `u8 → i8` 是位模式重解释，`i8 → f32` 是有符号展宽。
        fn one(elem: Self) -> f32 {
            f32::from(elem as i8)
        }
    }

    /// [`COLS`] 行并行点积：`w` 是连续 [`COLS`] 条行、每行 `m` 个元素，结果写进 `out` 的 [`COLS`] 个槽。
    ///
    /// 形状是 `COLS × DEPTH × SPAN` 的累加器：一行一份 `x` 段摊给 [`COLS`] 行共用，行内再拆
    /// [`DEPTH`] 条独立链去盖住乘加的 4 拍延迟。稳定版没有显式车道，靠的是循环体形状本身
    /// 够整齐，让 LLVM 自己向量化——所以 `SPAN · DEPTH` 必须是 2 的幂，否则整段退化成标量
    /// （实测 `COLS = 3` 掉到 30 GiB/s）。
    ///
    /// 三个常数都不是猜的：2026-11-22 在同机同口径下扫过 `COLS ∈ {2,4,8}`（55.1／48.9／44.7）
    /// 与 `SPAN ∈ {8,16,32,64}`、`DEPTH ∈ {1,2,3,4}`，定格在 2／8／2 —— 再大的 `COLS` 每多一条行流
    /// 就多一条 DRAM 预取流，与 `x` 的流争抢，收益反转为负。
    pub(crate) fn elementwise_four<T: Load>(w: &[T], x: &[f32], m: usize, out: &mut [f32]) {
        let mut acc = [[[0.0f32; SPAN]; DEPTH]; COLS];
        let group = SPAN * DEPTH;
        let full = m - m % group;
        let (x_head, x_tail) = x.split_at(full);
        for (g, xg) in x_head.as_chunks::<{ SPAN * DEPTH }>().0.iter().enumerate() {
            commit(&mut acc, w, m, xg, g * group);
        }
        for (r, (o, chain)) in out.iter_mut().zip(acc).enumerate() {
            *o = finish(w, m, r, full, x_tail, chain);
        }
    }

    /// 把一个 `x` 组（`SPAN · DEPTH` 个元素）乘进 [`COLS`] 行 × [`DEPTH`] 链的累加器。
    ///
    /// 单独成函数只为把嵌套压到门禁的 5 以内；`#[inline]` 保证与内联版逐指令一致——这是
    /// 排版，不是抽象层。
    #[inline]
    fn commit<T: Load>(
        acc: &mut [[[f32; SPAN]; DEPTH]; COLS],
        w: &[T],
        m: usize,
        xg: &[f32],
        at: usize,
    ) {
        for (r, row) in acc.iter_mut().enumerate() {
            for (d, chain) in row.iter_mut().enumerate() {
                let xs = &xg[d * SPAN..][..SPAN];
                let ws = &w[r * m + at + d * SPAN..][..SPAN];
                for (lane, (wv, xv)) in ws.iter().zip(xs).enumerate() {
                    chain[lane] = T::one(*wv).mul_add(*xv, chain[lane]);
                }
            }
        }
    }

    /// 归约一条链并吃掉行尾（`m` 不是 `SPAN · DEPTH` 整数倍时剩下的那几个）。
    #[inline]
    fn finish<T: Load>(
        w: &[T],
        m: usize,
        r: usize,
        full: usize,
        x_tail: &[f32],
        chain: [[f32; SPAN]; DEPTH],
    ) -> f32 {
        let mut total: f32 = chain.iter().flatten().sum();
        for (wc, xc) in w[r * m + full..][..m - full].iter().zip(x_tail) {
            total = T::one(*wc).mul_add(*xc, total);
        }
        total
    }

    /// 连续点积：`CHAINS` 条独立累加链轮转，行尾归约一次。
    pub(crate) fn elementwise<T: Load>(w: &[T], x: &[f32]) -> f32 {
        let (w_chunks, w_tail) = w.as_chunks::<WIDE>();
        let (x_chunks, x_tail) = x.as_chunks::<WIDE>();
        let mut acc = [[0.0f32; WIDE]; CHAINS];
        for (i, (w, x)) in w_chunks.iter().zip(x_chunks).enumerate() {
            let chain = &mut acc[i % CHAINS];
            for (k, (w, x)) in w.iter().zip(x).enumerate() {
                chain[k] = T::one(*w).mul_add(*x, chain[k]);
            }
        }
        let mut total: f32 = acc.iter().flatten().sum();
        for (w, x) in w_tail.iter().zip(x_tail) {
            total = T::one(*w).mul_add(*x, total);
        }
        total
    }

    /// 打包格式的通道参数占位：与 nightly 版的 `Affine` 对位的稳定退路类型；
    /// [`admit`] 恒为 `None`，这个值永远不会被构造出来。
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct StableChannel;

    /// 打包格式的通道参数：stable 上没有向量通道，恒为 `None`，调用方落到标量叶。
    pub(crate) fn admit(
        _cpb: usize,
        _n: usize,
        _ctx: ScaledRow,
        _dequant: &[f32],
    ) -> Option<StableChannel> {
        None
    }

    /// 只有 [`admit`] 返回 `Some` 才会走到这里，而 stable 上它永远不返回 `Some`。
    ///
    /// 留空壳只是为了与 nightly 版同名同形；参数用 [`StableChannel`] 而不是 `Infallible`，
    /// 避免编译器把这个退路分支推断成 never 发散而触发 `unreachable_code`。
    pub(crate) fn forward_batch(
        _rows: &[&[f32]],
        _weights: &[u8],
        _scales: &[f32],
        _channel: StableChannel,
        _out: &mut [f32],
    ) {
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
mod packed_test_support {
    include!("../tests/kernel_packed_test_support.rs");
}

/// 打包量化的分组长度：一行权重的元素个数是它的整数倍，标度按它分组记账。
pub const GROUP: usize = 32;

/// 生成一份「打包量化」GEMV 精度变体的完整公开面：`GROUP` 常量、`PackedRow` 标记、
/// [`annotation::model`] 别名、`forward` 入口与测试挂载。`i2`/`i4` 只差一行 `Row`；
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
            x: tensor::View<'_, f32>,
            weights: &[u8],
            scales: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            <$alias as $crate::gemv::Format>::forward(x, weights, scales, out)
        }

        /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只解包一次服务所有输入行。
        pub fn forward_batch(
            x: tensor::View<'_, f32>,
            weights: &[u8],
            scales: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            let (rows, m) = $crate::matmul::dim2(x.dims(), $op)?;
            let n = out.len().checked_div(rows).unwrap_or(0);
            let geo = $crate::gemv::Geometry::new(rows, m, n);
            $crate::gemv::packed::forward_batch(
                x.as_slice(),
                weights,
                scales.as_slice(),
                geo,
                out,
                $row.cpb,
                $row.table,
            )
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

        const FRAME: &'static str = "nn::gemv::formula";

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
                .map(|idx| ((idx * 37) % 251) as f32 * 0.125 - 8.0)
                .collect()
        }

        fn to_f32(elem: f32) -> f32 {
            elem
        }

        /// 稠密 `f32` 走 `simd::elementwise`：`WIDE` 条车道 × 独立累加链，整段乘加编译成向量指令。
        fn dot(w_row: &[f32], x: &[f32]) -> f32 {
            super::simd::elementwise(w_row, x)
        }

        /// 行并行：一份 `x` 段摊到 `COLS` 行，解码阶段每 64 B 权重的取数
        /// 从单行版的 8 次降到 6 次，是 K9 从 40 GiB/s 抬到 66 GiB/s 的主因。
        fn dot_four(w: &[f32], x: &[f32], m: usize, out: &mut [f32]) {
            super::simd::elementwise_four(w, x, m, out)
        }
    }

    /// 稠密格式的语义类型：分块宽度 [`BLOCK`]、任意输入宽度都合法。
    #[annotation::model(name = "gemv_f32", shape = [4096, 4096], dtype = "f32")]
    pub type F32 = PlainFmt<gemv::Dense<BLOCK>>;

    // 基座契约（ADR 0050）：粒度与三条行为从格式契约转发。
    crate::operator_via!(grained F32, "gemv_f32", 2, (4096, 4096), true, super, gemv::AXES);

    /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
    pub fn forward(
        x: tensor::View<'_, f32>,
        weights: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        <F32 as gemv::Format>::forward(x, weights, tensor::View::one(&[]), out)
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_gemv.rs");
    }
}
pub mod i2 {
    //! `gemv_i2` 的算子语义：2 位打包权重 + 每 [`GROUP`] 个元素一条标度。
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

        const FRAME: &'static str = "nn::gemv::formula";

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
                .map(|idx| ((idx * 37) % 16) as i8 - 8)
                .map(|value| value as u8)
                .collect()
        }

        fn to_f32(elem: u8) -> f32 {
            f32::from(elem as i8)
        }

        /// 对称 `int8` 走 `simd::elementwise`：位模式原样重解释成 `i8` 再展宽，零点 0 无需偏移。
        fn dot(w_row: &[u8], x: &[f32]) -> f32 {
            super::simd::elementwise(w_row, x)
        }
    }

    /// 对称 int8 格式的语义类型：声明粒度 [`GROUP`]、每行 `m` 个字节。
    #[annotation::model(name = "gemv_i8", shape = [4096, 4096], dtype = "f32")]
    pub type I8 = PlainFmt<gemv::SymI8<GROUP>>;

    // 基座契约（ADR 0050）：粒度与三条行为从格式契约转发；量化算子不可微。
    crate::operator_via!(grained I8, "gemv_i8", 3, (4096, 4096), false, super, gemv::AXES);

    /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
    pub fn forward(
        x: tensor::View<'_, f32>,
        weights: &[u8],
        scales: tensor::View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        <I8 as gemv::Format>::forward(x, weights, scales, out)
    }

    /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只读一遍。
    pub fn forward_batch(
        x: tensor::View<'_, f32>,
        weights: &[u8],
        scales: tensor::View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let (rows, m) = crate::matmul::dim2(x.dims(), "gemv_i8")?;
        let n = out.len().checked_div(rows).unwrap_or(0);
        let geo = gemv::Geometry::new(rows, m, n);
        gemv::forward_plain_batch::<gemv::SymI8<GROUP>>(
            x.as_slice(),
            weights,
            scales.as_slice(),
            geo,
            out,
        )
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_gemv_i8.rs");
    }
}

/// 每个格式别名的契约自证都必须在**本 crate**里被求值一次。
const _: () = <f32::F32 as Format>::CHECK;
const _: () = <i2::I2 as Format>::CHECK;
const _: () = <i4::I4 as Format>::CHECK;
const _: () = <i8::I8 as Format>::CHECK;
