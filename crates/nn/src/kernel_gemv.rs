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

#[path = "kernel_gemv_format.rs"]
mod format;
#[path = "kernel_gemv_forward.rs"]
mod forward;
#[path = "kernel_gemv_packed.rs"]
pub mod packed;
#[path = "kernel_gemv_shape.rs"]
mod shape;
#[cfg(nightly)]
#[path = "kernel_gemv_simd_nightly.rs"]
mod simd;
#[cfg(not(nightly))]
#[path = "kernel_gemv_simd_stable.rs"]
mod simd;
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

        /// 稠密 `f32` 走 [`simd::elementwise`]：`WIDE` 条车道 × 独立累加链，整段乘加编译成向量指令。
        fn dot(w_row: &[f32], x: &[f32]) -> f32 {
            super::simd::elementwise(w_row, x)
        }

        /// 行并行：一份 `x` 段摊到 [`COLS`](super::simd::COLS) 行，解码阶段每 64 B 权重的取数
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
    pub fn forward(x: &[f32], weights: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        <F32 as gemv::Format>::forward(x, weights, &[], out)
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
            (elem as i8) as f32
        }

        /// 对称 `int8` 走 [`simd::elementwise`]：位模式原样重解释成 `i8` 再展宽，零点 0 无需偏移。
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
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        <I8 as gemv::Format>::forward(x, weights, scales, out)
    }

    /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只读一遍。
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
        include!("../tests/kernel_gemv_i8.rs");
    }
}

/// 每个格式别名的契约自证都必须在**本 crate**里被求值一次。
const _: () = <f32::F32 as Format>::CHECK;
const _: () = <i2::I2 as Format>::CHECK;
const _: () = <i4::I4 as Format>::CHECK;
const _: () = <i8::I8 as Format>::CHECK;
