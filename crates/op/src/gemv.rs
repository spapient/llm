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

mod format;
mod forward;

pub mod packed;

mod shape;
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

pub mod f32;
pub mod i2;
pub mod i4;
pub mod i8;
