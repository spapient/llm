//! GEMM 家族共享的轴契约与形状投影。
//!
//! `gemm_f32`、`gemm_i8`、`gemm_i4` 是同一副三轴骨架（`m`/`k` 输入、`n` 输出）的三种
//! 权重格式：轴的角色、收缩轴长度、形状投影完全相同，差异只在代价、样本与内核算术。本模块
//! 收口这部分共享事实；各格式仍保留自己的 `cost`/`sample`/`reference`，并按格式单态化，
//! 不在热路径上引入运行时分发。与 [`crate::gemv`] 的两轴骨架对称，只是这里多一条收缩轴。

use crate::model::{AxisRole, AxisSpec, ShapeContext};
use diagnostic::{Code, Error};

/// 收缩轴长度（固定档）。
pub const K: usize = 8;

/// 三轴：`m`/`k` 输入，`n` 输出。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'm',
    },
    AxisSpec {
        role: AxisRole::Input,
        label: 'k',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'n',
    },
];

/// 形状：输出 `m×n`，收缩轴 `k` 固定。
///
/// # Errors
///
/// 当前实现不会失败。
pub fn project_shape(m: usize, n: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('m', m), ('k', K), ('n', n)])
}

/// 从形状上下文取一条轴；缺轴时按 `op` 标签报 [`Code::Invalid`]。
///
/// 三份格式对轴的判读逐字相同，只有诊断里点名的算子名不同——标签作为参数传入，避免三份
/// 平行的取轴辅助各抄一遍。
pub fn axis(shape: &ShapeContext, label: char, op: &str) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `{label}`")))
}
