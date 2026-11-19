//! 轴展开：把 [`OpDecl`] 的轴取值笛卡尔积物化成 [`TileCombo`] 候选，或只算候选数。
//!
//! 与声明解析（`decl.rs`）拆开：解析只产出 [`OpDecl`]，展开是机制层在拿到声明之后的第一刀。

use diagnostic::Error;

use super::{field, Axis, OpDecl, TileCombo, MAX_TILES};

/// tile 候选数：轴取值的乘积，与 [`expand`] 同一口径但不物化组合。
///
/// # Errors
///
/// 乘积超过 [`MAX_TILES`] 时返回说明（与 [`expand`] 同一个判据与消息）。
pub fn tiles(op: &OpDecl) -> diagnostic::Result<usize> {
    let mut count = 1usize;
    for axis in &op.axes {
        count = count.saturating_mul(axis.values.len());
        if count > MAX_TILES {
            return Err(too_many_tiles(op, axis, count));
        }
    }
    Ok(count)
}

/// 轴展开：笛卡尔积，超过 [`MAX_TILES`] 即拒绝并说明该剪哪个轴。
pub fn expand(op: &OpDecl) -> diagnostic::Result<Vec<TileCombo>> {
    let mut combos = vec![TileCombo { values: Vec::new() }];
    for axis in &op.axes {
        let mut next = Vec::with_capacity(combos.len() * axis.values.len());
        for combo in &combos {
            for value in &axis.values {
                let mut values = combo.values.clone();
                values.push((axis.name.clone(), *value));
                next.push(TileCombo { values });
            }
        }
        if next.len() > MAX_TILES {
            return Err(too_many_tiles(op, axis, next.len()));
        }
        combos = next;
    }
    Ok(combos)
}

/// 候选超限的诊断：点名该剪哪个轴，以及它至少要让出几个取值。
fn too_many_tiles(op: &OpDecl, axis: &Axis, count: usize) -> Error {
    field(
        &format!("{}.axes", op.name),
        format!(
            "展开后有 {count} 个 tile 候选，超过上限 {MAX_TILES}；最少可剪掉轴 `{}` 的{}个取值",
            axis.name,
            axis.values.len().saturating_sub(1)
        ),
    )
}
