//! 职责：写回尾块供给解析（`[[epilogue]]`）。
//!
//! 它和 `[[provides]]` 共享"见证即 `{ file, func, expect }`"这一条，但没有任何协商坐标：
//! 六种形式由 `spec::epilogue` 的语法写死，档案只回答"这台设备怎么实现它、证据到哪一级"。
use super::super::*;
use super::provide::witness_fields;

/// `[[epilogue]]` 数组：缺省为空（旧档案不写这一段），写了就要六种形式齐全。
pub(super) fn parse_epilogues(table: &Table) -> Result<Vec<EpiProvide>, ProfileError> {
    let tables = table.table_array("epilogue").unwrap_or_default();
    tables
        .iter()
        .enumerate()
        .map(|(index, entry)| parse_epilogue(entry, index))
        .collect()
}

/// 一条 `[[epilogue]]`：`form` / `impl` / `grade` 必填，`witness` 也必填——构建期要用它的
/// 函数名当镜像入口，缺了这条供给就没有落点。
fn parse_epilogue(entry: &Table, index: usize) -> Result<EpiProvide, ProfileError> {
    let form = closed::<Epilogue>(text(entry, "form", epi_at(index, "form"))?, epi_at(index, "form"))?;
    let implementation = entry.string("impl").unwrap_or("-");
    let grade = closed::<Grade>(text(entry, "grade", epi_at(index, "grade"))?, epi_at(index, "grade"))?;
    let witness = witness_of_epi(entry, index)?;
    Ok(EpiProvide {
        form,
        impl_name: implementation.to_string(),
        grade,
        witness: Some(witness),
    })
}

/// `witness`：与 `[[provides]]` 同一种内联表，只有路径的见证无法执行。
fn witness_of_epi(entry: &Table, index: usize) -> Result<Witness, ProfileError> {
    let Some(table) = entry.table("witness") else {
        return Err(missing(epi_at(index, "witness")));
    };
    witness_fields(table, |key| epi_witness_at(index, key))
}
