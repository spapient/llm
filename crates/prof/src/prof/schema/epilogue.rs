//! 职责：写回尾块供给解析（`[[epilogue]]`）。
//!
//! 它和 `[[provides]]` 共享"见证即 `{ file, func, expect }`"这一条，但没有任何协商坐标：
//! 六种形式由 `spec::epilogue` 的语法写死，档案只回答"这台设备怎么实现它、证据到哪一级"。
use super::super::*;
use super::provide::witness_fields;
use super::witness::shape_details;

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
    Ok(EpiProvide {
        form: form_of(entry, index)?,
        impl_name: entry.string("impl").unwrap_or("-").to_string(),
        grade: grade_of(entry, index)?,
        witness: Some(witness_of_epi(entry, index)?),
    })
}

/// `form`：`spec::epilogue` 的六种写回形式之一，闭集之外的写法没有装配位置。
fn form_of(entry: &Table, index: usize) -> Result<Epilogue, ProfileError> {
    let at = epi_at(index, "form");
    closed::<Epilogue>(text(entry, "form", at)?, at)
}

/// `grade`：证据等级；等级要得起见证时，`epi_shape` 会再要 `witness`。
fn grade_of(entry: &Table, index: usize) -> Result<Grade, ProfileError> {
    let at = epi_at(index, "grade");
    closed::<Grade>(text(entry, "grade", at)?, at)
}

/// `witness`：与 `[[provides]]` 同一种内联表，只有路径的见证无法执行。
fn witness_of_epi(entry: &Table, index: usize) -> Result<Witness, ProfileError> {
    let Some(table) = entry.table("witness") else {
        return Err(missing(epi_at(index, "witness")));
    };
    witness_fields(table, |key| epi_witness_at(index, key))
}

/// 一条写回供给的形状：等级要得起见证就必须有，写了就必须有入口名与期望值。
pub(super) fn epi_shape(epi: &EpiProvide, index: usize) -> Vec<ProfileError> {
    let Some(witness) = epi.witness.as_ref() else {
        let grade = epi.grade;
        return if grade.is_verified() {
            let detail = format!("{grade} 要求 `witness = {{ file, func, expect }}`");
            vec![bad(epi_at(index, "witness"), detail)]
        } else {
            Vec::new()
        };
    };
    let make = |(key, detail): (&'static str, String)| bad(epi_witness_at(index, key), detail);
    shape_details(witness).into_iter().map(make).collect()
}

/// 六种形式各一条：形式重名时"这台设备怎么实现 bias"就没有唯一答案；
/// 见证入口重名时，通过的结论无法归属到哪一种形式。
pub(super) fn epi_duplicates(all: &[EpiProvide]) -> Vec<ProfileError> {
    let mut forms: BTreeMap<Epilogue, usize> = BTreeMap::new();
    let mut entries: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut errors = Vec::new();
    for (index, epi) in all.iter().enumerate() {
        let form = epi.form;
        if let Some(first) = forms.insert(form, index) {
            let detail = format!("`{form}` 已经在 epilogue[{first}] 声明过：六种形式各一条");
            errors.push(bad(epi_at(index, "form"), detail));
        }
        let Some(witness) = epi.witness.as_ref() else {
            continue;
        };
        let key = (witness.file.as_str(), witness.func.as_str());
        if let Some(first) = entries.insert(key, index) {
            let detail = format!(
                "见证 `{}` 同时被 epilogue[{first}] 与 epilogue[{index}] 引用：一条见证只能证明一种形式",
                witness.func
            );
            errors.push(bad(epi_witness_at(index, "func"), detail));
        }
    }
    errors
}
