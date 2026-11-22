//! 声明表解析：`[op]`/`[axes]`/`[needs]`/`[layouts]`/`[grid]`/`[chunk]`/`[consts]` 各自一块。
use diagnostic::data::{Table, Value};

use spec::sch::layout::LayoutSpec;
use crate::atom::Need;

use super::build::parse_need;
use super::variant::variants;
use super::{Axis, Chunk, Grid, Phase, Skeleton, Variant, field};

/// 算子名：`[op].name`，非空。
pub fn op_name(op: &Table) -> diagnostic::Result<String> {
    let name = op.string("name").ok_or_else(|| field("op.name", "缺失"))?;
    if name.is_empty() {
        return Err(field("op.name", "不能为空"));
    }
    Ok(name.to_string())
}

/// `[op]` 段读出的身份：名字、骨架、摘要。
pub fn head(table: &Table) -> diagnostic::Result<(String, Skeleton, String)> {
    let op = table
        .table("op")
        .ok_or_else(|| field("op", "缺少 `[op]` 段"))?;
    let (skeleton, summary) = op_head(op)?;
    Ok((op_name(op)?, skeleton, summary))
}

/// `[op]` 的其余两键：骨架必须落在闭集内，摘要缺省为空串占位的 `-`。
pub fn op_head(op: &Table) -> diagnostic::Result<(Skeleton, String)> {
    let skeleton = Skeleton::parse(
        op.string("skeleton")
            .ok_or_else(|| field("op.skeleton", "缺失"))?,
    )
    .ok_or_else(|| {
        let allowed: Vec<&str> = Skeleton::ALL.iter().map(|item| item.as_str()).collect();
        field("op.skeleton", format!("闭集：{}", allowed.join(" | ")))
    })?;
    Ok((skeleton, op.string("summary").unwrap_or("-").to_string()))
}

/// 四个段（轴、需求、布局、变体链）：`declared` 按同一顺序解构。
pub type Sections = (
    Vec<Axis>,
    Vec<Need>,
    Vec<(String, LayoutSpec)>,
    Vec<Variant>,
);

pub fn sections(table: &Table) -> diagnostic::Result<Sections> {
    Ok((
        axes(table)?,
        needs(table)?,
        layouts(table)?,
        variants(table)?,
    ))
}

/// `[axes]`：每个键一组正整数取值；至少要有一条轴。
pub fn axes(table: &Table) -> diagnostic::Result<Vec<Axis>> {
    let Some(section) = table.table("axes") else {
        return Err(field("axes", "至少需要一个轴"));
    };
    let out: Vec<Axis> = section
        .entries()
        .iter()
        .map(|(key, value)| axis(key, value))
        .collect::<diagnostic::Result<_>>()?;
    if out.is_empty() {
        return Err(field("axes", "至少需要一个轴"));
    }
    Ok(out)
}

/// 一条轴：`axes.{名字}` 是非空的正整数数组。
pub fn axis(key: &str, value: &Value) -> diagnostic::Result<Axis> {
    let path = format!("axes.{key}");
    let Value::Array(items) = value else {
        return Err(field(&path, "必须是数组"));
    };
    let values = items
        .iter()
        .map(|item| positive(&path, item))
        .collect::<diagnostic::Result<Vec<u32>>>()?;
    if values.is_empty() {
        return Err(field(&path, "数组不能为空"));
    }
    Ok(Axis {
        name: key.to_string(),
        values,
    })
}

/// 轴的一个取值：正整数。
pub fn positive(path: &str, item: &Value) -> diagnostic::Result<u32> {
    match item {
        Value::Int(number) if *number > 0 => Ok(*number as u32),
        _ => Err(field(path, "每个取值必须是正整数")),
    }
}

/// `[needs]`：算子级公共需求，每条一个原语的能力签名。
pub fn needs(table: &Table) -> diagnostic::Result<Vec<Need>> {
    let Some(section) = table.table("needs") else {
        return Ok(Vec::new());
    };
    section
        .entries()
        .iter()
        .map(|(key, _)| parse_need(section, key, &format!("needs.{key}")))
        .collect()
}

/// `[layouts]`：槽位名 → 布局闭集里的一个词。
pub fn layouts(table: &Table) -> diagnostic::Result<Vec<(String, LayoutSpec)>> {
    let Some(section) = table.table("layouts") else {
        return Ok(Vec::new());
    };
    section
        .entries()
        .iter()
        .map(|(key, value)| layout(key, value))
        .collect()
}

/// 一条布局：`layouts.{槽位}` 必须是布局闭集内的字符串。
pub fn layout(key: &str, value: &Value) -> diagnostic::Result<(String, LayoutSpec)> {
    let path = format!("layouts.{key}");
    let Value::Str(text) = value else {
        return Err(field(&path, "必须是字符串"));
    };
    let spec = LayoutSpec::parse(text).map_err(|detail| field(&path, detail.message()))?;
    Ok((key.to_string(), spec))
}

/// `[grid]`：Gemm 骨架的网格映射（仅 Gemm 声明，其余骨架不需要）。缺省返回 `None`。
pub fn grid(table: &Table) -> diagnostic::Result<Option<Grid>> {
    let Some(section) = table.table("grid") else {
        return Ok(None);
    };
    Ok(Some(Grid {
        dims: grid_dims(section)?,
    }))
}

/// `grid.dims`：`"线程:标量"` 数组，至少一维、至多三维（Metal 网格上限）。
pub fn grid_dims(section: &Table) -> diagnostic::Result<Vec<(String, String)>> {
    let Some(value) = section.get("dims") else {
        return Err(field("grid.dims", "缺失"));
    };
    let Value::Array(items) = value else {
        return Err(field("grid.dims", "必须是数组"));
    };
    let dims = items
        .iter()
        .enumerate()
        .map(|(idx, item)| grid_item(idx, item))
        .collect::<diagnostic::Result<Vec<_>>>()?;
    check_grid_dims(&dims)?;
    Ok(dims)
}

/// 解析一维：`"线程:标量"`，两个名字都不能为空。
pub fn grid_item(idx: usize, item: &Value) -> diagnostic::Result<(String, String)> {
    let path = format!("grid.dims[{idx}]");
    let Value::Str(text) = item else {
        return Err(field(&path, "必须是 `线程:标量` 字符串"));
    };
    let Some((thread, scalar)) = text.split_once(':') else {
        return Err(field(&path, "必须是 `线程:标量` 字符串"));
    };
    if thread.is_empty() || scalar.is_empty() {
        return Err(field(&path, "线程名与标量名都不能为空"));
    }
    Ok((thread.to_string(), scalar.to_string()))
}

/// 网格维数约束：至少一维、至多三维（Metal 网格上限）。
pub fn check_grid_dims(dims: &[(String, String)]) -> diagnostic::Result<()> {
    if dims.is_empty() {
        return Err(field("grid.dims", "至少一维"));
    }
    if dims.len() > 3 {
        return Err(field("grid.dims", "至多三维（Metal 网格上限）"));
    }
    Ok(())
}

/// `[chunk]`：分块并行的块长（reduce/scan 骨架）。缺省返回 `None`（串行兜底）。
pub fn chunk(table: &Table) -> diagnostic::Result<Option<Chunk>> {
    let Some(section) = table.table("chunk") else {
        return Ok(None);
    };
    let size = section
        .int("size")
        .ok_or_else(|| field("chunk.size", "缺失"))?;
    if size < 1 {
        return Err(field("chunk.size", "至少 1"));
    }
    Ok(Some(Chunk {
        size: size as usize,
    }))
}

/// `[consts]`：模板常量的固定取值（非轴、非布局）。键给槽位名，值按十进制渲染成字符串，
/// 供 [`OpDecl::const_value`] 在 `{{槽位}}` 解析时查用。
pub fn consts(table: &Table) -> diagnostic::Result<Vec<(String, String)>> {
    let Some(section) = table.table("consts") else {
        return Ok(Vec::new());
    };
    section
        .entries()
        .iter()
        .map(|(key, value)| const_value(key, value))
        .collect()
}

/// 一个模板常量：整数/浮点/布尔按十进制渲染，字符串原样；表或数组不能当常量。
pub fn const_value(key: &str, value: &Value) -> diagnostic::Result<(String, String)> {
    let text = match value {
        Value::Str(text) => text.clone(),
        Value::Int(number) => number.to_string(),
        Value::Float(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        _ => return Err(field(&format!("consts.{key}"), "不能是表或数组")),
    };
    Ok((key.to_string(), text))
}

/// `[consts]` 与 `[grid]`、`[chunk]` 三块可选的补丁：`declared` 按同一顺序解构。
pub type Extras = (Vec<(String, String)>, Option<Grid>, Option<Chunk>);

/// `[consts]` 与 `[grid]`、`[chunk]`：声明表里三块可选的补丁，一起解析成 `(常量, 网格, 分块)`。
pub fn extras(table: &Table) -> diagnostic::Result<Extras> {
    Ok((consts(table)?, grid(table)?, chunk(table)?))
}

/// `[[phases]]`：数据流阶段（可选）。缺省 = 单阶段：名字取算子名，读/写槽位取自签名。
pub fn phases(
    table: &Table,
    op: &str,
    signature: &spec::sig::Signature,
) -> diagnostic::Result<Vec<Phase>> {
    let Some(rows) = table.table_array("phases") else {
        return Ok(vec![Phase {
            name: op.to_string(),
            reads: signature.buf_names(false),
            writes: signature.buf_names(true),
        }]);
    };
    rows.iter().map(|row| phase(op, row, signature)).collect()
}

/// 一条阶段：`name` 必填，`reads`/`writes` 是槽位名数组，每个名字都必须是签名的缓冲形参。
fn phase(op: &str, row: &Table, signature: &spec::sig::Signature) -> diagnostic::Result<Phase> {
    let name = row
        .string("name")
        .map(str::to_string)
        .ok_or_else(|| field(&format!("{op}.phases.name"), "缺失"))?;
    let reads = slot_names(op, row, "reads", signature)?;
    let writes = slot_names(op, row, "writes", signature)?;
    Ok(Phase {
        name,
        reads,
        writes,
    })
}

/// 阶段里的槽位名数组：每个名字都必须是签名的缓冲形参，否则拼装期就拦下（而不是漏缝）。
fn slot_names(
    op: &str,
    row: &Table,
    key: &str,
    signature: &spec::sig::Signature,
) -> diagnostic::Result<Vec<String>> {
    let Some(names) = row.string_array(key) else {
        return Ok(Vec::new());
    };
    for name in &names {
        let known = signature
            .params()
            .iter()
            .any(|param| param.ctype.is_buf() && param.name == *name);
        if !known {
            return Err(field(
                &format!("{op}.phases.{key}"),
                format!("`{name}` 不是签名的缓冲形参"),
            ));
        }
    }
    Ok(names.into_iter().map(str::to_string).collect())
}
