//! 算子语料解析：`ops/*.toml` → [`OpModel`] 中间模型。
//!
//! 纯函数层：不接触 `proc_macro`。`base` 继承合并、`tiers` 档展开、表 → 模型的逐字段解码
//! 都在这里，脱离宏可独立单测；渲染只消费 [`OpModel`]，不回头读 TOML。

use std::fs;
use std::path::{Path, PathBuf};
use toml::Table;
use toml::Value;

use crate::decl::{
    AttrModel, AxisModel, ChunkModel, GridModel, NeedModel, OpModel, PhaseModel, SpanModel,
    VariantModel,
};

/// 扫 `ops` 目录解析全部算子；继承合并与档展开完成后按名升序。
pub(crate) fn collect(ops_dir: &Path) -> Result<Vec<OpModel>, String> {
    let mut entries: Vec<OpModel> = Vec::new();
    for path in toml_files(ops_dir)? {
        collect_file(ops_dir, &path, &mut entries)?;
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// 单份档案：继承合并后按档展开，逐档收进结果。
fn collect_file(dir: &Path, path: &Path, entries: &mut Vec<OpModel>) -> Result<(), String> {
    let file_stem = path_stem(path)?;
    let resolved = resolve(dir, read_table(path)?)?;
    push_tiers(&file_stem, &resolved, entries)
}

/// 把一份档案展开出的各档算子压入结果。
fn push_tiers(file_stem: &str, resolved: &Table, entries: &mut Vec<OpModel>) -> Result<(), String> {
    for (name, table) in expand_tiers(file_stem, resolved)? {
        entries.push(parse_model(&name, &table)?);
    }
    Ok(())
}

/// 目录下全部 `.toml` 文件，按文件名升序（与生成序一致）。
fn toml_files(ops_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(ops_dir)
        .map_err(|error| format!("读算子目录 `{}`：{error}", ops_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
        .collect();
    files.sort();
    Ok(files)
}

/// 读一份 TOML 为顶层表；非表顶层直接失败（不让它悄悄变成运行期错误）。
fn read_table(path: &Path) -> Result<Table, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("读 `{}`：{error}", path.display()))?;
    toml::from_str::<Table>(&text).map_err(|error| format!("解析 `{}`：{error}", path.display()))
}

/// 文件名 stem；不是合法 UTF-8 就在编译期失败（生成代码要写进 `&'static str`）。
fn path_stem(path: &Path) -> Result<String, String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_string)
        .ok_or_else(|| format!("`{}` 文件名不是合法 UTF-8", path.display()))
}

// ── `base` 继承 ──────────────────────────────────────────────────────────

/// 解析 `base = "<name>"`：先继承 `<name>.toml`，再被自己的键覆盖；继承链可多级。
fn resolve(dir: &Path, mut table: Table) -> Result<Table, String> {
    let Some(Value::String(base)) = table.remove("base") else {
        return Ok(table);
    };
    let base_table = resolve(dir, read_table(&dir.join(format!("{base}.toml")))?)?;
    Ok(merge(base_table, table))
}

/// 深度合并：`over` 覆盖 `base`，同名子表递归合并（只覆盖差的那一个叶子）。
fn merge(mut base: Table, over: Table) -> Table {
    for (key, value) in over {
        match (base.get_mut(&key), value) {
            (Some(Value::Table(inner)), Value::Table(update)) => {
                *inner = merge(std::mem::take(inner), update);
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
    base
}

// ── `tiers` 档展开 ─────────────────────────────────────────────────────────

/// 按 `[op].tiers` 把一份算法档案展开成逐档算子；无 `tiers` 时原样返回单条（名 = 文件名）。
fn expand_tiers(file_stem: &str, table: &Table) -> Result<Vec<(String, Table)>, String> {
    let Some(tiers) = op_section(table)?.get("tiers").and_then(Value::as_array) else {
        return Ok(vec![(file_stem.to_string(), table.clone())]);
    };
    let algo = algo_name(table, file_stem)?;
    expand_collect(&algo, table, tiers)
}

/// 逐档克隆档案、替换精度并压入结果。
fn expand_collect(
    algo: &str,
    table: &Table,
    tiers: &[Value],
) -> Result<Vec<(String, Table)>, String> {
    let mut out = Vec::new();
    for tier in tiers {
        push_tier(&mut out, tier, algo, table)?;
    }
    Ok(out)
}

/// 算法名：取 `op.name` 末段档名之前的前缀；缺省回退文件名。
fn algo_name(table: &Table, file_stem: &str) -> Result<String, String> {
    let algo = op_section(table)?
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(file_stem);
    Ok(algo
        .rsplit_once('_')
        .map_or(algo.to_string(), |(head, _)| head.to_string()))
}

/// 单档：克隆档案、替换 `dt` 精度、剥掉 `tiers` 并改名为 `<算法>_<档>`。
fn push_tier(
    out: &mut Vec<(String, Table)>,
    tier: &Value,
    algo: &str,
    table: &Table,
) -> Result<(), String> {
    let tier = tier
        .as_str()
        .ok_or_else(|| "`tiers` 元素必须是字符串档名".to_string())?;
    let mut expanded = table.clone();
    substitute_dt(&mut expanded, tier);
    if let Some(op) = expanded.get_mut("op").and_then(Value::as_table_mut) {
        op.remove("tiers");
        op.insert("name".into(), Value::String(format!("{algo}_{tier}")));
    }
    out.push((format!("{algo}_{tier}"), expanded));
    Ok(())
}

/// 递归把所有 `dt` 数组里的精度 token 替换为本档。
fn substitute_dt(table: &mut Table, tier: &str) {
    for (key, value) in table.iter_mut() {
        apply_substitution(key, value, tier);
    }
}

/// 按值类型分流：子表递归，`dt` 数组逐项替换，其余不动。
fn apply_substitution(key: &str, value: &mut Value, tier: &str) {
    match value {
        Value::Table(inner) => substitute_dt(inner, tier),
        Value::Array(arr) if key == "dt" => substitute_dt_array(arr, tier),
        _ => {}
    }
}

/// `dt` 数组：字符串精度项换成本档，其余形态保留。
fn substitute_dt_array(arr: &mut [Value], tier: &str) {
    for item in arr.iter_mut() {
        if item.as_str().is_some() {
            *item = Value::String(tier.to_string());
        }
    }
}

// ── 表 → 模型 ─────────────────────────────────────────────────────────────

/// 一份档案解析成算子模型；模板正文占位由 #46 接 `TEMPLATES` 表按名配对。
fn parse_model(name: &str, table: &Table) -> Result<OpModel, String> {
    let (skeleton, summary) = header_of(table, name)?;
    let (axes, needs) = parse_axes_needs(table)?;
    let (layouts, grid, chunk, consts) = parse_layout_geometry(table)?;
    let (variants, phases) = parse_tail(name, table)?;
    Ok(OpModel {
        name: name.to_string(),
        skeleton,
        summary,
        axes,
        needs,
        layouts,
        grid,
        chunk,
        consts,
        variants,
        template: String::new(),
        phases,
    })
}

/// `[op]` 段内骨架与摘要两个标量字段。
fn header_of(table: &Table, name: &str) -> Result<(String, String), String> {
    let op = op_section(table)?;
    let skeleton = op
        .get("skeleton")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("`{name}.op.skeleton` 缺失"))?
        .to_string();
    let summary = op
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("-")
        .to_string();
    Ok((skeleton, summary))
}

/// 轴与需求两组能力签名。
fn parse_axes_needs(table: &Table) -> Result<(Vec<AxisModel>, Vec<NeedModel>), String> {
    let axes = parse_axes(table)?;
    let needs = parse_needs(table)?;
    Ok((axes, needs))
}

/// 布局 / 网格 / 分块 / 常量四类几何字段的解码结果。
type Geometry = (
    Vec<(String, String)>,
    Option<GridModel>,
    Option<ChunkModel>,
    Vec<(String, String)>,
);

/// 布局 / 网格 / 分块 / 常量四类几何字段。
fn parse_layout_geometry(table: &Table) -> Result<Geometry, String> {
    let layouts = parse_layouts(table)?;
    let grid = parse_grid(table)?;
    let chunk = parse_chunk(table)?;
    let consts = parse_consts(table)?;
    Ok((layouts, grid, chunk, consts))
}

/// 降级链与阶段列表。
fn parse_tail(name: &str, table: &Table) -> Result<(Vec<VariantModel>, Vec<PhaseModel>), String> {
    let variants = parse_variants(name, table)?;
    let phases = parse_phases(name, table)?;
    Ok((variants, phases))
}

/// `[op]` 段；缺失时报带算子名的说明。
fn op_section(table: &Table) -> Result<&Table, String> {
    table
        .get("op")
        .and_then(Value::as_table)
        .ok_or_else(|| "缺少 `[op]` 段".to_string())
}

/// `[axes]`：每个键一组正整数取值。
fn parse_axes(table: &Table) -> Result<Vec<AxisModel>, String> {
    let Some(section) = table.get("axes").and_then(Value::as_table) else {
        return Ok(Vec::new());
    };
    section
        .iter()
        .map(|(key, value)| parse_axis(key, value))
        .collect()
}

/// 单条轴：数组里逐项解析正整数。
fn parse_axis(key: &str, value: &Value) -> Result<AxisModel, String> {
    let items = value
        .as_array()
        .ok_or_else(|| format!("`axes.{key}` 必须是数组"))?;
    let values = items
        .iter()
        .map(parse_axis_value)
        .collect::<Result<Vec<u32>, String>>()?;
    Ok(AxisModel {
        name: key.to_string(),
        values,
    })
}

/// 单个轴取值：正整数。
fn parse_axis_value(item: &Value) -> Result<u32, String> {
    item.as_integer()
        .and_then(|number| u32::try_from(number).ok())
        .filter(|number| *number > 0)
        .ok_or_else(|| "`axes` 每个取值必须是正整数".to_string())
}

/// `[needs]`：算子级公共需求，每条一个原语的能力签名。
fn parse_needs(table: &Table) -> Result<Vec<NeedModel>, String> {
    let Some(section) = table.get("needs").and_then(Value::as_table) else {
        return Ok(Vec::new());
    };
    section
        .iter()
        .map(|(prim, detail)| parse_need(prim, detail))
        .collect()
}

/// 一条需求：维度键挂区间，`dt`/`space` 是集合，其余是能力属性。
fn parse_need(prim: &str, detail: &Value) -> Result<NeedModel, String> {
    let table = detail
        .as_table()
        .ok_or_else(|| format!("`needs.{prim}` 必须是内联表"))?;
    let mut need = NeedModel {
        prim: prim.to_string(),
        extent: SpanModel::Any,
        n: SpanModel::Any,
        k: SpanModel::Any,
        dt: Vec::new(),
        space: Vec::new(),
        attrs: Vec::new(),
    };
    for (name, value) in table {
        apply_field(&mut need, name, value)?;
    }
    Ok(need)
}

/// 字段分流：区间维度 → 槽位，集合键 → 列表，其余进属性。
fn apply_field(need: &mut NeedModel, name: &str, value: &Value) -> Result<(), String> {
    if is_span_key(name) {
        return apply_span_field(need, name, value);
    }
    if is_list_key(name) {
        return apply_list_field(need, name, value);
    }
    apply_attr(need, name, value)
}

/// 维度键是否挂区间槽位。
fn is_span_key(name: &str) -> bool {
    matches!(name, "m" | "bytes" | "extent" | "n" | "k")
}

/// 集合键是否挂字符串列表。
fn is_list_key(name: &str) -> bool {
    matches!(name, "dt" | "space")
}

/// 区间字段分派到对应槽位。
fn apply_span_field(need: &mut NeedModel, name: &str, value: &Value) -> Result<(), String> {
    let slot = match name {
        "m" | "bytes" | "extent" => &mut need.extent,
        "n" => &mut need.n,
        "k" => &mut need.k,
        _ => return Ok(()),
    };
    *slot = parse_span(value)?;
    Ok(())
}

/// 集合字段分派到对应列表。
fn apply_list_field(need: &mut NeedModel, name: &str, value: &Value) -> Result<(), String> {
    match name {
        "dt" => need.dt = names(value)?,
        "space" => need.space = names(value)?,
        _ => {}
    }
    Ok(())
}

/// 兜底进自由属性列表。
fn apply_attr(need: &mut NeedModel, name: &str, value: &Value) -> Result<(), String> {
    need.attrs.push(AttrModel {
        key: name.to_string(),
        values: names(value)?,
    });
    Ok(())
}

/// `[layouts]`：槽位名 → 布局写法原文。
fn parse_layouts(table: &Table) -> Result<Vec<(String, String)>, String> {
    let Some(section) = table.get("layouts").and_then(Value::as_table) else {
        return Ok(Vec::new());
    };
    section
        .iter()
        .map(|(slot, value)| {
            let spec = value
                .as_str()
                .ok_or_else(|| format!("`layouts.{slot}` 必须是字符串"))?;
            Ok((slot.clone(), spec.to_string()))
        })
        .collect()
}

/// `[grid]`：Gemm 网格映射；缺省返回 `None`。
fn parse_grid(table: &Table) -> Result<Option<GridModel>, String> {
    let Some(section) = table.get("grid").and_then(Value::as_table) else {
        return Ok(None);
    };
    let dims = section
        .get("dims")
        .and_then(Value::as_array)
        .ok_or_else(|| "`grid.dims` 缺失".to_string())?;
    let pairs = grid_pairs(dims)?;
    Ok(Some(GridModel { dims: pairs }))
}

/// 逐条解析网格项。
fn grid_pairs(dims: &[Value]) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    for item in dims {
        pairs.push(parse_grid_pair(item)?);
    }
    Ok(pairs)
}

/// 单条网格项：`线程:标量`，两侧都不能为空。
fn parse_grid_pair(item: &Value) -> Result<(String, String), String> {
    let text = item
        .as_str()
        .ok_or_else(|| "`grid.dims` 元素必须是字符串".to_string())?;
    let (thread, scalar) =
        split_grid(text).ok_or_else(|| format!("`grid.dims` 项 `{text}` 必须是 `线程:标量`"))?;
    Ok((thread.to_string(), scalar.to_string()))
}

/// 拆 `线程:标量`；任一侧为空则视为写法非法。
fn split_grid(text: &str) -> Option<(&str, &str)> {
    let (thread, scalar) = text.split_once(':')?;
    (!thread.is_empty() && !scalar.is_empty()).then_some((thread, scalar))
}

/// `[chunk]`：分块并行块长；缺省返回 `None`。
fn parse_chunk(table: &Table) -> Result<Option<ChunkModel>, String> {
    let Some(section) = table.get("chunk").and_then(Value::as_table) else {
        return Ok(None);
    };
    let size = section
        .get("size")
        .and_then(Value::as_integer)
        .ok_or_else(|| "`chunk.size` 缺失".to_string())?;
    let size = usize::try_from(size).map_err(|_| "`chunk.size` 不是合法块长".to_string())?;
    Ok(Some(ChunkModel { size }))
}

/// `[consts]`：模板常量；标量按十进制渲染成字符串，表 / 数组不能当常量。
fn parse_consts(table: &Table) -> Result<Vec<(String, String)>, String> {
    let Some(section) = table.get("consts").and_then(Value::as_table) else {
        return Ok(Vec::new());
    };
    section
        .iter()
        .map(|(key, value)| {
            let text = scalar_text(value)?;
            Ok((key.clone(), text))
        })
        .collect()
}

/// `[[variants]]`：降级链；`extra_needs` 先收进模型，渲染空切片占位待后续补全。
fn parse_variants(name: &str, table: &Table) -> Result<Vec<VariantModel>, String> {
    let Some(rows) = table.get("variants").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    rows.iter()
        .map(|row| parse_variant_row(name, row))
        .collect()
}

/// 单条变体行解析。
fn parse_variant_row(name: &str, row: &Value) -> Result<VariantModel, String> {
    let row = row
        .as_table()
        .ok_or_else(|| format!("`{name}.variants` 元素必须是表"))?;
    let vname = variant_name(name, row)?;
    let order = variant_order(name, row)?;
    let schedule = variant_schedule(row);
    let mma_share = variant_share(row);
    Ok(VariantModel {
        name: vname,
        order,
        schedule,
        mma_share,
        extra_needs: Vec::new(),
    })
}

/// 变体名；缺失时报带算子名的说明。
fn variant_name(name: &str, row: &Table) -> Result<String, String> {
    row.get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("`{name}.variants.name` 缺失"))
        .map(str::to_string)
}

/// 变体顺序号：非负整数。
fn variant_order(name: &str, row: &Table) -> Result<u32, String> {
    row.get("order")
        .and_then(Value::as_integer)
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| format!("`{name}.variants.order` 缺失"))
}

/// 变体调度写法；缺省直连。
fn variant_schedule(row: &Table) -> String {
    row.get("schedule")
        .and_then(Value::as_str)
        .unwrap_or("direct")
        .to_string()
}

/// `mma_share`：浮点优先，整数按浮点收；缺省 0。
fn variant_share(row: &Table) -> f64 {
    row.get("mma_share")
        .and_then(|value| {
            value
                .as_float()
                .or_else(|| value.as_integer().map(|number| number as f64))
        })
        .unwrap_or(0.0)
}

/// `[[phases]]`：显式数据流阶段；缺省空表，owned 化按签名派生单阶段。
fn parse_phases(name: &str, table: &Table) -> Result<Vec<PhaseModel>, String> {
    let Some(rows) = table.get("phases").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    rows.iter().map(|row| parse_phase_row(name, row)).collect()
}

/// 单条阶段行解析。
fn parse_phase_row(name: &str, row: &Value) -> Result<PhaseModel, String> {
    let row = row
        .as_table()
        .ok_or_else(|| format!("`{name}.phases` 元素必须是表"))?;
    let pname = phase_name(name, row)?;
    let (reads, writes) = phase_slots(row)?;
    Ok(PhaseModel {
        name: pname,
        reads,
        writes,
    })
}

/// 阶段名；缺失时报带算子名的说明。
fn phase_name(name: &str, row: &Table) -> Result<String, String> {
    row.get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("`{name}.phases.name` 缺失"))
        .map(str::to_string)
}

/// 阶段读 / 写槽位名列表。
fn phase_slots(row: &Table) -> Result<(Vec<String>, Vec<String>), String> {
    let reads = string_list(row.get("reads"))?;
    let writes = string_list(row.get("writes"))?;
    Ok((reads, writes))
}

// ── 值解码 ─────────────────────────────────────────────────────────────────

/// 一个标量或名字列表：标量按十进制渲染，数组逐项必须是字符串。
fn names(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "数组里只能是字符串".to_string())
            })
            .collect(),
        _ => scalar_text(value).map(|text| vec![text]),
    }
}

/// 一个标量的十进制文本：字符串原样，整数 / 浮点 / 布尔按十进制写。
fn scalar_text(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Integer(number) => Ok(number.to_string()),
        Value::Float(number) => Ok(number.to_string()),
        Value::Boolean(flag) => Ok(flag.to_string()),
        _ => Err("只接受标量".to_string()),
    }
}

/// 阶段槽位名数组；缺省为空。
fn string_list(value: Option<&Value>) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    match value {
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "数组里只能是字符串".to_string())
            })
            .collect(),
        Value::String(text) => Ok(vec![text.clone()]),
        _ => Err("必须是字符串或字符串数组".to_string()),
    }
}

/// 区间写法：`16` / `<=512` / `16..64` / `16..`；闭区间暂不可 const 构造，保留到模型层。
fn parse_span(value: &Value) -> Result<SpanModel, String> {
    let text = value
        .as_str()
        .ok_or_else(|| "区间必须是字符串（`16`、`16..64`、`<=512`）".to_string())?;
    span_from_text(text.trim())
}

/// 区间文本分派：上界前缀优先，其余按 `..` 切分。
fn span_from_text(text: &str) -> Result<SpanModel, String> {
    if let Some(rest) = text.strip_prefix("<=") {
        return Ok(SpanModel::AtMost(parse_u32(rest)?));
    }
    span_range(text)
}

/// 含 `..` 的区间：无上界为 AtLeast，两侧齐全为 Range，否则按精确值。
fn span_range(text: &str) -> Result<SpanModel, String> {
    match text.split_once("..") {
        Some((lo, hi)) => range_tail(lo, hi),
        None => Ok(SpanModel::Exact(parse_u32(text)?)),
    }
}

/// 已切出上下界：上界为空为 AtLeast，否则闭区间。
fn range_tail(lo: &str, hi: &str) -> Result<SpanModel, String> {
    if hi.trim().is_empty() {
        return Ok(SpanModel::AtLeast(parse_u32(lo)?));
    }
    span_range_pair(lo, hi)
}

/// 闭区间两段各自解析。
fn span_range_pair(lo: &str, hi: &str) -> Result<SpanModel, String> {
    Ok(SpanModel::Range(parse_u32(lo)?, parse_u32(hi)?))
}

/// 一个非负十进制整数。
fn parse_u32(text: &str) -> Result<u32, String> {
    text.trim()
        .parse::<u32>()
        .map_err(|_| format!("`{text}` 不是合法的非负整数"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一份完整算子表逐字段解码成模型。
    #[test]
    fn parses_a_full_op_table() {
        let text = r#"
            [op]
            name = "add_f32"
            skeleton = "elementwise"
            summary = "y[n] = a[n] + b[n]"

            [axes]
            N = [32]

            [needs]
            load = { bytes = "<=4096", space = ["GM", "FAST"], src = ["GM"] }

            [layouts]
            a = "row"

            [consts]
            binop = "+"

            [[variants]]
            name = "direct"
            order = 1
            schedule = "direct"
            mma_share = 0.0
        "#;
        let table = toml::from_str::<Table>(text).expect("合法 TOML");
        let model = parse_model("add_f32", &table).expect("解码成功");

        assert_eq!(model.name, "add_f32");
        assert_eq!(model.skeleton, "elementwise");
        assert_eq!(model.summary, "y[n] = a[n] + b[n]");
        assert_eq!(model.axes[0].name, "N");
        assert_eq!(model.axes[0].values, [32]);

        let load = &model.needs[0];
        assert_eq!(load.prim, "load");
        assert_eq!(load.extent, SpanModel::AtMost(4096));
        assert_eq!(load.space, ["GM", "FAST"]);
        assert_eq!(load.attrs[0].key, "src");
        assert_eq!(load.attrs[0].values, ["GM"]);

        assert_eq!(model.layouts[0], ("a".to_string(), "row".to_string()));
        assert_eq!(model.consts[0], ("binop".to_string(), "+".to_string()));
        let direct = &model.variants[0];
        assert_eq!(direct.name, "direct");
        assert_eq!(direct.order, 1);
        assert_eq!(direct.schedule, "direct");
        assert_eq!(direct.mma_share, 0.0);
        assert!(model.grid.is_none());
        assert!(model.chunk.is_none());
        assert!(model.phases.is_empty());
    }

    /// 区间四种写法分别落到对应模型形态。
    #[test]
    fn parses_span_forms() {
        let span = |text: &str| parse_span(&Value::String(text.to_string())).expect("区间合法");
        assert_eq!(span("16"), SpanModel::Exact(16));
        assert_eq!(span("<=8"), SpanModel::AtMost(8));
        assert_eq!(span("16.."), SpanModel::AtLeast(16));
        assert_eq!(span("16..64"), SpanModel::Range(16, 64));
    }

    /// 深度合并：同名子表递归合并，标量被覆盖。
    #[test]
    fn merge_deep_merges_nested_tables() {
        let base = toml::from_str::<Table>("[inner]\nx = 1\na = 1\n").expect("base");
        let over = toml::from_str::<Table>("[inner]\ny = 2\nb = 2\n").expect("over");
        let merged = merge(base, over);
        let inner = merged
            .get("inner")
            .and_then(Value::as_table)
            .expect("inner 表");
        assert!(inner.contains_key("x"));
        assert!(inner.contains_key("y"));
        assert!(inner.contains_key("a"));
        assert!(inner.contains_key("b"));
    }

    /// 档展开：每档替换 `dt` 精度并改名 `<算法>_<档>`。
    #[test]
    fn expands_tiers_and_substitutes_dt() {
        let table = toml::from_str::<Table>(
            r#"
            [op]
            name = "abs"
            tiers = ["f32", "f16"]

            [needs]
            cvt = { dt = ["f32"] }
            "#,
        )
        .expect("合法 TOML");
        let expanded = expand_tiers("abs_f32", &table).expect("档展开");
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].0, "abs_f32");
        assert_eq!(expanded[1].0, "abs_f16");
        let f16 = &expanded[1].1;
        let cvt = f16.get("needs").and_then(Value::as_table).expect("needs");
        let dt = cvt.get("cvt").and_then(Value::as_table).expect("cvt");
        let dt = dt.get("dt").and_then(Value::as_array).expect("dt 数组");
        assert_eq!(dt[0].as_str(), Some("f16"));
    }
}
