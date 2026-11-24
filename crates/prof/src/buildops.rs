//! 算子声明表生成：`ops/*.toml` → `declared.rs` + `template_list.rs` + `templates_table.rs`。
//!
//! 产物只有一棵 `Table` 树、一份 `$m!{ 名字 => 模板 }` 清单与一张模板正文表（ADR 0022：
//! 运行期无解析器）；生成顺序按文件 stem 升序，所以模板清单与注册表的
//! 顺序都可由文件名推出。模板名（`stem`）由语料自己声明：`[op].stem` 缺省为文件名，共享一份
//! 模板的算子写显式 stem（`base` 继承自动传播）——"哪些算子用哪份模板"只住在 `ops/` 里，
//! 代码没有第二份清单。

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::{buildgen, fail};

/// 算子声明表：`ops/*.toml` → `$OUT_DIR/declared.rs` + `template_list.rs` + `templates_table.rs`。
pub fn declarations_of(manifest: &Path) -> (String, String, String) {
    declarations(manifest).unwrap_or_else(|error| fail(format!("算子声明表：{error}")))
}

/// 三份产物：声明表（`declared.rs`）、`for_each_template!` 的清单（`template_list.rs`，
/// 与 `ops/` 同源所以没有第二份手写清单）与模板正文表（`templates_table.rs`）。
fn declarations(manifest: &Path) -> diagnostic::Result<(String, String, String)> {
    let ops = manifest.join("src").join("ops");
    println!("cargo:rerun-if-changed={}", ops.display());
    let files = buildgen::collect(&ops)?;
    let (names, stems, rows, tiers) = scan(&ops, manifest, &files)?;
    Ok((
        emit_declared(&rows),
        emit_list(&names, &stems),
        emit_templates(&ops, &names, &stems, &tiers),
    ))
}

/// 单份展开条目：算子名 / 模板 stem / 表字面量行 / 档名。
struct Entry {
    name: String,
    stem: String,
    row: String,
    tier: Option<String>,
}

/// 展开产物四元组：名字表 / 模板 stem 表 / 行清单 / 档名表，三份同序。
type Generated = (Vec<String>, Vec<String>, String, Vec<Option<String>>);

/// 逐文件扫描：`(算子名, 模板 stem, 表字面量行)` 三份同序清单，展开 `base` 继承与 `tiers` 档后生成。
///
/// 一份档案声明 `[op].tiers = ["f32", "f16", …]` 时，按档展开成多个算子名 `<算法>_<档>`，
/// 共享同一份模板 stem，`dt` 叶子与 `[op].name` 随档替换——「加一档=加一行」的数据驱动档维度。
/// 无 `tiers` 的档案逐字走旧路径（名=文件名）。展开后全体按名升序排序，二分查找的「有序」
/// 不变量因此与档展开兼容（注册表与目录仍同序）。
fn scan(ops: &Path, manifest: &Path, files: &[PathBuf]) -> diagnostic::Result<Generated> {
    let entries = collect_entries(ops, manifest, files)?;
    Ok(flatten(entries))
}

/// 读入并解析（含 `base` 继承）一份档案表。
fn resolved_table(ops: &Path, path: &Path) -> diagnostic::Result<toml::Table> {
    resolve(ops, buildgen::read(path)?)
}

/// 逐档生成一条表字面量行。
fn tier_row(exp_name: &str, exp_table: &toml::Table) -> diagnostic::Result<String> {
    Ok(format!(
        "            (\"{exp_name}\", {}),",
        buildgen::emit(exp_table, exp_name)?
    ))
}

/// 单份档案展开出的档条目；继承解析、模板 stem 校验都在本助手内闭包。
fn entries_for_file(ops: &Path, path: &Path) -> diagnostic::Result<Vec<Entry>> {
    let name = file_stem(path);
    let table = resolved_table(ops, path)?;
    let stem = stem_of(&table, name, ops);
    let mut out: Vec<Entry> = Vec::new();
    for (exp_name, exp_table, tier) in expand_tiers(name, &table)? {
        let row = tier_row(&exp_name, &exp_table)?;
        out.push(Entry {
            name: exp_name,
            stem: stem.clone(),
            row,
            tier,
        });
    }
    Ok(out)
}

/// 遍历档案文件，逐档展开成行条目（保持文件序，稍后由 [`flatten`] 统一排序）。
fn collect_entries(
    ops: &Path,
    manifest: &Path,
    files: &[PathBuf],
) -> diagnostic::Result<Vec<Entry>> {
    let mut entries: Vec<Entry> = Vec::new();
    for path in files {
        println!(
            "cargo:rerun-if-changed={}",
            buildgen::relative(manifest, path)
        );
        let mut file_entries = entries_for_file(ops, path)?;
        entries.append(&mut file_entries);
    }
    Ok(entries)
}

/// 条目按名升序，汇总成名字表 / 模板表 / 行清单 / 档名表四份同序清单。
fn flatten(entries: Vec<Entry>) -> Generated {
    let mut entries = entries;
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut names = Vec::new();
    let mut stems = Vec::new();
    let mut tiers = Vec::new();
    let mut rows = String::new();
    for entry in entries {
        names.push(entry.name);
        stems.push(entry.stem);
        tiers.push(entry.tier);
        let _ = writeln!(rows, "{}", entry.row);
    }
    (names, stems, rows, tiers)
}

/// 按 `[op].tiers` 把一份算法档案展开成逐档算子表；无 `tiers` 时原样返回单条（名=文件名）。
///
/// 档名取 `<算法>_<档>`：算法名是 `[op].name`（缺省文件名）去掉既有 `_<档>` 后缀。每档把
/// `[op].name` 设为展开名、所有 `dt` 数组里的精度 token 替换为本档，其余声明逐字共享。
fn expand_tiers(
    name: &str,
    table: &toml::Table,
) -> diagnostic::Result<Vec<(String, toml::Table, Option<String>)>> {
    let tiers = table
        .get("op")
        .and_then(|value| value.as_table())
        .and_then(|op| op.get("tiers"))
        .and_then(|value| value.as_array());
    let Some(tiers) = tiers else {
        return Ok(vec![(name.to_string(), table.clone(), None)]);
    };
    let algo = table
        .get("op")
        .and_then(|value| value.as_table())
        .and_then(|op| op.get("name"))
        .and_then(|value| value.as_str())
        .unwrap_or(name);
    let algo = algo.rsplit_once('_').map_or(algo, |(head, _)| head);
    let mut out = Vec::new();
    for tier in tiers {
        let tier = tier
            .as_str()
            .unwrap_or_else(|| fail(format!("`{name}.toml` 的 tiers 元素必须是字符串档名")));
        let mut expanded = table.clone();
        substitute_dt(&mut expanded, tier);
        if let Some(op) = expanded.get_mut("op").and_then(|v| v.as_table_mut()) {
            op.remove("tiers");
            op.insert("name".into(), toml::Value::String(format!("{algo}_{tier}")));
        }
        out.push((format!("{algo}_{tier}"), expanded, Some(tier.to_string())));
    }
    Ok(out)
}

/// 递归把表里所有 `dt` 数组的精度 token 替换为 `tier`（`[needs].*.dt` 等叶子）。
fn substitute_dt(table: &mut toml::Table, tier: &str) {
    for (key, value) in table.iter_mut() {
        match value {
            toml::Value::Table(inner) => substitute_dt(inner, tier),
            toml::Value::Array(arr) if key == "dt" => replace_dt_tokens(arr, tier),
            _ => {}
        }
    }
}

/// 把 `dt` 数组里的字符串精度 token 整体换成本档；非字符串项原样保留。
fn replace_dt_tokens(arr: &mut [toml::Value], tier: &str) {
    for item in arr.iter_mut() {
        if item.as_str().is_some() {
            *item = toml::Value::String(tier.to_string());
        }
    }
}

/// 文件名 stem；不是合法 UTF-8 就构建期失败（生成代码里要写进 `&'static str`）。
fn file_stem(path: &Path) -> &str {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_else(|| fail(format!("`{}` 的文件名不是合法 UTF-8", path.display())))
}

/// `declared.rs`：`Table` 树字面量 + 二分查找。
fn emit_declared(rows: &str) -> String {
    let mut code = String::from(
        "// @generated by crates/prof/build.rs — do not edit.\n\
         // Source: crates/prof/ops/*.toml, sorted by file stem.\n",
    );
    code.push_str(crate::TABLE_REGISTRY_HEAD);
    code.push_str(rows);
    code.push_str(
        "        ]\n\
         \x20   })\n\
         }\n\n\
         /// 按名字取编译期生成的声明表；名字不在表里时返回 `None`。表按名升序（生成序），二分查找。\n\
         pub fn declared(name: &str) -> Option<&'static Table> {\n\
         \x20   let idx = tables().binary_search_by_key(&name, |(entry, _)| entry).ok()?;\n\
         \x20   Some(&tables()[idx].1)\n\
         }\n\n",
    );
    code
}

/// `template_list.rs`：`$m!{ 名字 => 模板, ... }` 宏清单，与 `names` 同序（文件名升序）。
fn emit_list(names: &[String], stems: &[String]) -> String {
    let mut list = String::from(
        "// @generated by crates/prof/build.rs — do not edit.\n\
         // Source: crates/prof/ops/*.toml (`[op].stem`), sorted by file stem.\n",
    );
    list.push_str("$m! {\n");
    for (name, stem) in names.iter().zip(stems) {
        let _ = writeln!(list, "    {name} => \"{stem}\",");
    }
    list.push_str("}\n");
    list
}

/// `templates_table.rs`：名字 → 模板正文，`include_str!` 用绝对路径（生成文件住在 `$OUT_DIR`）。
fn emit_templates(
    ops: &Path,
    names: &[String],
    stems: &[String],
    tiers: &[Option<String>],
) -> String {
    let mut table = String::from(
        "// @generated by crates/prof/build.rs — do not edit.\n\
         // Source: crates/prof/ops/<stem>.k, sorted by operator name.\n\
         /// 算子名与模板正文，顺序与 `crate::for_each_template!` 一致。\n",
    );
    let _ = writeln!(
        table,
        "pub const TEMPLATES: [(&str, &str); {}] = [",
        names.len()
    );
    for ((name, stem), tier) in names.iter().zip(stems).zip(tiers) {
        let path = ops.join(format!("{stem}.k"));
        // 无档（旧档案）与 f32 档逐字共享 `include_str!`；其余档把模板里的 C 浮点类型按档替换，
        // 正文内联进生成表——「加一档=加一行声明」，模板只有一份。
        match tier.as_deref() {
            None | Some("f32") => {
                let _ = writeln!(
                    table,
                    "    (\"{name}\", include_str!({:?})),",
                    path.display()
                );
            }
            Some(tier) => {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| fail(format!("读模板 `{}`：{error}", path.display())));
                let text = substitute_ctype(&text, tier);
                let _ = writeln!(table, "    (\"{name}\", {text:?}),");
            }
        }
    }
    table.push_str("];\n");
    table
}

/// 档名 → 模板里的 C 浮点类型 token（与 `sig::CType::c_name` 同一份口径）。
fn c_token(tier: &str) -> &'static str {
    match tier {
        "f16" => "half",
        "bf16" => "bf16",
        "f64" => "double",
        _ => "float",
    }
}

/// 把模板正文里**独立单词** `float` 替换为本档 C 类型（词边界判定，不伤 `_cutile_exp_f32`
/// 这类标识符与中文注释）；f32 档原样返回。半精度字面量 `0.0f` 由 C 隐式转换，无需改后缀。
fn substitute_ctype(text: &str, tier: &str) -> String {
    let token = c_token(tier);
    if token == "float" {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (idx, _) in text.match_indices("float") {
        let end = idx + "float".len();
        if standalone(text, idx, end) {
            out.push_str(&text[last..idx]);
            out.push_str(token);
            last = end;
        }
    }
    out.push_str(&text[last..]);
    out
}

/// 字符是否属于 C 标识符（字母数字 / 下划线）。
fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `float` 在 `idx` 处是否为独立单词：前后都不接标识符字符。
fn standalone(text: &str, idx: usize, end: usize) -> bool {
    let before = idx == 0 || !text[..idx].chars().next_back().is_some_and(is_ident);
    let after = end >= text.len() || !text[end..].chars().next().is_some_and(is_ident);
    before && after
}

/// 一个算子的模板名：`[op].stem` 缺省为算子名；共享模板（`binary`/`gemm_packed`…）在语料里
/// 显式声明。模板文件必须存在，写错在构建期就失败。
fn stem_of(table: &toml::Table, name: &str, ops: &Path) -> String {
    let stem = table
        .get("op")
        .and_then(|value| value.as_table())
        .and_then(|op| op.get("stem"))
        .and_then(|value| value.as_str())
        .unwrap_or(name)
        .to_string();
    let template = ops.join(format!("{stem}.k"));
    if !template.is_file() {
        fail(format!(
            "`{name}.toml` 声明的模板 `{stem}.k` 不存在：stem 必须指向 `ops/` 里的同名文件"
        ));
    }
    stem
}

/// 解析 `base = "<name>"`：当前表先继承 `<name>.toml`，再被自己的键覆盖。
///
/// 兄弟继承（如 `gemv_i2` → `gemv_i4`）只差 `[op]`/`[consts]`/个别 `[needs]` 叶子，
/// `[needs]`/`[layouts]`/`[[variants]]` 一大段逐字相同，写在 base 里由继承共享。继承链可多级
/// （如 `adaptive_pool_f32` → `silu_f32` → `gelu_f32`），每级先展开自己的 base 再被当前表覆盖。
/// 声明语料无环，递归深度 ≤ 3。
fn resolve(ops: &Path, mut table: toml::Table) -> diagnostic::Result<toml::Table> {
    let Some(toml::Value::String(base)) = table.get("base").cloned() else {
        return Ok(table);
    };
    table.remove("base");
    let base = resolve(ops, buildgen::read(&ops.join(format!("{base}.toml")))?)?;
    Ok(merge(base, table))
}

/// 深度合并：`over` 覆盖 `base`，同名的子表递归合并（`[needs]` 只覆盖差的那一个叶子）。
fn merge(mut base: toml::Table, over: toml::Table) -> toml::Table {
    for (key, value) in over {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(inner)), toml::Value::Table(update)) => {
                *inner = merge(std::mem::take(inner), update);
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
    base
}
