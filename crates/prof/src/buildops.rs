//! 模板表生成：共享算子契约 + 有效后端内核 → 清单与正文表。
//!
//! 算子声明由 `prof_ops!()` 编译期展开成强类型 `OpView`（见 `crate::OPS`）。本模块只生成正文：
//! 后端优先使用自己的 `<vendor>/kernels/<stem>.k`，未特化时沿设备档案的 `[kernels].base`
//! 继承参考实现；`src/kernels.toml` 以 stem 为键配置调优，不随语义声明的 `base` 扩散。
//! 因此“完整覆盖”不再等于“每个后端复制一份相同文件”。

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::{Device, buildgen, fail};

/// 循环展开锚点：正文标记位置，`kernels.toml` 决定因子。
const UNROLL_MARKER: &str = "/* cutile-unroll */";

/// 模板清单与后端正文表。
pub fn declarations_of(manifest: &Path, devices: &[Device]) -> (String, String) {
    declarations(manifest, devices).unwrap_or_else(|error| fail(format!("算子模板表：{error}")))
}

/// 两份产物：`for_each_template!` 的清单与按 `(vendor, 算子名)` 排序的后端正文表。
fn declarations(manifest: &Path, devices: &[Device]) -> diagnostic::Result<(String, String)> {
    let src = manifest.join("src");
    let ops = src.join("ops");
    println!("cargo:rerun-if-changed={}", ops.display());
    let files = buildgen::collect(&ops)?;
    let entries = scan(&ops, manifest, &files)?;
    let tuning = tuning_of(
        buildgen::read(&src.join("kernels.toml"))?,
        devices,
        &entries,
    );
    Ok((
        emit_list(&entries),
        emit_templates(&src, devices, &entries, &tuning),
    ))
}

/// 一份展开后的算子正文配置。
struct Entry {
    name: String,
    stem: String,
    tier: Option<String>,
}

/// 全局内核调优：模板 stem →（`default` 或 vendor → 展开因子）。
#[derive(Default)]
struct Tuning {
    unroll: BTreeMap<String, BTreeMap<String, u32>>,
}

impl Tuning {
    /// 后端专用因子优先于 `default`；零与缺省都表示不展开。
    fn unroll_for(&self, stem: &str, vendor: &str) -> Option<u32> {
        let config = self.unroll.get(stem)?;
        config
            .get(vendor)
            .or_else(|| config.get("default"))
            .copied()
            .filter(|factor| *factor > 0)
    }
}

/// 逐文件扫描并展开共享声明 `base` 与 `tiers` 档；排序供生成表二分查找。
fn scan(ops: &Path, manifest: &Path, files: &[PathBuf]) -> diagnostic::Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for path in files {
        println!(
            "cargo:rerun-if-changed={}",
            buildgen::relative(manifest, path)
        );
        entries.append(&mut entries_for_file(ops, path)?);
    }
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(entries)
}

/// 读入并解析（含共享声明 `base` 继承）一份档案表。
fn resolved_table(ops: &Path, path: &Path) -> diagnostic::Result<toml::Table> {
    resolve(ops, buildgen::read(path)?)
}

/// 单份档案展开出的档条目；模板 stem 随共享声明继承。
fn entries_for_file(ops: &Path, path: &Path) -> diagnostic::Result<Vec<Entry>> {
    let name = file_stem(path);
    let table = resolved_table(ops, path)?;
    let stem = template_stem(&table, name);
    let entries = expand_tiers(name, &table)
        .into_iter()
        .map(|(name, tier)| Entry {
            name,
            stem: stem.clone(),
            tier,
        })
        .collect();
    Ok(entries)
}

/// 按 `[op].tiers` 展开名字与档名；声明本体由 `prof_ops!()` 展开，这里不克隆或改写整棵树。
fn expand_tiers(name: &str, table: &toml::Table) -> Vec<(String, Option<String>)> {
    let op = table.get("op").and_then(toml::Value::as_table);
    let tiers = op
        .and_then(|op| op.get("tiers"))
        .and_then(toml::Value::as_array);
    let Some(tiers) = tiers else {
        return vec![(name.to_string(), None)];
    };
    let algorithm = op
        .and_then(|op| op.get("name"))
        .and_then(toml::Value::as_str)
        .unwrap_or(name);
    let algorithm = algorithm
        .rsplit_once('_')
        .map_or(algorithm, |(head, _)| head);
    let mut entries = Vec::new();
    for tier in tiers {
        let tier = tier
            .as_str()
            .unwrap_or_else(|| fail(format!("`{name}.toml` 的 tiers 元素必须是字符串档名")));
        entries.push((format!("{algorithm}_{tier}"), Some(tier.to_string())));
    }
    entries
}

/// 文件名 stem；生成代码需要合法的 UTF-8 字符串。
fn file_stem(path: &Path) -> &str {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_else(|| fail(format!("`{}` 的文件名不是合法 UTF-8", path.display())))
}

/// 一个算子的模板名：`[op].stem` 缺省为算子名。
fn template_stem(table: &toml::Table, name: &str) -> String {
    table
        .get("op")
        .and_then(toml::Value::as_table)
        .and_then(|op| op.get("stem"))
        .and_then(toml::Value::as_str)
        .unwrap_or(name)
        .to_string()
}

/// 解析并严格校验 `kernels.toml`，拒绝未知段、模板、后端与非整数因子。
fn tuning_of(table: toml::Table, devices: &[Device], entries: &[Entry]) -> Tuning {
    validate_top_keys(&table);
    let mut tuning = Tuning::default();
    let Some(unroll) = table.get("unroll") else {
        return tuning;
    };
    let unroll = unroll
        .as_table()
        .unwrap_or_else(|| fail("`kernels.toml.unroll` 必须是表"));
    for (stem, config) in unroll {
        let factors = parse_stem_unroll(stem, config, devices, entries);
        tuning.unroll.insert(stem.clone(), factors);
    }
    tuning
}

/// 顶层只允许 `[unroll]`；其余键属于未登记的调优族。
fn validate_top_keys(table: &toml::Table) {
    for key in table.keys() {
        if key != "unroll" {
            fail(format!("`kernels.toml.{key}` 未知：目前只支持 `[unroll]`"));
        }
    }
}

/// 解析单份模板的展开配置：先确认 stem 已声明，再逐后端取因子。
fn parse_stem_unroll(
    stem: &str,
    config: &toml::Value,
    devices: &[Device],
    entries: &[Entry],
) -> BTreeMap<String, u32> {
    if !entries.iter().any(|entry| entry.stem == stem) {
        fail(format!("`kernels.toml.unroll.{stem}` 不是已声明的模板"));
    }
    let config = config
        .as_table()
        .unwrap_or_else(|| fail(format!("`kernels.toml.unroll.{stem}` 必须是内联表")));
    let mut factors = BTreeMap::new();
    for (backend, value) in config {
        let factor = parse_backend_factor(stem, backend, value, devices);
        factors.insert(backend.clone(), factor);
    }
    factors
}

/// 解析单个后端因子：后端必须是 `default` 或已登记 vendor，因子为非负整数。
fn parse_backend_factor(stem: &str, backend: &str, value: &toml::Value, devices: &[Device]) -> u32 {
    if backend != "default" && !devices.iter().any(|device| device.name == backend) {
        fail(format!(
            "`kernels.toml.unroll.{stem}.{backend}` 不是已登记后端或 `default`"
        ));
    }
    let factor = value.as_integer().unwrap_or_else(|| {
        fail(format!(
            "`kernels.toml.unroll.{stem}.{backend}` 必须是非负整数"
        ))
    });
    u32::try_from(factor).unwrap_or_else(|_| {
        fail(format!(
            "`kernels.toml.unroll.{stem}.{backend}` 必须在 0..={} 之间",
            u32::MAX
        ))
    })
}

/// `template_list.rs`：算子名与模板 stem 的宏清单，顺序与 [`crate::OPS`] 一致。
fn emit_list(entries: &[Entry]) -> String {
    let mut list = String::from(
        "// @generated by crates/prof/build.rs — do not edit.\n\
         // Source: crates/prof/src/ops/*.toml (`[op].stem`), sorted by operator name.\n",
    );
    list.push_str("$m! {\n");
    for entry in entries {
        let _ = writeln!(list, "    {} => \"{}\",", entry.name, entry.stem);
    }
    list.push_str("}\n");
    list
}

/// `templates_table.rs`：CPU 兼容表 + `(vendor, 名字) → 有效后端正文` 的完整矩阵。
fn emit_templates(src: &Path, devices: &[Device], entries: &[Entry], tuning: &Tuning) -> String {
    let mut table = String::from(
        "// @generated by crates/prof/build.rs — do not edit.\n\
         // Sources: backend overrides and their configured reference kernels.\n\
         /// CPU 算子名与模板正文，顺序与 `crate::for_each_template!` 一致。\n",
    );
    let _ = writeln!(
        table,
        "pub const TEMPLATES: [(&str, &str); {}] = [",
        entries.len()
    );
    let cpu = devices
        .iter()
        .find(|device| device.name == "cpu")
        .unwrap_or_else(|| fail("设备档案缺少 `cpu`：主机参考内核无处登记"));
    emit_entries(&mut table, src, devices, cpu, entries, tuning, false);
    table.push_str("];\n\n/// `(vendor, 算子名)` 与有效后端模板正文，按二元组升序。\n");
    let _ = writeln!(
        table,
        "pub static DEVICE_TEMPLATES: [(&str, &str, &str); {}] = [",
        entries.len() * devices.len()
    );
    for device in devices {
        emit_entries(&mut table, src, devices, device, entries, tuning, true);
    }
    table.push_str("];\n");
    table
}

/// 发射一台设备的全部算子；无需换类型或展开时直接 `include_str!`，不复制正文。
fn emit_entries(
    table: &mut String,
    src: &Path,
    devices: &[Device],
    device: &Device,
    entries: &[Entry],
    tuning: &Tuning,
    qualified: bool,
) {
    for entry in entries {
        let path = kernel_path(src, devices, device, &entry.stem);
        let prefix = if qualified {
            format!("(\"{}\", \"{}\", ", device.name, entry.name)
        } else {
            format!("(\"{}\", ", entry.name)
        };
        let unroll = tuning.unroll_for(&entry.stem, &device.name);
        match (entry.tier.as_deref(), unroll) {
            (None | Some("f32"), None) => {
                let _ = writeln!(table, "    {prefix}include_str!({:?})),", path.display());
            }
            (tier, _) => {
                let text = read(&path);
                let text = apply_unroll(text, unroll, &path);
                let text = match tier {
                    None | Some("f32") => text,
                    Some(tier) => substitute_ctype(&text, tier),
                };
                let _ = writeln!(table, "    {prefix}{text:?}),");
            }
        }
    }
}

/// 解析有效内核路径：本后端文件优先，否则沿 `[kernels].base` 查找参考实现。
///
/// 缺失基座、未知后端与继承环都在构建期失败；诊断携带完整查找链，不静默换实现。
fn kernel_path(src: &Path, devices: &[Device], device: &Device, stem: &str) -> PathBuf {
    let mut current = device;
    let mut chain = vec![current.name.as_str()];
    loop {
        let path = src
            .join(&current.name)
            .join("kernels")
            .join(format!("{stem}.k"));
        if path.is_file() {
            return path;
        }
        let Some(base) = current.kernel_base.as_deref() else {
            fail(format!(
                "算子模板 `{stem}.k` 在 `{}` 后端不存在，且未声明 `[kernels].base`",
                current.name
            ));
        };
        current = devices
            .iter()
            .find(|candidate| candidate.name == base)
            .unwrap_or_else(|| {
                fail(format!(
                    "`{}` 的 `[kernels].base = \"{base}\"` 指向未知后端",
                    current.name
                ))
            });
        if chain.contains(&current.name.as_str()) {
            fail(format!(
                "内核继承形成环：{} → {}",
                chain.join(" → "),
                current.name
            ));
        }
        chain.push(&current.name);
    }
}

/// 把唯一展开锚点替换为配置指令；配置了因子却没标位置属于模板错误。
fn apply_unroll(text: String, factor: Option<u32>, path: &Path) -> String {
    let markers = text.matches(UNROLL_MARKER).count();
    if markers > 1 {
        fail(format!(
            "`{}` 有 {markers} 个 `{UNROLL_MARKER}`：每份模板至多标注一个展开点",
            path.display()
        ));
    }
    let Some(factor) = factor else {
        return text;
    };
    if markers == 0 {
        fail(format!(
            "`{}` 配置了展开因子，却缺少 `{UNROLL_MARKER}` 锚点",
            path.display()
        ));
    }
    text.replacen(UNROLL_MARKER, &format!("#pragma unroll {factor}"), 1)
}

/// 读取一份微内核正文，I/O 错误直接终止构建并带上路径。
fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| fail(format!("读模板 `{}`：{error}", path.display())))
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

/// 把模板正文里独立单词 `float` 替换为本档 C 类型，不伤标识符与注释。
fn substitute_ctype(text: &str, tier: &str) -> String {
    let token = c_token(tier);
    if token == "float" {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (index, _) in text.match_indices("float") {
        let end = index + "float".len();
        if standalone(text, index, end) {
            out.push_str(&text[last..index]);
            out.push_str(token);
            last = end;
        }
    }
    out.push_str(&text[last..]);
    out
}

/// 字符是否属于 C 标识符（字母数字 / 下划线）。
fn is_ident(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// `float` 在 `index` 处是否为独立单词：前后都不接标识符字符。
fn standalone(text: &str, index: usize, end: usize) -> bool {
    let before = index == 0 || !text[..index].chars().next_back().is_some_and(is_ident);
    let after = end >= text.len() || !text[end..].chars().next().is_some_and(is_ident);
    before && after
}

/// 解析共享声明 `base = "<name>"`：当前表覆盖基座，同名子表递归合并。
fn resolve(ops: &Path, mut table: toml::Table) -> diagnostic::Result<toml::Table> {
    let Some(toml::Value::String(base)) = table.get("base").cloned() else {
        return Ok(table);
    };
    table.remove("base");
    let base = resolve(ops, buildgen::read(&ops.join(format!("{base}.toml")))?)?;
    Ok(merge(base, table))
}

/// 深度合并：`over` 覆盖 `base`，同名子表递归合并。
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
