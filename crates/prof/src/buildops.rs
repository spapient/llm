//! 模板表生成：共享算子契约 + 有效后端内核 → 清单与正文表。
//!
//! 算子声明由 `prof_ops!()` 编译期展开成强类型 `OpView`（见 `crate::OPS`）。本模块只生成正文：
//! 后端优先使用自己的 `<vendor>/<stem>.<ext>`，未特化时使用同目录的 `ops/<stem>.k`
//! 共享参考实现；`src/ops/kernels.toml` 以 stem 为键配置调优，不随语义声明的 `base` 扩散。
//! 声明与参考正文同属算子，设备目录只承担真实特化，完整覆盖不需要机械副本或继承配置。

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::{Device, buildgen, fail};

/// 循环展开锚点：正文标记位置，`kernels.toml` 决定因子。
const UNROLL_MARKER: &str = "/* cutile-unroll */";

/// Rust 常量与 C 模板槽位的唯一映射：槽位、微内核族、配置键。
const NATIVE_FIELDS: &[(&str, &str, &str)] = &[
    ("FMA_MR", "fma", "mr"),
    ("FMA_NR", "fma", "nr"),
    ("FMA_KB", "fma", "kb"),
    ("SDOT_MR", "sdot", "mr"),
    ("SDOT_NR", "sdot", "nr"),
    ("SDOT_CHAINS", "sdot", "chains"),
    ("SME_MR", "sme", "mr"),
    ("SME_NR", "sme", "nr"),
    ("SME_MIN_ELEMS", "sme", "min_elems"),
];

/// 模板清单与后端正文表。
pub fn template_tables_of(manifest: &Path, devices: &[Device]) -> (String, String) {
    render_template_tables(manifest, devices)
        .unwrap_or_else(|error| fail(format!("算子模板表：{error}")))
}

/// 两份产物：`for_each_template!` 的清单与按 `(vendor, 算子名)` 排序的后端正文表。
fn render_template_tables(
    manifest: &Path,
    devices: &[Device],
) -> diagnostic::Result<(String, String)> {
    let src = manifest.join("src");
    let ops = src.join("ops");
    println!("cargo:rerun-if-changed={}", ops.display());
    let files: Vec<_> = buildgen::collect(&ops)?
        .into_iter()
        .filter(|path| file_stem(path) != "kernels")
        .collect();
    let entries = scan(&ops, manifest, &files)?;
    let tuning = tuning_of(
        buildgen::read(&ops.join("kernels.toml"))?,
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
    /// 档案键优先于设备族，设备族优先于 `default`；零与缺省都表示不展开。
    fn unroll_for(&self, stem: &str, device: &Device) -> Option<u32> {
        let config = self.unroll.get(stem)?;
        config
            .get(&device.name)
            .or_else(|| config.get(&device.family))
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
    if let Some(duplicate) = entries.windows(2).find(|pair| pair[0].name == pair[1].name) {
        fail(format!("算子 `{}` 被重复声明", duplicate[0].name));
    }
    Ok(entries)
}

/// 读入并解析（含共享声明 `base` 继承）一份档案表。
fn resolved_table(ops: &Path, path: &Path) -> diagnostic::Result<toml::Table> {
    resolve(ops, buildgen::read(path)?)
}

/// 单份档案展开出的档条目；同族可共享公共段，每条只写自己的 `op` 与差异段。
fn entries_for_file(ops: &Path, path: &Path) -> diagnostic::Result<Vec<Entry>> {
    let name = file_stem(path);
    let table = resolved_table(ops, path)?;
    if !table.contains_key("op") && !table.contains_key("operators") {
        return Ok(Vec::new());
    }
    Ok(documents(name, &table)?
        .into_iter()
        .flat_map(|(name, table)| document_entries(name, &table))
        .collect())
}

/// 一条同族声明展开为带模板身份与档名的构建条目。
fn document_entries(name: String, table: &toml::Table) -> Vec<Entry> {
    let stem = template_stem(table, &name);
    expand_tiers(&name, table)
        .into_iter()
        .map(|(name, tier)| Entry {
            name,
            stem: stem.clone(),
            tier,
        })
        .collect()
}

/// 展开同族声明；`[op]` 与 `[[operators]]` 互斥，族内每条都必须有唯一职责名。
fn documents(
    file_stem: &str,
    table: &toml::Table,
) -> diagnostic::Result<Vec<(String, toml::Table)>> {
    if !table.contains_key("operators") {
        return Ok(vec![(file_stem.to_string(), table.clone())]);
    }
    let items = family_items(file_stem, table);
    let mut family = table.clone();
    family.remove("operators");
    Ok(items
        .iter()
        .map(|item| family_document(file_stem, &family, item))
        .collect())
}

/// 同族声明表：互斥关系、数组类型与非空约束集中在这里。
fn family_items<'a>(file_stem: &str, table: &'a toml::Table) -> &'a [toml::Value] {
    if table.contains_key("op") {
        fail(format!(
            "`{file_stem}` 不能同时包含 `[op]` 与 `[[operators]]`"
        ));
    }
    let items = table
        .get("operators")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| fail(format!("`{file_stem}.operators` 必须是算子表数组")));
    if items.is_empty() {
        fail(format!("`{file_stem}.operators` 不能为空"));
    }
    items
}

/// 展开一条同族声明，校验条目形状与唯一职责名。
fn family_document(
    file_stem: &str,
    family: &toml::Table,
    item: &toml::Value,
) -> (String, toml::Table) {
    let item = item
        .as_table()
        .unwrap_or_else(|| fail(format!("`{file_stem}.operators` 的每项都必须是表")));
    let name = item
        .get("op")
        .and_then(toml::Value::as_table)
        .and_then(|op| op.get("name"))
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| fail(format!("`{file_stem}.operators` 每项都必须声明 `op.name`")));
    (name.to_string(), merge(family.clone(), item.clone()))
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

/// 顶层闭集：设备模板 `[unroll]` 与生产微内核 `[gemm_micro]`。
fn validate_top_keys(table: &toml::Table) {
    for key in table.keys() {
        if !["unroll", "gemm_micro"].contains(&key.as_str()) {
            fail(format!("`kernels.toml.{key}` 不是已登记的调优族"));
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
         // Sources: backend overrides and shared ops reference kernels.\n\
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
    emit_entries(&mut table, src, cpu, entries, tuning, false);
    table.push_str("];\n\n/// `(vendor, 算子名)` 与有效后端模板正文，按二元组升序。\n");
    let _ = writeln!(
        table,
        "pub static DEVICE_TEMPLATES: [(&str, &str, &str); {}] = [",
        entries.len() * devices.len()
    );
    for device in devices {
        emit_entries(&mut table, src, device, entries, tuning, true);
    }
    table.push_str("];\n");
    table
}

/// 发射一台设备的全部算子；无需换类型或展开时直接 `include_str!`，不复制正文。
fn emit_entries(
    table: &mut String,
    src: &Path,
    device: &Device,
    entries: &[Entry],
    tuning: &Tuning,
    qualified: bool,
) {
    for entry in entries {
        let path = kernel_path(src, device, &entry.stem);
        let prefix = if qualified {
            format!("(\"{}\", \"{}\", ", device.name, entry.name)
        } else {
            format!("(\"{}\", ", entry.name)
        };
        let unroll = tuning.unroll_for(&entry.stem, device);
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

/// 解析有效内核路径：后端真实特化优先，否则使用 `ops/<stem>.k` 共享参考正文。
/// 两级查找无需 vendor 分支；两者都缺失时在构建期报告后端与参考路径。
fn kernel_path(src: &Path, device: &Device, stem: &str) -> PathBuf {
    let specialized = src
        .join(&device.family)
        .join(format!("{stem}.{}", device.ext));
    if specialized.is_file() {
        return specialized;
    }
    let reference = src.join("ops").join(format!("{stem}.k"));
    if !reference.is_file() {
        fail(format!(
            "算子模板 `{stem}.k` 无有效正文：`{}` 后端未特化，共享参考 `{}` 也不存在",
            device.name,
            reference.display()
        ));
    }
    reference
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
fn is_ident(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// `float` 在 `idx` 处是否为独立单词：前后都不接标识符字符。
fn standalone(text: &str, idx: usize, end: usize) -> bool {
    let before = idx == 0 || !text[..idx].chars().next_back().is_some_and(is_ident);
    let after = end >= text.len() || !text[end..].chars().next().is_some_and(is_ident);
    before && after
}

/// 解析共享声明 `base = "<name>"`：当前表覆盖基座，同名子表递归合并。
fn resolve(ops: &Path, mut table: toml::Table) -> diagnostic::Result<toml::Table> {
    let Some(toml::Value::String(base)) = table.get("base").cloned() else {
        return Ok(table);
    };
    table.remove("base");
    let mut base = resolve(ops, buildgen::read(&ops.join(format!("{base}.toml")))?)?;
    if table.contains_key("operators") {
        base.remove("op");
    }
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

// ── 主机 GEMM 微内核：配置 → 注解校验 → 静态 C 库 ─────────────────────

/// 构建 `gemm_micro.k`：同一正文供静态生产桥与既有验证 harness 使用。
pub fn build_native(manifest: &Path, out: &Path) {
    let ops = manifest.join("src").join("ops");
    let config = ops.join("kernels.toml");
    let template = ops.join("gemm_micro.k");
    println!("cargo:rerun-if-changed={}", config.display());
    println!("cargo:rerun-if-changed={}", template.display());
    let table = buildgen::read(&config).unwrap_or_else(|error| fail(format!("{error}")));
    let rendered = render_native(&read(&template), &table);
    let c = out.join("gemm_micro.c");
    std::fs::write(&c, rendered)
        .unwrap_or_else(|error| fail(format!("写 {} 失败：{error}", c.display())));
    let native = native_table(&table);
    let mut contract = String::new();
    for (slot, family, key) in NATIVE_FIELDS {
        let value = native_positive(native, family, key);
        let _ = writeln!(
            contract,
            "#[doc = \"gemm_micro 配置常量。\"] pub const {slot}: usize = {value};"
        );
    }
    contract.push_str("unsafe extern \"C\" {\n\
        #[cfg(not(all(target_arch = \"x86_64\", target_env = \"msvc\")))] fn cutile_fma_micro(a: *const f32, b: *const f32, out: *mut f32, row: i32, col: i32, n: i32, kb: i32, first: i32);\n\
        #[cfg(all(target_arch = \"x86_64\", target_env = \"msvc\"))] fn cutile_fma_micro_avx2(a: *const f32, b: *const f32, out: *mut f32, row: i32, col: i32, n: i32, kb: i32, first: i32);\n\
        #[cfg(target_arch = \"x86_64\")] fn cutile_fma_micro_scalar(a: *const f32, b: *const f32, out: *mut f32, row: i32, col: i32, n: i32, kb: i32, first: i32);\n\
        fn cutile_sdot_micro(a: *const i8, b: *const i8, out: *mut i32, row: i32, blk: i32, n: i32, k: i32, kvec: i32);\n\
        fn cutile_sme_micro(a: *const f32, b: *const f32, out: *mut f32, n: usize, k: usize);\n\
        fn cutile_sme_pack(a: *const f32, panel: *mut f32, k: usize, mblk0: usize, blocks: usize);\n\
        fn cutile_sme_stage(a: *const f32, b: *const f32, out: *mut f32, n: usize, k: usize, blocks: usize);\n\
    }\n\
    #[cfg(all(feature = \"bench\", target_arch = \"aarch64\"))] unsafe extern \"C\" {\n\
        fn cutile_fma_pure(rounds: usize);\n\
        fn cutile_fma_block(rounds: usize, a: *const f32, b: *const f32, k: i32);\n\
        fn cutile_sdot_pure(rounds: usize);\n\
        fn cutile_sdot_block(rounds: usize, a: *const i8, b: *const i8, k: i32);\n\
        fn cutile_sme_pure(rounds: usize);\n\
        fn cutile_sme_mix(rounds: usize, a: *const f32, b: *const f32);\n\
    }\n");
    contract.push_str(crate::sign::NATIVE_PROBE_RS);
    let contract_path = out.join("gemm_micro_contract.rs");
    std::fs::write(&contract_path, contract)
        .unwrap_or_else(|error| fail(format!("写 {} 失败：{error}", contract_path.display())));
    compile_native(&c);
}

/// MSVC 用同一正文分别裁剪基线和 AVX2 对象；其他工具链保留函数级 target 属性。
fn compile_native(c: &Path) {
    let msvc_x86 = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if msvc_x86 {
        cc::Build::new()
            .file(c)
            .opt_level(3)
            .define("CUTILE_SCALAR_ONLY", None)
            .compile("cutile_gemm_micro");
        cc::Build::new()
            .file(c)
            .opt_level(3)
            .define("CUTILE_AVX2_ONLY", None)
            .flag("/arch:AVX2")
            .compile("cutile_gemm_micro_avx2");
        return;
    }
    cc::Build::new()
        .file(c)
        .opt_level(3)
        .flag_if_supported("-std=c99")
        .compile("cutile_gemm_micro");
}

/// 生成三族配置常量并校验／渲染所有具名注解。
fn render_native(text: &str, table: &toml::Table) -> String {
    let native = native_table(table);
    validate_native_config(native);
    crate::sign::validate_native_annotations(text, fail);
    let out = replace_native_slots(native, text);
    if out.contains("{{") {
        fail("`gemm_micro.k` 仍有未登记的 `{{…}}` 槽位");
    }
    apply_native_unrolls(native, out)
}

fn native_table(table: &toml::Table) -> &toml::Table {
    table
        .get("gemm_micro")
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| fail("`kernels.toml.gemm_micro` 必须是表"))
}

fn validate_native_config(native: &toml::Table) {
    for (family, allowed) in [
        ("fma", &["mr", "nr", "kb", "unroll"][..]),
        ("sdot", &["mr", "nr", "chains", "unroll"][..]),
        ("sme", &["mr", "nr", "min_elems", "unroll"][..]),
    ] {
        validate_native_family(native, family, allowed);
    }
    for family in native.keys() {
        if !["fma", "sdot", "sme"].contains(&family.as_str()) {
            fail(format!("`gemm_micro.{family}` 不是已登记微内核族"));
        }
    }
}

fn validate_native_family(native: &toml::Table, family: &str, allowed: &[&str]) {
    let section = native
        .get(family)
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| fail("`gemm_micro.{family}` 必须是表"));
    for key in section.keys() {
        if !allowed.contains(&key.as_str()) {
            fail(format!("`gemm_micro.{family}.{key}` 不是已登记字段"));
        }
    }
    for key in allowed {
        let _ = native_positive(native, family, key);
    }
}

fn replace_native_slots(native: &toml::Table, text: &str) -> String {
    let mut out = text.to_owned();
    for (slot, family, key) in NATIVE_FIELDS {
        let value = native_positive(native, family, key);
        out = out.replace(&format!("{{{{{slot}}}}}"), &value.to_string());
    }
    out
}

fn apply_native_unrolls(native: &toml::Table, mut out: String) -> String {
    for (name, family) in [
        ("fma_neon", "fma"),
        ("fma_avx2", "fma"),
        ("fma_scalar", "fma"),
        ("sdot", "sdot"),
        ("sme", "sme"),
    ] {
        let marker = format!("/* cutile-unroll: {name} */");
        let factor = native_positive(native, family, "unroll");
        let count = out.matches(&marker).count();
        if factor > 1 && count != 1 {
            fail(format!(
                "`gemm_micro.k` 的 `{name}` 展开点应恰好一次，实得 {count}"
            ));
        }
        out = out.replace(&marker, &format!("#pragma unroll {factor}"));
    }
    out
}

/// 一个配置字段：必须是非零、可由 usize 承载的整数。
fn native_positive(native: &toml::Table, family: &str, key: &str) -> usize {
    let value = native
        .get(family)
        .and_then(toml::Value::as_table)
        .and_then(|section| section.get(key))
        .and_then(toml::Value::as_integer)
        .unwrap_or_else(|| fail(format!("`gemm_micro.{family}.{key}` 必须是正整数")));
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| fail(format!("`gemm_micro.{family}.{key}` 超出正 usize 范围")))
}

#[cfg(test)]
mod tests {
    use super::documents;

    #[test]
    fn family_entries_keep_shared_geometry_and_their_own_constants() {
        let table = toml::from_str(
            "[axes]\nN = [32]\n[[operators]]\nop = { name = 'neg_f32', stem = 'unary_fn' }\nconsts = { unexpr = '-x' }\n[[operators]]\nop = { name = 'square_f32', stem = 'unary_fn' }\nconsts = { unexpr = 'x * x' }\n",
        )
        .expect("合法同族声明");
        let documents = documents("unary", &table).expect("同族展开");
        assert_eq!(documents.len(), 2);
        assert!(
            documents
                .iter()
                .all(|(_, table)| table.contains_key("axes"))
        );
        assert_eq!(documents[0].0, "neg_f32");
        assert_eq!(documents[1].1["consts"]["unexpr"].as_str(), Some("x * x"));
    }

    #[test]
    #[should_panic(expected = "不能同时包含")]
    fn mixed_single_and_family_declarations_are_rejected() {
        let table =
            toml::from_str("[op]\nname = 'single'\n[[operators]]\nop = { name = 'family' }\n")
                .expect("合法 TOML");
        let _ = documents("mixed", &table);
    }
}
