//! 档案 schema：把编译期展开的表读成 [`Profile`]（`parse`），再回答"档案自己是否自洽"
//! （`check`）。字段解析、见证判据与见证源码定位都在这一份文件里。

use core::fmt;
use diagnostic::{Code, Error};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use crate::atom::Set;
use crate::prof::*;

pub(crate) use crate::eval::eval_expression;

// ── parse ─────────────────────────────────────────────────────────────

impl MatchRule {
    /// 解析规则文本：`contains:` → 择一 → 集合 → 区间 → 精确逐级降级，只回答规则种类。
    pub fn parse(text: &str) -> Self {
        let text = text.trim();
        if let Some(tokens) = text.strip_prefix("contains:") {
            return Self::Contains(split(tokens, '|'));
        }
        if text.contains('|') {
            return Self::OneOf(text.split('|').map(Self::parse).collect());
        }
        Self::parse_narrow(text)
    }
    /// 剩下三种：集合、区间、精确；没有 `|` 也没有 `contains:` 时才轮到这里。
    fn parse_narrow(text: &str) -> Self {
        if let Some((base, members)) = brace_set(text) {
            return Self::Set { base, members };
        }
        if let Some((lo, hi)) = text.split_once("..") {
            let (lo, hi) = (lo.trim().to_string(), hi.trim().to_string());
            return Self::Range { lo, hi };
        }
        Self::Exact(text.to_string())
    }
}
/// 花括号集合语法 `base{a,b}`：括号必须成对且有序，否则不认（退回区间/精确）。
fn brace_set(text: &str) -> Option<(String, Vec<String>)> {
    let (open, close) = (text.find('{')?, text.rfind('}')?);
    if open >= close {
        return None;
    }
    let base = text[..open].trim().to_string();
    Some((base, split(&text[open + 1..close], ',')))
}
fn split(text: &str, delimiter: char) -> Vec<String> {
    let parts = text.split(delimiter).map(|part| part.trim().to_string());
    parts.filter(|part| !part.is_empty()).collect()
}
impl Profile {
    /// 从构建期展开的表解析（表由 [`crate::embedded`] 给出）；闭集与见证校验留给 [`Profile::validate`]。
    pub fn parse(table: &Table) -> Result<Self, ProfileError> {
        let meta = meta_of(table)?;
        let device = fingerprint_and_spaces(table)?;
        let capability = capability_sections(table)?;
        let (flags, preamble, headers) = parse_compile(table);
        Ok(Self {
            meta,
            fingerprint: device.fingerprint,
            spaces: device.spaces,
            mem: parse_mem(table),
            simt: parse_simt(table),
            units: capability.units,
            provides: capability.provides,
            peak: capability.peak,
            flags,
            preamble,
            headers,
        })
    }
}
/// `[meta]` 段定位：缺段报 `meta`，其余交给 [`parse_meta`]。
fn meta_of(table: &Table) -> Result<Meta, ProfileError> {
    let meta = table.table("meta").ok_or_else(|| missing("meta"))?;
    parse_meta(meta)
}
/// `[meta]` 段：厂商、型号、输出语言、工具名与核对日期；不核对工具链与本机是否一致。
fn parse_meta(table: &Table) -> Result<Meta, ProfileError> {
    Ok(Meta {
        vendor: named(table, "vendor", "meta.vendor")?,
        id: named(table, "id", "meta.id")?,
        emit: emit_of(table)?,
        tool: named(table, "tools", "meta.tools")?,
        tool_ver: or_dash(table, "tool_ver"),
        checked: or_dash(table, "checked"),
    })
}
/// 一段必填字符串：缺键或不是字符串都报 `meta.<键>` 定位。
fn named(table: &Table, key: &str, field: &str) -> Result<String, ProfileError> {
    Ok(text(table, key, field)?.to_string())
}
/// `[meta].emit`：必须在 [`Emit`] 闭集里。
fn emit_of(table: &Table) -> Result<Emit, ProfileError> {
    closed::<Emit>(text(table, "emit", "meta.emit")?, "meta.emit")
}
/// 指纹与存储层：[`Profile::parse`] 的两段定位描述，失败信息互不牵连。
fn fingerprint_and_spaces(table: &Table) -> Result<Device, ProfileError> {
    Ok(Device {
        fingerprint: parse_fingerprint(table)?,
        spaces: parse_spaces(table)?,
    })
}
/// 设备段的前两节：指纹规则与存储层名字映射。
struct Device {
    fingerprint: Vec<(String, MatchRule)>,
    spaces: Vec<(Space, String)>,
}
/// 能力三件套：执行单元、峰值与 `[[provides]]`（协商只看这三段）。
fn capability_sections(table: &Table) -> Result<Capability, ProfileError> {
    Ok(Capability {
        units: parse_units(table)?,
        peak: parse_peak(table)?,
        provides: parse_provides(table)?,
    })
}
/// 设备段的后三节：执行单元、峰值与能力条目。
struct Capability {
    units: Vec<Unit>,
    peak: Peak,
    provides: Vec<Provide>,
}
/// `[fingerprint]` 段：规则直接写在段下，或写在 `match = { ... }` 里（README §5.3）。
fn parse_fingerprint(table: &Table) -> Result<Vec<(String, MatchRule)>, ProfileError> {
    let section = table.table("fingerprint");
    let rules = section.map(|section| section.table("match").unwrap_or(section));
    let fingerprint = rules.map_or(Ok(Vec::new()), |rules| entries(rules, fingerprint_rule))?;
    if fingerprint.is_empty() {
        return Err(missing("fingerprint.match"));
    }
    Ok(fingerprint)
}
/// 一条指纹规则：值必须是字符串（规则文法由 [`MatchRule::parse`] 解释）。
fn fingerprint_rule(key: &str, value: &Value) -> Result<(String, MatchRule), ProfileError> {
    let Value::Str(text) = value else {
        return Err(bad(format!("fingerprint.match.{key}"), "只接受字符串规则"));
    };
    Ok((key.to_string(), MatchRule::parse(text)))
}
/// `[space]` 段：语义存储名 → 物理名，三级存储必须齐全。
fn parse_spaces(table: &Table) -> Result<Vec<(Space, String)>, ProfileError> {
    let section = table.table("space");
    let spaces = section.map_or(Ok(Vec::new()), |map| entries(map, space_entry))?;
    require_spaces(&spaces)?;
    Ok(spaces)
}
/// 一条存储映射：键是闭集里的语义名，值是物理名。
fn space_entry(key: &str, value: &Value) -> Result<(Space, String), ProfileError> {
    let space = closed::<Space>(key, "space")?;
    let Value::Str(name) = value else {
        return Err(bad(format!("space.{key}"), "只接受字符串"));
    };
    Ok((space, name.clone()))
}
/// 三级存储的语义名必须齐全：缺一条就拼不出完整的物理名映射。
fn require_spaces(spaces: &[(Space, String)]) -> Result<(), ProfileError> {
    let has = |wanted: Space| spaces.iter().any(|(have, _)| *have == wanted);
    let unfilled = Space::ALL.iter().copied().find(|space| !has(*space));
    unfilled.map_or(Ok(()), |space| Err(unfilled_space(space)))
}
fn unfilled_space(space: Space) -> ProfileError {
    let name = space.as_str();
    missing(format!("space.{name}（三级存储的语义名必须齐全）"))
}
/// `[mem]` 段：容量与 bank 数，缺项按 0。
fn parse_mem(table: &Table) -> Mem {
    let mem = table.table("mem");
    let fast = mem.and_then(|mem| mem.table("FAST"));
    let reg = mem.and_then(|mem| mem.table("REG"));
    Mem {
        fast_bytes: count(fast, "bytes") as u64,
        fast_bank: count(fast, "bank") as u32,
        reg_bytes: count(reg, "bytes") as u64,
    }
}
fn count(table: Option<&Table>, key: &str) -> i64 {
    table.and_then(|table| table.int(key)).unwrap_or(0).max(0)
}
/// `[simt]` 段：缺段表示"不是 SIMT"，不能拿默认值 1 当作单 lane 的证据。
fn parse_simt(table: &Table) -> Simt {
    let simt = table.table("simt");
    Simt {
        group: count(simt, "group").max(1) as u32,
        groups_max: count(simt, "groups_max").max(1) as u32,
        declared: simt.is_some(),
    }
}
/// `[units.<name>]` 段：执行单元及其可访问的存储层；缺段表示档案不区分单元。
fn parse_units(table: &Table) -> Result<Vec<Unit>, ProfileError> {
    table
        .table("units")
        .map_or(Ok(Vec::new()), |section| entries(section, unit_entry))
}
/// 一个执行单元：必须是 `[units.<name>]` 形式的表，`space` 必填。
fn unit_entry(name: &str, value: &Value) -> Result<Unit, ProfileError> {
    let Value::Table(inner) = value else {
        return Err(bad(format!("units.{name}"), UNIT_DETAIL));
    };
    let path = format!("units.{name}.space");
    let space = parse_space_set(&path, &list(inner, "space", &path)?)?;
    Ok(Unit {
        name: name.to_string(),
        space,
    })
}
/// `[[provides]]` 数组：至少一条，否则协商没有可选的能力。
fn parse_provides(table: &Table) -> Result<Vec<Provide>, ProfileError> {
    let tables = table.table_array("provides").unwrap_or_default();
    let each = |(idx, entry): (usize, &&Table)| parse_provide(entry, idx);
    let provides: Vec<Provide> = tables
        .iter()
        .enumerate()
        .map(each)
        .collect::<Result<_, _>>()?;
    if provides.is_empty() {
        return Err(missing("provides"));
    }
    Ok(provides)
}
/// `[peak]` 段：roofline 用的峰值常数，两项都可以缺。
fn parse_peak(table: &Table) -> Result<Peak, ProfileError> {
    let empty = Table::new();
    let peak = table.table("peak").unwrap_or(&empty);
    Ok(Peak {
        flops_f32: parse_constant(peak, "flops_f32")?,
        bw: parse_constant(peak, "bw")?,
    })
}
/// `[compile]` 段：参与缓存指纹的参数与源文件前置声明，缺项给空。
fn parse_compile(table: &Table) -> (Vec<String>, String, Vec<(String, String)>) {
    let compile = table.table("compile");
    let flags = compile
        .and_then(|table| table.string_array("flags"))
        .unwrap_or_default();
    let flags = flags.into_iter().map(str::to_string).collect();
    let preamble = compile
        .and_then(|table| table.string("preamble"))
        .unwrap_or("");
    let headers = compile
        .and_then(|table| table.table("headers"))
        .map(|table| {
            table
                .entries()
                .iter()
                .filter_map(|(name, value)| match value {
                    ::diagnostic::data::Value::Str(text) => Some((name.clone(), text.clone())),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();
    (flags, preamble.to_string(), headers)
}
/// 峰值常数：`value` 必须等于 `src` 的推导式结果；抄来的数字与带变量名的公式都不收。
fn parse_constant(table: &Table, key: &str) -> Result<Option<Constant>, ProfileError> {
    table
        .table(key)
        .map(|inner| constant(inner, &format!("peak.{key}")))
        .transpose()
}
/// 一条峰值常数：两个必填项加上一次对拍。
fn constant(table: &Table, field: &str) -> Result<Constant, ProfileError> {
    let path = format!("{field}.value");
    let value = table.float("value").ok_or(missing(&path))?;
    let src = text(table, "src", format!("{field}.src"))?.to_string();
    let derived = derive(field, &src)?;
    // 允许字面量的常规舍入（如 `2.1e12`），但容不下 10% 量级的"抄错一位"。
    if (derived - value).abs() > value.abs().max(1.0) * 1e-6 {
        let detail = format!("推导式 `{src}` = {derived:e}，与 value = {value:e} 不一致");
        return Err(bad(format!("{field}.src"), detail));
    }
    Ok(Constant { value, src })
}
/// 推导式求值；失败一律是 `invalid`（推导式是档案的一部分，写错就是声明写错）。
fn derive(field: &str, src: &str) -> Result<f64, ProfileError> {
    let fail = |error: Error| {
        let detail = format!("推导式 `{src}` 求值失败：{}", error.message());
        bad(format!("{field}.src"), detail)
    };
    eval_expression(src).map_err(fail)
}
/// 推导式写错：位置与非法内容由调用方给全。
pub(crate) fn invalid(detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, detail)
}

// ── provides 字段 ─────────────────────────────────────────────────────

/// 一个"名字集合"字段：标量字符串或字符串数组，逐项交给 `each`。
/// `dt` 与 `space` 是同一形状，且都只把名字解析成位集——这里不克隆 `String`。
fn each_name(
    field: impl fmt::Display,
    value: &Value,
    mut each: impl FnMut(&str) -> Result<(), ProfileError>,
) -> Result<(), ProfileError> {
    match value {
        Value::Str(text) => each(text),
        Value::Array(items) => {
            for item in items {
                let Value::Str(text) = item else {
                    return Err(bad(&field, "数组里只能是字符串"));
                };
                each(text)?;
            }
            Ok(())
        }
        _ => Err(bad(field, "只接受字符串或字符串数组")),
    }
}
/// 一组存储层名字：每个都必须在 [`Space`] 闭集里。
fn parse_space_set(
    field: impl fmt::Display,
    items: &[&str],
) -> Result<SpaceSet, ProfileError> {
    let mut set = SpaceSet::empty();
    for item in items {
        set.insert(closed::<Space>(item, &field)?);
    }
    Ok(set)
}
/// 一条 `[[provides]]`：认得的键各有分支，其余一律当属性（键还要过 [`AttrKey`] 闭集）。
fn parse_provide(entry: &Table, idx: usize) -> Result<Provide, ProfileError> {
    let mut provide = head_of(entry, idx)?;
    provide_axes(&mut provide, entry, idx)?;
    provide_types(&mut provide, entry, idx)?;
    provide_tail(&mut provide, entry, idx)?;
    Ok(provide)
}
/// 一条 `[[provides]]` 的头两项：`prim` 必填且必须在闭集里，`impl` 缺省为 `-`。
fn head_of(entry: &Table, idx: usize) -> Result<Provide, ProfileError> {
    let field = at(idx, "prim");
    let prim = closed::<Prim>(text(entry, "prim", field)?, field)?;
    Ok(Provide::of(prim, entry.string("impl").unwrap_or("-")))
}
/// 三条区间：主维度（`m` / `bytes` / `extent` 三选一）与 `n` / `k`。
fn provide_axes(provide: &mut Provide, entry: &Table, idx: usize) -> Result<(), ProfileError> {
    provide.extent = span_of(entry, idx, &["m", "bytes", "extent"], EXTENT_DETAIL)?;
    provide.n = span_of(entry, idx, &["n"], AXIS_DETAIL)?;
    provide.k = span_of(entry, idx, &["k"], AXIS_DETAIL)?;
    Ok(())
}
/// `dt` / `space` 缺省为空集，`grade` 必填。
fn provide_types(provide: &mut Provide, entry: &Table, idx: usize) -> Result<(), ProfileError> {
    provide.dt = closed_set(entry, idx, "dt", DTypeSet::empty(), DType::parse_name)?;
    provide.space = closed_set(entry, idx, "space", SpaceSet::empty(), Space::parse)?;
    provide.grade = grade_of(entry, idx)?;
    Ok(())
}
/// 见证与其余属性：两者都只在签名写完之后才看得懂。
fn provide_tail(provide: &mut Provide, entry: &Table, idx: usize) -> Result<(), ProfileError> {
    provide.witness = witness_of(entry, idx)?;
    attrs(entry, idx, provide)
}
/// 一组同义维度：没写就是不限，多写就是错，写了就要能解析。
fn span_of(
    entry: &Table,
    idx: usize,
    names: &[&str],
    detail: &str,
) -> Result<Span, ProfileError> {
    let hit = |name: &&str| entry.contains(name);
    let used: Vec<&str> = names.iter().copied().filter(hit).collect();
    let Some(alias) = used.first().copied() else {
        return Ok(Span::any());
    };
    if used.len() > 1 {
        let count = used.len();
        let aliases = "主维度有三个同义名字（`m` / `bytes` / `extent`），只能写一个";
        let detail = format!("{aliases}，这里写了 {count}");
        return Err(bad(at(idx, alias), detail));
    }
    let field = at(idx, alias);
    let Some(text) = entry.string(alias) else {
        return Err(bad(field, detail));
    };
    parse_span(field, text)
}
/// `dt` / `space` 的公共形状：标量或数组都接受，缺省是空集，闭集外取值报 `Unknown`。
fn closed_set<E: Closed, const WORDS: usize>(
    entry: &Table,
    idx: usize,
    key: &str,
    empty: Set<E, WORDS>,
    parse: impl Fn(&str) -> Option<E>,
) -> Result<Set<E, WORDS>, ProfileError> {
    let Some(value) = entry.get(key) else {
        return Ok(empty);
    };
    let field = at(idx, key);
    let mut set = Set::empty();
    each_name(field, value, |name| {
        let item = parse(name).ok_or_else(|| unknown(field, name, &closed_names(E::ALL)))?;
        set.insert(item);
        Ok(())
    })?;
    Ok(set)
}
/// `grade`：必填，闭集外取值报 `Unknown`。
fn grade_of(entry: &Table, idx: usize) -> Result<Grade, ProfileError> {
    let path = at(idx, "grade");
    closed::<Grade>(text(entry, "grade", path)?, &path)
}
/// `witness`：内联表 `{ file, func, expect }`；只有路径的见证无法执行。
fn witness_of(entry: &Table, idx: usize) -> Result<Option<Witness>, ProfileError> {
    let Some(table) = entry.table("witness") else {
        if entry.contains("witness") {
            return Err(bad(at(idx, "witness"), WITNESS_DETAIL));
        }
        return Ok(None);
    };
    Ok(Some(witness_fields(table, |key| witness_at(idx, key))?))
}
/// 见证的三个字段：`file` / `func` 必填，`expect` 是数值数组。字段路径由调用方给，
/// 报错位置因此落在声明它的那一段里。
fn witness_fields(
    table: &Table,
    at: impl Fn(&'static str) -> Field<'static>,
) -> Result<Witness, ProfileError> {
    let file = text(table, "file", at("file"))?.to_string();
    let func = text(table, "func", at("func"))?.to_string();
    let expect = expect_values(table, at("expect"))?;
    Ok(Witness { file, func, expect })
}
/// `expect`：数值数组，按顺序与见证输出对拍。
fn expect_values(table: &Table, path: Field<'_>) -> Result<Vec<f32>, ProfileError> {
    let items = table.array("expect").ok_or_else(|| missing(path))?;
    let number = |value: &Value| expect_number(value, path);
    items.iter().map(number).collect()
}
fn expect_number(value: &Value, path: impl fmt::Display) -> Result<f32, ProfileError> {
    match value {
        Value::Float(number) => Ok(*number as f32),
        Value::Int(number) => Ok(*number as f32),
        _ => Err(bad(path, "只能是数值")),
    }
}
/// 其余键一律视为枚举型属性；闭集外的键直接报错，避免"写了但没人看"。
fn attrs(entry: &Table, idx: usize, provide: &mut Provide) -> Result<(), ProfileError> {
    for (key, value) in entry.entries() {
        if !parsed_key(key) {
            provide.attrs.push(one_attr(idx, key, value)?);
        }
    }
    Ok(())
}
/// 已经有专属解析分支的键：再当属性处理就会重复计入（与诊断同一种 `a | b` 写法）。
fn parsed_key(key: &str) -> bool {
    matches!(
        key,
        "prim" | "impl" | "m" | "bytes" | "extent" | "n" | "k" | "dt" | "space" | "grade"
            | "witness"
    )
}
/// 一条属性：键在 [`AttrKey`] 闭集里，取值按标量或数组展开。
fn one_attr(idx: usize, key: &str, value: &Value) -> Result<Attr, ProfileError> {
    let attr_key = closed::<AttrKey>(key, &at(idx, key))?;
    Ok(Attr::new(attr_key, attr_values(idx, key, value)?))
}
/// 属性取值：标量或数组；表在这里没有意义。
fn attr_values(
    idx: usize,
    key: &str,
    value: &Value,
) -> Result<Vec<String>, ProfileError> {
    let path = at(idx, key);
    match value {
        Value::Array(items) => items.iter().map(|item| one_string(path, item)).collect(),
        Value::Table(_) => Err(bad(path, "属性不能是表")),
        scalar => scalar_texts(scalar).ok_or_else(|| bad(path, "属性不能是表")),
    }
}
fn one_string(path: impl fmt::Display, item: &Value) -> Result<String, ProfileError> {
    match item {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad(path, "属性数组里只能是字符串")),
    }
}
/// 区间文本：解析失败时把原话带上，字段路径由调用方给全。
fn parse_span(field: impl fmt::Display, text: &str) -> Result<Span, ProfileError> {
    Span::parse(text).map_err(|error| bad(&field, error.message().to_string()))
}

// ── check ─────────────────────────────────────────────────────────────

/// 写了 `exec` 却没声明执行单元。
const EXEC_DETAIL: &str = "写 `exec` 前必须先用 `[units.<name>]` 声明执行单元";
impl Profile {
    /// 闭集与见证校验：只回答"档案自己是否自洽"，不判断它是否能在某台机器上跑对。
    pub fn validate(&self, witnesses: Option<&dyn WitnessLookup>) -> Result<(), Vec<ProfileError>> {
        let mut errors = self.head_errors();
        let mut files = WitnessFiles::new(witnesses);
        for provide in &self.provides {
            errors.extend(witness_shape(provide));
            errors.extend(files.probe(provide));
            errors.extend(signature_fields(provide));
        }
        errors.extend(self.simt_lanes());
        errors.extend(self.exec_units());
        let passed = errors.is_empty().then_some(());
        passed.ok_or(errors)
    }
    /// 单一根下的校验：见证文件都相对 `root` 解析（`<root>/<vendor>/witness.<ext>`）。
    pub fn validate_in(&self, root: &Path) -> Result<(), Vec<ProfileError>> {
        self.validate(Some(&Root(root)))
    }
    /// 跨供给的重复项与属性键闭集：`impl` 重名、见证复用、无关属性键。
    fn head_errors(&self) -> Vec<ProfileError> {
        let mut errors = self.duplicate_impls();
        errors.extend(self.attr_keys());
        errors.extend(self.duplicate_witnesses());
        errors
    }
    /// 同一原语的多条签名必须用不同 `impl`，否则协商只会取到其中一条。
    fn duplicate_impls(&self) -> Vec<ProfileError> {
        let mut seen: BTreeMap<(Prim, &str), usize> = BTreeMap::new();
        let mut errors = Vec::new();
        for (idx, provide) in self.provides.iter().enumerate() {
            let key = (provide.prim, provide.impl_name.as_str());
            let Some(first) = seen.insert(key, idx) else {
                continue;
            };
            let detail = format!(
                "`{}` 的实现名与 provides[{first}] 重名：同一原语的多条签名必须用不同 impl 区分",
                provide.prim
            );
            errors.push(bad(at(idx, "impl"), detail));
        }
        errors
    }
    /// 属性键必须属于该原语的能力签名：多写的键永远不会被协商比对，闭集在这里就拒收。
    fn attr_keys(&self) -> Vec<ProfileError> {
        let mut errors = Vec::new();
        for provide in &self.provides {
            for attr in &provide.attrs {
                if !provide.prim.accepts(attr.key()) {
                    errors.push(attr_key_error(provide, attr.key()));
                }
            }
        }
        errors
    }
    /// 一条见证只证明一条签名：两段签名指向同一入口时，"都有见证"就无法归属到哪一条。
    fn duplicate_witnesses(&self) -> Vec<ProfileError> {
        let mut seen: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut errors = Vec::new();
        for (idx, provide) in self.provides.iter().enumerate() {
            let Some(witness) = provide.witness.as_ref() else {
                continue;
            };
            let key = (witness.file.as_str(), witness.func.as_str());
            let Some(first) = seen.insert(key, idx) else {
                continue;
            };
            let previous = &self.provides[first];
            let was = format!("{}/{}", previous.prim, previous.impl_name);
            let now = format!("{}/{}", provide.prim, provide.impl_name);
            let detail = format!(
                "见证 `{}` 同时被 `{was}` 与 `{now}` 引用：一条见证只能证明一条签名",
                witness.func
            );
            errors.push(witness_error(provide, detail));
        }
        errors
    }
    /// 子组内交换/广播以"有第二条 lane"为前提；只看 `group` 不够，还要 `declared`。
    fn simt_lanes(&self) -> Vec<ProfileError> {
        let group = self.simt.group;
        let swap = |p: &Provide| matches!(p.prim, Prim::Shuffle | Prim::Broadcast);
        let on_one_lane = |p: &Provide| swap(p) && self.simt.declared && group <= 1;
        let errors = self.provides.iter().filter(|p| on_one_lane(p));
        errors
            .map(|p| {
                bad(
                    prim_at(p.prim, ""),
                    format!(
                        "`{}` 要求 `[simt].group > 1`（子组内交换/广播），本档案声明的是 {group}：单 lane 设备没有可交换的对象，这条能力应当不声明",
                        p.prim
                    ),
                )
            })
            .collect()
    }
    /// `exec` 必须对得上本档案声明的执行单元，且该单元的存储层要覆盖这条供给。
    fn exec_units(&self) -> Vec<ProfileError> {
        let mut errors = Vec::new();
        for provide in &self.provides {
            if let Some(exec) = exec_attr(provide) {
                errors.extend(self.exec_errors(provide, exec));
            }
        }
        errors
    }
    /// 一条 `exec` 断言：单元要在，且它的存储层要覆盖供给声明的 `space`。
    fn exec_errors(&self, provide: &Provide, exec: &Attr) -> Vec<ProfileError> {
        let field = prim_at(provide.prim, "exec");
        if self.units.is_empty() {
            return vec![bad(field, EXEC_DETAIL)];
        }
        let (reachable, mut errors) = self.exec_reach(field, exec.values());
        if !provide.space.is_subset(reachable) {
            let unreachable = provide.space.difference(reachable);
            let detail = format!("`{}` 访问不到 [{unreachable}]", exec.values().join("+"));
            errors.push(bad(prim_at(provide.prim, "space"), detail));
        }
        errors
    }
    /// `exec` 取值能触及的存储层并集；本档案没声明的单元逐个报 `Unknown`。
    fn exec_reach(
        &self,
        field: impl fmt::Display,
        values: &[String],
    ) -> (SpaceSet, Vec<ProfileError>) {
        let allowed: Vec<&str> = self.units.iter().map(|unit| unit.name.as_str()).collect();
        let (mut reachable, mut undefined) = (SpaceSet::empty(), Vec::new());
        for value in values {
            match self.units.iter().find(|unit| unit.name == *value) {
                Some(unit) => unit.space.iter().for_each(|space| reachable.insert(space)),
                None => undefined.push(unknown(&field, value, &allowed)),
            }
        }
        (reachable, undefined)
    }
}

/// 见证的形状：`L2` 必须有见证，声明了就要有入口函数名与期望值（不碰文件）。
fn witness_shape(provide: &Provide) -> Vec<ProfileError> {
    let Some(witness) = provide.witness.as_ref() else {
        return missing_witness(provide).into_iter().collect();
    };
    let make = |(_, detail): (&'static str, String)| witness_error(provide, detail);
    shape_details(witness).into_iter().map(make).collect()
}
/// 见证形状的判据本体：键是 `func` / `expect`，两种供给（原语与写回形式）共用一份。
fn shape_details(witness: &Witness) -> Vec<(&'static str, String)> {
    let len = witness.expect.len();
    let too_many = format!("期望输出 {len} 个，超过上限 {}", Witness::MAX_OUT);
    let checks = [
        (
            witness.func.is_empty(),
            "func",
            "见证必须给出入口函数名，否则无法执行".to_string(),
        ),
        (
            witness.expect.is_empty(),
            "expect",
            "见证必须给出期望输出，否则无法对拍".to_string(),
        ),
        (len > Witness::MAX_OUT, "expect", too_many),
    ];
    let failed = checks.into_iter().filter(|(failed, ..)| *failed);
    failed.map(|(_, key, detail)| (key, detail)).collect()
}
/// 没有见证时唯一说得通的理由是等级还不到 `L2`——等级是执行的结论，不是自述。
fn missing_witness(provide: &Provide) -> Option<ProfileError> {
    if !provide.grade.is_verified() {
        return None;
    }
    let grade = provide.grade;
    let detail = format!("{grade} 要求 `witness = {{ file, func, expect }}`");
    Some(witness_error(provide, detail))
}
/// 签名的必填项：有数据类型的原语不能给空集；`L2` 及以上必须写实现名。
fn signature_fields(provide: &Provide) -> Vec<ProfileError> {
    let empty_dt = provide.dt.is_empty() && provide.prim.has_dtype();
    let no_impl = provide.grade.is_verified() && provide.impl_name.is_empty();
    let checks = [
        (empty_dt, "dt", "数据类型集合不能为空"),
        (no_impl, "impl", "实现名不能为空"),
    ];
    let failed = checks.into_iter().filter(|(failed, ..)| *failed);
    let make = |(_, key, detail): (bool, &str, &str)| bad(prim_at(provide.prim, key), detail);
    failed.map(make).collect()
}
/// 闭集外的属性键：闭集是该原语签名键的并集 `exec`（执行单元是档案级属性）。
fn attr_key_error(provide: &Provide, key: AttrKey) -> ProfileError {
    let keys = provide.prim.attr_keys();
    let mut allowed: Vec<&str> = keys.iter().map(|key| key.as_str()).collect();
    allowed.push("exec");
    unknown(prim_at(provide.prim, key.as_str()), key.as_str(), &allowed)
}
/// 一条供给声明的执行单元列表；没写 `exec` 就没有这条判据。
fn exec_attr(provide: &Provide) -> Option<&Attr> {
    let is_exec = |attr: &&Attr| attr.key() == AttrKey::Exec;
    provide.attrs.iter().find(is_exec)
}
/// 见证源码的定位方式：把 `witness.file`（`<vendor>/witness.<ext>`）解析成一个可读路径。
pub trait WitnessLookup {
    /// 见证文件的可读路径；`None` 表示这个 vendor 没有任何落盘位置。
    fn path(&self, file: &str) -> Option<PathBuf>;
}
/// 单一根下的定位：`file` 直接拼在根上。
struct Root<'a>(&'a Path);
impl WitnessLookup for Root<'_> {
    fn path(&self, file: &str) -> Option<PathBuf> {
        Some(self.0.join(file))
    }
}
/// 见证文件的一次性读取缓存：同一份源码只读一次，同一个缺失文件只报一次。
struct WitnessFiles<'a> {
    /// 定位方式；`None` 表示本轮不做文件检查（单元测试）。
    witnesses: Option<&'a dyn WitnessLookup>,
    /// 文件 → 源码；`None` 表示不存在或不可读。
    sources: BTreeMap<String, Option<String>>,
    /// 已经报过的缺失路径。
    reported: BTreeSet<String>,
}
impl<'a> WitnessFiles<'a> {
    /// 空缓存。
    fn new(witnesses: Option<&'a dyn WitnessLookup>) -> Self {
        Self {
            witnesses,
            sources: BTreeMap::new(),
            reported: BTreeSet::new(),
        }
    }
    /// 一条供给的文件级判据：文件要在，且真的声明了那个入口函数；不判断跑不跑得通。
    fn probe(&mut self, provide: &Provide) -> Vec<ProfileError> {
        let make = |detail: String| witness_error(provide, detail);
        let (Some(witnesses), Some(witness)) = (self.witnesses, provide.witness.as_ref()) else {
            return Vec::new();
        };
        let source = self.source(witnesses, &witness.file);
        let found = source.map(|text| declares(text, witness));
        match found {
            None => self.report(witnesses, &witness.file),
            Some(false) => vec![make(absent_function(witness))],
            Some(true) => Vec::new(),
        }
    }
    /// 读一份见证源码：每个文件最多读一次盘。
    fn source(&mut self, witnesses: &dyn WitnessLookup, file: &str) -> Option<&str> {
        let text = self.sources.entry(file.to_string());
        let text = text.or_insert_with(|| {
            let path = witnesses.path(file)?;
            std::fs::read_to_string(path).ok()
        });
        text.as_deref()
    }
    /// 缺失文件只报第一次：多条供给引用同一份缺失文件时，原因只有一个。
    fn report(&mut self, witnesses: &dyn WitnessLookup, file: &str) -> Vec<ProfileError> {
        let path = witnesses
            .path(file)
            .map_or_else(|| file.to_string(), |path| path.display().to_string());
        if !self.reported.insert(path.clone()) {
            return Vec::new();
        }
        vec![ProfileError::WitnessFile {
            file: path,
            detail: "不存在或不可读".to_string(),
        }]
    }
}
fn declares(text: &str, witness: &Witness) -> bool {
    text.contains(&format!("{}(", witness.func))
}
fn absent_function(witness: &Witness) -> String {
    format!(
        "`{}` 里没有函数 `{}`：声明的见证无法执行",
        witness.file, witness.func
    )
}
