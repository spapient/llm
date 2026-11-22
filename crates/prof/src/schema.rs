//! 档案 schema：把编译期展开的表读成 [`Profile`]（`parse`），再回答"档案自己是否自洽"
//! （`check`）。

use core::fmt;
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::path::Path;
use crate::prof::*;

pub(crate) use crate::eval::eval_expression;
use crate::provide::*;
pub use crate::witness::WitnessLookup;
use crate::witness::*;

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
/// 指纹与存储层：[`Self::parse`] 的两段定位描述，失败信息互不牵连。
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
            errors.push(bad(at(idx, "impl"), dup_impl(first, provide)));
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
            let detail = shared_witness(witness, previous, provide);
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
            .map(|p| bad(prim_at(p.prim, ""), lane_swap_detail(p.prim, group)))
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
            let detail = reach_missing(exec, unreachable);
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
