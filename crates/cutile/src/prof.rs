//! 档案层：设备指纹、参数化能力供给、证据等级与字段校验（DESIGN §5.3、§11.2）。
//!
//! 档案是"设备只建档"的那一列：同一份模板之所以能跑在八台设备上，是因为每个变量
//! （存储层物理名、输出语言、工具链、能力区间）都在这份数据里，而不是在代码里。

use crate::atom::{
    Attr, AttrKey, DType, DTypeSet, Grade, Need, Prim, Provide, Space, SpaceSet, Span,
};
use crate::toml::{self, Table};
use core::fmt;
use std::collections::BTreeMap;
use std::path::Path;

/// 输出语言闭集（DESIGN §5.4 的 `emit` 列）。设备与输出语言是两个维度：
/// 同一条 `cuda` emit 同时服务 NVIDIA 与 MUSA。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Emit {
    /// CUDA C++。
    Cuda,
    /// PTX 汇编。
    Ptx,
    /// HIP。
    Hip,
    /// OpenCL C。
    Opencl,
    /// SPIR-V。
    Spirv,
    /// Metal Shading Language。
    Metal,
    /// 昇腾 AscendC。
    Ascendc,
    /// 可移植 C。
    C,
    /// WGSL。
    Wgsl,
}

impl Emit {
    /// 闭集全量。
    pub const ALL: [Self; 9] = [
        Self::Cuda,
        Self::Ptx,
        Self::Hip,
        Self::Opencl,
        Self::Spirv,
        Self::Metal,
        Self::Ascendc,
        Self::C,
        Self::Wgsl,
    ];

    /// 档案与缓存里使用的名字。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Ptx => "ptx",
            Self::Hip => "hip",
            Self::Opencl => "opencl",
            Self::Spirv => "spirv",
            Self::Metal => "metal",
            Self::Ascendc => "ascendc",
            Self::C => "c",
            Self::Wgsl => "wgsl",
        }
    }

    /// 生成的源文件扩展名。
    pub const fn ext(self) -> &'static str {
        match self {
            Self::Cuda => "cu",
            Self::Ptx => "ptx",
            Self::Hip => "hip",
            Self::Opencl => "cl",
            Self::Spirv => "spv",
            Self::Metal => "metal",
            Self::Ascendc => "cpp",
            Self::C => "c",
            Self::Wgsl => "wgsl",
        }
    }

    /// 从名字解析。
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|emit| emit.as_str() == name)
    }
}

impl fmt::Display for Emit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 档案的元信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Meta {
    /// 厂商键（`nvidia` / `cpu` …）。
    pub vendor: String,
    /// 具体型号键（`sm100` / `x86_64` …），指纹最具体者胜出。
    pub id: String,
    /// 输出语言。
    pub emit: Emit,
    /// 编译工具名。
    pub tool: String,
    /// 工具版本。
    pub tool_ver: String,
    /// 上次核对日期。
    pub checked: String,
}

/// 指纹匹配规则（一份档案可以有多条，全部满足才算命中）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchRule {
    /// 精确相等。
    Exact(String),
    /// 版本区间，如 `10.0..10.3`。
    Range {
        /// 下界（含）。
        lo: String,
        /// 上界（含）。
        hi: String,
    },
    /// 包含任一子串，如 `contains: B200|GB200`。
    Contains(Vec<String>),
    /// 集合语法，如 `x86_64{avx2,avx512}`。
    Set {
        /// 前缀。
        base: String,
        /// 必须同时出现的成员。
        members: Vec<String>,
    },
    /// 择一语法，如 `x86_64|aarch64`；每个分支各自按完整规则解析。
    OneOf(Vec<MatchRule>),
}

impl MatchRule {
    /// 解析规则文本。
    pub fn parse(text: &str) -> Self {
        let text = text.trim();
        if let Some(rest) = text.strip_prefix("contains:") {
            return Self::Contains(
                rest.split('|')
                    .map(|part| part.trim().to_string())
                    .filter(|part| !part.is_empty())
                    .collect(),
            );
        }
        if text.contains('|') {
            return Self::OneOf(text.split('|').map(Self::parse).collect());
        }
        if let (Some(open), Some(close)) = (text.find('{'), text.rfind('}'))
            && open < close
        {
            let base = text[..open].trim().to_string();
            let members = text[open + 1..close]
                .split(',')
                .map(|part| part.trim().to_string())
                .filter(|part| !part.is_empty())
                .collect();
            return Self::Set { base, members };
        }
        if let Some((lo, hi)) = text.split_once("..") {
            return Self::Range {
                lo: lo.trim().to_string(),
                hi: hi.trim().to_string(),
            };
        }
        Self::Exact(text.to_string())
    }

    /// 是否匹配一个实际探测值。
    pub fn matches(&self, actual: &str) -> bool {
        match self {
            Self::Exact(expected) => actual == expected,
            Self::Contains(tokens) => tokens.iter().any(|token| actual.contains(token)),
            Self::Set { base, members } => {
                actual.starts_with(base.as_str())
                    && members
                        .iter()
                        .all(|member| actual.contains(member.as_str()))
            }
            Self::OneOf(branches) => branches.iter().any(|branch| branch.matches(actual)),
            Self::Range { lo, hi } => {
                let Some(value) = parse_version(actual) else {
                    return false;
                };
                parse_version(lo).is_some_and(|low| low <= value)
                    && parse_version(hi).is_some_and(|high| value <= high)
            }
        }
    }
}

impl fmt::Display for MatchRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(text) => f.write_str(text),
            Self::Range { lo, hi } => write!(f, "{lo}..{hi}"),
            Self::Contains(tokens) => write!(f, "contains: {}", tokens.join("|")),
            Self::Set { base, members } => write!(f, "{base}{{{}}}", members.join(",")),
            Self::OneOf(branches) => {
                let parts: Vec<String> = branches.iter().map(ToString::to_string).collect();
                f.write_str(&parts.join("|"))
            }
        }
    }
}

fn parse_version(text: &str) -> Option<Vec<u32>> {
    text.trim()
        .split('.')
        .map(|part| part.parse::<u32>().ok())
        .collect()
}

/// 一台设备实际探测到的指纹：只如实记录，不做判断。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Detected {
    attrs: BTreeMap<String, String>,
}

impl Detected {
    /// 空指纹（测试与模拟用）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一项探测结果。
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<String>) -> Self {
        self.attrs.insert(key.to_string(), value.into());
        self
    }

    /// 取一项。
    pub fn get(&self, key: &str) -> Option<&str> {
        self.attrs.get(key).map(String::as_str)
    }

    /// 全部项。
    pub fn attrs(&self) -> &BTreeMap<String, String> {
        &self.attrs
    }
}

impl fmt::Display for Detected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .attrs
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect();
        f.write_str(&parts.join(" "))
    }
}

/// 存储容量与 bank 数。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Mem {
    /// 片上快速存储字节数。
    pub fast_bytes: u64,
    /// 片上快速存储的 bank 数（swizzle 合法性用）。
    pub fast_bank: u32,
    /// 寄存器字节数。
    pub reg_bytes: u64,
}

/// SIMT 执行模型参数（CPU 也有一份，`group = 1`）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Simt {
    /// 子组宽度。
    pub group: u32,
    /// 每块最大子组数。
    pub groups_max: u32,
}

/// 一个带推导式出处的常数（DESIGN §12.1）。
#[derive(Debug, Clone, PartialEq)]
pub struct Constant {
    /// 数值。
    pub value: f64,
    /// 推导式或来源。
    pub src: String,
}

/// roofline 所需的峰值常数。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Peak {
    /// f32 峰值（FLOP/s）。
    pub flops_f32: Option<Constant>,
    /// 带宽（B/s）。
    pub bw: Option<Constant>,
}

/// 一份设备档案。
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    /// 元信息。
    pub meta: Meta,
    /// 指纹规则，全部满足才命中。
    pub fingerprint: Vec<(String, MatchRule)>,
    /// 语义存储名 → 物理名。
    pub spaces: Vec<(Space, String)>,
    /// 容量与 bank。
    pub mem: Mem,
    /// SIMT 参数。
    pub simt: Simt,
    /// 能力供给。
    pub provides: Vec<Provide>,
    /// 峰值常数。
    pub peak: Peak,
    /// 编译参数（参与缓存指纹）。
    pub flags: Vec<String>,
    /// 源文件前置声明（`#include` 之类）。
    pub preamble: String,
}

/// 档案加载 / 校验错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    /// 解析失败。
    Syntax(String),
    /// 缺字段。
    Missing {
        /// 字段路径。
        field: String,
    },
    /// 闭集外取值。
    Unknown {
        /// 字段路径。
        field: String,
        /// 非法取值。
        value: String,
        /// 闭集内容。
        allowed: String,
    },
    /// 证据等级不满足要求。
    Witness {
        /// 原语。
        prim: Prim,
        /// 说明。
        detail: String,
    },
    /// 数值非法。
    BadValue {
        /// 字段路径。
        field: String,
        /// 说明。
        detail: String,
    },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(detail) => write!(f, "解析失败：{detail}"),
            Self::Missing { field } => write!(f, "缺字段 `{field}`"),
            Self::Unknown {
                field,
                value,
                allowed,
            } => write!(f, "`{field} = {value}` 不在闭集内（闭集：{allowed}）"),
            Self::Witness { prim, detail } => write!(f, "`{prim}` 的见证不成立：{detail}"),
            Self::BadValue { field, detail } => write!(f, "`{field}` 非法：{detail}"),
        }
    }
}

impl std::error::Error for ProfileError {}

fn missing(field: &str) -> ProfileError {
    ProfileError::Missing {
        field: field.to_string(),
    }
}

fn unknown(field: &str, value: &str, allowed: &[&str]) -> ProfileError {
    ProfileError::Unknown {
        field: field.to_string(),
        value: value.to_string(),
        allowed: allowed.join(" | "),
    }
}

impl Profile {
    /// 从档案文本解析（不做闭集与见证校验，校验在 [`Profile::validate`]）。
    pub fn parse(source: &str) -> Result<Self, ProfileError> {
        let table = toml::parse(source).map_err(|error| ProfileError::Syntax(error.to_string()))?;

        let meta_table = table.table("meta").ok_or_else(|| missing("meta"))?;
        let vendor = meta_table
            .string("vendor")
            .ok_or_else(|| missing("meta.vendor"))?
            .to_string();
        let id = meta_table
            .string("id")
            .ok_or_else(|| missing("meta.id"))?
            .to_string();
        let emit_name = meta_table
            .string("emit")
            .ok_or_else(|| missing("meta.emit"))?;
        let emit = Emit::parse(emit_name).ok_or_else(|| {
            let names: Vec<&str> = Emit::ALL.iter().map(|emit| emit.as_str()).collect();
            unknown("meta.emit", emit_name, &names)
        })?;
        let tool = meta_table
            .string("tool")
            .ok_or_else(|| missing("meta.tool"))?
            .to_string();
        let tool_ver = meta_table.string("tool_ver").unwrap_or("-").to_string();
        let checked = meta_table.string("checked").unwrap_or("-").to_string();

        let mut fingerprint = Vec::new();
        // 两种写法都接受：`[fingerprint]` 下直接写规则（P0 的内置档案），
        // 或 `[fingerprint] match = { ... }`（DESIGN §5.3 的写法）。
        let rules = table
            .table("fingerprint")
            .map(|section| section.table("match").unwrap_or(section));
        if let Some(rules) = rules {
            for (key, value) in rules.entries() {
                let crate::toml::Value::Str(text) = value else {
                    return Err(ProfileError::BadValue {
                        field: format!("fingerprint.match.{key}"),
                        detail: "只接受字符串规则".to_string(),
                    });
                };
                fingerprint.push((key.clone(), MatchRule::parse(text)));
            }
        }
        if fingerprint.is_empty() {
            return Err(missing("fingerprint.match"));
        }

        let mut spaces = Vec::new();
        if let Some(map) = table.table("space") {
            for (key, value) in map.entries() {
                let Some(space) = Space::parse(key) else {
                    let names: Vec<&str> = Space::ALL.iter().map(|space| space.as_str()).collect();
                    return Err(unknown("space", key, &names));
                };
                let crate::toml::Value::Str(name) = value else {
                    return Err(ProfileError::BadValue {
                        field: format!("space.{key}"),
                        detail: "只接受字符串".to_string(),
                    });
                };
                spaces.push((space, name.clone()));
            }
        }
        for space in Space::ALL {
            if !spaces.iter().any(|(have, _)| *have == space) {
                return Err(missing(&format!(
                    "space.{}（三级存储的语义名必须齐全）",
                    space.as_str()
                )));
            }
        }

        let empty = Table::new();
        let mem_table = table.table("mem").unwrap_or(&empty);
        let fast = mem_table.table("FAST");
        let reg = mem_table.table("REG");
        let mem = Mem {
            fast_bytes: fast.and_then(|t| t.int("bytes")).unwrap_or(0).max(0) as u64,
            fast_bank: fast.and_then(|t| t.int("bank")).unwrap_or(0).max(0) as u32,
            reg_bytes: reg.and_then(|t| t.int("bytes")).unwrap_or(0).max(0) as u64,
        };

        let simt_table = table.table("simt");
        let simt = Simt {
            group: simt_table.and_then(|t| t.int("group")).unwrap_or(1).max(1) as u32,
            groups_max: simt_table
                .and_then(|t| t.int("groups_max"))
                .unwrap_or(1)
                .max(1) as u32,
        };

        let mut provides = Vec::new();
        for (index, entry) in table
            .table_array("provides")
            .unwrap_or_default()
            .into_iter()
            .enumerate()
        {
            provides.push(parse_provide(entry, index)?);
        }
        if provides.is_empty() {
            return Err(missing("provides"));
        }

        let peak_table = table.table("peak").unwrap_or(&empty);
        let peak = Peak {
            flops_f32: parse_constant(peak_table, "flops_f32")?,
            bw: parse_constant(peak_table, "bw")?,
        };

        let compile = table.table("compile");
        let flags = compile
            .and_then(|t| t.string_array("flags"))
            .unwrap_or_default()
            .into_iter()
            .map(str::to_string)
            .collect();
        let preamble = compile
            .and_then(|t| t.string("preamble"))
            .unwrap_or("")
            .to_string();

        Ok(Self {
            meta: Meta {
                vendor,
                id,
                emit,
                tool,
                tool_ver,
                checked,
            },
            fingerprint,
            spaces,
            mem,
            simt,
            provides,
            peak,
            flags,
            preamble,
        })
    }

    /// 闭集与见证校验；`witness_root` 为 `None` 时跳过文件存在性检查（仅用于单元测试）。
    pub fn validate(&self, witness_root: Option<&Path>) -> Result<(), Vec<ProfileError>> {
        let mut errors = Vec::new();

        for provide in &self.provides {
            if provide.grade.is_verified() {
                match provide.witness.as_deref() {
                    None => errors.push(ProfileError::Witness {
                        prim: provide.prim,
                        detail: format!("{} 要求 `witness` 字段", provide.grade),
                    }),
                    Some(path) => {
                        if let Some(root) = witness_root {
                            let full = root.join(path);
                            if !full.is_file() {
                                errors.push(ProfileError::Witness {
                                    prim: provide.prim,
                                    detail: format!("见证文件 `{}` 不存在", full.display()),
                                });
                            }
                        }
                    }
                }
            }
            if provide.dt.is_empty() && provide.prim.has_dtype() {
                errors.push(ProfileError::BadValue {
                    field: format!("provides[{}].dt", provide.prim),
                    detail: "数据类型集合不能为空".to_string(),
                });
            }
            if provide.grade.is_verified() && provide.impl_name.is_empty() {
                errors.push(ProfileError::BadValue {
                    field: format!("provides[{}].impl", provide.prim),
                    detail: "实现名不能为空".to_string(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// 语义存储名映射到物理名。
    pub fn physical_space(&self, space: Space) -> Option<&str> {
        self.spaces
            .iter()
            .find(|(have, _)| *have == space)
            .map(|(_, name)| name.as_str())
    }

    /// 指纹是否命中（全部规则满足）。
    pub fn matches(&self, detected: &Detected) -> bool {
        self.fingerprint
            .iter()
            .all(|(key, rule)| detected.get(key).is_some_and(|actual| rule.matches(actual)))
    }

    /// 指纹的具体度：命中的规则条数，多者优先（继承旧工程的 `pick_specific`）。
    pub fn specificity(&self) -> usize {
        self.fingerprint.len()
    }

    /// 某原语的供给（同一原语只允许一条，校验保证）。
    pub fn provider(&self, prim: Prim) -> Option<&Provide> {
        self.provides.iter().find(|provide| provide.prim == prim)
    }

    /// 需求是否被满足：只接受 `L2` 及以上的供给（未验证视为不具备）。
    pub fn satisfies(&self, need: &Need) -> bool {
        self.provides
            .iter()
            .any(|provide| provide.grade.is_verified() && provide.covers(need))
    }
}

fn parse_constant(table: &Table, key: &str) -> Result<Option<Constant>, ProfileError> {
    let Some(inner) = table.table(key) else {
        return Ok(None);
    };
    let value = inner
        .float("value")
        .ok_or_else(|| missing(&format!("peak.{key}.value")))?;
    let src = inner.string("src").unwrap_or("-").to_string();
    Ok(Some(Constant { value, src }))
}

fn parse_provide(entry: &Table, index: usize) -> Result<Provide, ProfileError> {
    let field = |name: &str| format!("provides[{index}].{name}");
    let prim_name = entry
        .string("prim")
        .ok_or_else(|| missing(&field("prim")))?;
    let prim = Prim::parse(prim_name).ok_or_else(|| {
        let names: Vec<&str> = Prim::ALL.iter().map(|prim| prim.as_str()).collect();
        unknown(&field("prim"), prim_name, &names)
    })?;

    let impl_name = entry.string("impl").unwrap_or("-").to_string();
    let mut provide = Provide::of(prim, impl_name);

    if let Some(text) = entry.string("m") {
        provide = provide.with_extent(parse_span(&field("m"), text)?);
    }
    if let Some(text) = entry.string("n") {
        provide = provide.with_n(parse_span(&field("n"), text)?);
    }
    if let Some(text) = entry.string("k") {
        provide = provide.with_k(parse_span(&field("k"), text)?);
    }
    if let Some(items) = entry.string_array("dt") {
        let mut set = DTypeSet::empty();
        for item in items {
            let dt = DType::parse(item).ok_or_else(|| {
                let names: Vec<&str> = DType::ALL.iter().map(|dt| dt.as_str()).collect();
                unknown(&field("dt"), item, &names)
            })?;
            set.insert(dt);
        }
        provide = provide.with_dt(set);
    }
    if let Some(items) = entry.string_array("space") {
        let mut set = SpaceSet::empty();
        for item in items {
            let space = Space::parse(item).ok_or_else(|| {
                let names: Vec<&str> = Space::ALL.iter().map(|space| space.as_str()).collect();
                unknown(&field("space"), item, &names)
            })?;
            set.insert(space);
        }
        provide = provide.with_space(set);
    }
    if let Some(text) = entry.string("grade") {
        let grade = Grade::parse(text).ok_or_else(|| {
            let names: Vec<&str> = Grade::ALL.iter().map(|grade| grade.as_str()).collect();
            unknown(&field("grade"), text, &names)
        })?;
        provide = provide.with_grade(grade);
    } else {
        return Err(missing(&field("grade")));
    }
    if let Some(path) = entry.string("witness") {
        provide = provide.with_witness(path);
    }

    // 其余键一律视为枚举型属性；闭集外的键直接报错，避免"写了但没人看"。
    for (key, value) in entry.entries() {
        if matches!(
            key.as_str(),
            "prim" | "impl" | "m" | "n" | "k" | "dt" | "space" | "grade" | "witness"
        ) {
            continue;
        }
        let Some(attr_key) = AttrKey::parse(key) else {
            let names: Vec<&str> = AttrKey::ALL.iter().map(|key| key.as_str()).collect();
            return Err(unknown(&field(key), key, &names));
        };
        let values: Vec<String> = match value {
            crate::toml::Value::Str(text) => vec![text.clone()],
            crate::toml::Value::Array(items) => items
                .iter()
                .map(|item| match item {
                    crate::toml::Value::Str(text) => Ok(text.clone()),
                    _ => Err(ProfileError::BadValue {
                        field: field(key),
                        detail: "属性数组里只能是字符串".to_string(),
                    }),
                })
                .collect::<Result<_, _>>()?,
            crate::toml::Value::Int(number) => vec![number.to_string()],
            crate::toml::Value::Float(number) => vec![number.to_string()],
            crate::toml::Value::Bool(flag) => vec![flag.to_string()],
            crate::toml::Value::Table(_) => {
                return Err(ProfileError::BadValue {
                    field: field(key),
                    detail: "属性不能是表".to_string(),
                });
            }
        };
        provide = provide.with_attr(Attr::new(attr_key, values));
    }

    Ok(provide)
}

fn parse_span(field: &str, text: &str) -> Result<Span, ProfileError> {
    Span::parse(text).map_err(|detail| ProfileError::BadValue {
        field: field.to_string(),
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPU: &str = include_str!("../devices/cpu.toml");

    #[test]
    fn parses_builtin_cpu_profile() {
        let profile = Profile::parse(CPU).expect("内置 CPU 档案必须可解析");
        assert_eq!(profile.meta.vendor, "cpu");
        assert_eq!(profile.meta.emit, Emit::C);
        assert_eq!(profile.physical_space(Space::Fast), Some("L2"));
        assert_eq!(profile.simt.group, 1);
        assert!(profile.provider(Prim::Dot).is_some());
    }

    #[test]
    fn builtin_cpu_profile_validates_against_witness_root() {
        let profile = Profile::parse(CPU).expect("可解析");
        profile
            .validate(Some(Path::new(env!("CARGO_MANIFEST_DIR"))))
            .expect("CPU 档案的见证文件必须齐全");
    }

    #[test]
    fn missing_emit_is_rejected() {
        let broken = CPU.replace("emit     = \"c\"", "emit     = \"nope\"");
        let error = Profile::parse(&broken).unwrap_err();
        assert!(matches!(error, ProfileError::Unknown { .. }), "{error}");
    }

    #[test]
    fn l2_without_witness_is_rejected() {
        let mut profile = Profile::parse(CPU).expect("可解析");
        for provide in &mut profile.provides {
            provide.witness = None;
        }
        let errors = profile.validate(None).unwrap_err();
        assert!(
            errors
                .iter()
                .all(|e| matches!(e, ProfileError::Witness { .. }))
        );
    }

    #[test]
    fn fingerprint_rules_match_versions_and_sets() {
        assert!(MatchRule::parse("10.0..10.3").matches("10.2"));
        assert!(!MatchRule::parse("10.0..10.3").matches("9.0"));
        assert!(MatchRule::parse("contains: B200|GB200").matches("NVIDIA B200"));
        assert!(!MatchRule::parse("contains: B200|GB200").matches("H100"));
        assert!(MatchRule::parse("x86_64{avx2,avx512}").matches("x86_64+avx2+avx512"));
        assert!(!MatchRule::parse("x86_64{avx2,avx512}").matches("x86_64+avx2"));
        assert!(MatchRule::parse("sm90").matches("sm90"));
        assert!(MatchRule::parse("x86_64|aarch64").matches("aarch64"));
        assert!(!MatchRule::parse("x86_64|aarch64").matches("riscv64"));
    }

    #[test]
    fn profile_matches_detected_fingerprint() {
        let profile = Profile::parse(CPU).expect("可解析");
        let arch = std::env::consts::ARCH;
        let detected = Detected::new()
            .with("arch", arch)
            .with("features", format!("{arch}+avx2"));
        assert!(profile.matches(&detected), "{detected}");
        assert!(!profile.matches(&Detected::new().with("arch", "sparc")));
    }

    #[test]
    fn only_verified_provides_satisfy_needs() {
        let profile = Profile::parse(CPU).expect("可解析");
        let need = Need::of(Prim::Atomic);
        // CPU 档案把 atomic 声明在 L1：未验证视为不具备。
        assert!(!profile.satisfies(&need));
        assert!(profile.provider(Prim::Atomic).is_some());
    }
}
