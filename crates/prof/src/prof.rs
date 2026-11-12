//! 档案层：设备指纹、参数化能力供给、证据等级与字段校验（DESIGN §5.3、§11.2）。
//!
//! 档案是"设备只建档"的那一列：同一份模板之所以能跑在八台设备上，是因为每个变量
//! （存储层物理名、输出语言、工具链、能力区间）都在这份数据里，而不是在代码里。

use crate::atom::{
    Attr, AttrKey, DType, DTypeSet, Grade, Need, Prim, Provide, Space, SpaceSet, Span, Witness,
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

/// 一个执行单元：名字 + 它能访问的存储层（`[units.<name>]`）。
///
/// 名字是数据而不是枚举：同一套机制既服务 SIMT 的 `warp`，也服务昇腾的 `cube`/`vector`，
/// 代码里没有任何一个具体单元名。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// 单元名，由档案定义。
    pub name: String,
    /// 该单元访问得到的存储层。
    pub space: SpaceSet,
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
    /// 执行单元及其可访问的存储层；为空表示档案不区分单元。
    pub units: Vec<Unit>,
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

        let mut units = Vec::new();
        if let Some(section) = table.table("units") {
            for (name, value) in section.entries() {
                let crate::toml::Value::Table(inner) = value else {
                    return Err(ProfileError::BadValue {
                        field: format!("units.{name}"),
                        detail: "只接受 `[units.<name>]` 形式的表".to_string(),
                    });
                };
                let items = inner
                    .string_array("space")
                    .ok_or_else(|| missing(&format!("units.{name}.space")))?;
                units.push(Unit {
                    name: name.clone(),
                    space: parse_space_set(&format!("units.{name}.space"), &items)?,
                });
            }
        }

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
            units,
            provides,
            peak,
            flags,
            preamble,
        })
    }

    /// 闭集与见证校验；`witness_root` 为 `None` 时跳过文件存在性检查（仅用于单元测试）。
    pub fn validate(&self, witness_root: Option<&Path>) -> Result<(), Vec<ProfileError>> {
        let mut errors = Vec::new();

        // 同一份见证源码只读一次：一台设备的 `provides` 通常共用一个文件。
        let mut sources: BTreeMap<String, Option<String>> = BTreeMap::new();

        // 同一原语的每条签名必须有不同 `impl`：同名的两条供给无法区分"谁实现了什么"，
        // 协商也只会取到其中一条。
        let mut seen: BTreeMap<(Prim, &str), usize> = BTreeMap::new();
        for (index, provide) in self.provides.iter().enumerate() {
            if let Some(first) = seen.insert((provide.prim, provide.impl_name.as_str()), index) {
                errors.push(ProfileError::BadValue {
                    field: format!("provides[{index}].impl"),
                    detail: format!(
                        "`{}` 的实现名与 provides[{first}] 重名：同一原语的多条签名必须用不同 impl 区分",
                        provide.prim
                    ),
                });
            }
        }

        for provide in &self.provides {
            match provide.witness.as_ref() {
                // L2 及以上必须有见证：等级是执行的结论，不是自述。
                None => {
                    if provide.grade.is_verified() {
                        errors.push(ProfileError::Witness {
                            prim: provide.prim,
                            detail: format!(
                                "{} 要求 `witness = {{ file, func, expect }}`",
                                provide.grade
                            ),
                        });
                    }
                }
                // 声明了见证就校验它，即使等级还是 L0/L1：证据可以先落成数据，
                // 文件名/函数名写错不该等到升 L2 才暴露。
                Some(witness) => {
                    if witness.func.is_empty() {
                        errors.push(ProfileError::Witness {
                            prim: provide.prim,
                            detail: "见证必须给出入口函数名，否则无法执行".to_string(),
                        });
                    }
                    if witness.expect.is_empty() {
                        errors.push(ProfileError::Witness {
                            prim: provide.prim,
                            detail: "见证必须给出期望输出，否则无法对拍".to_string(),
                        });
                    }
                    if witness.expect.len() > Witness::MAX_OUT {
                        errors.push(ProfileError::Witness {
                            prim: provide.prim,
                            detail: format!(
                                "期望输出 {} 个，超过上限 {}",
                                witness.expect.len(),
                                Witness::MAX_OUT
                            ),
                        });
                    }
                    if let Some(root) = witness_root {
                        let text = sources.entry(witness.file.clone()).or_insert_with(|| {
                            std::fs::read_to_string(root.join(&witness.file)).ok()
                        });
                        match text.as_deref() {
                            None => errors.push(ProfileError::Witness {
                                prim: provide.prim,
                                detail: format!(
                                    "见证文件 `{}` 不存在或不可读",
                                    root.join(&witness.file).display()
                                ),
                            }),
                            Some(text) if !text.contains(&format!("{}(", witness.func)) => {
                                errors.push(ProfileError::Witness {
                                    prim: provide.prim,
                                    detail: format!(
                                        "`{}` 里没有函数 `{}`：声明的见证无法执行",
                                        witness.file, witness.func
                                    ),
                                });
                            }
                            Some(_) => {}
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

        // `exec` 的取值必须对得上本档案声明的执行单元，且该单元的存储层必须覆盖这条供给：
        // "Cube 只能访问 L1/L0"因此是一条可校验的声明，而不是一句注释。
        let unit_names: Vec<&str> = self.units.iter().map(|unit| unit.name.as_str()).collect();
        for provide in &self.provides {
            let Some(exec) = provide
                .attrs
                .iter()
                .find(|attr| attr.key() == AttrKey::Exec)
            else {
                continue;
            };
            let field = format!("provides[{}].exec", provide.prim);
            if self.units.is_empty() {
                errors.push(ProfileError::BadValue {
                    field,
                    detail: "写 `exec` 前必须先用 `[units.<name>]` 声明执行单元".to_string(),
                });
                continue;
            }
            let mut reachable = SpaceSet::empty();
            for value in exec.values() {
                match self.units.iter().find(|unit| unit.name == *value) {
                    Some(unit) => {
                        for space in Space::ALL {
                            if unit.space.contains(space) {
                                reachable.insert(space);
                            }
                        }
                    }
                    None => errors.push(ProfileError::Unknown {
                        field: field.clone(),
                        value: value.clone(),
                        allowed: unit_names.join(" | "),
                    }),
                }
            }
            if !provide.space.is_subset(reachable) {
                errors.push(ProfileError::BadValue {
                    field: format!("provides[{}].space", provide.prim),
                    detail: format!(
                        "`{}` 访问不到 [{}]",
                        exec.values().join("+"),
                        provide.space.difference(reachable)
                    ),
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

    /// 某原语的全部供给。同一原语可以有多条：一条签名只能描述一种实现，
    /// 而 `dot` 的 `dp4a` 与标量 `fma` 覆盖的是不同的 `dt`/`lanes`，合并成一条就只剩少写字段。
    pub fn providers(&self, prim: Prim) -> impl Iterator<Item = &Provide> {
        self.provides
            .iter()
            .filter(move |provide| provide.prim == prim)
    }

    /// 需求是否被满足：只接受 `L2` 及以上的供给（未验证视为不具备）。
    pub fn satisfies(&self, need: &Need) -> bool {
        self.provides
            .iter()
            .any(|provide| provide.grade.is_verified() && provide.covers(need))
    }
}

/// 峰值常数：`value` 必须等于 `src` 这段推导式的结果。
///
/// 抄来的数字无法核验；带变量名的公式也无法核验（本层没有变量表）。所以闭集只接受
/// 数值字面量与 `+ - * /`、括号组成的表达式——量纲与物理含义写在该行的 TOML 注释里。
/// 这样 `value` 与 `src` 才有唯一事实源：改一个而不改另一个会在装载期失败（DESIGN §12.1）。
fn parse_constant(table: &Table, key: &str) -> Result<Option<Constant>, ProfileError> {
    let Some(inner) = table.table(key) else {
        return Ok(None);
    };
    let field = format!("peak.{key}");
    let value = inner
        .float("value")
        .ok_or_else(|| missing(&format!("{field}.value")))?;
    let src = inner
        .string("src")
        .ok_or_else(|| missing(&format!("{field}.src")))?
        .to_string();
    let derived = eval_expression(&src).map_err(|detail| ProfileError::BadValue {
        field: format!("{field}.src"),
        detail: format!("推导式 `{src}` 求值失败：{detail}"),
    })?;
    // 允许字面量的常规舍入（如 `2.1e12`），但容不下 10% 量级的"抄错一位"。
    let tolerance = value.abs().max(1.0) * 1e-6;
    if (derived - value).abs() > tolerance {
        return Err(ProfileError::BadValue {
            field: format!("{field}.src"),
            detail: format!("推导式 `{src}` = {derived:e}，与 value = {value:e} 不一致"),
        });
    }
    Ok(Some(Constant { value, src }))
}

/// 求值一个只用数值字面量、`+ - * /` 与括号的推导式。
fn eval_expression(text: &str) -> Result<f64, String> {
    let bytes = text.as_bytes();
    let mut cursor = 0usize;
    let value = eval_sum(text, bytes, &mut cursor)?;
    skip_spaces(bytes, &mut cursor);
    if cursor != bytes.len() {
        return Err(format!("位置 {cursor} 起有多余内容"));
    }
    Ok(value)
}

fn skip_spaces(bytes: &[u8], cursor: &mut usize) {
    while bytes
        .get(*cursor)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        *cursor += 1;
    }
}

fn eval_sum(text: &str, bytes: &[u8], cursor: &mut usize) -> Result<f64, String> {
    let mut value = eval_product(text, bytes, cursor)?;
    loop {
        skip_spaces(bytes, cursor);
        match bytes.get(*cursor) {
            Some(b'+') => {
                *cursor += 1;
                value += eval_product(text, bytes, cursor)?;
            }
            Some(b'-') => {
                *cursor += 1;
                value -= eval_product(text, bytes, cursor)?;
            }
            _ => return Ok(value),
        }
    }
}

fn eval_product(text: &str, bytes: &[u8], cursor: &mut usize) -> Result<f64, String> {
    let mut value = eval_atom(text, bytes, cursor)?;
    loop {
        skip_spaces(bytes, cursor);
        match bytes.get(*cursor) {
            Some(b'*') => {
                *cursor += 1;
                value *= eval_atom(text, bytes, cursor)?;
            }
            Some(b'/') => {
                *cursor += 1;
                let divisor = eval_atom(text, bytes, cursor)?;
                if divisor == 0.0 {
                    return Err("除数为 0".to_string());
                }
                value /= divisor;
            }
            _ => return Ok(value),
        }
    }
}

fn eval_atom(text: &str, bytes: &[u8], cursor: &mut usize) -> Result<f64, String> {
    skip_spaces(bytes, cursor);
    match bytes.get(*cursor) {
        Some(b'(') => {
            *cursor += 1;
            let value = eval_sum(text, bytes, cursor)?;
            skip_spaces(bytes, cursor);
            if bytes.get(*cursor) != Some(&b')') {
                return Err("括号没有闭合".to_string());
            }
            *cursor += 1;
            Ok(value)
        }
        Some(byte) if byte.is_ascii_digit() || *byte == b'.' => {
            let start = *cursor;
            let mut exponent = false;
            while let Some(&byte) = bytes.get(*cursor) {
                match byte {
                    b'0'..=b'9' | b'.' => *cursor += 1,
                    b'e' | b'E' if !exponent => {
                        exponent = true;
                        *cursor += 1;
                        if bytes
                            .get(*cursor)
                            .is_some_and(|sign| matches!(sign, b'+' | b'-'))
                        {
                            *cursor += 1;
                        }
                    }
                    _ => break,
                }
            }
            text[start..*cursor].parse::<f64>().map_err(|error| {
                format!(
                    "位置 {start} 的数字 `{}` 非法：{error}",
                    &text[start..*cursor]
                )
            })
        }
        Some(byte) => Err(format!(
            "位置 {cursor} 处不允许字符 `{}`",
            char::from(*byte)
        )),
        None => Err("表达式意外结束".to_string()),
    }
}

fn parse_space_set(field: &str, items: &[&str]) -> Result<SpaceSet, ProfileError> {
    let mut set = SpaceSet::empty();
    for item in items {
        let space = Space::parse(item).ok_or_else(|| {
            let names: Vec<&str> = Space::ALL.iter().map(|space| space.as_str()).collect();
            unknown(field, item, &names)
        })?;
        set.insert(space);
    }
    Ok(set)
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
        provide = provide.with_space(parse_space_set(&field("space"), &items)?);
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
    if let Some(table) = entry.table("witness") {
        let file = table
            .string("file")
            .ok_or_else(|| missing(&field("witness.file")))?
            .to_string();
        let func = table
            .string("func")
            .ok_or_else(|| missing(&field("witness.func")))?
            .to_string();
        let items = table
            .array("expect")
            .ok_or_else(|| missing(&field("witness.expect")))?;
        let expect = items
            .iter()
            .map(|value| match value {
                crate::toml::Value::Float(number) => Ok(*number as f32),
                crate::toml::Value::Int(number) => Ok(*number as f32),
                _ => Err(ProfileError::BadValue {
                    field: field("witness.expect"),
                    detail: "只能是数值".to_string(),
                }),
            })
            .collect::<Result<Vec<f32>, _>>()?;
        provide = provide.with_witness(Witness { file, func, expect });
    } else if entry.get("witness").is_some() {
        return Err(ProfileError::BadValue {
            field: field("witness"),
            detail: "必须是内联表 `{ file, func, expect }`：没有入口函数与期望值的见证无法执行"
                .to_string(),
        });
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

    /// 内置 CPU 档案原文。
    ///
    /// 事实源在 `prof`，本模块只解析内容。
    fn cpu() -> &'static str {
        crate::embedded("cpu").expect("内置 cpu 档案")
    }

    #[test]
    fn parses_builtin_cpu_profile() {
        let profile = Profile::parse(cpu()).expect("内置 CPU 档案必须可解析");
        assert_eq!(profile.meta.vendor, "cpu");
        assert_eq!(profile.meta.emit, Emit::C);
        assert_eq!(profile.physical_space(Space::Fast), Some("L2"));
        assert_eq!(profile.simt.group, 1);
        assert!(profile.providers(Prim::Dot).next().is_some());
    }

    #[test]
    fn builtin_cpu_profile_is_schema_valid() {
        let profile = Profile::parse(cpu()).expect("可解析");
        // 见证文件属于片段层（`crates/prim`），本 crate 只校验 schema 与 `[units]`；
        // "见证真的存在"由 `cutile` 的 `Catalogue::validate(Some(root))` 负责。
        profile.validate(None).expect("CPU 档案必须自洽");
    }

    #[test]
    fn missing_emit_is_rejected() {
        let broken = cpu().replace("emit     = \"c\"", "emit     = \"nope\"");
        let error = Profile::parse(&broken).unwrap_err();
        assert!(matches!(error, ProfileError::Unknown { .. }), "{error}");
    }

    #[test]
    fn l2_without_witness_is_rejected() {
        let mut profile = Profile::parse(cpu()).expect("可解析");
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
    fn a_peak_must_equal_its_own_formula() {
        // 108.8e9 是旧档案里的手抄值：与 8*2.6e9*8*2 差了三倍，装载期必须报出来。
        let broken = cpu().replacen("value = 332.8e9", "value = 108.8e9", 1);
        let error = Profile::parse(&broken).expect_err("value 与 src 不一致必须是错误");
        assert!(error.to_string().contains("不一致"), "{error}");
    }

    #[test]
    fn a_peak_formula_with_a_unit_name_is_rejected() {
        let broken = cpu().replacen(
            "src = \"8*2.6e9*8*2\"",
            "src = \"8 cores * 2.6 GHz * 8 lanes * 2 flop\"",
            1,
        );
        let error = Profile::parse(&broken).expect_err("带单位的推导式无法核验");
        assert!(error.to_string().contains("求值失败"), "{error}");
    }

    #[test]
    fn formulas_are_four_operations_with_parentheses() {
        assert!((eval_expression("2*3+4").expect("可求值") - 10.0).abs() < 1e-9);
        assert!((eval_expression("(1+2)*3/4").expect("可求值") - 2.25).abs() < 1e-9);
        assert!((eval_expression("2*4096/8*8e9").expect("可求值") - 8.192e12).abs() < 1.0);
        assert!(eval_expression("1/0").is_err());
        assert!(eval_expression("8*").is_err());
        assert!(eval_expression("(1").is_err());
        assert!(eval_expression("2 3").is_err());
    }

    #[test]
    fn a_witness_is_file_plus_function_plus_expectation() {
        let profile = Profile::parse(cpu()).expect("可解析");
        let witness = profile
            .providers(Prim::Quant)
            .find_map(|provide| provide.witness.as_ref())
            .expect("quant 的见证");
        assert_eq!(witness.file, "cpu/witness.c");
        assert_eq!(witness.func, "witness_quant");
        assert_eq!(witness.expect, vec![-4.0, 3.5]);
    }

    #[test]
    fn a_witness_path_without_a_function_is_rejected() {
        let broken = cpu().replacen(
            "{ file = \"cpu/witness.c\", func = \"witness_quant\", expect = [-4.0, 3.5] }",
            "\"cpu/witness.c\"",
            1,
        );
        let error = Profile::parse(&broken).expect_err("只有路径的见证必须是错误");
        assert!(error.to_string().contains("内联表"), "{error}");
    }

    #[test]
    fn an_empty_expectation_is_rejected() {
        let mut profile = Profile::parse(cpu()).expect("可解析");
        profile.provides[0]
            .witness
            .as_mut()
            .expect("见证")
            .expect
            .clear();
        let errors = profile.validate(None).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("期望输出")),
            "{errors:?}"
        );
    }

    #[test]
    fn the_witness_function_must_exist_in_the_file() {
        let dir = std::env::temp_dir().join(format!("prof-witness-{}", std::process::id()));
        let vendor = dir.join("cpu");
        std::fs::create_dir_all(&vendor).expect("建临时目录");
        std::fs::write(
            vendor.join("witness.c"),
            "int witness_dot(float *out) { out[0] = 70.0f; return 1; }\n",
        )
        .expect("写见证");
        let mut profile = Profile::parse(cpu()).expect("可解析");
        profile.provides[0].witness.as_mut().expect("见证").func = "witness_absent".to_string();
        let errors = profile.validate(Some(&dir)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("witness_absent")),
            "{errors:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 见证可以先落成数据：等级还是 L0/L1 时，声明过的文件与函数名同样要校验，
    /// 否则命名写错会一直潜伏到升 L2 才暴露。
    #[test]
    fn a_declared_witness_is_validated_below_l2() {
        let dir = std::env::temp_dir().join(format!("prof-witness-l1-{}", std::process::id()));
        let vendor = dir.join("nvidia");
        std::fs::create_dir_all(&vendor).expect("建临时目录");
        std::fs::write(
            vendor.join("witness.cu"),
            "int witness_dot(float *out) { out[0] = 70.0f; return 1; }\n",
        )
        .expect("写见证");
        let mut profile =
            Profile::parse(crate::embedded("nvidia").expect("内置 nvidia 档案")).expect("可解析");
        // 只留一条声明，其余见证置空（L1 允许没有见证），这样错误只指向被测的那条。
        for provide in &mut profile.provides {
            if provide.prim != Prim::Dot {
                provide.witness = None;
            }
        }
        let dot = profile
            .provides
            .iter_mut()
            .find(|provide| provide.prim == Prim::Dot)
            .expect("dot 供给");
        assert_eq!(dot.grade, Grade::L1, "前提：这条签名还没到 L2");
        dot.witness.as_mut().expect("见证").func = "witness_absent".to_string();
        let errors = profile.validate(Some(&dir)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("witness_absent")),
            "{errors:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
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
        let profile = Profile::parse(cpu()).expect("可解析");
        let arch = std::env::consts::ARCH;
        let detected = Detected::new()
            .with("arch", arch)
            .with("features", format!("{arch}+avx2"));
        assert!(profile.matches(&detected), "{detected}");
        assert!(!profile.matches(&Detected::new().with("arch", "sparc")));
    }

    #[test]
    fn only_verified_provides_satisfy_needs() {
        let profile = Profile::parse(cpu()).expect("可解析");
        let need = Need::of(Prim::Atomic);
        // CPU 档案把 atomic 声明在 L1：未验证视为不具备。
        assert!(!profile.satisfies(&need));
        assert!(profile.providers(Prim::Atomic).next().is_some());
    }

    /// 昇腾档案原文：执行单元与"哪一层只有谁能访问"全在数据里。
    fn ascend() -> &'static str {
        crate::embedded("ascend").expect("内置 ascend 档案")
    }

    fn ascend_profile() -> Profile {
        let profile = Profile::parse(ascend()).expect("内置 ascend 档案必须可解析");
        profile.validate(None).expect("内置 ascend 档案必须自洽");
        profile
    }

    fn exec(values: &[&str]) -> Attr {
        Attr::new(AttrKey::Exec, values.iter().copied())
    }

    #[test]
    fn parses_builtin_ascend_profile_with_units() {
        let profile = ascend_profile();
        assert_eq!(profile.meta.vendor, "ascend");
        assert_eq!(profile.meta.emit, Emit::Ascendc);
        assert_eq!(profile.physical_space(Space::Fast), Some("L1"));
        assert_eq!(profile.physical_space(Space::Reg), Some("L0C"));

        let cube = profile
            .units
            .iter()
            .find(|unit| unit.name == "cube")
            .expect("cube 单元");
        assert!(cube.space.contains(Space::Fast));
        assert!(cube.space.contains(Space::Reg));
        assert!(!cube.space.contains(Space::Gm), "Cube 访问不到 GM");

        let dot = profile
            .providers(Prim::Dot)
            .find(|provide| provide.attrs.iter().any(|attr| attr.key() == AttrKey::Exec))
            .expect("dot 供给");
        let exec = dot
            .attrs
            .iter()
            .find(|attr| attr.key() == AttrKey::Exec)
            .expect("dot 必须声明执行单元");
        assert_eq!(exec.values(), ["cube".to_string()]);
    }

    #[test]
    fn exec_must_name_a_declared_unit() {
        let broken = ascend().replace("exec    = [\"cube\"]", "exec    = [\"cube0\"]");
        let profile = Profile::parse(&broken).expect("解析期只做闭集校验");
        let errors = profile.validate(None).unwrap_err();
        assert!(
            errors.iter().any(|error| matches!(
                error,
                ProfileError::Unknown { value, .. } if value == "cube0"
            )),
            "{errors:?}"
        );
    }

    #[test]
    fn a_unit_cannot_reach_a_space_it_does_not_have() {
        let broken = ascend().replace(
            "impl    = \"mmad\"\ndt      = [\"f16\", \"bf16\"]\nspace   = [\"FAST\", \"REG\"]",
            "impl    = \"mmad\"\ndt      = [\"f16\", \"bf16\"]\nspace   = [\"FAST\", \"REG\", \"GM\"]",
        );
        let profile = Profile::parse(&broken).expect("`GM` 是闭集内的存储名");
        let errors = profile.validate(None).unwrap_err();
        assert!(
            errors.iter().any(|error| matches!(
                error,
                ProfileError::BadValue { field, detail }
                    if field == "provides[dot].space" && detail.contains("GM")
            )),
            "{errors:?}"
        );
    }

    #[test]
    fn one_primitive_can_have_several_signatures_but_not_the_same_impl_twice() {
        let mut profile = Profile::parse(cpu()).expect("可解析");
        let mut extra = profile.provides[0].clone();
        extra.grade = Grade::L1;
        extra.witness = None;
        // 换一个 impl 名就是另一条签名：同一原语下允许并存。
        extra.impl_name = "scalar_i4_wide".to_string();
        profile.provides.push(extra.clone());
        profile
            .validate(None)
            .expect("不同 impl 的两条签名可以并存");
        // 同名 = 无法区分谁实现了什么。
        profile.provides.push(extra);
        let errors = profile.validate(None).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("重名")),
            "{errors:?}"
        );
    }

    #[test]
    fn a_need_can_demand_an_execution_unit() {
        let profile = ascend_profile();
        let dots: Vec<&Provide> = profile.providers(Prim::Dot).collect();
        assert!(
            dots.iter()
                .any(|dot| dot.covers(&Need::of(Prim::Dot).with_attr(exec(&["cube"]))))
        );
        // 档案里没有任何一条 dot 由 vector 实现：需求不能被"另一条签名"顶替。
        let vector = Need::of(Prim::Dot).with_attr(exec(&["vector"]));
        assert!(dots.iter().all(|dot| !dot.covers(&vector)));
        assert!(dots.iter().all(|dot| dot.diff(&vector).len() == 1));
        // 只有 L0：形状与单元都对得上，但仍然"不具备"。
        assert!(!profile.satisfies(&Need::of(Prim::Dot).with_attr(exec(&["cube"]))));
    }
}
