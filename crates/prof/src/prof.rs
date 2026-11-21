//! 档案层：设备指纹、参数化能力供给、证据等级与字段校验（cutile README §5.3、§11.2）。
//!
//! 本模块只放数据类型与它们与判据无关的读取方法；解析在 `parse`，校验在 `check`。
use crate::atom::{
    Attr, AttrKey, Closed, DType, DTypeSet, Grade, Need, Prim, Provide, Space, SpaceSet, Span,
    Witness,
};
use core::fmt;
use diagnostic::data::{Table, Value};
use diagnostic::{Code, Error};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
/// 输出语言闭集（cutile README §5.4 的 `emit` 列）。设备与输出语言是两个维度：
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
impl Closed for Emit {
    /// 闭集全量；顺序即 `Emit::SPEC` 的下标，也就是变体判别值。
    const ALL: &'static [Self] = &[
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
    fn as_str(self) -> &'static str {
        Self::SPEC[self as usize].0
    }
    fn index(self) -> usize {
        self as usize
    }
}
crate::inherent_names!(Emit);
impl Emit {
    /// 名字与源文件扩展名，一行一个变体（顺序与 [`Emit::ALL`] 一致）：
    /// 名字与后缀是同一份事实，写成一张表，改名时不会漏掉 `ext`。
    const SPEC: &'static [(&'static str, &'static str)] = &[
        ("cuda", "cu"),
        ("ptx", "ptx"),
        ("hip", "hip"),
        ("opencl", "cl"),
        ("spirv", "spv"),
        ("metal", "metal"),
        ("ascendc", "cpp"),
        ("c", "c"),
        ("wgsl", "wgsl"),
    ];
    /// 生成的源文件扩展名（只回答后缀，不回答这种语言能否编出产物）。
    pub const fn ext(self) -> &'static str {
        Self::SPEC[self as usize].1
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
    /// 编译工具名（命令名，`cc` / `nvcc` / `xcrun` …）。
    pub tool: String,
    /// **登记**的工具版本：证据是在哪个工具链上取得的，与 `checked` 同一性质；本机实测值由
    /// `cutile::toolchain::Toolchain` 从 `--version` 解析，两者不必相等，所以不参与缓存键与判定。
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
    /// 是否匹配一个实际探测值：只比较一个键的取值，不判断其他规则、也不看档案。
    pub fn matches(&self, actual: &str) -> bool {
        match self {
            Self::Exact(expected) => actual == expected,
            Self::Contains(tokens) => tokens.iter().any(|token| actual.contains(token)),
            Self::Set { base, members } => set_matches(base, members, actual),
            Self::OneOf(branches) => branches.iter().any(|branch| branch.matches(actual)),
            Self::Range { lo, hi } => within(lo, hi, actual),
        }
    }
}
fn set_matches(base: &str, members: &[String], actual: &str) -> bool {
    let hit = |member: &String| actual.contains(member.as_str());
    actual.starts_with(base) && members.iter().all(hit)
}
/// 版本区间：两端都必须能解析成版本号，实际值落在闭区间内。
fn within(lo: &str, hi: &str, actual: &str) -> bool {
    let Some(value) = parse_version(actual) else {
        return false;
    };
    parse_version(lo).is_some_and(|low| low <= value)
        && parse_version(hi).is_some_and(|high| value <= high)
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
    let parts = text.trim().split('.');
    parts.map(|part| part.parse().ok()).collect()
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
        let pair = |(key, value): (&String, &String)| format!("{key}={value}");
        let parts: Vec<String> = self.attrs.iter().map(pair).collect();
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
    /// 档案是否写了 `[simt]` 段。缺段不等于"没有并行"：昇腾的 lane 交换由向量单元
    /// （`vgather`/`vbroadcast`）承担，本来就不是 SIMT，所以默认值 1 不是"单 lane"的证据。
    pub declared: bool,
}
/// 一个带推导式出处的常数（cutile README §12.1）。
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
/// 一个执行单元：名字 + 它能访问的存储层（`[units.<name>]`）。名字是数据而不是枚举：
/// 同一套机制既服务 SIMT 的 `warp`，也服务昇腾的 `cube`/`vector`。
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
    /// 工具链头文件垫片：文件名 → 内容（`[compile.headers]`）。
    ///
    /// 生成的程序只写一种头文件名（`<cuda_runtime.h>`），而不同工具链的名字不同（MUSA 只有
    /// `<musa_runtime.h>`）。把"缺哪个头、缺的那份长什么样"登记在档案里，编译入口才有依据落地，
    /// 不必让每个后端各写一份——头文件名也是设备知识。
    pub headers: Vec<(String, String)>,
}
/// 档案加载 / 校验错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
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
    /// 见证文件不在或读不出来（给定了片段层根时）。
    WitnessFile {
        /// 路径（相对片段层根）。
        file: String,
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
            Self::Missing { field } => write!(f, "缺字段 `{field}`"),
            Self::Unknown {
                field,
                value,
                allowed,
            } => write!(f, "`{field} = {value}` 不在闭集内（闭集：{allowed}）"),
            Self::Witness { prim, detail } => write!(f, "`{prim}` 的见证不成立：{detail}"),
            Self::WitnessFile { file, detail } => write!(f, "见证文件 `{file}` {detail}"),
            Self::BadValue { field, detail } => write!(f, "`{field}` 非法：{detail}"),
        }
    }
}
impl ProfileError {
    /// 这条诊断属于哪一类：声明与数字写错都是 `invalid`，而“见证文件不在”是 `missing`
    /// ——前者是档案的问题，后者是片段层没装到位，脚本里该分开处理。
    pub fn code(&self) -> Code {
        let missing_file = matches!(self, Self::WitnessFile { .. });
        if missing_file {
            Code::Missing
        } else {
            Code::Invalid
        }
    }
}
impl From<ProfileError> for Error {
    /// 消息与 [`ProfileError`] 的 `Display` 逐字相同，只补上类别与结构化字段。
    fn from(error: ProfileError) -> Self {
        Self::new(error.code(), error.to_string())
    }
}
impl std::error::Error for ProfileError {}
fn missing(field: &str) -> ProfileError {
    ProfileError::Missing {
        field: field.to_string(),
    }
}
/// 闭集外取值；`allowed` 是闭集成员的名字表。
fn unknown(field: &str, value: &str, allowed: &[&str]) -> ProfileError {
    ProfileError::Unknown {
        field: field.to_string(),
        value: value.to_string(),
        allowed: allowed.join(" | "),
    }
}
/// 字段非法：`detail` 说明为什么非法，不重复字段名。
fn bad(field: impl Into<String>, detail: impl Into<String>) -> ProfileError {
    ProfileError::BadValue {
        field: field.into(),
        detail: detail.into(),
    }
}
/// 证据不成立：`prim` 是被考察的那条供给。
fn witness_error(provide: &Provide, detail: impl Into<String>) -> ProfileError {
    ProfileError::Witness {
        prim: provide.prim,
        detail: detail.into(),
    }
}
/// 闭集成员的名字表；诊断里的「闭集：a | b」按声明顺序给出。
fn closed_names<E: Closed>(all: &'static [E]) -> Vec<&'static str> {
    all.iter().map(|item| item.as_str()).collect()
}
/// 数据类型闭集的名字表：`DType` 是 L1 外部类型，用它自己的 `name()`。
fn dtype_names() -> Vec<&'static str> {
    DType::ALL.iter().map(|dt| dt.name()).collect()
}
/// 解析闭集里的名字；闭集外一律 `Unknown`，并把闭集内容一起报出来。
fn closed<E: Closed>(name: &str, path: &str) -> Result<E, ProfileError> {
    E::parse(name).ok_or_else(|| unknown(path, name, &closed_names(E::ALL)))
}
fn at(index: usize, key: &str) -> String {
    format!("provides[{index}].{key}")
}
fn witness_at(index: usize, key: &str) -> String {
    format!("provides[{index}].witness.{key}")
}
fn prim_at(prim: Prim, key: &str) -> String {
    format!("provides[{prim}].{key}")
}
fn text<'a>(table: &'a Table, key: &str, path: &str) -> Result<&'a str, ProfileError> {
    table.string(key).ok_or_else(|| missing(path))
}
fn or_dash(table: &Table, key: &str) -> String {
    table.string(key).unwrap_or("-").to_string()
}
fn list<'a>(table: &'a Table, key: &str, path: &str) -> Result<Vec<&'a str>, ProfileError> {
    table.string_array(key).ok_or_else(|| missing(path))
}
/// 逐项解析一张表的每个键值对：解析段里所有"按键"段落共用这一种形状。
fn entries<T>(
    table: &Table,
    each: impl Fn(&str, &Value) -> Result<T, ProfileError>,
) -> Result<Vec<T>, ProfileError> {
    table.entries().iter().map(|(k, v)| each(k, v)).collect()
}
/// 标量的字符串渲染：字符串原样，整数/浮点/布尔按十进制写；数组与表不是标量，返回 `None`
/// （错误由调用方按自己的字段路径给）。档案层与声明层的标量通道都走这一条。
pub fn scalar_texts(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Str(text) => Some(vec![text.clone()]),
        Value::Int(number) => Some(vec![number.to_string()]),
        Value::Float(number) => Some(vec![number.to_string()]),
        Value::Bool(flag) => Some(vec![flag.to_string()]),
        _ => None,
    }
}
/// 主维度只接受区间字符串。
const EXTENT_DETAIL: &str = "主维度只能是区间字符串（如 `<=4096`）";
/// `n` / `k` 只接受区间字符串。
const AXIS_DETAIL: &str = "只能是区间字符串（如 `16`、`16..64`）";
/// `[units.<name>]` 的取值必须是表。
const UNIT_DETAIL: &str = "只接受 `[units.<name>]` 形式的表";
/// `witness` 必须是内联表。
const WITNESS_DETAIL: &str =
    "必须是内联表 `{ file, func, expect }`：没有入口函数与期望值的见证无法执行";
impl Profile {
    /// 语义存储名映射到物理名。
    pub fn physical_space(&self, space: Space) -> Option<&str> {
        let found = self.spaces.iter().find(|pair| pair.0 == space);
        found.map(|(_, name)| name.as_str())
    }
    /// 指纹是否命中（全部规则满足）。
    pub fn matches(&self, detected: &Detected) -> bool {
        let hit =
            |(key, rule): &(String, MatchRule)| detected.get(key).is_some_and(|v| rule.matches(v));
        self.fingerprint.iter().all(hit)
    }
    /// 指纹的具体度：命中的规则条数，多者优先（继承旧工程的 `pick_specific`）。
    pub fn specificity(&self) -> usize {
        self.fingerprint.len()
    }
    /// 某原语的全部供给。同一原语可以有多条：一条签名只能描述一种实现，
    /// 而 `dot` 的 `dp4a` 与标量 `fma` 覆盖的是不同的 `dt`/`lanes`，合并成一条就只剩少写字段。
    pub fn providers(&self, prim: Prim) -> impl Iterator<Item = &Provide> {
        self.provides.iter().filter(move |p| p.prim == prim)
    }
    /// 需求是否被满足：只接受 `L2` 及以上的供给（未验证视为不具备）。
    pub fn satisfies(&self, need: &Need) -> bool {
        let ok = |p: &Provide| p.grade.is_verified() && p.covers(need);
        self.provides.iter().any(ok)
    }
}

/// 解析全部内置档案；构建期已过校验，解析失败只可能是表结构漂移，在此过滤掉。
///
/// 编排层（组合根）探测本机后把这份列表交给 [`select`]；档案清单从这里取一份，
/// 不在调用方重复 `names → embedded → parse` 三段遍历。
///
/// 结果在进程内只算一次：档案表是编译期内嵌的只读数据，每次调用都重新解析等于把同一份
/// 字符串重新拼成几十个 `String`/`Vec`。返回 `&'static` 让调用方不必为"再看一眼"付克隆代价
/// （设备 crate 的用例每条都取一次档案）。
pub fn builtin_profiles() -> &'static [Profile] {
    static PROFILES: OnceLock<Vec<Profile>> = OnceLock::new();
    PROFILES.get_or_init(|| {
        crate::names()
            .iter()
            .filter_map(|name| crate::embedded(name))
            .filter_map(|table| Profile::parse(table).ok())
            .collect()
    })
}

/// 按探测指纹从一组档案里挑出唯一命中、且规则最具体的一份。
///
/// 命中规则条数（[`Profile::specificity`]）最多者胜出；与最优者同等具体的另一份也算
/// 配置错误——“选谁”没有唯一答案就不该静默挑一份。没有任何档案命中时返回
/// [`Code::Unsupported`]。选择判定是档案层职责，机制层只传入解析好的档案列表。
pub fn select<'a>(profiles: &'a [Profile], detected: &Detected) -> diagnostic::Result<&'a Profile> {
    let Some((index, profile)) = profiles
        .iter()
        .enumerate()
        .filter(|(_, profile)| profile.matches(detected))
        .max_by_key(|(_, profile)| profile.specificity())
    else {
        return Err(
            Error::new(Code::Unsupported, format!("没有档案匹配该指纹：{detected}"))
                .with("prof::select"),
        );
    };
    match rival(profiles, detected, index, profile.specificity()) {
        Some(other) => Err(Error::new(
            Code::Invalid,
            format!(
                "`{}` 与 `{}` 的指纹同等具体，无法决定用哪一份（指纹：{detected}）",
                profile.meta.vendor, other.meta.vendor
            ),
        )
        .with("prof::select")),
        None => Ok(profile),
    }
}

/// 与最优者同等具体、同样命中的另一份档案：选择没有唯一答案。
fn rival<'a>(
    profiles: &'a [Profile],
    detected: &Detected,
    best: usize,
    top: usize,
) -> Option<&'a Profile> {
    profiles
        .iter()
        .enumerate()
        .find(|(index, profile)| {
            *index != best && profile.specificity() == top && profile.matches(detected)
        })
        .map(|(_, profile)| profile)
}
mod schema;
pub use schema::WitnessLookup;
#[cfg(test)]
#[path = "../tests/prof.rs"]
mod tests;
