//! 档案层：设备指纹、参数化能力供给、证据等级、字段校验与见证查找（cutile README §5.3、§11.2）。
//!
//! 一份文件装两节，因为它们是同一条链的两端：`crate::schema` 把构建期展开的表读成 [`Profile`]，
//! 本节给它判定（字段形状、见证入口、`exec` 单元、重复项）并回答"见证从哪读"（[`WitnessLookup`]）。
/// 同层文件（`schema`）用 `use crate::prof::*` 取词汇表，所以是 `pub(crate) use` 而不是 `use`——
/// 这里就是本 crate 唯一的词汇表入口。
pub(crate) use crate::atom::{
    Attr, AttrKey, Closed, DType, DTypeSet, Grade, Need, Prim, Provide, Space, SpaceSet, Span,
    Witness,
};
use core::fmt;
pub(crate) use diagnostic::data::{Table, Value};
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::sync::OnceLock;

crate::closed_enum! {
    /// 输出语言闭集（cutile README §5.4 的 `emit` 列）。设备与输出语言是两个维度：
    /// 同一条 `cuda` emit 同时服务 NVIDIA 与 MUSA。附加列 `ext` 是生成的源文件扩展名
    /// （只回答后缀，不回答这种语言能否编出产物）。
    Emit {
        ext:
        // CUDA C++。
        Cuda => ("cuda", "cu"),
        // PTX 汇编。
        Ptx => ("ptx", "ptx"),
        // HIP。
        Hip => ("hip", "hip"),
        // OpenCL C。
        Opencl => ("opencl", "cl"),
        // SPIR-V。
        Spirv => ("spirv", "spv"),
        // Metal Shading Language。
        Metal => ("metal", "metal"),
        // 昇腾 AscendC。
        Ascendc => ("ascendc", "cpp"),
        // 可移植 C。
        C => ("c", "c"),
        // WGSL。
        Wgsl => ("wgsl", "wgsl"),
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
pub(crate) fn set_matches(base: &str, members: &[String], actual: &str) -> bool {
    let hit = |member: &String| actual.contains(member.as_str());
    actual.starts_with(base) && members.iter().all(hit)
}
/// 版本区间：两端都必须能解析成版本号，实际值落在闭区间内。
pub(crate) fn within(lo: &str, hi: &str, actual: &str) -> bool {
    let Some(value) = parse_version(actual) else {
        return false;
    };
    parse_version(lo).is_some_and(|low| low <= value)
        && parse_version(hi).is_some_and(|high| value <= high)
}
impl fmt::Display for MatchRule {
    /// 五个变体各走一条只写一处的排版分支：`match` 之外没有一个 `?`，
    /// 于是这条 `Display` 的复杂度就是臂数本身。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(text) => f.write_str(text),
            Self::Range { lo, hi } => write!(f, "{lo}..{hi}"),
            Self::Contains(tokens) => marked(f, "contains: ", "|", tokens),
            Self::Set { base, members } => braced(f, base, members),
            Self::OneOf(branches) => joined(f, "|", branches),
        }
    }
}

/// `前缀 + 用分隔符串起来的条目`（`contains: a|b`）。
fn marked<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    head: &str,
    sep: &str,
    items: &[T],
) -> fmt::Result {
    f.write_str(head)?;
    joined(f, sep, items)
}

/// `基名{条目}`（`x86{avx2,sve}`）。
fn braced(f: &mut fmt::Formatter<'_>, base: &str, members: &[String]) -> fmt::Result {
    write!(f, "{base}{{")?;
    joined(f, ",", members)?;
    f.write_str("}")
}

/// 用分隔符把一串条目写进格式化器：不先收集 `Vec<String>`，也不为每个成员造 `String`。
///
/// 首个条目之后交给 `try_for_each`：分隔符与条目在同一个写点，函数里只剩一个 `?`。
fn joined<T: fmt::Display>(f: &mut fmt::Formatter<'_>, sep: &str, items: &[T]) -> fmt::Result {
    let mut rest = items.iter();
    let Some(first) = rest.next() else {
        return Ok(());
    };
    write!(f, "{first}")?;
    rest.try_for_each(|item| write!(f, "{sep}{item}"))
}
pub(crate) fn parse_version(text: &str) -> Option<Vec<u32>> {
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
        for (at, (key, value)) in self.attrs.iter().enumerate() {
            if at > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{key}={value}")?;
        }
        Ok(())
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
pub(crate) fn missing(field: impl fmt::Display) -> ProfileError {
    ProfileError::Missing {
        field: field.to_string(),
    }
}
/// 闭集外取值；`allowed` 是闭集成员的名字表。
pub(crate) fn unknown(field: impl fmt::Display, value: &str, allowed: &[&str]) -> ProfileError {
    ProfileError::Unknown {
        field: field.to_string(),
        value: value.to_string(),
        allowed: allowed.join(" | "),
    }
}
/// 字段非法：`detail` 说明为什么非法，不重复字段名。
pub(crate) fn bad(field: impl fmt::Display, detail: impl Into<String>) -> ProfileError {
    ProfileError::BadValue {
        field: field.to_string(),
        detail: detail.into(),
    }
}
/// 证据不成立：`prim` 是被考察的那条供给。
pub(crate) fn witness_error(provide: &Provide, detail: impl Into<String>) -> ProfileError {
    ProfileError::Witness {
        prim: provide.prim,
        detail: detail.into(),
    }
}
/// 闭集成员的名字表；诊断里的「闭集：a | b」按声明顺序给出。
pub(crate) fn closed_names<E: Closed>(all: &'static [E]) -> Vec<&'static str> {
    all.iter().map(|item| item.as_str()).collect()
}
/// 解析闭集里的名字；闭集外一律 `Unknown`，并把闭集内容一起报出来。
pub(crate) fn closed<E: Closed>(name: &str, field: impl fmt::Display) -> Result<E, ProfileError> {
    E::parse(name).ok_or_else(|| unknown(field, name, &closed_names(E::ALL)))
}
/// 字段路径的惰性写法。一次 `Profile::parse` 会走几十条供给、每条十几个字段，
/// 成功路径上不该为诊断拼任何字符串——`format!` 只在真出错时才发生。
#[derive(Clone, Copy)]
pub(crate) enum Field<'a> {
    /// `provides[<下标>].<key>`；`witness` 为真时多一层 `witness.`。
    Provide {
        idx: usize,
        key: &'a str,
        witness: bool,
    },
    /// `provides[<原语名>].<key>`：校验按原语而不是下标报位置。
    Prim { prim: Prim, key: &'a str },
}
impl fmt::Display for Field<'_> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Provide {
                idx,
                key,
                witness: true,
            } => write!(out, "provides[{idx}].witness.{key}"),
            Self::Provide {
                idx,
                key,
                witness: false,
            } => write!(out, "provides[{idx}].{key}"),
            Self::Prim { prim, key } => write!(out, "provides[{prim}].{key}"),
        }
    }
}
pub(crate) const fn at(idx: usize, key: &str) -> Field<'_> {
    Field::Provide {
        idx,
        key,
        witness: false,
    }
}
pub(crate) const fn witness_at(idx: usize, key: &str) -> Field<'_> {
    Field::Provide {
        idx,
        key,
        witness: true,
    }
}
pub(crate) const fn prim_at(prim: Prim, key: &str) -> Field<'_> {
    Field::Prim { prim, key }
}
pub(crate) fn text<'a>(
    table: &'a Table,
    key: &str,
    field: impl fmt::Display,
) -> Result<&'a str, ProfileError> {
    table.string(key).ok_or_else(|| missing(field))
}
pub(crate) fn or_dash(table: &Table, key: &str) -> String {
    table.string(key).unwrap_or("-").to_string()
}
pub(crate) fn list<'a>(
    table: &'a Table,
    key: &str,
    field: impl fmt::Display,
) -> Result<Vec<&'a str>, ProfileError> {
    table.string_array(key).ok_or_else(|| missing(field))
}
/// 逐项解析一张表的每个键值对：解析段里所有"按键"段落共用这一种形状。
pub(crate) fn entries<T>(
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
pub(crate) const EXTENT_DETAIL: &str = "主维度只能是区间字符串（如 `<=4096`）";
/// `n` / `k` 只接受区间字符串。
pub(crate) const AXIS_DETAIL: &str = "只能是区间字符串（如 `16`、`16..64`）";
/// `[units.<name>]` 的取值必须是表。
pub(crate) const UNIT_DETAIL: &str = "只接受 `[units.<name>]` 形式的表";
/// `witness` 必须是内联表。
pub(crate) const WITNESS_DETAIL: &str =
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

/// 一趟扫描的结果：最高具体度、它**第一个**出现在哪、**最后一个**出现在哪。
///
/// 「选谁」与「跟谁并列」是两个位置：选后者是 `max_by_key` 的语义（同分取靠后的那份），
/// 而报错时点名的是并列里的第一个。分开记这两个下标，一次遍历就够——不必先把命中项收成
/// `Vec` 再扫两遍，也不为一次选择分配内存。
#[derive(Default)]
struct Best {
    /// 目前见过的最高具体度。
    top: usize,
    /// 达到 `top` 的第一份档案下标。
    first: Option<usize>,
    /// 达到 `top` 的最后一份档案下标（同分时选它）。
    last: usize,
}

impl Best {
    /// 登记一次命中：更高则重开一段并列，同分则推进「最后一份」，更低一概不动。
    fn offer(&mut self, at: usize, specificity: usize) {
        match self.first {
            Some(_) if specificity < self.top => {}
            Some(_) if specificity == self.top => self.last = at,
            _ => {
                self.top = specificity;
                self.first = Some(at);
                self.last = at;
            }
        }
    }
}

/// 按探测指纹从一组档案里挑出唯一命中、且规则最具体的一份。
pub fn select<'a>(profiles: &'a [Profile], detected: &Detected) -> diagnostic::Result<&'a Profile> {
    let mut best = Best::default();
    for (at, profile) in profiles.iter().enumerate() {
        if profile.matches(detected) {
            best.offer(at, profile.specificity());
        }
    }
    let Some(first) = best.first else {
        return Err(
            Error::new(Code::Unsupported, format!("没有档案匹配该指纹：{detected}"))
                .with("prof::select"),
        );
    };
    if first == best.last {
        return Ok(&profiles[best.last]);
    }
    Err(tie(&profiles[best.last], &profiles[first], detected))
}

/// 两份档案同等具体：消息点名「选中的那份」与「并列里的第一份」，与旧实现逐字一致。
fn tie(chosen: &Profile, rival: &Profile, detected: &Detected) -> Error {
    let detail = format!(
        "`{}` 与 `{}` 的指纹同等具体，无法决定用哪一份（指纹：{detected}）",
        chosen.meta.vendor, rival.meta.vendor
    );
    Error::new(Code::Invalid, detail).with("prof::select")
}

// ── 见证与校验 ────────────────────────────────────────────────
//
// 同一条链的另一端：`crate::schema` 把表读成 [`Profile`]，这里给它判定（字段形状、见证入口、
// `exec` 单元、重复项），并回答"见证从哪读"。判据本体只产出 [`ProfileError`]、不碰文件；
// 读文件的那一步在 [`WitnessFiles`] 里，磁盘与内嵌表两种来源共用同一套判定。

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// 见证的形状：`L2` 必须有见证，声明了就要有入口函数名与期望值（不碰文件）。
pub(crate) fn witness_shape(provide: &Provide) -> Vec<ProfileError> {
    let Some(witness) = provide.witness.as_ref() else {
        return missing_witness(provide).into_iter().collect();
    };
    let make = |(_, detail): (&'static str, String)| witness_error(provide, detail);
    shape_details(witness).into_iter().map(make).collect()
}
/// 见证形状的判据本体：键是 `func` / `expect`，两种供给（原语与写回形式）共用一份。
///
/// 文案按需渲染：通过校验的供给（构建期里的绝大多数）一条消息都不拼。
pub(crate) fn shape_details(witness: &Witness) -> Vec<(&'static str, String)> {
    let mut failed = Vec::new();
    if witness.func.is_empty() {
        failed.push(("func", "见证必须给出入口函数名，否则无法执行".to_string()));
    }
    if witness.expect.is_empty() {
        failed.push(("expect", "见证必须给出期望输出，否则无法对拍".to_string()));
    }
    if witness.expect.len() > Witness::MAX_OUT {
        failed.push(("expect", too_many_expect(witness)));
    }
    failed
}

/// `expect` 超长的说明：只有真超了才会被调用，个数与上限在这里现算。
fn too_many_expect(witness: &Witness) -> String {
    format!(
        "期望输出 {} 个，超过上限 {}",
        witness.expect.len(),
        Witness::MAX_OUT
    )
}
/// 没有见证时唯一说得通的理由是等级还不到 `L2`——等级是执行的结论，不是自述。
pub(crate) fn missing_witness(provide: &Provide) -> Option<ProfileError> {
    if !provide.grade.is_verified() {
        return None;
    }
    let grade = provide.grade;
    let detail = format!("{grade} 要求 `witness = {{ file, func, expect }}`");
    Some(witness_error(provide, detail))
}
/// 签名的必填项：有数据类型的原语不能给空集；`L2` 及以上必须写实现名。
pub(crate) fn signature_fields(provide: &Provide) -> Vec<ProfileError> {
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
pub(crate) fn attr_key_error(provide: &Provide, key: AttrKey) -> ProfileError {
    let keys = provide.prim.attr_keys();
    let mut allowed: Vec<&str> = keys.iter().map(|key| key.as_str()).collect();
    allowed.push("exec");
    unknown(prim_at(provide.prim, key.as_str()), key.as_str(), &allowed)
}
/// 一条供给声明的执行单元列表；没写 `exec` 就没有这条判据。
pub(crate) fn exec_attr(provide: &Provide) -> Option<&Attr> {
    let is_exec = |attr: &&Attr| attr.key() == AttrKey::Exec;
    provide.attrs.iter().find(is_exec)
}
/// 见证源码的定位方式：把 `witness.file`（`<vendor>/witness.<ext>`）解析成一个可读路径。
pub trait WitnessLookup {
    /// 见证文件的可读路径；`None` 表示这个 vendor 没有任何落盘位置。
    fn path(&self, file: &str) -> Option<PathBuf>;
}
/// 单一根下的定位：`file` 直接拼在根上。
pub(crate) struct Root<'a>(pub(crate) &'a Path);
impl WitnessLookup for Root<'_> {
    fn path(&self, file: &str) -> Option<PathBuf> {
        Some(self.0.join(file))
    }
}
/// 见证文件的一次性读取缓存：同一份源码只读一次，同一个缺失文件只报一次。
pub(crate) struct WitnessFiles<'a> {
    /// 定位方式；`None` 表示本轮不做文件检查（单元测试）。
    witnesses: Option<&'a dyn WitnessLookup>,
    /// 文件 → 源码；`None` 表示不存在或不可读。
    sources: BTreeMap<String, Option<String>>,
    /// 已经报过的缺失路径。
    reported: BTreeSet<String>,
}
impl<'a> WitnessFiles<'a> {
    /// 空缓存。
    pub(crate) fn new(witnesses: Option<&'a dyn WitnessLookup>) -> Self {
        Self {
            witnesses,
            sources: BTreeMap::new(),
            reported: BTreeSet::new(),
        }
    }
    /// 一条供给的文件级判据：文件要在，且真的声明了那个入口函数；不判断跑不跑得通。
    pub(crate) fn probe(&mut self, provide: &Provide) -> Vec<ProfileError> {
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
    /// 读一份见证源码：每个文件最多读一次盘（命中缓存的那次连键都不必再分配）。
    fn source(&mut self, witnesses: &dyn WitnessLookup, file: &str) -> Option<&str> {
        if !self.sources.contains_key(file) {
            let text = witnesses
                .path(file)
                .and_then(|path| std::fs::read_to_string(path).ok());
            self.sources.insert(file.to_string(), text);
        }
        self.sources.get(file)?.as_deref()
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
/// 见证源码里是否声明了入口函数：找「`func` 紧跟 `(`」的位置，不为每次探测拼一个 `func(`。
pub(crate) fn declares(text: &str, witness: &Witness) -> bool {
    let func = witness.func.as_str();
    text.match_indices(func)
        .any(|(at, _)| text[at + func.len()..].starts_with('('))
}
pub(crate) fn absent_function(witness: &Witness) -> String {
    format!(
        "`{}` 里没有函数 `{}`：声明的见证无法执行",
        witness.file, witness.func
    )
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

#[cfg(test)]
#[path = "../tests/prof.rs"]
mod tests;
