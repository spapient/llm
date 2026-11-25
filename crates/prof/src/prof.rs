//! 档案层：设备指纹、参数化能力供给、证据等级与字段校验（cutile README §5.3、§11.2）。
/// 同层校验与测试共用的词汇表入口。
pub(crate) use crate::atom::{
    Attr, AttrKey, DType, Grade, Need, Prim, Provide, Set, Space, SpaceSet, Span, Witness,
};
use crate::decl::AttrView;
use core::fmt;
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::sync::LazyLock;

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
    /// 解析规则文本：`contains:` → 择一 → 集合 → 区间 → 精确逐级降级。
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

/// 花括号集合语法 `base{a,b}`：括号必须成对且有序，否则退回区间 / 精确。
fn brace_set(text: &str) -> Option<(String, Vec<String>)> {
    let (open, close) = (text.find('{')?, text.rfind('}')?);
    if open >= close {
        return None;
    }
    let base = text[..open].trim().to_string();
    Some((base, split(&text[open + 1..close], ',')))
}

fn split(text: &str, delimiter: char) -> Vec<String> {
    text.split(delimiter)
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mem {
    /// 片上快速存储字节数。
    pub fast_bytes: u64,
    /// 片上快速存储的 bank 数（swizzle 合法性用）。
    pub fast_bank: u32,
    /// 寄存器字节数。
    pub reg_bytes: u64,
}
/// SIMT 执行模型参数（CPU 也有一份，`group = 1`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
/// 字段路径按需渲染；成功路径不为诊断拼字符串。
#[derive(Clone, Copy)]
pub(crate) enum Field<'a> {
    /// 按供给下标定位。
    Provide { idx: usize, key: &'a str },
    /// 按原语定位，供跨供给校验使用。
    Prim { prim: Prim, key: &'a str },
}

impl fmt::Display for Field<'_> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Provide { idx, key } => write!(out, "provides[{idx}].{key}"),
            Self::Prim { prim, key } => write!(out, "provides[{prim}].{key}"),
        }
    }
}

pub(crate) const fn at(idx: usize, key: &str) -> Field<'_> {
    Field::Provide { idx, key }
}

pub(crate) const fn prim_at(prim: Prim, key: &str) -> Field<'_> {
    Field::Prim { prim, key }
}

impl Profile {
    /// 从编译期强类型视图构造档案；保留字段校验与既有入口名。
    pub fn parse(view: &crate::prof::ProfileView) -> Result<Self, ProfileError> {
        view.into_owned()
    }

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

/// 全部内置档案：静态视图只在首次使用时 owned 化并缓存，成功与失败都可重读。
///
/// # Errors
///
/// 任一生成档案违反字段约束时返回对应错误，不跳过设备，也不在生产路径 panic。
pub fn builtin_profiles() -> Result<&'static [Profile], ProfileError> {
    static PROFILES: LazyLock<Result<Vec<Profile>, ProfileError>> =
        LazyLock::new(|| crate::PROFILE_VIEWS.iter().map(Profile::parse).collect());
    PROFILES.as_ref().map(Vec::as_slice).map_err(Clone::clone)
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
pub use crate::witness::WitnessLookup;
#[cfg(test)]
#[path = "../tests/prof.rs"]
mod tests;

/// 一条能力供给的静态视图。
#[derive(Debug, Clone, Copy)]
pub struct ProvideView {
    /// 原语。
    pub prim: Prim,
    /// 实现名。
    pub imp: &'static str,
    /// 主维度区间原文；缺省表示不限。
    pub extent: Option<&'static str>,
    /// `n` 区间原文。
    pub n: Option<&'static str>,
    /// `k` 区间原文。
    pub k: Option<&'static str>,
    /// 数据类型集合。
    pub dt: &'static [DType],
    /// 存储空间集合。
    pub space: &'static [Space],
    /// 证据等级。
    pub grade: Grade,
    /// 见证：`(文件, 入口, 期望输出)`。
    pub witness: Option<(&'static str, &'static str, &'static [f32])>,
    /// 枚举型属性。
    pub attrs: &'static [AttrView],
}

impl ProvideView {
    /// 转成现有供给类型；区间仍由唯一 [`Span::parse`] 校验。
    pub fn into_owned(self) -> Result<Provide, ProfileError> {
        Ok(Provide {
            prim: self.prim,
            extent: view_span(self.extent)?,
            n: view_span(self.n)?,
            k: view_span(self.k)?,
            dt: Set::of(self.dt.iter().copied()),
            space: Set::of(self.space.iter().copied()),
            attrs: self.attrs.iter().map(|attr| attr.into_owned()).collect(),
            impl_name: self.imp.to_string(),
            grade: self.grade,
            witness: self.witness.map(|(file, func, expect)| Witness {
                file: file.to_string(),
                func: func.to_string(),
                expect: expect.to_vec(),
            }),
        })
    }
}

/// 可选区间仍走唯一区间解析器，拒绝手工构造的非法视图。
fn view_span(text: Option<&str>) -> Result<Span, ProfileError> {
    text.map(|text| Span::parse(text).map_err(|error| bad("provides", error.message())))
        .transpose()
        .map(|span| span.unwrap_or_else(Span::any))
}

/// 一份设备档案的强类型只读视图，不包含 `String`、`Vec` 或通用 `Table`。
#[derive(Debug, Clone, Copy)]
pub struct ProfileView {
    /// 厂商键。
    pub vendor: &'static str,
    /// 具体型号键。
    pub id: &'static str,
    /// 输出语言。
    pub emit: Emit,
    /// 编译工具名。
    pub tools: &'static str,
    /// 登记的工具版本。
    pub tool_ver: &'static str,
    /// 上次核对日期。
    pub checked: &'static str,
    /// 指纹 `(键, 规则原文)`。
    pub fingerprint: &'static [(&'static str, &'static str)],
    /// 语义存储到物理名的映射。
    pub spaces: &'static [(Space, &'static str)],
    /// 容量与 bank。
    pub mem: Mem,
    /// SIMT 参数。
    pub simt: Simt,
    /// 执行单元 `(名字, 可访问空间)`。
    pub units: &'static [(&'static str, &'static [Space])],
    /// 能力供给。
    pub provides: &'static [ProvideView],
    /// f32 峰值 `(数值, 推导式)`。
    pub flops_f32: Option<(f64, &'static str)>,
    /// 带宽 `(数值, 推导式)`。
    pub bw: Option<(f64, &'static str)>,
    /// 编译参数。
    pub flags: &'static [&'static str],
    /// 源文件前置声明。
    pub preamble: &'static str,
    /// 工具链头文件垫片 `(文件名, 内容)`。
    pub headers: &'static [(&'static str, &'static str)],
}

impl ProfileView {
    /// 转成现有档案类型；保留缺段、三级存储与峰值推导式校验。
    pub fn into_owned(self) -> Result<Profile, ProfileError> {
        self.check()?;
        Ok(Profile {
            meta: Meta {
                vendor: self.vendor.to_string(),
                id: self.id.to_string(),
                emit: self.emit,
                tool: self.tools.to_string(),
                tool_ver: self.tool_ver.to_string(),
                checked: self.checked.to_string(),
            },
            fingerprint: self
                .fingerprint
                .iter()
                .map(|(key, rule)| (key.to_string(), MatchRule::parse(rule)))
                .collect(),
            spaces: self
                .spaces
                .iter()
                .map(|(space, name)| (*space, name.to_string()))
                .collect(),
            mem: self.mem,
            simt: self.simt,
            units: self
                .units
                .iter()
                .map(|(name, space)| Unit {
                    name: name.to_string(),
                    space: Set::of(space.iter().copied()),
                })
                .collect(),
            provides: self
                .provides
                .iter()
                .map(|provide| provide.into_owned())
                .collect::<Result<_, _>>()?,
            peak: Peak {
                flops_f32: self
                    .flops_f32
                    .map(|constant| constant_owned("flops_f32", constant))
                    .transpose()?,
                bw: self
                    .bw
                    .map(|constant| constant_owned("bw", constant))
                    .transpose()?,
            },
            flags: self.flags.iter().map(ToString::to_string).collect(),
            preamble: self.preamble.to_string(),
            headers: self
                .headers
                .iter()
                .map(|(name, text)| (name.to_string(), text.to_string()))
                .collect(),
        })
    }

    /// 与旧表解析入口一致的最小结构约束。
    fn check(&self) -> Result<(), ProfileError> {
        if self.fingerprint.is_empty() {
            return Err(missing("fingerprint.match"));
        }
        if self.provides.is_empty() {
            return Err(missing("provides"));
        }
        for space in Space::ALL.iter().copied() {
            if !self.spaces.iter().any(|(have, _)| *have == space) {
                return Err(missing(format!(
                    "space.{}（三级存储的语义名必须齐全）",
                    space.as_str()
                )));
            }
        }
        Ok(())
    }
}

/// 峰值常数：数值必须与推导式在既有相对误差预算内一致。
fn constant_owned(field: &str, (value, src): (f64, &str)) -> Result<Constant, ProfileError> {
    let derived = crate::einsum::formula::eval_expression(src).map_err(|error| {
        bad(
            format!("peak.{field}.src"),
            format!("推导式 `{src}` 求值失败：{}", error.message()),
        )
    })?;
    if (derived - value).abs() > value.abs().max(1.0) * 1e-6 {
        return Err(bad(
            format!("peak.{field}.src"),
            format!("推导式 `{src}` = {derived:e}，与 value = {value:e} 不一致"),
        ));
    }
    Ok(Constant {
        value,
        src: src.to_string(),
    })
}
