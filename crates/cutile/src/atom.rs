//! 原语闭集与参数化能力签名。
//!
//! 能力不是"有没有 `mma`"，而是"哪种 `mma`、支持多宽的形状、什么精度、在哪个存储层"
//! （DESIGN §5.2）。因此 [`Provide::covers`] 只做一次区间包含判定，协商不需要任何启发式。

use core::fmt;

/// 原语闭集：模板唯一能调用的 16 个指令级操作（DESIGN §5.1 的 P1–P16）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Prim {
    /// `D = A × B + C`，tile 粒度、累积。
    Mma,
    /// 整数点积（dp4a / dpas / dot4）。
    Dot,
    /// 全局 → 片上，同步搬运。
    Load,
    /// 异步搬运 + 完成令牌。
    LoadAsync,
    /// 异步组提交 / 等待。
    CommitWait,
    /// 片上 → 全局。
    Store,
    /// 块内屏障 / arrive-wait。
    Bar,
    /// 子组内 lane 交换。
    Shuffle,
    /// 子组内广播。
    Broadcast,
    /// 沿轴归约。
    Reduce,
    /// 沿轴包含前缀。
    Scan,
    /// 快速 2 的幂。
    Exp2,
    /// 倒数 / rsqrt。
    Rcp,
    /// 数据类型转换（含饱和转换）。
    Cvt,
    /// 缩放 / 零点应用、低位宽打包。
    Quant,
    /// 全局原子。
    Atomic,
}

impl Prim {
    /// 闭集全量，顺序与 DESIGN §5.1 一致。
    pub const ALL: [Self; 16] = [
        Self::Mma,
        Self::Dot,
        Self::Load,
        Self::LoadAsync,
        Self::CommitWait,
        Self::Store,
        Self::Bar,
        Self::Shuffle,
        Self::Broadcast,
        Self::Reduce,
        Self::Scan,
        Self::Exp2,
        Self::Rcp,
        Self::Cvt,
        Self::Quant,
        Self::Atomic,
    ];

    /// 声明层、档案层与片段层共用的名字。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mma => "mma",
            Self::Dot => "dot",
            Self::Load => "load",
            Self::LoadAsync => "load_async",
            Self::CommitWait => "commit_wait",
            Self::Store => "store",
            Self::Bar => "bar",
            Self::Shuffle => "shuffle",
            Self::Broadcast => "broadcast",
            Self::Reduce => "reduce",
            Self::Scan => "scan",
            Self::Exp2 => "exp2",
            Self::Rcp => "rcp",
            Self::Cvt => "cvt",
            Self::Quant => "quant",
            Self::Atomic => "atomic",
        }
    }

    /// 从闭集名字解析；闭集外返回 `None`，由调用方补全"闭集是什么"的报错文案。
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|prim| prim.as_str() == name)
    }

    /// 主维度在诊断里的名字：`mma` 是 `m`、搬运是 `bytes`、`dot` 是 `lanes`。
    pub const fn extent_label(self) -> &'static str {
        match self {
            Self::Mma => "m",
            Self::Dot => "lanes",
            Self::Load | Self::LoadAsync | Self::Store => "bytes",
            Self::Reduce | Self::Scan => "axis",
            _ => "extent",
        }
    }

    /// 是否带数据类型语义（`commit_wait` / `bar` 这类同步原语没有）。
    pub const fn has_dtype(self) -> bool {
        !matches!(self, Self::CommitWait | Self::Bar)
    }

    /// 是否属于搬运三件套（`load` / `load_async` / `store`）。
    pub const fn is_transfer(self) -> bool {
        matches!(self, Self::Load | Self::LoadAsync | Self::Store)
    }
}

impl fmt::Display for Prim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 数据类型闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// IEEE 半精度。
    F16,
    /// bfloat16。
    Bf16,
    /// 单精度。
    F32,
    /// FP8 E4M3。
    F8e4m3,
    /// FP4。
    F4,
    /// 有符号 8 位整数。
    I8,
    /// 有符号 4 位整数。
    I4,
    /// 无符号 8 位整数。
    U8,
    /// 有符号 32 位整数。
    I32,
}

impl DType {
    /// 闭集全量。
    pub const ALL: [Self; 9] = [
        Self::F16,
        Self::Bf16,
        Self::F32,
        Self::F8e4m3,
        Self::F4,
        Self::I8,
        Self::I4,
        Self::U8,
        Self::I32,
    ];

    /// DSL 与档案里使用的名字。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F8e4m3 => "f8e4m3",
            Self::F4 => "f4",
            Self::I8 => "i8",
            Self::I4 => "i4",
            Self::U8 => "u8",
            Self::I32 => "i32",
        }
    }

    /// 单元素位宽。
    pub const fn size_bits(self) -> u32 {
        match self {
            Self::F4 | Self::I4 => 4,
            Self::U8 | Self::I8 | Self::F8e4m3 => 8,
            Self::F16 | Self::Bf16 => 16,
            Self::F32 | Self::I32 => 32,
        }
    }

    /// 在闭集里的索引，供 [`DTypeSet`] 做位集。
    const fn index(self) -> usize {
        self as usize
    }

    /// 从闭集名字解析。
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|dt| dt.as_str() == name)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 数据类型集合（位集），协商时做子集判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct DTypeSet(u32);

impl DTypeSet {
    /// 空集。
    pub const fn empty() -> Self {
        Self(0)
    }

    /// 加入一个类型。
    pub fn insert(&mut self, dt: DType) {
        self.0 |= 1 << dt.index();
    }

    /// 由类型列表构造。
    pub fn of(dts: impl IntoIterator<Item = DType>) -> Self {
        let mut set = Self::empty();
        for dt in dts {
            set.insert(dt);
        }
        set
    }

    /// 是否包含某类型。
    pub const fn contains(self, dt: DType) -> bool {
        self.0 & (1 << dt.index()) != 0
    }

    /// 是否为空集。
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// 元素个数。
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// `self` 是否是 `other` 的子集。
    pub const fn is_subset(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// 在 `self` 中而不在 `other` 中的元素。
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// 逐名字列出（升序，与闭集顺序一致）。
    pub fn names(self) -> Vec<&'static str> {
        DType::ALL
            .into_iter()
            .filter(|dt| self.contains(*dt))
            .map(DType::as_str)
            .collect()
    }
}

impl fmt::Display for DTypeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("-");
        }
        f.write_str(&self.names().join(","))
    }
}

/// 三级存储的语义名（DESIGN §6.3）：物理名由档案映射。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Space {
    /// 全局内存。
    Gm,
    /// 片上快速存储（NVIDIA `shared`、Intel `slm`、昇腾 `L1`…）。
    Fast,
    /// 寄存器。
    Reg,
}

impl Space {
    /// 闭集全量。
    pub const ALL: [Self; 3] = [Self::Gm, Self::Fast, Self::Reg];

    /// 模板与档案里使用的名字（大写，与 DESIGN 一致）。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gm => "GM",
            Self::Fast => "FAST",
            Self::Reg => "REG",
        }
    }

    /// 在闭集里的索引。
    const fn index(self) -> usize {
        self as usize
    }

    /// 从名字解析。
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|space| space.as_str() == name)
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 存储空间集合（位集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct SpaceSet(u32);

impl SpaceSet {
    /// 空集。
    pub const fn empty() -> Self {
        Self(0)
    }

    /// 加入一个存储空间。
    pub fn insert(&mut self, space: Space) {
        self.0 |= 1 << space.index();
    }

    /// 由列表构造。
    pub fn of(spaces: impl IntoIterator<Item = Space>) -> Self {
        let mut set = Self::empty();
        for space in spaces {
            set.insert(space);
        }
        set
    }

    /// 是否包含某空间。
    pub const fn contains(self, space: Space) -> bool {
        self.0 & (1 << space.index()) != 0
    }

    /// 是否为空集。
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `self` 是否是 `other` 的子集。
    pub const fn is_subset(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// 在 `self` 中而不在 `other` 中的元素。
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// 逐名字列出。
    pub fn names(self) -> Vec<&'static str> {
        Space::ALL
            .into_iter()
            .filter(|space| self.contains(*space))
            .map(Space::as_str)
            .collect()
    }
}

impl fmt::Display for SpaceSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("-");
        }
        f.write_str(&self.names().join(","))
    }
}

/// 一个闭区间（含两端）。`exact(16)` 表示"只能 16"，`at_most(512)` 表示"≤ 512"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    lo: u32,
    hi: u32,
}

impl Span {
    /// 不限。
    pub const fn any() -> Self {
        Self {
            lo: 0,
            hi: u32::MAX,
        }
    }

    /// 精确值。
    pub const fn exact(value: u32) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    /// 上界（下界 0）。
    pub const fn at_most(value: u32) -> Self {
        Self { lo: 0, hi: value }
    }

    /// 下界（上界不限）。
    pub const fn at_least(value: u32) -> Self {
        Self {
            lo: value,
            hi: u32::MAX,
        }
    }

    /// 区间；`lo > hi` 视为非法。
    pub fn range(lo: u32, hi: u32) -> Result<Self, String> {
        if lo > hi {
            return Err(format!("区间下界 {lo} 大于上界 {hi}"));
        }
        Ok(Self { lo, hi })
    }

    /// 下界。
    pub const fn lo(self) -> u32 {
        self.lo
    }

    /// 上界。
    pub const fn hi(self) -> u32 {
        self.hi
    }

    /// 是否不限。
    pub const fn is_any(self) -> bool {
        self.lo == 0 && self.hi == u32::MAX
    }

    /// `self` 是否覆盖 `need`（区间包含）。
    pub const fn covers(self, need: Self) -> bool {
        self.lo <= need.lo && need.hi <= self.hi
    }

    /// 解析三种写法：`16`、`16..64`、`<=512`、`16..`。
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if let Some(rest) = text.strip_prefix("<=") {
            let hi = parse_u32(rest)?;
            return Ok(Self::at_most(hi));
        }
        if let Some((lo, hi)) = text.split_once("..") {
            let lo = parse_u32(lo)?;
            let hi = if hi.trim().is_empty() {
                u32::MAX
            } else {
                parse_u32(hi)?
            };
            return Self::range(lo, hi);
        }
        Ok(Self::exact(parse_u32(text)?))
    }
}

fn parse_u32(text: &str) -> Result<u32, String> {
    text.trim()
        .parse::<u32>()
        .map_err(|_| format!("`{}` 不是合法的非负整数", text.trim()))
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_any() {
            return f.write_str("*");
        }
        if self.lo == self.hi {
            return write!(f, "{}", self.lo);
        }
        if self.lo == 0 {
            return write!(f, "<={}", self.hi);
        }
        if self.hi == u32::MAX {
            return write!(f, "{}..", self.lo);
        }
        write!(f, "{}..{}", self.lo, self.hi)
    }
}

/// 能力签名里的枚举型属性键（闭集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttrKey {
    /// 归约 / 原子的运算种类。
    Op,
    /// 归约 / 扫描的轴名。
    Axis,
    /// 变体类操作（`rsqrt` / `rcp`）。
    Kind,
    /// 是否饱和。
    Sat,
    /// 精度档。
    Accuracy,
    /// 量化分组。
    Group,
    /// 点积 lanes。
    Lanes,
    /// 搬运源。
    Src,
    /// 搬运目标。
    Dst,
    /// 预取级数。
    Stages,
}

impl AttrKey {
    /// 闭集全量。
    pub const ALL: [Self; 10] = [
        Self::Op,
        Self::Axis,
        Self::Kind,
        Self::Sat,
        Self::Accuracy,
        Self::Group,
        Self::Lanes,
        Self::Src,
        Self::Dst,
        Self::Stages,
    ];

    /// 声明层里使用的键名。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Op => "op",
            Self::Axis => "axis",
            Self::Kind => "kind",
            Self::Sat => "sat",
            Self::Accuracy => "accuracy",
            Self::Group => "group",
            Self::Lanes => "lanes",
            Self::Src => "src",
            Self::Dst => "dst",
            Self::Stages => "stages",
        }
    }

    /// 从键名解析；闭集外返回 `None`。
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|key| key.as_str() == name)
    }
}

impl fmt::Display for AttrKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 一个属性键上的取值集合；协商要求"需求方的取值全部被供给方覆盖"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    key: AttrKey,
    values: Vec<String>,
}

impl Attr {
    /// 构造。
    pub fn new<S: Into<String>>(key: AttrKey, values: impl IntoIterator<Item = S>) -> Self {
        Self {
            key,
            values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// 键。
    pub const fn key(&self) -> AttrKey {
        self.key
    }

    /// 取值。
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// 是否含某取值。
    pub fn contains(&self, value: &str) -> bool {
        self.values.iter().any(|item| item == value)
    }
}

/// 需求：模板 / 算子声明对某原语的要求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Need {
    /// 原语。
    pub prim: Prim,
    /// 主维度区间：`mma` 用 `m`、搬运用 `bytes`、`dot` 用 `lanes`。
    pub extent: Span,
    /// `mma` 的 `n`。
    pub n: Span,
    /// `mma` 的 `k`。
    pub k: Span,
    /// 数据类型集合。
    pub dt: DTypeSet,
    /// 存储空间集合。
    pub space: SpaceSet,
    /// 枚举型属性。
    pub attrs: Vec<Attr>,
}

impl Need {
    /// 只约束原语本身，其余维度不限。
    pub fn of(prim: Prim) -> Self {
        Self {
            prim,
            extent: Span::any(),
            n: Span::any(),
            k: Span::any(),
            dt: DTypeSet::empty(),
            space: SpaceSet::empty(),
            attrs: Vec::new(),
        }
    }

    /// 设置主维度。
    #[must_use]
    pub fn with_extent(mut self, extent: Span) -> Self {
        self.extent = extent;
        self
    }

    /// 设置 `n`。
    #[must_use]
    pub fn with_n(mut self, n: Span) -> Self {
        self.n = n;
        self
    }

    /// 设置 `k`。
    #[must_use]
    pub fn with_k(mut self, k: Span) -> Self {
        self.k = k;
        self
    }

    /// 设置数据类型集合。
    #[must_use]
    pub fn with_dt(mut self, dt: DTypeSet) -> Self {
        self.dt = dt;
        self
    }

    /// 设置存储空间集合。
    #[must_use]
    pub fn with_space(mut self, space: SpaceSet) -> Self {
        self.space = space;
        self
    }

    /// 追加一个属性。
    #[must_use]
    pub fn with_attr(mut self, attr: Attr) -> Self {
        self.attrs.push(attr);
        self
    }

    /// 按签名文本渲染，供诊断使用：`mma{m:16..64, n:8..128, dt:[f16,bf16], space:[FAST]}`。
    pub fn signature(&self) -> String {
        let mut parts = Vec::new();
        if !self.extent.is_any() {
            parts.push(format!("{}:{}", self.prim.extent_label(), self.extent));
        }
        for (label, span) in [("n", self.n), ("k", self.k)] {
            if !span.is_any() {
                parts.push(format!("{label}:{span}"));
            }
        }
        if !self.dt.is_empty() {
            parts.push(format!("dt:[{}]", self.dt));
        }
        if !self.space.is_empty() {
            parts.push(format!("space:[{}]", self.space));
        }
        for attr in &self.attrs {
            parts.push(format!("{}:{}", attr.key, attr.values.join("|")));
        }
        format!("{}{{{}}}", self.prim, parts.join(", "))
    }
}

/// 证据等级（DESIGN §11.2）。协商只接受 `L2` 及以上。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Grade {
    /// 档案里声明了。
    L0,
    /// 片段能编译。
    L1,
    /// 与参考对拍通过。
    L2,
    /// 真机性能达到 roofline 的 70%。
    L3,
}

impl Grade {
    /// 闭集全量，从低到高。
    pub const ALL: [Self; 4] = [Self::L0, Self::L1, Self::L2, Self::L3];

    /// 渲染名。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L0 => "L0",
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
        }
    }

    /// 解析 `L0`–`L3`。
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|grade| grade.as_str() == text)
    }

    /// 是否达到"可用于协商"的门槛（`L2`）。
    pub const fn is_verified(self) -> bool {
        matches!(self, Self::L2 | Self::L3)
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 供给：某设备档案对某原语的实际能力。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provide {
    /// 原语。
    pub prim: Prim,
    /// 实际支持的主维度区间。
    pub extent: Span,
    /// 实际支持的 `n`。
    pub n: Span,
    /// 实际支持的 `k`。
    pub k: Span,
    /// 实际支持的数据类型。
    pub dt: DTypeSet,
    /// 实际所在存储空间。
    pub space: SpaceSet,
    /// 枚举型属性。
    pub attrs: Vec<Attr>,
    /// 实现名，如 `wgmma` / `dpas` / `neon`。
    pub impl_name: String,
    /// 证据等级。
    pub grade: Grade,
    /// 见证文件相对路径（`L2` 及以上必填）。
    pub witness: Option<String>,
}

impl Provide {
    /// 构造一个只声明原语本身的供给，便于增量补字段。
    pub fn of(prim: Prim, impl_name: impl Into<String>) -> Self {
        Self {
            prim,
            extent: Span::any(),
            n: Span::any(),
            k: Span::any(),
            dt: DTypeSet::empty(),
            space: SpaceSet::empty(),
            attrs: Vec::new(),
            impl_name: impl_name.into(),
            grade: Grade::L0,
            witness: None,
        }
    }

    /// 设置主维度。
    #[must_use]
    pub fn with_extent(mut self, extent: Span) -> Self {
        self.extent = extent;
        self
    }

    /// 设置 `n`。
    #[must_use]
    pub fn with_n(mut self, n: Span) -> Self {
        self.n = n;
        self
    }

    /// 设置 `k`。
    #[must_use]
    pub fn with_k(mut self, k: Span) -> Self {
        self.k = k;
        self
    }

    /// 设置数据类型集合。
    #[must_use]
    pub fn with_dt(mut self, dt: DTypeSet) -> Self {
        self.dt = dt;
        self
    }

    /// 设置存储空间集合。
    #[must_use]
    pub fn with_space(mut self, space: SpaceSet) -> Self {
        self.space = space;
        self
    }

    /// 追加属性。
    #[must_use]
    pub fn with_attr(mut self, attr: Attr) -> Self {
        self.attrs.push(attr);
        self
    }

    /// 设置证据等级。
    #[must_use]
    pub fn with_grade(mut self, grade: Grade) -> Self {
        self.grade = grade;
        self
    }

    /// 设置见证路径。
    #[must_use]
    pub fn with_witness(mut self, witness: impl Into<String>) -> Self {
        self.witness = Some(witness.into());
        self
    }

    /// 是否覆盖某个需求。
    pub fn covers(&self, need: &Need) -> bool {
        self.diff(need).is_empty()
    }

    /// 逐项比对，返回全部不满足项；空向量等价于覆盖。
    pub fn diff(&self, need: &Need) -> Vec<Mismatch> {
        if self.prim != need.prim {
            return vec![Mismatch::PrimAbsent { prim: need.prim }];
        }

        let mut out = Vec::new();
        if !need.extent.is_any() && !self.extent.covers(need.extent) {
            out.push(Mismatch::ExtentTooSmall {
                label: need.prim.extent_label(),
                need: need.extent,
                have: self.extent,
            });
        }
        for (label, need_span, have_span) in [("n", need.n, self.n), ("k", need.k, self.k)] {
            if need_span.is_any() || have_span.covers(need_span) {
                continue;
            }
            out.push(Mismatch::ExtentTooSmall {
                label,
                need: need_span,
                have: have_span,
            });
        }
        if !need.dt.is_subset(self.dt) {
            out.push(Mismatch::DtMissing(need.dt.difference(self.dt)));
        }
        if !need.space.is_subset(self.space) {
            out.push(Mismatch::SpaceMissing(need.space.difference(self.space)));
        }
        for attr in &need.attrs {
            match self.attrs.iter().find(|have| have.key == attr.key) {
                None => out.push(Mismatch::AttrMissing {
                    key: attr.key,
                    missing: attr.values.clone(),
                }),
                Some(have) => {
                    let missing: Vec<String> = attr
                        .values
                        .iter()
                        .filter(|value| !have.contains(value))
                        .cloned()
                        .collect();
                    if !missing.is_empty() {
                        out.push(Mismatch::AttrMissing {
                            key: attr.key,
                            missing,
                        });
                    }
                }
            }
        }
        out
    }

    /// 单行签名，供 `doctor` 与诊断打印（DESIGN §8.2 的 `have` 行）。
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.extent.is_any() {
            parts.push(format!("{}:{}", self.prim.extent_label(), self.extent));
        }
        for (label, span) in [("n", self.n), ("k", self.k)] {
            if !span.is_any() {
                parts.push(format!("{label}:{span}"));
            }
        }
        if !self.dt.is_empty() {
            parts.push(format!("dt:[{}]", self.dt));
        }
        if !self.space.is_empty() {
            parts.push(format!("space:[{}]", self.space));
        }
        for attr in &self.attrs {
            parts.push(format!("{}:{}", attr.key, attr.values.join("|")));
        }
        parts.push(format!("impl:{}", self.impl_name));
        parts.push(format!("grade:{}", self.grade));
        format!("{}{{{}}}", self.prim, parts.join(", "))
    }
}

/// 需求与供给之间的单条不满足项。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mismatch {
    /// 档案未声明该原语。
    PrimAbsent {
        /// 缺失的原语。
        prim: Prim,
    },
    /// 区间不够宽。
    ExtentTooSmall {
        /// 维度名（`m` / `n` / `k` / `bytes` / `lanes` / `axis`）。
        label: &'static str,
        /// 需求区间。
        need: Span,
        /// 供给区间。
        have: Span,
    },
    /// 数据类型缺失。
    DtMissing(DTypeSet),
    /// 存储空间缺失。
    SpaceMissing(SpaceSet),
    /// 属性缺失。
    AttrMissing {
        /// 属性键。
        key: AttrKey,
        /// 缺失的取值。
        missing: Vec<String>,
    },
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrimAbsent { prim } => write!(f, "该档案未声明 `{prim}`"),
            Self::ExtentTooSmall { label, need, have } => {
                write!(f, "{label} {} < {}", have.hi, need.lo)
            }
            Self::DtMissing(set) => write!(f, "dt 缺 {set}"),
            Self::SpaceMissing(set) => write!(f, "space 缺 {set}"),
            Self::AttrMissing { key, missing } => {
                write!(f, "{key} 缺 {}", missing.join(","))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bf16() -> DTypeSet {
        DTypeSet::of([DType::F16, DType::Bf16])
    }

    #[test]
    fn closed_sets_round_trip() {
        assert_eq!(Prim::ALL.len(), 16);
        for prim in Prim::ALL {
            assert_eq!(Prim::parse(prim.as_str()), Some(prim));
        }
        for dt in DType::ALL {
            assert_eq!(DType::parse(dt.as_str()), Some(dt));
        }
        for space in Space::ALL {
            assert_eq!(Space::parse(space.as_str()), Some(space));
        }
        for key in AttrKey::ALL {
            assert_eq!(AttrKey::parse(key.as_str()), Some(key));
        }
        assert_eq!(Prim::parse("tma_bulk"), None);
    }

    #[test]
    fn span_parses_four_forms() {
        assert_eq!(Span::parse("16").unwrap(), Span::exact(16));
        assert_eq!(Span::parse("16..64").unwrap(), Span::range(16, 64).unwrap());
        assert_eq!(Span::parse("<=512").unwrap(), Span::at_most(512));
        assert_eq!(Span::parse("32..").unwrap(), Span::at_least(32));
        assert!(Span::parse("64..16").is_err());
        assert!(Span::parse("abc").is_err());
    }

    #[test]
    fn covers_requires_range_containment() {
        let have = Span::range(16, 256).unwrap();
        assert!(have.covers(Span::range(16, 64).unwrap()));
        assert!(have.covers(Span::exact(256)));
        assert!(
            !have.covers(Span::range(8, 16).unwrap()),
            "下界越界不算覆盖"
        );
        assert!(!have.covers(Span::any()), "不限的需求只能由不限的供给覆盖");
    }

    #[test]
    fn diff_reports_every_missing_field() {
        let need = Need::of(Prim::Mma)
            .with_extent(Span::range(16, 64).unwrap())
            .with_n(Span::range(8, 128).unwrap())
            .with_k(Span::exact(16))
            .with_dt(f16_bf16())
            .with_space(SpaceSet::of([Space::Fast]))
            .with_attr(Attr::new(AttrKey::Op, ["sum"]));
        let have = Provide::of(Prim::Mma, "cube")
            .with_extent(Span::exact(16))
            .with_n(Span::exact(16))
            .with_k(Span::exact(16))
            .with_dt(DTypeSet::of([DType::F16]))
            .with_space(SpaceSet::of([Space::Fast]))
            .with_grade(Grade::L2);

        assert!(!have.covers(&need));
        let diff = have.diff(&need);
        let text = diff.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(
            text.contains(&"m 16 < 16".to_string()) || text.iter().any(|t| t.starts_with("m "))
        );
        assert!(text.contains(&"n 16 < 8".to_string()));
        assert!(text.contains(&"dt 缺 bf16".to_string()));
        assert!(text.contains(&"op 缺 sum".to_string()));
        assert!(!text.iter().any(|t| t.starts_with("space")));
    }

    #[test]
    fn exact_supply_covers_identical_need() {
        let need = Need::of(Prim::Load)
            .with_extent(Span::at_most(512))
            .with_space(SpaceSet::of([Space::Gm, Space::Fast]));
        let have = Provide::of(Prim::Load, "neon")
            .with_extent(Span::at_most(1024))
            .with_space(SpaceSet::of([Space::Gm, Space::Fast]));
        assert!(have.covers(&need));

        let too_wide = Provide::of(Prim::Load, "neon")
            .with_extent(Span::at_most(256))
            .with_space(SpaceSet::of([Space::Gm, Space::Fast]));
        assert_eq!(
            too_wide.diff(&need),
            vec![Mismatch::ExtentTooSmall {
                label: "bytes",
                need: Span::at_most(512),
                have: Span::at_most(256),
            }]
        );
    }

    #[test]
    fn missing_prim_is_reported_first() {
        let need = Need::of(Prim::LoadAsync);
        let have = Provide::of(Prim::Load, "memcpy");
        assert_eq!(
            have.diff(&need),
            vec![Mismatch::PrimAbsent {
                prim: Prim::LoadAsync
            }]
        );
    }

    #[test]
    fn signatures_render_like_design_diagnostic() {
        let need = Need::of(Prim::Mma)
            .with_extent(Span::range(16, 64).unwrap())
            .with_n(Span::range(8, 128).unwrap())
            .with_k(Span::exact(16))
            .with_dt(f16_bf16())
            .with_space(SpaceSet::of([Space::Fast]));
        assert_eq!(
            need.signature(),
            "mma{m:16..64, n:8..128, k:16, dt:[f16,bf16], space:[FAST]}"
        );

        let have = Provide::of(Prim::Load, "neon")
            .with_extent(Span::at_most(1024))
            .with_grade(Grade::L2);
        assert_eq!(have.summary(), "load{bytes:<=1024, impl:neon, grade:L2}");
    }

    #[test]
    fn grade_gate_is_l2() {
        assert!(!Grade::L0.is_verified());
        assert!(!Grade::L1.is_verified());
        assert!(Grade::L2.is_verified());
        assert!(Grade::L3.is_verified());
        assert!(Grade::L3 > Grade::L2);
    }

    #[test]
    fn dtype_set_ops() {
        let a = DTypeSet::of([DType::F16]);
        let b = f16_bf16();
        assert!(a.is_subset(b));
        assert!(!b.is_subset(a));
        assert_eq!(b.difference(a).names(), vec!["bf16"]);
        assert_eq!(DType::F4.size_bits(), 4);
    }
}
