//! 原语闭集与参数化能力签名。
//!
//! 能力不是"有没有 `mma`"，而是"哪种 `mma`、支持多宽的形状、什么精度、在哪个存储层"
//! （cutile README §5.2）。因此 [`Provide::covers`] 只做一次区间包含判定，协商不需要启发式。

use core::fmt;
use core::hash::Hash;
use core::marker::PhantomData;
use diagnostic::{Code, Error};
/// 闭集：词汇表本身。`ALL` 是全量与顺序的唯一事实源，`as_str` 是跨层共用的名字（TOML、诊断、
/// 模板字符串同一份），`index` 是位集下标；[`Self::parse`] 由前两者默认实现，闭集外返回 `None`。
pub trait Closed: Copy + Eq + Ord + Hash + 'static {
    /// 闭集全量；顺序即 [`Self::index`]，也是诊断里的顺序。
    const ALL: &'static [Self];
    /// 跨层名字。
    fn as_str(self) -> &'static str;
    /// 位集下标；必须落在 `0..ALL.len()`。
    fn index(self) -> usize;
    /// 按名字解析；闭集外返回 `None`。
    fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|item| item.as_str() == name)
    }
}

/// 一个闭集的位集：`WORDS` 个 `u64`（const 参数），`Copy`、无堆分配；所有位运算只有这一份实现。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Set<E: Closed, const WORDS: usize> {
    bits: [u64; WORDS],
    element: PhantomData<fn() -> E>,
}
/// 每个 `u64` 能表示的元素个数。
const WORD_BITS: usize = u64::BITS as usize;
impl<E: Closed, const WORDS: usize> Set<E, WORDS> {
    /// 容量自证：引用它的构造点会在单态化时求值，写错宽度是编译错误。
    const FITS: () = assert!(
        E::ALL.len() <= WORDS * WORD_BITS,
        "位集容量不足：加大 `WORDS`"
    );
    /// 空集。
    pub const fn empty() -> Self {
        let () = Self::FITS;
        Self {
            bits: [0; WORDS],
            element: PhantomData,
        }
    }
    /// 由元素列表构造。
    pub fn of(items: impl IntoIterator<Item = E>) -> Self {
        let mut set = Self::empty();
        for item in items {
            set.insert(item);
        }
        set
    }
    /// 加入一个元素。
    pub fn insert(&mut self, item: E) {
        let () = Self::FITS;
        self.bits[item.index() / WORD_BITS] |= 1 << (item.index() % WORD_BITS);
    }
    /// 是否包含某元素。
    pub fn contains(self, item: E) -> bool {
        self.bits[item.index() / WORD_BITS] & (1 << (item.index() % WORD_BITS)) != 0
    }
    /// 是否为空集。
    pub const fn is_empty(self) -> bool {
        let mut index = 0;
        while index < WORDS && self.bits[index] == 0 {
            index += 1;
        }
        index == WORDS
    }
    /// 元素个数。
    pub fn len(self) -> u32 {
        self.bits.iter().map(|word| word.count_ones()).sum()
    }
    /// `self` 是否是 `other` 的子集。
    pub fn is_subset(self, other: Self) -> bool {
        self.difference(other).is_empty()
    }
    /// 在 `self` 中而不在 `other` 中的元素。
    pub fn difference(self, other: Self) -> Self {
        let mut bits = self.bits;
        bits.iter_mut()
            .zip(other.bits)
            .for_each(|(bits, other)| *bits &= !other);
        Self {
            bits,
            element: PhantomData,
        }
    }
    /// 按闭集顺序迭代（不分配）。
    pub fn iter(self) -> impl Iterator<Item = E> {
        E::ALL
            .iter()
            .copied()
            .filter(move |item| self.contains(*item))
    }
    /// 逐名字列出（升序，与闭集顺序一致）。
    pub fn names(self) -> Vec<&'static str> {
        self.iter().map(E::as_str).collect()
    }
}
impl<E: Closed, const WORDS: usize> Default for Set<E, WORDS> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<E: Closed, const WORDS: usize> fmt::Display for Set<E, WORDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("-");
        }
        f.write_str(&self.names().join(","))
    }
}

/// 为每个闭集补一份**同名**的固有转发：`Prim::ALL` / `prim.as_str()` / `Prim::parse(..)`。
///
/// 三个符号都转到 [`crate::atom::Closed`]，名字表仍然只有一份；调用方不必为了叫出名字去
/// `use` 一个 trait。其他层（`op` 的骨架与调度）用同一份转发。
#[macro_export]
macro_rules! inherent_names {
    ($($ty:ty),* $(,)?) => {
        $(
            impl $ty {
                /// 闭集全量（见 [`crate::atom::Closed::ALL`]）。
                pub const ALL: &'static [Self] = <Self as $crate::atom::Closed>::ALL;
                /// 跨层名字（见 [`crate::atom::Closed::as_str`]）。
                #[must_use]
                pub fn as_str(self) -> &'static str { <Self as $crate::atom::Closed>::as_str(self) }
                /// 按名字解析（见 [`crate::atom::Closed::parse`]）。
                #[must_use]
                pub fn parse(name: &str) -> Option<Self> {
                    <Self as $crate::atom::Closed>::parse(name)
                }
            }
        )*
    };
}

/// 声明一个闭集：变体、`ALL`、名字表与 `Display` 由同一张表展开，名字因此只有一份。
/// 变体顺序即 `ALL` 顺序、即位集下标；`as_str` 按 `index` 查同序名字表。
macro_rules! vocabulary {
    (
        $(#[$doc:meta])*
        $name:ident { $($(#[$variant_doc:meta])* $variant:ident => $text:literal),* $(,)? }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $name { $($(#[$variant_doc])* $variant,)* }
        impl Closed for $name {
            const ALL: &'static [Self] = &[$(Self::$variant),*];
            fn as_str(self) -> &'static str {
                const NAMES: &[&str] = &[$($text),*];
                NAMES[self.index()]
            }
            fn index(self) -> usize { self as usize }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
        }
    };
}
vocabulary! {
    /// 原语闭集：模板唯一能调用的 16 个指令级操作（cutile README §5.1 的 P1–P16）。
    Prim {
        #[doc = "`D = A × B + C`，tile 粒度、累积。"] Mma => "mma",
        #[doc = "整数点积（dp4a / dpas / dot4）。"] Dot => "dot",
        #[doc = "全局 → 片上，同步搬运。"] Load => "load",
        #[doc = "异步搬运 + 完成令牌。"] LoadAsync => "load_async",
        #[doc = "异步组提交 / 等待。"] CommitWait => "commit_wait",
        #[doc = "片上 → 全局。"] Store => "store",
        #[doc = "块内屏障 / arrive-wait。"] Bar => "bar",
        #[doc = "子组内 lane 交换。"] Shuffle => "shuffle",
        #[doc = "子组内广播。"] Broadcast => "broadcast",
        #[doc = "沿轴归约。"] Reduce => "reduce",
        #[doc = "沿轴包含前缀。"] Scan => "scan",
        #[doc = "快速 2 的幂。"] Exp2 => "exp2",
        #[doc = "倒数 / rsqrt。"] Rcp => "rcp",
        #[doc = "数据类型转换（含饱和转换）。"] Cvt => "cvt",
        #[doc = "缩放 / 零点应用、低位宽打包。"] Quant => "quant",
        #[doc = "全局原子。"] Atomic => "atomic",
    }
}
inherent_names!(Prim);

impl Prim {
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
    /// 这条原语的能力签名里允许出现的属性键（`exec` 之外），顺序即签名参数的顺序。
    ///
    /// 键不是全局通用的：`lanes` 对 `dot`/`shuffle` 有意义，对 `load` 没有。把"签名由哪些参数
    /// 构成"写成每原语的闭集后，声明层与档案层对同一份签名给出同一个答案。
    pub fn attr_keys(self) -> &'static [AttrKey] {
        const ACCURACY: &[AttrKey] = &[AttrKey::Accuracy];
        const LANES: &[AttrKey] = &[AttrKey::Lanes];
        const ALIGNED: &[AttrKey] = &[AttrKey::Src, AttrKey::Dst, AttrKey::Align];
        const AXIS_OP: &[AttrKey] = &[AttrKey::Axis, AttrKey::Op];
        const ATTR_KEYS: [&[AttrKey]; 16] = [
            ACCURACY,                                                       // Mma
            LANES,                                                          // Dot
            ALIGNED,                                                        // Load
            &[AttrKey::Src, AttrKey::Dst, AttrKey::Stages, AttrKey::Align], // LoadAsync
            &[AttrKey::Group],                                              // CommitWait
            ALIGNED,                                                        // Store
            &[],                                                            // Bar
            LANES,                                                          // Shuffle
            LANES,                                                          // Broadcast
            AXIS_OP,                                                        // Reduce
            AXIS_OP,                                                        // Scan
            ACCURACY,                                                       // Exp2
            &[AttrKey::Kind, AttrKey::Accuracy],                            // Rcp
            &[AttrKey::Sat],                                                // Cvt
            &[AttrKey::Group],                                              // Quant
            &[AttrKey::Op],                                                 // Atomic
        ];
        ATTR_KEYS[self as usize]
    }
    /// 某个属性键能否出现在这条原语的签名里；`exec` 对每条原语都成立。
    pub fn accepts(self, key: AttrKey) -> bool {
        key == AttrKey::Exec || self.attr_keys().contains(&key)
    }
}

/// 数据类型闭集：唯一事实源在 L1 `dtype`，这里只再导出并接上 [`Closed`] 契约。
pub use dtype::DType;
/// L1 的每个精度闭集都实现了 `dtype::Format`，因此 [`Closed`] 只需这一份 blanket 实现。
impl<T: dtype::Format> Closed for T {
    const ALL: &'static [Self] = T::ALL;
    fn as_str(self) -> &'static str {
        dtype::Format::name(self)
    }
    fn index(self) -> usize {
        dtype::Format::index(self)
    }
    fn parse(name: &str) -> Option<Self> {
        <Self as dtype::Format>::parse_name(name)
    }
}
/// 数据类型集合（位集），协商时做子集判定。
pub type DTypeSet = Set<DType, 1>;

vocabulary! {
    /// 三级存储的语义名（cutile README §6.3）：物理名由档案映射。
    Space {
        #[doc = "全局内存。"] Gm => "GM",
        #[doc = "片上快速存储（NVIDIA `shared`、Intel `slm`、昇腾 `L1`…）。"] Fast => "FAST",
        #[doc = "寄存器。"] Reg => "REG",
    }
}
inherent_names!(Space);
/// 存储空间集合（位集）。
pub type SpaceSet = Set<Space, 1>;

/// 一个闭区间（含两端）。`exact(16)` 表示"只能 16"，`at_most(512)` 表示"≤ 512"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    lo: u32,
    hi: u32,
}
impl Span {
    /// 两端构造；四个构造口径共用的私有入口。
    const fn new(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }
    /// 不限。
    pub const fn any() -> Self {
        Self::new(0, u32::MAX)
    }
    /// 精确值。
    pub const fn exact(value: u32) -> Self {
        Self::new(value, value)
    }
    /// 上界（下界 0）。
    pub const fn at_most(value: u32) -> Self {
        Self::new(0, value)
    }
    /// 下界（上界不限）。
    pub const fn at_least(value: u32) -> Self {
        Self::new(value, u32::MAX)
    }
    /// 区间；`lo > hi` 视为非法。
    pub fn range(lo: u32, hi: u32) -> diagnostic::Result<Self> {
        if lo > hi {
            return Err(Error::new(
                Code::Invalid,
                format!("区间下界 {lo} 大于上界 {hi}"),
            ));
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
    /// 解析 `16`、`16..64`、`<=512`、`16..` 四种写法；越界或写法不认识时返回 `invalid`。
    pub fn parse(text: &str) -> diagnostic::Result<Self> {
        let text = text.trim();
        if let Some(rest) = text.strip_prefix("<=") {
            return parse_u32(rest).map(Self::at_most);
        }
        match text.split_once("..") {
            Some((lo, hi)) => Self::parse_range(lo, hi),
            None => parse_u32(text).map(Self::exact),
        }
    }
    /// 区间写法 `lo..hi` / `lo..`（空上界表示不限）。
    fn parse_range(lo: &str, hi: &str) -> diagnostic::Result<Self> {
        let lo = parse_u32(lo)?;
        if hi.trim().is_empty() {
            return Ok(Self::at_least(lo));
        }
        Self::range(lo, parse_u32(hi)?)
    }
}
fn parse_u32(text: &str) -> diagnostic::Result<u32> {
    text.trim().parse::<u32>().map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("`{}` 不是合法的非负整数", text.trim()),
        )
    })
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

vocabulary! {
    /// 能力签名里的枚举型属性键（闭集）。
    AttrKey {
        #[doc = "归约 / 原子的运算种类。"] Op => "op",
        #[doc = "归约 / 扫描的轴名。"] Axis => "axis",
        #[doc = "变体类操作（`rsqrt` / `rcp`）。"] Kind => "kind",
        #[doc = "是否饱和。"] Sat => "sat",
        #[doc = "精度档。"] Accuracy => "accuracy",
        #[doc = "量化分组。"] Group => "group",
        #[doc = "点积 lanes。"] Lanes => "lanes",
        #[doc = "搬运源。"] Src => "src",
        #[doc = "搬运目标。"] Dst => "dst",
        #[doc = "预取级数。"] Stages => "stages",
        #[doc = "实现这个原语的硬件单元；单元名由档案的 `[units]` 给出，机制层不认识具体名字。"]
        Exec => "exec",
        #[doc = "加载 / 存储的起始对齐字节数（1 = 任意对齐）。"]
        Align => "align",
    }
}
inherent_names!(AttrKey);

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

/// 需求与供给共有的赋值式 builder：字段名与类型一致，用一张表展开，避免各抄一遍。
macro_rules! constraint_builders {
    ($($(#[$doc:meta])* $method:ident($field:ident: $ty:ty)),* $(,)?) => {
        $(
            $(#[$doc])*
            #[must_use]
            pub fn $method(mut self, $field: $ty) -> Self { self.$field = $field; self }
        )*
        /// 追加一个属性。
        #[must_use]
        pub fn with_attr(mut self, attr: Attr) -> Self { self.attrs.push(attr); self }
    };
}

/// 追加一个具名区间（不限的区间不渲染）。
fn push_span(parts: &mut Vec<String>, label: &str, span: Span) {
    if !span.is_any() {
        parts.push(format!("{label}:{span}"));
    }
}
/// 追加一个具名位集（空集不渲染）。
fn push_set<E: Closed, const WORDS: usize>(
    parts: &mut Vec<String>,
    label: &str,
    set: Set<E, WORDS>,
) {
    if !set.is_empty() {
        parts.push(format!("{label}:[{set}]"));
    }
}
/// 渲染需求/供给共享的具名字段；字段顺序是（原语, 主维度, `n`, `k`, 数据类型, 存储空间）。
fn constraint_parts(
    fields: (Prim, Span, Span, Span, DTypeSet, SpaceSet),
    attrs: &[Attr],
) -> Vec<String> {
    let (prim, extent, n, k, dt, space) = fields;
    let mut parts = Vec::new();
    push_span(&mut parts, prim.extent_label(), extent);
    push_span(&mut parts, "n", n);
    push_span(&mut parts, "k", k);
    push_set(&mut parts, "dt", dt);
    push_set(&mut parts, "space", space);
    parts.extend(
        attrs
            .iter()
            .map(|attr| format!("{}:{}", attr.key, attr.values.join("|"))),
    );
    parts
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
    constraint_builders! {
        #[doc = "设置主维度。"] with_extent(extent: Span),
        #[doc = "设置 `n`。"] with_n(n: Span),
        #[doc = "设置 `k`。"] with_k(k: Span),
        #[doc = "设置数据类型集合。"] with_dt(dt: DTypeSet),
        #[doc = "设置存储空间集合。"] with_space(space: SpaceSet),
    }
    /// 按签名文本渲染，供诊断使用：`mma{m:16..64, n:8..128, dt:[f16,bf16], space:[FAST]}`。
    pub fn signature(&self) -> String {
        let parts = constraint_parts(
            (self.prim, self.extent, self.n, self.k, self.dt, self.space),
            &self.attrs,
        );
        format!("{}{{{}}}", self.prim, parts.join(", "))
    }
}

vocabulary! {
    /// 证据等级（cutile README §11.2）。协商只接受 `L2` 及以上。
    Grade {
        #[doc = "档案里声明了。"] L0 => "L0",
        #[doc = "片段能编译。"] L1 => "L1",
        #[doc = "与参考对拍通过。"] L2 => "L2",
        #[doc = "真机性能达到 roofline 的 70%。"] L3 => "L3",
    }
}
inherent_names!(Grade);
impl Grade {
    /// 是否达到"可用于协商"的门槛（`L2`）。
    pub const fn is_verified(self) -> bool {
        matches!(self, Self::L2 | Self::L3)
    }
}

/// 一条可执行的见证（cutile README §11.3）：源码、入口函数与期望输出。
///
/// 路径相对片段层根目录（`cpu/witness.c`）。函数统一为 `int func(float *out)`：返回写入的
/// 元素个数，值按 f32 读回后与 [`Witness::expect`] 对拍。
#[derive(Debug, Clone, PartialEq)]
pub struct Witness {
    /// 源码路径（相对片段层根）。
    pub file: String,
    /// 入口函数名。
    pub func: String,
    /// 期望输出，按顺序；长度必须等于函数的返回值。
    pub expect: Vec<f32>,
}
impl Witness {
    /// 见证函数最多写入的元素个数（执行器据此定缓冲大小）。
    pub const MAX_OUT: usize = 8;
}

/// 供给：某设备档案对某原语的实际能力。
#[derive(Debug, Clone, PartialEq)]
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
    /// 可执行的见证（`L2` 及以上必填）。
    pub witness: Option<Witness>,
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
    constraint_builders! {
        #[doc = "设置主维度。"] with_extent(extent: Span),
        #[doc = "设置 `n`。"] with_n(n: Span),
        #[doc = "设置 `k`。"] with_k(k: Span),
        #[doc = "设置数据类型集合。"] with_dt(dt: DTypeSet),
        #[doc = "设置存储空间集合。"] with_space(space: SpaceSet),
    }
    /// 设置证据等级。
    #[must_use]
    pub fn with_grade(mut self, grade: Grade) -> Self {
        self.grade = grade;
        self
    }
    /// 设置见证。
    #[must_use]
    pub fn with_witness(mut self, witness: Witness) -> Self {
        self.witness = Some(witness);
        self
    }
    /// 是否覆盖某个需求。
    pub fn covers(&self, need: &Need) -> bool {
        self.diff(need).is_empty()
    }
    /// 逐项比对，返回全部不满足项；空向量等价于覆盖。顺序固定为：原语、主维度、`n`、`k`、
    /// 数据类型、存储空间、属性。
    pub fn diff(&self, need: &Need) -> Vec<Mismatch> {
        if self.prim != need.prim {
            return vec![Mismatch::PrimAbsent { prim: need.prim }];
        }
        let mut out = extent_mismatches(self, need);
        if !need.dt.is_subset(self.dt) {
            out.push(Mismatch::DtMissing(need.dt.difference(self.dt)));
        }
        if !need.space.is_subset(self.space) {
            out.push(Mismatch::SpaceMissing(need.space.difference(self.space)));
        }
        out.extend(missing_attrs(&need.attrs, &self.attrs));
        out
    }
    /// 单行签名，供 `doctor` 与诊断打印（cutile README §8.2 的 `have` 行）。
    pub fn summary(&self) -> String {
        let mut parts = constraint_parts(
            (self.prim, self.extent, self.n, self.k, self.dt, self.space),
            &self.attrs,
        );
        parts.push(format!("impl:{}", self.impl_name));
        parts.push(format!("grade:{}", self.grade));
        format!("{}{{{}}}", self.prim, parts.join(", "))
    }
}
/// 一条区间判据：`need` 不限视为通过，否则要求 `have` 覆盖它。
fn span_mismatch(label: &'static str, need: Span, have: Span) -> Option<Mismatch> {
    if need.is_any() || have.covers(need) {
        return None;
    }
    Some(Mismatch::ExtentTooSmall { label, need, have })
}
/// 主维度、`n`、`k` 三条区间判据，按诊断顺序收集。
fn extent_mismatches(have: &Provide, need: &Need) -> Vec<Mismatch> {
    [
        (need.prim.extent_label(), need.extent, have.extent),
        ("n", need.n, have.n),
        ("k", need.k, have.k),
    ]
    .into_iter()
    .filter_map(|(label, need, have)| span_mismatch(label, need, have))
    .collect()
}
/// 需求里的属性逐键比对：供给没有该键、或取值不全，都算缺失。
fn missing_attrs(need: &[Attr], have: &[Attr]) -> Vec<Mismatch> {
    need.iter()
        .filter_map(|attr| missing_attr(attr, have))
        .collect()
}
/// 单个属性键的缺失项；供给里没有该键或取值不全时返回缺失项，全部覆盖时 `None`。
fn missing_attr(attr: &Attr, have: &[Attr]) -> Option<Mismatch> {
    let found = have.iter().find(|item| item.key == attr.key);
    let missing: Vec<String> = attr
        .values
        .iter()
        .filter(|value| found.is_none_or(|found| !found.contains(value)))
        .cloned()
        .collect();
    (!missing.is_empty()).then_some(Mismatch::AttrMissing {
        key: attr.key,
        missing,
    })
}

/// 一条内核实现的登记键（ADR 0027 §1 七元组）。
///
/// 选择只走档案见证过的能力闭集；`arch` 与 `block` 是登记标签，不进热路径 `match`（ADR 0027
/// 中性段）。P0/P1 句柄是 prim 片段文本 + 档案见证入口；P3 由独立后端边界 crate 挂设备内核，
/// 本结构只承载身份，不持有可执行句柄。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    /// 原语闭集。
    pub prim: Prim,
    /// 实现名，与档案 `provides.impl_name` 同源（`scalar_f32` / `fma` / `dp4a` …）。
    pub impl_name: String,
    /// 元素类型；键取单点，不是位集。
    pub dtype: DType,
    /// 存储层级；键取单点。
    pub space: Space,
    /// 子组 / 向量宽度；标量实现记 `1`。
    pub lanes: u32,
    /// 指令集 / 后端登记标签（`cpu` / `neon` / `metal` / `cuda` …），不产生热路径分发。
    pub arch: String,
    /// 分块宽度（tile 尺寸 / 量化块宽）；标量实现记 `1`。
    pub block: u32,
}

impl KernelKey {
    /// 登记一条内核键；`lanes` / `block` 为 `0` 按标量归一成 `1`。
    pub fn new(
        prim: Prim,
        impl_name: impl Into<String>,
        dtype: DType,
        space: Space,
        lanes: u32,
        arch: impl Into<String>,
        block: u32,
    ) -> Self {
        Self {
            prim,
            impl_name: impl_name.into(),
            dtype,
            space,
            lanes: lanes.max(1),
            arch: arch.into(),
            block: block.max(1),
        }
    }

    /// 签名渲染：`dot{dt:f32, space:FAST, lanes:4, arch:cpu, block:16}/impl:scalar_f32`。
    pub fn signature(&self) -> String {
        format!(
            "{}{{dt:{}, space:{}, lanes:{}, arch:{}, block:{}}}/impl:{}",
            self.prim,
            self.dtype.name(),
            self.space.as_str(),
            self.lanes,
            self.arch,
            self.block,
            self.impl_name
        )
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
                write!(f, "{label}: 需求 {need} 不被 {have} 覆盖")
            }
            Self::DtMissing(set) => write!(f, "dt 缺 {set}"),
            Self::SpaceMissing(set) => write!(f, "space 缺 {set}"),
            Self::AttrMissing { key, missing } => write!(f, "{key} 缺 {}", missing.join(",")),
        }
    }
}

#[cfg(test)]
#[path = "../tests/atom_tests.rs"]
mod tests;
