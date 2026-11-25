//! 能力词汇表与参数化签名。

use core::fmt;
use core::hash::Hash;
use core::marker::PhantomData;

pub use crate::need::{Attr, Need, Provide, Witness};

// ── vocab ──────────────────────────────────────────────────────────────

/// 闭集：词汇表本身。`ALL` 是全量与顺序的唯一事实源，`as_str` 是跨层共用的名字（TOML、诊断、
/// 模板字符串同一份），`idx` 是位集下标；[`Self::parse`] 由前两者默认实现，闭集外返回 `None`。
pub trait Closed: Copy + Eq + Ord + Hash + 'static {
    /// 闭集全量；顺序即 [`Self::idx`]，也是诊断里的顺序。
    const ALL: &'static [Self];
    /// 跨层名字。
    fn as_str(self) -> &'static str;
    /// 位集下标；必须落在 `0..ALL.len()`。
    fn idx(self) -> usize;
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
        self.bits[item.idx() / WORD_BITS] |= 1 << (item.idx() % WORD_BITS);
    }
    /// 是否包含某元素。
    pub fn contains(self, item: E) -> bool {
        self.bits[item.idx() / WORD_BITS] & (1 << (item.idx() % WORD_BITS)) != 0
    }
    /// 是否为空集。
    pub const fn is_empty(self) -> bool {
        let mut idx = 0;
        while idx < WORDS && self.bits[idx] == 0 {
            idx += 1;
        }
        idx == WORDS
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
/// 变体顺序即 `ALL` 顺序、即位集下标；`as_str` 按 `idx` 查同序名字表。
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
                NAMES[self.idx()]
            }
            fn idx(self) -> usize { self as usize }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
        }
    };
}

vocabulary! {
    /// 原语闭集：模板与后端能力唯一能调用的操作（含登记过的设备级矩阵乘）。
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
    /// 这条原语的能力签名里允许出现的属性键（`exec` 之外），顺序即签名参数的顺序。
    pub fn attr_keys(self) -> &'static [AttrKey] {
        const ACCURACY: &[AttrKey] = &[AttrKey::Accuracy];
        const LANES: &[AttrKey] = &[AttrKey::Lanes];
        const ALIGNED: &[AttrKey] = &[AttrKey::Src, AttrKey::Dst, AttrKey::Align, AttrKey::Lanes];
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

vocabulary! {
    /// 写回尾块（epilogue）闭集：`gemm` / `conv` / `device` 写回阶段的六个步骤（[`crate::epilogue`] 的文法）。
    Epilogue {
        #[doc = "加列偏置。"] Bias => "bias",
        #[doc = "逐元素激活（参数在 [`ActKind`]）。"] Act => "act",
        #[doc = "加残差输入。"] Residual => "residual",
        #[doc = "乘标量。"] Scale => "scale",
        #[doc = "量化到整数位宽再反量化。"] Quant => "quant",
        #[doc = "写回。"] Store => "store",
    }
}
inherent_names!(Epilogue);

vocabulary! {
    /// 激活闭集：`Epilogue::Act` 的参数。
    ActKind {
        #[doc = "ReLU：`max(x, 0)`。"] Relu => "relu",
        #[doc = "GELU（精确 erf 版）。"] Gelu => "gelu",
        #[doc = "SiLU：`x·sigmoid(x)`。"] Silu => "silu",
        #[doc = "LeakyReLU（斜率 0.01）。"] LeakyRelu => "leaky_relu",
        #[doc = "HardSigmoid：`clamp(x/6 + 0.5, 0, 1)`。"] HardSigmoid => "hard_sigmoid",
    }
}
inherent_names!(ActKind);

/// 数据类型闭集：唯一事实源在 L1 `dtype`，这里只再导出并接上 [`Closed`] 契约。
pub use dtype::DType;
/// L1 的每个精度闭集都实现了 `dtype::Format`，因此 [`Closed`] 只需这一份 blanket 实现。
impl<T: dtype::Format> Closed for T {
    const ALL: &'static [Self] = T::ALL;
    fn as_str(self) -> &'static str {
        dtype::Format::name(self)
    }
    fn idx(self) -> usize {
        dtype::Format::idx(self)
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

// ── span ──────────────────────────────────────────────────────────────

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
            return Err(invalid(format!("区间下界 {lo} 大于上界 {hi}")));
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
    text.trim()
        .parse::<u32>()
        .map_err(|_| invalid(format!("`{}` 不是合法的非负整数", text.trim())))
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

/// 词汇层唯一的 `invalid` 构造点：闭集外取值、写法不合法、区间颠倒都从这里出门。
///
/// 收在一处是为了让"词汇层报错"只有一个口径（类别固定 `Invalid`），各模块不再各写一遍。
pub(crate) fn invalid(detail: impl Into<String>) -> diagnostic::Error {
    diagnostic::Error::new(diagnostic::Code::Invalid, detail)
}

// ── attr ──────────────────────────────────────────────────────────────

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
// ── key ───────────────────────────────────────────────────────────────
/// 一条内核实现的登记键（ADR 0027 §1 七元组）。
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

    /// 签名渲染：`dot{dt:f32, space:FAST, lanes:4, algorithm:cpu, block:16}/impl:scalar_f32`。
    pub fn signature(&self) -> String {
        format!(
            "{}{{dt:{}, space:{}, lanes:{}, algorithm:{}, block:{}}}/impl:{}",
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

#[cfg(test)]
#[path = "../tests/spec/atom.rs"]
mod tests;
