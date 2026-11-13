//! 原语闭集与参数化能力签名。
//!
//! 能力不是"有没有 `mma`"，而是"哪种 `mma`、支持多宽的形状、什么精度、在哪个存储层"
//! （DESIGN §5.2）。因此 [`Provide::covers`] 只做一次区间包含判定，协商不需要任何启发式。

use core::fmt;
use core::hash::Hash;
use core::marker::PhantomData;
use diagnostic::{Code, Error};

/// 闭集：词汇表本身，而不是"一个枚举"。
///
/// 每个词汇表都要回答三件事，因此它们写成契约而不是各抄一遍：`ALL` 是全量与顺序的**唯一**
/// 事实源，`as_str` 是跨层共用的名字（TOML、诊断、模板字符串是同一个），`index` 是位集下标。
/// [`Self::parse`] 由前两者默认实现：闭集外的名字一律 `None`，"闭集里有哪些"的文案由调用方给
/// （它比这里更清楚读者在写哪一行）。
///
/// 契约是**可静态判定**的：`ALL` 与 `index` 都是常量，`Set` 的位宽因此能在编译期核对，
/// 词汇表的增减只改一个 `impl`，不牵动任何分发。
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

/// 一个闭集的位集。
///
/// 宽度是 `WORDS` 个 `u64`（const 参数），因此集合是 `Copy`、无堆分配，容量在编译期核对：
/// `Set<E, 1>` 装不下 `E::ALL.len() > 64` 的闭集，那是编译错误而不是越界。
///
/// 一份实现覆盖所有闭集："空集 / 插入 / 包含 / 子集 / 差集 / 按闭集顺序列名字"只有这一处代码，
/// `DTypeSet` / `SpaceSet` 这类别名因此不必各自再写一遍位运算。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Set<E: Closed, const WORDS: usize> {
    bits: [u64; WORDS],
    element: PhantomData<fn() -> E>,
}

/// 每个 `u64` 能表示的元素个数。
const WORD_BITS: usize = u64::BITS as usize;

impl<E: Closed, const WORDS: usize> Set<E, WORDS> {
    /// 容量自证：闭集放得进 `WORDS` 个 word。
    ///
    /// 引用它的每个构造点都会让编译器在**单态化时**求值，因此写错宽度是编译错误。
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
        while index < WORDS {
            if self.bits[index] != 0 {
                return false;
            }
            index += 1;
        }
        true
    }

    /// 元素个数。
    pub const fn len(self) -> u32 {
        let mut total = 0;
        let mut index = 0;
        while index < WORDS {
            total += self.bits[index].count_ones();
            index += 1;
        }
        total
    }

    /// `self` 是否是 `other` 的子集。
    pub const fn is_subset(self, other: Self) -> bool {
        let mut index = 0;
        while index < WORDS {
            if self.bits[index] & !other.bits[index] != 0 {
                return false;
            }
            index += 1;
        }
        true
    }

    /// 在 `self` 中而不在 `other` 中的元素。
    pub const fn difference(self, other: Self) -> Self {
        let mut bits = [0; WORDS];
        let mut index = 0;
        while index < WORDS {
            bits[index] = self.bits[index] & !other.bits[index];
            index += 1;
        }
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

impl Closed for Prim {
    /// 闭集全量，顺序与 DESIGN §5.1 一致。
    const ALL: &'static [Self] = &[
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
    fn as_str(self) -> &'static str {
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

    fn index(self) -> usize {
        self as usize
    }
}

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

    /// 这条原语的能力签名里允许出现的属性键（`exec` 之外）。
    ///
    /// 属性键不是全局通用的：`lanes` 对 `dot`/`shuffle` 有意义，对 `load` 没有。把"一个签名
    /// 由哪些参数构成"写成每原语的闭集后，声明层（`op::decl`）与档案层
    /// （[`crate::prof::Profile::validate`]）对同一份签名给出同一个答案——否则一个永远不被
    /// 协商检查的键（`load.sat`）会让"声明了"与"生效了"分家。
    pub fn attr_keys(self) -> &'static [AttrKey] {
        match self {
            Self::Mma => &[AttrKey::Accuracy],
            Self::Dot | Self::Shuffle | Self::Broadcast => &[AttrKey::Lanes],
            Self::Load | Self::Store => &[AttrKey::Src, AttrKey::Dst],
            Self::LoadAsync => &[AttrKey::Src, AttrKey::Dst, AttrKey::Stages],
            Self::CommitWait => &[AttrKey::Group],
            Self::Bar => &[],
            Self::Reduce | Self::Scan => &[AttrKey::Axis, AttrKey::Op],
            Self::Exp2 => &[AttrKey::Accuracy],
            Self::Rcp => &[AttrKey::Kind, AttrKey::Accuracy],
            Self::Cvt => &[AttrKey::Sat],
            Self::Quant => &[AttrKey::Group],
            Self::Atomic => &[AttrKey::Op],
        }
    }

    /// 某个属性键能否出现在这条原语的签名里。
    ///
    /// `exec` 对每条原语都成立：它说的是"由哪个硬件单元实现"，与语义无关。
    pub fn accepts(self, key: AttrKey) -> bool {
        key == AttrKey::Exec || self.attr_keys().contains(&key)
    }
}

impl fmt::Display for Prim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 数据类型闭集：唯一事实源在 L1 `dtype`，这里只再导出并接上 [`Closed`] 契约。
pub use dtype::DType;

/// L1 的每个精度闭集都实现了 `dtype::Format`（全量表 + 判别值 + 规范名），因此 [`Closed`]
/// 只需这一份 blanket 实现：`ALL`/`as_str`/`index` 不再按类型手写，也不会与 L1 漂移。
impl<T: dtype::Format> Closed for T {
    const ALL: &'static [Self] = T::ALL;

    fn as_str(self) -> &'static str {
        dtype::Format::name(self)
    }

    fn index(self) -> usize {
        dtype::Format::index(self)
    }

    /// 按名解析同样转调 L1：闭集的名字解析只有一份实现（`Format::parse_name`）。
    fn parse(name: &str) -> Option<Self> {
        <Self as dtype::Format>::parse_name(name)
    }
}

/// 数据类型集合（位集），协商时做子集判定。
pub type DTypeSet = Set<DType, 1>;

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

impl Closed for Space {
    /// 闭集全量。
    const ALL: &'static [Self] = &[Self::Gm, Self::Fast, Self::Reg];

    /// 模板与档案里使用的名字（大写，与 DESIGN 一致）。
    fn as_str(self) -> &'static str {
        match self {
            Self::Gm => "GM",
            Self::Fast => "FAST",
            Self::Reg => "REG",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 存储空间集合（位集）。
pub type SpaceSet = Set<Space, 1>;

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

    /// 解析三种写法：`16`、`16..64`、`<=512`、`16..`。
    ///
    /// # Errors
    ///
    /// 写法不在上述三种之内、或数字越界时返回 `invalid`。
    pub fn parse(text: &str) -> diagnostic::Result<Self> {
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
    /// 执行单元：实现这个原语的硬件单元（`cube` / `vector` / `mte` / `simt` …）。
    ///
    /// 单元名由档案的 `[units]` 给出，机制层不认识任何具体名字：因此"某一层存储只有
    /// 某个单元访问得到"（昇腾 `cube` 只能到 `L1`/`L0`）是数据里的一条声明，而不是代码里的
    /// 一个分支。
    Exec,
}

impl Closed for AttrKey {
    /// 闭集全量。
    const ALL: &'static [Self] = &[
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
        Self::Exec,
    ];

    /// 声明层里使用的键名。
    fn as_str(self) -> &'static str {
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
            Self::Exec => "exec",
        }
    }

    fn index(self) -> usize {
        self as usize
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

/// 需求与供给共有的六个 builder：字段名与类型在两个结构体里一致，用一份模板展开，避免各抄一遍。
macro_rules! constraint_builders {
    () => {
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
    };
}

/// 渲染需求/供给共享的具名字段：主维度、`n`、`k`、数据类型、存储空间与属性。
fn constraint_parts(
    prim: Prim,
    extent: Span,
    n: Span,
    k: Span,
    dt: DTypeSet,
    space: SpaceSet,
    attrs: &[Attr],
) -> Vec<String> {
    let mut parts = Vec::new();
    if !extent.is_any() {
        parts.push(format!("{}:{}", prim.extent_label(), extent));
    }
    for (label, span) in [("n", n), ("k", k)] {
        if !span.is_any() {
            parts.push(format!("{label}:{span}"));
        }
    }
    if !dt.is_empty() {
        parts.push(format!("dt:[{}]", dt));
    }
    if !space.is_empty() {
        parts.push(format!("space:[{}]", space));
    }
    for attr in attrs {
        parts.push(format!("{}:{}", attr.key, attr.values.join("|")));
    }
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

    constraint_builders!();

    /// 按签名文本渲染，供诊断使用：`mma{m:16..64, n:8..128, dt:[f16,bf16], space:[FAST]}`。
    pub fn signature(&self) -> String {
        let parts = constraint_parts(
            self.prim,
            self.extent,
            self.n,
            self.k,
            self.dt,
            self.space,
            &self.attrs,
        );
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

impl Closed for Grade {
    /// 闭集全量，从低到高。
    const ALL: &'static [Self] = &[Self::L0, Self::L1, Self::L2, Self::L3];

    /// 渲染名。
    fn as_str(self) -> &'static str {
        match self {
            Self::L0 => "L0",
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

impl Grade {
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

/// 为每个闭集补一份**同名**的固有转发：`Prim::ALL` / `prim.as_str()` / `Prim::parse(..)`。
///
/// 转发而不是复制逻辑：三个符号都直接转到 [`crate::atom::Closed`]，所以"名字表只有一份"这条
/// 不变量仍然成立；而调用方不必为了叫出一个词汇表的名字去 `use` 一个 trait —— `as_str` 是跨层
/// 到处都在用的名字，让它既是契约的一部分、又仍然可以直接调用，比让每个调用点补一行 `use`
/// 更可审计。其他层（`op` 的骨架与调度）用同一份转发，词汇表因此不是"每个 crate 抄一遍"。
#[macro_export]
macro_rules! inherent_names {
    ($($ty:ty),* $(,)?) => {
        $(
            impl $ty {
                /// 闭集全量（见 [`crate::atom::Closed::ALL`]）。
                pub const ALL: &'static [Self] = <Self as $crate::atom::Closed>::ALL;

                /// 跨层名字（见 [`crate::atom::Closed::as_str`]）。
                #[must_use]
                pub fn as_str(self) -> &'static str {
                    <Self as $crate::atom::Closed>::as_str(self)
                }

                /// 按名字解析（见 [`crate::atom::Closed::parse`]）。
                #[must_use]
                pub fn parse(name: &str) -> Option<Self> {
                    <Self as $crate::atom::Closed>::parse(name)
                }

            }
        )*
    };
}

// DType 是 L1 外部类型，不能在本 crate 写固有 impl；它的 `ALL` / `name` / `parse_name`
// 已在 dtype crate 内定义，调用点直接用 L1 名字。
inherent_names!(Prim, Space, AttrKey, Grade);

/// 一条可执行的见证（DESIGN §11.3）：源码、入口函数与期望输出。
///
/// 路径相对片段层根目录（`cpu/witness.c`）。函数统一为 `int func(float *out)`：返回写入的
/// 元素个数，值按 f32 读回后与 [`Witness::expect`] 对拍。有了期望值，见证才是判据而不是注释。
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
    /// 见证函数最多写入的元素个数（执行器据此定缓冲大小，避免被恶意返回值带偏）。
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

    constraint_builders!();

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
        let mut parts = constraint_parts(
            self.prim,
            self.extent,
            self.n,
            self.k,
            self.dt,
            self.space,
            &self.attrs,
        );
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
        DTypeSet::of([DType::F16, DType::BF16])
    }

    #[test]
    fn closed_sets_round_trip() {
        assert_eq!(Prim::ALL.len(), 16);
        for prim in Prim::ALL.iter().copied() {
            assert_eq!(Prim::parse(prim.as_str()), Some(prim));
        }
        for dt in DType::ALL.iter().copied() {
            assert_eq!(DType::parse_name(dt.name()), Some(dt));
        }
        for space in Space::ALL.iter().copied() {
            assert_eq!(Space::parse(space.as_str()), Some(space));
        }
        for key in AttrKey::ALL.iter().copied() {
            assert_eq!(AttrKey::parse(key.as_str()), Some(key));
        }
        assert_eq!(Prim::parse("tma_bulk"), None);
    }

    #[test]
    fn index_is_the_position_in_the_closed_set() {
        // 位集下标就是 `ALL` 里的位置：这条不变量让“加一个原语只改一个 impl”成立，
        // 也让 `Set<E, 1>` 的容量自证有意义（超过 64 个元素要显式加宽）。
        fn positions<T: Closed>(label: &str) {
            for (position, item) in T::ALL.iter().copied().enumerate() {
                assert_eq!(item.index(), position, "{label}: {}", item.as_str());
            }
        }
        positions::<Prim>("prim");
        positions::<DType>("dtype");
        positions::<Space>("space");
        positions::<AttrKey>("attr");
        positions::<Grade>("grade");
    }

    #[test]
    fn a_set_holds_every_element_of_its_closed_set() {
        let all = SpaceSet::of(Space::ALL.iter().copied());
        assert_eq!(all.len(), 3);
        assert!(all.iter().eq(Space::ALL.iter().copied()));
        assert_eq!(all.names(), vec!["GM", "FAST", "REG"]);
        assert_eq!(all.to_string(), "GM,FAST,REG");
        assert!(SpaceSet::empty().is_subset(all));
        assert!(!all.is_subset(SpaceSet::empty()));
        assert_eq!(all.difference(all).len(), 0);
        assert_eq!(SpaceSet::of([Space::Fast]).difference(all).len(), 0);
        assert_eq!(
            all.difference(SpaceSet::of([Space::Fast])).to_string(),
            "GM,REG"
        );
        // 宽度自证：`Set<E, 1>` 装得下所有闭集，因此常量构造在编译期就通过检查。
        const EMPTY: SpaceSet = SpaceSet::empty();
        assert!(EMPTY.is_empty());
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
        assert_eq!(DType::F4.bits(), 4);
    }
}
