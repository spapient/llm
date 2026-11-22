//! 需求与供给的代数：`Need` 说"算子要什么"，`Provide` 说"这台设备给什么"。
//!
//! 协商只有一个判定（[`Provide::covers`]）与一个解释（[`Provide::diff`]：逐条报出没被覆盖的
//! 取值，而不是只回一个 `false`）。所有取值都在闭集与区间里，没有自由文本；[`Attr`] 是属性键
//! 上的一组取值（`sat = 1`、`kind = row`），需求的每个取值都必须被供给覆盖。

use crate::atom::*;

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

// ── signature ──────────────────────────────────────────────────────────────

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
            .map(|attr| format!("{}:{}", attr.key(), attr.values().join("|"))),
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

/// 一条可执行的见证（cutile README §11.3）：源码、入口函数与期望输出。
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
    let found = have.iter().find(|item| item.key() == attr.key());
    let missing: Vec<String> = attr
        .values()
        .iter()
        .filter(|value| found.is_none_or(|found| !found.contains(value)))
        .cloned()
        .collect();
    (!missing.is_empty()).then_some(Mismatch::AttrMissing {
        key: attr.key(),
        missing,
    })
}
