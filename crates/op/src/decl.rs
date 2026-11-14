//! 声明解析与轴展开（cutile README §4、§10.2 的 ①）。
//!
//! 文本来自声明层（磁盘与内嵌两种来源，见 [`crate::Op`]），本模块只解释内容、不决定数据放在哪。
//! 解析的结果是 [`OpDecl`]：机制层（`cutile`）从这一步之后只消费它，不再看 TOML。

use crate::layout::LayoutSpec;
use crate::sch;
use crate::sch::Schedule;
use crate::tpl::{Arg, Template};
use core::fmt;
use diagnostic::data::{Table, Value};
use diagnostic::{Code, Error};
use prof::atom::{Attr, AttrKey, DType, DTypeSet, Need, Prim, Space, SpaceSet, Span};

/// 变体数上限（cutile README §4.1）。
pub const MAX_VARIANTS: usize = 8;
/// 轴展开后的 tile 候选上限（cutile README §4.1）。
pub const MAX_TILES: usize = 8;

/// 骨架闭集（cutile README 附录 B）：决定用哪个模板基座。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Skeleton {
    /// 逐元素。
    Elementwise,
    /// 归约。
    Reduce,
    /// 矩阵乘。
    Gemm,
    /// 注意力。
    Attention,
    /// 扫描。
    Scan,
}

impl prof::atom::Closed for Skeleton {
    /// 闭集全量。
    const ALL: &'static [Self] = &[
        Self::Elementwise,
        Self::Reduce,
        Self::Gemm,
        Self::Attention,
        Self::Scan,
    ];

    /// 声明层写法。
    fn as_str(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Reduce => "reduce",
            Self::Gemm => "gemm",
            Self::Attention => "attention",
            Self::Scan => "scan",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

prof::inherent_names!(Skeleton);

impl fmt::Display for Skeleton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 一个轴：有限枚举，展开后成为 tile 候选。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axis {
    /// 轴名（也是模板里的槽位名）。
    pub name: String,
    /// 候选取值。
    pub values: Vec<u32>,
}

/// 降级链上的一个变体。
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// 变体名。
    pub name: String,
    /// 优先级，小者优先。
    pub order: u32,
    /// 调度模板。
    pub schedule: Schedule,
    /// 非 MMA 占比上界。
    pub mma_share: f64,
    /// 变体级增量需求：只能**加强** `[needs]`，不能少写模板用到的原语。
    pub extra_needs: Vec<Need>,
}

impl Variant {
    /// 该变体的完整需求 = 算子级公共需求 + 变体级增量需求。
    pub fn needs<'a>(&'a self, op: &'a OpDecl) -> impl Iterator<Item = &'a Need> {
        op.needs.iter().chain(self.extra_needs.iter())
    }
}

/// 一个算子的声明加模板。
#[derive(Debug, Clone, PartialEq)]
pub struct OpDecl {
    /// 算子名。
    pub name: String,
    /// 骨架。
    pub skeleton: Skeleton,
    /// 一句话说明。
    pub summary: String,
    /// 轴。
    pub axes: Vec<Axis>,
    /// 算子级公共需求。
    pub needs: Vec<Need>,
    /// 布局契约：槽位 → 构造子。
    pub layouts: Vec<(String, LayoutSpec)>,
    /// 降级链，按 `order` 升序。
    pub variants: Vec<Variant>,
    /// 模板文本。
    pub template: String,
    /// 内核签名：从模板的入口函数解析，是 ABI 的唯一事实源。
    pub signature: crate::sig::Signature,
}

/// 轴展开出的一组具体值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileCombo {
    values: Vec<(String, u32)>,
}

impl TileCombo {
    /// 取一个轴的值。
    pub fn get(&self, name: &str) -> Option<u32> {
        self.values
            .iter()
            .find(|(axis, _)| axis == name)
            .map(|(_, value)| *value)
    }

    /// 全部取值。
    pub fn values(&self) -> &[(String, u32)] {
        &self.values
    }

    /// 渲染成 `GMAX=32` 形式。
    pub fn summary(&self) -> String {
        self.values
            .iter()
            .map(|(axis, value)| format!("{axis}={value}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl fmt::Display for TileCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// 字段级错误：`invalid` 类别，消息里带字段路径（声明层的一切错误都是"输入不合法"）。
fn field(path: &str, detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("`{path}` 非法：{}", detail.into()))
}

/// 算子名：`[op].name`，非空。
fn op_name(op: &Table) -> diagnostic::Result<String> {
    let name = op.string("name").ok_or_else(|| field("op.name", "缺失"))?;
    if name.is_empty() {
        return Err(field("op.name", "不能为空"));
    }
    Ok(name.to_string())
}

/// `[op]` 段读出的身份：名字、骨架、摘要。
fn head(table: &Table) -> diagnostic::Result<(String, Skeleton, String)> {
    let op = table
        .table("op")
        .ok_or_else(|| field("op", "缺少 `[op]` 段"))?;
    let (skeleton, summary) = op_head(op)?;
    Ok((op_name(op)?, skeleton, summary))
}

/// `[op]` 的其余两键：骨架必须落在闭集内，摘要缺省为空串占位的 `-`。
fn op_head(op: &Table) -> diagnostic::Result<(Skeleton, String)> {
    let skeleton = Skeleton::parse(
        op.string("skeleton")
            .ok_or_else(|| field("op.skeleton", "缺失"))?,
    )
    .ok_or_else(|| {
        let allowed: Vec<&str> = Skeleton::ALL.iter().map(|item| item.as_str()).collect();
        field("op.skeleton", format!("闭集：{}", allowed.join(" | ")))
    })?;
    Ok((skeleton, op.string("summary").unwrap_or("-").to_string()))
}

/// 四个段（轴、需求、布局、变体链）：`declared` 按同一顺序解构。
type Sections = (
    Vec<Axis>,
    Vec<Need>,
    Vec<(String, LayoutSpec)>,
    Vec<Variant>,
);

fn sections(table: &Table) -> diagnostic::Result<Sections> {
    Ok((
        axes(table)?,
        needs(table)?,
        layouts(table)?,
        variants(table)?,
    ))
}

/// `[axes]`：每个键一组正整数取值；至少要有一条轴。
fn axes(table: &Table) -> diagnostic::Result<Vec<Axis>> {
    let Some(section) = table.table("axes") else {
        return Err(field("axes", "至少需要一个轴"));
    };
    let out: Vec<Axis> = section
        .entries()
        .iter()
        .map(|(key, value)| axis(key, value))
        .collect::<diagnostic::Result<_>>()?;
    if out.is_empty() {
        return Err(field("axes", "至少需要一个轴"));
    }
    Ok(out)
}

/// 一条轴：`axes.{名字}` 是非空的正整数数组。
fn axis(key: &str, value: &Value) -> diagnostic::Result<Axis> {
    let path = format!("axes.{key}");
    let Value::Array(items) = value else {
        return Err(field(&path, "必须是数组"));
    };
    let values = items
        .iter()
        .map(|item| positive(&path, item))
        .collect::<diagnostic::Result<Vec<u32>>>()?;
    if values.is_empty() {
        return Err(field(&path, "数组不能为空"));
    }
    Ok(Axis {
        name: key.to_string(),
        values,
    })
}

/// 轴的一个取值：正整数。
fn positive(path: &str, item: &Value) -> diagnostic::Result<u32> {
    match item {
        Value::Int(number) if *number > 0 => Ok(*number as u32),
        _ => Err(field(path, "每个取值必须是正整数")),
    }
}

/// `[needs]`：算子级公共需求，每条一个原语的能力签名。
fn needs(table: &Table) -> diagnostic::Result<Vec<Need>> {
    let Some(section) = table.table("needs") else {
        return Ok(Vec::new());
    };
    section
        .entries()
        .iter()
        .map(|(key, _)| parse_need(section, key, &format!("needs.{key}")))
        .collect()
}

/// `[layouts]`：槽位名 → 布局闭集里的一个词。
fn layouts(table: &Table) -> diagnostic::Result<Vec<(String, LayoutSpec)>> {
    let Some(section) = table.table("layouts") else {
        return Ok(Vec::new());
    };
    section
        .entries()
        .iter()
        .map(|(key, value)| layout(key, value))
        .collect()
}

/// 一条布局：`layouts.{槽位}` 必须是布局闭集内的字符串。
fn layout(key: &str, value: &Value) -> diagnostic::Result<(String, LayoutSpec)> {
    let path = format!("layouts.{key}");
    let Value::Str(text) = value else {
        return Err(field(&path, "必须是字符串"));
    };
    let spec = LayoutSpec::parse(text).map_err(|detail| field(&path, detail.message()))?;
    Ok((key.to_string(), spec))
}

/// 变体链：逐个解析、按 `order` 排序，再判数量与链路。
fn variants(table: &Table) -> diagnostic::Result<Vec<Variant>> {
    let mut variants: Vec<Variant> = table
        .table_array("variants")
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(index, entry)| variant(index, entry))
        .collect::<diagnostic::Result<_>>()?;
    check_count(&variants)?;
    variants.sort_by_key(|variant| variant.order);
    check_chain(&variants)?;
    Ok(variants)
}

/// 变体数：至少一个、至多 [`MAX_VARIANTS`] 个。
fn check_count(variants: &[Variant]) -> diagnostic::Result<()> {
    if variants.is_empty() {
        return Err(field("variants", "至少需要一个变体"));
    }
    if variants.len() > MAX_VARIANTS {
        return Err(field(
            "variants",
            format!("{} 个变体超过上限 {MAX_VARIANTS}", variants.len()),
        ));
    }
    Ok(())
}

/// 降级链的判据：名字唯一、非 MMA 占比单调，且首选的掩码能落地。
///
/// 掩码只判首选：兜底变体的存在理由就是"设备缺原语时仍能跑"（见 `sch::check_masking`）。
fn check_chain(variants: &[Variant]) -> diagnostic::Result<()> {
    for pair in variants.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(field("variants", format!("变体名 `{}` 重复", pair[0].name)));
        }
    }
    // 借用变体名：这条判据只读名字与占比，没有理由在装载期为它拷一份字符串。
    let chain: Vec<(&str, f64)> = variants
        .iter()
        .map(|variant| (variant.name.as_str(), variant.mma_share))
        .collect();
    sch::check_monotonic(&chain).map_err(|detail| field("variants", detail.message()))?;
    let Some(head) = variants.first() else {
        return Ok(());
    };
    sch::check_masking(&head.name, sch::Masking::of(head.mma_share, &head.schedule))
        .map_err(|detail| field("variants", detail.message()))
}

/// 一个 `[[variants]]` 段：名字与优先级、调度、非 MMA 占比、增量需求。
fn variant(index: usize, entry: &Table) -> diagnostic::Result<Variant> {
    let (name, order) = identity(index, entry)?;
    let schedule = schedule(index, entry)?;
    let mma_share = share(index, entry)?;
    Ok(Variant {
        name,
        order,
        schedule,
        mma_share,
        extra_needs: extra_needs(index, entry)?,
    })
}

/// 变体名与优先级：两个键都必填，负的 `order` 按 0 处理。
fn identity(index: usize, entry: &Table) -> diagnostic::Result<(String, u32)> {
    let name = entry
        .string("name")
        .ok_or_else(|| field(&field_path(index, "name"), "缺失"))?
        .to_string();
    let order = entry
        .int("order")
        .ok_or_else(|| field(&field_path(index, "order"), "缺失"))?
        .max(0) as u32;
    Ok((name, order))
}

/// 调度：`schedule` 必须落在调度闭集内。
fn schedule(index: usize, entry: &Table) -> diagnostic::Result<Schedule> {
    let path = field_path(index, "schedule");
    let text = entry
        .string("schedule")
        .ok_or_else(|| field(&path, "缺失"))?;
    Schedule::parse(text).map_err(|detail| field(&path, detail.message()))
}

/// 非 MMA 占比：`0..=1` 之间。
fn share(index: usize, entry: &Table) -> diagnostic::Result<f64> {
    let path = field_path(index, "mma_share");
    let mma_share = entry
        .float("mma_share")
        .ok_or_else(|| field(&path, "缺失"))?;
    if (0.0..=1.0).contains(&mma_share) {
        return Ok(mma_share);
    }
    Err(field(&path, "必须在 0..=1 之间"))
}

/// `extra_needs`：变体自己追加的需求。
fn extra_needs(index: usize, entry: &Table) -> diagnostic::Result<Vec<Need>> {
    let Some(section) = entry.table("extra_needs") else {
        return Ok(Vec::new());
    };
    let prefix = field_path(index, "");
    section
        .entries()
        .iter()
        .map(|(key, _)| parse_need(section, key, &format!("{prefix}.extra_needs.{key}")))
        .collect()
}

/// `variants[i].{name}` 形式的字段路径；`name` 为空时只剩前缀。
fn field_path(index: usize, name: &str) -> String {
    format!("variants[{index}].{name}")
}

impl OpDecl {
    /// 从编译期生成的声明表解析声明，并解析同名模板。
    ///
    /// 按「键 → 取值语法」分工；模板侧的三条判据见 `Self::verified`。
    pub fn parse(table: &Table, template: &str) -> diagnostic::Result<Self> {
        let op_decl = Self::declared(table, template)?;
        op_decl.verified(template)?;
        Ok(op_decl)
    }

    /// 声明本体：头、四个段与签名。模板侧校验不在这里（见 [`Self::verified`]）。
    fn declared(table: &Table, template: &str) -> diagnostic::Result<Self> {
        let (name, skeleton, summary) = head(table)?;
        // 返回顺序：轴、需求、布局、变体链。
        let (axes, needs, layouts, variants) = sections(table)?;
        let signature = crate::sig::Signature::parse(&name, template)?;
        Ok(Self {
            name,
            skeleton,
            summary,
            axes,
            needs,
            layouts,
            variants,
            template: template.to_string(),
            signature,
        })
    }

    /// 模板侧的三条判据：槽位有定义、每条原语都被申请、调用点参数落在签名内。
    ///
    /// 模板只切一遍：三处校验看同一份 [`Template`]，不再各 `lex` 一次。
    fn verified(&self, template: &str) -> diagnostic::Result<()> {
        let segments = Template::parse(template);
        tiles(self)?;
        self.check_slots(&segments)?;
        self.check_needs(&segments)?;
        self.check_call_arguments(&segments)
    }

    /// 按名字取变体。
    pub fn variant(&self, name: &str) -> Option<&Variant> {
        self.variants.iter().find(|variant| variant.name == name)
    }

    /// 按槽位名取布局。
    pub fn layout(&self, slot: &str) -> Option<LayoutSpec> {
        self.layouts
            .iter()
            .find(|(name, _)| name == slot)
            .map(|(_, spec)| *spec)
    }

    /// 按名字取轴。
    pub fn axis(&self, name: &str) -> Option<&Axis> {
        self.axes.iter().find(|axis| axis.name == name)
    }

    /// 模板里的每个 `{{槽位}}` 必须能被轴、布局或 `op` 解析（提前拦住 `{{GMAXX}}` 这类笔误）。
    fn check_slots(&self, template: &Template<'_>) -> diagnostic::Result<()> {
        for (line, slot) in template.slots() {
            if slot == "op" {
                continue;
            }
            if self.axis(slot).is_none() && self.layout(slot).is_none() {
                return Err(field(
                    &format!("{}.k:{line}", self.name),
                    format!("槽位 `{{{{{slot}}}}}` 在 [axes] 和 [layouts] 里都没有定义"),
                ));
            }
        }
        Ok(())
    }

    /// 每个变体的需求必须覆盖模板里出现的全部 `@原语`：模板是各变体共享的，所以按变体判，
    /// 不取并集（否则某个变体漏申请会被掩盖），也不等到协商时才发现。
    fn check_needs(&self, template: &Template<'_>) -> diagnostic::Result<()> {
        for variant in &self.variants {
            let mut declared: Vec<Prim> = variant.needs(self).map(|need| need.prim).collect();
            declared.extend(variant.schedule.required_prims().iter().copied());
            for (line, prim, _) in template.calls() {
                if declared.contains(&prim) {
                    continue;
                }
                return Err(field(
                    &format!("{}.k:{line}", self.name),
                    format!(
                        "变体 `{}` 没有声明模板用到的 `@{prim}`：模板由所有变体共享，每条原语都要在该变体的需求里",
                        variant.name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// 模板调用点的能力参数必须落在该变体 `[needs]` 声明的能力签名之内：否则设备按一条签名
    /// 验证、代码按另一条生成。只判闭集内的属性键，闭集外的命名参数留给片段。
    fn check_call_arguments(&self, template: &Template<'_>) -> diagnostic::Result<()> {
        for variant in &self.variants {
            for (line, prim, args) in template.calls() {
                for (key, value) in args.iter().filter_map(Arg::named) {
                    self.check_argument(variant, prim, line, key, value)?;
                }
            }
        }
        Ok(())
    }

    /// 一个能力参数：闭集内的键必须在 `needs.{prim}` 里声明过，且取值被声明允许。
    ///
    /// 只回答"声明与生成是否一致"，不回答"这个参数该不该传"。
    fn check_argument(
        &self,
        variant: &Variant,
        prim: Prim,
        line: usize,
        key: &str,
        value: &str,
    ) -> diagnostic::Result<()> {
        // 闭集之外的命名参数是片段级提示，不构成能力声明。
        let Some(attr_key) = AttrKey::parse(key) else {
            return Ok(());
        };
        // 只收这一条原语声明的属性：`needs.{prim}.{key}` 是该调用点的全部签名。
        let declared: Vec<&Attr> = variant
            .needs(self)
            .filter(|need| need.prim == prim)
            .flat_map(|need| need.attrs.iter())
            .filter(|attr| attr.key() == attr_key)
            .collect();
        let at = format!("{}.k:{line}", self.name);
        if declared.is_empty() {
            return Err(field(
                &at,
                format!(
                    "变体 `{}` 给 `@{prim}` 传了能力参数 `{key}: {value}`，但该变体的 `needs.{prim}` 没有声明 `{key}`：协商不会检查它",
                    variant.name
                ),
            ));
        }
        if let Some(attr) = declared.iter().find(|attr| !attr.contains(value)) {
            return Err(field(
                &at,
                format!(
                    "变体 `{}` 给 `@{prim}` 传 `{key}: {value}`，但 `needs.{prim}.{key}` 只允许 [{}]：设备被按声明的值验证，代码却按模板的值生成",
                    variant.name,
                    attr.values().join(" | ")
                ),
            ));
        }
        Ok(())
    }
}

/// tile 候选数：轴取值的乘积，与 [`expand`] 同一口径但不物化组合。
///
/// # Errors
///
/// 乘积超过 [`MAX_TILES`] 时返回说明（与 [`expand`] 同一个判据与消息）。
pub fn tiles(op: &OpDecl) -> diagnostic::Result<usize> {
    let mut count = 1usize;
    for axis in &op.axes {
        count = count.saturating_mul(axis.values.len());
        if count > MAX_TILES {
            return Err(too_many_tiles(op, axis, count));
        }
    }
    Ok(count)
}

/// 轴展开：笛卡尔积，超过 [`MAX_TILES`] 即拒绝并说明该剪哪个轴。
pub fn expand(op: &OpDecl) -> diagnostic::Result<Vec<TileCombo>> {
    let mut combos = vec![TileCombo { values: Vec::new() }];
    for axis in &op.axes {
        let mut next = Vec::with_capacity(combos.len() * axis.values.len());
        for combo in &combos {
            for value in &axis.values {
                let mut values = combo.values.clone();
                values.push((axis.name.clone(), *value));
                next.push(TileCombo { values });
            }
        }
        if next.len() > MAX_TILES {
            return Err(too_many_tiles(op, axis, next.len()));
        }
        combos = next;
    }
    Ok(combos)
}

/// 候选超限的诊断：点名该剪哪个轴，以及它至少要让出几个取值。
fn too_many_tiles(op: &OpDecl, axis: &Axis, count: usize) -> Error {
    field(
        &format!("{}.axes", op.name),
        format!(
            "展开后有 {count} 个 tile 候选，超过上限 {MAX_TILES}；最少可剪掉轴 `{}` 的{}个取值",
            axis.name,
            axis.values.len().saturating_sub(1)
        ),
    )
}

/// 从 `[needs]`/`[extra_needs]` 的一个条目解析能力签名。
fn parse_need(section: &Table, key: &str, path: &str) -> Result<Need, Error> {
    let prim = prim_key(key, path)?;
    let detail = need_table(section, key, path)?;
    // 不拷这张表：解析只读它，为一棵内联表克隆一次等于每条需求多付一次分配。
    detail
        .entries()
        .iter()
        .try_fold(Need::of(prim), |need, (name, value)| {
            let value_path = format!("{path}.{name}");
            match dimension(&need, name, &value_path, value)? {
                Some(need) => Ok(need),
                None => collection(need, prim, &value_path, name, value),
            }
        })
}

/// 原语名：必须在闭集内。
fn prim_key(key: &str, path: &str) -> Result<Prim, Error> {
    Prim::parse(key).ok_or_else(|| {
        let allowed: Vec<&str> = Prim::ALL.iter().map(|prim| prim.as_str()).collect();
        field(path, format!("闭集：{}", allowed.join(" | ")))
    })
}

/// 一条需求的取值：必须是内联表。
fn need_table<'a>(section: &'a Table, key: &str, path: &str) -> Result<&'a Table, Error> {
    let Some(value) = section.get(key) else {
        return Err(field(path, "缺失"));
    };
    let Value::Table(detail) = value else {
        return Err(field(path, "必须是内联表"));
    };
    Ok(detail)
}

/// 维度键：`m`/`bytes`/`extent` 共用跨度写法，`n`/`k` 各一条轴；都不是返回 `None`。
fn dimension(need: &Need, name: &str, path: &str, value: &Value) -> Result<Option<Need>, Error> {
    let attach: fn(Need, Span) -> Need = match name {
        "m" | "bytes" | "extent" => Need::with_extent,
        "n" => Need::with_n,
        "k" => Need::with_k,
        _ => return Ok(None),
    };
    Ok(Some(attach(need.clone(), parse_span(path, value)?)))
}

/// 非跨度键：`dt`/`space` 各一条集合，其余是能力属性。
fn collection(
    need: Need,
    prim: Prim,
    path: &str,
    name: &str,
    value: &Value,
) -> Result<Need, Error> {
    match name {
        "dt" => Ok(need.with_dt(parse_dt(path, value)?)),
        "space" => Ok(need.with_space(parse_space(path, value)?)),
        _ => attribute(need, prim, path, name, value),
    }
}

/// 能力属性：键必须属于该原语的签名，取值是字符串列表。
fn attribute(need: Need, prim: Prim, path: &str, name: &str, value: &Value) -> Result<Need, Error> {
    let attr_key =
        AttrKey::parse(name).ok_or_else(|| field(path, "既不是维度也不是闭集内的属性键"))?;
    // 属性键属于原语自己的签名：`sat` 不属于 `load`，`lanes` 不属于 `mma`。协商只比对需求里
    // 出现的键，所以一个与该原语无关的键既不会被检查、也不会被片段消费——它只会让声明看起来
    // 更具体。装载期就拒收。
    if !prim.accepts(attr_key) {
        let mut allowed: Vec<&str> = prim.attr_keys().iter().map(|key| key.as_str()).collect();
        allowed.push("exec");
        return Err(field(
            path,
            format!(
                "`{prim}` 的能力签名里没有 `{attr_key}`（该原语的属性键：{}）",
                allowed.join(" | ")
            ),
        ));
    }
    Ok(need.with_attr(Attr::new(attr_key, string_list(path, value)?)))
}

fn parse_span(path: &str, value: &Value) -> Result<Span, Error> {
    let Value::Str(text) = value else {
        return Err(field(path, "必须是字符串（`16`、`16..64`、`<=512`）"));
    };
    Span::parse(text).map_err(|error| field(path, error.message()))
}

fn parse_dt(path: &str, value: &Value) -> Result<DTypeSet, Error> {
    let mut set = DTypeSet::empty();
    for name in string_list(path, value)? {
        let dt = DType::parse_name(&name).ok_or_else(|| {
            let allowed: Vec<&str> = DType::ALL.iter().map(|dt| dt.name()).collect();
            field(
                path,
                format!("`{name}` 不在闭集内（{}）", allowed.join(" | ")),
            )
        })?;
        set.insert(dt);
    }
    Ok(set)
}

fn parse_space(path: &str, value: &Value) -> Result<SpaceSet, Error> {
    let mut set = SpaceSet::empty();
    for name in string_list(path, value)? {
        let space = Space::parse(&name).ok_or_else(|| {
            let allowed: Vec<&str> = Space::ALL.iter().map(|space| space.as_str()).collect();
            field(
                path,
                format!("`{name}` 不在闭集内（{}）", allowed.join(" | ")),
            )
        })?;
        set.insert(space);
    }
    Ok(set)
}

/// 一个键的字符串列表：允许 `["a", "b"]`、`"a"` 与标量写法（标量按十进制渲染）。
fn string_list(path: &str, value: &Value) -> Result<Vec<String>, Error> {
    match value {
        Value::Array(items) => items.iter().map(|item| as_string(path, item)).collect(),
        scalar => scalar_list(path, scalar),
    }
}

/// 非数组写法：字符串原样；整数/浮点/布尔按十进制渲染；表不能当列表。
fn scalar_list(path: &str, value: &Value) -> Result<Vec<String>, Error> {
    match value {
        Value::Str(text) => Ok(vec![text.clone()]),
        Value::Int(number) => Ok(vec![number.to_string()]),
        Value::Float(number) => Ok(vec![number.to_string()]),
        Value::Bool(flag) => Ok(vec![flag.to_string()]),
        _ => Err(field(path, "不能是表")),
    }
}

/// 数组里的一个元素：只能是字符串。
fn as_string(path: &str, item: &Value) -> Result<String, Error> {
    match item {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(field(path, "数组里只能是字符串")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::data::Value as TValue;

    fn builtin() -> OpDecl {
        // 按名字取 gemv_f32：注册表按名字排序，新增算子后不应再假设索引 0 是哪条。
        let op = crate::builtin_named("gemv_f32").expect("gemv_f32 内置");
        OpDecl::parse(op.declaration(), op.template()).expect("内置声明必须自洽")
    }

    /// 计数与展开必须同一个口径：装载期只数不物化，调用方仍然可以拿到组合。
    #[test]
    fn the_tile_count_matches_the_expanded_combinations() {
        let op = builtin();
        assert_eq!(
            tiles(&op).expect("未超限"),
            expand(&op).expect("可展开").len()
        );
    }

    #[test]
    fn a_tile_overflow_names_the_axis_to_prune() {
        let op = crate::builtin_named("gemv_f32").expect("gemv_f32");
        let mut broken = op.declaration().clone();
        broken.set(
            "axes.GMAX",
            TValue::array(vec![
                TValue::int(1),
                TValue::int(2),
                TValue::int(4),
                TValue::int(8),
                TValue::int(16),
                TValue::int(32),
                TValue::int(64),
                TValue::int(128),
                TValue::int(256),
            ]),
        );
        let error = OpDecl::parse(&broken, op.template()).expect_err("9 个候选超过上限 8");
        assert_eq!(error.code(), Code::Invalid);
        let text = error.to_string();
        assert!(text.contains("GMAX"), "{text}");
        assert!(text.contains("超过上限 8"), "{text}");
        assert!(text.contains("9 个 tile 候选"), "{text}");
    }

    /// 属性键属于原语自己的签名：`sat` 是 `cvt` 的参数，写在 `store` 上既不会被协商检查，
    /// 也不会被片段消费。判据必须点名原语、键与该原语真正允许的参数。
    #[test]
    fn an_attr_key_must_belong_to_its_primitive() {
        let op = crate::builtin_named("gemv_f32").expect("gemv_f32");
        let mut broken = op.declaration().clone();
        broken.set("needs.store.sat", TValue::array(vec![TValue::text("off")]));
        let store = broken
            .table("needs")
            .and_then(|needs| needs.table("store"))
            .expect("store 需求");
        assert!(store.contains("sat"), "前提：测试要改的是 store 的签名");
        let error = OpDecl::parse(&broken, op.template()).expect_err("`store.sat` 不属于签名");
        let text = error.to_string();
        assert!(text.contains("`store`"), "{text}");
        assert!(text.contains("`sat`"), "{text}");
        assert!(text.contains("src | dst | exec"), "{text}");
    }

    /// 降级链的单调性判据接受借用名：装载期不为它拷一份 `String`。
    #[test]
    fn the_monotonic_chain_is_checked_on_borrowed_names() {
        let chain = [("first", 0.5), ("second", 0.5)];
        assert!(sch::check_monotonic(&chain).is_ok());
        let reversed = [("first", 0.4), ("second", 0.9)];
        assert!(sch::check_monotonic(&reversed).is_err());
    }
}
