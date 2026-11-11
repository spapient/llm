//! 声明层：`ops/<op>.toml` + `ops/<op>.k` 的解析与轴展开（DESIGN §4、§10.2 的 ①）。

use crate::atom::{Attr, AttrKey, DType, DTypeSet, Need, Prim, Space, SpaceSet, Span};
use crate::layout::LayoutSpec;
use crate::sch;
use crate::sch::Schedule;
use crate::toml::{self, Table, Value};
use core::fmt;

/// 变体数上限（DESIGN §4.1）。
pub const MAX_VARIANTS: usize = 8;
/// 轴展开后的 tile 候选上限（DESIGN §4.1）。
pub const MAX_TILES: usize = 8;

/// 骨架闭集（DESIGN 附录 B）：决定用哪个模板基座。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Skeleton {
    /// 闭集全量。
    pub const ALL: [Self; 5] = [
        Self::Elementwise,
        Self::Reduce,
        Self::Gemm,
        Self::Attention,
        Self::Scan,
    ];

    /// 声明层写法。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Reduce => "reduce",
            Self::Gemm => "gemm",
            Self::Attention => "attention",
            Self::Scan => "scan",
        }
    }

    /// 从名字解析。
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|item| item.as_str() == name)
    }
}

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
    /// 变体级增量需求（`naive` 不需要 `load_async`，就靠这里表达）。
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
        let parts: Vec<String> = self
            .values
            .iter()
            .map(|(axis, value)| format!("{axis}={value}"))
            .collect();
        parts.join(" ")
    }
}

impl fmt::Display for TileCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// 声明层错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpError {
    /// TOML 语法错误。
    Syntax(String),
    /// 字段非法。
    Field {
        /// 字段路径。
        path: String,
        /// 说明。
        detail: String,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(detail) => write!(f, "解析失败：{detail}"),
            Self::Field { path, detail } => write!(f, "`{path}` 非法：{detail}"),
        }
    }
}

impl std::error::Error for OpError {}

fn field(path: &str, detail: impl Into<String>) -> OpError {
    OpError::Field {
        path: path.to_string(),
        detail: detail.into(),
    }
}

impl OpDecl {
    /// 解析声明与模板。
    pub fn parse(declaration: &str, template: &str) -> Result<Self, OpError> {
        let table = toml::parse(declaration).map_err(|error| OpError::Syntax(error.to_string()))?;

        let op = table
            .table("op")
            .ok_or_else(|| field("op", "缺少 `[op]` 段"))?;
        let name = op
            .string("name")
            .ok_or_else(|| field("op.name", "缺失"))?
            .to_string();
        if name.is_empty() {
            return Err(field("op.name", "不能为空"));
        }
        let skeleton_name = op
            .string("skeleton")
            .ok_or_else(|| field("op.skeleton", "缺失"))?;
        let skeleton = Skeleton::parse(skeleton_name).ok_or_else(|| {
            let allowed: Vec<&str> = Skeleton::ALL.iter().map(|item| item.as_str()).collect();
            field("op.skeleton", format!("闭集：{}", allowed.join(" | ")))
        })?;
        let summary = op.string("summary").unwrap_or("-").to_string();

        let mut axes = Vec::new();
        if let Some(section) = table.table("axes") {
            for (key, value) in section.entries() {
                let values = match value {
                    Value::Array(items) => items
                        .iter()
                        .map(|item| match item {
                            Value::Int(number) if *number > 0 => Ok(*number as u32),
                            _ => Err(field(&format!("axes.{key}"), "每个取值必须是正整数")),
                        })
                        .collect::<Result<Vec<u32>, _>>()?,
                    _ => return Err(field(&format!("axes.{key}"), "必须是数组")),
                };
                if values.is_empty() {
                    return Err(field(&format!("axes.{key}"), "数组不能为空"));
                }
                axes.push(Axis {
                    name: key.clone(),
                    values,
                });
            }
        }
        if axes.is_empty() {
            return Err(field("axes", "至少需要一个轴"));
        }

        let mut needs = Vec::new();
        if let Some(section) = table.table("needs") {
            for (key, _) in section.entries() {
                needs.push(parse_need(section, key, &format!("needs.{key}"))?);
            }
        }

        let mut layouts = Vec::new();
        if let Some(section) = table.table("layouts") {
            for (key, value) in section.entries() {
                let Value::Str(text) = value else {
                    return Err(field(&format!("layouts.{key}"), "必须是字符串"));
                };
                let spec = LayoutSpec::parse(text)
                    .map_err(|detail| field(&format!("layouts.{key}"), detail))?;
                layouts.push((key.clone(), spec));
            }
        }

        let mut variants = Vec::new();
        for (index, entry) in table
            .table_array("variants")
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let path = |name: &str| format!("variants[{index}].{name}");
            let name = entry
                .string("name")
                .ok_or_else(|| field(&path("name"), "缺失"))?
                .to_string();
            let order = entry
                .int("order")
                .ok_or_else(|| field(&path("order"), "缺失"))?
                .max(0) as u32;
            let schedule_text = entry
                .string("schedule")
                .ok_or_else(|| field(&path("schedule"), "缺失"))?;
            let schedule = Schedule::parse(schedule_text)
                .map_err(|detail| field(&path("schedule"), detail))?;
            let mma_share = entry
                .float("mma_share")
                .ok_or_else(|| field(&path("mma_share"), "缺失"))?;
            if !(0.0..=1.0).contains(&mma_share) {
                return Err(field(&path("mma_share"), "必须在 0..=1 之间"));
            }
            let mut extra_needs = Vec::new();
            if let Some(section) = entry.table("extra_needs") {
                for (key, _) in section.entries() {
                    extra_needs.push(parse_need(
                        section,
                        key,
                        &format!("{}.extra_needs.{key}", path("")),
                    )?);
                }
            }
            variants.push(Variant {
                name,
                order,
                schedule,
                mma_share,
                extra_needs,
            });
        }
        if variants.is_empty() {
            return Err(field("variants", "至少需要一个变体"));
        }
        if variants.len() > MAX_VARIANTS {
            return Err(field(
                "variants",
                format!("{} 个变体超过上限 {MAX_VARIANTS}", variants.len()),
            ));
        }
        variants.sort_by_key(|variant| variant.order);
        for pair in variants.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(field("variants", format!("变体名 `{}` 重复", pair[0].name)));
            }
        }
        let chain: Vec<(String, f64)> = variants
            .iter()
            .map(|variant| (variant.name.clone(), variant.mma_share))
            .collect();
        sch::check_monotonic(&chain).map_err(|detail| field("variants", detail))?;

        let op_decl = Self {
            name,
            skeleton,
            summary,
            axes,
            needs,
            layouts,
            variants,
            template: template.to_string(),
        };
        expand(&op_decl)?;
        op_decl.check_slots()?;
        Ok(op_decl)
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
    fn check_slots(&self) -> Result<(), OpError> {
        for (line, slot) in crate::lower::substitute::slots(&self.template) {
            if slot == "op" {
                continue;
            }
            if self.axis(&slot).is_none() && self.layout(&slot).is_none() {
                return Err(OpError::Field {
                    path: format!("{}.k:{line}", self.name),
                    detail: format!("槽位 `{{{{{slot}}}}}` 在 [axes] 和 [layouts] 里都没有定义"),
                });
            }
        }
        Ok(())
    }
}

/// 轴展开：笛卡尔积，超过 [`MAX_TILES`] 即拒绝并说明该剪哪个轴。
pub fn expand(op: &OpDecl) -> Result<Vec<TileCombo>, OpError> {
    let mut combos = vec![TileCombo { values: Vec::new() }];
    for axis in &op.axes {
        let mut next = Vec::new();
        for combo in &combos {
            for value in &axis.values {
                let mut values = combo.values.clone();
                values.push((axis.name.clone(), *value));
                next.push(TileCombo { values });
            }
        }
        if next.len() > MAX_TILES {
            return Err(OpError::Field {
                path: format!("{}.axes", op.name),
                detail: format!(
                    "展开后有 {} 个 tile 候选，超过上限 {MAX_TILES}；最少可剪掉轴 `{}` 的{}个取值",
                    next.len(),
                    axis.name,
                    axis.values.len().saturating_sub(1)
                ),
            });
        }
        combos = next;
    }
    Ok(combos)
}

/// 从 `[needs]`/`[extra_needs]` 的一个条目解析能力签名。
fn parse_need(section: &Table, key: &str, path: &str) -> Result<Need, OpError> {
    let prim = Prim::parse(key).ok_or_else(|| {
        let allowed: Vec<&str> = Prim::ALL.iter().map(|prim| prim.as_str()).collect();
        field(path, format!("闭集：{}", allowed.join(" | ")))
    })?;
    let Value::Table(detail) = section.get(key).expect("调用方已确认键存在").clone()
    else {
        return Err(field(path, "必须是内联表"));
    };

    let mut need = Need::of(prim);
    for (name, value) in detail.entries() {
        let value_path = format!("{path}.{name}");
        match name.as_str() {
            "m" | "bytes" | "extent" => {
                need = need.with_extent(parse_span(&value_path, value)?);
            }
            "n" => need = need.with_n(parse_span(&value_path, value)?),
            "k" => need = need.with_k(parse_span(&value_path, value)?),
            "dt" => need = need.with_dt(parse_dt(&value_path, value)?),
            "space" => need = need.with_space(parse_space(&value_path, value)?),
            other => {
                let attr_key = AttrKey::parse(other)
                    .ok_or_else(|| field(&value_path, "既不是维度也不是闭集内的属性键"))?;
                need = need.with_attr(Attr::new(attr_key, string_list(&value_path, value)?));
            }
        }
    }
    Ok(need)
}

fn parse_span(path: &str, value: &Value) -> Result<Span, OpError> {
    let Value::Str(text) = value else {
        return Err(field(path, "必须是字符串（`16`、`16..64`、`<=512`）"));
    };
    Span::parse(text).map_err(|detail| field(path, detail))
}

fn parse_dt(path: &str, value: &Value) -> Result<DTypeSet, OpError> {
    let mut set = DTypeSet::empty();
    for name in string_list(path, value)? {
        let dt = DType::parse(&name).ok_or_else(|| {
            let allowed: Vec<&str> = DType::ALL.iter().map(|dt| dt.as_str()).collect();
            field(
                path,
                format!("`{name}` 不在闭集内（{}）", allowed.join(" | ")),
            )
        })?;
        set.insert(dt);
    }
    Ok(set)
}

fn parse_space(path: &str, value: &Value) -> Result<SpaceSet, OpError> {
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

fn string_list(path: &str, value: &Value) -> Result<Vec<String>, OpError> {
    match value {
        Value::Str(text) => Ok(vec![text.clone()]),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::Str(text) => Ok(text.clone()),
                _ => Err(field(path, "数组里只能是字符串")),
            })
            .collect(),
        Value::Int(number) => Ok(vec![number.to_string()]),
        Value::Float(number) => Ok(vec![number.to_string()]),
        Value::Bool(flag) => Ok(vec![flag.to_string()]),
        Value::Table(_) => Err(field(path, "不能是表")),
    }
}
