//! 声明层：算子声明的类型、轴展开与降级链解析（cutile README §4、§10.2 的 ①）。
//!
//! 一份文件装三件事，因为它们是同一条链上的三节：类型（[`OpDecl`] / [`Variant`] / [`Fusion`]）
//! 是声明的形状，[`expand`] 把轴展开成 tile 候选，`variants` 保证降级有序。
//! 字段级解析在同文件的 `ingest` 模块（表 → 类型；本文件只留类型与代数）。

use crate::atom::Need;
use crate::sch::Schedule;
use crate::sch::layout::LayoutSpec;
use crate::sig::CType;
use core::fmt;
use diagnostic::{Code, Error};

/// 变体数上限（cutile README §4.1）。
pub const MAX_VARIANTS: usize = 8;
/// 轴展开后的 tile 候选上限（cutile README §4.1）。
pub const MAX_TILES: usize = 8;

crate::closed_enum! {
    /// 骨架闭集（cutile README 附录 B）：决定用哪个模板基座。
    Skeleton {
        // 逐元素。
        Elementwise => "elementwise",
        // 归约。
        Reduce => "reduce",
        // 矩阵乘。
        Gemm => "gemm",
        // 扫描。
        Scan => "scan",
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

/// Gemm 骨架的网格映射：声明哪些串行外层循环按网格维拆线程（N 维、每维一个 `线程:标量` 对）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    /// 网格维（按序）：每项 `(线程名, 边界标量名)`。线程名 = 模板循环变量名，网格大小 = 标量样本值。
    pub dims: Vec<(String, String)>,
}

/// 分块并行声明：把长轴切成 `size` 元素一块，网格 = `ceil(N / size)`，每线程处理一块。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// 块长（元素/线程）。
    pub size: usize,
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
    /// Gemm 骨架的网格映射（仅 Gemm 算子声明；其余骨架不需要）。
    pub grid: Option<Grid>,
    /// 分块并行的块长（reduce/scan 骨架；`None` 表示串行兜底）。
    pub chunk: Option<Chunk>,
    /// 模板常量：`{{槽位}}` 的固定取值（非轴、非布局），按十进制渲染成字符串。
    pub consts: Vec<(String, String)>,
    /// 降级链，按 `order` 升序。
    pub variants: Vec<Variant>,
    /// 模板文本。
    pub template: String,
    /// 内核签名：从模板的入口函数解析，是 ABI 的唯一事实源。
    pub signature: crate::sig::Signature,
    /// 数据流阶段：谁在哪一阶段读/写哪个槽位；单内核算子就是一段（名字 = 算子名）。
    pub phases: Vec<Phase>,
}

/// 一个数据流阶段：把一次执行切成「读哪些槽位 → 写哪些槽位」的有序片段。`compose` 靠它判定
/// 生产者的写阶段与消费者的读阶段能不能对成一条缝。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase {
    /// 阶段名（缺省 = 算子名）。
    pub name: String,
    /// 本阶段读的槽位名（缓冲形参名）。
    pub reads: Vec<String>,
    /// 本阶段写的槽位名。
    pub writes: Vec<String>,
}

/// 一条融合缝：生产者的写槽位与消费者的读槽位同类型，就是"能拼"的那一处。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seam {
    /// 生产者的写槽位名。
    pub from: String,
    /// 消费者的读槽位名（融合时以临时缓冲桥接，名字不必相同）。
    pub to: String,
    /// 缝上的类型（两侧一致才合法；只可能是缓冲类型 F32 / U8）。
    pub ctype: crate::sig::CType,
}

/// 一次合法的生产者→消费者融合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fusion {
    /// 生产者算子名。
    pub producer: String,
    /// 消费者算子名。
    pub consumer: String,
    /// 全部融合缝（非空才有融合）。
    pub seams: Vec<Seam>,
}

impl Fusion {
    /// 本融合要求生产者产出的槽位，按 `from` 去重（保持声明顺序）。
    ///
    /// 同一输出可以桥接到消费者的多个读槽位（一条输出、两条缝），因此**缝数不等于输出数**：
    /// 对拍必须按这个分组逐条桥接，只桥一条就等于测了另一个程序。
    pub fn outputs(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for seam in &self.seams {
            if !out.contains(&seam.from.as_str()) {
                out.push(&seam.from);
            }
        }
        out
    }

    /// 某个生产者输出桥接到的全部消费者读槽位。
    pub fn targets<'a>(&'a self, from: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.seams
            .iter()
            .filter(move |seam| seam.from == from)
            .map(|seam| seam.to.as_str())
    }

    /// 是不是自融合：同一算子拼到自己前面是循环携带依赖，不是缝。
    pub fn is_self(&self) -> bool {
        self.producer == self.consumer
    }
}

impl OpDecl {
    /// 槽位的声明类型：缓冲形参才有；标量或未知名返回 `None`。
    pub fn slot_type(&self, name: &str) -> Option<crate::sig::CType> {
        self.signature
            .params()
            .iter()
            .find(|param| param.name == name && param.ctype.is_buf())
            .map(|param| param.ctype)
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

    /// 按名字取模板常量。
    pub fn const_value(&self, name: &str) -> Option<&str> {
        self.consts
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// 按名字取轴。
    pub fn axis(&self, name: &str) -> Option<&Axis> {
        self.axes.iter().find(|axis| axis.name == name)
    }
}

/// 槽位代数：判定"生产者 A 能不能拼到消费者 B 前面"（cutile README §9 的槽位代数）。
///
/// 合法当且仅当 A、B 是**不同**算子且存在至少一条**缝**——A 的某个写阶段写出一个槽位，B 的某个
/// 读阶段读入一个**同类型**的槽位（名字不必相同：融合时以临时缓冲桥接）。写写、写读都对不上、
/// 类型不一致、或 `A == B` 都返回 `None`（不是错误：融合是候选剪枝，不是输入不合法）。
///
/// 自融合被排除是代数层面的要求，不是偏保守：A 读的槽位要等 A 这一轮写完才有值，那是循环携带
/// 依赖，拼出来的"融合核"无法定义执行顺序；把它放进候选表还会白占一个预算名额。
pub fn compose(producer: &OpDecl, consumer: &OpDecl) -> Option<Fusion> {
    let seams = seams_between(producer, consumer);
    (!seams.is_empty()).then(|| Fusion {
        producer: producer.name.clone(),
        consumer: consumer.name.clone(),
        seams,
    })
}

/// 生产者每条写槽位 × 消费者每条读槽位里同类型的配对。自融合的窗口从 `ops[i]` 往前数半径，
/// 天生包含自己，所以这里第一件事就是把 `A == B` 摘掉。
fn seams_between(producer: &OpDecl, consumer: &OpDecl) -> Vec<Seam> {
    let mut seams = Vec::new();
    if producer.name == consumer.name {
        return seams;
    }
    for phase in &producer.phases {
        for from in &phase.writes {
            let Some(ctype) = producer.slot_type(from) else {
                continue;
            };
            record_seams(&mut seams, consumer, from, ctype);
        }
    }
    seams
}

/// 把一个生产者写槽位与消费者全部读槽位配对。
fn record_seams(seams: &mut Vec<Seam>, consumer: &OpDecl, from: &str, ctype: CType) {
    for phase in &consumer.phases {
        for to in &phase.reads {
            record_seam(seams, consumer, (from, to), ctype);
        }
    }
}

/// 登记一条缝：两侧同类型才成立，同一对 `(from, to)` 只登记一次。
fn record_seam(seams: &mut Vec<Seam>, consumer: &OpDecl, pair: (&str, &str), ctype: CType) {
    let (from, to) = pair;
    if consumer.slot_type(to) != Some(ctype) {
        return;
    }
    if seams.iter().any(|seam| seam.from == from && seam.to == to) {
        return;
    }
    seams.push(Seam {
        from: from.to_string(),
        to: to.to_string(),
        ctype,
    });
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
pub fn field(path: &str, detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("`{path}` 非法：{}", detail.into()))
}

// ── 轴展开 ─────────────────────────────────────────────────────────────

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

// ── 变体链 ─────────────────────────────────────────────────────────────

/// 变体链：逐个解析、按 `order` 排序，再判数量与链路。
pub(crate) fn variants(table: &diagnostic::data::Table) -> diagnostic::Result<Vec<Variant>> {
    let mut variants: Vec<Variant> = table
        .table_array("variants")
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(idx, entry)| variant(idx, entry))
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
    crate::sch::check_monotonic(&chain).map_err(|detail| field("variants", detail.message()))?;
    let Some(head) = variants.first() else {
        return Ok(());
    };
    crate::sch::check_masking(
        &head.name,
        crate::sch::Masking::of(head.mma_share, &head.schedule),
    )
    .map_err(|detail| field("variants", detail.message()))
}

/// 一个 `[[variants]]` 段：名字与优先级、调度、非 MMA 占比、增量需求。
fn variant(idx: usize, entry: &diagnostic::data::Table) -> diagnostic::Result<Variant> {
    let (name, order) = identity(idx, entry)?;
    let schedule = schedule(idx, entry)?;
    let mma_share = share(idx, entry)?;
    Ok(Variant {
        name,
        order,
        schedule,
        mma_share,
        extra_needs: extra_needs(idx, entry)?,
    })
}

/// 变体名与优先级：两个键都必填，负的 `order` 按 0 处理。
fn identity(idx: usize, entry: &diagnostic::data::Table) -> diagnostic::Result<(String, u32)> {
    let name = entry
        .string("name")
        .ok_or_else(|| field(&field_path(idx, "name"), "缺失"))?
        .to_string();
    let order = entry
        .int("order")
        .ok_or_else(|| field(&field_path(idx, "order"), "缺失"))?
        .max(0) as u32;
    Ok((name, order))
}

/// 调度：`schedule` 必须落在调度闭集内。
fn schedule(idx: usize, entry: &diagnostic::data::Table) -> diagnostic::Result<Schedule> {
    let path = field_path(idx, "schedule");
    let text = entry
        .string("schedule")
        .ok_or_else(|| field(&path, "缺失"))?;
    Schedule::parse(text).map_err(|detail| field(&path, detail.message()))
}

/// 非 MMA 占比：`0..=1` 之间。
fn share(idx: usize, entry: &diagnostic::data::Table) -> diagnostic::Result<f64> {
    let path = field_path(idx, "mma_share");
    let mma_share = entry
        .float("mma_share")
        .ok_or_else(|| field(&path, "缺失"))?;
    if (0.0..=1.0).contains(&mma_share) {
        return Ok(mma_share);
    }
    Err(field(&path, "必须在 0..=1 之间"))
}

/// `extra_needs`：变体自己追加的需求。
fn extra_needs(idx: usize, entry: &diagnostic::data::Table) -> diagnostic::Result<Vec<Need>> {
    let Some(section) = entry.table("extra_needs") else {
        return Ok(Vec::new());
    };
    let prefix = field_path(idx, "");
    section
        .entries()
        .iter()
        .map(|(key, _)| {
            crate::decl::ingest::parse_need(section, key, &format!("{prefix}.extra_needs.{key}"))
        })
        .collect()
}

/// `variants[i].{name}` 形式的字段路径；`name` 为空时只剩前缀。
fn field_path(idx: usize, name: &str) -> String {
    format!("variants[{idx}].{name}")
}

/// 语料摄入：声明表 → [`OpDecl`] 的字段级解析与模板侧校验，平铺在 `src/ingest.rs`
/// （模块路径是 `decl::ingest`，文件不套子目录）。
#[path = "ingest.rs"]
mod ingest;

#[cfg(test)]
#[path = "../tests/decl.rs"]
mod tests;
