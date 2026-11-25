//! 声明层：算子声明的类型、轴展开与降级链解析（cutile README §4、§10.2 的 ①）。
//!
//! 强类型视图（[`OpView`]）一次性 owned 化为声明类型；[`expand`] 展开轴候选，
//! 变体链与模板侧校验共用同一份代数，不再保留动态表解析旁路。

use crate::atom::{Attr, AttrKey, DType, Need, Prim, Set, Space, Span};
use crate::sch::Schedule;
use crate::sch::layout::LayoutSpec;
use crate::sig::CType;
use crate::tpl::{Arg, Template};
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
    /// 从编译期强类型视图构造；模板正文仍由注册表按算子名配对。
    pub fn from_view(view: &OpView, template: &str) -> diagnostic::Result<Self> {
        let declaration = view.into_decl(template)?;
        declaration.verified(template)?;
        Ok(declaration)
    }

    /// 无归约 / 点积、单标量的逐元素算子可走低维逐点网格快路径。
    pub fn pointwise(&self) -> bool {
        self.skeleton == Skeleton::Elementwise
            && !self
                .needs
                .iter()
                .any(|need| matches!(need.prim, Prim::Reduce | Prim::Dot))
            && self
                .signature
                .params()
                .iter()
                .filter(|param| !param.ctype.is_buf())
                .count()
                == 1
            && self.same_element_type()
    }

    fn same_element_type(&self) -> bool {
        let mut types = self
            .signature
            .params()
            .iter()
            .filter(|param| param.ctype.is_buf())
            .map(|param| param.ctype);
        let Some(first) = types.next() else {
            return false;
        };
        types.all(|ctype| ctype == first)
    }

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

    /// 模板侧校验：槽位、能力需求、调用点参数与数据流阶段必须一致。
    fn verified(&self, template: &str) -> diagnostic::Result<()> {
        let segments = Template::parse(template);
        tiles(self)?;
        self.check_phases()?;
        self.check_slots(&segments)?;
        self.check_needs(&segments)?;
        self.check_call_arguments(&segments)
    }

    fn check_phases(&self) -> diagnostic::Result<()> {
        for param in self
            .signature
            .params()
            .iter()
            .filter(|param| param.ctype.is_buf())
        {
            let covered = self.phases.iter().any(|phase| {
                phase.reads.iter().any(|slot| slot == &param.name)
                    || phase.writes.iter().any(|slot| slot == &param.name)
            });
            if !covered {
                return Err(field(
                    &format!("{}.phases", self.name),
                    format!("缓冲 `{}` 没有出现在任何阶段里", param.name),
                ));
            }
        }
        Ok(())
    }

    fn check_slots(&self, template: &Template<'_>) -> diagnostic::Result<()> {
        for (line, slot) in template.slots() {
            if slot == "op" || self.slot_known(slot) {
                continue;
            }
            return Err(field(
                &format!("{}.k:{line}", self.name),
                format!("槽位 `{{{{{slot}}}}}` 在 [axes]、[layouts] 和 [consts] 里都没有定义"),
            ));
        }
        Ok(())
    }

    fn slot_known(&self, slot: &str) -> bool {
        self.axis(slot).is_some() || self.layout(slot).is_some() || self.const_value(slot).is_some()
    }

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
    fn check_argument(
        &self,
        variant: &Variant,
        prim: Prim,
        line: usize,
        key: &str,
        value: &str,
    ) -> diagnostic::Result<()> {
        let Some(attr_key) = AttrKey::parse(key) else {
            return Ok(());
        };
        let needs = variant.needs(self).filter(|need| need.prim == prim);
        let declared: Vec<&Attr> = needs
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
                    "变体 `{variant}` 给 `@{prim}` 传 `{key}: {value}`，但 `needs.{prim}.{key}` 只允许 [{}]：设备被按声明的值验证，代码却按模板的值生成",
                    attr.values().join(" | "),
                    variant = variant.name
                ),
            ));
        }
        Ok(())
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

/// 强类型变体一次性 owned 化，按优先级排序后执行完整链路校验。
fn variants(views: &[VariantView]) -> diagnostic::Result<Vec<Variant>> {
    let mut variants: Vec<Variant> = views.iter().map(|view| view.into_owned()).collect();
    check_count(&variants)?;
    variants.sort_by_key(|variant| variant.order);
    check_chain(&variants)?;
    Ok(variants)
}

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

/// 名字全链路唯一、占比有限且单调，首选变体的掩码必须可落地。
fn check_chain(variants: &[Variant]) -> diagnostic::Result<()> {
    for (idx, variant) in variants.iter().enumerate() {
        check_variant(&variants[..idx], variant, idx)?;
    }
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

/// 单条变体的局部判据；下标同时用于全链路重名检测与精确错误定位。
fn check_variant(earlier: &[Variant], variant: &Variant, idx: usize) -> diagnostic::Result<()> {
    if earlier.iter().any(|have| have.name == variant.name) {
        return Err(field("variants", format!("变体名 `{}` 重复", variant.name)));
    }
    if !(0.0..=1.0).contains(&variant.mma_share) {
        return Err(field(
            &format!("variants[{idx}].mma_share"),
            "必须在 0..=1 之间",
        ));
    }
    Ok(())
}

// ---- 编译期强类型只读视图（方案 3） -----------------------------------------
//
// 宏在编译期把算子配置填充成这些 `const` 结构；字段只用 `&'static str`、`&'static [..]`、
// 闭集枚举、标量与 [`Span`] 等可在编译期字面量构造的类型。运行期再一次性 owned 化成上面的
// 归属类型；视图本身不分配 `String` / `Vec`。位集（数据类型 / 存储空间）在视图层用元素切片
// 表示，owned 化时重建为 [`Set`]。

/// 轴视图：名字 + 正整数候选值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisView {
    /// 轴名（也是模板槽位名）。
    pub name: &'static str,
    /// 候选取值。
    pub values: &'static [u32],
}

impl AxisView {
    /// owned 化成 [`Axis`]。
    pub fn into_owned(self) -> Axis {
        Axis {
            name: self.name.to_string(),
            values: self.values.to_vec(),
        }
    }
}

/// 属性取值视图：一个属性键上的一组取值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttrView {
    /// 属性键闭集。
    pub key: AttrKey,
    /// 取值列表。
    pub values: &'static [&'static str],
}

impl AttrView {
    /// owned 化成 [`Attr`]。
    pub fn into_owned(self) -> Attr {
        Attr::new(self.key, self.values.iter().copied())
    }
}

/// 需求视图：模板 / 算子声明对某原语的要求。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedView {
    /// 原语闭集。
    pub prim: Prim,
    /// 主维度区间。
    pub extent: Span,
    /// `mma` 的 `n`。
    pub n: Span,
    /// `mma` 的 `k`。
    pub k: Span,
    /// 数据类型取值（视图层用元素切片）。
    pub dt: &'static [DType],
    /// 存储空间取值（视图层用元素切片）。
    pub space: &'static [Space],
    /// 枚举型属性。
    pub attrs: &'static [AttrView],
}

impl NeedView {
    /// owned 化成 [`Need`]；位集由元素切片重建。
    pub fn into_owned(self) -> Need {
        Need {
            prim: self.prim,
            extent: self.extent,
            n: self.n,
            k: self.k,
            dt: Set::of(self.dt.iter().copied()),
            space: Set::of(self.space.iter().copied()),
            attrs: self.attrs.iter().map(|a| a.into_owned()).collect(),
        }
    }
}

/// 网格映射视图：`Gemm` 骨架按输出行拆线程，一维或多维。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridView {
    /// 网格维（按序）：每项 `(线程名, 边界标量名)`。
    pub dims: &'static [(&'static str, &'static str)],
}

impl GridView {
    /// owned 化成 [`Grid`]。
    pub fn into_owned(self) -> Grid {
        Grid {
            dims: self
                .dims
                .iter()
                .map(|(thread, scalar)| (thread.to_string(), scalar.to_string()))
                .collect(),
        }
    }
}

/// 分块并行视图：把长轴切成固定大小的块。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkView {
    /// 块长（元素 / 线程）。
    pub size: usize,
}

impl ChunkView {
    /// owned 化成 [`Chunk`]。
    pub fn into_owned(self) -> Chunk {
        Chunk { size: self.size }
    }
}

/// 降级变体视图：降级链上的一个变体。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariantView {
    /// 变体名。
    pub name: &'static str,
    /// 优先级，小者优先。
    pub order: u32,
    /// 调度模板。
    pub schedule: Schedule,
    /// 非 MMA 占比上界。
    pub mma_share: f64,
    /// 变体级增量需求（只能加强公共需求）。
    pub extra_needs: &'static [NeedView],
}

impl VariantView {
    /// owned 化成 [`Variant`]。
    pub fn into_owned(self) -> Variant {
        Variant {
            name: self.name.to_string(),
            order: self.order,
            schedule: self.schedule,
            mma_share: self.mma_share,
            extra_needs: self.extra_needs.iter().map(|n| n.into_owned()).collect(),
        }
    }
}

/// 数据流阶段视图：一次执行切成「读哪些槽位 → 写哪些槽位」的有序片段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseView {
    /// 阶段名（缺省 = 算子名）。
    pub name: &'static str,
    /// 本阶段读的槽位名。
    pub reads: &'static [&'static str],
    /// 本阶段写的槽位名。
    pub writes: &'static [&'static str],
}

impl PhaseView {
    /// owned 化成 [`Phase`]。
    pub fn into_owned(self) -> Phase {
        Phase {
            name: self.name.to_string(),
            reads: self.reads.iter().map(ToString::to_string).collect(),
            writes: self.writes.iter().map(ToString::to_string).collect(),
        }
    }
}

/// 算子声明视图：编译期可 const 构造的算子静态形状。
///
/// 模板正文由 `include_str!` 挂入；内核签名 [`crate::sig::Signature`] 与缺省阶段在
/// owned 化时从模板派生，视图层不重复存储。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpView {
    /// 算子名。
    pub name: &'static str,
    /// 骨架闭集。
    pub skeleton: Skeleton,
    /// 一句话说明。
    pub summary: &'static str,
    /// 轴。
    pub axes: &'static [AxisView],
    /// 算子级公共需求。
    pub needs: &'static [NeedView],
    /// 布局契约：槽位 → 布局闭集。
    pub layouts: &'static [(&'static str, LayoutSpec)],
    /// 网格映射（仅 Gemm）。
    pub grid: Option<GridView>,
    /// 分块并行块长（reduce / scan）。
    pub chunk: Option<ChunkView>,
    /// 模板常量：`{{槽位}}` 的固定取值。
    pub consts: &'static [(&'static str, &'static str)],
    /// 降级链，按 `order` 升序。
    pub variants: &'static [VariantView],
    /// 显式数据流阶段；为空时 owned 化按签名派生单阶段。
    pub phases: &'static [PhaseView],
}

impl OpView {
    /// owned 化成 [`OpDecl`]：模板正文由调用方按名从 [`crate::catalogue::TEMPLATES`] 配对传入，
    /// 内核签名从模板解析，阶段缺省时按签名派生单阶段。
    ///
    /// # Errors
    ///
    /// 模板入口 / 形参不在闭集，或显式阶段引用了签名之外的槽位时返回带字段路径的说明。
    pub fn into_decl(self, template: &str) -> diagnostic::Result<OpDecl> {
        let signature = crate::sig::Signature::parse(self.name, template)?;
        let phases = declare_phases(self.name, self.phases, &signature)?;
        Ok(OpDecl {
            name: self.name.to_string(),
            skeleton: self.skeleton,
            summary: self.summary.to_string(),
            axes: self.axes.iter().map(|a| a.into_owned()).collect(),
            needs: self.needs.iter().map(|n| n.into_owned()).collect(),
            layouts: self
                .layouts
                .iter()
                .map(|(slot, spec)| (slot.to_string(), *spec))
                .collect(),
            grid: self.grid.map(GridView::into_owned),
            chunk: self.chunk.map(ChunkView::into_owned),
            consts: self
                .consts
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
            variants: variants(self.variants)?,
            template: template.to_string(),
            signature,
            phases,
        })
    }
}

/// 阶段：空视图按签名派生单阶段；非空逐条转 owned 并校验槽位落在签名形参内。
fn declare_phases(
    op: &str,
    views: &[PhaseView],
    signature: &crate::sig::Signature,
) -> diagnostic::Result<Vec<Phase>> {
    if views.is_empty() {
        return Ok(vec![Phase {
            name: op.to_string(),
            reads: signature.buf_names(false),
            writes: signature.buf_names(true),
        }]);
    }
    views
        .iter()
        .map(|phase| view_phase(op, phase, signature))
        .collect()
}

/// 一条显式阶段：校验读写槽位后转 owned。
fn view_phase(
    op: &str,
    phase: &PhaseView,
    signature: &crate::sig::Signature,
) -> diagnostic::Result<Phase> {
    let reads = check_phase_slots(op, "reads", phase.reads, signature)?;
    let writes = check_phase_slots(op, "writes", phase.writes, signature)?;
    Ok(Phase {
        name: phase.name.to_string(),
        reads,
        writes,
    })
}

/// 校验阶段读写槽位都落在签名缓冲形参内，返回 owned 名字表。
fn check_phase_slots(
    op: &str,
    key: &str,
    names: &[&'static str],
    signature: &crate::sig::Signature,
) -> diagnostic::Result<Vec<String>> {
    let unknown = names
        .iter()
        .find(|name| !signature.params().iter().any(|param| param.name == **name));
    match unknown {
        Some(name) => Err(field(
            &format!("{op}.phases.{key}"),
            format!("`{name}` 不是签名的缓冲形参"),
        )),
        None => Ok(names.iter().map(ToString::to_string).collect()),
    }
}

#[cfg(test)]
#[path = "../tests/decl.rs"]
mod tests;
