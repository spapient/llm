//! 声明解析与轴展开（cutile README §4、§10.2 的 ①）。
//!
//! 文本来自声明层（磁盘与内嵌两种来源，见 [`crate::Op`]），本模块只解释内容、不决定数据放在哪。
//! 解析的结果是 [`OpDecl`]：机制层（`cutile`）从这一步之后只消费它，不再看 TOML。

use crate::layout::LayoutSpec;
use crate::sch::Schedule;
use core::fmt;
use diagnostic::{Code, Error};
use prof::atom::Need;

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
    /// 扫描。
    Scan,
}

impl prof::atom::Closed for Skeleton {
    /// 闭集全量。
    const ALL: &'static [Self] = &[Self::Elementwise, Self::Reduce, Self::Gemm, Self::Scan];

    /// 声明层写法。
    fn as_str(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Reduce => "reduce",
            Self::Gemm => "gemm",
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

/// Gemm 骨架的网格映射：声明哪些串行外层循环按网格维拆线程（N 维、每维一个 `线程:标量` 对）。
///
/// 这是「配置化优先」的落点：串行模板的外层循环 `for (int t = 0; t < S; ++t)` 由机制层把循环头
/// 绑定到线程下标（`t = tid.x`、循环只跑一次），正文里的跨步（`i*K`、`p*N`）原样保留——因此既能
/// 覆盖 gemv 的一维行网格，也能覆盖 gemm 的二维网格（右操作数的规约维跨步 `p*N` 不被误压缩）。
/// 线程名必须与模板里的循环变量同名，标量名是该循环的边界；两者都由声明层显式给出，机制层
/// （cutile）只做循环头替换、不索引分析模板。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    /// 网格维（按序）：每项 `(线程名, 边界标量名)`。线程名 = 模板循环变量名，网格大小 = 标量样本值。
    pub dims: Vec<(String, String)>,
}

/// 分块并行声明：把长轴切成 `size` 元素一块，网格 = `ceil(N / size)`，每线程处理一块。
///
/// 与 [`Grid`]（循环头绑定到线程下标、正文跨步原样保留）的分工是「规约轴交给谁」：归约/扫描
/// 的跨元素依赖不能靠逐元素拆线程（每线程会各算一整遍），只能切成连续块、每线程串行处理自己
/// 那一块，再把块级部分结果交给主机折叠（归约）或做 carry 前缀（扫描）。块长由声明层显式给出，
/// 机制层只做 `start = g*size` / `len = min(size, N-start)` 的切块，不索引分析模板。
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
pub(crate) fn field(path: &str, detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("`{path}` 非法：{}", detail.into()))
}

mod build {
    use diagnostic::Error;
    use diagnostic::data::{Table, Value};

    use crate::layout::LayoutSpec;
    use crate::tpl::{Arg, Template};
    use prof::atom::{Attr, AttrKey, Closed, DTypeSet, Need, Prim, Set, SpaceSet, Span};
    use prof::prof::scalar_texts;

    use super::expand::tiles;
    use super::parse::{extras, head, sections};
    use super::{Axis, OpDecl, Skeleton, Variant, field};

    impl OpDecl {
        /// 从编译期生成的声明表解析声明，并解析同名模板。
        ///
        /// 按「键 → 取值语法」分工；模板侧的三条判据见 `Self::verified`。
        pub fn parse(table: &Table, template: &str) -> diagnostic::Result<Self> {
            let op_decl = Self::declared(table, template)?;
            op_decl.verified(template)?;
            Ok(op_decl)
        }

        /// 逐点网格判定：无归约/点积、单标量的 Elementwise 算子走 `crate::harness` 的逐点网格快路径——
        /// 线程下标就是元素下标、标量槽透传 `1`，网格必须覆盖整行。该判定是 lowering 分派的单一事实源，
        /// 机制层（cutile）与 cli 编排共用，避免两处各写一遍分派条件。
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
                    .filter(|param| !param.ctype.is_buffer())
                    .count()
                    == 1
                && self.same_element_type()
        }

        /// 全部缓冲同型：`y[n] = f(x[n])` 只有在此情形下才「线程下标就是元素下标」。
        ///
        /// 打包反量化是反例——`u8` 字节输入按 `cpb` 个 `f32` 输出展开，两侧元素计数不是 1:1，
        /// 按元素拆线程会把每个字节解 0 次、结果全零。判据取「同型」而非「无打包常量」：同型是
        /// 逐点成立的充分条件，打包族因此回落到整段串行（语义不变，只是不并行）。
        fn same_element_type(&self) -> bool {
            let mut types = self
                .signature
                .params()
                .iter()
                .filter(|param| param.ctype.is_buffer())
                .map(|param| param.ctype);
            let Some(first) = types.next() else {
                return false;
            };
            types.all(|ctype| ctype == first)
        }

        /// 声明本体：头、四个段与签名。模板侧校验不在这里（见 [`Self::verified`]）。
        fn declared(table: &Table, template: &str) -> diagnostic::Result<Self> {
            let (name, skeleton, summary) = head(table)?;
            // 返回顺序：轴、需求、布局、变体链。
            let (axes, needs, layouts, variants) = sections(table)?;
            let (consts, grid, chunk) = extras(table)?;
            let signature = crate::sig::Signature::parse(&name, template)?;
            Ok(Self {
                name,
                skeleton,
                summary,
                axes,
                needs,
                layouts,
                grid,
                chunk,
                consts,
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

        /// 模板里的每个 `{{槽位}}` 必须能被轴、布局、常量或 `op` 解析（提前拦住 `{{GMAXX}}` 这类笔误）。
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

        /// 槽位是否可解析：轴、布局或模板常量任一命中。
        fn slot_known(&self, slot: &str) -> bool {
            self.axis(slot).is_some()
                || self.layout(slot).is_some()
                || self.const_value(slot).is_some()
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

    /// 从 `[needs]`/`[extra_needs]` 的一个条目解析能力签名。
    pub(super) fn parse_need(section: &Table, key: &str, path: &str) -> Result<Need, Error> {
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
    fn dimension(
        need: &Need,
        name: &str,
        path: &str,
        value: &Value,
    ) -> Result<Option<Need>, Error> {
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
    fn attribute(
        need: Need,
        prim: Prim,
        path: &str,
        name: &str,
        value: &Value,
    ) -> Result<Need, Error> {
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

    /// 从字符串列表解析一个闭集位集：`["f32", "f16"]` 或单个标量都行；闭集外的名字报错并列出全量。
    ///
    /// `DType` / `Space` 都是 [`Closed`] 闭集，位集都是 [`Set`]，因此解析循环只有这一份；新增一种闭集
    /// 只需加一行薄包装，不重写"逐名解析 + 闭集外诊断"的结构。
    fn parse_closed_set<E: Closed, const WORDS: usize>(
        path: &str,
        value: &Value,
    ) -> Result<Set<E, WORDS>, Error> {
        let mut set = Set::<E, WORDS>::empty();
        for name in string_list(path, value)? {
            let item = E::parse(&name).ok_or_else(|| {
                let allowed: Vec<&str> = E::ALL.iter().map(|item| item.as_str()).collect();
                field(
                    path,
                    format!("`{name}` 不在闭集内（{}）", allowed.join(" | ")),
                )
            })?;
            set.insert(item);
        }
        Ok(set)
    }

    fn parse_dt(path: &str, value: &Value) -> Result<DTypeSet, Error> {
        parse_closed_set(path, value)
    }

    fn parse_space(path: &str, value: &Value) -> Result<SpaceSet, Error> {
        parse_closed_set(path, value)
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
        scalar_texts(value).ok_or_else(|| field(path, "不能是表"))
    }

    /// 数组里的一个元素：只能是字符串。
    fn as_string(path: &str, item: &Value) -> Result<String, Error> {
        match item {
            Value::Str(text) => Ok(text.clone()),
            _ => Err(field(path, "数组里只能是字符串")),
        }
    }
}

mod expand {
    //! 轴展开：把 [`OpDecl`] 的轴取值笛卡尔积物化成 [`TileCombo`] 候选，或只算候选数。
    //!
    //! 与声明解析（`decl.rs`）拆开：解析只产出 [`OpDecl`]，展开是机制层在拿到声明之后的第一刀。
    use diagnostic::Error;

    use super::{Axis, MAX_TILES, OpDecl, TileCombo, field};

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
}

mod parse {
    //! 声明表解析：`[op]`/`[axes]`/`[needs]`/`[layouts]`/`[grid]`/`[chunk]`/`[consts]` 各自一块。
    //!
    //! 每块一个函数，只做「取值 → 类型检查 → 结构」；跨块的语义校验（槽位、需求、参数）在
    //! [`super::build`]。字段级错误一律走 [`super::field`]，消息带 `variants[i].name` 式路径。
    use diagnostic::data::{Table, Value};

    use crate::layout::LayoutSpec;
    use prof::atom::Need;

    use super::build::parse_need;
    use super::variant::variants;
    use super::{Axis, Chunk, Grid, Skeleton, Variant, field};

    /// 算子名：`[op].name`，非空。
    pub(super) fn op_name(op: &Table) -> diagnostic::Result<String> {
        let name = op.string("name").ok_or_else(|| field("op.name", "缺失"))?;
        if name.is_empty() {
            return Err(field("op.name", "不能为空"));
        }
        Ok(name.to_string())
    }

    /// `[op]` 段读出的身份：名字、骨架、摘要。
    pub(super) fn head(table: &Table) -> diagnostic::Result<(String, Skeleton, String)> {
        let op = table
            .table("op")
            .ok_or_else(|| field("op", "缺少 `[op]` 段"))?;
        let (skeleton, summary) = op_head(op)?;
        Ok((op_name(op)?, skeleton, summary))
    }

    /// `[op]` 的其余两键：骨架必须落在闭集内，摘要缺省为空串占位的 `-`。
    pub(super) fn op_head(op: &Table) -> diagnostic::Result<(Skeleton, String)> {
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
    pub(super) type Sections = (
        Vec<Axis>,
        Vec<Need>,
        Vec<(String, LayoutSpec)>,
        Vec<Variant>,
    );

    pub(super) fn sections(table: &Table) -> diagnostic::Result<Sections> {
        Ok((
            axes(table)?,
            needs(table)?,
            layouts(table)?,
            variants(table)?,
        ))
    }

    /// `[axes]`：每个键一组正整数取值；至少要有一条轴。
    pub(super) fn axes(table: &Table) -> diagnostic::Result<Vec<Axis>> {
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
    pub(super) fn axis(key: &str, value: &Value) -> diagnostic::Result<Axis> {
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
    pub(super) fn positive(path: &str, item: &Value) -> diagnostic::Result<u32> {
        match item {
            Value::Int(number) if *number > 0 => Ok(*number as u32),
            _ => Err(field(path, "每个取值必须是正整数")),
        }
    }

    /// `[needs]`：算子级公共需求，每条一个原语的能力签名。
    pub(super) fn needs(table: &Table) -> diagnostic::Result<Vec<Need>> {
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
    pub(super) fn layouts(table: &Table) -> diagnostic::Result<Vec<(String, LayoutSpec)>> {
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
    pub(super) fn layout(key: &str, value: &Value) -> diagnostic::Result<(String, LayoutSpec)> {
        let path = format!("layouts.{key}");
        let Value::Str(text) = value else {
            return Err(field(&path, "必须是字符串"));
        };
        let spec = LayoutSpec::parse(text).map_err(|detail| field(&path, detail.message()))?;
        Ok((key.to_string(), spec))
    }

    /// `[grid]`：Gemm 骨架的网格映射（仅 Gemm 声明，其余骨架不需要）。缺省返回 `None`。
    ///
    /// 一个键族：`dims` 是 `"线程:标量"` 数组（网格维，按序）。线程名 = 模板里的循环变量名，标量名 =
    /// 该循环的边界。
    pub(super) fn grid(table: &Table) -> diagnostic::Result<Option<Grid>> {
        let Some(section) = table.table("grid") else {
            return Ok(None);
        };
        Ok(Some(Grid {
            dims: grid_dims(section)?,
        }))
    }

    /// `grid.dims`：`"线程:标量"` 数组，至少一维、至多三维（Metal 网格上限）。
    pub(super) fn grid_dims(section: &Table) -> diagnostic::Result<Vec<(String, String)>> {
        let Some(value) = section.get("dims") else {
            return Err(field("grid.dims", "缺失"));
        };
        let Value::Array(items) = value else {
            return Err(field("grid.dims", "必须是数组"));
        };
        let dims = items
            .iter()
            .enumerate()
            .map(|(index, item)| grid_item(index, item))
            .collect::<diagnostic::Result<Vec<_>>>()?;
        check_grid_dims(&dims)?;
        Ok(dims)
    }

    /// 解析一维：`"线程:标量"`，两个名字都不能为空。
    pub(super) fn grid_item(index: usize, item: &Value) -> diagnostic::Result<(String, String)> {
        let path = format!("grid.dims[{index}]");
        let Value::Str(text) = item else {
            return Err(field(&path, "必须是 `线程:标量` 字符串"));
        };
        let Some((thread, scalar)) = text.split_once(':') else {
            return Err(field(&path, "必须是 `线程:标量` 字符串"));
        };
        if thread.is_empty() || scalar.is_empty() {
            return Err(field(&path, "线程名与标量名都不能为空"));
        }
        Ok((thread.to_string(), scalar.to_string()))
    }

    /// 网格维数约束：至少一维、至多三维（Metal 网格上限）。
    pub(super) fn check_grid_dims(dims: &[(String, String)]) -> diagnostic::Result<()> {
        if dims.is_empty() {
            return Err(field("grid.dims", "至少一维"));
        }
        if dims.len() > 3 {
            return Err(field("grid.dims", "至多三维（Metal 网格上限）"));
        }
        Ok(())
    }

    /// `[chunk]`：分块并行的块长（reduce/scan 骨架）。缺省返回 `None`（串行兜底）。
    pub(super) fn chunk(table: &Table) -> diagnostic::Result<Option<Chunk>> {
        let Some(section) = table.table("chunk") else {
            return Ok(None);
        };
        let size = section
            .int("size")
            .ok_or_else(|| field("chunk.size", "缺失"))?;
        if size < 1 {
            return Err(field("chunk.size", "至少 1"));
        }
        Ok(Some(Chunk {
            size: size as usize,
        }))
    }

    /// `[consts]`：模板常量的固定取值（非轴、非布局）。键给槽位名，值按十进制渲染成字符串，
    /// 供 [`OpDecl::const_value`] 在 `{{槽位}}` 解析时查用。
    pub(super) fn consts(table: &Table) -> diagnostic::Result<Vec<(String, String)>> {
        let Some(section) = table.table("consts") else {
            return Ok(Vec::new());
        };
        section
            .entries()
            .iter()
            .map(|(key, value)| const_value(key, value))
            .collect()
    }

    /// 一个模板常量：整数/浮点/布尔按十进制渲染，字符串原样；表或数组不能当常量。
    pub(super) fn const_value(key: &str, value: &Value) -> diagnostic::Result<(String, String)> {
        let text = match value {
            Value::Str(text) => text.clone(),
            Value::Int(number) => number.to_string(),
            Value::Float(number) => number.to_string(),
            Value::Bool(flag) => flag.to_string(),
            _ => return Err(field(&format!("consts.{key}"), "不能是表或数组")),
        };
        Ok((key.to_string(), text))
    }

    /// `[consts]` 与 `[grid]`、`[chunk]` 三块可选的补丁：`declared` 按同一顺序解构。
    pub(super) type Extras = (Vec<(String, String)>, Option<Grid>, Option<Chunk>);

    /// `[consts]` 与 `[grid]`、`[chunk]`：声明表里三块可选的补丁，一起解析成 `(常量, 网格, 分块)`。
    pub(super) fn extras(table: &Table) -> diagnostic::Result<Extras> {
        Ok((consts(table)?, grid(table)?, chunk(table)?))
    }
}

mod variant {
    //! 变体链：`[variants]` 的解析与链路校验（数量、`order` 连续、共享比例范围）。
    //!
    //! 变体是同一算子的多种落地形态（dtype/调度），`[variants]` 的顺序由 `order` 字段定序后
    //! 才参与 `MAX_VARIANTS` 与链路检查。
    use diagnostic::data::Table;

    use crate::sch;
    use crate::sch::Schedule;
    use prof::atom::Need;

    use super::build::parse_need;
    use super::{MAX_VARIANTS, Variant, field};

    /// 变体链：逐个解析、按 `order` 排序，再判数量与链路。
    pub(super) fn variants(table: &Table) -> diagnostic::Result<Vec<Variant>> {
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
}
pub use expand::{expand, tiles};

#[cfg(test)]
#[path = "../tests/decl_tests.rs"]
mod tests;
