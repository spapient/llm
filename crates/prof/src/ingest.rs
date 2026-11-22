//! 声明语料的摄入：`ops/*.toml` 的编译期表 → [`OpDecl`]，含各段字段解析与模板侧校验；形状与代数在 [`super`]（`decl.rs`）。

use diagnostic::Error;
use diagnostic::data::{Table, Value};

use crate::atom::{Attr, AttrKey, Closed, Need, Prim, Set, Span};
use crate::decode::{self, Absent, Fail, rows};
use crate::sch::layout::LayoutSpec;
use crate::tpl::{Arg, Template};

use super::{Axis, Chunk, Grid, OpDecl, Phase, Skeleton, Variant, field, tiles, variants};

/// 声明层的失败口径：声明写错都是同一类（`invalid`），路径由 [`field`] 统一带出。
impl Fail for Error {
    fn bad(path: &str, detail: &str) -> Self {
        field(path, detail)
    }
    fn unknown(path: &str, value: &str, allowed: &str) -> Self {
        field(path, format!("`{value}` 不在闭集内（闭集：{allowed}）"))
    }
}

/// 表里的全部段落一次取齐：构造函数本身没有分支，圈复杂度不该被解析步骤堆上去。
fn parts(table: &Table) -> diagnostic::Result<(Head, Sections, Extras)> {
    Ok((head(table)?, sections(table)?, extras(table)?))
}

/// `head` 的三元组：名字、骨架、摘要。
type Head = (String, Skeleton, String);

impl OpDecl {
    /// 从编译期生成的声明表解析声明，并解析同名模板。
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
                .filter(|param| !param.ctype.is_buf())
                .count()
                == 1
            && self.same_element_type()
    }

    /// 全部缓冲同型：`y[n] = f(x[n])` 只有在此情形下才「线程下标就是元素下标」。
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

    /// 声明本体：头、四个段与签名。模板侧校验不在这里（见 [`Self::verified`]）。
    fn declared(table: &Table, template: &str) -> diagnostic::Result<Self> {
        let ((name, skeleton, summary), (axes, needs, layouts, variants), (consts, grid, chunk)) =
            parts(table)?;
        let signature = crate::sig::Signature::parse(&name, template)?;
        let phases = phases(table, &name, &signature)?;
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
            phases,
        })
    }

    /// 模板侧的三条判据：槽位有定义、每条原语都被申请、调用点参数落在签名内。
    fn verified(&self, template: &str) -> diagnostic::Result<()> {
        let segments = Template::parse(template);
        tiles(self)?;
        self.check_phases()?;
        self.check_slots(&segments)?;
        self.check_needs(&segments)?;
        self.check_call_arguments(&segments)
    }

    /// 每个缓冲形参都必须出现在至少一个阶段里：漏写会让 `compose` 静默漏缝。
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
        self.axis(slot).is_some() || self.layout(slot).is_some() || self.const_value(slot).is_some()
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
pub(crate) fn parse_need(section: &Table, key: &str, path: &str) -> Result<Need, Error> {
    let prim = decode::closed::<Prim, Error>(path, key)?;
    let detail = need_table(section, key, path)?;
    // 不拷这张表：解析只读它，为一棵内联表克隆一次等于每条需求多付一次分配。
    detail
        .entries()
        .iter()
        .try_fold(Need::of(prim), |need, (name, value)| {
            need_field(need, prim, path, name, value)
        })
}

/// 需求表里的一个键：维度键（`m`/`bytes`/`extent`/`n`/`k`）挂跨度，其余是能力属性。
fn need_field(
    need: Need,
    prim: Prim,
    path: &str,
    name: &String,
    value: &Value,
) -> Result<Need, Error> {
    let value_path = format!("{path}.{name}");
    let attach: fn(Need, Span) -> Need = match name.as_str() {
        "m" | "bytes" | "extent" => Need::with_extent,
        "n" => Need::with_n,
        "k" => Need::with_k,
        _ => return collection(need, prim, &value_path, name, value),
    };
    let Value::Str(text) = value else {
        return Err(field(
            &value_path,
            "必须是字符串（`16`、`16..64`、`<=512`）",
        ));
    };
    Ok(attach(need, decode::span::<Error>(&value_path, text)?))
}

/// 位集解码：`WORDS` 由目标类型给，失败类型固定本层（`decode` 那边只有一份实现）。
fn closed_set<E: Closed, const WORDS: usize>(
    path: &str,
    value: &Value,
) -> Result<Set<E, WORDS>, Error> {
    decode::closed_set(path, value)
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

/// 非跨度键：`dt`/`space` 各一条集合，其余是能力属性。
fn collection(
    need: Need,
    prim: Prim,
    path: &str,
    name: &str,
    value: &Value,
) -> Result<Need, Error> {
    match name {
        "dt" => Ok(need.with_dt(closed_set(path, value)?)),
        "space" => Ok(need.with_space(closed_set(path, value)?)),
        _ => attribute(need, prim, path, name, value),
    }
}

/// 能力属性：键必须属于该原语的签名（`sat` 不属于 `load`），取值是字符串列表。
fn attribute(need: Need, prim: Prim, path: &str, name: &str, value: &Value) -> Result<Need, Error> {
    let attr_key = decode::closed::<AttrKey, Error>(path, name)?;
    // 无关键既不会被协商比对、也不会被片段消费——它只会让声明看起来更具体，装载期就拒收。
    if !prim.accepts(attr_key) {
        let mut allowed: Vec<&str> = prim.attr_keys().iter().map(|key| key.as_str()).collect();
        allowed.push("exec");
        let detail = format!(
            "`{prim}` 的能力签名里没有 `{attr_key}`（该原语的属性键：{}）",
            allowed.join(" | ")
        );
        return Err(field(path, detail));
    }
    Ok(need.with_attr(Attr::new(attr_key, decode::names::<Error>(path, value)?)))
}

// ── 段解析 ─────────────────────────────────────────────────────────────

/// 算子名：`[op].name`，非空。
pub fn op_name(op: &Table) -> diagnostic::Result<String> {
    let name = op.string("name").ok_or_else(|| field("op.name", "缺失"))?;
    if name.is_empty() {
        return Err(field("op.name", "不能为空"));
    }
    Ok(name.to_string())
}

/// `[op]` 段读出的身份：名字、骨架、摘要。
pub fn head(table: &Table) -> diagnostic::Result<(String, Skeleton, String)> {
    let op = table
        .table("op")
        .ok_or_else(|| field("op", "缺少 `[op]` 段"))?;
    let (skeleton, summary) = op_head(op)?;
    Ok((op_name(op)?, skeleton, summary))
}

/// `[op]` 的其余两键：骨架必须落在闭集内，摘要缺省为空串占位的 `-`。
pub fn op_head(op: &Table) -> diagnostic::Result<(Skeleton, String)> {
    let skeleton = decode::closed::<Skeleton, Error>(
        "op.skeleton",
        op.string("skeleton")
            .ok_or_else(|| field("op.skeleton", "缺失"))?,
    )?;
    Ok((skeleton, op.string("summary").unwrap_or("-").to_string()))
}

/// 四个段（轴、需求、布局、变体链）：`declared` 按同一顺序解构。
pub type Sections = (
    Vec<Axis>,
    Vec<Need>,
    Vec<(String, LayoutSpec)>,
    Vec<Variant>,
);

pub fn sections(table: &Table) -> diagnostic::Result<Sections> {
    Ok((
        axes(table)?,
        needs(table)?,
        layouts(table)?,
        variants(table)?,
    ))
}

/// `[axes]`：每个键一组正整数取值；至少要有一条轴。
pub fn axes(table: &Table) -> diagnostic::Result<Vec<Axis>> {
    let out: Vec<Axis> = rows(
        table,
        "axes",
        Absent::Required("至少需要一个轴"),
        |_, key, value| axis(key, value),
    )?;
    if out.is_empty() {
        return Err(field("axes", "至少需要一个轴"));
    }
    Ok(out)
}
/// 一条轴：`axes.{名字}` 是非空的正整数数组。
pub fn axis(key: &str, value: &Value) -> diagnostic::Result<Axis> {
    let path = format!("axes.{key}");
    let Value::Array(items) = value else {
        return Err(field(&path, "必须是数组"));
    };
    let values = items
        .iter()
        .map(|item| decode::positive::<Error>(&path, item))
        .collect::<diagnostic::Result<Vec<u32>>>()?;
    if values.is_empty() {
        return Err(field(&path, "数组不能为空"));
    }
    Ok(Axis {
        name: key.to_string(),
        values,
    })
}

/// `[needs]`：算子级公共需求，每条一个原语的能力签名。
pub fn needs(table: &Table) -> diagnostic::Result<Vec<Need>> {
    rows(table, "needs", Absent::Empty, |section, key, _| {
        parse_need(section, key, &format!("needs.{key}"))
    })
}

/// `[layouts]`：槽位名 → 布局闭集里的一个词。
pub fn layouts(table: &Table) -> diagnostic::Result<Vec<(String, LayoutSpec)>> {
    rows(table, "layouts", Absent::Empty, |_, key, value| {
        layout(key, value)
    })
}

/// 一条布局：`layouts.{槽位}` 必须是布局闭集内的字符串。
pub fn layout(key: &str, value: &Value) -> diagnostic::Result<(String, LayoutSpec)> {
    let path = format!("layouts.{key}");
    let Value::Str(text) = value else {
        return Err(field(&path, "必须是字符串"));
    };
    let spec = LayoutSpec::parse(text).map_err(|detail| field(&path, detail.message()))?;
    Ok((key.to_string(), spec))
}

/// `[grid]`：Gemm 骨架的网格映射（仅 Gemm 声明，其余骨架不需要）。缺省返回 `None`。
pub fn grid(table: &Table) -> diagnostic::Result<Option<Grid>> {
    let Some(section) = table.table("grid") else {
        return Ok(None);
    };
    Ok(Some(Grid {
        dims: grid_dims(section)?,
    }))
}

/// `grid.dims`：`"线程:标量"` 数组，至少一维、至多三维（Metal 网格上限）。
pub fn grid_dims(section: &Table) -> diagnostic::Result<Vec<(String, String)>> {
    let Some(value) = section.get("dims") else {
        return Err(field("grid.dims", "缺失"));
    };
    let Value::Array(items) = value else {
        return Err(field("grid.dims", "必须是数组"));
    };
    let dims = items
        .iter()
        .enumerate()
        .map(|(idx, item)| grid_item(idx, item))
        .collect::<diagnostic::Result<Vec<_>>>()?;
    check_grid_dims(&dims)?;
    Ok(dims)
}

/// 解析一维：`"线程:标量"`，两个名字都不能为空。
pub fn grid_item(idx: usize, item: &Value) -> diagnostic::Result<(String, String)> {
    let path = format!("grid.dims[{idx}]");
    let Value::Str(text) = item else {
        return Err(field(&path, "必须是 `线程:标量` 字符串"));
    };
    let Some((thread, scalar)) = text
        .split_once(':')
        .filter(|(thread, scalar)| !thread.is_empty() && !scalar.is_empty())
    else {
        return Err(field(&path, "必须是 `线程:标量` 字符串"));
    };
    Ok((thread.to_string(), scalar.to_string()))
}

/// 网格维数约束：至少一维、至多三维（Metal 网格上限）。
pub fn check_grid_dims(dims: &[(String, String)]) -> diagnostic::Result<()> {
    if dims.is_empty() {
        return Err(field("grid.dims", "至少一维"));
    }
    if dims.len() > 3 {
        return Err(field("grid.dims", "至多三维（Metal 网格上限）"));
    }
    Ok(())
}

/// `[chunk]`：分块并行的块长（reduce/scan 骨架）。缺省返回 `None`（串行兜底）。
pub fn chunk(table: &Table) -> diagnostic::Result<Option<Chunk>> {
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

/// `[consts]`：模板常量的固定取值（非轴、非布局）。键给槽位名，值按十进制渲染成字符串， 供 [`OpDecl::const_value`] 在 `{{槽位}}` 解析时查用。
pub fn consts(table: &Table) -> diagnostic::Result<Vec<(String, String)>> {
    rows(table, "consts", Absent::Empty, |_, key, value| {
        const_value(key, value)
    })
}

/// 一个模板常量：整数/浮点/布尔按十进制渲染，字符串原样；表或数组不能当常量。
pub fn const_value(key: &str, value: &Value) -> diagnostic::Result<(String, String)> {
    let path = format!("consts.{key}");
    Ok((key.to_string(), decode::scalar_text::<Error>(&path, value)?))
}

/// `[consts]` 与 `[grid]`、`[chunk]` 三块可选的补丁：`declared` 按同一顺序解构。
pub type Extras = (Vec<(String, String)>, Option<Grid>, Option<Chunk>);

pub fn extras(table: &Table) -> diagnostic::Result<Extras> {
    Ok((consts(table)?, grid(table)?, chunk(table)?))
}

/// `[[phases]]`：数据流阶段（可选）。缺省 = 单阶段：名字取算子名，读/写槽位取自签名。
pub fn phases(
    table: &Table,
    op: &str,
    signature: &crate::sig::Signature,
) -> diagnostic::Result<Vec<Phase>> {
    let Some(rows) = table.table_array("phases") else {
        return Ok(vec![Phase {
            name: op.to_string(),
            reads: signature.buf_names(false),
            writes: signature.buf_names(true),
        }]);
    };
    rows.iter().map(|row| phase(op, row, signature)).collect()
}

/// 一条阶段：`name` 必填，`reads`/`writes` 是槽位名数组，每个名字都必须是签名的缓冲形参。
fn phase(op: &str, row: &Table, signature: &crate::sig::Signature) -> diagnostic::Result<Phase> {
    let name = row
        .string("name")
        .map(str::to_string)
        .ok_or_else(|| field(&format!("{op}.phases.name"), "缺失"))?;
    Ok(Phase {
        name,
        reads: slot_names(op, row, "reads", signature)?,
        writes: slot_names(op, row, "writes", signature)?,
    })
}

/// 阶段里的槽位名数组：每个名字都必须是签名的缓冲形参，否则拼装期就拦下（而不是漏缝）。
fn slot_names(
    op: &str,
    row: &Table,
    key: &str,
    signature: &crate::sig::Signature,
) -> diagnostic::Result<Vec<String>> {
    let Some(names) = row.string_array(key) else {
        return Ok(Vec::new());
    };
    let unknown = names
        .iter()
        .find(|name| !signature.params().iter().any(|param| param.name == **name));
    let Some(name) = unknown else {
        return Ok(names.into_iter().map(str::to_string).collect());
    };
    Err(field(
        &format!("{op}.phases.{key}"),
        format!("`{name}` 不是签名的缓冲形参"),
    ))
}
