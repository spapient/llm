//! 职责：算子声明构建：模板实参与布局规格的实例化。
use diagnostic::Error;
use diagnostic::data::{Table, Value};

use crate::layout::LayoutSpec;
use crate::tpl::{Arg, Template};
use prof::atom::{Attr, AttrKey, Closed, DTypeSet, Need, Prim, Set, SpaceSet, Span};
use prof::prof::scalar_texts;

use super::expand::tiles;
use super::parse::{Extras, Sections, extras, head, phases, sections};
use super::{Axis, OpDecl, Skeleton, Variant, field};

/// 表里的全部段落一次取齐。`head`/`sections`/`extras` 各自的 `?` 收在这一处：
/// 构造函数本身没有分支，圈复杂度不该被几个"不可能失败"的解析步骤堆上去。
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
                .filter(|param| !param.ctype.is_buffer())
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
            .filter(|param| param.ctype.is_buffer())
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
            .filter(|param| param.ctype.is_buffer())
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

/// 从字符串列表解析一个闭集位集：`["f32", "f16"]` 或单个标量都行；闭集外的名字报错并列出全量。
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
