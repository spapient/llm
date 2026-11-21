//! `docs/ARCHITECTURE.md` 的三张表：层表、允许的边、所有权表。
//!
//! 三张表是同一份事实源（[`DOC`]，编译期嵌入），所以解析只做一次（[`Spec::parse`]），门禁各自
//! 取查询结果去比对 `cargo metadata` 与源码：改文档或改 `Cargo.toml` 而不同步另一侧即失败。
//! markdown 文法只在这里写一遍——门禁不复制第二份解析口径。
//!
//! 每张表解析为空都当场失败：「表在，但一条也没解出来」只会是格式漂移，让门禁静默全绿。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::edge::{Edge, short};

/// 契约文本编译期嵌入：门禁不依赖当前工作目录，也读不到另一份副本。
pub(crate) const DOC: &str = include_str!("../../../docs/ARCHITECTURE.md");

/// 成员在层表里的位置：`layer` 越小越靠下游，同层 `order` 越小越靠左。
///
/// `Ord` 就是"允许被谁依赖"——合法边即 `to < from`，方向判据因此是一次比较而不是两个分支。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Rank {
    /// 层号。
    pub(crate) layer: usize,
    /// 层内书写次序。
    pub(crate) order: usize,
}

/// 文档对边数的**声明**；实际条数与它另有门禁核对。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Totals {
    /// 全部边（含 `+dev`）。
    pub(crate) edges: usize,
    /// 其中 `+dev` 的条数。
    pub(crate) dev: usize,
}

/// 三张表解析后的事实面；只由 [`Spec::parse`] 构造，只读。
pub(crate) struct Spec {
    layers: BTreeMap<String, Rank>,
    tools: BTreeSet<String>,
    edges: BTreeSet<Edge>,
    totals: Totals,
    owners: BTreeMap<String, BTreeSet<String>>,
}

impl Spec {
    /// 进程内只解析一次：所有门禁读同一份文档，三张表也就只解析一遍。
    pub(crate) fn parse() -> &'static Spec {
        static SPEC: OnceLock<Spec> = OnceLock::new();
        SPEC.get_or_init(Spec::read)
    }

    /// 层表：运行时成员 → 位置。工具成员不在表内（它们没有层）。
    pub(crate) fn layers(&self) -> &BTreeMap<String, Rank> {
        &self.layers
    }

    /// 单个成员在层表里的位置。
    pub(crate) fn rank(&self, member: &str) -> Option<Rank> {
        self.layers.get(member).copied()
    }

    /// 工具成员：装配与验证的入口，不参与层序。
    pub(crate) fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    /// 文档是否登记过这个成员（运行时层或工具成员）。
    pub(crate) fn declares(&self, member: &str) -> bool {
        self.layers.contains_key(member) || self.tools.contains(member)
    }

    /// 文档声明的边集。
    pub(crate) fn edges(&self) -> &BTreeSet<Edge> {
        &self.edges
    }

    /// 文档声明的边数。
    pub(crate) fn totals(&self) -> Totals {
        self.totals
    }

    /// 所有权表：主人 → 它声称拥有的类型名。
    pub(crate) fn owners(&self) -> &BTreeMap<String, BTreeSet<String>> {
        &self.owners
    }

    /// 所有权表逐条 `(主人, 类型名)`；主人按名排序，输出因此稳定可 diff。
    pub(crate) fn owned(&self) -> impl Iterator<Item = (&str, &str)> {
        self.owners
            .iter()
            .flat_map(|(owner, names)| names.iter().map(move |name| (owner.as_str(), name.as_str())))
    }

    /// 所有权表登记的全部类型名，与主人无关。
    pub(crate) fn listed_types(&self) -> BTreeSet<&str> {
        self.owners
            .values()
            .flatten()
            .map(String::as_str)
            .collect()
    }

    fn read() -> Spec {
        let (edges, totals) = edges();
        Spec {
            layers: layers(),
            tools: tools(),
            edges,
            totals,
            owners: owners(),
        }
    }
}

/// `| L<n> | ... |` 表格行：成员格里的每个反引号片段是一个成员，书写次序即层内位置。
fn layers() -> BTreeMap<String, Rank> {
    let mut table = BTreeMap::new();
    for line in DOC.lines() {
        let Some(rest) = line.strip_prefix("| L") else {
            continue;
        };
        let Some((level, tail)) = rest.split_once(' ') else {
            continue;
        };
        let Ok(level) = level.parse::<usize>() else {
            continue;
        };
        let Some(members) = tail.split('|').nth(1) else {
            continue;
        };
        // `L7` 的成员格还带 bin 说明（`llm`（bin：`llm` 与 `cutile`）），括号内不是成员。
        let members = members.split('（').next().unwrap_or_default();
        for (order, member) in backticks(members).into_iter().enumerate() {
            table.insert(short(&member), Rank { layer: level, order });
        }
    }
    assert!(
        !table.is_empty(),
        "层表解析为 0 条：`| L<n> |` 表格行的格式变了"
    );
    table
}

/// 「… 是**工具成员**」那一行的反引号片段。
fn tools() -> BTreeSet<String> {
    let line = DOC
        .lines()
        .find(|line| line.contains("是**工具成员**"))
        .expect("`ARCHITECTURE.md` 必须有一段声明工具成员");
    backticks(line).into_iter().map(|name| short(&name)).collect()
}

/// 「允许的边」一节：代码块里的边集，与标题行声明的总数。
fn edges() -> (BTreeSet<Edge>, Totals) {
    let mut lines = DOC.lines();
    let header = lines
        .find(|line| line.contains("允许的边（当前全量"))
        .expect("`ARCHITECTURE.md` 必须有「允许的边」一节");
    let totals = Totals {
        edges: count_after(header, "共 "),
        dev: count_after(header, "含 "),
    };
    let mut edges = BTreeSet::new();
    for entry in fence_block(lines) {
        add_row(&mut edges, &entry);
    }
    assert!(
        !edges.is_empty(),
        "边表解析为 0 条：代码块的格式变了"
    );
    (edges, totals)
}

/// 紧跟标题的第一个 ``` 代码块，一行一条边；续行拼到上一条（文档里长行会折行）。
fn fence_block<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    let mut opened = false;
    for line in lines {
        let trimmed = line.trim();
        if trimmed == "```" {
            if opened {
                break;
            }
            opened = true;
            continue;
        }
        if !opened || trimmed.is_empty() {
            continue;
        }
        if line.contains('→') {
            entries.push(trimmed.to_string());
        } else if let Some(last) = entries.last_mut() {
            last.push(' ');
            last.push_str(trimmed);
        }
    }
    entries
}

/// 一行 `a/b → c, d（叶）+dev: e`。
fn add_row(edges: &mut BTreeSet<Edge>, entry: &str) {
    let (lhs, rhs) = entry.split_once('→').expect("每条边都有箭头");
    let (runtime, fixture) = match rhs.split_once("+dev:") {
        Some((runtime, fixture)) => (runtime, names(fixture)),
        None => (rhs, Vec::new()),
    };
    // 「叶」只表示没有运行时出边；同一行仍可以登记 `+dev` 测试夹具边。
    let runtime = if runtime.contains("（叶") {
        Vec::new()
    } else {
        names(runtime)
    };
    for from in lhs.split('/').map(short) {
        for dev in [false, true] {
            let targets = if dev { &fixture } else { &runtime };
            for to in targets {
                edges.insert(Edge {
                    from: from.clone(),
                    to: to.clone(),
                    dev,
                });
            }
        }
    }
}

/// 逗号分隔的成员名，去掉括号说明。
fn names(part: &str) -> Vec<String> {
    part.split(',')
        .map(|name| name.split('（').next().unwrap_or_default())
        .map(short)
        .filter(|name| !name.is_empty())
        .collect()
}

/// 反引号包裹的片段：成对出现，奇数位置即内容。
pub(crate) fn backticks(text: &str) -> Vec<String> {
    text.split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

/// `prefix` 之后的十进制数；缺前缀或不是数字时失败——这两个数写在正文里，解析不出来就是格式漂了。
fn count_after(text: &str, prefix: &str) -> usize {
    let start = text
        .find(prefix)
        .unwrap_or_else(|| panic!("文档缺 `{prefix}`"))
        + prefix.len();
    text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("`{prefix}` 后不是数字"))
}

/// 所有权表：`| 类型 | 主人 | ...` 行，主人格里的反引号片段是主人。
fn owners() -> BTreeMap<String, BTreeSet<String>> {
    let mut table: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in section("## 三、所有权表") {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') || trimmed.starts_with("|---") {
            continue;
        }
        let cells: Vec<&str> = trimmed.split('|').map(str::trim).collect();
        let owner = cells
            .get(2)
            .and_then(|cell| backticks(cell).into_iter().next());
        let types = cells.get(1);
        if let (Some(owner), Some(types)) = (owner, types) {
            table.entry(short(&owner)).or_default().extend(backticks(types));
        }
    }
    assert!(
        !table.is_empty(),
        "所有权表解析为 0 条：`## 三、所有权表` 的表头格式变了"
    );
    table
}

/// `title` 开头的那一节，直到下一个二级标题之前。
fn section(title: &str) -> Vec<&'static str> {
    DOC.lines()
        .skip_while(move |line| !line.starts_with(title))
        .skip(1)
        .take_while(|line| !line.starts_with("## "))
        .collect()
}
