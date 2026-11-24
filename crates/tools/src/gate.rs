//! 质量门配置：从编译期生成的 `lints/vocabulary.toml` 表读取强类型阈值与存量棘轮。

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use diagnostic::data::Table;
use diagnostic::{Code, Error, Result};

/// 静态质量门：阈值与存量债基线。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gate {
    max_ccn: usize,
    max_nloc: usize,
    max_file_lines: usize,
    min_file_lines: usize,
    max_test_lines: usize,
    debt: BTreeMap<String, Debt>,
    /// 行数下限存量基线：这些既存正交模块暂不受 [`min_lines`](Self::min_lines) 约束；
    /// 新开口子必须达到下限或归入既有模块。条目达标（≥下限）后应从清单删除。
    min_baseline: BTreeSet<String>,
}

/// 一个存量债项的登记度量（只登记仍超标的字段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Debt {
    /// 登记的圈复杂度。
    ccn: Option<usize>,
    /// 登记的函数长度。
    nloc: Option<usize>,
    /// 登记的文件行数。
    lines: Option<usize>,
    /// 登记的禁止形态数量。
    call: Option<usize>,
}

impl Debt {
    /// 某个度量的登记值；未登记的度量没有预算，等同「新增超标」。
    #[must_use]
    const fn allowed(&self, metric: Metric) -> Option<usize> {
        match metric {
            Metric::Ccn => self.ccn,
            Metric::Nloc => self.nloc,
            Metric::Lines => self.lines,
            Metric::Call => self.call,
        }
    }

    /// 已登记的（度量, 登记值）对，顺序即棘轮逐条结清的顺序。
    fn registered(self) -> impl Iterator<Item = (Metric, usize)> {
        [
            (Metric::Ccn, self.ccn),
            (Metric::Nloc, self.nloc),
            (Metric::Lines, self.lines),
            (Metric::Call, self.call),
        ]
        .into_iter()
        .filter_map(|(metric, value)| value.map(|value| (metric, value)))
    }
}

/// 度量类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Metric {
    /// 圈复杂度。
    Ccn,
    /// 函数长度。
    Nloc,
    /// 文件行数。
    Lines,
    /// 禁止形态数量。
    Call,
}

/// 一个当前观测到的超标项：身份、度量、当前值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Over {
    /// 身份键（文件用路径，函数用 `路径:名#序号`）。
    pub key: String,
    /// 度量类别。
    pub metric: Metric,
    /// 当前值。
    pub value: usize,
}

impl Gate {
    /// 读取编译期生成的词汇表：`lints/vocabulary.toml` 的 `[gate]`。
    pub fn load() -> Result<Self> {
        Self::from_table(crate::vocabulary())
    }

    /// 由词汇表构造阈值与存量棘轮；校验与错误文案与旧的运行期解析一致。
    pub fn from_table(root: &Table) -> Result<Self> {
        let table = root
            .table("gate")
            .ok_or_else(|| invalid("vocabulary.toml 缺 [gate]"))?;
        let [
            max_ccn,
            max_nloc,
            max_file_lines,
            min_file_lines,
            max_test_lines,
        ] = thresholds(table)?;
        let gate = Self {
            max_ccn,
            max_nloc,
            max_file_lines,
            min_file_lines,
            max_test_lines,
            debt: debts(table)?,
            min_baseline: min_baseline(table),
        };
        gate.check_thresholds().map(|()| gate)
    }

    /// 某个度量适用的限值：门禁唯一的限值出口。
    #[must_use]
    pub const fn limit(&self, metric: Metric) -> usize {
        match metric {
            Metric::Ccn => self.max_ccn,
            Metric::Nloc => self.max_nloc,
            Metric::Lines => self.max_file_lines,
            Metric::Call => 0,
        }
    }

    /// 测试代码的行数上限：`src` 只放生产代码，测试同样不许长成巨石。
    ///
    /// 与 [`Gate::limit`] 分开是因为它不是棘轮项：测试文件没有历史债，一超就是新增。
    #[must_use]
    pub const fn test_limit(&self) -> usize {
        self.max_test_lines
    }

    /// 生产文件行数下限：薄入口与 libFuzzer 垫片由扫描侧豁免，其余独立文件不足此行数不允许单开。
    #[must_use]
    pub const fn min_lines(&self) -> usize {
        self.min_file_lines
    }

    /// 该相对路径是否在行数下限存量基线内（既存正交模块，暂不受下限约束）。
    #[must_use]
    pub fn in_min_baseline(&self, key: &str) -> bool {
        self.min_baseline.contains(key)
    }

    /// 判定一个**已超限**的测量；返回 `Some(理由)` 即门禁失败。
    #[must_use]
    pub fn reject_reason(&self, over: &Over) -> Option<String> {
        let allowed = self
            .debt
            .get(&over.key)
            .and_then(|debt| debt.allowed(over.metric));
        let Some(allowed) = allowed else {
            return Some(new_entry(over));
        };
        match over.value.cmp(&allowed) {
            Ordering::Equal => None,
            Ordering::Greater => Some(format!(
                "{}: {:?} {} 恶化 > 登记 {allowed}",
                over.key, over.metric, over.value
            )),
            Ordering::Less => Some(format!(
                "{}: {:?} 实测 {} < 登记 {allowed}：把基线条目改成 {}",
                over.key, over.metric, over.value, over.value
            )),
        }
    }

    /// 已登记、但本次**没有观测到超限**的度量：都必须删除，避免永久豁免。
    #[must_use]
    pub fn stale(&self, live: &BTreeMap<String, BTreeSet<Metric>>) -> Vec<String> {
        let empty = BTreeSet::new();
        let mut out = Vec::new();
        for (key, debt) in &self.debt {
            for (metric, _) in debt.registered() {
                if !live.get(key).unwrap_or(&empty).contains(&metric) {
                    out.push(format!("{key}: {metric:?} 已达标，删除基线条目"));
                }
            }
        }
        out
    }

    fn check_thresholds(&self) -> Result<()> {
        let positive = [
            self.max_ccn,
            self.max_nloc,
            self.max_file_lines,
            self.min_file_lines,
            self.max_test_lines,
        ];
        if positive.contains(&0) {
            return Err(invalid("[gate] 阈值必须为正"));
        }
        if self.min_file_lines > self.max_file_lines {
            return Err(invalid("[gate] file_min_lines 不能大于 file_max_lines"));
        }
        Ok(())
    }
}

/// `[gate]` 里的五个阈值键，顺序与 [`Gate::check_thresholds`] 的校验一致。
const THRESHOLDS: [&str; 5] = [
    "ccn_max",
    "nloc_max",
    "file_max_lines",
    "file_min_lines",
    "test_max_lines",
];

/// `[gate]` 的闭集：五个阈值键 + `debt` 表与 `min_lines_baseline` 数组。拼错的键会被拒绝，而不是静默地不起作用——
/// 一条从未被读到的「档位」比没有这条档位更糟：它让人以为门禁在管那件事。
fn unknown_keys(table: &Table) -> Result<()> {
    match table.entries().iter().find(|(name, _)| {
        name != "debt" && name != "min_lines_baseline" && !THRESHOLDS.contains(&name.as_str())
    }) {
        Some((name, _)) => Err(invalid(&format!("[gate] 未知键 {name}"))),
        None => Ok(()),
    }
}

/// 未登记度量的失败文案；两个失败点共用一份措辞。
fn new_entry(over: &Over) -> String {
    format!("{}: 新增 {:?}={}", over.key, over.metric, over.value)
}

/// 一条门禁配置错误：与旧的运行期解析同一套文案，上下文固定为 `tools::gate`。
fn invalid(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("tools::gate")
}

fn int(table: &Table, key: &str) -> Result<usize> {
    table
        .int(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(&format!("[gate] 缺正整数 {key}")))
}

fn opt_int(table: &Table, key: &str) -> Option<usize> {
    table.int(key).and_then(|value| usize::try_from(value).ok())
}

/// 校验键集合并按 [`THRESHOLDS`] 的顺序读出五个阈值。
fn thresholds(table: &Table) -> Result<[usize; 5]> {
    unknown_keys(table)?;
    let mut limits = [0; THRESHOLDS.len()];
    for (slot, key) in limits.iter_mut().zip(THRESHOLDS) {
        *slot = int(table, key)?;
    }
    Ok(limits)
}

/// `[[gate.debt]]` 行展开成 `key → 登记度量`；没有度量的行是配置错误。
fn debts(table: &Table) -> Result<BTreeMap<String, Debt>> {
    let mut debt = BTreeMap::new();
    for row in table.table_array("debt").unwrap_or_default() {
        let key = row
            .string("key")
            .ok_or_else(|| invalid("[gate.debt] 缺 key"))?
            .to_string();
        let entry = Debt {
            ccn: opt_int(row, "ccn"),
            nloc: opt_int(row, "nloc"),
            lines: opt_int(row, "lines"),
            call: opt_int(row, "call"),
        };
        if entry == Debt::default() {
            return Err(invalid(&format!("[gate.debt] {key} 无度量")));
        }
        debt.insert(key, entry);
    }
    Ok(debt)
}

/// 行数下限存量基线：相对仓库根的正斜杠路径集合；缺省为空（没有豁免条目）。
fn min_baseline(table: &Table) -> BTreeSet<String> {
    table
        .string_array("min_lines_baseline")
        .map_or_default(|items| items.iter().map(ToString::to_string).collect())
}
