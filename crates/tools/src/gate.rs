//! 质量门配置：从编译期生成的 `lints/vocabulary.toml` 表读取强类型阈值与存量棘轮。
//!
//! `build.rs` 在编译期把 `lints/vocabulary.toml` 展开成 `diagnostic::data::Table`，运行期只遍历
//! 这棵树：本 crate 不读配置文件（配置解析只发生在 `cli::settings`，ADR 0048），语料仍在构建期
//! 解析（ADR 0022）。与 burn 的 Config 同思想：标准是数据，
//! 行为是纯函数。
//!
//! 三条边界缺一条，这套门禁就退化成「数字散落在调用方」：
//!
//! 1. **限值只有 [`Gate::limit`] 一个出口**。测量方只报「测到多少」，不持有阈值、也不决定
//!    「哪个度量用哪个数」；`[gate]` 是闭集，拼错的键报错而不是被忽略。
//! 2. **棘轮按（身份, 度量）结清**。同一函数登记了 `ccn` 与 `nloc` 时，改好其中一条就必须删掉
//!    那一条，否则豁免会永久留在表里（[`Gate::stale`]）。
//! 3. **新增必进表，恶化必失败**：全新身份、或某度量的值超过登记值即失败。
//!
//! 能保证的是「不留过期豁免、不允许恶化」。它**不**保证「登记过的项一定在变好」：仍超标但
//! 比登记值改善的值会通过，因为静态门禁没有第二份历史事实可比——那属于版本库，属于 `git`，
//! 不在这里假装拥有。

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use diagnostic::data::Table;
use diagnostic::{Code, Error, Result};

/// 静态质量门：阈值与存量债基线。
///
/// 阈值字段私有是有意的：唯一出口是 [`Gate::limit`]，调用方拿到的是「某个度量的限值」，
/// 而不是「三个可随意组合的数字」。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gate {
    max_ccn: usize,
    max_nloc: usize,
    max_file_lines: usize,
    debt: BTreeMap<String, Debt>,
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
    ///
    /// 表写在用它的地方而不是导出一个 `Metric::ALL`：门禁只在这里需要「全部度量」，
    /// 导出成公开闭集就得接 `prof::atom::Closed`，而 `tools` 是零业务的静态分析器。
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
///
/// 可排序：棘轮按（身份, 度量）结清，`Metric` 是 `BTreeSet` 的键，所以顺序必须由类型给出。
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
///
/// 限值不在里面：它是门禁的策略（[`Gate::limit`]），测量方只能提供「测到多少」。
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
        let [max_ccn, max_nloc, max_file_lines] = thresholds(table)?;
        let gate = Self {
            max_ccn,
            max_nloc,
            max_file_lines,
            debt: debts(table)?,
        };
        gate.check_thresholds().map(|()| gate)
    }

    /// 某个度量适用的限值：门禁唯一的限值出口。
    ///
    /// `Call` 的限值是 0——`unwrap`/`expect`/`panic!` 在 `crates/*/src` 里没有预算，登记值只是
    /// 存量；其余度量读 `[gate]` 的对应阈值。
    #[must_use]
    pub const fn limit(&self, metric: Metric) -> usize {
        match metric {
            Metric::Ccn => self.max_ccn,
            Metric::Nloc => self.max_nloc,
            Metric::Lines => self.max_file_lines,
            Metric::Call => 0,
        }
    }

    /// 判定一个**已超限**的测量；返回 `Some(理由)` 即门禁失败。
    ///
    /// 调用方先用 [`Gate::limit`] 过滤：未超限的测量不是门禁的输入，它对应的「删除基线条目」
    /// 由 [`Gate::stale`] 按度量逐条结清。
    ///
    /// 登记值必须等于实测值：**高于**实测即失败，要求同一提交里把基线降下来。这是「棘轮」真正
    /// 的含义——登记表是快照，不是预算；只写「不许超过登记值」时，降到 6 的 ccn 可以顶着 47 的
    /// 登记值一直活下去，降级永远不会被要求记账。
    #[must_use]
    pub fn check(&self, over: &Over) -> Option<String> {
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
    ///
    /// `live` 是本次观测结果（身份 → 仍超限的度量）。按度量而不是按身份比较，是因为一个函数
    /// 可以同时登记 `ccn` 与 `nloc`：只按身份比较时，改好其中一条也永远不会被要求删表。
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
        let positive = [self.max_ccn, self.max_nloc, self.max_file_lines];
        if positive.contains(&0) {
            return Err(invalid("[gate] 阈值必须为正"));
        }
        Ok(())
    }
}

/// `[gate]` 里的三个阈值键，顺序与 [`Gate::check_thresholds`] 的校验一致。
const THRESHOLDS: [&str; 3] = ["ccn_max", "nloc_max", "file_max_lines"];

/// `[gate]` 的闭集：三个阈值键 + `debt` 表。拼错的键会被拒绝，而不是静默地不起作用——
/// 一条从未被读到的「档位」比没有这条档位更糟：它让人以为门禁在管那件事。
fn unknown_keys(table: &Table) -> Result<()> {
    match table
        .entries()
        .iter()
        .find(|(name, _)| name != "debt" && !THRESHOLDS.contains(&name.as_str()))
    {
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

/// 校验键集合并按 [`THRESHOLDS`] 的顺序读出三个阈值。
fn thresholds(table: &Table) -> Result<[usize; 3]> {
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

#[cfg(test)]
mod tests {
    use diagnostic::data::Value;

    use super::*;

    /// 本仓库的三个阈值：改动它们就改 `lints/vocabulary.toml`。
    const SOUND: [i64; 3] = [5, 60, 1200];

    /// 一份与仓库同阈值、没有存量债的 `[gate]` 表。
    fn sound() -> Table {
        let gate = THRESHOLDS
            .iter()
            .zip(SOUND)
            .map(|(key, value)| (*key, Value::int(value)))
            .collect();
        Table::from_pairs(vec![("gate", Value::table(gate))])
    }

    /// 造一张只含一条 ccn 债务的 gate 与一份实测超限：登记值 `ccn`、实测 `measured`。
    fn debt_gate(ccn: i64, measured: usize) -> (Gate, Over) {
        let mut config = sound();
        config.push(
            "gate.debt",
            Value::table(vec![
                ("key", Value::text("crates/x/src/y.rs:run#0")),
                ("ccn", Value::int(ccn)),
            ]),
        );
        let gate = Gate::from_table(&config).expect("表合法");
        let over = Over {
            key: "crates/x/src/y.rs:run#0".into(),
            metric: Metric::Ccn,
            value: measured,
        };
        (gate, over)
    }

    /// 项目配置：三个阈值来自构建期生成的 `lints/vocabulary.toml`；存量债务只允许挂在入口编排层
    ///（`crates/cli/src/` 下的编排与二进制入口），运行时生产代码 `crates/*/src` 一条都不能有。
    #[test]
    fn the_project_config_lists_only_root_entry_legacy_debt() {
        let gate = Gate::load().expect("项目配置合法");
        let limits = [Metric::Ccn, Metric::Nloc, Metric::Lines, Metric::Call]
            .map(|metric| gate.limit(metric));
        assert_eq!(
            limits,
            [5, 60, 1200, 0],
            "阈值来自构建期生成的 `lints/vocabulary.toml`；`Call` 没有预算"
        );
        for key in gate.debt.keys() {
            assert!(
                key.starts_with("crates/cli/src/"),
                "运行时生产代码没有债务条目，存量只能是入口编排既有逻辑：{key}"
            );
        }
    }

    #[test]
    fn every_threshold_must_be_positive() {
        for key in THRESHOLDS {
            let mut config = sound();
            config.set(&format!("gate.{key}"), Value::int(0));
            assert_eq!(
                Gate::from_table(&config).unwrap_err().message(),
                "[gate] 阈值必须为正",
                "`{key} = 0` 必须被拒"
            );
        }
    }

    #[test]
    fn an_unknown_gate_key_is_an_error_not_noise() {
        let mut config = sound();
        config.set("gate.simd_nloc", Value::int(96));
        assert_eq!(
            Gate::from_table(&config).unwrap_err().message(),
            "[gate] 未知键 simd_nloc",
            "从未被读到的档位必须报错，而不是静默失效"
        );
    }

    #[test]
    fn a_threshold_must_be_a_positive_integer() {
        let mut config = sound();
        config.set("gate.ccn_max", Value::text("5"));
        assert_eq!(
            Gate::from_table(&config).unwrap_err().message(),
            "[gate] 缺正整数 ccn_max"
        );
    }

    #[test]
    fn a_vocabulary_without_a_gate_section_is_rejected() {
        assert_eq!(
            Gate::from_table(&Table::new()).unwrap_err().message(),
            "vocabulary.toml 缺 [gate]"
        );
    }

    #[test]
    fn a_debt_row_without_a_metric_is_rejected() {
        let mut config = sound();
        config.push(
            "gate.debt",
            Value::table(vec![("key", Value::text("crates/x/src/y.rs:run#0"))]),
        );
        assert_eq!(
            Gate::from_table(&config).unwrap_err().message(),
            "[gate.debt] crates/x/src/y.rs:run#0 无度量"
        );
    }

    #[test]
    fn a_debt_row_registers_only_the_metrics_it_names() {
        let (gate, over) = debt_gate(9, 9);
        assert!(gate.check(&over).is_none(), "登记范围内应通过");
        assert!(
            gate.check(&Over { value: 10, ..over }).is_some(),
            "超过登记值应失败"
        );
        assert_eq!(
            gate.stale(&BTreeMap::new()).len(),
            1,
            "未观测到的条目应提示删除"
        );
    }

    /// 登记值高于实测即失败：基线是快照，不是预算，改善必须在同一提交里记账。
    #[test]
    fn a_register_above_the_measurement_must_be_lowered() {
        let (gate, over) = debt_gate(47, 6);
        assert_eq!(
            gate.check(&over).unwrap(),
            "crates/x/src/y.rs:run#0: Ccn 实测 6 < 登记 47：把基线条目改成 6",
            "顶着 47 的登记值活着的 ccn=6 必须被要求记账"
        );
    }

    /// 同一身份登记 ccn 与 nloc：只观测到 ccn 仍超限时，nloc 那条必须被要求删除。
    ///
    /// 这是旧实现的实际漏洞：按身份判断时，`live` 只要含这个 key，两条都算「仍有超标」，
    /// 于是改好的那半永远留在表里。
    #[test]
    fn a_satisfied_metric_must_drop_even_when_the_identity_is_still_live() {
        let mut config = sound();
        config.push(
            "gate.debt",
            Value::table(vec![
                ("key", Value::text("crates/x/src/y.rs:run#0")),
                ("ccn", Value::int(9)),
                ("nloc", Value::int(90)),
            ]),
        );
        let gate = Gate::from_table(&config).expect("表合法");
        let live = BTreeMap::from([(
            "crates/x/src/y.rs:run#0".to_string(),
            BTreeSet::from([Metric::Ccn]),
        )]);
        assert_eq!(
            gate.stale(&live),
            vec!["crates/x/src/y.rs:run#0: Nloc 已达标，删除基线条目".to_string()],
            "身份仍在册不能让已达标的那条度量继续豁免"
        );
        assert_eq!(
            gate.stale(&BTreeMap::new()).len(),
            2,
            "一条都没观测到就该全删"
        );
    }

    /// 三种判定：恶化失败、登记范围内通过、全新身份失败。
    ///
    /// 登记值由本地表给出：项目基线清零后，仓库里没有可以借用的登记值。
    #[test]
    fn ratchet_blocks_new_and_worse_allows_within_budget() {
        let (gate, over) = debt_gate(9, 10);
        assert!(gate.check(&over).is_some(), "恶化必须失败");
        assert!(
            gate.check(&Over { value: 9, ..over }).is_none(),
            "登记范围内应通过（此时仍超阈值）"
        );
        assert!(
            gate.check(&Over {
                key: "crates/x/src/y.rs:new#0".into(),
                ..over
            })
            .is_some(),
            "全新身份直接失败"
        );
    }
}
