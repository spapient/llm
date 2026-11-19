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
    /// 导出成公开闭集就得接 `prof::atom::Closed`，而 `tool` 是零业务的静态分析器。
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

/// 一条门禁配置错误：与旧的运行期解析同一套文案，上下文固定为 `tool::gate`。
fn invalid(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("tool::gate")
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

/// 禁止调用形态：生产代码里的裸 `unwrap`/`expect` 与 panic 宏。
///
/// 纯函数：基于 [`crate::lex`] token 精确匹配，字符串/注释里的 `.unwrap()` 不会误报。
/// 点号方法形态与 UFCS 关联函数形态（`Type::unwrap(x)`）都认。
/// 命名 snake_case 与 `todo!`/`unimplemented!` 已由编译器与 workspace lints 强制，这里不重复。
///
/// `unreachable!` 必须由本模块兜住：它是 panic 路径，却没有任何 lint 覆盖——`todo`/
/// `unimplemented` 被 `clippy` 拒了，`unreachable` 没有。「零 panic」这句话少这一条就只是
/// 「零一部分 panic」。
///
/// `assert!`/`assert_eq!` 不在本门形态内：它们表达逻辑不变量，测试代码先由 `strip_tests`
/// 剔除；生产路径用断言兜底不可信输入属评审纪律，不由本扫描器判定。
pub mod call {
    use crate::lex::{Delim, Pos, Token, parse};

    /// 禁止形态类别。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Kind {
        /// 裸 `.unwrap()`。
        Unwrap,
        /// 裸 `.expect(..)`。
        Expect,
        /// `panic!(..)`。
        Panic,
        /// `unreachable!(..)`。
        Unreachable,
    }

    /// 一处禁止形态：类别与位置。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Call {
        /// 类别。
        pub kind: Kind,
        /// 起始位置。
        pub pos: Pos,
    }

    /// 扫描全部禁止形态。
    pub fn scan(text: &str) -> diagnostic::Result<Vec<Call>> {
        let tokens = parse(text)?;
        let mut out = Vec::new();
        walk(&tokens, &mut out);
        Ok(out)
    }

    /// 递归扫描 token 树，收集禁止形态。
    fn walk(tokens: &[Token], out: &mut Vec<Call>) {
        for (idx, token) in tokens.iter().enumerate() {
            out.extend(method_call(tokens, idx));
            out.extend(ufcs_call(tokens, idx));
            out.extend(macro_call(tokens, idx));
            if let Token::Group { items, .. } = token {
                walk(items, out);
            }
        }
    }

    /// `.unwrap()` / `.expect(..)` 形态。
    fn method_call(tokens: &[Token], idx: usize) -> Option<Call> {
        let Token::Punct { ch: '.', pos } = &tokens[idx] else {
            return None;
        };
        let Some(Token::Ident { text, .. }) = tokens.get(idx + 1) else {
            return None;
        };
        let kind = unwrap_expect(text)?;
        let paren = has_paren(tokens, idx + 2);
        paren.then_some(Call { kind, pos: *pos })
    }

    /// UFCS 关联函数形态：`Type::unwrap(..)` / `Type::expect(..)`，与点号方法同属裸解包。
    fn ufcs_call(tokens: &[Token], idx: usize) -> Option<Call> {
        let (text, pos) = ufcs_target(tokens, idx)?;
        let kind = unwrap_expect(text)?;
        let paren = has_paren(tokens, idx + 4);
        paren.then_some(Call { kind, pos })
    }

    /// 该下标处是否为圆括号分组；方法与 UFCS 两条形态共用同一种判定。
    fn has_paren(tokens: &[Token], index: usize) -> bool {
        matches!(
            tokens.get(index),
            Some(Token::Group {
                delim: Delim::Paren,
                ..
            })
        )
    }

    /// UFCS 前缀 `Type::method`：返回方法名与位置；前缀不成立则 `None`。
    fn ufcs_target(tokens: &[Token], idx: usize) -> Option<(&str, Pos)> {
        let [
            Some(Token::Ident { .. }),
            Some(Token::Punct { ch: ':', .. }),
            Some(Token::Punct { ch: ':', .. }),
            Some(Token::Ident { text, pos }),
        ] = [
            tokens.get(idx),
            tokens.get(idx + 1),
            tokens.get(idx + 2),
            tokens.get(idx + 3),
        ]
        else {
            return None;
        };
        Some((text.as_str(), *pos))
    }

    /// 解包方法名 → 对应禁止形态；其余名字不是裸解包。
    fn unwrap_expect(text: &str) -> Option<Kind> {
        match text {
            "unwrap" => Some(Kind::Unwrap),
            "expect" => Some(Kind::Expect),
            _ => None,
        }
    }

    /// panic 类宏形态；其它宏（如 `write!`/`println!`）不算。
    fn macro_call(tokens: &[Token], idx: usize) -> Option<Call> {
        let Token::Ident { text, pos } = &tokens[idx] else {
            return None;
        };
        let kind = match text.as_str() {
            "panic" => Kind::Panic,
            "unreachable" => Kind::Unreachable,
            _ => return None,
        };
        let bang = matches!(tokens.get(idx + 1), Some(Token::Punct { ch: '!', .. }));
        bang.then_some(Call { kind, pos: *pos })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bare_unwrap_expect_and_panic_are_caught() {
            let src = "fn f() { let a = x.unwrap(); let b = y.expect(\"m\"); panic!(\"p\"); }";
            let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
            assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect, Kind::Panic]);
        }

        /// `unreachable!` 是 panic 路径但没有 lint 覆盖：漏掉它，「零 panic」就只覆盖一部分。
        #[test]
        fn unreachable_is_caught_but_its_plain_identifier_is_not() {
            let src = "fn f() { let unreachable = 1; g(unreachable); unreachable!(\"x\"); }";
            let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
            assert_eq!(kinds, vec![Kind::Unreachable], "只有带 `!` 的那次是宏调用");
        }

        #[test]
        fn unwrap_inside_a_string_is_not_flagged() {
            let src = "let s = \"x.unwrap()\";";
            assert!(scan(src).unwrap().is_empty());
        }

        #[test]
        fn ufcs_unwrap_expect_are_caught() {
            let src = "fn f() { let a = Option::unwrap(x); let b = Result::expect(y, \"m\"); }";
            let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
            assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect]);
        }

        #[test]
        fn ordinary_associated_function_is_not_flagged() {
            let src = "fn f() { let v = Type::new(x); }";
            assert!(scan(src).unwrap().is_empty());
        }

        #[test]
        fn result_returning_code_is_clean() {
            let src = "fn f() -> Result<u32, E> { let v = ok()?; Ok(v) }";
            assert!(scan(src).unwrap().is_empty());
        }
    }
}

/// 文档漂移：markdown 里相对链接的目标文件是否存在。
///
/// 纯函数：提取文档中的相对链接（排除 http/https/mailto 与纯锚点），由调用方核对目标文件。
/// 不自己读盘，因此在测试里用任意虚拟文档即可验证，不依赖仓库布局。
pub mod docs {
    /// 一个待校验的相对链接：目标路径与所在行号。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Link {
        /// 目标（去掉 `#anchor` 部分）。
        pub target: String,
        /// 文档内行号（从 1 起）。
        pub line: usize,
    }

    /// 提取全部相对链接。
    #[must_use]
    pub fn links(text: &str) -> Vec<Link> {
        let mut out = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            line_links(line, line_no + 1, &mut out);
        }
        out
    }

    /// 一行的全部相对链接；外部链接、纯锚点与空路径都不计。
    fn line_links(line: &str, line_no: usize, out: &mut Vec<Link>) {
        let mut rest = line;
        while let Some(open) = rest.find("](") {
            let after = &rest[open + 2..];
            let Some(close) = after.find(')') else {
                return;
            };
            let raw = after[..close].trim();
            let target = raw.split('#').next().unwrap_or(raw).trim();
            if is_relative(raw) && !target.is_empty() {
                out.push(Link {
                    target: target.to_string(),
                    line: line_no,
                });
            }
            rest = &after[close + 1..];
        }
    }

    fn is_relative(raw: &str) -> bool {
        let external = ["http://", "https://", "mailto:", "ftp://"];
        !external.iter().any(|prefix| raw.starts_with(prefix)) && !raw.starts_with('#')
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn relative_links_are_collected_and_anchors_stripped() {
            let doc = "见 [架构](ARCHITECTURE.md#二) 与 [代码](../crates/tool/src/lib.rs)。";
            assert_eq!(
                links(doc),
                vec![
                    Link {
                        target: "ARCHITECTURE.md".to_string(),
                        line: 1
                    },
                    Link {
                        target: "../crates/tool/src/lib.rs".to_string(),
                        line: 1
                    },
                ]
            );
        }

        #[test]
        fn external_and_pure_anchor_links_are_skipped() {
            let doc = "[site](https://x.io/a) [top](#section) [mail](mailto:a@b.c)";
            assert!(links(doc).is_empty());
        }
    }
}

/// 单文件行数棘轮。
///
/// 重复块检测**不在本模块**：主门禁用 jscpd（`.jscpd.json` 的 `threshold` 重复率硬阈值），
/// 语义巡检用 `cargo dupes`（AST 归一化识别近重复）。这里只保留两者都不管的单文件行数上限。
/// 纯函数：调用方传入文本，超限时返回实际行数，不读盘、不打印。
pub mod dup {
    use super::{Gate, Metric};

    /// 文件行数超限时返回实际行数；限值由门禁给出，本函数不持有阈值。
    #[must_use]
    pub fn file_lines(text: &str, gate: &Gate) -> Option<usize> {
        let lines = text.lines().count();
        (lines > gate.limit(Metric::Lines)).then_some(lines)
    }

    #[cfg(test)]
    mod tests {
        use diagnostic::data::{Table, Value};

        use super::*;

        /// 一份只有阈值的 `[gate]`：行数上限故意收紧到 10，便于用短文本验证边界。
        fn gate() -> Gate {
            let thresholds = vec![
                ("ccn_max", Value::int(5)),
                ("nloc_max", Value::int(60)),
                ("file_max_lines", Value::int(10)),
            ];
            Gate::from_table(&Table::from_pairs(vec![("gate", Value::table(thresholds))]))
                .expect("测试阈值合法")
        }

        #[test]
        fn an_oversized_file_is_flagged() {
            let text = (0..12).map(|_| "x").collect::<Vec<_>>().join("\n");
            assert_eq!(file_lines(&text, &gate()), Some(12));
        }

        #[test]
        fn a_file_within_budget_is_silent() {
            let text = (0..8).map(|_| "x").collect::<Vec<_>>().join("\n");
            assert_eq!(file_lines(&text, &gate()), None);
        }
    }
}
