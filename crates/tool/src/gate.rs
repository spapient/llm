//! 质量门配置：从项目 `lints/vocabulary.toml` 加载强类型阈值与存量棘轮。
//!
//! 与 burn 的 Config 同思想：标准是数据，行为是纯函数。阈值与存量债都在配置里，分析器不持有
//! 魔数。棘轮只准改善：新增超标直接失败，登记的存量项只准降，降到阈值后必须删除其基线条目。

use std::collections::BTreeMap;

use config::toml;
use diagnostic::{Code, Error, Result};

/// 静态质量门：阈值与存量债基线。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gate {
    /// 圈复杂度上限。
    pub max_ccn: usize,
    /// 函数体有效行数上限。
    pub max_nloc: usize,
    /// SIMD 微核的函数长度上限。
    pub simd_nloc: usize,
    /// 单文件行数硬上限。
    pub max_file_lines: usize,
    pub(crate) debt: BTreeMap<String, Debt>,
}

/// 一个存量债项的登记度量（只登记仍超标的字段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Debt {
    /// 登记的圈复杂度。
    pub ccn: Option<usize>,
    /// 登记的函数长度。
    pub nloc: Option<usize>,
    /// 登记的文件行数。
    pub lines: Option<usize>,
    /// 登记的禁止形态数量。
    pub call: Option<usize>,
}

/// 度量类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// 一个当前观测到的超标项：身份、度量、当前值、目标阈值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Over {
    /// 身份键（文件用路径，函数用 `路径:名#序号`）。
    pub key: String,
    /// 度量类别。
    pub metric: Metric,
    /// 当前值。
    pub value: usize,
    /// 目标阈值。
    pub limit: usize,
}

impl Gate {
    /// 解析编译期嵌入的项目配置。
    pub fn load() -> Result<Self> {
        let root = toml::parse(include_str!("../../../lints/vocabulary.toml")).map_err(|err| {
            Error::new(Code::Invalid, format!("vocabulary.toml: {err}")).with("tool::gate")
        })?;
        let table = root.table("gate").ok_or_else(|| {
            Error::new(Code::Invalid, "vocabulary.toml 缺 [gate]").with("tool::gate")
        })?;
        let mut gate = Self {
            max_ccn: int(table, "ccn_max")?,
            max_nloc: int(table, "nloc_max")?,
            simd_nloc: int(table, "simd_nloc")?,
            max_file_lines: int(table, "file_max_lines")?,
            debt: BTreeMap::new(),
        };
        for row in table.table_array("debt").unwrap_or_default() {
            let key = row
                .string("key")
                .ok_or_else(|| Error::new(Code::Invalid, "[gate.debt] 缺 key").with("tool::gate"))?
                .to_string();
            let debt = Debt {
                ccn: opt_int(row, "ccn"),
                nloc: opt_int(row, "nloc"),
                lines: opt_int(row, "lines"),
                call: opt_int(row, "call"),
            };
            if debt == Debt::default() {
                return Err(
                    Error::new(Code::Invalid, format!("[gate.debt] {key} 无度量"))
                        .with("tool::gate"),
                );
            }
            gate.debt.insert(key, debt);
        }
        gate.check_thresholds()?;
        Ok(gate)
    }

    /// 棘轮判定一个超标项；返回 Some(理由) 即门禁失败。
    #[must_use]
    pub fn check(&self, over: &Over) -> Option<String> {
        let Some(budget) = self.debt.get(&over.key) else {
            return Some(format!(
                "{}: 新增 {:?}={}",
                over.key, over.metric, over.value
            ));
        };
        let allowed = match over.metric {
            Metric::Ccn => budget.ccn,
            Metric::Nloc => budget.nloc,
            Metric::Lines => budget.lines,
            Metric::Call => budget.call,
        };
        let Some(allowed) = allowed else {
            return Some(format!(
                "{}: 新增 {:?}={}",
                over.key, over.metric, over.value
            ));
        };
        if over.value > allowed {
            return Some(format!(
                "{}: {:?} {} 恶化 > 登记 {allowed}",
                over.key, over.metric, over.value
            ));
        }
        if over.value <= over.limit {
            return Some(format!(
                "{}: {:?} 已达标，删除基线条目",
                over.key, over.metric
            ));
        }
        None
    }

    /// 基线条目是否仍有对应超标；没有则应删除，避免永久豁免。
    #[must_use]
    pub fn stale(&self, keys: &std::collections::BTreeSet<String>) -> Vec<String> {
        self.debt
            .keys()
            .filter(|key| !keys.contains(*key))
            .map(|key| format!("{key}: 已全部达标，删除基线条目"))
            .collect()
    }

    fn check_thresholds(&self) -> Result<()> {
        let positive = [
            self.max_ccn,
            self.max_nloc,
            self.simd_nloc,
            self.max_file_lines,
        ];
        if positive.contains(&0) {
            return Err(Error::new(Code::Invalid, "[gate] 阈值必须为正").with("tool::gate"));
        }
        if self.simd_nloc < self.max_nloc {
            return Err(Error::new(Code::Invalid, "simd_nloc 不得小于 nloc_max").with("tool::gate"));
        }
        Ok(())
    }
}

fn int(table: &toml::Table, key: &str) -> Result<usize> {
    table
        .int(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            Error::new(Code::Invalid, format!("[gate] 缺正整数 {key}")).with("tool::gate")
        })
}

fn opt_int(table: &toml::Table, key: &str) -> Option<usize> {
    table.int(key).and_then(|value| usize::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_project_config_loads_with_baseline() {
        let gate = Gate::load().expect("项目配置合法");
        assert_eq!(gate.max_ccn, 5);
        assert!(!gate.debt.is_empty(), "存量基线被加载");
    }

    #[test]
    fn ratchet_blocks_new_and_worse_allows_within_budget() {
        let gate = Gate::load().unwrap();
        // 取一个真实的 ccn 基线 key 验证三种判定。
        let (key, debt) = gate.debt.iter().find(|(_, d)| d.ccn.is_some()).unwrap();
        let (metric, allowed) = (Metric::Ccn, debt.ccn.unwrap());
        let limit = gate.max_ccn;
        // 恶化：超过登记值。
        assert!(
            gate.check(&Over {
                key: key.clone(),
                metric,
                value: allowed + 1,
                limit
            })
            .is_some()
        );
        // 登记范围内：通过（当前值仍超阈值）。
        if allowed > limit {
            assert!(
                gate.check(&Over {
                    key: key.clone(),
                    metric,
                    value: allowed,
                    limit
                })
                .is_none()
            );
        }
        // 全新身份：直接失败。
        assert!(
            gate.check(&Over {
                key: "crates/x/src/y.rs:new#0".into(),
                metric,
                value: 9,
                limit
            })
            .is_some()
        );
    }
}
