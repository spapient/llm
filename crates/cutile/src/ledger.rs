//! `ledger`：跨档案的后端能力台账——每条供给这次有什么证据，折成一个闭集状态。
//!
//! 台账**不产生新证据**：复用 [`witness::establish`] 已经跑出的 [`Evidence`] / [`Proof`]，
//! 这里只做聚合。状态闭集与 [`Grade`] 分开：`grade` 是档案自述，[`Status`] 是这次执行到的结论，
//! 自述不能冒充设备证据。渲染归前端，本模块只给结构化结论。

use crate::builtin::Catalogue;
use crate::cap::Proof;
use crate::dirs::Dirs;
use crate::witness::{self, Basis, Evidence};
use prof::atom::{Grade, Prim, Provide};
use prof::prof::Profile;
use std::collections::BTreeMap;

/// 一条供给在本次台账里的结论。闭集全量，渲染层不得临时新造一个词。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// 设备真机执行并与标量参考对拍通过：依据是 [`Basis::Executed`]，且这条供给在执行结论里。
    Verified,
    /// 本机没有该设备：见证在 host C ABI 编过、算得对，语言门通过；**不构成设备证据**。
    SelfCheck,
    /// 只有档案自述（L0/L1，或 L2 自述但本次没在本机执行）。
    Declared,
    /// 这次执行留下了不成立的结论：编译 / 执行 / 对拍 / 语言门任一步失败。
    Failed,
    /// 这份档案根本没声明这条原语。
    Unprovided,
}

impl Status {
    /// 台账格子的短标签：同一份事实只允许这一种写法。
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Verified => "✓ 真机对拍",
            Self::SelfCheck => "自检（host C）",
            Self::Declared => "仅档案声明",
            Self::Failed => "✗ 不成立",
            Self::Unprovided => "—",
        }
    }
}

/// 台账里的一格：一台设备对一条原语的一条供给。
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerCell {
    /// 原语。
    pub prim: Prim,
    /// 实现名。
    pub impl_name: String,
    /// 档案登记的证据等级（自述，不等于本次结论）。
    pub grade: Grade,
    /// 本次执行到的结论。
    pub status: Status,
    /// 补充说明：为什么是这一档，缺的是哪一步。
    pub note: String,
}

/// 一台设备的台账行：证据依据、三档计数与逐原语格子。
#[derive(Debug, Clone, PartialEq)]
pub struct BackendRow {
    /// 档案元信息（厂商、型号、emit、登记工具链）。
    pub profile: Profile,
    /// 这次协商的依据：执行还是自述。
    pub basis: Basis,
    /// 设备执行通过的见证条数。
    pub device: usize,
    /// 本机自检通过的见证条数。
    pub host: usize,
    /// 语言门结论（人读文本）。
    pub gate: String,
    /// 本机是否有这台设备的执行前端。
    pub available: bool,
    /// 逐原语格子；未声明的原语也占一格 [`Status::Unprovided`]，台账不留空格。
    pub cells: Vec<LedgerCell>,
}

/// 跨全部档案聚合一份台账：每台设备跑一遍 [`witness::establish`]，再把供给对到格子状态。
#[must_use]
pub fn ledger(catalogue: &Catalogue, dirs: &Dirs) -> Vec<BackendRow> {
    catalogue
        .profiles()
        .iter()
        .map(|profile| row(profile, dirs))
        .collect()
}

/// 一台设备的行：证据先取一次，格子只在这份证据上分类。
fn row(profile: &Profile, dirs: &Dirs) -> BackendRow {
    let evidence = witness::establish(profile, dirs);
    let proof = evidence.proof(profile);
    let available = witness::device_available(profile);
    let gate = evidence.language.to_string();

    // 供给按原语归一次组：随后逐原语取组内条目，不再每个原语都全量过滤一遍 `provides`。
    let mut by_prim: BTreeMap<Prim, Vec<&Provide>> = BTreeMap::new();
    for have in &profile.provides {
        by_prim.entry(have.prim).or_default().push(have);
    }

    let mut cells = Vec::new();
    for prim in Prim::ALL {
        let mut declared = by_prim.get(prim).cloned().unwrap_or_default();
        if declared.is_empty() {
            cells.push(LedgerCell {
                prim: *prim,
                impl_name: String::new(),
                grade: Grade::L0,
                status: Status::Unprovided,
                note: "档案未声明".to_string(),
            });
            continue;
        }
        // 同一原语多条供给（不同实现/签名）逐条入格，不合并成一行。
        declared.sort_by(|a, b| a.impl_name.cmp(&b.impl_name));
        for have in declared {
            cells.push(cell(profile, &evidence, &proof, have, available, &gate));
        }
    }

    BackendRow {
        profile: profile.clone(),
        basis: evidence.basis,
        device: evidence.device,
        host: evidence.host,
        gate,
        available,
        cells,
    }
}

/// 一格的分类：失败先于一切；其次设备执行持有；再其次 host 自检；其余落自述。
fn cell(
    profile: &Profile,
    evidence: &Evidence,
    proof: &Proof,
    have: &Provide,
    available: bool,
    gate: &str,
) -> LedgerCell {
    let (status, note) = if let Some(failure) = &evidence.failure {
        (Status::Failed, failure.to_string())
    } else if verified(evidence, proof, have) {
        (
            Status::Verified,
            format!(
                "设备执行 {device} 条；参数 {params}",
                device = evidence.device,
                params = evidence.device_params,
            ),
        )
    } else if self_check(evidence, have) {
        (
            Status::SelfCheck,
            format!("host C ABI 对拍通过；{gate}；不含设备执行"),
        )
    } else {
        (
            Status::Declared,
            declared_note(profile, evidence, available, have),
        )
    };

    LedgerCell {
        prim: have.prim,
        impl_name: have.impl_name.clone(),
        grade: have.grade,
        status,
        note,
    }
}

/// 设备执行依据成立且这条供给在执行结论里：才记真机对拍。
fn verified(evidence: &Evidence, proof: &Proof, have: &Provide) -> bool {
    evidence.basis == Basis::Executed && proof.holds(have)
}

/// 档案自述依据、本机自检过、且这条供给带了见证：记 host 自检（不构成设备证据）。
fn self_check(evidence: &Evidence, have: &Provide) -> bool {
    evidence.basis == Basis::Declared && evidence.host > 0 && have.witness.is_some()
}

/// 自述格的理由：设备在本机却没持有、本机缺设备、还是只有 L0/L1 声明，三种说法分开。
fn declared_note(
    profile: &Profile,
    evidence: &Evidence,
    available: bool,
    have: &Provide,
) -> String {
    if available && evidence.basis == Basis::Executed {
        "设备在本机，但这条供给没进设备执行结论".to_string()
    } else if have.grade.is_verified() {
        format!(
            "档案自述 {}，本机执行不了 `{}`（emit={}）",
            have.grade, profile.meta.vendor, profile.meta.emit
        )
    } else {
        format!("{} 声明，无可执行结论", have.grade)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::Catalogue;

    fn catalogue() -> Catalogue {
        Catalogue::try_builtin().expect("内置数据必须合法")
    }

    #[test]
    fn every_profile_covers_the_whole_prim_closed_set() {
        let catalogue = catalogue();
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-ledger-test"));
        for row in ledger(&catalogue, &dirs) {
            for prim in Prim::ALL {
                assert!(
                    row.cells.iter().any(|cell| cell.prim == *prim),
                    "{} 缺原语 {prim}",
                    row.profile.meta.vendor
                );
            }
        }
    }

    #[test]
    fn verified_cells_only_come_from_executed_basis() {
        let catalogue = catalogue();
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-ledger-test"));
        for row in ledger(&catalogue, &dirs) {
            for cell in &row.cells {
                if cell.status == Status::Verified {
                    assert_eq!(row.basis, Basis::Executed, "自述依据不能记成真机对拍");
                }
            }
        }
    }

    #[test]
    fn self_check_cells_only_come_from_declared_basis() {
        let catalogue = catalogue();
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-ledger-test"));
        for row in ledger(&catalogue, &dirs) {
            for cell in &row.cells {
                if cell.status == Status::SelfCheck {
                    assert_eq!(row.basis, Basis::Declared, "自检依据不能记成设备执行");
                }
            }
        }
    }

    #[test]
    fn declared_provisions_each_get_a_cell_and_unprovided_are_marked() {
        let catalogue = catalogue();
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-ledger-test"));
        for row in ledger(&catalogue, &dirs) {
            for prim in Prim::ALL {
                let declared = row
                    .profile
                    .provides
                    .iter()
                    .filter(|have| have.prim == *prim)
                    .count();
                let matched: Vec<&LedgerCell> =
                    row.cells.iter().filter(|cell| cell.prim == *prim).collect();
                if declared == 0 {
                    assert_eq!(matched.len(), 1, "未声明的原语占一格 Unprovided");
                    assert!(
                        matched.iter().all(|cell| cell.status == Status::Unprovided),
                        "未声明的原语必须占一格 Unprovided"
                    );
                } else {
                    assert_eq!(
                        matched.len(),
                        declared,
                        "档案 {vendor} 原语 {prim}：多条供给必须逐条入格，不合并",
                        vendor = row.profile.meta.vendor
                    );
                    assert!(
                        matched.iter().all(|cell| cell.status != Status::Unprovided),
                        "已声明的供给不得落成 Unprovided"
                    );
                }
            }
        }
    }
}
