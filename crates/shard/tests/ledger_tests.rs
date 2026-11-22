use super::*;
use cutile::builtin::Catalogue;

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

/// K14：设备执行一个都拿不到的档案，每一格要么有见证、要么自报"本机起不了前端"。
/// 口径：`Declared` 格只允许出现在整行零见证的档案上；行里有见证而某原语没见证 = 真漏洞。
#[test]
fn no_provided_cell_falls_back_to_declaration_while_the_row_has_witnesses() {
    let catalogue = catalogue();
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-ledger-test"));
    for row in ledger(&catalogue, &dirs) {
        if row.host == 0 {
            continue;
        }
        for cell in &row.cells {
            assert_ne!(
                cell.status,
                Status::Declared,
                "档案 {vendor} 行内已有 {host} 条 host 见证，原语 {prim} 却停在只有自述：{note}",
                vendor = row.profile.meta.vendor,
                host = row.host,
                prim = cell.prim,
                note = cell.note
            );
        }
    }
}
