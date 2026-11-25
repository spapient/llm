use super::*;
use crate::fixtures::catalogue;
use crate::witness::Basis;
use cutile::atom::Prim;

fn dirs() -> Dirs {
    Dirs::under(std::env::temp_dir().join("cutile-doctor-tests"))
}

/// 独立缓存目录：`--why` 会读真目录，不能与其他用例共用。
fn cache_dirs(name: &str) -> Dirs {
    let root = std::env::temp_dir().join(format!("cutile-why-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    Dirs::under(root)
}

#[test]
fn doctor_reports_the_full_chain() {
    let report = doctor(&catalogue(), &dirs(), "gemv_i4", "cpu").expect("doctor 应成功");
    assert_eq!(report.path.as_ref().expect("有路径").variant, "pipe2");
    assert!(!report.layouts.is_empty(), "{report:?}");
    let source = report.source.as_ref().expect("有 emit 产物");
    assert!(source.fragment.lines > 0);
    assert!(source.fragment.within_limit(), "{source:?}");
    let hex = source.key.hex();
    assert_eq!(hex.len(), 16, "缓存键是 16 位十六进制");
    assert!(hex.chars().all(|digit| digit.is_ascii_hexdigit()));
}

#[test]
fn doctor_reports_each_variant_as_data() {
    let report = doctor(&catalogue(), &dirs(), "gemv_i4", "cpu").expect("doctor 应成功");
    assert!(report.no_path.is_none(), "{report:?}");
    assert_eq!(report.variants.len(), 2);
    assert!(
        report
            .variants
            .iter()
            .all(|check| check.failures.is_empty())
    );
}

#[test]
fn doctor_explains_why_a_gpu_is_rejected() {
    // `ascend` 的 P0 原语仍未验证：作为证据不足样本。
    let report = doctor(&catalogue(), &dirs(), "gemv_i4", "ascend").expect("仍应返回报告");
    assert!(report.path.is_none(), "{report:?}");
    let no_path = report.no_path.as_ref().expect("失败要带回诊断");
    assert!(no_path.to_string().contains("L2"), "{no_path}");
}

/// 协商用的是这次执行到的证据：条数与依据都要能对上档案。
#[test]
fn doctor_reports_the_evidence_it_ran() {
    if crate::toolchain::skip_without_host_compiler() {
        return;
    }
    let catalogue = catalogue();
    let report = doctor(&catalogue, &dirs(), "gemv_i4", "cpu").expect("doctor 应成功");
    let profile = catalogue.profile("cpu").expect("cpu 档案");
    assert_eq!(
        report.evidence.basis,
        Basis::Executed,
        "C 档案在本机就是设备"
    );
    assert!(
        report.evidence.failure.is_none(),
        "证据不成立：{:?}",
        report.evidence.failure
    );
    assert_eq!(
        report.evidence.device,
        profile.provides.len(),
        "设备就在本机：每一条已声明的见证都要在这次证据里（不按 `grade` 筛）"
    );
    assert_eq!(report.path.as_ref().expect("有路径").variant, "pipe2");
}

/// 本机执行不了的设备：依据必须写成"档案自述"，不能假装执行过。
/// 用 ascend（两台机都缺席）而非 nvidia：后者在 CUDA 真机上设备执行，依据是 Executed。
#[test]
fn a_device_that_cannot_run_here_is_negotiated_on_declarations() {
    let report = doctor(&catalogue(), &dirs(), "gemv_i4", "ascend").expect("仍应返回报告");
    assert_eq!(report.evidence.basis, Basis::Declared);
    assert_eq!(report.evidence.device, 0, "本机没有设备执行");
    assert!(
        report.evidence.host > 0,
        "自检仍要跑：见证源码算得对不对在这里是可验证的"
    );
    assert!(
        report
            .device
            .toolchain
            .as_deref()
            .is_none_or(|id| id.starts_with("ccec@")),
        "工具链身份只能是档案声明的 `ccec`，不能是宿主编译器：{:?}",
        report.device.toolchain
    );
}

/// 见证对不上时能力就不成立：同一份档案只把一处期望值写错，协商也必须全线不可行。
#[test]
fn a_witness_that_does_not_hold_removes_the_capability() {
    if crate::toolchain::skip_without_host_compiler() {
        return;
    }
    let mut profile =
        cutile::prof::Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案可解析");
    // `quant` 的见证解包 `0xF0` 得 `-4.0`：把期望值改一位，执行器必须发现。
    let quant = profile
        .provides
        .iter_mut()
        .find(|provide| provide.prim == Prim::Quant)
        .expect("cpu 有 quant");
    quant.witness.as_mut().expect("quant 有见证").expect[0] = -5.0;

    let evidence = witness::establish(&profile, &dirs());
    assert!(
        evidence.failure.is_some(),
        "见证对不上就是证据不成立：{evidence:?}"
    );
    assert!(evidence.proof(&profile).is_empty(), "一条都不该被证明");

    let op = catalogue().op("gemv_i4").expect("算子").clone();
    let error = cutile::cap::negotiate_with(&op, &profile, &evidence.proof(&profile)).unwrap_err();
    assert!(error.to_string().contains("没有这次执行的见证"), "{error}");
}

#[test]
fn storage_reports_the_archive_capacity_next_to_the_declared_traffic() {
    let report = doctor(&catalogue(), &dirs(), "gemv_i4", "cpu").expect("doctor 应成功");
    assert_eq!(report.storage.len(), 3, "三层都要报：{report:?}");
    let fast = report
        .storage
        .iter()
        .find(|row| row.space == cutile::atom::Space::Fast)
        .expect("有 FAST 行");
    assert_eq!(fast.capacity, Some(524_288), "容量取档案声明，不是中转缓冲");
    assert!(
        fast.declared
            .iter()
            .any(|usage| usage.prim == "load" && usage.bytes == 4096),
        "用量取构建期契约比较过的声明：{fast:?}"
    );
    let reg = report
        .storage
        .iter()
        .find(|row| row.space == cutile::atom::Space::Reg)
        .expect("有 REG 行");
    assert_eq!(reg.capacity, Some(65_536));
    assert!(
        reg.declared.is_empty(),
        "没有声明字节数的原语不占 REG：{reg:?}"
    );
    let gm = report
        .storage
        .iter()
        .find(|row| row.space == cutile::atom::Space::Gm)
        .expect("有 GM 行");
    assert_eq!(gm.capacity, None, "GM 不受限额，不编一个容量");
}

#[test]
fn the_planned_shape_comes_from_the_operator_model() {
    let model = crate::pipeline::model_for("gemv_i4").expect("有模型");
    let report = doctor(&catalogue(), &dirs(), "gemv_i4", "cpu").expect("doctor 应成功");
    assert_eq!(report.shape, model.shape);
    assert!(
        report.layouts.iter().all(|row| row.spec == "row"),
        "{report:?}"
    );
}

#[test]
fn why_separates_half_written_from_never_written() {
    let dirs = cache_dirs("cache-why-partial");
    let report = doctor(&catalogue(), &dirs, "gemv_i4", "cpu").expect("doctor 应成功");
    let key = report.source.expect("有产物").key;
    assert_eq!(why(&dirs, &key, &key.hex()).state(), CacheState::Match);

    let dir = dirs.cache().join(&key.hex()[..2]).join(key.hex());
    std::fs::remove_file(dir.join("source")).expect("拿掉 source");
    let partial = why(&dirs, &key, &key.hex());
    assert_eq!(partial.state(), CacheState::Partial, "{partial:?}");
    assert!(partial.entry.is_some(), "半截目录仍是一条记录：{partial:?}");
    assert!(
        partial.changes.is_empty(),
        "没有 source 不代表键不同：{partial:?}"
    );

    assert_eq!(
        why(&dirs, &key, "0000000000000000").state(),
        CacheState::Never
    );
    let _ = std::fs::remove_dir_all(dirs.cache());
}

#[test]
fn why_names_the_field_that_invalidated_the_cache() {
    let dirs = cache_dirs("cache-why");
    let report = doctor(&catalogue(), &dirs, "gemv_i4", "cpu").expect("doctor 应成功");
    let key = report.source.expect("有产物").key;

    let hit = why(&dirs, &key, &key.hex());
    assert!(hit.entry.is_some(), "doctor 已落盘：{hit:?}");
    assert!(hit.changes.is_empty(), "同一份键不应有差异：{hit:?}");

    let mut other = key.clone();
    other.tile = "GMAX=64".to_string();
    let miss = why(&dirs, &other, &key.hex());
    let fields: Vec<&str> = miss
        .changes
        .iter()
        .map(|change| change.field.as_str())
        .collect();
    assert_eq!(fields, ["tile"], "只该指出真正变了的那项：{miss:?}");
    assert_eq!(miss.current, other.hex());

    let absent = why(&dirs, &key, "0000000000000000");
    assert!(absent.entry.is_none(), "{absent:?}");
    assert!(
        !absent.changes.is_empty(),
        "没有缓存记录就不能声称无差异：{absent:?}"
    );
    let _ = std::fs::remove_dir_all(dirs.cache());
}

/// 这条用例的前提是"doctor 已经把内核源与见证程序都落了盘"，因此它跟着 `doctor` 住在执行半边：
/// 判据本身是 `cutile::cache` 的 GC，但只有跑过一条流水线才有可回收的东西。
#[test]
fn cache_gc_reclaims_through_the_cache_directory() {
    let dirs =
        Dirs::under(std::env::temp_dir().join(format!("cutile-gc-cache-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(dirs.cache());
    let catalogue = catalogue();
    let report = crate::doctor::doctor(&catalogue, &dirs, "gemv_i4", "cpu").expect("doctor");
    assert!(report.source.is_some(), "doctor 应已落盘");
    let before = cutile::cache::cache_gc(&dirs, None).expect("可回收");
    assert_eq!(
        before.limit,
        cutile::cache::CACHE_LIMIT,
        "缺省上限是命令策略"
    );
    assert!(before.before > 0, "doctor 刚落过盘：{before:?}");
    assert_eq!(before.removed, 0, "未超上限时不删：{before:?}");
    // 条目个数不再是"doctor 只落 1 条"：它现在还要把见证程序登记进缓存。
    let entries = cutile::cache::Cache::from_dirs(&dirs).entries().len();
    assert!(entries > 1, "内核源之外还有见证程序：{entries}");
    let after = cutile::cache::cache_gc(&dirs, Some(0)).expect("可回收");
    assert_eq!(after.removed, entries, "{after:?}");
    assert_eq!(after.after, 0, "{after:?}");
    let _ = std::fs::remove_dir_all(dirs.cache());
}

// ---- ledger_tests.rs ----

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
