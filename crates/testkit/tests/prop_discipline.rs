//! 性质测试的**工作区纪律**：[`crate::prop`] 是唯一配置来源，谁都不能自建一套。
//!
//! [`super::prop_config`] 验"装置本身对不对"，这里验"有没有人绕过装置"。
//! 分两份是因为前者改装置时红、后者改任何 crate 时红，故障定位不同。
//!
//! 为什么值得一道门：`nn` 与 `algorithm` 曾经各有一份手抄的 `ProptestConfig`，
//! 两份的 `cases` 不同、`max_shrink_iters` 也不同，于是同一性质在两个 crate 里
//! 的失败率与最小反例都不一样。把配置收成一处只是修好当下；让第二处**无法出现**
//! 才是纪律。手抄配置不会让任何测试变红——它只会让性质测试静默地变弱。

use crate::architecture::{rust_files, source_text, src_root, workspace, workspace_root};
use std::path::{Path, PathBuf};

/// 唯一允许出现 `ProptestConfig` 构造的位置。
const HARNESS: &str = "crates/testkit/src/prop.rs";

fn relative(path: &Path) -> String {
    let root = workspace_root();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    path.strip_prefix(&root)
        .unwrap_or(&path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// 成员名 → 需要扫描的源码目录（`src`、`tests`、`benches`，存在才扫）。
fn scan_dirs(member: &str) -> Vec<PathBuf> {
    let src = src_root(member);
    let Some(manifest_dir) = src.parent() else {
        return Vec::new();
    };
    ["src", "tests", "benches"]
        .iter()
        .map(|name| manifest_dir.join(name))
        .filter(|dir| dir.is_dir())
        .collect()
}

/// 成员名 → 该成员全部 Rust 源文件的 `(相对路径, 文本)`。
fn member_sources(member: &str) -> Vec<(String, String)> {
    scan_dirs(member)
        .iter()
        .flat_map(|dir| rust_files(dir))
        .filter_map(|path| source_text(&path).map(|text| (relative(&path), text)))
        .collect()
}

/// 成员名 → 清单文本；读不到时返回 `None`（虚拟成员没有清单）。
fn manifest_text(member: &str) -> Option<String> {
    let manifest = src_root(member).parent()?.join("Cargo.toml");
    source_text(&manifest)
}

/// 工作区里声明了 `proptest` 的成员。
fn property_crates() -> Vec<String> {
    let (members, _) = workspace();
    members
        .iter()
        .filter(|member| manifest_text(member).is_some_and(|text| text.contains("proptest")))
        .cloned()
        .collect()
}

/// 去掉字符串字面量与注释，只留下真正的代码。
///
/// 剥字符串不是为了更聪明，是为了让这道门**可以讨论自己**：这个文件的断言语里就写着
/// `"ProptestConfig {"`，不剥的话门会被自己的说明文字打红，然后就有人去把它改成空转。
fn code_of(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(current) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if current == '\\' {
                escaped = true;
            } else if current == '"' {
                in_string = false;
            }
            continue;
        }
        if current == '"' {
            in_string = true;
            continue;
        }
        if current == '/' && matches!(chars.clone().next(), Some('/') | Some('*')) {
            break;
        }
        out.push(current);
    }
    out
}

/// 只认**构造**，不认提及：散文里写 `ProptestConfig` 是合法的，
/// 真绕过装置的是 `ProptestConfig { .. }` 或 `ProptestConfig::default()`。
fn constructs_a_config(line: &str) -> bool {
    let code = code_of(line);
    code.contains("ProptestConfig {")
        || code.contains("ProptestConfig{")
        || code.contains("ProptestConfig::")
}

#[test]
fn no_crate_hand_rolls_its_own_proptest_config() {
    let (members, _) = workspace();
    let mut offenders = Vec::new();
    for member in &members {
        for (path, text) in member_sources(member) {
            if path == HARNESS {
                continue;
            }
            for (index, line) in text.lines().enumerate() {
                if constructs_a_config(line) {
                    offenders.push(format!("{path}:{}", index + 1));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "这些位置自建了 `ProptestConfig`，请改用 `testkit::prop::config()`：{offenders:?}\n\
         手抄配置不会让测试变红，只会让性质测试静默变弱（样本数与收缩预算各不相同）"
    );
}

#[test]
fn every_property_crate_depends_on_the_harness() {
    let mut offenders = Vec::new();
    for member in property_crates() {
        let Some(manifest) = manifest_text(&member) else {
            continue;
        };
        if !manifest.contains("testkit") {
            offenders.push(member);
        }
    }
    assert!(
        offenders.is_empty(),
        "这些成员声明了 `proptest` 却没有 `testkit` 依赖，因此拿不到共享配置：{offenders:?}"
    );
}

#[test]
fn every_property_crate_reaches_the_shared_configuration() {
    let mut offenders = Vec::new();
    for member in property_crates() {
        // 装置自己的 lib 内部写 `crate::prop`，只有它免检。
        if member == "testkit" {
            continue;
        }
        let reached = member_sources(&member)
            .iter()
            .any(|(_, text)| text.contains("testkit::prop"));
        if !reached {
            offenders.push(member);
        }
    }
    assert!(
        offenders.is_empty(),
        "这些成员有 `testkit` 依赖却从不调用 `testkit::prop::*`，性质测试没走共享装置：{offenders:?}"
    );
}

/// 纪律门的自检：扫描面必须真的覆盖到已经收口的四个 crate。
///
/// 没有这条，"扫描恒返回空集"这种失效会和"纪律完好"长得一模一样——
/// `mirror` 的注解扫描就因为写死 `crate_name == "op"` 而在拆分后静默归零过。
#[test]
fn the_scan_actually_sees_the_property_crates() {
    let found = property_crates();
    for expected in crate::prop::ledger::COVERED {
        assert!(
            found.contains(&expected.to_string()),
            "扫描面漏了 `{expected}`：台账说它有性质测试，扫描却没在清单里看到 proptest，\
             说明这条路没有真的读文件"
        );
    }
    let total: usize = found
        .iter()
        .map(|member| member_sources(member).len())
        .sum();
    assert!(total > 0, "一个源文件都没读到，扫描面是空的");
}

/// 台账不能有个成员落空：工作区成员集与"已覆盖 ∪ 已登记缺口 ∪ 有理由缺席"逐成员相等。
///
/// 单向检查（"台账里的成员都存在"）没意义——漏掉一个成员时它同样通过。
#[test]
fn the_ledger_accounts_for_every_member_exactly_once() {
    use crate::prop::ledger::{ABSENT, COVERED, PLANNED};
    let (members, _) = workspace();
    let mut accounted = std::collections::BTreeSet::new();
    let mut duplicates = Vec::new();
    let mut entries: Vec<&str> = COVERED.to_vec();
    entries.extend(PLANNED.iter().map(|(member, _)| *member));
    entries.extend(
        ABSENT
            .iter()
            .flat_map(|absence| absence.members.iter().copied()),
    );
    for member in entries {
        if !accounted.insert(member) {
            duplicates.push(member);
        }
    }
    assert!(
        duplicates.is_empty(),
        "台账里重复登记的成员：{duplicates:?}"
    );
    let unaccounted: Vec<&str> = members
        .iter()
        .map(String::as_str)
        .filter(|member| !accounted.contains(member))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "这些成员在台账里既不在 COVERED、也不在 PLANNED、也没有缺席理由：{unaccounted:?}\n\
         新成员必须归一次类：写性质测试，或说清它由什么替代覆盖"
    );
    let ghosts: Vec<&str> = accounted
        .iter()
        .copied()
        .filter(|member| !members.iter().any(|real| real == member))
        .collect();
    assert!(
        ghosts.is_empty(),
        "台账里有工作区中不存在的成员：{ghosts:?}"
    );
}

/// 台账与源码必须**双向**一致：说覆盖的真的有性质测试块，说没覆盖的真的没有。
#[test]
fn the_ledger_agrees_with_the_sources() {
    use crate::prop::ledger::{ABSENT, COVERED, PLANNED};
    let has_block = |member: &str| {
        member_sources(member)
            .iter()
            .any(|(_, text)| text.contains("proptest!"))
    };
    let missing: Vec<&str> = COVERED
        .iter()
        .copied()
        .filter(|member| !has_block(member))
        .collect();
    assert!(
        missing.is_empty(),
        "台账说这些成员有性质测试，但源码里找不到性质测试块：{missing:?}"
    );
    let mut covered_but_absent = Vec::new();
    for member in PLANNED.iter().map(|(member, _)| *member) {
        if has_block(member) {
            covered_but_absent.push(member);
        }
    }
    for member in ABSENT
        .iter()
        .flat_map(|absence| absence.members.iter().copied())
    {
        if has_block(member) {
            covered_but_absent.push(member);
        }
    }
    assert!(
        covered_but_absent.is_empty(),
        "这些成员已经写了性质测试，台账却还把它们记作\"不需要\"或\"待补\"：{covered_but_absent:?}\n\
         台账落后于代码，就等于没有台账"
    );
}

/// 缺席理由必须**具体**：太短或彼此重复的理由无法指导下一个成员归哪类。
#[test]
fn every_absence_reason_is_specific() {
    use crate::prop::ledger::ABSENT;
    let mut reasons: Vec<&str> = Vec::new();
    for absence in ABSENT {
        assert!(
            !absence.members.is_empty(),
            "有一条缺席理由没挂任何成员，它是死规则：{}",
            absence.reason
        );
        let length = absence.reason.chars().count();
        assert!(
            length >= 40,
            "缺席理由太短（{length} 字），不足以说明替代覆盖在哪：{}",
            absence.reason
        );
        assert!(
            !reasons.contains(&absence.reason),
            "两条缺席理由一字不差，说明其中一条是复制来的：{}",
            absence.reason
        );
        reasons.push(absence.reason);
    }
}

/// 已登记缺口只能减。允许"还没写"，不允许"越积越多"。
#[test]
fn the_known_gaps_are_capped() {
    use crate::prop::ledger::{PLANNED, PLANNED_MAX};
    assert!(
        PLANNED.len() <= PLANNED_MAX,
        "已登记缺口涨到 {} 条（上限 {PLANNED_MAX}）：先补上一条，再加新的",
        PLANNED.len()
    );
    for (member, reason) in PLANNED {
        assert!(
            reason.chars().count() >= 40,
            "`{member}` 的缺口说明太短，读的人不知道要补什么：{reason}"
        );
    }
}

/// 清单里声明了 `proptest` 的成员集与台账的已覆盖集必须相等。
#[test]
fn manifest_declarations_match_the_ledger() {
    use crate::prop::ledger::COVERED;
    let mut declared = property_crates();
    let mut covered: Vec<String> = COVERED.iter().map(|member| (*member).to_string()).collect();
    declared.sort();
    covered.sort();
    assert_eq!(
        declared, covered,
        "清单声明与台账不一致：加了 `proptest` 就要登记进 COVERED，删了就要改理由"
    );
}
