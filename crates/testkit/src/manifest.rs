//! `cargo metadata` 的事实面：工作区成员，以及成员之间的依赖边。
//!
//! 成员集合来自 `packages[].name`，边来自 `packages[].dependencies[].{name, kind}`（`kind ==
//! "dev"` 即 `+dev`）。解析交给 `serde_json`，不写第二份 JSON 文法。
//!
//! 进程内只取一次：`cargo metadata` 是一次进程启动，而九个门禁都要这份事实。

use std::collections::BTreeSet;
use std::process::Command;
use std::sync::OnceLock;

use crate::edge::{Edge, short};
use crate::source::root;

/// 工作区图：成员与成员之间的边。
pub(crate) struct Workspace {
    members: BTreeSet<String>,
    edges: BTreeSet<Edge>,
}

impl Workspace {
    /// 进程内只装载一次。
    pub(crate) fn load() -> &'static Workspace {
        static WORKSPACE: OnceLock<Workspace> = OnceLock::new();
        WORKSPACE.get_or_init(read)
    }

    /// 工作区成员（短名）。
    pub(crate) fn members(&self) -> &BTreeSet<String> {
        &self.members
    }

    /// 成员之间的依赖边（两端都是短名）。
    pub(crate) fn edges(&self) -> &BTreeSet<Edge> {
        &self.edges
    }
}

fn read() -> Workspace {
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
        ])
        .arg(root().join("Cargo.toml"))
        .output()
        .expect("`cargo metadata` 无法启动");
    assert!(
        output.status.success(),
        "`cargo metadata` 失败：{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("`cargo metadata` 的输出不是 JSON");
    let packages = metadata["packages"]
        .as_array()
        .expect("`cargo metadata` 的输出不含 `packages[]`");
    let mut members = BTreeSet::new();
    let mut edges = BTreeSet::new();
    for package in packages {
        let from = short(package["name"].as_str().expect("每个包都有名字"));
        members.insert(from.clone());
        let dependencies = package["dependencies"].as_array().into_iter().flatten();
        for dependency in dependencies {
            edges.insert(Edge {
                from: from.clone(),
                to: short(dependency["name"].as_str().expect("每条依赖都有名字")),
                dev: dependency["kind"].as_str() == Some("dev"),
            });
        }
    }
    assert!(!members.is_empty(), "工作区没有成员");
    Workspace { members, edges }
}
