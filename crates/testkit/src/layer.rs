//! 层表与依赖边：`docs/ARCHITECTURE.md` 的「层表」与「允许的边」两节的可执行版本。
//!
//! 四道判据各自守一侧：成员集合逐项相等（不多不少）、边集合逐项相等（方向与 `+dev` 都在内）、
//! 每条边只指向更低层或同层左侧、每个成员至少被用或被用。任一方向漂移都会在这里红。

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeSet;

    use crate::architecture::tests::{Edge, contract, declared_edges, workspace};

    #[test]
    fn the_layer_table_lists_exactly_the_workspace_members() {
        let (members, _) = workspace();
        let contract = contract();
        let listed: BTreeSet<String> = contract
            .layers
            .keys()
            .chain(contract.tools.iter())
            .cloned()
            .collect();
        assert_eq!(
            members,
            listed,
            "层表与 Cargo.toml 的成员不一致：多 {:?}，缺 {:?}",
            members.difference(&listed).collect::<Vec<_>>(),
            listed.difference(&members).collect::<Vec<_>>()
        );
        let overlap: Vec<&String> = contract
            .tools
            .iter()
            .filter(|tool| contract.layers.contains_key(*tool))
            .collect();
        assert!(
            overlap.is_empty(),
            "工具成员不能同时出现在运行时层里：{overlap:?}"
        );
    }

    #[test]
    fn the_declared_edges_are_exactly_the_workspace_edges() {
        let (_, actual) = workspace();
        let (declared, total, dev) = declared_edges();
        let missing: Vec<&Edge> = actual.difference(&declared).collect();
        let extra: Vec<&Edge> = declared.difference(&actual).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "文档与 cargo metadata 的边集不一致：文档缺 {missing:?}，文档多 {extra:?}"
        );
        assert_eq!(
            actual.len(),
            total,
            "文档声称共 {total} 条边，实际 {} 条",
            actual.len()
        );
        let actual_dev = actual.iter().filter(|edge| edge.dev).count();
        assert_eq!(
            actual_dev, dev,
            "文档声称 {dev} 条 +dev 边，实际 {actual_dev} 条"
        );
    }

    #[test]
    fn every_edge_points_to_a_lower_layer_or_to_the_left() {
        let (_, edges) = workspace();
        let contract = contract();
        for edge in &edges {
            // 工具成员没有运行时层级，它们的边不参与方向判定（ARCHITECTURE §一）。
            // `+dev` 边只服务测试/夹具，不进任何产品二进制，同样不约束运行时分层方向。
            if edge.dev || contract.tools.contains(&edge.from) || contract.tools.contains(&edge.to)
            {
                continue;
            }
            let from = contract
                .layers
                .get(&edge.from)
                .unwrap_or_else(|| panic!("层表缺成员 {}", edge.from));
            let to = contract
                .layers
                .get(&edge.to)
                .unwrap_or_else(|| panic!("层表缺成员 {}", edge.to));
            assert!(
                to.0 < from.0 || (to.0 == from.0 && to.1 < from.1),
                "{} → {} 违反依赖方向：只能指向更低的层或同层更靠左的成员",
                edge.from,
                edge.to
            );
        }
    }

    #[test]
    fn every_member_is_consumed_or_consumes_something() {
        let (members, edges) = workspace();
        let contract = contract();
        let used: BTreeSet<&String> = edges
            .iter()
            .flat_map(|edge| [&edge.from, &edge.to])
            .collect();
        for member in &members {
            if contract.tools.contains(member) {
                continue;
            }
            assert!(
                used.contains(member),
                "{member} 没有任何依赖边：它只是一个目录，不是模块（ARCHITECTURE §五）"
            );
        }
    }
}
