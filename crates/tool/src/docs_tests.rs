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
