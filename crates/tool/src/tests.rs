use super::*;

#[test]
fn report_pads_labels_to_the_widest_row() {
    let report = Report::new("标题").row("op", "gemv").row("device", "cpu");
    assert_eq!(report.len(), 2);
    assert!(!report.is_empty());
    let text = report.render();
    assert!(text.starts_with("标题\n"));
    let rows: Vec<&str> = text.lines().skip(1).collect();
    assert!(
        rows.iter()
            .any(|row| row.contains("op") && row.contains("gemv"))
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("device") && row.contains("cpu"))
    );
}

#[test]
fn report_with_only_title_is_empty() {
    assert!(Report::new("空").is_empty());
    assert_eq!(Report::new("空").render(), "空\n");
}

#[test]
fn join_names_separates_with_dunhao_and_dashes_empty() {
    assert_eq!(join_names(std::iter::empty::<&str>()), "—");
    assert_eq!(join_names(["a", "b"].iter().copied()), "a、b");
}
