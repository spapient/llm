use super::*;

testkit::test_cases! {
fn panel_pads_labels_to_the_widest_row() {
    let panel = Panel::new("标题").row("op", "gemv").row("device", "cpu");
    let text = panel.render();
    assert_eq!(text.lines().count(), 3);
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

/// 中英混排的面板按**显示列**对齐，不按字符数：「条目数」3 个字符占 6 列，与 10 列的
/// `V_artifact` 并排时补 4 个空格才对齐。按 `chars().count()` 会补 7 个，两行的值列差 3 列。
fn panel_aligns_east_asian_labels_by_display_columns() {
    let panel = Panel::new("t").row("条目数", "1").row("V_artifact", "2");

    assert_eq!(panel.render(), "t\n  条目数      1\n  V_artifact  2\n");
}

/// 数学符号属歧义宽度，终端按一列排：`×` 不能让整行多出一列。
fn ambiguous_width_symbols_take_one_column() {
    assert_eq!(span("a×b"), 3);
    assert_eq!(span("条目数"), 6);
    assert_eq!(span("MiB"), 3);
}

fn panel_with_only_title_renders_one_line() {
    assert_eq!(Panel::new("空").render(), "空\n");
}

fn sheet_writes_header_separator_and_rows_in_one_place() {
    let grid = Sheet::new(["原语", "实现", "次数"])
        .align([Align::Left, Align::Left, Align::Right])
        .row(["`gemv`", "i8", "3"])
        .row(["`gemm`", "f32", "7"]);
    assert_eq!(
        grid.render(),
        "| 原语 | 实现 | 次数 |\n|---|---|---:|\n| `gemv` | i8 | 3 |\n| `gemm` | f32 | 7 |\n"
    );
}

fn sheet_pads_short_rows_and_truncates_long_ones() {
    let grid = Sheet::new(["a", "b"]).row(["only"]).row(["x", "y", "z"]);
    let text = grid.render();
    assert!(text.contains("| only |  |\n"), "{text}");
    assert!(text.contains("| x | y |\n"), "{text}");
}

fn join_names_separates_with_dunhao_and_dashes_empty() {
    assert_eq!(join_names(std::iter::empty::<&str>()), "—");
    assert_eq!(join_names(["a", "b"].iter().copied()), "a、b");
}

fn byte_sizes_are_printable_next_to_the_fast_capacity() {
    assert_eq!(human_bytes(148), "148 B");
    assert_eq!(human_bytes(10_485_760), "10.0 MiB");
    assert_eq!(human_bytes(524_288), "512.0 KiB");
}

fn a_sub_microsecond_kernel_is_reported_in_nanoseconds() {
    assert_eq!(per_run(1.9), "1.9 ns/次");
    assert_eq!(per_run(13_300_000.0), "13.300 ms/次");
}
}
