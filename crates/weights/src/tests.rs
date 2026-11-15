use super::*;

fn entry(name: &str, offset: usize) -> Entry {
    Entry::new(name, DType::F32, Shape::new([2, 3])).at(offset)
}

#[test]
fn a_valid_header_sorts_entries_and_reports_the_data_span() {
    let header = Header::new(vec![entry("b", 88), entry("a", 64)], 64).expect("合法");
    let names: Vec<&str> = header.entries().iter().map(Entry::name).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(header.data_start(), 64);
    assert_eq!(header.data_bytes(), 48);
}

#[test]
fn duplicate_names_are_rejected() {
    let error = Header::new(vec![entry("a", 64), entry("a", 88)], 64).expect_err("重名");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("重复"), "{error}");
}

#[test]
fn overlapping_intervals_are_rejected() {
    let error = Header::new(vec![entry("a", 64), entry("b", 80)], 64).expect_err("重叠");
    assert!(error.to_string().contains("重叠"), "{error}");
}

#[test]
fn an_entry_before_the_data_section_is_rejected() {
    let error = Header::new(vec![entry("a", 8)], 64).expect_err("早于数据段");
    assert!(error.to_string().contains("数据段"), "{error}");
}

#[test]
fn a_zero_sized_entry_is_allowed_and_costs_nothing() {
    let empty = Entry::new("empty", DType::U4, Shape::new([0])).at(64);
    assert!(empty.is_empty());
    let header = Header::new(vec![empty], 64).expect("合法");
    assert_eq!(header.data_bytes(), 0);
    assert!(header.get("missing").is_none());
}
