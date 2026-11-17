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

#[test]
fn decode_f32_unpacks_little_endian_words() {
    let bytes = [0u8, 0, 128, 63, 0, 0, 0, 0];
    let values = decode_f32(&bytes).expect("完整 f32 段");
    assert_eq!(values, [1.0, 0.0]);
}

#[test]
fn decode_f32_rejects_a_partial_tail_word() {
    let error = decode_f32(&[0u8, 1, 2]).expect_err("余字节");
    assert_eq!(error.code(), Code::Invalid);
}

/// 测试用 Loader：按条目长度把持有字节拷进缓冲，边界判断复用 [`check_capacity`]。
struct StubLoader(Vec<u8>);

impl Loader for StubLoader {
    fn copy_into(&self, entry: &Entry, out: &mut [u8]) -> Result<()> {
        check_capacity(entry, out)?;
        out.copy_from_slice(&self.0[..out.len()]);
        Ok(())
    }
}

#[test]
fn load_f32_decodes_an_entry_into_the_resident_buffer() {
    let bytes: Vec<u8> = [1.0f32, 0.0, 2.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let entry = Entry::new("w", DType::F32, Shape::new([3])).at(64);
    let values = load_f32(&entry, &StubLoader(bytes)).expect("装载");
    assert_eq!(values, [1.0, 0.0, 2.5]);
}

#[test]
fn a_non_f32_entry_is_not_unpacked_into_the_resident_buffer() {
    let entry = Entry::new("q", DType::F16, Shape::new([4])).at(64);
    let error = load_f32(&entry, &StubLoader(vec![0u8; 8])).expect_err("非 F32 必须拒绝");
    assert_eq!(error.code(), Code::Unsupported);
}

#[test]
fn decode_entry_f32_passes_through_the_partial_tail_word_rejection() {
    let entry = Entry::new("w", DType::F32, Shape::new([2])).at(64);
    let error = decode_entry_f32(&entry, &[0u8, 1, 2]).expect_err("余字节");
    assert_eq!(error.code(), Code::Invalid);
}
