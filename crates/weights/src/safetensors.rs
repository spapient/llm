//! safetensors 头部：把文件字节段映射成 [`Header`] 与它的 [`Reader`]。
//!
//! 格式只有两段：前 8 字节是**小端**的头部长度 `N`，紧跟 `N` 字节 UTF-8 JSON，之后是数据段。
//! JSON 是一个对象：每个键都是条目名（`__metadata__` 除外），值里写 `dtype` / `shape` /
//! `data_offsets`（相对数据段起点的 `[begin, end)`）。
//!
//! 这里做的是**自证合法**，不是"尽力读懂"：形状推出的字节数必须与声明的区间长度相等、数据段必须
//! 正好被条目覆盖、字段只允许那三个。任一条不成立都直接报错——否则下半段会以一个更难查的方式
//! 崩在某个越界读上。
//!
//! 元素类型映射到 [`DType`] 闭集（`F32` / `F16` / `BF16` / `I8`）。int4 权重在 safetensors 里
//! 以 8 位条目存两个 nibble 一个字节，分组由伴随的标度条目给出（见 [`crate::quant`]）；`U4` 是
//! 解包后的视图，文件里不出现，因此这里没有 `U4` 的映射。
//!
//! # 例子
//!
//! ```
//! use llm_weights::safetensors::Archive;
//!
//! // 头部是一段 JSON，数据段是 2×3 个 f32（24 字节）。
//! let json = br#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
//! let mut file = (json.len() as u64).to_le_bytes().to_vec();
//! file.extend_from_slice(json);
//! file.resize(file.len() + 24, 0);
//!
//! let archive = Archive::parse(&file).expect("头部自证合法");
//! let header = archive.header();
//! assert_eq!(header.entries().len(), 1);
//! assert_eq!(header.get("w").map(|entry| entry.len()), Some(24));
//! ```

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

use crate::json::{self, Value};
use crate::{Entry, Header, Loader, Reader, check_capacity};

/// 头部里与条目并列的元数据键：不是条目，跳过。
pub const METADATA: &str = "__metadata__";

/// 支持的 `dtype` 名 ↔ 本仓元素类型闭集。
///
/// `I8` 与 `U8` 映射到同一个元素类型 [`DType::I8`]：本仓的 8 位元素类型本来就定义为"8 位整数，
/// 有符号与否由算子声明决定"，而**文件里的有符号性不决定它怎么用**——真实的 4 位量化权重就是用
/// `U8` 存两个 nibble 一个字节。拒绝 `U8` 会让"支持真实量化模型"这句话不成立。
///
/// 那它为什么不再是"猜"：一个 8 位条目到底是"逐元素 int8"还是"打包的两个 nibble"，由**字节数**
/// 决定（`out × in` 还是 `out × ceil(in / 2)`），由 [`crate::Header::quantization`] 当场验证。
/// 符号性交给算子声明，长度交给文件自证，两者都不靠 dtype 名。
///
/// `I16` / `F64` / `F8_E4M3` 之类仍拒绝：它们没有对应的元素类型，映射过去只能丢精度或改语义。
pub const DTYPES: [(&str, DType); 5] = [
    ("F32", DType::F32),
    ("F16", DType::F16),
    ("BF16", DType::BF16),
    ("I8", DType::I8),
    ("U8", DType::I8),
];

/// 一个 safetensors 文件：已校验的头部 + 数据段切片。
///
/// 生命周期绑定在文件字节上：不复制数据，也不持有文件句柄。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Archive<'a> {
    header: Header,
    data: &'a [u8],
}

impl<'a> Archive<'a> {
    /// 解析并校验整份文件。
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`]：长度前缀或 JSON 文法坏、字段缺失/多余、声明区间与形状不符、
    /// 数据段不足或有多余字节、条目重名/重叠。
    ///
    /// [`Code::Unsupported`]：`dtype` 不在 [`DTYPES`] 闭集内。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let (header_end, text) = header_slice(bytes)?;
        let header = parse_header(text, header_end)?;
        let data = data_after(bytes, &header, header_end)?;
        Ok(Self { header, data })
    }

    /// 已校验的头部。
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// 数据段（从头部之后到文件末尾）。
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// 各条目实际占用的字节总数（不含条目之间的空隙）。
    pub fn payload_bytes(&self) -> usize {
        self.header.entries().iter().map(Entry::len).sum()
    }
}

impl Reader for Archive<'_> {
    fn read(&self, entry: &Entry) -> Result<&[u8]> {
        let Some(start) = entry.offset().checked_sub(self.header.data_start()) else {
            return Err(internal(format!(
                "条目 `{}` 的偏移 {} 早于数据段起点 {}",
                entry.name(),
                entry.offset(),
                self.header.data_start()
            )));
        };
        let Some(end) = start.checked_add(entry.len()) else {
            return Err(internal(format!("条目 `{}` 的长度溢出", entry.name())));
        };
        let Some(bytes) = self.data.get(start..end) else {
            return Err(internal(format!(
                "条目 `{}` 声明 {start}..{end}，数据段只有 {} 字节",
                entry.name(),
                self.data.len()
            )));
        };
        Ok(bytes)
    }
}

impl Loader for Archive<'_> {
    fn copy_into(&self, entry: &Entry, out: &mut [u8]) -> Result<()> {
        check_capacity(entry, out)?;
        let bytes = self.read(entry)?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

/// 磁盘后端（只解析头部、按条目回读），定义在 [`safetensors_file`] 以保持本文件只做内存适配。
pub use crate::safetensors_file::File;

/// 解 8 字节小端长度前缀：给出头部 JSON 的长度与头部终点（即数据段起点）。
pub(crate) fn split_prefix(prefix: [u8; 8]) -> Result<(usize, usize)> {
    let declared = u64::from_le_bytes(prefix);
    let declared = usize::try_from(declared)
        .map_err(|_| invalid(format!("头部长度 {declared} 超出本机可寻址范围")))?;
    let end = 8usize
        .checked_add(declared)
        .ok_or_else(|| invalid(format!("头部长度 {declared} 与 8 字节前缀相加溢出")))?;
    Ok((declared, end))
}

/// 由头部 JSON 构造并校验 [`Header`]：[`Archive`] 与 [`File`] 共用这一份。
pub(crate) fn parse_header(text: &[u8], data_start: usize) -> Result<Header> {
    let root = root_value(text)?;
    let fields = root
        .object()
        .map_err(|error| error.with(format!("safetensors 头部（{} 字节）的顶层", text.len())))?;
    let entries = collect_entries(fields, data_start)?;
    Header::new(entries, data_start)
}

/// 按文件顺序：元数据只校验，其余解析成条目。
fn collect_entries(fields: &[(String, Value)], data_start: usize) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for (name, value) in fields {
        if name.as_str() == METADATA {
            check_metadata(value)?;
            continue;
        }
        entries.push(entry(name, value, data_start)?);
    }
    Ok(entries)
}

/// 头部 JSON 根值；不是 UTF-8 或文法坏都在这里报错。
fn root_value(text: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(text).map_err(|_| invalid("头部不是 UTF-8".to_string()))?;
    json::parse(text).map_err(|error| error.with("weights::safetensors 头部"))
}

/// 元数据不参与布局，但它的值必须是对象。
fn check_metadata(value: &Value) -> Result<()> {
    value
        .object()
        .map_err(|error| error.with(format!("`{METADATA}` 必须是对象")))?;
    Ok(())
}


/// 把一条 JSON 条目解析成 [`Entry`]：字段闭集、形状与区间必须自洽。
fn entry(name: &str, value: &Value, data_start: usize) -> Result<Entry> {
    if name.is_empty() {
        return Err(invalid("条目名不能为空".to_string()).with("weights::safetensors"));
    }
    let context = format!("权重条目 `{name}`");
    let fields = value.object().map_err(|error| error.with(&context))?;
    let (dtype, shape, offsets) = collect_fields(fields, &context)?;
    let (dtype, shape, offsets) = required_fields(dtype, shape, offsets, &context)?;
    build(name, dtype, shape, offsets, data_start, &context)
}

/// 条目三个字段的可选槽位。
type Slots = (Option<DType>, Option<Shape>, Option<(usize, usize)>);

/// 三个字段；重复与未知字段按文件顺序拒绝。
fn collect_fields(fields: &[(String, Value)], context: &str) -> Result<Slots> {
    let mut dtype = None;
    let mut shape = None;
    let mut offsets = None;
    for (key, value) in fields {
        field(key, value, context, &mut dtype, &mut shape, &mut offsets)?;
    }
    Ok((dtype, shape, offsets))
}

/// 一条字段写进对应槽位；未知字段直接报错。
fn field(
    key: &str,
    value: &Value,
    context: &str,
    dtype: &mut Option<DType>,
    shape: &mut Option<Shape>,
    offsets: &mut Option<(usize, usize)>,
) -> Result<()> {
    match key {
        "dtype" => dtype_slot(dtype, value, key, context),
        "shape" => shape_slot(shape, value, key, context),
        "data_offsets" => offsets_slot(offsets, value, key, context),
        other => Err(invalid(format!(
            "未知字段 `{other}`（条目只允许 dtype / shape / data_offsets）"
        ))
        .with(context)),
    }
}

fn dtype_slot(slot: &mut Option<DType>, value: &Value, key: &str, context: &str) -> Result<()> {
    set_once(slot, dtype_of(value)?, key, context)
}

fn shape_slot(slot: &mut Option<Shape>, value: &Value, key: &str, context: &str) -> Result<()> {
    set_once(slot, shape_of(value)?, key, context)
}

fn offsets_slot(
    slot: &mut Option<(usize, usize)>,
    value: &Value,
    key: &str,
    context: &str,
) -> Result<()> {
    set_once(slot, offsets_of(value)?, key, context)
}

/// 字段只允许出现一次。
fn set_once<T>(slot: &mut Option<T>, value: T, key: &str, context: &str) -> Result<()> {
    if slot.is_some() {
        return Err(invalid(format!("字段 `{key}` 重复")).with(context));
    }
    *slot = Some(value);
    Ok(())
}

/// 三个必填字段，缺一即报错并指出名字。
fn required_fields(
    dtype: Option<DType>,
    shape: Option<Shape>,
    offsets: Option<(usize, usize)>,
    context: &str,
) -> Result<(DType, Shape, (usize, usize))> {
    let dtype = required(dtype, "dtype", context)?;
    let shape = required(shape, "shape", context)?;
    let offsets = required(offsets, "data_offsets", context)?;
    Ok((dtype, shape, offsets))
}

fn required<T>(slot: Option<T>, key: &str, context: &str) -> Result<T> {
    slot.ok_or_else(|| invalid(format!("缺字段 `{key}`")).with(context))
}

/// 区间长度与形状自洽，并把相对起点转成绝对偏移。
fn build(
    name: &str,
    dtype: DType,
    shape: Shape,
    (begin, end): (usize, usize),
    data_start: usize,
    context: &str,
) -> Result<Entry> {
    let expected = shape.element_bytes(dtype);
    if end - begin != expected {
        return Err(invalid(format!(
            "区间 {begin}..{end} 是 {} 字节，但 {:?} {:?} 推出 {expected} 字节",
            end - begin,
            dtype,
            shape.dims()
        ))
        .with(context));
    }
    let offset = data_start
        .checked_add(begin)
        .ok_or_else(|| invalid(format!("偏移 {begin} 与数据段起点相加溢出")).with(context))?;
    Ok(Entry::new(name, dtype, shape).at(offset))
}

/// `dtype` → [`DType`]；不认识的名字直接报错并列出闭集。
fn dtype_of(value: &Value) -> Result<DType> {
    let name = value.string().map_err(|error| error.with("字段 `dtype`"))?;
    DTYPES
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, dtype)| *dtype)
        .ok_or_else(|| {
            let known: Vec<&str> = DTYPES.iter().map(|(name, _)| *name).collect();
            Error::new(
                Code::Unsupported,
                format!("`dtype` 是 `{name}`，闭集里只有 {}", known.join(" / ")),
            )
            .with("字段 `dtype`")
        })
}

fn shape_of(value: &Value) -> Result<Shape> {
    let items = value.array().map_err(|error| error.with("字段 `shape`"))?;
    let mut dims = Vec::with_capacity(items.len());
    for item in items {
        let number = item
            .integer()
            .map_err(|error| error.with("字段 `shape` 的轴长度"))?;
        let axis = usize::try_from(number)
            .map_err(|_| invalid(format!("轴长度 {number} 不能为负")).with("字段 `shape`"))?;
        dims.push(axis);
    }
    Ok(Shape::new(dims))
}

fn offsets_of(value: &Value) -> Result<(usize, usize)> {
    let (begin, end) = pair_of(value)?;
    let begin = offset(begin)?;
    let end = offset(end)?;
    if begin > end {
        return Err(
            invalid(format!("`data_offsets` 的起点 {begin} 晚于终点 {end}"))
                .with("字段 `data_offsets`"),
        );
    }
    Ok((begin, end))
}

/// `data_offsets` 必须是恰好两个元素的数组。
fn pair_of(value: &Value) -> Result<(&Value, &Value)> {
    let items = value
        .array()
        .map_err(|error| error.with("字段 `data_offsets`"))?;
    match items {
        [begin, end] => Ok((begin, end)),
        _ => Err(invalid(format!(
            "`data_offsets` 必须是两个整数，实际 {} 个",
            items.len()
        ))
        .with("字段 `data_offsets`")),
    }
}

/// 一个非负偏移值；负数越界即报错。
fn offset(value: &Value) -> Result<usize> {
    let number = value
        .integer()
        .map_err(|error| error.with("字段 `data_offsets`"))?;
    usize::try_from(number)
        .map_err(|_| invalid(format!("偏移 {number} 不能为负")).with("字段 `data_offsets`"))
}

/// 8 字节长度前缀与头部 JSON 切片；两者都必须落在文件内。
fn header_slice(bytes: &[u8]) -> Result<(usize, &[u8])> {
    let Some(prefix) = bytes.get(..8) else {
        return Err(invalid(format!(
            "文件只有 {} 字节，读不出 8 字节头部长度",
            bytes.len()
        )));
    };
    let mut word = [0u8; 8];
    word.copy_from_slice(prefix);
    let (declared, header_end) = split_prefix(word)?;
    let Some(text) = bytes.get(8..header_end) else {
        return Err(invalid(format!(
            "头部声明 {declared} 字节，但 {header_end} 已超出文件长度 {}",
            bytes.len()
        )));
    };
    Ok((header_end, text))
}

/// 数据段：必须正好被条目覆盖，不许多余字节。
fn data_after<'b>(bytes: &'b [u8], header: &Header, header_end: usize) -> Result<&'b [u8]> {
    let data = &bytes[header_end..];
    let covered = header.data_bytes();
    if covered == data.len() {
        return Ok(data);
    }
    Err(invalid(format!(
        "数据段应为 {covered} 字节（由条目的最大结束偏移推出），实际 {} 字节",
        data.len()
    ))
    .with("weights::safetensors::Archive"))
}





/// 读端内部错误：偏移越界只可能是头部与数据段漂移。
fn internal(message: String) -> Error {
    Error::new(Code::Internal, message).with("weights::safetensors::Archive::read")
}

pub(crate) fn invalid(message: String) -> Error {
    Error::new(Code::Invalid, message).with("weights::safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 按 safetensors 的布局拼一个文件：8 字节长度前缀 + 头部 + 数据段。
    fn file(json: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn two_entries_are_read_back_byte_for_byte() {
        let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
                       "b": {"dtype": "F16", "shape": [4], "data_offsets": [24, 32]}}"#;
        let data: Vec<u8> = (0..32).collect();
        let bytes = file(json, &data);
        let archive = Archive::parse(&bytes).expect("合法");

        let names: Vec<&str> = archive.header().entries().iter().map(Entry::name).collect();
        assert_eq!(names, ["a", "b"], "条目按数据段偏移排序");
        assert_eq!(archive.payload_bytes(), 32);

        let a = archive.header().get("a").expect("条目 a");
        assert_eq!(archive.read(a).expect("可读"), &data[..24]);
        let b = archive.header().get("b").expect("条目 b");
        assert_eq!(archive.read(b).expect("可读"), &data[24..]);
    }

    #[test]
    fn metadata_is_skipped_but_must_still_be_an_object() {
        let json = r#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}"#;
        let bytes = file(json, &[1, 2]);
        let archive = Archive::parse(&bytes).expect("合法");
        assert_eq!(archive.header().entries().len(), 1);

        let broken =
            r#"{"__metadata__": "pt", "w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}"#;
        let error = Archive::parse(&file(broken, &[1, 2])).expect_err("元数据不是对象");
        assert!(error.to_string().contains("必须是对象"), "{error}");
    }

    #[test]
    fn an_unknown_dtype_is_unsupported_rather_than_guessed() {
        let json = r#"{"w": {"dtype": "I16", "shape": [1], "data_offsets": [0, 2]}}"#;
        let error = Archive::parse(&file(json, &[1, 2])).expect_err("I16 不在闭集");
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("I16"), "{error}");
        assert!(
            error.to_string().contains("F32"),
            "错误里要列出闭集：{error}"
        );
    }

    #[test]
    fn an_unsigned_byte_entry_is_the_same_eight_bit_element() {
        // 真实的 4 位量化权重存成 `U8`（两个 nibble 一个字节）：拒绝它等于拒绝真实模型。
        let json = r#"{"w": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}"#;
        let bytes = file(json, &[1, 2]);
        let archive = Archive::parse(&bytes).expect("U8 是 8 位元素");
        let entry = &archive.header().entries()[0];
        assert_eq!(entry.dtype(), DType::I8);
        assert_eq!(archive.read(entry).expect("可读"), &[1, 2]);
    }

    #[test]
    fn a_declared_interval_must_match_the_shape() {
        let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 16]}}"#;
        let error = Archive::parse(&file(json, &[0; 16])).expect_err("少了 8 字节");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("推出 24 字节"), "{error}");
    }

    #[test]
    fn unknown_and_missing_fields_are_rejected() {
        let unknown =
            r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "layout": "row"}}"#;
        let error = Archive::parse(&file(unknown, &[0; 4])).expect_err("未知字段");
        assert!(error.to_string().contains("未知字段 `layout`"), "{error}");

        let missing = r#"{"w": {"dtype": "F32", "shape": [1]}}"#;
        let error = Archive::parse(&file(missing, &[])).expect_err("缺字段");
        assert!(
            error.to_string().contains("缺字段 `data_offsets`"),
            "{error}"
        );
    }

    #[test]
    fn a_data_section_that_does_not_match_the_header_is_rejected() {
        let json = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;
        for data in [&[0u8; 0][..], &[0u8; 5][..]] {
            let error = Archive::parse(&file(json, data)).expect_err("字节数不符");
            assert!(error.to_string().contains("数据段应为 4 字节"), "{error}");
        }
    }

    #[test]
    fn a_broken_length_prefix_is_rejected() {
        assert!(Archive::parse(&[0u8; 4]).is_err(), "不足 8 字节");
        let mut oversized = 4096u64.to_le_bytes().to_vec();
        oversized.extend_from_slice(b"{}");
        let error = Archive::parse(&oversized).expect_err("头部超长");
        assert!(error.to_string().contains("超出文件长度"), "{error}");
        assert!(Archive::parse(&file("[]", &[])).is_err(), "顶层必须是对象");
    }

    #[test]
    fn overlapping_and_reversed_intervals_are_rejected() {
        let overlap = r#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                          "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}"#;
        let error = Archive::parse(&file(overlap, &[0; 12])).expect_err("区间重叠");
        assert!(error.to_string().contains("重叠"), "{error}");

        let reversed = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}"#;
        let error = Archive::parse(&file(reversed, &[])).expect_err("起点晚于终点");
        assert!(error.to_string().contains("晚于"), "{error}");
    }

    #[test]
    fn an_empty_archive_is_a_valid_zero_byte_file() {
        let bytes = file("{}", &[]);
        let archive = Archive::parse(&bytes).expect("空头部合法");
        assert!(archive.header().entries().is_empty());
        assert_eq!(archive.payload_bytes(), 0);
        assert!(archive.data().is_empty());
    }

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("llm-weights-{}-{name}", std::process::id()))
    }

    /// 落一个数据段长度为 `data_len` 的文件：数据段用 `set_len` 拉开（稀疏文件，不占内存）。
    fn sparse(name: &str, json: &str, data_len: usize) -> std::path::PathBuf {
        use std::io::Write as _;
        let target = path(name);
        let mut file = std::fs::File::create(&target).expect("临时文件可写");
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("写长度前缀");
        file.write_all(json.as_bytes()).expect("写头部");
        file.set_len(8 + json.len() as u64 + data_len as u64)
            .expect("拉开数据段");
        target
    }

    #[test]
    fn a_gigabyte_of_weights_is_opened_by_reading_only_the_header() {
        // 1 GiB 数据段 + 尾部一个 4 字节条目：打开只读前缀与头部，读小条目只读那 4 字节。
        let json = r#"{"big": {"dtype": "F32", "shape": [268435456], "data_offsets": [0, 1073741824]},
                       "__metadata__": {"format": "pt"},
                       "tail": {"dtype": "I8", "shape": [4], "data_offsets": [1073741824, 1073741828]}}"#;
        let target = sparse("huge.safetensors", json, 1_073_741_828);
        let file = File::open(&target).expect("头部自证合法");
        assert_eq!(file.header_len(), 8 + json.len(), "只读了前缀与头部");
        assert_eq!(file.file_len(), 8 + json.len() + 1_073_741_828);
        assert_eq!(file.header().entries().len(), 2, "`__metadata__` 不是条目");
        assert_eq!(file.header().data_bytes(), 1_073_741_828);

        let tail = file.header().get("tail").expect("条目 tail");
        assert_eq!(Loader::load(&file, tail).expect("可读"), vec![0u8; 4]);

        // 缓冲区不够时先在长度上拒绝，不会去磁盘上搬 1 GiB。
        let big = file.header().get("big").expect("条目 big");
        let mut short = [0u8; 4];
        let error = file.copy_into(big, &mut short).expect_err("缓冲区太小");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("缓冲区 4 字节"), "{error}");
        let _ = std::fs::remove_file(target);
    }

    #[test]
    fn a_truncated_or_padded_file_is_rejected_when_it_is_opened() {
        let json = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
        let truncated = sparse("truncated.safetensors", json, 16);
        let error = File::open(&truncated).expect_err("少了 8 字节");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("文件被截断"), "{error}");
        let _ = std::fs::remove_file(truncated);

        let padded = sparse("padded.safetensors", json, 25);
        let error = File::open(&padded).expect_err("多出 1 字节");
        assert!(error.to_string().contains("尾部有多余字节"), "{error}");
        let _ = std::fs::remove_file(padded);
    }

    #[test]
    fn a_prefix_that_promises_more_than_the_file_holds_is_rejected() {
        let short = path("short.safetensors");
        std::fs::write(&short, 4096u64.to_le_bytes()).expect("临时文件可写");
        let error = File::open(&short).expect_err("头部超长");
        assert!(error.to_string().contains("头部声明 4096 字节"), "{error}");
        let _ = std::fs::remove_file(short);

        let tiny = path("tiny.safetensors");
        std::fs::write(&tiny, [0u8; 4]).expect("临时文件可写");
        let error = File::open(&tiny).expect_err("不足 8 字节");
        assert!(error.to_string().contains("读不出 8 字节"), "{error}");
        let _ = std::fs::remove_file(tiny);
    }

    #[test]
    fn a_missing_file_is_an_io_error_not_a_panic() {
        let error = File::open("/nonexistent/llm.safetensors").expect_err("打不开");
        assert_eq!(error.code(), Code::Io);
        assert!(error.to_string().contains("打开文件"), "{error}");
    }

    #[test]
    fn both_backends_produce_the_same_header_and_the_same_bytes() {
        let json = r#"{"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
                       "b": {"dtype": "I8", "shape": [4], "data_offsets": [24, 28]}}"#;
        let data: Vec<u8> = (0..28).collect();
        let bytes = file(json, &data);
        let archive = Archive::parse(&bytes).expect("合法");
        let target = path("agree.safetensors");
        std::fs::write(&target, &bytes).expect("临时文件可写");
        let on_disk = File::open(&target).expect("头部自证合法");

        assert_eq!(on_disk.header(), archive.header(), "两个后端解出同一个头部");
        for entry in archive.header().entries() {
            assert_eq!(
                Loader::load(&on_disk, entry).expect("可读"),
                Loader::load(&archive, entry).expect("可读"),
                "条目 `{}`",
                entry.name()
            );
        }
        let _ = std::fs::remove_file(target);
    }

    #[test]
    fn a_wrong_sized_buffer_is_rejected_by_the_slice_backend_too() {
        let json = r#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;
        let bytes = file(json, &[0; 4]);
        let archive = Archive::parse(&bytes).expect("合法");
        let entry = &archive.header().entries()[0];

        let mut too_big = [0u8; 8];
        let error = archive
            .copy_into(entry, &mut too_big)
            .expect_err("缓冲区太大");
        assert!(error.to_string().contains("缓冲区 8 字节"), "{error}");

        let mut exact = [9u8; 4];
        archive.copy_into(entry, &mut exact).expect("正好");
        assert_eq!(exact, [0u8; 4]);
    }
}
