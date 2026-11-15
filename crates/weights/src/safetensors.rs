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
//! 元素类型映射到 [`DType`] 闭集（`F32` / `F16` / `BF16` / `I8` / `U8`）。int4 权重在 safetensors
//! 里以 `U8` 条目存两个 nibble 一个字节，分组由伴随的标度条目给出（见 [`crate::quant`]）；`U4` 是
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

use std::fs::File as Source;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

use crate::json::{self, Value};
use crate::{Entry, Header, Loader, Reader, check_capacity};

/// 头部里与条目并列的元数据键：不是条目，跳过。
pub const METADATA: &str = "__metadata__";

/// 支持的 `dtype` 名 ↔ 本仓元素类型闭集。
///
/// `I8` 与 `U8` 保留各自的符号性：`I8` 是逐元素有符号字节，`U8` 是无符号字节（真实 4 位量化
/// 权重就用 `U8` 缓冲存两个 nibble 一个字节）。符号性留在解析层作为分流宽度的事实，weights 不
/// 认识算子分组档——分组仍由伴随标度条目的形状推出（见 [`crate::quant`]）。
///
/// 字节长度由 `shape × dtype` 推出，safetensors 要求声明区间等于这个乘积，因此长度是结构性
/// 自证，不靠 dtype 名猜。`I16` / `F64` / `F8_E4M3` 之类仍拒绝：它们没有对应的元素类型，映射
/// 过去只能丢精度或改语义。
pub const DTYPES: [(&str, DType); 5] = [
    ("F32", DType::F32),
    ("F16", DType::F16),
    ("BF16", DType::BF16),
    ("I8", DType::I8),
    ("U8", DType::U8),
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

/// 磁盘上的 safetensors 文件：只解析头部，数据段按条目读。
///
/// 与 [`Archive`] 的差别在内存，不在校验：`Archive` 要求调用方把整份文件交进来（内存或映射），
/// `File` 只读 8 字节前缀与头部 JSON，数据段留在磁盘上。7B 级权重是十几 GB，"整份读入"在真实模型
/// 上不可用，而按条目读只需要一次 `seek + read`。
///
/// 自证合法的口径一个字都没放宽：打开时用**文件长度**核对"头部 + 条目声明的数据段"是否正好等于
/// 整个文件——与 [`Archive::parse`] 用切片长度核对是同一个判断，只是长度从 `metadata` 来。
#[derive(Debug)]
pub struct File {
    source: Source,
    header: Header,
    header_len: usize,
    file_len: usize,
}

impl File {
    /// 打开并校验一个 safetensors 文件，只读头部。
    ///
    /// # Errors
    ///
    /// [`Code::Io`]：打不开、取不到长度、`seek` 或 `read` 失败。
    ///
    /// [`Code::Invalid`] / [`Code::Unsupported`]：与 [`Archive::parse`] 同口径，外加"文件长度与头部
    /// 声明不符"——被截断或尾部有多余字节都在这里拦住，而不是等到读某个条目时。
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (mut source, file_len) = open_source(path)?;
        let (declared, header_len) = read_prefix(&mut source, path, file_len)?;
        let header = read_header(&mut source, path, declared, header_len)?;
        check_file_len(path, &header, header_len, file_len)?;
        Ok(Self {
            source,
            header,
            header_len,
            file_len,
        })
    }

    /// 已校验的头部。
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// 打开时从磁盘读了多少字节（8 字节前缀 + 头部 JSON），与数据段大小无关。
    pub fn header_len(&self) -> usize {
        self.header_len
    }

    /// 文件总长度。
    pub fn file_len(&self) -> usize {
        self.file_len
    }
}

impl Loader for File {
    fn copy_into(&self, entry: &Entry, out: &mut [u8]) -> Result<()> {
        check_capacity(entry, out)?;
        let mut source = &self.source;
        source
            .seek(SeekFrom::Start(entry.offset() as u64))
            .map_err(|error| {
                Error::new(
                    Code::Io,
                    format!(
                        "定位到条目 `{}` 的偏移 {} 失败：{error}",
                        entry.name(),
                        entry.offset()
                    ),
                )
                .with("weights::safetensors::File::copy_into")
            })?;
        source.read_exact(out).map_err(|error| {
            Error::new(
                Code::Io,
                format!(
                    "读条目 `{}`（{} 字节，偏移 {}）失败：{error}",
                    entry.name(),
                    out.len(),
                    entry.offset()
                ),
            )
            .with("weights::safetensors::File::copy_into")
        })
    }
}

/// 打开文件并取长度；超出 usize 是明确错误而不是截断。
fn open_source(path: &Path) -> Result<(Source, usize)> {
    let source = Source::open(path).map_err(|error| io_error(error, "打开", path))?;
    let metadata = source
        .metadata()
        .map_err(|error| io_error(error, "读取长度", path))?;
    let file_len = usize::try_from(metadata.len()).map_err(|_| {
        invalid(format!(
            "文件 `{}` 的 {} 字节超出本机可寻址范围",
            path.display(),
            metadata.len()
        ))
    })?;
    Ok((source, file_len))
}

/// 8 字节前缀；头部声明的长度不得越过文件末尾。
fn read_prefix(source: &mut Source, path: &Path, file_len: usize) -> Result<(usize, usize)> {
    let mut prefix = [0u8; 8];
    source.read_exact(&mut prefix).map_err(|_| {
        invalid(format!(
            "文件 `{}` 只有 {file_len} 字节，读不出 8 字节头部长度",
            path.display()
        ))
        .with("weights::safetensors::File::open")
    })?;
    let (declared, header_len) = split_prefix(prefix)?;
    if header_len > file_len {
        return Err(invalid(format!(
            "头部声明 {declared} 字节，但文件 `{}` 只有 {file_len} 字节",
            path.display()
        ))
        .with("weights::safetensors::File::open"));
    }
    Ok((declared, header_len))
}

/// 读头部 JSON 并解析；分配前已确认声明长度不超过文件。
fn read_header(
    source: &mut Source,
    path: &Path,
    declared: usize,
    header_len: usize,
) -> Result<Header> {
    let mut text = vec![0u8; declared];
    source.read_exact(&mut text).map_err(|error| {
        io_error(error, "读取头部", path).with("weights::safetensors::File::open")
    })?;
    parse_header(&text, header_len)
}

/// 头部 + 声明数据段必须正好等于文件长度。
fn check_file_len(path: &Path, header: &Header, header_len: usize, file_len: usize) -> Result<()> {
    let declared_total = header_len
        .checked_add(header.data_bytes())
        .ok_or_else(|| invalid(format!("头部长度 {header_len} 与数据段长度相加溢出")))?;
    if declared_total == file_len {
        return Ok(());
    }
    let hint = if declared_total > file_len {
        "文件被截断"
    } else {
        "尾部有多余字节"
    };
    Err(invalid(format!(
        "文件 `{}` 是 {file_len} 字节，但头部（{header_len} 字节）与条目声明的数据段（{} 字节）合起来是 {declared_total} 字节：{hint}",
        path.display(),
        header.data_bytes()
    ))
    .with("weights::safetensors::File::open"))
}

fn io_error(error: std::io::Error, what: &str, path: &Path) -> Error {
    Error::new(
        Code::Io,
        format!("{what}文件 `{}` 失败：{error}", path.display()),
    )
    .with("weights::safetensors::File")
}

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
#[path = "../tests/safetensors_tests.rs"]
mod tests;
