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
        let prefix = bytes.get(..8).ok_or_else(|| {
            invalid(format!(
                "文件只有 {} 字节，读不出 8 字节头部长度",
                bytes.len()
            ))
        })?;
        let (declared, header_end) = split_prefix(prefix.try_into().expect("恰为 8 字节"))?;
        let text = bytes.get(8..header_end).ok_or_else(|| {
            invalid(format!(
                "头部声明 {declared} 字节，但 {header_end} 已超出文件长度 {}",
                bytes.len()
            ))
        })?;
        let header = parse_header(text, header_end)?;
        let data = bytes.get(header_end..).expect("上方已确认头部在文件内");
        let covered = header.data_bytes();
        if covered != data.len() {
            return Err(invalid(format!(
                "数据段应为 {covered} 字节（由条目的最大结束偏移推出），实际 {} 字节",
                data.len()
            ))
            .with("weights::safetensors::Archive"));
        }
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
        let start = entry
            .offset()
            .checked_sub(self.header.data_start())
            .ok_or_else(|| {
                Error::new(
                    Code::Internal,
                    format!(
                        "条目 `{}` 的偏移 {} 早于数据段起点 {}",
                        entry.name(),
                        entry.offset(),
                        self.header.data_start()
                    ),
                )
                .with("weights::safetensors::Archive::read")
            })?;
        let end = start.checked_add(entry.len()).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!("条目 `{}` 的长度溢出", entry.name()),
            )
            .with("weights::safetensors::Archive::read")
        })?;
        self.data.get(start..end).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!(
                    "条目 `{}` 声明 {start}..{end}，数据段只有 {} 字节",
                    entry.name(),
                    self.data.len()
                ),
            )
            .with("weights::safetensors::Archive::read")
        })
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
/// 与 [`Archive`] 的差别在**内存**，不在校验：`Archive` 要求调用方把整份文件交进来（内存或
/// 映射），`File` 只读 8 字节前缀与头部 JSON，数据段留在磁盘上。7B 级权重是十几 GB，
/// "整份读入"在真实模型上不可用，而按条目读只需要一次 `seek + read`。
///
/// 自证合法的口径一个字都没放宽：打开时用**文件长度**核对"头部 + 条目声明的数据段"是否正好等于
/// 整个文件——与 [`Archive::parse`] 用切片长度核对是同一个判断，只是长度从 `metadata` 来。
///
/// # 例子
///
/// ```
/// use llm_weights::safetensors::File;
/// use llm_weights::Loader;
///
/// let json = br#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
/// let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
/// bytes.extend_from_slice(json);
/// bytes.resize(bytes.len() + 24, 7);
/// let path = std::env::temp_dir().join(format!("llm-weights-doc-{}.safetensors", std::process::id()));
/// std::fs::write(&path, &bytes).expect("临时文件可写");
///
/// let file = File::open(&path).expect("头部自证合法");
/// assert_eq!(file.header_len(), 8 + json.len(), "只读了前缀与头部");
/// assert_eq!(file.file_len(), bytes.len());
/// let entry = &file.header().entries()[0];
/// assert_eq!(Loader::load(&file, entry).expect("可读"), vec![7u8; 24]);
/// let _ = std::fs::remove_file(path);
/// ```
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
    /// [`Code::Invalid`] / [`Code::Unsupported`]：与 [`Archive::parse`] 同口径，外加
    /// "文件长度与头部声明不符"——被截断或尾部有多余字节都在这里拦住，而不是等到读某个条目时。
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut source = Source::open(path).map_err(|error| io_error(error, "打开", path))?;
        let file_len = source
            .metadata()
            .map_err(|error| io_error(error, "读取长度", path))
            .and_then(|metadata| {
                usize::try_from(metadata.len()).map_err(|_| {
                    invalid(format!(
                        "文件 `{}` 的 {} 字节超出本机可寻址范围",
                        path.display(),
                        metadata.len()
                    ))
                })
            })?;

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

        // 只读头部：`declared` 由文件自己声明，所以先按它分配——这正是 `header_len > file_len`
        // 必须先判的原因，否则一个坏掉的 8 字节前缀就能让这里替它分配一块任意大的内存。
        let mut text = vec![0u8; declared];
        source.read_exact(&mut text).map_err(|error| {
            io_error(error, "读取头部", path).with("weights::safetensors::File::open")
        })?;
        let header = parse_header(&text, header_len)?;

        let declared_total = header_len
            .checked_add(header.data_bytes())
            .ok_or_else(|| invalid(format!("头部长度 {header_len} 与数据段长度相加溢出")))?;
        if declared_total != file_len {
            let hint = if declared_total > file_len {
                "文件被截断"
            } else {
                "尾部有多余字节"
            };
            return Err(invalid(format!(
                "文件 `{}` 是 {file_len} 字节，但头部（{header_len} 字节）与条目声明的数据段（{} 字节）合起来是 {declared_total} 字节：{hint}",
                path.display(),
                header.data_bytes()
            ))
            .with("weights::safetensors::File::open"));
        }

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

/// 解 8 字节小端长度前缀：给出头部 JSON 的长度与头部终点（即数据段起点）。
fn split_prefix(prefix: [u8; 8]) -> Result<(usize, usize)> {
    let declared = u64::from_le_bytes(prefix);
    let declared = usize::try_from(declared)
        .map_err(|_| invalid(format!("头部长度 {declared} 超出本机可寻址范围")))?;
    let end = 8usize
        .checked_add(declared)
        .ok_or_else(|| invalid(format!("头部长度 {declared} 与 8 字节前缀相加溢出")))?;
    Ok((declared, end))
}

/// 由头部 JSON 构造并校验 [`Header`]：[`Archive`] 与 [`File`] 共用这一份，不各写一遍。
fn parse_header(text: &[u8], data_start: usize) -> Result<Header> {
    let text = std::str::from_utf8(text).map_err(|_| invalid("头部不是 UTF-8".to_string()))?;
    let root = json::parse(text).map_err(|error| error.with("weights::safetensors 头部"))?;
    let fields = root
        .object()
        .map_err(|error| error.with(format!("safetensors 头部（{} 字节）的顶层", text.len())))?;
    let mut entries = Vec::new();
    for (name, value) in fields {
        if name == METADATA {
            // 元数据不参与布局，但形状必须是对象：写成字符串的文件一定是被别的工具改坏了。
            value
                .object()
                .map_err(|error| error.with(format!("`{METADATA}` 必须是对象")))?;
            continue;
        }
        entries.push(entry(name, value, data_start)?);
    }
    Header::new(entries, data_start)
}

fn io_error(error: std::io::Error, what: &str, path: &Path) -> Error {
    Error::new(
        Code::Io,
        format!("{what}文件 `{}` 失败：{error}", path.display()),
    )
    .with("weights::safetensors::File")
}

/// 把一条 JSON 条目解析成 [`Entry`]：字段闭集、形状与区间必须自洽。
fn entry(name: &str, value: &Value, data_start: usize) -> Result<Entry> {
    let context = format!("权重条目 `{name}`");
    if name.is_empty() {
        return Err(invalid("条目名不能为空".to_string()).with("weights::safetensors"));
    }
    let fields = value.object().map_err(|error| error.with(&context))?;
    let mut dtype = None;
    let mut shape = None;
    let mut offsets = None;
    for (key, value) in fields {
        match key.as_str() {
            "dtype" => {
                if dtype.is_some() {
                    return Err(invalid("字段 `dtype` 重复".to_string()).with(&context));
                }
                dtype = Some(dtype_of(value)?);
            }
            "shape" => {
                if shape.is_some() {
                    return Err(invalid("字段 `shape` 重复".to_string()).with(&context));
                }
                shape = Some(shape_of(value)?);
            }
            "data_offsets" => {
                if offsets.is_some() {
                    return Err(invalid("字段 `data_offsets` 重复".to_string()).with(&context));
                }
                offsets = Some(offsets_of(value)?);
            }
            other => {
                return Err(invalid(format!(
                    "未知字段 `{other}`（条目只允许 dtype / shape / data_offsets）"
                ))
                .with(&context));
            }
        }
    }
    let dtype = dtype.ok_or_else(|| invalid("缺字段 `dtype`".to_string()).with(&context))?;
    let shape = shape.ok_or_else(|| invalid("缺字段 `shape`".to_string()).with(&context))?;
    let (begin, end) =
        offsets.ok_or_else(|| invalid("缺字段 `data_offsets`".to_string()).with(&context))?;

    let expected = shape.element_bytes(dtype);
    if end - begin != expected {
        return Err(invalid(format!(
            "区间 {begin}..{end} 是 {} 字节，但 {:?} {:?} 推出 {expected} 字节",
            end - begin,
            dtype,
            shape.dims()
        ))
        .with(&context));
    }
    let offset = data_start
        .checked_add(begin)
        .ok_or_else(|| invalid(format!("偏移 {begin} 与数据段起点相加溢出")).with(&context))?;
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
    let items = value
        .array()
        .map_err(|error| error.with("字段 `data_offsets`"))?;
    if items.len() != 2 {
        return Err(invalid(format!(
            "`data_offsets` 必须是两个整数，实际 {} 个",
            items.len()
        ))
        .with("字段 `data_offsets`"));
    }
    let mut pair = [0usize; 2];
    for (slot, item) in pair.iter_mut().zip(items) {
        let number = item
            .integer()
            .map_err(|error| error.with("字段 `data_offsets`"))?;
        *slot = usize::try_from(number)
            .map_err(|_| invalid(format!("偏移 {number} 不能为负")).with("字段 `data_offsets`"))?;
    }
    if pair[0] > pair[1] {
        return Err(invalid(format!(
            "`data_offsets` 的起点 {} 晚于终点 {}",
            pair[0], pair[1]
        ))
        .with("字段 `data_offsets`"));
    }
    Ok((pair[0], pair[1]))
}

fn invalid(message: String) -> Error {
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
