//! 磁盘后端：只解析头部，数据段按条目 `seek + read`，不整份读入。
//!
//! 与内存后端 [`crate::safetensors::Archive`] 的差别在**所有权**，不在校验：头部解析与
//! 条目自洽判据两边共用同一份（见 [`crate::safetensors::parse_header`]），这里只负责打开、
//! 取长度、读前缀/头部与按条目回读。

use std::fs::File as Source;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;

use diagnostic::{Code, Error, Result};

use crate::{Entry, Header, Loader, check_capacity};
use crate::safetensors::{invalid, parse_header, split_prefix};

/// 磁盘上的 safetensors 文件：只解析头部，数据段按条目读。
///
/// 与 [`crate::safetensors::Archive`] 的差别在内存，不在校验：`Archive` 要求调用方把整份
/// 文件交进来（内存或映射），`File` 只读 8 字节前缀与头部 JSON，数据段留在磁盘上。7B 级
/// 权重是十几 GB，"整份读入"在真实模型上不可用，而按条目读只需要一次 `seek + read`。
///
/// 自证合法的口径一个字都没放宽：打开时用**文件长度**核对"头部 + 条目声明的数据段"是否
/// 正好等于整个文件——与 [`crate::safetensors::Archive::parse`] 用切片长度核对是同一个判断，
/// 只是长度从 `metadata` 来。
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
    /// [`Code::Invalid`] / [`Code::Unsupported`]：与 [`crate::safetensors::Archive::parse`]
    /// 同口径，外加"文件长度与头部声明不符"——被截断或尾部有多余字节都在这里拦住，而不是
    /// 等到读某个条目时。
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
