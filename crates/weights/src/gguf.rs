//! GGUF v3 只读解析：头部、KV 元数据、张量信息表与反量化。
//!
//! 布局：`magic "GGUF" + version(u32) + tensor_count(u64) + kv_count(u64)` → KV 表 → 张量信息表
//! → 按 `general.alignment`（默认 32）对齐 → 数据区。本模块只消费调用方持有的字节切片，不做写入、
//! 不做 I/O；所有偏移与长度都经过边界检查，未信任输入不会触发越界。
//!
//! 支持 GGML 类型码：`F32=0`、`F16=1`、`Q8_0=8`、`Q4_K=12`、`Q5_K=13`、`Q6_K=14`。
//! 反量化只求正确，不做 SIMD；性能路径由上层决定。零第三方依赖，纯安全 Rust。

use crate::bad;
use crate::gguf_dequant::dequantize;
use crate::gguf_reader::{Reader, data_start, tables};
use diagnostic::{Code, Error, Result};
use dtype::{DType, QType};

/// 文件 magic。
pub const MAGIC: [u8; 4] = *b"GGUF";
/// 支持的容器版本。
pub const VERSION: u32 = 3;
/// 未声明 `general.alignment` 时的数据区对齐。
pub const DEFAULT_ALIGN: usize = 32;

/// GGML 磁盘类型码的薄映射（ADR 0018 决策 5 的外部码镜像）：只声明本解码器支持哪些类型，
/// 每块元素数与每块字节数一律委托 [`QType`]（块格式）或 [`DType`]（标量元素）的显式布局，
/// 本模块不再自带 `SPAN`/`ELEM` 表。
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 32 位浮点（GGML 码 0）。
    F32 = 0,
    /// 16 位浮点（GGML 码 1）。
    F16 = 1,
    /// 8 位量化块（GGML 码 8，f16 尺度 + 32×i8）。
    Q8_0 = 8,
    /// 4 位 K 超块（GGML 码 12）。
    Q4_K = 12,
    /// 5 位 K 超块（GGML 码 13）。
    Q5_K = 13,
    /// 6 位 K 超块（GGML 码 14）。
    Q6_K = 14,
}

/// 一个已支持格式的布局来源：标量元素按 [`DType`] 计宽，块格式全部委托 [`QType`]。
///
/// 几何（`elems`/`span`/`bytes_for`）只有一份实现，住在 `dtype::Layout<B>`：这里只把块参数固化成
/// [`QType`]，标量手臂与块手臂的几何都不再另写一份。
pub type Layout = dtype::Layout<QType>;

/// 本解码器**已实现**反量化的格式表：GGML 码 → 布局来源。
///
/// 这是「支持哪些格式」的唯一事实：[`Kind::from_code`]、[`Kind::elems`] 与 [`Kind::span`]
/// 都读这张表；新增一种可解码格式只加一行。`dtype` 的 [`QType`] 声明了 25 个块格式，
/// 这里只实现其中 4 个，其余 21 个尚无解码器——这张表让「表里有、解不了」成为可读清单，
/// 而不是散在几个 `match` 里的隐式事实。
pub const SUPPORTED: &[(Kind, Layout)] = &[
    (Kind::F32, Layout::Scalar(DType::F32)),
    (Kind::F16, Layout::Scalar(DType::F16)),
    (Kind::Q8_0, Layout::Block(QType::Q8_0)),
    (Kind::Q4_K, Layout::Block(QType::Q4_K)),
    (Kind::Q5_K, Layout::Block(QType::Q5_K)),
    (Kind::Q6_K, Layout::Block(QType::Q6_K)),
];

impl Kind {
    /// 磁盘类型码。
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// 由磁盘码解析；未支持的码显式失败。
    pub fn from_code(code: u32) -> Result<Self> {
        SUPPORTED
            .iter()
            .find_map(|(kind, _)| (kind.code() == code).then_some(*kind))
            .ok_or_else(|| bad("不支持的 ggml 类型码"))
    }

    /// 布局来源；每个变体都必须在 [`SUPPORTED`] 里出现（由 `supported_table_covers_kind` 锁定）。
    pub fn layout(self) -> Result<Layout> {
        SUPPORTED
            .iter()
            .find(|(kind, _)| *kind == self)
            .map(|(_, layout)| *layout)
            .ok_or_else(|| {
                Error::new(Code::Internal, "Kind 的每个变体都必须在 SUPPORTED 里").with("gguf")
            })
    }

    /// 每块元素数：块格式取 [`QType::block_len`]，标量元素按 1。
    pub fn elems(self) -> Result<usize> {
        Ok(self.layout()?.elems() as usize)
    }

    /// 每块字节数：块格式取 [`QType::block_bytes`]，标量元素取 [`DType::bytes_for`] 的单元素值。
    pub fn span(self) -> Result<usize> {
        Ok(self.layout()?.span() as usize)
    }
}

/// 元数据表：键值对按书写顺序。
pub(crate) type Meta = Vec<(String, Value)>;

/// 元数据值：整数按有无符号归类，浮点统一 f64。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 无符号整数。
    Uint(u64),
    /// 有符号整数。
    Int(i64),
    /// 浮点（f32 升 f64）。
    Float(f64),
    /// 布尔。
    Bool(bool),
    /// UTF-8 字符串。
    Str(String),
    /// 同类型数组。
    Arr(Vec<Value>),
}

impl Value {
    /// 字符串视图。
    pub fn text(&self) -> Option<&str> {
        match self {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    /// 无符号整数视图。
    pub fn uint(&self) -> Option<u64> {
        match self {
            Value::Uint(value) => Some(*value),
            _ => None,
        }
    }

    /// 有符号整数视图。
    pub fn int(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// 浮点视图。
    pub fn float(&self) -> Option<f64> {
        match self {
            Value::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// 布尔视图。
    pub fn bool(&self) -> Option<bool> {
        match self {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// 数组视图。
    pub fn arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// 字符串数组：元素必须同为 [`Value::Str`]，混型或缺失返回 `None`。
    pub fn text_array(&self) -> Option<Vec<&str>> {
        self.arr()?.iter().map(Value::text).collect()
    }

    /// 浮点数组：元素必须同为 [`Value::Float`]，混型或缺失返回 `None`。
    pub fn float_array(&self) -> Option<Vec<f64>> {
        self.arr()?.iter().map(Value::float).collect()
    }
}

/// 张量信息表条目；offset 相对数据区起点。
#[derive(Debug, Clone)]
pub struct Tensor {
    /// 张量名。
    pub name: String,
    /// 元素类型。
    pub kind: Kind,
    /// 形状。
    pub shape: Vec<u64>,
    /// 数据区相对偏移。
    pub offset: u64,
}

impl Tensor {
    /// 元素总数，乘法溢出显式失败。
    pub fn count(&self) -> Result<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &dim| {
                acc.checked_mul(usize::try_from(dim).ok()?)
            })
            .ok_or_else(|| bad("张量形状溢出"))
    }

    /// 磁盘字节数；量化块必须整除元素数。
    pub fn bytes(&self) -> Result<usize> {
        let count = self.count()?;
        let elem = self.kind.elems()?;
        self.blocks(count, elem)
    }

    /// 块数 × 每块字节数；块大小必须整除元素数，乘法溢出显式失败。
    fn blocks(&self, count: usize, elem: usize) -> Result<usize> {
        if !count.is_multiple_of(elem) {
            return Err(bad("量化块大小不整除元素数"));
        }
        (count / elem)
            .checked_mul(self.kind.span()?)
            .ok_or_else(|| bad("张量大小溢出"))
    }
}

/// 已解析的 GGUF 文件，持有调用方字节切片的借用。
pub struct Gguf<'a> {
    bytes: &'a [u8],
    meta: Vec<(String, Value)>,
    tensors: Vec<Tensor>,
    data: usize,
}

impl<'a> Gguf<'a> {
    /// 解析头部、KV 与张量信息表，定位数据区。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (tensors, meta) = tables(&mut reader)?;
        let data = data_start(&meta, reader.pos, bytes.len())?;
        Ok(Self {
            bytes,
            meta,
            tensors,
            data,
        })
    }

    /// 按 key 取元数据。
    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.meta
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// 全部元数据。
    #[must_use]
    pub fn metas(&self) -> &[(String, Value)] {
        &self.meta
    }

    /// 全部张量信息。
    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// 按名取张量信息。
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// `general.name`。
    pub fn name(&self) -> Option<&str> {
        self.meta("general.name").and_then(Value::text)
    }

    /// `general.architecture`。
    pub fn arch(&self) -> Option<&str> {
        self.meta("general.architecture").and_then(Value::text)
    }

    /// 张量原始字节视图。
    pub fn raw(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let offset = usize::try_from(tensor.offset).ok();
        let start = offset
            .and_then(|offset| self.data.checked_add(offset))
            .ok_or_else(|| bad("张量偏移溢出"))?;
        let len = tensor.bytes()?;
        self.bytes
            .get(start..start.saturating_add(len))
            .ok_or_else(|| bad("张量字节越界"))
    }

    /// 按名反量化为 f32。
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.named(name)?;
        let raw = self.raw(tensor)?;
        let count = tensor.count()?;
        let mut out = vec![0.0f32; count];
        dequantize(tensor.kind, raw, &mut out)?;
        Ok(out)
    }

    /// 按名取张量信息；缺失时显式报错（[`Gguf::tensor`] 的 `Option` 留给调用方自己决定）。
    fn named(&self, name: &str) -> Result<&Tensor> {
        self.tensor(name)
            .ok_or_else(|| bad(format!("无张量 `{name}`")))
    }
}

#[cfg(test)]
#[path = "../tests/gguf.rs"]
mod tests;
