//! 检查点文件：[`Snapshot`] 把一次训练恢复所需的事实与权重条目表编成字节。
//!
//! **拥有**：`Format`、`Snapshot`。**不拥有**：训练循环状态（`train::Checkpoint` 是内存里的
//! 最小事实集，本 crate 只负责它怎么落盘、怎么读回）、参数数据（`weights::Header` 被引用，
//! 条目表不复制）、张量存储（`tensor`）、字节容器的去重与 CRC（`storage`）。
//! **依赖**：`weights`、`tensor`、`diagnostic`。**被依赖**：`train`。
//!
//! 解码不是"读出来就用"：`weights::Header` 会重新校验名字唯一、区间不重叠、起点不早于数据段，
//! 因此手改过的检查点不会等到恢复后的第 N 步才炸。
//!
//! # 例子
//!
//! ```
//! use llm_checkpoint::Snapshot;
//! use tensor::{DType, Shape};
//! use weights::{Entry, Header};
//!
//! let header = Header::new(
//!     vec![Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64)],
//!     64,
//! )
//! .expect("头部");
//! let snapshot = Snapshot::new(7, 2, 2, 3, header).expect("快照");
//! let bytes = snapshot.encode().expect("编码");
//! let restored = Snapshot::decode(&bytes).expect("解码");
//! assert_eq!(restored.step(), 7);
//! assert_eq!(restored.weights().entries().len(), 1);
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};
use weights::{Entry, Header};

/// 解码失败的上下文：错误只由这一层产生，排查时不必再猜是哪一段。
const CONTEXT: &str = "checkpoint::Snapshot::decode";

/// 检查点文件头：4 字节魔数 + 2 字节小端版本号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format;

impl Format {
    /// 魔数 `LLMC`。
    pub const MAGIC: [u8; 4] = *b"LLMC";

    /// 当前格式版本：事实集、条目表或编码顺序变化时必须递增。
    pub const VERSION: u16 = 1;

    /// 文件头字节数（魔数 + 版本）。
    pub const HEADER_BYTES: usize = Self::MAGIC.len() + 2;

    /// 校验文件头：魔数一致且版本相同。
    pub fn check(bytes: &[u8]) -> Result<()> {
        let magic = bytes.get(..Self::MAGIC.len()).ok_or_else(|| truncated(0))?;
        if magic != Self::MAGIC.as_slice() {
            return Err(
                Error::new(Code::Invalid, format!("检查点魔数不是 `LLMC`：{magic:?}"))
                    .with("checkpoint::Format::check"),
            );
        }
        let version = bytes
            .get(Self::MAGIC.len()..Self::HEADER_BYTES)
            .ok_or_else(|| truncated(Self::MAGIC.len()))?;
        let version = u16::from_le_bytes([version[0], version[1]]);
        if version != Self::VERSION {
            return Err(Error::new(
                Code::Invalid,
                format!("检查点版本 {version} 与当前版本 {} 不同", Self::VERSION),
            )
            .with("checkpoint::Format::check"));
        }
        Ok(())
    }
}

/// 一次落盘的快照：恢复所需的事实 + 权重条目表。
///
/// 四个事实必须在恢复后逐项相同，否则读出来的参数形状正确、内容却错位；因此它们和条目表
/// 一起进同一个文件，而不是各自找地方存。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    step: u64,
    world: usize,
    layers: usize,
    batch_rows: usize,
    weights: Header,
}

impl Snapshot {
    /// 构造：并行度、层数与批行数必须非零；权重条目表由 [`weights::Header`] 自证合法。
    pub fn new(
        step: u64,
        world: usize,
        layers: usize,
        batch_rows: usize,
        weights: Header,
    ) -> Result<Self> {
        for (name, value) in [("并行度", world), ("层数", layers), ("批行数", batch_rows)] {
            if value == 0 {
                return Err(Error::new(Code::Invalid, format!("检查点的{name}不能为 0"))
                    .with("checkpoint::Snapshot::new"));
            }
        }
        Ok(Self {
            step,
            world,
            layers,
            batch_rows,
            weights,
        })
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 保存时的并行度（恢复时必须一致）。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 训练层数。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// 权重条目表：解码时已经重新校验过偏移与重名。
    pub fn weights(&self) -> &Header {
        &self.weights
    }

    /// 编码成字节：文件头 → 事实集 → 权重数据段起点 → 条目表。
    pub fn encode(&self) -> Result<Vec<u8>> {
        let entries = self.weights.entries();
        let mut bytes = Vec::with_capacity(Format::HEADER_BYTES + 32 + entries.len() * 32);
        bytes.extend_from_slice(Format::MAGIC.as_slice());
        bytes.extend_from_slice(&Format::VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.step.to_le_bytes());
        bytes.extend_from_slice(&(self.world as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.layers as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.batch_rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.weights.data_start() as u64).to_le_bytes());
        let count = u32::try_from(entries.len()).map_err(|_| invalid("权重条目超过 u32 上限"))?;
        bytes.extend_from_slice(&count.to_le_bytes());
        for entry in entries {
            let name = entry.name().as_bytes();
            let len = u16::try_from(name.len()).map_err(|_| {
                invalid(format!("权重条目 `{}` 的名字超过 65535 字节", entry.name()))
            })?;
            bytes.extend_from_slice(&len.to_le_bytes());
            bytes.extend_from_slice(name);
            bytes.push(tag(entry.dtype()));
            let dims = entry.shape().dims();
            let rank = u16::try_from(dims.len())
                .map_err(|_| invalid(format!("权重条目 `{}` 的秩超过 65535", entry.name())))?;
            bytes.extend_from_slice(&rank.to_le_bytes());
            for dim in dims {
                bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            bytes.extend_from_slice(&(entry.offset() as u64).to_le_bytes());
        }
        Ok(bytes)
    }

    /// 从字节解码：文件头 → 事实集 → 权重数据段起点 → 条目表。
    ///
    /// 截断、版本不符、尾部多余字节都直接拒绝；条目表交给 [`weights::Header::new`] 重新校验，
    /// 解码路径与磁盘读取路径因此共用同一套判据。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Format::check(bytes)?;
        let mut cursor = Cursor::new(bytes, Format::HEADER_BYTES);
        let step = cursor.u64()?;
        let world = cursor.count()?;
        let layers = cursor.count()?;
        let batch_rows = cursor.count()?;
        let data_start = cursor.count()?;
        let count = cursor.u32()?;
        let mut entries = Vec::new();
        for index in 0..count {
            let len = usize::from(cursor.u16()?);
            let name = cursor.take(len)?;
            let name = std::str::from_utf8(name)
                .map_err(|error| invalid(format!("第 {index} 个条目名不是 UTF-8：{error}")))?;
            let dtype = dtype(cursor.u8()?)?;
            let rank = usize::from(cursor.u16()?);
            let mut dims = Vec::new();
            for _ in 0..rank {
                dims.push(cursor.count()?);
            }
            let offset = cursor.count()?;
            entries.push(Entry::new(name, dtype, Shape::new(dims)).at(offset));
        }
        if cursor.remaining() > 0 {
            return Err(invalid(format!(
                "检查点尾部多出 {} 字节",
                cursor.remaining()
            )));
        }
        let weights = Header::new(entries, data_start).map_err(|error| error.with(CONTEXT))?;
        Self::new(step, world, layers, batch_rows, weights)
    }
}

/// 逐字节读取器：每次读取都检查边界，越界时报出当前的偏移。
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| truncated(self.offset))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| truncated(self.offset))?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let raw = self.take(2)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let raw = self.take(8)?;
        Ok(u64::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
        ]))
    }

    fn count(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid("检查点的计数超出 usize 上限"))
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

/// 元素类型在字节里的标签；穷尽 `match` 保证新增 [`DType`] 变体时这里编译失败。
fn tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I8 => 3,
        DType::U4 => 4,
    }
}

fn dtype(tag: u8) -> Result<DType> {
    match tag {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::I8),
        4 => Ok(DType::U4),
        other => Err(invalid(format!("未知的元素类型标签 {other}"))),
    }
}

fn truncated(offset: usize) -> Error {
    Error::new(Code::Invalid, format!("检查点在偏移 {offset} 处截断")).with(CONTEXT)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(CONTEXT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Snapshot {
        let weights = Header::new(
            vec![
                Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([2, 3])).at(64),
                Entry::new("blk.0.scales", DType::U4, Shape::new([4])).at(88),
            ],
            64,
        )
        .expect("头部");
        Snapshot::new(7, 2, 2, 3, weights).expect("快照")
    }

    #[test]
    fn a_snapshot_round_trips_through_its_bytes() {
        let snapshot = snapshot();
        let bytes = snapshot.encode().expect("编码");
        let restored = Snapshot::decode(&bytes).expect("解码");
        assert_eq!(restored, snapshot);
        assert_eq!(restored.weights().entries().len(), 2);
        assert_eq!(restored.weights().entries()[1].name(), "blk.0.scales");
    }

    #[test]
    fn a_foreign_magic_is_rejected() {
        let mut bytes = snapshot().encode().expect("编码");
        bytes[0] = b'X';
        let error = Snapshot::decode(&bytes).expect_err("魔数不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("LLMC"), "{error}");
    }

    #[test]
    fn another_version_is_rejected() {
        let mut bytes = snapshot().encode().expect("编码");
        bytes[4] = 9;
        let error = Snapshot::decode(&bytes).expect_err("版本不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("版本"), "{error}");
    }

    #[test]
    fn a_truncated_snapshot_names_the_offset() {
        let bytes = snapshot().encode().expect("编码");
        let error = Snapshot::decode(&bytes[..bytes.len() - 3]).expect_err("截断");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("截断"), "{error}");
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = snapshot().encode().expect("编码");
        bytes.push(0);
        let error = Snapshot::decode(&bytes).expect_err("尾部多余");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("尾部"), "{error}");
    }

    #[test]
    fn a_snapshot_without_layers_is_rejected() {
        let weights = Header::new(Vec::new(), 0).expect("空头部");
        assert_eq!(
            Snapshot::new(0, 1, 0, 1, weights).unwrap_err().code(),
            Code::Invalid
        );
    }
}
