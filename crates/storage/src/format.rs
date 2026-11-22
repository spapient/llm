//! 字节格式的版本与恢复事实：`LLMC` 魔数、当前版本号，以及恢复后必须逐项相同的五个数。

use diagnostic::{Code, Error, Result};

use crate::codec::need;

/// 检查点文件头：4 字节魔数 + 2 字节小端版本号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format;

impl Format {
    /// 魔数 `LLMC`。
    pub const MAGIC: [u8; 4] = *b"LLMC";

    /// 当前格式版本：字段集、编码顺序或校验方式变化时必须递增；版本之间不兼容，只支持当前版本。
    pub const VERSION: u16 = 5;

    /// 文件头字节数（魔数 + 版本）。
    pub const HEADER_BYTES: usize = Self::MAGIC.len() + 2;

    /// 尾部校验和字节数（小端 `u32`）。
    pub const FOOTER_BYTES: usize = 4;

    /// 校验文件头：魔数一致且版本相同，不符时返回 [`Code::Invalid`]。
    pub fn check_bytes(bytes: &[u8]) -> Result<()> {
        let magic = need(bytes, 0..Self::MAGIC.len(), 0)?;
        let raw = need(
            bytes,
            Self::MAGIC.len()..Self::HEADER_BYTES,
            Self::MAGIC.len(),
        )?;
        if magic != Self::MAGIC.as_slice() {
            let message = format!("检查点魔数不是 `LLMC`：{magic:?}");
            return Err(Error::new(Code::Invalid, message).with("storage::snapshot::Format::check"));
        }
        let version = u16::from_le_bytes([raw[0], raw[1]]);
        if version == Self::VERSION {
            return Ok(());
        }
        Err(version_error(version))
    }
}

/// 版本不符时说清是哪一边旧：读端旧要重存，写端旧要升级。
fn version_error(version: u16) -> Error {
    let current = Format::VERSION;
    let message = if version > current {
        format!("检查点版本 {version} 比当前版本 {current} 新：请用写它的那个版本读取")
    } else {
        format!(
            "检查点版本 {version} 比当前版本 {current} 旧：本 crate 只读第 {current} 版，请重存一次"
        )
    };
    Error::new(Code::Invalid, message).with("storage::snapshot::Format::check")
}

/// 一次训练恢复后必须逐项相同的事实：步号、并行度、层数与批几何（行、列整体比较才是判据）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    step: u64,
    world: usize,
    layers: usize,
    batch_rows: usize,
    batch_cols: usize,
}

impl Facts {
    /// 构造：并行度、层数与批行列必须非零（步号可以为 0）；任一为零时返回 [`Code::Invalid`]。
    pub fn new(
        step: u64,
        world: usize,
        layers: usize,
        batch_rows: usize,
        batch_cols: usize,
    ) -> Result<Self> {
        let zeros = [
            ("并行度", world),
            ("层数", layers),
            ("批行数", batch_rows),
            ("批列宽", batch_cols),
        ];
        for (name, value) in zeros {
            require_nonzero(name, value)?;
        }
        Ok(Self {
            step,
            world,
            layers,
            batch_rows,
            batch_cols,
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

    /// 批的列宽。
    pub fn batch_cols(&self) -> usize {
        self.batch_cols
    }
}

/// 一项恢复事实为零：把字段名带在消息里。
fn require_nonzero(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        let message = format!("检查点的{name}不能为 0");
        return Err(Error::new(Code::Invalid, message).with("storage::snapshot::Facts::new"));
    }
    Ok(())
}
