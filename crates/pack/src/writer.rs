//! 写端：收集段，按对齐布局一次产出 canonical 字节。
//!
//! 数据段在描述符之后顺序写出，每段先填零对齐；段级 CRC 覆盖逻辑字节，全局 CRC 覆盖从头到
//! 数据区末尾。写端不缓冲"整份文件大小"的副本——段的字节本来就在 [`crate::seg::Segment`] 里，
//! 这里只做一次顺序拼接。

use diagnostic::{Code, Error, Result};

use crate::crc;
use crate::leb;
use crate::seg::Segment;

/// 魔数 `"LLPK"`。
pub const MAGIC: u32 = 0x4B50_4C4C;

/// 当前格式版本。
pub const VERSION: u16 = 1;

/// `flags` 位：全局 CRC 存在。
pub const GLOBAL_CRC: u32 = 1 << 0;

/// `flags` 位：段级 CRC 存在。
pub const SEG_CRC: u32 = 1 << 1;

/// 最大对齐指数（2^16 = 64 KiB）。
pub const MAX_ALIGN_LOG: u8 = 16;

/// 权重写者。
#[derive(Debug, Default)]
pub struct Writer {
    segs: Vec<Segment>,
    checksum: bool,
}

impl Writer {
    /// 空写者，默认开启两级 CRC。
    pub fn new() -> Self {
        Self {
            segs: Vec::new(),
            checksum: true,
        }
    }

    /// 开关两级校验和（默认开）。
    pub fn checksum(&mut self, on: bool) -> &mut Self {
        self.checksum = on;
        self
    }

    /// 追加一个已自洽的段；重名在这里被拒。
    pub fn push(&mut self, segment: Segment) -> Result<&mut Self> {
        segment.validate()?;
        if self.segs.iter().any(|known| known.name == segment.name) {
            return Err(
                Error::new(Code::Invalid, format!("段 `{}` 重复", segment.name))
                    .with("pack::Writer::push"),
            );
        }
        self.segs.push(segment);
        Ok(self)
    }

    /// 段数。
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// 是否没有段。
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// 产出容器字节；`align_log` 是对齐边界的 2 的幂指数（0..=16，常用 6 = 64 B）。
    pub fn finish(&self, align_log: u8) -> Result<Vec<u8>> {
        if align_log > MAX_ALIGN_LOG {
            return Err(Error::new(
                Code::Invalid,
                format!("align_log {align_log} 超过 {MAX_ALIGN_LOG}"),
            )
            .with("pack::Writer::finish"));
        }
        let flags = if self.checksum {
            GLOBAL_CRC | SEG_CRC
        } else {
            0
        };

        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.push(align_log);
        leb::write_u32(&mut buf, self.segs.len() as u32);

        // 描述符：段级 CRC 在此算好并写进头部。
        for segment in &self.segs {
            let seg_crc = if self.checksum {
                crc::crc32(&segment.data)
            } else {
                0
            };
            segment.encode_desc(&mut buf, seg_crc)?;
        }

        // 数据区：顺序对齐 + 写逻辑字节。
        let align = 1usize << align_log;
        for segment in &self.segs {
            let target = align_up(buf.len(), align)?;
            buf.resize(target, 0);
            buf.extend_from_slice(&segment.data);
        }

        // 全局 CRC 覆盖从魔数到数据区末尾（含对齐填充）。
        let global = if self.checksum { crc::crc32(&buf) } else { 0 };
        buf.extend_from_slice(&global.to_le_bytes());
        Ok(buf)
    }
}

/// 向上对齐到 2 的幂边界。
fn align_up(pos: usize, align: usize) -> Result<usize> {
    let mask = align - 1;
    pos.checked_add(mask)
        .map(|wrapped| wrapped & !mask)
        .ok_or_else(|| Error::new(Code::Invalid, "对齐位置超出可寻址范围").with("pack::Writer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Wire;
    use crate::reader::Archive;
    use crate::seg::Segment;

    fn seg(name: &str, data: Vec<u8>) -> Segment {
        Segment::new(name, Wire::F32, vec![data.len() as u64 / 4], None, data).expect("自洽")
    }

    #[test]
    fn an_empty_container_round_trips_with_a_footer() {
        let bytes = Writer::new().finish(6).expect("空容器");
        // 固定头 11 + 1 字节段计数(0) + 4 字节全局 CRC；CRC 覆盖头部，本就非 0。
        assert_eq!(bytes.len(), 11 + 1 + 4);
        assert!(Archive::parse(&bytes).is_ok(), "空容器能被解析回");
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut writer = Writer::new();
        writer.push(seg("a", vec![0u8; 4])).expect("首次");
        assert!(writer.push(seg("a", vec![0u8; 4])).is_err());
    }

    #[test]
    fn a_bad_align_log_is_rejected() {
        assert!(Writer::new().finish(17).is_err());
    }
}
