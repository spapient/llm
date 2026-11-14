//! 测试共享夹具：f32 段构造与归档写盘，mode/patch 等模块的测试共用同一份。

use crate::Wire;
use crate::seg::Segment;
use crate::writer::Writer;

/// 由 f32 值构造一个单维 F32 段。
#[cfg(test)]
pub(crate) fn f32_segment(name: &str, vals: &[f32]) -> Segment {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Segment::new(name, Wire::F32, vec![vals.len() as u64], None, bytes).expect("段")
}

/// 把段依次写进归档，返回对齐到 `2^6` 的字节。
#[cfg(test)]
pub(crate) fn archive(segments: Vec<Segment>) -> Vec<u8> {
    let mut w = Writer::new();
    for seg in segments {
        w.push(seg).expect("push");
    }
    w.finish(6).expect("字节")
}
