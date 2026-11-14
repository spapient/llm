//! 容器双模式：训练断点 / 推理发布。
//!
//! 不新增二进制机制：模式写在保留名段 [`MODE_NAME`]，优化器状态是 `opt.` 命名空间下的普通段，
//! 完整复用块表、CRC 与压缩。发布即一次过滤写入，硬件无关。无该段的旧文件按推理模式读取。

use diagnostic::{Code, Error, Result};

use crate::Wire;
use crate::reader::Archive;
use crate::seg::Segment;

/// 模式标记段的保留名。
pub const MODE_NAME: &str = "llm.mode";
/// 优化器状态段的命名空间前缀。
const OPT_PREFIX: &str = "opt.";

/// 容器生命周期模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 训练断点：含优化器/动量/方差等 `opt.` 状态。
    Train,
    /// 推理发布：剥离训练状态的精简产物。
    Infer,
}

impl Mode {
    fn code(self) -> u8 {
        match self {
            Mode::Train => 0,
            Mode::Infer => 1,
        }
    }

    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Mode::Train),
            1 => Ok(Mode::Infer),
            other => Err(bad(&format!("mode: 未知码 {other}"))),
        }
    }

    /// 编码为可写进容器的标记段：保留名、`U8` 单轴、1 字节。
    pub fn to_segment(self) -> Result<Segment> {
        Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![self.code()])
    }
}

/// 读取模式；无标记段的旧文件按推理模式处理。
pub fn mode_of(archive: &Archive<'_>) -> Mode {
    archive
        .get(MODE_NAME)
        .and_then(|seg| archive.read(seg).first().copied())
        .map_or(Mode::Infer, |code| {
            Mode::from_code(code).unwrap_or(Mode::Infer)
        })
}

/// 段名是否属于优化器状态。
pub fn is_opt(name: &str) -> bool {
    name.starts_with(OPT_PREFIX)
}

/// 发布转换：剥离所有 `opt.` 段，保留其余段并标记为推理模式。
pub fn release(src: &Archive<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    for seg in src.segments() {
        if !is_opt(seg.name()) {
            out.push(keep_segment(src, seg)?);
        }
    }
    out.push(Mode::Infer.to_segment()?);
    Ok(out)
}

/// 把读端段还原成写端段：加载逻辑字节，沿用名字/类型/形状/量化方案。
fn keep_segment(src: &Archive<'_>, seg: &crate::reader::Stored) -> Result<Segment> {
    let data = src.load(seg)?.into_owned();
    Segment::new(
        seg.name(),
        seg.dtype(),
        seg.shape().to_vec(),
        seg.quant().cloned(),
        data,
    )
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("mode")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{archive, f32_segment};

    #[test]
    fn release_strips_opt_and_marks_infer() {
        let bytes = archive(vec![
            f32_segment("w", &[1.0, 2.0]),
            f32_segment("opt.m.w", &[0.5, 0.5]),
        ]);
        let src = Archive::parse(&bytes).expect("src");

        let released = release(&src).expect("release");
        let names: Vec<&str> = released.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"w"));
        assert!(!names.iter().any(|n| is_opt(n)));

        let released_bytes = archive(released);
        let released_archive = Archive::parse(&released_bytes).expect("x");
        assert_eq!(mode_of(&released_archive), Mode::Infer);
    }

    #[test]
    fn missing_marker_defaults_infer() {
        let bytes = archive(vec![f32_segment("w", &[1.0])]);
        assert_eq!(mode_of(&Archive::parse(&bytes).expect("a")), Mode::Infer);
    }

    #[test]
    fn train_marker_roundtrips() {
        let mut segments = vec![f32_segment("w", &[1.0])];
        segments.push(Mode::Train.to_segment().expect("标记"));
        let bytes = archive(segments);
        let parsed = Archive::parse(&bytes).expect("a");
        assert_eq!(mode_of(&parsed), Mode::Train);
    }
}
