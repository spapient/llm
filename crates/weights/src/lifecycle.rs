//! 容器生命周期模式与增量补丁：训练/发布判定、`opt.` 状态剥离、逐字节 XOR 补丁。
//!
//! 模式判定与剥离只依赖已读 [`Archive`]；补丁是自逆 XOR：base 缺失或不等长按全量保留，
//! 不做第二份容错。

use diagnostic::{Code, Error, Result};

use crate::{Archive, Segment, Stored, Wire};

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
///
/// 有标记段但码位不在闭集内时显式报错，不静默落进推理模式：未知码通常意味着文件损坏
/// 或版本错位，按推理放行会丢掉训练状态段。
pub fn mode_of(archive: &Archive<'_>) -> Result<Mode> {
    match archive
        .get(MODE_NAME)
        .and_then(|seg| archive.read(seg).first().copied())
    {
        None => Ok(Mode::Infer),
        Some(code) => Mode::from_code(code),
    }
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

/// 把读端段还原成写端段：沿用名字/类型/形状/量化方案，`data` 是已就位的逻辑字节。
/// `release` 与补丁重建共用这一份字段装配，避免四处各拼一遍 `Segment::new`。
fn to_segment(seg: &Stored, data: Vec<u8>) -> Result<Segment> {
    Segment::new(
        seg.name(),
        seg.dtype(),
        seg.shape().to_vec(),
        seg.quant().cloned(),
        data,
    )
}

/// 把读端段还原成写端段：先加载逻辑字节，再交 [`to_segment`] 装配。
fn keep_segment(src: &Archive<'_>, seg: &Stored) -> Result<Segment> {
    let data = src.load(seg)?.into_owned();
    to_segment(seg, data)
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("mode")
}

// —— 增量补丁：逐字节 XOR，读端按 base 自逆合成 ——

/// 计算 cur 相对 base 的补丁段（逻辑字节，未压缩；交给 Writer 落盘并按需 rans）。
pub fn diff(base: &Archive<'_>, cur: &Archive<'_>) -> Result<Vec<Segment>> {
    rebuild(base, cur)
}

/// 用 base 与补丁归档合成完整段。
pub fn join(base: &Archive<'_>, patch: &Archive<'_>) -> Result<Vec<Segment>> {
    rebuild(base, patch)
}

/// diff 与 join 同构：对 right 每个段，base 有同名等长段就 XOR，否则按全量保留。
fn rebuild(base: &Archive<'_>, right: &Archive<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    for seg in right.segments() {
        let bytes = right.load(seg)?.into_owned();
        let data = apply(base, seg.name(), bytes)?;
        out.push(to_segment(seg, data)?);
    }
    Ok(out)
}

/// 同名等长则 XOR；base 缺失或长度不同则原样返回（全量）。
fn apply(base: &Archive<'_>, name: &str, bytes: Vec<u8>) -> Result<Vec<u8>> {
    let Some(prev) = base.get(name) else {
        return Ok(bytes);
    };
    let prev_bytes = base.load(prev)?;
    if prev_bytes.len() != bytes.len() {
        return Ok(bytes);
    }
    Ok(xor(&bytes, &prev_bytes))
}

fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

#[cfg(test)]
mod mode_tests {
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
        let names: Vec<&str> = released.iter().map(|s| s.name()).collect();
        assert!(names.contains(&"w"));
        assert!(!names.iter().any(|n| is_opt(n)));

        let released_bytes = archive(released);
        let released_archive = Archive::parse(&released_bytes).expect("x");
        assert_eq!(mode_of(&released_archive).expect("模式"), Mode::Infer);
    }

    #[test]
    fn missing_marker_defaults_infer() {
        let bytes = archive(vec![f32_segment("w", &[1.0])]);
        assert_eq!(
            mode_of(&Archive::parse(&bytes).expect("a")).expect("模式"),
            Mode::Infer
        );
    }

    #[test]
    fn train_marker_roundtrips() {
        let mut segments = vec![f32_segment("w", &[1.0])];
        segments.push(Mode::Train.to_segment().expect("标记"));
        let bytes = archive(segments);
        let parsed = Archive::parse(&bytes).expect("a");
        assert_eq!(mode_of(&parsed).expect("模式"), Mode::Train);
    }

    #[test]
    fn an_unknown_mode_code_is_rejected_not_fallen_back_to_infer() {
        let bad = Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![7]).expect("段");
        let bytes = archive(vec![f32_segment("w", &[1.0]), bad]);
        let parsed = Archive::parse(&bytes).expect("a");
        let error = mode_of(&parsed).expect_err("未知码必须报错");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("mode"), "{error}");
    }
}

#[cfg(test)]
mod patch_tests {
    use super::*;
    use crate::testutil::{archive, f32_segment};
    use crate::writer::Writer;

    #[test]
    fn xor_roundtrip_recovers_cur() {
        let base_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.0])]);
        let base = Archive::parse(&base_bytes).expect("base");
        let cur_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.5])]);
        let cur = Archive::parse(&cur_bytes).expect("cur");

        let patch_bytes = archive(diff(&base, &cur).expect("diff"));
        let patch = Archive::parse(&patch_bytes).expect("patch");

        let joined = join(&base, &patch).expect("join");
        let expected = cur.load(&cur.segments()[0]).expect("cur 字节");
        assert_eq!(joined[0].data, &expected[..]);
    }

    #[test]
    fn small_change_patch_is_smaller() {
        let base_bytes = archive(vec![f32_segment("w", &[1.0; 4096])]);
        let base = Archive::parse(&base_bytes).expect("base");
        let mut changed = vec![1.0f32; 4096];
        changed[10] = 1.0001;
        let cur_bytes = archive(vec![f32_segment("w", &changed)]);
        let cur = Archive::parse(&cur_bytes).expect("cur");

        let mut w = Writer::new();
        w.compress(true);
        for seg in diff(&base, &cur).expect("diff") {
            w.push(seg).expect("push");
        }
        assert!(w.finish(6).expect("patch").len() < cur_bytes.len());
    }

    #[test]
    fn missing_base_is_full_not_garbage() {
        let base_bytes = archive(vec![f32_segment("a", &[1.0])]);
        let base = Archive::parse(&base_bytes).expect("base");
        let cur_bytes = archive(vec![f32_segment("b", &[9.0])]);
        let cur = Archive::parse(&cur_bytes).expect("cur");

        let patch = diff(&base, &cur).expect("diff");
        assert_eq!(patch[0].name(), "b");
        assert_eq!(patch[0].data, f32_segment("b", &[9.0]).data);
    }
}
