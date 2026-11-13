//! 增量补丁：逐字节 XOR 差值，读端按 base 自逆合成。
//!
//! XOR 是自逆的（`join` 也是一次 XOR），且不需要任何额外标记：base 有同名等长段就 XOR，
//! 否则该段按全量保留。两侧判定对称——
//! - 同名等长：`diff` 存 `cur ^ base`，`join` 得 `(cur ^ base) ^ base = cur`；
//! - base 缺失/不等长：`diff` 存全量 `cur`，`join` 直接采纳。
//!
//! 训练前后张量表不变、仅数值微调时，XOR 结果大量为零，再经 [`crate::Writer`] 的 rans 压到很小。

use diagnostic::Result;

use crate::reader::Archive;
use crate::seg::Segment;

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
        out.push(Segment::new(
            seg.name(),
            seg.dtype(),
            seg.shape().to_vec(),
            seg.quant().cloned(),
            data,
        )?);
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
mod tests {
    use super::*;
    use crate::reader::Archive;
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
        assert_eq!(patch[0].name, "b");
        assert_eq!(patch[0].data, f32_segment("b", &[9.0]).data);
    }
}
