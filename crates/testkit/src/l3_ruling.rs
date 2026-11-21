//! `Ruling::Lossless` 的实测先决条件：真实容器的往返必须逐位还原。
//!
//! `ε = 0` 只是定理侧的一项；"产物确实还原"只能由写盘 → 解析 → 逐段解码之后的字节比较给出。
//! 往返走 `weights::Writer`/`Archive` 本体，比的是落盘字节读回来的结果，不是一份"应该相等"的
//! 预期；反方向（同样 `ε = 0` 但往返不逐位 → `Unverified`）属纯代数，由 `weights::l3` 的单测守。

use weights::l3::{Chain, Ruling};
use weights::{Archive, Segment, Stored, Wire, Writer};

use super::l3_measure::{frobenius, rung, scaled};

#[test]
fn a_bitwise_artifact_round_trip_completes_the_lossless_ruling() {
    let w1 = scaled(64, 6, 0.25);
    let w2 = scaled(64, 7, 0.25);

    let mut writer = Writer::new();
    for (name, values) in [("layer.0.w", &w1), ("layer.1.w", &w2)] {
        writer.push(segment(name, values)).expect("段名与载荷自洽");
    }
    let bytes = writer.finish(6).expect("落盘");
    let archive = Archive::parse(&bytes).expect("解析");

    let decoded: Vec<Vec<f32>> = archive
        .segments()
        .iter()
        .map(|stored| decode(&archive, stored))
        .collect();
    let bitwise = decoded == [w1.clone(), w2.clone()];
    assert!(bitwise, "无损容器的往返必须逐位相等");

    // 两项先决条件都成立才叫无损：`ε = 0`（定理项）**且**产物逐位还原（这里量出来的实测项）。
    let chain = Chain::new(
        frobenius(&w1),
        vec![rung(&w1, &w1, 1.0, 0.0), rung(&w2, &w2, 1.0, 0.0)],
    )
    .expect("链");
    assert_eq!(
        chain.bracket().expect("界").ruling(bitwise),
        Ruling::Lossless
    );
}

/// 一段 F32 权重：`Wire::F32` 的载荷就是小端字面量，没有第二套编码。
fn segment(name: &str, values: &[f32]) -> Segment {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let shape = vec![values.len() as u64];
    Segment::new(name, Wire::F32, shape, None, bytes).expect("段")
}

/// 逐段解码回权重；`F32` 段的字节数必须是 4 的整数倍，否则格式本身就不自洽。
fn decode(archive: &Archive<'_>, stored: &Stored) -> Vec<f32> {
    let raw = archive.load(stored).expect("解码");
    let (words, rest) = raw.as_chunks::<4>();
    assert!(rest.is_empty(), "F32 段的载荷长度必须是 4 的倍数");
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}
