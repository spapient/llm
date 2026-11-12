//! 端到端契约：全 wire 类型往返、量化段往返、对齐布局，以及两级 CRC 对损坏/截断的拦截。

use llm_pack::{Archive, Quant, Segment, Wire, Writer};

/// 给每个 wire 类型造一个 `n` 元素的段，数据填递增字节，长度正好是形状推出的字节数。
fn segment_of(wire: Wire, index: usize) -> Segment {
    let numel = 11u64; // 奇数，顺带钉亚字节类型的向上取整
    let bytes = wire.element_bytes(numel) as usize;
    let data: Vec<u8> = (0..bytes)
        .map(|b| (b as u8).wrapping_add(index as u8))
        .collect();
    Segment::new(format!("seg_{index}"), wire, vec![numel], None, data).expect("段自洽")
}

#[test]
fn every_wire_type_round_trips_byte_for_byte() {
    let mut writer = Writer::new();
    for (index, wire) in Wire::ALL.iter().enumerate() {
        writer.push(segment_of(*wire, index)).expect("追加");
    }
    let bytes = writer.finish(6).expect("64 B 对齐");
    let archive = Archive::parse(&bytes).expect("自证合法");
    assert_eq!(archive.segments().len(), Wire::ALL.len());

    for (index, wire) in Wire::ALL.iter().enumerate() {
        let expected = segment_of(*wire, index);
        let stored = archive.get(&format!("seg_{index}")).expect("按名查");
        assert_eq!(stored.dtype(), *wire);
        assert_eq!(stored.shape(), &[11]);
        assert_eq!(archive.read(stored), expected.data.as_slice());
        // 容器存储与 tensor 映射的口径在两端一致。
        assert_eq!(stored.dtype().to_tensor(), wire.to_tensor());
    }
}

#[test]
fn a_quantized_segment_keeps_its_scheme() {
    let quant = Quant::new(1, 0, 32, false, Wire::F16, None).expect("per-block");
    let segment = Segment::new(
        "blk.0.q.weight",
        Wire::U8,
        vec![8, 64],
        Some(quant),
        vec![3u8; 512],
    )
    .expect("量化段");

    let mut writer = Writer::new();
    writer.push(segment).expect("追加");
    let bytes = writer.finish(8).expect("256 B 对齐");
    let archive = Archive::parse(&bytes).expect("解析");
    let stored = archive.get("blk.0.q.weight").expect("段存在");
    assert_eq!(stored.quant(), Some(&quant));
    assert_eq!(archive.read(stored), vec![3u8; 512].as_slice());
}

#[test]
fn a_flipped_bit_is_rejected_by_the_global_crc() {
    let bytes = Writer::new()
        .push(segment_of(Wire::F32, 0))
        .expect("追加")
        .finish(6)
        .expect("产出");

    let mut damaged = bytes.clone();
    let flip = damaged.len() / 2;
    damaged[flip] ^= 0x01;
    let error = Archive::parse(&damaged).expect_err("比特翻转必须被拦");
    assert_eq!(error.code().as_str(), "invalid");
}

#[test]
fn a_truncated_file_is_rejected() {
    let bytes = Writer::new()
        .push(segment_of(Wire::F16, 0))
        .expect("追加")
        .finish(6)
        .expect("产出");
    let cut = &bytes[..bytes.len() - 7];
    assert!(Archive::parse(cut).is_err());
}

#[test]
fn checksums_can_be_disabled_and_then_a_flip_survives() {
    let mut writer = Writer::new();
    writer.checksum(false);
    writer.push(segment_of(Wire::F32, 0)).expect("追加");
    let bytes = writer.finish(6).expect("产出");

    let source = Archive::parse(&bytes).expect("原文件合法");
    let start = source.segments()[0].start();
    let mut damaged = bytes.clone();
    damaged[start + 1] ^= 0x01; // 精确翻数据段逻辑字节，不碰头部与对齐填充
    assert!(Archive::parse(&damaged).is_ok(), "关闭校验后读端不验 CRC");
}
