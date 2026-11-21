use super::*;
use std::mem::size_of;

#[test]
fn discriminant_fits_one_byte() {
    assert_eq!(size_of::<QType>(), 1, "repr(u8) 加 25 个格式应正好一字节");
}

#[test]
fn layout_table_is_locked() {
    // 一行一个格式：规范名、块长、块字节、标称位宽、scale 形状、编码族。整表与 `ALL` 同序，
    // 长度由 `ALL.len()` 锚定。反例就在表里：`q8_k` 属 KQuant 却用 F32 scale，
    // `iq4_nl` 属 Grid 却只有单个 f16 d——这正是 `Scale` 与 `Family` 分开的理由。
    const LOCKED: [(&str, u32, u32, u32, Scale, Family); QType::ALL.len()] = [
        ("q1_0", 128, 18, 1, Scale::F16, Family::Plain),
        ("q2_0", 64, 18, 2, Scale::F16, Family::Plain),
        ("q4_0", 32, 18, 4, Scale::F16, Family::Plain),
        ("q4_1", 32, 20, 4, Scale::F16Pair, Family::Plain),
        ("q5_0", 32, 22, 5, Scale::F16, Family::Plain),
        ("q5_1", 32, 24, 5, Scale::F16Pair, Family::Plain),
        ("q8_0", 32, 34, 8, Scale::F16, Family::Plain),
        ("q8_1", 32, 36, 8, Scale::F16Pair, Family::Plain),
        ("q2_k", 256, 84, 2, Scale::SubScale, Family::KQuant),
        ("q3_k", 256, 110, 3, Scale::SubScale, Family::KQuant),
        ("q4_k", 256, 144, 4, Scale::SubScale, Family::KQuant),
        ("q5_k", 256, 176, 5, Scale::SubScale, Family::KQuant),
        ("q6_k", 256, 210, 6, Scale::SubScale, Family::KQuant),
        ("q8_k", 256, 292, 8, Scale::F32, Family::KQuant),
        ("iq1_s", 256, 50, 1, Scale::F16, Family::Grid),
        ("iq1_m", 256, 56, 1, Scale::SubScale, Family::Grid),
        ("iq2_xxs", 256, 66, 2, Scale::F16, Family::Grid),
        ("iq2_xs", 256, 74, 2, Scale::SubScale, Family::Grid),
        ("iq2_s", 256, 82, 2, Scale::SubScale, Family::Grid),
        ("iq3_xxs", 256, 98, 3, Scale::F16, Family::Grid),
        ("iq3_s", 256, 110, 3, Scale::SubScale, Family::Grid),
        ("iq4_xs", 256, 136, 4, Scale::SubScale, Family::Grid),
        ("iq4_nl", 32, 18, 4, Scale::F16, Family::Grid),
        ("tq1_0", 256, 54, 1, Scale::F16, Family::Ternary),
        ("tq2_0", 256, 66, 2, Scale::F16, Family::Ternary),
    ];
    for (qtype, (name, block_len, block_bytes, bits, scale, family)) in
        QType::ALL.iter().zip(LOCKED)
    {
        let spec = qtype.spec();
        assert_eq!(
            (
                spec.name,
                spec.block_len,
                spec.block_bytes,
                spec.bits,
                spec.scale,
                spec.family,
            ),
            (name, block_len, block_bytes, bits, scale, family),
            "{qtype:?} 的布局行被改动"
        );
    }
}

#[test]
fn accessors_read_the_layout_row() {
    // 行表锁的是字段，公开访问器锁的是「取哪一列」：`bits()` 与 `block_bytes()` 同为
    // `u32`，把列取错不会让任何一条别的测试变红。`scale()`/`family()` 在该文件外没有
    // 任何读取点，只能由这里对齐。
    for qtype in QType::ALL {
        let spec = qtype.spec();
        assert_eq!(qtype.name(), spec.name);
        assert_eq!(qtype.block_len(), spec.block_len);
        assert_eq!(qtype.block_bytes(), spec.block_bytes);
        assert_eq!(qtype.bits(), spec.bits);
        assert_eq!(qtype.scale(), spec.scale);
        assert_eq!(qtype.family(), spec.family);
    }
}

#[test]
fn block_bytes_cover_the_payload() {
    // 「事实是这个数」由上一张表锁定；这里只锁结构性下界，与公式的写法无关。
    for qtype in QType::ALL {
        assert!(
            u64::from(qtype.block_bytes()) * 8
                >= u64::from(qtype.block_len()) * u64::from(qtype.bits()),
            "{} 的字节装不下元素位",
            qtype.name()
        );
    }
}

#[test]
fn bytes_for_pads_to_whole_blocks() {
    assert_eq!(QType::Q4_K.bytes_for(256), 144);
    assert_eq!(QType::Q4_K.bytes_for(255), 144, "不足一块按整块计");
    assert_eq!(QType::Q4_K.bytes_for(512), 288);
    assert_eq!(QType::Q8_0.bytes_for(64), 68);
    assert_eq!(QType::Q1_0.bytes_for(128), 18);
    assert_eq!(QType::Q8_K.bytes_for(256), 292);
}
