//! `rans` 模块的单元测试。

use super::*;



/// 任意字节往返：解码必须与输入逐字节相等。

fn assert_roundtrip(src: &[u8]) {

    let frame = encode(src);

    let back = decode(&frame, src.len()).expect("解码");

    assert_eq!(back, src, "往返不一致");

}



#[test]

fn empty_roundtrips() {

    assert_roundtrip(b"");

}



#[test]

fn single_byte_roundtrips() {

    assert_roundtrip(&[42]);

}



#[test]

fn repeated_value_roundtrips() {

    assert_roundtrip(&vec![7u8; 10_000]);

}



#[test]

fn all_symbols_roundtrips() {

    let src: Vec<u8> = (0..=255u16).map(|b| b as u8).collect();

    assert_roundtrip(&src);

    assert_roundtrip(&src.repeat(64));

}



#[test]

fn pseudo_random_roundtrips() {

    // 线性同余生成确定性的"高熵"字节，覆盖近似均匀分布。

    let mut x = 0x1234_5678u32;

    let src: Vec<u8> = (0..50_000)

        .map(|_| {

            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);

            (x >> 16) as u8

        })

        .collect();

    assert_roundtrip(&src);

}



#[test]

fn skewed_distribution_roundtrips() {

    // 90% 为 0，其余少量符号，压缩后应明显变小。

    let mut src = vec![0u8; 20_000];

    for (i, slot) in src.iter_mut().enumerate() {

        if i % 10 == 0 {

            *slot = (i % 7) as u8 + 1;

        }

    }

    let frame = encode(&src);

    assert!(frame.len() < src.len(), "高偏斜数据应被压缩");

    assert_roundtrip(&src);

}



#[test]

fn truncated_frame_is_rejected() {

    let frame = encode(&vec![3u8; 1000]);

    for cut in [0, 1, 4, 33, frame.len() - 1] {

        assert!(

            decode(&frame[..cut], 1000).is_err(),

            "截断到 {cut} 必须报错"

        );

    }

}



#[test]

fn garbage_is_rejected() {

    assert!(decode(&[0xFF; 64], 64).is_err());

}



#[test]

fn a_trailing_extra_byte_is_rejected() {

    let mut frame = encode(&vec![3u8; 1000]);

    frame.push(0x77);

    let error = decode(&frame, 1000).expect_err("帧尾多余字节必须报错");

    assert!(error.to_string().contains("多余字节"), "{error}");

}



#[test]

fn a_lying_length_is_rejected_before_allocating() {

    // 帧头的原始长度来自文件，不能当分配依据：把它改成 u32::MAX 之后必须当场拒绝，

    // 而不是先按它分配再想办法。

    let mut frame = encode(&[7u8; 64]);

    frame[..4].copy_from_slice(&u32::MAX.to_le_bytes());

    let error = decode(&frame, 64).expect_err("帧自报长度不能当分配依据");

    assert!(error.to_string().contains("超过上限"), "{error}");

}



#[test]

fn normalized_freq_sums_exactly() {

    let raw = [

        counts(&[1u8, 1, 2, 3, 3, 3]),

        counts(&(0..255u16).map(|b| b as u8).collect::<Vec<u8>>()),

    ];

    for r in raw {

        let freq = normalize(&r);

        assert_eq!(freq.iter().sum::<u32>(), TOTAL_FREQ);

    }

}



/// 香农熵（bits/符号），无损压缩的物理下界。

fn entropy(src: &[u8]) -> f64 {

    let n = src.len() as f64;

    let mut h = 0.0f64;

    for &c in &counts(src) {

        if c > 0 {

            let p = c as f64 / n;

            h -= p * p.log2();

        }

    }

    h

}



/// 从帧头解析出纯 rans payload 长度（8 条 lane 流之和）。

fn payload_len(frame: &[u8]) -> usize {

    let mut cur = frame;

    take::<4>(&mut cur).expect("src_len");

    let bitmap = take::<32>(&mut cur).expect("bitmap");

    let present = bitmap.iter().map(|b| b.count_ones()).sum::<u32>() as usize;

    for _ in 0..present {

        take::<2>(&mut cur).expect("freq");

    }

    for _ in 0..NLANES {

        take::<4>(&mut cur).expect("state");

    }

    let mut total = 0usize;

    for _ in 0..NLANES {

        total += u32::from_le_bytes(take::<4>(&mut cur).expect("lane len")) as usize;

    }

    total

}



#[test]

fn payload_approaches_entropy_bound() {

    // 确定性的偏斜分布：80% 为 0，其余在 4 个符号间，约 1 bit/符号。

    let n = 200_000usize;

    let src: Vec<u8> = (0..n)

        .map(|i| if i % 5 == 0 { (i % 4) as u8 + 1 } else { 0 })

        .collect();



    let h = entropy(&src);

    let ideal = (n as f64) * h / 8.0;

    let frame = encode(&src);

    let payload = payload_len(&frame) as f64;



    // rans 核心效率不低于 99%：payload 与熵下界之差在 1% 内。

    let efficiency = ideal / payload;

    assert!(

        efficiency >= 0.99,

        "rANS 效率 {efficiency:.4} 低于 0.99（payload={payload}, 熵下界={ideal}）"

    );



    // 帧总开销（头部）是有界常数，不随 n 增长：头部 ≤ 200 字节。

    let overhead = frame.len() as f64 - payload;

    assert!(overhead <= 200.0, "头部开销 {overhead} 应为有界常数");

}



#[test]

fn uniform_data_is_not_amplified() {

    // 近似均匀（熵≈8 bit）的数据：rans 帧不会比原始短，写端因此回退 store。

    let mut x = 0xABCDEF01u32;

    let src: Vec<u8> = (0..10_000)

        .map(|_| {

            x = x.wrapping_mul(1_103_515_245).wrapping_add(1);

            (x >> 8) as u8

        })

        .collect();

    let frame = encode(&src);

    // 编码帧本身允许含头部，但写端策略保证只有严格变小才选 rans。

    assert!(frame.len() >= src.len(), "均匀数据不应被 rans 压缩");

}
