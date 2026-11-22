use super::*;

#[test]
fn same_key_stream_idx_produces_the_same_block() {
    let first = block(0x1234_5678_9ABC_DEF0, 7, 100);
    let again = block(0x1234_5678_9ABC_DEF0, 7, 100);
    assert_eq!(first, again, "计数器派生必须是确定性的");
}

#[test]
fn different_streams_do_not_share_a_sequence() {
    let seed = 42u64;
    let a = block(seed, 0, 0);
    let b = block(seed, 1, 0);
    assert_ne!(a, b, "独立流号必须给出不同序列");
}

#[test]
fn rng_is_deterministic_and_bounded() {
    let mut left = Rng::new(1234);
    let mut right = Rng::new(1234);
    for _ in 0..17 {
        assert_eq!(left.next_u32(), right.next_u32(), "同种子序列一致");
    }
}

#[test]
fn next_f32_stays_in_the_half_open_unit_interval() {
    let mut rng = Rng::new(7);
    let mut seen_nonzero = false;
    for _ in 0..64 {
        let value = rng.next_f32();
        assert!(
            (0.0..1.0).contains(&value),
            "均匀量必须落在 [0,1)，得到 {value}"
        );
        seen_nonzero |= value > 0.0;
    }
    assert!(seen_nonzero, "高 24 位映射不能塌缩成全零");
}

/// 已知答案向量：与独立参考实现逐词核对的 Philox4x32-10 输出。
///
/// 覆盖零种子、非零种子、跨 32 位边界的种子、同一流内连续三个计数器；任何轮函数、
/// 乘法高低位或轮密钥递推写偏都会在这里被抓住。
#[test]
fn known_answer_vectors_match_the_reference_philox() {
    let cases = [
        (
            0u64,
            0u32,
            0u64,
            [0x01BF_6E88, 0x0113_DC1E, 0x0FF2_CC34, 0x6B72_E23A],
        ),
        (
            123_456,
            7,
            100,
            [0xB99B_391E, 0x068B_CDA9, 0x83BC_9EFA, 0x3292_3104],
        ),
        (
            0x1234_5678_9ABC_DEF0,
            7,
            100,
            [0xD45C_6E74, 0x6860_CD9F, 0xD917_D187, 0x9734_C29E],
        ),
        (
            9,
            3,
            0,
            [0x8AA2_DF03, 0x713D_E75A, 0x704C_584E, 0x581A_FDAC],
        ),
        (
            9,
            3,
            1,
            [0xC2E8_B289, 0x77BF_5997, 0x45C2_B652, 0x9850_3DDF],
        ),
        (
            9,
            3,
            2,
            [0xFDBC_B116, 0x27FB_AB4E, 0xE85F_C832, 0xBAC6_B546],
        ),
    ];
    for (key, stream, idx, expected) in cases {
        assert_eq!(
            block(key, stream, idx),
            expected,
            "key {key:#x} stream {stream} index {idx}"
        );
    }
}

#[test]
fn sequential_blocks_match_direct_counter_derivation() {
    let mut rng = Rng::with_stream(9, 3);
    for idx in 0..3u64 {
        let direct = block(9, 3, idx);
        let buffered = std::array::from_fn(|_| rng.next_u32());
        assert_eq!(buffered, direct, "顺序缓冲必须与直接派生逐词一致");
    }
}
