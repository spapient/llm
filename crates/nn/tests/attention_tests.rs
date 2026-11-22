use super::*;

#[test]
fn naive_and_flash_bind_their_own_operators() {
    let naive = Attention::new("blk.0.attn", 16, 64, AttentionAlgo::Naive).expect("合法");
    assert_eq!(naive.spca().ops(), [Attention::NAIVE]);
    let flash = Attention::new("blk.0.attn", 16, 64, AttentionAlgo::Flash).expect("合法");
    assert_eq!(flash.spca().ops(), [Attention::FLASH]);
    assert_eq!(naive.output().dims(), [16, 64]);
    assert_eq!(naive.seq(), 16);
    assert_eq!(naive.head_dim(), 64);
    assert_eq!(naive.algo(), AttentionAlgo::Naive);
}

#[test]
fn attention_floor_follows_the_shared_skeleton() {
    for algo in [AttentionAlgo::Naive, AttentionAlgo::Flash] {
        let layer = Attention::new("attn", 16, 64, algo).expect("合法");
        let floor = layer.bounds().expect("合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_op(algo.operator(), 16, 64).expect("同一份语义")
        );
        assert!(floor.flops() > 0.0 && floor.read() > 0 && floor.write() > 0);
    }
}

#[test]
fn a_zero_dimension_attention_is_rejected() {
    assert_eq!(
        Attention::new("x", 0, 64, AttentionAlgo::Naive)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(
        Attention::new("x", 16, 0, AttentionAlgo::Flash)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn latent_compresses_to_the_latent_width_and_spreads_rows_by_heads() {
    let layer = LatentAttention::new("blk.0.latent", 16, 32, 4).expect("合法");
    assert_eq!(layer.spca().ops(), [LatentAttention::OPERATOR]);
    assert_eq!(layer.output().dims(), [16, 32]);
    assert_eq!(layer.seq(), 16);
    assert_eq!(layer.latent(), 32);
    assert_eq!(layer.heads(), 4);
    let flat = LatentAttention::new("blk.0.latent", 64, 32, 1).expect("合法");
    assert_eq!(
        layer.bounds().expect("合法"),
        flat.bounds().expect("合法"),
        "代价只认摊平行数 seq · h：16×4 与 64×1 同价"
    );
    assert_ne!(
        layer.bounds().expect("合法"),
        LatentAttention::new("blk.0.latent", 16, 32, 1)
            .expect("合法")
            .bounds()
            .expect("合法"),
        "头数纯摊平行数，不进形状"
    );
}

#[test]
fn a_zero_dimension_latent_attention_is_rejected() {
    for (seq, latent, heads) in [(0, 32, 4), (16, 0, 4), (16, 32, 0)] {
        let error = LatentAttention::new("x", seq, latent, heads).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("潜空间注意力层"), "{error}");
    }
}

#[test]
fn decode_binds_its_own_operators_and_keeps_history_out_of_the_shape() {
    for (algo, op) in [
        (AttentionAlgo::Naive, DecodeAttention::NAIVE),
        (AttentionAlgo::Flash, DecodeAttention::FLASH),
    ] {
        let layer = DecodeAttention::new("blk.0.decode", 1, 64, algo).expect("合法");
        assert_eq!(layer.spca().ops(), [op]);
        assert_eq!(layer.output().dims(), [1, 64]);
        assert_eq!(layer.q_rows(), 1);
        assert_eq!(layer.head_dim(), 64);
        assert_eq!(layer.algo(), algo);
    }
}

#[test]
fn decode_floor_is_linear_in_query_rows_because_history_is_a_declared_tier() {
    let one = DecodeAttention::new("d", 1, 64, AttentionAlgo::Flash).expect("合法");
    let two = DecodeAttention::new("d", 2, 64, AttentionAlgo::Flash).expect("合法");
    let one = one.bounds().expect("合法").expect("有算子");
    let two = two.bounds().expect("合法").expect("有算子");
    assert_eq!(
        two.flops(),
        2.0 * one.flops(),
        "历史轴固定，代价只随 query 行数线性走"
    );
    assert!(one.read() > 0 && one.write() > 0);
    let prefill = Attention::new("p", 1, 64, AttentionAlgo::Flash)
        .expect("合法")
        .bounds()
        .expect("合法")
        .expect("有算子");
    assert_ne!(one, prefill, "解码骨架与 prefill 不同，层不许假装两者同价");
}

#[test]
fn a_zero_dimension_decode_is_rejected() {
    for (q_rows, head_dim) in [(0, 64), (1, 0)] {
        let error = DecodeAttention::new("x", q_rows, head_dim, AttentionAlgo::Naive).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("解码注意力层"), "{error}");
    }
}
