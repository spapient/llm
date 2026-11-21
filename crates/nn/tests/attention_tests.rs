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
