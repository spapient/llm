use super::*;

fn block() -> AttentionBlock {
    AttentionBlock::new(
        "blk.0.attn",
        32,
        16,
        8,
        4,
        AttentionAlgo::Naive,
        WeightFormat::F32,
    )
    .expect("合法几何")
}

#[test]
fn it_groups_qkv_attention_and_output_under_one_prefix() {
    let layer = block();
    assert_eq!(layer.name(), "blk.0.attn");
    assert_eq!(layer.output().dims(), [32]);
    assert_eq!(layer.spca().ops(), [Linear::DENSE, Attention::NAIVE]);

    assert_eq!(layer.q().name(), "blk.0.attn.q");
    assert_eq!(layer.k().name(), "blk.0.attn.k");
    assert_eq!(layer.v().name(), "blk.0.attn.v");
    assert_eq!(layer.out().name(), "blk.0.attn.o");
    assert_eq!(layer.attention().name(), "blk.0.attn.attn");

    assert_eq!(layer.hidden(), 32);
    assert_eq!(layer.kv_width(), 16);
    assert_eq!(layer.head_dim(), 8);
    assert_eq!(layer.heads(), 2);
    assert_eq!(layer.seq(), 4);
    assert_eq!(layer.algo(), AttentionAlgo::Naive);
}

#[test]
fn projection_widths_go_hidden_to_kv_then_back() {
    let layer = block();
    assert_eq!(layer.q().in_features(), 32);
    assert_eq!(layer.q().out_features(), 16);
    assert_eq!(layer.k().out_features(), 16);
    assert_eq!(layer.v().out_features(), 16);
    assert_eq!(layer.out().in_features(), 16);
    assert_eq!(layer.out().out_features(), 32);
}

#[test]
fn weight_entry_names_carry_the_block_prefix() {
    let layer = block();
    assert_eq!(
        AttentionBlock::weight_entry("q", layer.name()),
        "blk.0.attn.q.weight"
    );
    assert_eq!(
        AttentionBlock::weight_entry("o", layer.name()),
        "blk.0.attn.o.weight"
    );
}

#[test]
fn its_floor_is_the_sum_of_each_child_floor() {
    let layer = block();
    let floor = layer.bounds().expect("合法").expect("有算子");

    let mut expected = Bounds::ZERO;
    for projection in [layer.q(), layer.k(), layer.v(), layer.out()] {
        expected = expected.plus(projection.bounds().expect("合法").expect("有算子"));
    }
    expected = expected.plus(layer.attention().bounds().expect("合法").expect("有算子"));

    assert_eq!(floor, expected);
    assert!(floor.flops() > 0.0 && floor.read() > 0 && floor.write() > 0);
}

#[test]
fn it_is_reached_through_the_contract_view() {
    let dynamic: &dyn Contract = &block();
    let Some(view) = dynamic.attention_block() else {
        panic!("组合块应经 Contract 视图取回");
    };
    assert_eq!(view.hidden(), 32);
    assert!(dynamic.linear().is_none());
}

#[test]
fn a_zero_dimension_block_is_rejected() {
    assert_eq!(
        AttentionBlock::new("x", 0, 16, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(
        AttentionBlock::new("x", 32, 0, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn kv_width_must_be_divisible_by_head_dim() {
    let error = AttentionBlock::new("x", 32, 17, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
        .unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("整数倍"), "{error}");
}
