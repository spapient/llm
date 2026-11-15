use super::*;

#[test]
fn an_embedding_declares_its_table_and_output_shape() {
    let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
    assert_eq!(layer.name(), "tok_emb");
    assert_eq!(layer.vocab(), 32_000);
    assert_eq!(layer.dim(), 128);
    assert_eq!(layer.tokens(), 8);
    assert_eq!(layer.output().dims(), [8, 128]);
    assert_eq!(layer.spca().ops(), [Embedding::OP]);
}

#[test]
fn an_embedding_floor_follows_the_gather_semantics() {
    let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
    let floor = layer.bounds().expect("合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_context(Embedding::OP, 8, 1).expect("同一份语义")
    );
}

#[test]
fn a_zero_dimension_embedding_is_rejected() {
    assert_eq!(
        Embedding::new("x", 0, 128, 8).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Embedding::new("x", 32, 0, 8).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Embedding::new("x", 32, 128, 0).unwrap_err().code(),
        Code::Invalid
    );
}
