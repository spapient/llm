use super::*;

/// 单步查表：装配出的图取索引对应整行，参考路径同一 `embed_f32` 串行重算。
#[test]
fn an_embedding_single_step_matches_the_reference() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let idx = [2.0f32];

    let mut reference = vec![0.0; dim];
    nn::embed_f32::forward(
        tensor::View::one(&idx),
        tensor::View::one(&table),
        &mut reference,
    )
    .expect("参考：查表");

    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", vocab, dim, 1).expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[("tok_emb.weight", table)]);

    forward_close(stack, params, &idx, &reference);
}

/// 查表后接投影：嵌入行向量作为投影输入，参考路径串行 `embed → gemv` 重算。
#[test]
fn an_embedding_then_projection_matches_the_reference() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let idx = [2.0f32];
    // [2, 4] 行主序权重
    let w = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let embedded = {
        let mut e = vec![0.0; dim];
        nn::embed_f32::forward(tensor::View::one(&idx), tensor::View::one(&table), &mut e)
            .expect("参考：查表");
        e
    };
    let mut reference = vec![0.0; 2];
    nn::gemv_f32::forward(tensor::View::one(&embedded), &w, &mut reference).expect("参考：投影");

    let stack = embed_projection_stack(vocab, dim);
    let params = params_of(&[("tok_emb.weight", table), ("proj.weight", w.to_vec())]);

    forward_close(stack, params, &idx, &reference);
}

/// 查表缺驻留词表：装配期按名报 `Missing`。
#[test]
fn a_missing_embedding_table_is_reported_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", 8, 4, 1).expect("层"),
    )])
    .expect("栈");

    rejected_at_assembly(stack, Params::new(), Code::Missing);
}

/// 词表元素数与 `vocab × dim` 不符：装配期报 `Invalid`。
#[test]
fn an_embedding_table_length_mismatch_is_rejected_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", 8, 4, 1).expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[("tok_emb.weight", vec![0.0; 8 * 4 - 1])]);

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 多 token 查表：`tokens` 个索引各查一行，输出 `[tokens, dim]` 展平，与串行 gather 参考对拍。
#[test]
fn a_multi_token_embedding_gathers_multiple_rows() {
    let (vocab, dim, tokens) = (8, 4, 3);
    let table = embed_table(vocab, dim);
    let idx = [2.0f32, 5.0, 1.0];

    let mut reference = vec![0.0; tokens * dim];
    nn::embed_f32::forward(
        tensor::View::one(&idx),
        tensor::View::one(&table),
        &mut reference,
    )
    .expect("参考：多 token 查表");

    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", vocab, dim, tokens).expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[("tok_emb.weight", table)]);

    forward_close(stack, params, &idx, &reference);
}

/// Predict 一步前向：序列末位 token 折成索引，查表→投影主链对拍张量级 forward。
#[test]
fn predict_logits_from_an_embedding_sequence_match_the_reference() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let w = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let sequence = Sequence::new(0, &[2u32], 2, 16, Stream::new(0)).expect("序列");

    let stack = embed_projection_stack(vocab, dim);
    let params = params_of(&[("tok_emb.weight", table), ("proj.weight", w.to_vec())]);

    let mut model = assemble(stack, params);
    let mut batch = plain_batch();
    let produced = model.logits(&sequence, &mut batch).expect("一步前向");
    let reference = model.forward(&[2.0]).expect("张量级参考");

    assert_close(&produced, &reference);
}

/// 无块查表→投影链整批：两条序列一次前向，各行与串行 embed→gemv 参考对拍，行序对齐准入序。
#[test]
fn logits_batch_runs_two_sequences_through_the_flat_projection_chain() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let w = projection_matrix(2, dim, 1.0);

    let stack = embed_projection_stack(vocab, dim);
    let params = params_of(&[
        ("tok_emb.weight", table.clone()),
        ("proj.weight", w.clone()),
    ]);
    let mut model = assemble(stack, params);
    let mut batch = plain_batch();
    let a = Sequence::new(0, &[2u32], 2, 16, Stream::new(0)).expect("序列 A");
    let b = Sequence::new(1, &[5u32], 2, 16, Stream::new(0)).expect("序列 B");

    let rows = model.logits_batch(&[a, b], &mut batch).expect("整批前向");
    assert_eq!(rows.len(), 2, "两行 logits 与输入切片一一对应");

    assert_close(&rows[0], &reference_row(&table, &w, 2, dim));
    assert_close(&rows[1], &reference_row(&table, &w, 5, dim));
}

/// 条件记忆层不能作首层：要加回的是主链已有的隐向量，没有主链就没有可加的东西。
#[test]
fn a_memory_layer_cannot_open_the_stack() {
    let stack = Stack::of(vec![Box::new(
        MemoryBlock::new("mem", 2, 2, 7, 11, 1.0).expect("层"),
    )])
    .expect("栈");

    rejected_at_assembly(stack, Params::new(), Code::Invalid);
}

/// 记忆行宽与主链宽度不符：装配期报 `Invalid`，不靠运行期逐元素对不上才发现。
#[test]
fn a_memory_width_mismatch_is_rejected_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", 8, 2, 1).expect("层")),
        Box::new(MemoryBlock::new("mem", 4, 2, 7, 11, 1.0).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[
        ("tok_emb.weight", embed_table(8, 2)),
        ("mem.weight", vec![0.0; 11 * 4]),
    ]);

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 条件记忆旁路：历史达到阶才按槽加回门控行，历史不足时门控压成零、主链逐位不变。
#[test]
fn memory_bypass_adds_the_gated_row_only_once_history_reaches_the_order() {
    let (vocab, dim, order, seed, buckets, gate) = (8usize, 2usize, 2usize, 7u64, 11usize, 1.0f32);
    let table = embed_table(vocab, dim);
    let slot = nn::slot_of(&[3, 5], order, seed, buckets).expect("历史够阶");
    let mut memory_table = vec![0.0f32; buckets * dim];
    memory_table[slot * dim] = 1.5;
    memory_table[slot * dim + 1] = -2.0;

    let params = params_of(&[
        ("tok_emb.weight", table.clone()),
        ("mem.weight", memory_table.clone()),
    ]);
    let mut model = assemble(memory_stack(vocab, dim, order, seed, buckets, gate), params);

    let long = Sequence::new(0, &[3u32, 5], 2, 16, Stream::new(0)).expect("两 token 序列");
    let produced = model
        .logits(&long, &mut plain_batch())
        .expect("历史够阶：记忆加回");
    assert_close(
        &produced,
        &memory_reference(
            5,
            &[3, 5],
            &table,
            &memory_table,
            order,
            seed,
            buckets,
            gate,
            dim,
        ),
    );

    let short = Sequence::new(1, &[5u32], 2, 16, Stream::new(0)).expect("单 token 序列");
    let produced = model
        .logits(&short, &mut plain_batch())
        .expect("历史不足：不加记忆");
    assert_close(
        &produced,
        &memory_reference(
            5,
            &[5],
            &table,
            &memory_table,
            order,
            seed,
            buckets,
            gate,
            dim,
        ),
    );
}

/// Token-only Predict 无法提供非 Embedding 首层所需的真实隐向量，必须明确拒绝。
#[test]
fn predict_rejects_a_projection_first_layer_without_vector_input() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", projection_matrix(2, 3, 1.0))]);

    let mut model = assemble(stack, params);
    let sequence = Sequence::new(0, &[2u32], 2, 16, Stream::new(0)).expect("序列");
    let mut batch = plain_batch();
    let error = model
        .logits(&sequence, &mut batch)
        .expect_err("缺少向量输入必须拒绝");
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("真实向量输入"));
}
