use super::*;

/// 两行批整块归一化：`gamma` 按行宽复用，两段独立 RMS、互不串段；参考逐行 `forward` 重算。
#[test]
fn rmsnorm_groups_batch_rows_against_the_single_row_reference() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma = [1.0, 1.0, 1.0];
    let mut reference = vec![0.0; 6];
    for row in 0..2 {
        let start = row * 3;
        op::rmsnorm_f32::forward(
            &x[start..start + 3],
            &gamma,
            &mut reference[start..start + 3],
        )
        .expect("参考：逐行归一化");
    }

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind(&mut executor, 1, &gamma);
    let produced = run_one(&mut executor, "rmsnorm_f32", &[0, 1], 2, 6);

    assert_close(&produced, &reference);
}

/// 两段批 RoPE：第二路位置表逐序列给位置，段内独立旋转、互不串段；参考逐段 `forward_heads` 重算。
#[test]
fn rope_rotates_batch_segments_by_the_position_table() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let positions = [2.0, 5.0];
    let head_dim = 2;
    let mut reference = vec![0.0; 4];
    for (seq, &pos) in positions.iter().enumerate() {
        let start = seq * head_dim;
        op::rope_f32::forward_heads(
            pos as usize,
            &x[start..start + head_dim],
            &mut reference[start..start + head_dim],
            head_dim,
        )
        .expect("参考：逐段旋转");
    }

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind(&mut executor, 1, &positions);
    executor.bind(2, vec![0.0; 4]);
    let mut node = Node::new(
        "rope_f32",
        Launch::new([1, 1, 1], [head_dim as u32, head_dim as u32, 1]),
    );
    node.bind_input(0).bind_input(1).bind_output(2);
    executor.run(&node).expect("执行");
    let produced = executor.get(2).expect("输出缓冲").to_vec();

    assert_close(&produced, &reference);
}

/// 两段批 decode 注意力：第四路段长表给各段历史行数，段间紧排、互不串段；参考逐段 `decode_forward_heads` 重算。
#[test]
fn decode_attention_groups_batch_segments_by_the_history_table() {
    let (heads, d, q_rows) = (1usize, 2usize, 1usize);
    let hist = [2usize, 2];
    let q = [0.1, 0.2, 0.3, 0.4];
    let k = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let v = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    let hist_table: Vec<f32> = hist.iter().map(|&h| h as f32).collect();

    let q_seq = heads * q_rows * d;
    let mut reference = vec![0.0; hist.len() * q_seq];
    let mut kv_cursor = 0;
    for (seq, &h) in hist.iter().enumerate() {
        let kv_seq = heads * h * d;
        op::attention::decode_forward_heads(
            &q[seq * q_seq..(seq + 1) * q_seq],
            &k[kv_cursor..kv_cursor + kv_seq],
            &v[kv_cursor..kv_cursor + kv_seq],
            &mut reference[seq * q_seq..(seq + 1) * q_seq],
            q_rows,
            h,
            d,
        )
        .expect("参考：逐段 decode");
        kv_cursor += kv_seq;
    }

    let mut executor = cpu();
    bind(&mut executor, 0, &q);
    bind(&mut executor, 1, &k);
    bind(&mut executor, 2, &v);
    bind(&mut executor, 3, &hist_table);
    executor.bind(4, vec![0.0; reference.len()]);
    let mut node = Node::new(
        "decode_attention_f32",
        Launch::new([1, 1, 1], [(heads * d) as u32, d as u32, 1]),
    );
    node.bind_input(0)
        .bind_input(1)
        .bind_input(2)
        .bind_input(3)
        .bind_output(4);
    executor.run(&node).expect("执行");
    let produced = executor.get(4).expect("输出缓冲").to_vec();

    assert_close(&produced, &reference);
}

/// 注意力块 decode 一步：embed → q/k/v → decode 注意力 → o → 残差旁路，与独立参考对拍；
/// 当前步 K/V 写回页行并按 `[k|v]` 切回对拍。
#[test]
fn an_attention_block_decodes_against_the_reference_and_roundtrips_kv() {
    let (vocab, hidden, kv) = (8, 8, 8);
    let hist = 1;

    let (mut model, table, (wq, wk, wv, wo)) = block_fixture(vocab, hidden, kv);
    let (hist_k, hist_v) = history_kv(kv);
    let history = history_row(&hist_k, &hist_v);

    let (mut batch, sequence) = decode_batch(kv, &history);
    let produced = model.logits(&sequence, &mut batch).expect("一步 decode");

    let embedded = &table[2 * hidden..3 * hidden];
    // q/k 先按当前步位置过 RoPE 旋转，再进注意力；写回页行的 K 同样取旋转后一路。
    let (reference, k_rot, v) = reference_block(
        embedded,
        BlockWeights::Dense((&wq, &wk, &wv, &wo)),
        (&hist_k, &hist_v),
        hist,
        kv,
        kv,
        hidden,
        sequence.len(),
    );

    assert_close(&produced, &reference);

    assert_kv_row(&batch, sequence.len(), kv, &k_rot, &v);
}

/// 两条序列同批走块链整批前向（`logits_batch`）：行序对齐准入序，各自对拍参考块，KV 写回各归各的页行。
#[test]
fn logits_batch_runs_two_sequences_in_input_order_and_keeps_their_own_kv() {
    let (mut model, mut batch, sequences, histories, table, projections) = two_seq_fixture(8, 8, 8);
    let history_refs: Vec<(&[f32], &[f32])> = histories
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    run_block_pair(
        &mut model,
        &mut batch,
        &sequences,
        8,
        &table,
        &projections,
        &history_refs,
        &[1, 1],
        8,
        8,
        &[2, 5],
    )
    .expect("批量前向");
}

/// 多头（`heads = kv / head_dim = 2`）两条序列同批走块链整批前向：头优先段内重排与写回按
/// 头宽分段，两行各自对拍参考块，KV 写回各归各的页行。
#[test]
fn logits_batch_runs_two_multi_head_sequences_with_their_own_kv() {
    let (mut model, mut batch, sequences, histories, table, projections) =
        two_seq_fixture(16, 8, 4);
    let history_refs: Vec<(&[f32], &[f32])> = histories
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    run_block_pair(
        &mut model,
        &mut batch,
        &sequences,
        16,
        &table,
        &projections,
        &history_refs,
        &[1, 1],
        8,
        8,
        &[2, 5],
    )
    .expect("多头批量前向");
}

/// 不等长历史（ragged）两条序列同批：A 缓存 1 行、B 缓存 2 行，段紧排游标按各自 `hist_b` 累计；
/// 两行 logits 与各自参考块对拍，KV 写回互不串段。
#[test]
fn logits_batch_runs_two_sequences_with_unequal_history_lengths() {
    let (vocab, hidden, kv) = (8, 8, 8);

    let (mut model, table, projections) = block_fixture(vocab, hidden, kv);

    // A：一行种子历史。
    let (hist_k_a, hist_v_a) = history_kv(kv);
    let history_a = history_row(&hist_k_a, &hist_v_a);
    // B：两行种子历史，两行取不同小值；段内按两行头优先排布。
    let (k0, v0) = history_kv(kv);
    let k1: Vec<f32> = k0.iter().map(|value| value * 1.7).collect();
    let v1: Vec<f32> = v0.iter().map(|value| value * 1.3).collect();
    let row_b0 = history_row(&k0, &v0);
    let row_b1 = history_row(&k1, &v1);
    let hist_k_b: Vec<f32> = k0.iter().chain(k1.iter()).copied().collect();
    let hist_v_b: Vec<f32> = v0.iter().chain(v1.iter()).copied().collect();

    let pager = Pager::new(Cache::new(4, 4).expect("页池"), 2 * kv).expect("行宽");
    let mut batch = Batch::new(pager);
    let a = Sequence::new(0, &[2u32], 2, 4, Stream::new(0)).expect("序列 A");
    let b = Sequence::new(1, &[5u32, 3u32], 2, 4, Stream::new(0)).expect("序列 B");
    batch.admit(a.clone()).expect("准入 A");
    batch.admit(b.clone()).expect("准入 B");
    batch.write_kv(0, 0, &history_a).expect("种子历史 A");
    batch.write_kv(1, 0, &row_b0).expect("种子历史 B 首行");
    batch.write_kv(1, 1, &row_b1).expect("种子历史 B 次行");

    let sequences = [a.clone(), b.clone()];
    let histories = [
        (hist_k_a.as_slice(), hist_v_a.as_slice()),
        (hist_k_b.as_slice(), hist_v_b.as_slice()),
    ];
    run_block_pair(
        &mut model,
        &mut batch,
        &sequences,
        kv,
        &table,
        &projections,
        &histories,
        &[1, 2],
        kv,
        hidden,
        &[2, 3],
    )
    .expect("ragged 批量前向");
}

/// 四条序列同批走块链整批前向：行序对齐准入序，四段历史紧排、四行输出拆回、KV 各归各页行。
#[test]
fn logits_batch_runs_four_sequences_in_input_order() {
    let (vocab, hidden, kv) = (8, 8, 8);
    let hist = 1;
    let tokens = [2u32, 5, 3, 6];

    let (mut model, table, projections) = block_fixture(vocab, hidden, kv);
    let pager = Pager::new(Cache::new(8, 4).expect("页池"), 2 * kv).expect("行宽");
    let mut batch = Batch::new(pager);
    let mut sequences = Vec::new();
    let mut histories = Vec::new();
    for (index, &token) in tokens.iter().enumerate() {
        let sequence = Sequence::new(index as u64, &[token], 2, 4, Stream::new(0)).expect("序列");
        let (hist_k, hist_v) = history_kv(kv);
        let history = history_row(&hist_k, &hist_v);
        batch.admit(sequence.clone()).expect("准入");
        batch.write_kv(index as u64, 0, &history).expect("种子历史");
        histories.push((hist_k, hist_v));
        sequences.push(sequence);
    }

    let token_ids: Vec<usize> = tokens.iter().map(|&token| token as usize).collect();
    let history_refs: Vec<(&[f32], &[f32])> = histories
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    run_block_pair(
        &mut model,
        &mut batch,
        &sequences,
        kv,
        &table,
        &projections,
        &history_refs,
        &[hist; 4],
        kv,
        hidden,
        &token_ids,
    )
    .expect("四条批量前向");
}

/// 块内一个投影条目缺失：装配期按名报 `Missing`。
#[test]
fn a_missing_block_projection_weight_is_reported_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        AttentionBlock::new("blk.0", 8, 8, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
            .expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[
        ("blk.0.k.weight", vec![0.0; 64]),
        ("blk.0.v.weight", vec![0.0; 64]),
        ("blk.0.o.weight", vec![0.0; 64]),
    ]);

    rejected_at_assembly(stack, params, Code::Missing);
}

/// 上一层输出宽度与块隐藏宽度不符：装配期报 `Invalid`。
#[test]
fn a_block_width_chain_break_is_rejected_at_assembly() {
    let stack = embed_block_stack(8, 4, 8, 4);
    let mut params = block_zero_params();
    params.insert("tok_emb.weight", embed_table(8, 4));

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 块内权重元素数与 `in × out` 不符：装配期报 `Invalid`。
#[test]
fn a_block_weight_length_mismatch_is_rejected_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        AttentionBlock::new("blk.0", 8, 8, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
            .expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[
        ("blk.0.q.weight", vec![0.0; 63]),
        ("blk.0.k.weight", vec![0.0; 64]),
        ("blk.0.v.weight", vec![0.0; 64]),
        ("blk.0.o.weight", vec![0.0; 64]),
    ]);

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 量化 int4 注意力块 decode：四路投影走打包字节 + 标度，q/k/v/o 输出仍是 f32（`gemv_i4` 输出
/// 浮点），KV 页行写回不交叠；与独立 `gemv_i4`/RoPE/注意力/残差串行参考对拍。
#[test]
fn a_quantized_attention_block_decodes_against_the_reference() {
    let (vocab, hidden, kv) = (8, 32, 32);
    let head_dim = kv;
    let hist = 1;

    let table = embed_table(vocab, hidden);
    let (wq, sq) = quant_projection(kv, hidden, 0);
    let (wk, sk) = quant_projection(kv, hidden, 1);
    let (wv, sv) = quant_projection(kv, hidden, 2);
    let (wo, so) = quant_projection(hidden, kv, 3);

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, hidden, 1).expect("层")),
        Box::new(
            AttentionBlock::new(
                "blk.0",
                hidden,
                kv,
                head_dim,
                4,
                AttentionAlgo::Naive,
                WeightFormat::INT4,
            )
            .expect("层"),
        ),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", table.clone());
    params.insert_quantized(
        "blk.0.q.weight",
        wq.clone(),
        "blk.0.q.weight_scale",
        sq.clone(),
    );
    params.insert_quantized(
        "blk.0.k.weight",
        wk.clone(),
        "blk.0.k.weight_scale",
        sk.clone(),
    );
    params.insert_quantized(
        "blk.0.v.weight",
        wv.clone(),
        "blk.0.v.weight_scale",
        sv.clone(),
    );
    params.insert_quantized(
        "blk.0.o.weight",
        wo.clone(),
        "blk.0.o.weight_scale",
        so.clone(),
    );

    let mut model = assemble(stack, params);
    let (hist_k, hist_v) = history_kv(kv);
    let history = history_row(&hist_k, &hist_v);
    let (mut batch, sequence) = decode_batch(kv, &history);
    let produced = model.logits(&sequence, &mut batch).expect("一步 decode");

    let embedded = &table[2 * hidden..3 * hidden];
    let (reference, k_rot, v) = reference_block(
        embedded,
        BlockWeights::Int4((&wq, &sq), (&wk, &sk), (&wv, &sv), (&wo, &so)),
        (&hist_k, &hist_v),
        hist,
        kv,
        head_dim,
        hidden,
        sequence.len(),
    );
    assert_close(&produced, &reference);

    assert_kv_row(&batch, sequence.len(), kv, &k_rot, &v);
}

/// 多头展平（`kv_width = heads · head_dim`）端到端：嵌入 → q/k/v + 分段 RoPE → 多头 decode 注意力 →
/// o 投影 → 残差；CPU 参考数值链按同一头优先块布局串行重算并对拍，K/V 页行回环一致。
#[test]
fn a_multi_head_block_decodes_against_the_reference() {
    let (vocab, hidden, kv, head_dim) = (8, 8, 16, 8);
    let hist = 1;

    let table = embed_table(vocab, hidden);
    let (wq, wk, wv, wo) = block_projections(hidden, kv);
    let (hist_k, hist_v) = history_kv(kv);
    let history = history_row(&hist_k, &hist_v);

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, hidden, 1).expect("层")),
        Box::new(
            AttentionBlock::new(
                "blk.0",
                hidden,
                kv,
                head_dim,
                4,
                AttentionAlgo::Naive,
                WeightFormat::F32,
            )
            .expect("层"),
        ),
    ])
    .expect("栈");
    let params = block_params(
        table.clone(),
        wq.clone(),
        wk.clone(),
        wv.clone(),
        wo.clone(),
    );

    let mut model = assemble(stack, params);
    let (mut batch, sequence) = decode_batch(kv, &history);
    let produced = model.logits(&sequence, &mut batch).expect("一步 decode");

    let embedded = &table[2 * hidden..3 * hidden];
    let (reference, k_rot, v) = reference_block(
        embedded,
        BlockWeights::Dense((&wq, &wk, &wv, &wo)),
        (&hist_k, &hist_v),
        hist,
        kv,
        head_dim,
        hidden,
        sequence.len(),
    );
    assert_close(&produced, &reference);

    assert_kv_row(&batch, sequence.len(), kv, &k_rot, &v);
}
