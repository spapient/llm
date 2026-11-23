use super::*;

/// 两层 F32 线性链：装配出的图串行跑通，参考路径用同一批权重串行 gemv 重算。
#[test]
fn a_two_layer_chain_forward_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let (stack, params, w1, w2) = two_layer_f32_fixture();

    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    nn::gemv_f32::forward(tensor::View::one(&hidden), &w2, &mut reference).expect("参考：第二层");

    forward_close(stack, params, &x, &reference);
}

/// 装饰器叠在执行基座上：同一个模型换 `Trace<CpuExecutor>`，前向逐位不变，提交序可审计。
#[test]
fn a_decorated_executor_runs_the_same_model_and_records_submissions() {
    let x = [1.0, 2.0, 3.0];
    let (stack, params, ..) = two_layer_f32_fixture();

    let mut plain = CpuModel::new(&stack, &params).expect("装配");
    let expected = plain.forward(&x).expect("前向");
    let mut decorated = CpuModel::with_executor(&stack, &params, Trace::new(cpu())).expect("装配");
    let produced = decorated.forward(&x).expect("前向");

    assert_eq!(produced, expected, "装饰器不改执行语义");
    assert_eq!(
        decorated.executor().submitted().len(),
        1,
        "一次前向一次提交"
    );
    assert!(
        decorated.executor().submitted()[0]
            .iter()
            .any(|name| name == "gemv_f32"),
        "记录的是真实提交的节点名：{:?}",
        decorated.executor().submitted()
    );
}

/// 描述表把编号映射回条目名与声明形状；激活缓冲没有条目名，形状只记单轴长度（D10 / ADR 0044）。
#[test]
fn descriptors_map_ids_back_to_entries_and_shapes() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 2, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[
        ("blk.0.weight", vec![0.0; 6]),
        ("blk.1.weight", vec![0.0; 4]),
    ]);
    let model = assemble(stack, params);

    let descriptors = model.descriptors();
    let weight = |entry: &str| {
        descriptors
            .iter()
            .find(|desc| desc.entry.as_deref() == Some(entry))
            .expect("权重描述")
    };
    assert_eq!(weight("blk.0.weight").shape, [2, 3]);
    assert_eq!(weight("blk.1.weight").shape, [2, 2]);

    let activations: Vec<_> = descriptors
        .iter()
        .filter(|desc| desc.entry.is_none())
        .collect();
    assert_eq!(activations.len(), 2, "两层各一条激活输出");
    assert!(activations.iter().all(|desc| desc.shape.len() == 1));
}

/// 权重元素数与形状不符：装配期就拦住，不带进执行期。
#[test]
fn a_weight_length_mismatch_is_rejected_at_assembly() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", vec![0.0; 5])]); // 需要 3×2 = 6

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 参数表缺条目：装配期按名报 `Missing`。
#[test]
fn a_missing_weight_is_rejected_at_assembly() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");

    rejected_at_assembly(stack, Params::new(), Code::Missing);
}

/// 量化（默认 int4）层配了稠密 f32 权重：装配期按 `Invalid` 拒绝，不把 f32 当打包字节解释。
#[test]
fn a_quantized_layer_with_dense_weights_is_rejected_at_assembly() {
    let stack = Stack::new(vec![Linear::new("blk.0", 32, 2).expect("层")]).expect("栈");
    let params = params_of(&[("blk.0.weight", vec![0.0; 64])]);

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 量化（默认 int4）层缺标度条目：装配期按 `Missing` 拒绝，不静默走单输入节点。
#[test]
fn a_quantized_layer_without_its_scale_is_rejected_at_assembly() {
    let stack = Stack::new(vec![Linear::new("blk.0", 32, 1).expect("层")]).expect("栈");
    let mut params = Params::new();
    params.insert_packed("blk.0.weight", pack_codes(4, &[1i8; 32]));

    rejected_at_assembly(stack, params, Code::Missing);
}

/// 量化 int4 投影装配端到端：打包字节 + 标度两条驻留条目，前向与 `gemv_i4` 参考对拍。
#[test]
fn a_quantized_int4_projection_assembles_and_matches_the_reference() {
    let x: Vec<f32> = (0..32).map(|idx| 1.0 + idx as f32 * 0.1).collect();
    let codes: Vec<i8> = (0..32).map(|idx| (idx as i8 % 5) - 2).collect();
    let w = pack_codes(4, &codes);
    let scales = [0.5f32];
    let mut reference = vec![0.0; 1];
    nn::gemv_i4::forward(
        tensor::View::one(&x),
        &w,
        tensor::View::one(&scales),
        &mut reference,
    )
    .expect("参考");

    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::INT4, 32, 1).expect("层"),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert_quantized("blk.0.weight", w, "blk.0.weight_scale", scales.to_vec());

    forward_close(stack, params, &x, &reference);
}

/// 量化 int8 投影装配端到端：逐元素字节 + 标度，前向与 `gemv_i8` 参考对拍。
#[test]
fn a_quantized_int8_projection_assembles_and_matches_the_reference() {
    let x: Vec<f32> = (0..32).map(|idx| 1.0 + idx as f32 * 0.1).collect();
    let w: Vec<u8> = (0..32).map(|idx| ((idx as i8 % 5) - 2) as u8).collect();
    let scales = [0.5f32];
    let mut reference = vec![0.0; 1];
    nn::gemv_i8::forward(
        tensor::View::one(&x),
        &w,
        tensor::View::one(&scales),
        &mut reference,
    )
    .expect("参考");

    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::INT8, 32, 1).expect("层"),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert_quantized("blk.0.weight", w, "blk.0.weight_scale", scales.to_vec());

    forward_close(stack, params, &x, &reference);
}

/// 量化 int4 归档经 `from_stack` 全链装载：打包字节 + 标度两条条目按头部自动成对，
/// 装配后前向与 `gemv_i4` 参考对拍——覆盖 `insert_named` 的成对登记与标度回链。
#[test]
fn a_quantized_int4_archive_loads_and_matches_the_reference() {
    let in_features = 32usize;
    let x: Vec<f32> = (0..in_features).map(|idx| 1.0 + idx as f32 * 0.1).collect();
    let codes: Vec<i8> = (0..in_features).map(|idx| (idx as i8 % 5) - 2).collect();
    let w = pack_codes(4, &codes);
    let scales = [0.25f32];
    let mut reference = vec![0.0; 1];
    nn::gemv_i4::forward(
        tensor::View::one(&x),
        &w,
        tensor::View::one(&scales),
        &mut reference,
    )
    .expect("参考");

    let bytes = archive_of(&[
        ArchiveEntry {
            name: "blk.0.weight",
            shape: [1, 16],
            dtype: "U8",
            bytes: w.clone(),
        },
        ArchiveEntry {
            name: "blk.0.weight_scale",
            shape: [1, 1],
            dtype: "F32",
            bytes: f32_block(&scales),
        },
    ]);
    let archive = Archive::parse(&bytes).expect("归档合法");
    let stack = Stack::from_header(archive.header()).expect("层栈");
    let params = Params::from_stack(&stack, archive.header(), &archive).expect("装载");

    forward_close(stack, params, &x, &reference);
}

/// 层间宽度链断裂：上一层输出宽度对不上本层输入宽度，装配期报 `Invalid`。
#[test]
fn a_width_chain_break_is_rejected_at_assembly() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[
        ("blk.0.weight", vec![0.0; 6]),
        ("blk.1.weight", vec![0.0; 6]),
    ]);

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 内存归档装载：头部推出层栈、Loader 取出条目字节、解码进驻留表，装配后 forward 与串行参考对拍。
#[test]
fn entries_from_an_in_memory_archive_load_into_params() {
    let (x, bytes, reference) = two_layer_archive_fixture();
    let archive = Archive::parse(&bytes).expect("归档合法");
    let stack = Stack::from_header(archive.header()).expect("层栈");
    let params = Params::from_stack(&stack, archive.header(), &archive).expect("装载");
    let mut model = assemble(stack, params);
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 归档同时实现 [`Reader`]：借用路径不经字节拷贝装载，装配后前向与串行参考对拍，
/// 与 Loader 拷贝路径同判据、同产物。
#[test]
fn entries_from_a_reader_source_load_into_params() {
    let (x, bytes, reference) = two_layer_archive_fixture();
    let archive = Archive::parse(&bytes).expect("归档合法");
    let stack = Stack::from_header(archive.header()).expect("层栈");
    let params = Params::from_reader_stack(&stack, archive.header(), &archive).expect("借用装载");
    let mut model = assemble(stack, params);
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 真实权重文件走装配根 [`CpuModel::from_path`]：打开头部 → 层栈 → 驻留参数 → 图装配，
/// 前向与串行参考对拍；当前构建按后端类别走映射借用（Unix + mmap）或文件拷贝，产物同判据。
#[test]
fn a_real_weight_file_assembles_through_from_path() {
    let (x, bytes, reference) = two_layer_archive_fixture();
    let path = write_archive_file("from-path", &bytes);
    let mut model = CpuModel::from_path(&path).expect("文件装配");
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 头部找不到对应条目：装载期按名报 `Missing`，不带进装配期。
#[test]
fn a_missing_header_entry_is_reported_at_load() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let bytes = empty_archive();
    let archive = Archive::parse(&bytes).expect("空归档");

    let error = Params::from_stack(&stack, archive.header(), &archive).expect_err("缺条目必须拒绝");
    assert_eq!(error.code(), Code::Missing);
}

/// 投影后接一元激活：装配图串行跑通，参考路径串行 `gemv → silu` 重算。
#[test]
fn a_projection_then_unary_activation_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    nn::silu_f32::forward(tensor::View::one(&hidden), &mut reference).expect("参考：激活");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Silu::new("act", 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", w1.to_vec())]);

    forward_close(stack, params, &x, &reference);
}

/// 投影后接归一化：gamma 作为驻留权重整块装配，参考路径串行 `gemv → rmsnorm` 重算。
#[test]
fn a_projection_then_rmsnorm_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let gamma = [1.0, 1.0];

    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    nn::rmsnorm_f32::forward(
        tensor::View::one(&hidden),
        tensor::View::one(&gamma),
        &mut reference,
    )
    .expect("参考：归一化");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(RmsNorm::new("norm.0", 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[
        ("blk.0.weight", w1.to_vec()),
        ("norm.0.weight", gamma.to_vec()),
    ]);

    forward_close(stack, params, &x, &reference);
}

/// 一元层宽度与上一层输出不等长：装配期报 `Invalid`。
#[test]
fn a_unary_width_chain_break_is_rejected_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Silu::new("act", 3).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", vec![0.0; 6])]);

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 归一化缺驻留 gamma：装配期按名报 `Missing`。
#[test]
fn a_missing_rmsnorm_gamma_is_reported_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(RmsNorm::new("norm.0", 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", vec![0.0; 6])]);

    rejected_at_assembly(stack, params, Code::Missing);
}

/// 逐元素二元层：主链输入 + 旁路激活两路接线，参考路径对旁路持有张量逐元素相加。
#[test]
fn a_binary_layer_takes_a_bypass_activation_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    nn::add_f32::forward(
        tensor::View::one(&hidden),
        tensor::View::one(&hidden),
        &mut reference,
    )
    .expect("参考：二元相加");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Add::new("res", 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", w1.to_vec())]);

    forward_close(stack, params, &x, &reference);
}

/// 逐元素乘法同族接线：参考路径对旁路持有张量逐元素相乘。
#[test]
fn a_binary_mul_layer_takes_a_bypass_activation_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    nn::mul_f32::forward(
        tensor::View::one(&hidden),
        tensor::View::one(&hidden),
        &mut reference,
    )
    .expect("参考：二元相乘");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Mul::new("gate", 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", w1.to_vec())]);

    forward_close(stack, params, &x, &reference);
}

/// 二元层前面没有可借用的旁路激活（首层即二元）：装配期报 `Invalid`，不静默走单输入。
#[test]
fn a_binary_layer_without_a_bypass_activation_is_rejected_at_assembly() {
    let stack = Stack::of(vec![Box::new(Add::new("res", 2).expect("层"))]).expect("栈");
    rejected_at_assembly(stack, Params::new(), Code::Invalid);
}

/// 非 F32 条目不在装载期解包；参数表留空，装配期按格式报 `Unsupported`。
#[test]
fn a_non_f32_entry_is_skipped_and_rejected_at_assembly() {
    let bytes = archive_of(&[ArchiveEntry {
        name: "blk.0.weight",
        shape: [2, 3],
        dtype: "I8",
        bytes: vec![0u8; 6],
    }]);
    let archive = Archive::parse(&bytes).expect("归档合法");
    let stack = Stack::from_header(archive.header()).expect("层栈");
    let params = Params::from_stack(&stack, archive.header(), &archive).expect("装载");
    assert!(params.is_empty());

    rejected_at_assembly(stack, params, Code::Unsupported);
}

/// 两层合成模型端到端：嵌入 → 注意力块（q/k/v + RoPE + decode 注意力 + 残差）→ 归一化 →
/// 线性输出；CPU 参考数值链串行重算并与 `logits` 对拍。
#[test]
fn a_two_layer_composite_model_logits_match_the_reference() {
    let (vocab, hidden, kv, out) = (8, 8, 8, 2);
    let hist = 1;

    let table = embed_table(vocab, hidden);
    let (wq, wk, wv, wo) = block_projections(hidden, kv);
    let gamma = vec![1.0; hidden];
    let w2 = projection_matrix(out, hidden, 4.0);
    let (hist_k, hist_v) = history_kv(kv);
    let history = history_row(&hist_k, &hist_v);

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, hidden, 1).expect("层")),
        Box::new(
            AttentionBlock::new(
                "blk.0",
                hidden,
                kv,
                kv,
                2,
                AttentionAlgo::Naive,
                WeightFormat::F32,
            )
            .expect("层"),
        ),
        Box::new(RmsNorm::new("norm.0", hidden).expect("层")),
        Box::new(Linear::with_format("head", WeightFormat::F32, hidden, out).expect("层")),
    ])
    .expect("栈");
    let mut params = block_params(
        table.clone(),
        wq.clone(),
        wk.clone(),
        wv.clone(),
        wo.clone(),
    );
    params.insert("norm.0.weight", gamma.clone());
    params.insert("head.weight", w2.clone());

    let mut model = assemble(stack, params);
    let (mut batch, sequence) = decode_batch(kv, &history);
    let produced = model.logits(&sequence, &mut batch).expect("一步 decode");

    // 参考数值链：embed → q/k/v → RoPE(当前步位置) → decode 注意力 → o → 残差 → rmsnorm → 输出投影。
    let embedded = &table[2 * hidden..3 * hidden];
    let (residual, _, _) = reference_block(
        embedded,
        BlockWeights::Dense((&wq, &wk, &wv, &wo)),
        (&hist_k, &hist_v),
        hist,
        kv,
        kv,
        hidden,
        sequence.len(),
    );
    let mut normed = vec![0.0; hidden];
    nn::rmsnorm_f32::forward(
        tensor::View::one(&residual),
        tensor::View::one(&gamma),
        &mut normed,
    )
    .expect("参考归一化");
    let mut reference = vec![0.0; out];
    nn::gemv_f32::forward(tensor::View::one(&normed), &w2, &mut reference).expect("参考输出投影");

    assert_close(&produced, &reference);
}

/// 合成模型宽度断裂：嵌入维度与块隐藏宽度对不上，装配期报 `Invalid`，不带进执行期。
#[test]
fn a_composite_width_break_is_rejected_at_assembly() {
    let stack = embed_block_stack(8, 4, 8, 2);
    let mut params = block_zero_params();
    params.insert("tok_emb.weight", embed_table(8, 4));

    rejected_at_assembly(stack, params, Code::Invalid);
}

/// 合成模型缺输出投影条目：装配期按名报 `Missing`，不 panic、不静默跳过。
#[test]
fn a_composite_missing_projection_weight_is_reported_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", 8, 4, 1).expect("层")),
        Box::new(RmsNorm::new("norm.0", 4).expect("层")),
        Box::new(Linear::with_format("head", WeightFormat::F32, 4, 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[
        ("tok_emb.weight", embed_table(8, 4)),
        ("norm.0.weight", vec![1.0; 4]),
        // 故意不插 `head.weight`
    ]);

    rejected_at_assembly(stack, params, Code::Missing);
}

/// 合成模型缺嵌入词表条目：装配期按名报 `Missing`，不 panic、不静默跳过。
#[test]
fn a_composite_missing_embedding_table_is_reported_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", 8, 4, 1).expect("层")),
        Box::new(Linear::with_format("proj", WeightFormat::F32, 4, 2).expect("层")),
    ])
    .expect("栈");
    let params = params_of(&[("proj.weight", vec![0.0; 8])]);

    rejected_at_assembly(stack, params, Code::Missing);
}
