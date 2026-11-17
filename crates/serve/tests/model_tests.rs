use super::*;
use crate::Predict;
use crate::assembly::Params;
use crate::model_host::CpuModel;
use diagnostic::Code;
use engine::exec::{CpuExecutor, Device, Executor, Graph, Launch, Node, Stream};
use engine::kv::{Cache, Pager};
use engine::{Batch, Sequence};
use nn::{
    Add, AttentionAlgo, AttentionBlock, Embedding, Linear, Mul, RmsNorm, Silu, Stack, WeightFormat,
};
use testkit::{all_finite, close};
use weights::safetensors::Archive;

/// 登记完表内内核的空 CPU 执行器。
fn cpu() -> CpuExecutor {
    let mut executor = CpuExecutor::new(Device::new("cpu", "generic"));
    register_cpu(&mut executor).expect("登记 CPU 内核");
    executor
}

/// 一条空批（无历史 KV）：查表/投影主链的 Predict 调用不读页行。
fn plain_batch() -> Batch {
    Batch::new(Pager::new(Cache::new(2, 16).expect("页池"), 2).expect("行宽"))
}

/// 绑一块输入缓冲。
fn bind(executor: &mut CpuExecutor, id: u32, values: &[f32]) {
    executor.bind(id, values.to_vec());
}

/// 绑一块打包字节缓冲（量化 lane 权重 / 码点）。
fn bind_u8(executor: &mut CpuExecutor, id: u32, values: &[u8]) {
    executor.bind_u8(id, values.to_vec());
}

/// 跑一个单节点：输出整块零初始化，执行后取回。
fn run_one(
    executor: &mut CpuExecutor,
    op: &str,
    inputs: &[u32],
    output: u32,
    len: usize,
) -> Vec<f32> {
    executor.bind(output, vec![0.0; len]);
    let mut node = Node::new(op, Launch::new([1, 1, 1], [len as u32, 0, 1]));
    for &id in inputs {
        node.bind_input(id);
    }
    node.bind_output(output);
    executor.run(&node).expect("执行");
    executor.get(output).expect("输出缓冲").to_vec()
}

/// 对拍口径：先排非有限值，再走绝对 + 相对双项预算。同一份 `op` 语义两次调用，
/// 位级应一致，容差只兜浮点读回误差。
fn assert_close(produced: &[f32], reference: &[f32]) {
    assert!(all_finite(produced), "生产路径出现非有限值");
    assert!(all_finite(reference), "参考路径出现非有限值");
    assert!(
        close(produced, reference, 1e-6, 1e-6),
        "生产 {produced:?} 与参考 {reference:?} 不一致"
    );
}

/// 参数表：按名驻留整块 f32 条目；无条目时传空切片。
fn params_of(entries: &[(&str, Vec<f32>)]) -> Params {
    let mut params = Params::new();
    for (name, values) in entries {
        params.insert(*name, values.clone());
    }
    params
}

/// 装配一份栈与参数表；对拍路径上装配失败即测试失败。
fn assemble(stack: Stack, params: Params) -> CpuModel {
    CpuModel::new(&stack, &params).expect("装配")
}

/// 装配后跑张量级前向并与参考对拍；输出长度与参考向量对齐核对。
fn forward_close(stack: Stack, params: Params, input: &[f32], reference: &[f32]) {
    let mut model = assemble(stack, params);
    assert_eq!(model.output_len(), reference.len());
    let produced = model.forward(input).expect("前向");
    assert_close(&produced, reference);
}

/// 装配必须被既有错误路径拒绝，并核对诊断码；不 panic、不静默跳过。
fn rejected_at_assembly(stack: Stack, params: Params, expected: Code) {
    let error = CpuModel::new(&stack, &params).expect_err("装配必须拒绝");
    assert_eq!(error.code(), expected);
}

/// 种子历史页行：`[k | v]` 按行宽拼回一条历史行。
fn history_row(hist_k: &[f32], hist_v: &[f32]) -> Vec<f32> {
    let mut row = Vec::with_capacity(hist_k.len() + hist_v.len());
    row.extend_from_slice(hist_k);
    row.extend_from_slice(hist_v);
    row
}

/// 一条带一行种子历史的 decode 批：页池 4 页、行宽 `2·kv`，序列位 2。
fn decode_batch(kv: usize, history: &[f32]) -> (Batch, Sequence) {
    let pager = Pager::new(Cache::new(4, 4).expect("页池"), 2 * kv).expect("行宽");
    let mut batch = Batch::new(pager);
    let sequence = Sequence::new(0, &[2u32], 2, 4, Stream::new(0)).expect("序列");
    batch.admit(sequence.clone()).expect("准入");
    batch.write_kv(0, 0, history).expect("种子历史");
    (batch, sequence)
}

/// 两条各带一行种子历史的 decode 批：A 取 token 2、B 取 token 5，页行按序列号归属；
/// 批量前向回归基线夹具——当前走 `logits_batch` 默认逐条回退，整批几何接线后仍跑同一份对拍。
fn decode_batch_two(
    kv: usize,
    history_a: &[f32],
    history_b: &[f32],
) -> (Batch, Sequence, Sequence) {
    let pager = Pager::new(Cache::new(4, 4).expect("页池"), 2 * kv).expect("行宽");
    let mut batch = Batch::new(pager);
    let a = Sequence::new(0, &[2u32], 2, 4, Stream::new(0)).expect("序列 A");
    let b = Sequence::new(1, &[5u32], 2, 4, Stream::new(0)).expect("序列 B");
    batch.admit(a.clone()).expect("准入 A");
    batch.admit(b.clone()).expect("准入 B");
    batch.write_kv(0, 0, history_a).expect("种子历史 A");
    batch.write_kv(1, 0, history_b).expect("种子历史 B");
    (batch, a, b)
}

/// 参考链一层 gemv：`x · w` 写进新向量返回，输出宽度按权重行数推得。
fn gemv_hidden(x: &[f32], w: &[f32]) -> Vec<f32> {
    let out = w.len() / x.len();
    let mut h = vec![0.0; out];
    op::gemv_f32::forward(x, w, &mut h).expect("参考：投影");
    h
}

/// 块四路投影小值矩阵：q/k/v 取种子 0/1/2，o 取种子 3。
fn block_projections(hidden: usize, kv: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        projection_matrix(kv, hidden, 0.0),
        projection_matrix(kv, hidden, 1.0),
        projection_matrix(kv, hidden, 2.0),
        projection_matrix(hidden, kv, 3.0),
    )
}

/// 一行 decode 历史：k 路负偏移、v 路正偏移。
fn history_kv(kv: usize) -> (Vec<f32>, Vec<f32>) {
    let hist_k = (0..kv).map(|i| -0.3 + i as f32 * 0.05).collect();
    let hist_v = (0..kv).map(|i| 0.2 - i as f32 * 0.04).collect();
    (hist_k, hist_v)
}

/// 嵌入 + 注意力块层栈；`seq` 为序列容量。
fn embed_block_stack(vocab: usize, hidden: usize, kv: usize, seq: usize) -> Stack {
    Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, hidden, 1).expect("层")),
        Box::new(
            AttentionBlock::new(
                "blk.0",
                hidden,
                kv,
                kv,
                seq,
                AttentionAlgo::Naive,
                WeightFormat::F32,
            )
            .expect("层"),
        ),
    ])
    .expect("栈")
}

/// 嵌入 + 线性投影层栈（投影出宽 2）。
fn embed_projection_stack(vocab: usize, dim: usize) -> Stack {
    Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, dim, 1).expect("层")),
        Box::new(Linear::with_format("proj", WeightFormat::F32, dim, 2).expect("层")),
    ])
    .expect("栈")
}

/// 块四路驻留条目：词表 + q/k/v/o 投影权重。
fn block_params(table: Vec<f32>, wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>) -> Params {
    params_of(&[
        ("tok_emb.weight", table),
        ("blk.0.q.weight", wq),
        ("blk.0.k.weight", wk),
        ("blk.0.v.weight", wv),
        ("blk.0.o.weight", wo),
    ])
}

/// 嵌入 + 单头注意力块装配夹具：返回模型、嵌入表与 q/k/v/o 四路投影，decode 对拍链复用。
fn block_fixture(
    vocab: usize,
    hidden: usize,
    kv: usize,
) -> (CpuModel, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let table = embed_table(vocab, hidden);
    let (wq, wk, wv, wo) = block_projections(hidden, kv);
    let stack = embed_block_stack(vocab, hidden, kv, 4);
    let params = block_params(
        table.clone(),
        wq.clone(),
        wk.clone(),
        wv.clone(),
        wo.clone(),
    );
    let model = assemble(stack, params);
    (model, table, wq, wk, wv, wo)
}

/// 块四路零权重参数表（几何/缺键负例共用）。
fn block_zero_params() -> Params {
    params_of(&[
        ("blk.0.q.weight", vec![0.0; 64]),
        ("blk.0.k.weight", vec![0.0; 64]),
        ("blk.0.v.weight", vec![0.0; 64]),
        ("blk.0.o.weight", vec![0.0; 64]),
    ])
}

/// 参考注意力块：embed 行 → q/k/v gemv → RoPE（按头宽分段）→ decode 多头 → o 投影 → 残差；
/// 返回残差输出、旋转后 K 与明文 V（写回页行对拍用）。`head_dim` 为单头头宽，头数 = `kv/head_dim`。
#[allow(clippy::too_many_arguments)]
fn reference_block(
    embedded: &[f32],
    (wq, wk, wv, wo): (&[f32], &[f32], &[f32], &[f32]),
    (hist_k, hist_v): (&[f32], &[f32]),
    hist: usize,
    kv: usize,
    head_dim: usize,
    hidden: usize,
    pos: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; kv];
    op::gemv_f32::forward(embedded, wq, &mut q).expect("参考 q");
    let mut k = vec![0.0; kv];
    op::gemv_f32::forward(embedded, wk, &mut k).expect("参考 k");
    let mut v = vec![0.0; kv];
    op::gemv_f32::forward(embedded, wv, &mut v).expect("参考 v");
    let mut q_rot = vec![0.0; kv];
    op::rope_f32::forward_heads(pos, &q, &mut q_rot, head_dim).expect("参考 rope q");
    let mut k_rot = vec![0.0; kv];
    op::rope_f32::forward_heads(pos, &k, &mut k_rot, head_dim).expect("参考 rope k");
    let mut attn = vec![0.0; kv];
    op::attention::decode_forward_heads(&q_rot, hist_k, hist_v, &mut attn, 1, hist, head_dim)
        .expect("参考注意力");
    let mut o = vec![0.0; hidden];
    op::gemv_f32::forward(&attn, wo, &mut o).expect("参考 o");
    let mut residual = vec![0.0; hidden];
    op::residual_add_f32::forward(&o, embedded, &mut residual).expect("参考残差");
    (residual, k_rot, v)
}

/// 归档条目：名、形状 `[out, in]`、dtype 名与原始字节段。
struct ArchiveEntry {
    name: &'static str,
    shape: [usize; 2],
    dtype: &'static str,
    bytes: Vec<u8>,
}

/// 把 f32 整块写成小端字节段。
fn f32_block(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// 构造一份内存 safetensors 归档；数据段按条目顺序拼接，偏移写进头部 JSON。
fn archive_of(entries: &[ArchiveEntry]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(entries.len());
    for entry in entries {
        offsets.push(data.len());
        data.extend_from_slice(&entry.bytes);
    }
    let mut json = String::from("{");
    for (index, entry) in entries.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let begin = offsets[index];
        let end = begin + entry.bytes.len();
        json.push_str(&format!(
            "\"{}\": {{\"dtype\":\"{}\",\"shape\":[{},{}],\"data_offsets\":[{begin},{end}]}}",
            entry.name, entry.dtype, entry.shape[0], entry.shape[1]
        ));
    }
    json.push('}');
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json.as_bytes());
    file.extend(data);
    file
}

/// 两层 3→2→2 合成模型夹具：输入、归档字节与串行参考输出；装载路线由各用例自选。
fn two_layer_archive_fixture() -> ([f32; 3], Vec<u8>, Vec<f32>) {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let w2 = [1.0, 1.0, 2.0, 0.0];
    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    op::gemv_f32::forward(&hidden, &w2, &mut reference).expect("参考：第二层");
    let bytes = archive_of(&[
        ArchiveEntry {
            name: "blk.0.weight",
            shape: [2, 3],
            dtype: "F32",
            bytes: f32_block(&w1),
        },
        ArchiveEntry {
            name: "blk.1.weight",
            shape: [2, 2],
            dtype: "F32",
            bytes: f32_block(&w2),
        },
    ]);
    (x, bytes, reference)
}

/// 空头部归档：没有任何条目。
fn empty_archive() -> Vec<u8> {
    let json = b"{}";
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json);
    file
}

#[test]
fn gemv_runs_through_the_registered_kernel() {
    let x = [1.0, 2.0, 3.0];
    let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut reference = vec![0.0; 2];
    op::gemv_f32::forward(&x, &weights, &mut reference).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind(&mut executor, 1, &weights);
    let produced = run_one(&mut executor, "gemv_f32", &[0, 1], 2, 2);

    assert_close(&produced, &reference);
}

/// 行批投影：整块两行共享同一份权重，逐行 gemv 与串行参考逐行重算对拍。
#[test]
fn gemv_row_batch_runs_each_row_against_the_shared_weight() {
    let (m, n) = (3, 2);
    let w: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.5).collect();
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut reference = vec![0.0; n * 2];
    for row in 0..2 {
        let xs = &x[row * m..(row + 1) * m];
        op::gemv_f32::forward(xs, &w, &mut reference[row * n..(row + 1) * n]).expect("参考行");
    }

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind(&mut executor, 1, &w);
    let output = 2;
    executor.bind(output, vec![0.0; n * 2]);
    let mut node = Node::new("gemv_f32", Launch::new([1, 1, 1], [n as u32, m as u32, 1]));
    node.bind_input(0);
    node.bind_input(1);
    node.bind_output(output);
    executor.run(&node).expect("行批执行");
    let produced = executor.get(output).expect("输出缓冲").to_vec();

    assert_close(&produced, &reference);
}

/// 行批几何不自洽：输入不是行宽整数倍时拒绝，不静默截位。
#[test]
fn gemv_row_batch_rejects_inconsistent_geometry() {
    let mut executor = cpu();
    bind(&mut executor, 0, &[1.0, 2.0]);
    bind(&mut executor, 1, &[0.0; 6]);
    let output = 2;
    executor.bind(output, vec![0.0; 4]);
    let mut node = Node::new("gemv_f32", Launch::new([1, 1, 1], [2, 3, 1]));
    node.bind_input(0);
    node.bind_input(1);
    node.bind_output(output);

    assert!(executor.run(&node).is_err());
}

/// 把 [-8, 7] 的 int4 码按低 nibble 在前打成字节，口径与 `op::gemv_i4` 参考一致。
fn packed_i4(codes: &[i8]) -> Vec<u8> {
    codes
        .chunks(2)
        .map(|pair| {
            let low = (pair[0] + 8) as u8 & 0x0F;
            let high = pair.get(1).copied().unwrap_or(0) + 8;
            low | ((high as u8) & 0x0F) << 4
        })
        .collect()
}

/// 4 位量化乘加：`x: f32, w: u8, scales: f32 → y: f32`，与独立 `op` 参考对拍。
#[test]
fn gemv_i4_runs_through_the_registered_kernel() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let w = packed_i4(&[1, 2, 3, 4]);
    let scales = [0.5f32];
    let mut reference = vec![0.0; 1];
    op::gemv_i4::forward(&x, &w, &scales, &mut reference).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind_u8(&mut executor, 1, &w);
    bind(&mut executor, 2, &scales);
    let produced = run_one(&mut executor, "gemv_i4", &[0, 1, 2], 3, 1);

    assert_close(&produced, &reference);
}

/// 8 位量化乘加：权重逐元素 int8 位模式打包进 `u8`，与独立 `op` 参考对拍。
#[test]
fn gemv_i8_runs_through_the_registered_kernel() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let w: Vec<u8> = [1i8, 2, 3, 4].iter().map(|value| *value as u8).collect();
    let scales = [0.5f32];
    let mut reference = vec![0.0; 1];
    op::gemv_i8::forward(&x, &w, &scales, &mut reference).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind_u8(&mut executor, 1, &w);
    bind(&mut executor, 2, &scales);
    let produced = run_one(&mut executor, "gemv_i8", &[0, 1, 2], 3, 1);

    assert_close(&produced, &reference);
}

/// 4 位反量化：打包字节每字节两个 nibble，scale/zero 取样本常量，与独立 `op` 参考对拍。
#[test]
fn dequant_i4_runs_through_the_registered_kernel() {
    let q = [0x52u8];
    let mut reference = vec![0.0; 2];
    op::dequant_i4::forward(
        op::dequant_i4::SCALE,
        op::dequant_i4::ZERO,
        &q,
        &mut reference,
    )
    .expect("参考");

    let mut executor = cpu();
    bind_u8(&mut executor, 0, &q);
    let produced = run_one(&mut executor, "dequant_i4", &[0], 1, 2);

    assert_close(&produced, &reference);
}

/// 8 位反量化：码点本身是 `f32`，scale/zero 取样本常量，与独立 `op` 参考对拍。
#[test]
fn dequant_i8_runs_through_the_registered_kernel() {
    let q = [8.0f32, 12.0];
    let mut reference = vec![0.0; 2];
    op::dequant_i8::forward(
        op::dequant_i8::SCALE,
        op::dequant_i8::ZERO,
        &q,
        &mut reference,
    )
    .expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &q);
    let produced = run_one(&mut executor, "dequant_i8", &[0], 1, 2);

    assert_close(&produced, &reference);
}

/// 量化权重槽误绑成浮点缓冲：执行期形态核对给 `Invalid`，不静默按字节解释。
#[test]
fn a_quantized_weight_bound_as_f32_is_rejected_by_the_kernel() {
    let mut executor = cpu();
    bind(&mut executor, 0, &[1.0; 4]);
    bind(&mut executor, 1, &[0.0; 2]);
    bind(&mut executor, 2, &[0.5]);
    executor.bind(3, vec![0.0; 1]);

    let mut node = Node::new("gemv_i4", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0)
        .bind_input(1)
        .bind_input(2)
        .bind_output(3);

    let error = executor.run(&node).expect_err("权重形态不符必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn rmsnorm_runs_through_the_registered_kernel() {
    let x = [1.0, 2.0, 3.0];
    let gamma = [1.0, 1.0, 1.0];
    let mut reference = vec![0.0; 3];
    op::rmsnorm_f32::forward(&x, &gamma, &mut reference).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &x);
    bind(&mut executor, 1, &gamma);
    let produced = run_one(&mut executor, "rmsnorm_f32", &[0, 1], 2, 3);

    assert_close(&produced, &reference);
}

#[test]
fn binary_elementwise_runs_through_the_registered_kernel() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0];

    let mut add_ref = vec![0.0; 2];
    op::add_f32::forward(&a, &b, &mut add_ref).expect("参考");
    let mut mul_ref = vec![0.0; 2];
    op::mul_f32::forward(&a, &b, &mut mul_ref).expect("参考");
    let mut residual_ref = vec![0.0; 2];
    op::residual_add_f32::forward(&a, &b, &mut residual_ref).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &a);
    bind(&mut executor, 1, &b);

    let add = run_one(&mut executor, "add_f32", &[0, 1], 2, 2);
    let mul = run_one(&mut executor, "mul_f32", &[0, 1], 3, 2);
    let residual = run_one(&mut executor, "residual_add_f32", &[0, 1], 4, 2);

    assert_close(&add, &add_ref);
    assert_close(&mul, &mul_ref);
    assert_close(&residual, &residual_ref);
}

#[test]
fn unary_elementwise_runs_through_the_registered_kernel() {
    let a = [-1.0, 0.0, 1.0, 2.0];

    let mut silu_ref = vec![0.0; 4];
    op::silu_f32::forward(&a, &mut silu_ref).expect("参考");
    let mut gelu_ref = vec![0.0; 4];
    op::gelu_f32::forward(&a, &mut gelu_ref).expect("参考");
    let mut scale_ref = vec![0.0; 4];
    op::scale_f32::forward(&a, &mut scale_ref).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &a);

    let silu = run_one(&mut executor, "silu_f32", &[0], 1, 4);
    let gelu = run_one(&mut executor, "gelu_f32", &[0], 2, 4);
    let scale = run_one(&mut executor, "scale_f32", &[0], 3, 4);

    assert_close(&silu, &silu_ref);
    assert_close(&gelu, &gelu_ref);
    assert_close(&scale, &scale_ref);
}

#[test]
fn softmax_and_reduce_run_through_the_registered_kernel() {
    let x = [3.0, 1.0, 2.0];

    let mut softmax_ref = vec![0.0; 3];
    op::softmax_f32::forward(&x, &mut softmax_ref).expect("参考");
    let mut max_ref = vec![0.0; 1];
    op::reduce_max_f32::forward(&x, &mut max_ref).expect("参考");
    let mut sum_ref = vec![0.0; 1];
    op::reduce_sum_f32::forward(&x, &mut sum_ref).expect("参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &x);

    let softmax = run_one(&mut executor, "softmax_f32", &[0], 1, 3);
    let max = run_one(&mut executor, "reduce_max_f32", &[0], 2, 1);
    let sum = run_one(&mut executor, "reduce_sum_f32", &[0], 3, 1);

    assert_close(&softmax, &softmax_ref);
    assert_close(&max, &max_ref);
    assert_close(&sum, &sum_ref);
}

/// 注意力三路 q/k/v 经登记内核执行，朴素与 Flash 分别对拍同算子参考实现。
#[test]
fn attention_runs_through_the_registered_kernel() {
    let sample = op::attention::sample(2, 8);
    let q = sample.f32("q").expect("q");
    let k = sample.f32("k").expect("k");
    let v = sample.f32("v").expect("v");
    let d = op::attention::MODEL.shape.1;
    let s = q.len() / d;

    let mut plain = vec![0.0; q.len()];
    op::attention::forward(q, k, v, &mut plain, s, d).expect("朴素参考");
    let mut flash = vec![0.0; q.len()];
    op::attention::flash_forward(q, k, v, &mut flash, s, d).expect("flash 参考");

    let mut executor = cpu();
    bind(&mut executor, 0, q);
    bind(&mut executor, 1, k);
    bind(&mut executor, 2, v);

    let produced_plain = run_one(&mut executor, "attention_f32", &[0, 1, 2], 3, q.len());
    let produced_flash = run_one(&mut executor, "flash_attention_f32", &[0, 1, 2], 4, q.len());

    assert_close(&produced_plain, &plain);
    assert_close(&produced_flash, &flash);
}

/// 图序就是执行序：第二个节点读到第一个节点写过的编号，参考路径串行重算同一结果。
#[test]
fn a_graph_chains_through_the_same_bindings() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0];

    let mut middle = vec![0.0; 2];
    op::add_f32::forward(&a, &b, &mut middle).expect("参考：加和");
    let mut reference = vec![0.0; 2];
    op::silu_f32::forward(&middle, &mut reference).expect("参考：激活");

    let mut executor = cpu();
    bind(&mut executor, 0, &a);
    bind(&mut executor, 1, &b);
    executor.bind(2, vec![0.0; 2]);
    executor.bind(3, vec![0.0; 2]);

    let mut summed = Node::new("add_f32", Launch::new([1, 1, 1], [2, 1, 1]));
    summed.bind_input(0).bind_input(1).bind_output(2);
    let mut activated = Node::new("silu_f32", Launch::new([1, 1, 1], [2, 1, 1]));
    activated.bind_input(2).bind_output(3);

    let mut graph = Graph::new();
    graph.push(summed);
    graph.push(activated);
    executor.submit(&graph, Stream::new(0)).expect("入队");
    executor.sync(Stream::new(0)).expect("执行");

    let produced = executor.get(3).expect("末节点输出").to_vec();
    assert_close(&produced, &reference);
}

/// 表未覆盖的算子不登记；执行时按名拒绝，而不是静默走空入口。
#[test]
fn an_uncovered_operator_is_refused_by_name() {
    let mut executor = cpu();
    executor.bind(0, vec![1.0, 2.0]);
    let mut node = Node::new("gemm_f32", Launch::new([1, 1, 1], [2, 1, 1]));
    node.bind_input(0).bind_output(1);

    let error = executor.run(&node).expect_err("未覆盖算子必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("gemm_f32"), "{error}");
}

/// 两层 F32 线性链：装配出的图串行跑通，参考路径用同一批权重串行 gemv 重算。
#[test]
fn a_two_layer_chain_forward_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    // [2, 3] 行主序：行 0 取 x[0]，行 1 取 x[1]
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    // [2, 2] 行主序
    let w2 = [1.0, 1.0, 2.0, 0.0];

    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    op::gemv_f32::forward(&hidden, &w2, &mut reference).expect("参考：第二层");

    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 2, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", w1.to_vec()), ("blk.1.weight", w2.to_vec())]);

    forward_close(stack, params, &x, &reference);
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

/// 非 F32 编码（默认 int4）：P1 只接稠密浮点，装配期报 `Unsupported`。
#[test]
fn a_non_f32_layer_is_rejected_at_assembly() {
    let stack = Stack::new(vec![Linear::new("blk.0", 3, 2).expect("层")]).expect("栈");
    let params = params_of(&[("blk.0.weight", vec![0.0; 6])]);

    rejected_at_assembly(stack, params, Code::Unsupported);
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
    op::silu_f32::forward(&hidden, &mut reference).expect("参考：激活");

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
    op::rmsnorm_f32::forward(&hidden, &gamma, &mut reference).expect("参考：归一化");

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
    op::add_f32::forward(&hidden, &hidden, &mut reference).expect("参考：二元相加");

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
    op::mul_f32::forward(&hidden, &hidden, &mut reference).expect("参考：二元相乘");

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

/// 查表行：行 `k` 的第 `d` 维为 `k + d·0.1`，便于逐行核对 gather 取的是整行。
fn embed_table(vocab: usize, dim: usize) -> Vec<f32> {
    (0..vocab * dim)
        .map(|index| {
            let row = index / dim;
            let col = index % dim;
            row as f32 + col as f32 * 0.1
        })
        .collect()
}

/// 单步查表：装配出的图取索引对应整行，参考路径同一 `embed_f32` 串行重算。
#[test]
fn an_embedding_single_step_matches_the_reference() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let idx = [2.0f32];

    let mut reference = vec![0.0; dim];
    op::embed_f32::forward(&idx, &table, &mut reference).expect("参考：查表");

    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", vocab, dim, 1).expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[("tok_emb.weight", table.clone())]);

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
        op::embed_f32::forward(&idx, &table, &mut e).expect("参考：查表");
        e
    };
    let mut reference = vec![0.0; 2];
    op::gemv_f32::forward(&embedded, &w, &mut reference).expect("参考：投影");

    let stack = embed_projection_stack(vocab, dim);
    let params = params_of(&[
        ("tok_emb.weight", table.clone()),
        ("proj.weight", w.to_vec()),
    ]);

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

/// 多 token 查表需要矩阵主链通路，当前向量主链按 `Unsupported` 拒绝，留连续批刀落地。
#[test]
fn a_multi_token_embedding_is_not_assembled_yet() {
    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", 8, 4, 3).expect("层"),
    )])
    .expect("栈");
    let params = params_of(&[("tok_emb.weight", embed_table(8, 4))]);

    rejected_at_assembly(stack, params, Code::Unsupported);
}

/// Predict 一步前向：序列末位 token 折成索引，查表→投影主链对拍张量级 forward。
#[test]
fn predict_logits_from_an_embedding_sequence_match_the_reference() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let w = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let sequence = Sequence::new(0, &[2u32], 2, 16, Stream::new(0)).expect("序列");

    let stack = embed_projection_stack(vocab, dim);
    let params = params_of(&[
        ("tok_emb.weight", table.clone()),
        ("proj.weight", w.to_vec()),
    ]);

    let mut model = assemble(stack, params);
    let mut batch = plain_batch();
    let produced = model.logits(&sequence, &mut batch).expect("一步前向");
    let reference = model.forward(&[2.0]).expect("张量级参考");

    assert_close(&produced, &reference);
}

/// 串行参考：单 token 查表 → 投影一行；权重 `[out, in]`，输出宽 `w.len()/dim`。
fn reference_row(table: &[f32], w: &[f32], token: u32, dim: usize) -> Vec<f32> {
    let mut embedded = vec![0.0; dim];
    op::embed_f32::forward(&[token as f32], table, &mut embedded).expect("参考查表");
    let mut out = vec![0.0; w.len() / dim];
    op::gemv_f32::forward(&embedded, w, &mut out).expect("参考投影");
    out
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

    let rows = model
        .logits_batch(&[a.clone(), b.clone()], &mut batch)
        .expect("整批前向");
    assert_eq!(rows.len(), 2, "两行 logits 与输入切片一一对应");

    assert_close(&rows[0], &reference_row(&table, &w, 2, dim));
    assert_close(&rows[1], &reference_row(&table, &w, 5, dim));
}

/// 投影首层拿不到隐向量：序列只持有 token id，Predict 通路按 `Unsupported` 拒绝。
#[test]
fn predict_with_a_projection_first_layer_is_not_assembled_yet() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", vec![0.0; 6])]);

    let mut model = assemble(stack, params);
    let sequence = Sequence::new(0, &[2u32], 2, 16, Stream::new(0)).expect("序列");
    let mut batch = plain_batch();
    let error = model
        .logits(&sequence, &mut batch)
        .expect_err("投影首层不给向量输入");
    assert_eq!(error.code(), Code::Unsupported);
}

/// 投影权重 `[out, in]`：确定性小值矩阵，便于参考路径串行重算。
fn projection_matrix(out: usize, width_in: usize, seed: f32) -> Vec<f32> {
    (0..out * width_in)
        .map(|index| seed + index as f32 * 0.01)
        .collect()
}

/// decode 两条注意力内核经登记表跑通：与独立 `op` decode 参考对拍。
#[test]
fn decode_attention_kernels_match_the_reference() {
    let sample = op::decode_attention::sample(1, 8);
    let q = sample.f32("q").expect("q").to_vec();
    let k = sample.f32("k").expect("k").to_vec();
    let v = sample.f32("v").expect("v").to_vec();
    let (d, hist) = (8, 4);

    let mut naive_reference = vec![0.0; d];
    op::attention::decode_forward(&q, &k, &v, &mut naive_reference, 1, hist, d).expect("朴素参考");
    let mut flash_reference = vec![0.0; d];
    op::attention::decode_flash_forward(&q, &k, &v, &mut flash_reference, 1, hist, d)
        .expect("flash 参考");

    let mut executor = cpu();
    bind(&mut executor, 0, &q);
    bind(&mut executor, 1, &k);
    bind(&mut executor, 2, &v);
    let produced_naive = run_one(&mut executor, "decode_attention_f32", &[0, 1, 2], 3, d);
    let produced_flash = run_one(
        &mut executor,
        "decode_flash_attention_f32",
        &[0, 1, 2],
        4,
        d,
    );

    assert_close(&produced_naive, &naive_reference);
    assert_close(&produced_flash, &flash_reference);
}

/// 空历史走 decode 内核：`check_rows` 报 `Invalid`，不带进执行期。
#[test]
fn an_empty_decode_history_is_rejected_by_the_kernel() {
    let d = 8;
    let mut executor = cpu();
    bind(&mut executor, 0, &vec![0.0; d]);
    bind(&mut executor, 1, &[]);
    bind(&mut executor, 2, &[]);
    executor.bind(3, vec![0.0; d]);

    let mut node = Node::new(
        "decode_attention_f32",
        Launch::new([1, 1, 1], [d as u32, 1, 1]),
    );
    node.bind_input(0);
    node.bind_input(1);
    node.bind_input(2);
    node.bind_output(3);

    let error = executor.run(&node).expect_err("空历史必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 注意力块 decode 一步：embed → q/k/v → decode 注意力 → o → 残差旁路，与独立参考对拍；
/// 当前步 K/V 写回页行并按 `[k|v]` 切回对拍。
#[test]
fn an_attention_block_decodes_against_the_reference_and_roundtrips_kv() {
    let (vocab, hidden, kv) = (8, 8, 8);
    let hist = 1;

    let (mut model, table, wq, wk, wv, wo) = block_fixture(vocab, hidden, kv);
    let (hist_k, hist_v) = history_kv(kv);
    let history = history_row(&hist_k, &hist_v);

    let (mut batch, sequence) = decode_batch(kv, &history);
    let produced = model.logits(&sequence, &mut batch).expect("一步 decode");

    let embedded = &table[2 * hidden..3 * hidden];
    // q/k 先按当前步位置过 RoPE 旋转，再进注意力；写回页行的 K 同样取旋转后一路。
    let (reference, k_rot, v) = reference_block(
        embedded,
        (&wq, &wk, &wv, &wo),
        (&hist_k, &hist_v),
        hist,
        kv,
        kv,
        hidden,
        sequence.len(),
    );

    assert_close(&produced, &reference);

    let row = batch.kv(0, sequence.len()).expect("当前步页行");
    let (written_k, written_v) = row.split_at(kv);
    assert_close(written_k, &k_rot);
    assert_close(written_v, &v);
}

/// 两条序列同批走 `logits_batch`：行序对齐准入序，各自对拍参考块，KV 写回各归各的页行。
/// 当前默认逐条回退；整批几何接线后本用例作为回归基线，不改变对拍口径。
#[test]
fn logits_batch_runs_two_sequences_in_input_order_and_keeps_their_own_kv() {
    let (vocab, hidden, kv) = (8, 8, 8);
    let hist = 1;

    let (mut model, table, wq, wk, wv, wo) = block_fixture(vocab, hidden, kv);
    let (hist_k_a, hist_v_a) = history_kv(kv);
    let hist_k_b: Vec<f32> = hist_k_a.iter().map(|value| value * 1.7).collect();
    let hist_v_b: Vec<f32> = hist_v_a.iter().map(|value| value * 1.3).collect();
    let history_a = history_row(&hist_k_a, &hist_v_a);
    let history_b = history_row(&hist_k_b, &hist_v_b);

    let (mut batch, a, b) = decode_batch_two(kv, &history_a, &history_b);
    let rows = model
        .logits_batch(&[a.clone(), b.clone()], &mut batch)
        .expect("批量前向");
    assert_eq!(rows.len(), 2, "两行 logits 与输入切片一一对应");

    let embedded_a = &table[2 * hidden..3 * hidden];
    let embedded_b = &table[5 * hidden..6 * hidden];
    let (ref_a, k_a, v_a) = reference_block(
        embedded_a,
        (&wq, &wk, &wv, &wo),
        (&hist_k_a, &hist_v_a),
        hist,
        kv,
        kv,
        hidden,
        a.len(),
    );
    let (ref_b, k_b, v_b) = reference_block(
        embedded_b,
        (&wq, &wk, &wv, &wo),
        (&hist_k_b, &hist_v_b),
        hist,
        kv,
        kv,
        hidden,
        b.len(),
    );
    assert_close(&rows[0], &ref_a);
    assert_close(&rows[1], &ref_b);

    let row_a = batch.kv(0, a.len()).expect("A 当前步页行");
    let row_b = batch.kv(1, b.len()).expect("B 当前步页行");
    let (written_k_a, written_v_a) = row_a.split_at(kv);
    let (written_k_b, written_v_b) = row_b.split_at(kv);
    assert_close(written_k_a, &k_a);
    assert_close(written_v_a, &v_a);
    assert_close(written_k_b, &k_b);
    assert_close(written_v_b, &v_b);
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
        (&wq, &wk, &wv, &wo),
        (&hist_k, &hist_v),
        hist,
        kv,
        head_dim,
        hidden,
        sequence.len(),
    );
    assert_close(&produced, &reference);

    let row = batch.kv(0, sequence.len()).expect("当前步页行");
    let (written_k, written_v) = row.split_at(kv);
    assert_close(written_k, &k_rot);
    assert_close(written_v, &v);
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
        (&wq, &wk, &wv, &wo),
        (&hist_k, &hist_v),
        hist,
        kv,
        kv,
        hidden,
        sequence.len(),
    );
    let mut normed = vec![0.0; hidden];
    op::rmsnorm_f32::forward(&residual, &gamma, &mut normed).expect("参考归一化");
    let mut reference = vec![0.0; out];
    op::gemv_f32::forward(&normed, &w2, &mut reference).expect("参考输出投影");

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

/// 记录逐条调用次序的桩：默认批量回退必须按输入顺序逐条调 [`Predict::logits`]。
struct OrderProbe {
    calls: Vec<u64>,
}

impl Predict for OrderProbe {
    fn logits(&mut self, sequence: &Sequence, _batch: &mut Batch) -> diagnostic::Result<Vec<f32>> {
        self.calls.push(sequence.id());
        Ok(vec![sequence.id() as f32])
    }
}

/// 默认 `logits_batch` 逐条回退：行序与输入序列一一对应，空批返回空，不吞调用次序。
#[test]
fn logits_batch_default_runs_each_sequence_in_input_order() {
    let pager = Pager::new(Cache::new(2, 16).expect("页池"), 2).expect("行宽");
    let mut batch = Batch::new(pager);
    let first = Sequence::new(7, &[2u32], 2, 16, Stream::new(0)).expect("序列");
    let second = Sequence::new(3, &[2u32], 2, 16, Stream::new(0)).expect("序列");
    batch.admit(first.clone()).expect("准入");
    batch.admit(second.clone()).expect("准入");
    let sequences: Vec<Sequence> = batch.sequences().to_vec();

    let mut model = OrderProbe { calls: Vec::new() };
    let rows = model
        .logits_batch(&sequences, &mut batch)
        .expect("批量回退");

    assert_eq!(model.calls, [7, 3], "默认回退必须按准入顺序逐条调用");
    assert_eq!(rows.len(), 2, "返回行数与输入条数一致");
    assert_eq!(rows[0], [7.0]);
    assert_eq!(rows[1], [3.0]);
    assert!(
        model
            .logits_batch(&[], &mut batch)
            .expect("空批")
            .is_empty(),
        "空序列批量返回空行"
    );
}
