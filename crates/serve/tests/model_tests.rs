use super::*;
use crate::Predict;
use crate::assembly::Params;
use crate::model::CpuModel;
use crate::proto::{Decoder, Envelope, HEADER_LEN, Kind, MAX_PAYLOAD, TRAILER_LEN, Version};
use crate::{Response, Route, Status};
use diagnostic::Code;
use engine::exec::{CpuExecutor, Device, Executor, Graph, Launch, Node, Stream, Trace};
use engine::kv::{Cache, Pager};
use engine::{Batch, Scheduler, Sequence};
use nn::{
    Add, AttentionAlgo, AttentionBlock, Embedding, Linear, MemoryBlock, Mul, RmsNorm, Silu, Stack,
    WeightFormat,
};
use testkit::{all_finite, close, pack_codes, safetensors_bytes, scratch_dir};
use tokenizer::{Bytes, Tokenizer};
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

/// 对拍口径：先排非有限值，再走绝对 + 相对双项预算。同一份 `nn` 语义两次调用，
/// 位级应一致，容差只兜浮点读回误差。
fn assert_close(produced: &[f32], reference: &[f32]) {
    assert!(all_finite(produced), "生产路径出现非有限值");
    assert!(all_finite(reference), "参考路径出现非有限值");
    assert!(
        close(produced, reference, 1e-6, 1e-6),
        "生产 {produced:?} 与参考 {reference:?} 不一致"
    );
}

/// 当前步 K/V 页行回环：拆半行核对写回的 K 与 V 各自与参考链一致。
fn assert_kv_row(batch: &Batch, position: usize, kv: usize, k_rot: &[f32], v: &[f32]) {
    let row = batch.kv(0, position).expect("当前步页行");
    let (written_k, written_v) = row.split_at(kv);
    assert_close(written_k, k_rot);
    assert_close(written_v, v);
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
/// 批量前向回归基线夹具——块链整批一次提交，历史段紧排装填、当前步按段写回各自页行。
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

/// 两条一行种子历史的块链夹具：装配模型 + 两条序列批 + 两行历史引用；`head_dim` 与 `seq` 显式。
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn two_seq_fixture(
    kv: usize,
    head_dim: usize,
    seq: usize,
) -> (
    CpuModel,
    Batch,
    [Sequence; 2],
    Vec<(Vec<f32>, Vec<f32>)>,
    Vec<f32>,
    Projections,
) {
    let (model, table, projections) = block_fixture_dim(8, 8, kv, head_dim, seq);
    let (hist_k_a, hist_v_a) = history_kv(kv);
    let hist_k_b: Vec<f32> = hist_k_a.iter().map(|value| value * 1.7).collect();
    let hist_v_b: Vec<f32> = hist_v_a.iter().map(|value| value * 1.3).collect();
    let history_a = history_row(&hist_k_a, &hist_v_a);
    let history_b = history_row(&hist_k_b, &hist_v_b);
    let (batch, a, b) = decode_batch_two(kv, &history_a, &history_b);
    let histories = vec![(hist_k_a, hist_v_a), (hist_k_b, hist_v_b)];
    (model, batch, [a, b], histories, table, projections)
}

/// 参考链一层 gemv：`x · w` 写进新向量返回，输出宽度按权重行数推得。
fn gemv_hidden(x: &[f32], w: &[f32]) -> Vec<f32> {
    let out = w.len() / x.len();
    let mut h = vec![0.0; out];
    nn::gemv_f32::forward(x, w, &mut h).expect("参考：投影");
    h
}

/// 块四路投影向量：q/k/v/o。
type Projections = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// 块四路投影小值矩阵：q/k/v 取种子 0/1/2，o 取种子 3。
fn block_projections(hidden: usize, kv: usize) -> Projections {
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

/// 嵌入 + 注意力块层栈；`seq` 为序列容量，头宽默认等于整段 `kv`（单头）。
fn embed_block_stack(vocab: usize, hidden: usize, kv: usize, seq: usize) -> Stack {
    embed_block_stack_dim(vocab, hidden, kv, kv, seq)
}

/// 嵌入 + 注意力块层栈：`head_dim` 显式给定，头数 = `kv / head_dim`；`seq` 为序列容量。
fn embed_block_stack_dim(
    vocab: usize,
    hidden: usize,
    kv: usize,
    head_dim: usize,
    seq: usize,
) -> Stack {
    Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, hidden, 1).expect("层")),
        Box::new(
            AttentionBlock::new(
                "blk.0",
                hidden,
                kv,
                head_dim,
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
fn block_fixture(vocab: usize, hidden: usize, kv: usize) -> (CpuModel, Vec<f32>, Projections) {
    block_fixture_dim(vocab, hidden, kv, kv, 4)
}

/// 嵌入 + 注意力块装配夹具（头宽/序列容量显式）：返回模型、嵌入表与四路投影，多头/ragged 链复用。
fn block_fixture_dim(
    vocab: usize,
    hidden: usize,
    kv: usize,
    head_dim: usize,
    seq: usize,
) -> (CpuModel, Vec<f32>, Projections) {
    let table = embed_table(vocab, hidden);
    let projections = block_projections(hidden, kv);
    let stack = embed_block_stack_dim(vocab, hidden, kv, head_dim, seq);
    let params = block_params(
        table.clone(),
        projections.0.clone(),
        projections.1.clone(),
        projections.2.clone(),
        projections.3.clone(),
    );
    let model = assemble(stack, params);
    (model, table, projections)
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

/// 参考块的输入束：嵌入行、KV 历史与几何（历史行数、宽度、头宽、当前步位置）。
///
/// 这些参数在两族参考块之间完全相同，收成一束后核心链只需一处转发。
#[derive(Clone, Copy)]
struct BlockInput<'a> {
    embedded: &'a [f32],
    hist_k: &'a [f32],
    hist_v: &'a [f32],
    hist: usize,
    kv: usize,
    head_dim: usize,
    hidden: usize,
    pos: usize,
}

/// 参考块的投影权重：稠密 f32 四路，或 int4 打包码 + 标度四路。
///
/// 两族的数值链其余部分（RoPE / decode 注意力 / 残差）逐位同族，只有一次 gemv 的调用形状不同。
enum BlockWeights<'a> {
    /// 稠密 f32：q/k/v/o 四路行主序权重。
    Dense((&'a [f32], &'a [f32], &'a [f32], &'a [f32])),
    /// int4 打包：每路一份码字节与一条按 32 元素分组的标度。
    Int4(
        (&'a [u8], &'a [f32]),
        (&'a [u8], &'a [f32]),
        (&'a [u8], &'a [f32]),
        (&'a [u8], &'a [f32]),
    ),
}

/// 参考注意力块核心链：q/k/v 投影 → RoPE（按头宽分段）→ decode 多头 → o 投影 → 残差。
/// 稠密（`gemv_f32`）与量化（`gemv_i4`）只差投影闭包，其余链同族，故此收敛。
/// 返回残差输出、旋转后 K 与明文 V（写回页行对拍用）。`head_dim` 为单头头宽，头数 = `kv/head_dim`。
fn reference_block_core(
    input: BlockInput<'_>,
    project_q: impl Fn(&mut [f32]),
    project_k: impl Fn(&mut [f32]),
    project_v: impl Fn(&mut [f32]),
    project_o: impl Fn(&[f32], &mut [f32]),
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let BlockInput {
        embedded,
        hist_k,
        hist_v,
        hist,
        kv,
        head_dim,
        hidden,
        pos,
    } = input;
    let mut q = vec![0.0; kv];
    project_q(&mut q);
    let mut k = vec![0.0; kv];
    project_k(&mut k);
    let mut v = vec![0.0; kv];
    project_v(&mut v);
    let mut q_rot = vec![0.0; kv];
    nn::rope_f32::forward_heads(pos, tensor::View::one(&q), &mut q_rot, head_dim)
        .expect("参考 rope q");
    let mut k_rot = vec![0.0; kv];
    nn::rope_f32::forward_heads(pos, tensor::View::one(&k), &mut k_rot, head_dim)
        .expect("参考 rope k");
    let mut attn = vec![0.0; kv];
    nn::conv::attention::decode_forward_heads(&q_rot, hist_k, hist_v, &mut attn, 1, hist, head_dim)
        .expect("参考注意力");
    let mut o = vec![0.0; hidden];
    project_o(&attn, &mut o);
    let mut residual = vec![0.0; hidden];
    nn::residual_add_f32::forward(
        tensor::View::one(&o),
        tensor::View::one(embedded),
        &mut residual,
    )
    .expect("参考残差");
    (residual, k_rot, v)
}

/// 参考块：四路投影按 `weights` 走 `gemv_f32` 或 `gemv_i4`，其余链交给
/// [`reference_block_core`]。
#[allow(clippy::too_many_arguments)]
fn reference_block(
    embedded: &[f32],
    weights: BlockWeights<'_>,
    (hist_k, hist_v): (&[f32], &[f32]),
    hist: usize,
    kv: usize,
    head_dim: usize,
    hidden: usize,
    pos: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input = BlockInput {
        embedded,
        hist_k,
        hist_v,
        hist,
        kv,
        head_dim,
        hidden,
        pos,
    };
    match weights {
        BlockWeights::Dense((wq, wk, wv, wo)) => reference_block_core(
            input,
            |q| nn::gemv_f32::forward(embedded, wq, q).expect("参考 q"),
            |k| nn::gemv_f32::forward(embedded, wk, k).expect("参考 k"),
            |v| nn::gemv_f32::forward(embedded, wv, v).expect("参考 v"),
            |attn, o| nn::gemv_f32::forward(attn, wo, o).expect("参考 o"),
        ),
        BlockWeights::Int4((wq, sq), (wk, sk), (wv, sv), (wo, so)) => reference_block_core(
            input,
            |q| nn::gemv_i4::forward(embedded, wq, sq, q).expect("参考 q"),
            |k| nn::gemv_i4::forward(embedded, wk, sk, k).expect("参考 k"),
            |v| nn::gemv_i4::forward(embedded, wv, sv, v).expect("参考 v"),
            |attn, o| nn::gemv_i4::forward(attn, wo, so, o).expect("参考 o"),
        ),
    }
}

/// 对拍一条批内序列：参考块残差与生产 logits 对齐，当前步 K/V 页行拆半行核对写回归属。
#[allow(clippy::too_many_arguments)]
fn assert_batch_row(
    batch: &Batch,
    id: u64,
    position: usize,
    kv: usize,
    produced: &[f32],
    embedded: &[f32],
    projections: &Projections,
    history: (&[f32], &[f32]),
    hist: usize,
    head_dim: usize,
    hidden: usize,
) {
    let (wq, wk, wv, wo) = projections;
    let (reference, k_rot, v) = reference_block(
        embedded,
        BlockWeights::Dense((wq, wk, wv, wo)),
        history,
        hist,
        kv,
        head_dim,
        hidden,
        position,
    );
    assert_close(produced, &reference);
    let row = batch.kv(id, position).expect("当前步页行");
    let (written_k, written_v) = row.split_at(kv);
    assert_close(written_k, &k_rot);
    assert_close(written_v, &v);
}

/// 成对批链整批前向并对拍：一次提交后按准入序逐行套 [`assert_batch_row`]，`tokens` 取各行
/// 嵌入词，`hists` 取各行历史行数。
#[allow(clippy::too_many_arguments)]
fn run_block_pair(
    model: &mut CpuModel,
    batch: &mut Batch,
    sequences: &[Sequence],
    kv: usize,
    table: &[f32],
    projections: &Projections,
    histories: &[(&[f32], &[f32])],
    hists: &[usize],
    head_dim: usize,
    hidden: usize,
    tokens: &[usize],
) -> diagnostic::Result<()> {
    let rows = model.logits_batch(sequences, batch)?;
    assert_eq!(rows.len(), sequences.len(), "行 logits 与输入切片一一对应");
    for (idx, sequence) in sequences.iter().enumerate() {
        let embedded = &table[tokens[idx] * hidden..(tokens[idx] + 1) * hidden];
        assert_batch_row(
            batch,
            sequence.id(),
            sequence.len(),
            kv,
            &rows[idx],
            embedded,
            projections,
            histories[idx],
            hists[idx],
            head_dim,
            hidden,
        );
    }
    Ok(())
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
    for (idx, entry) in entries.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        let begin = offsets[idx];
        let end = begin + entry.bytes.len();
        json.push_str(&format!(
            "\"{}\": {{\"dtype\":\"{}\",\"shape\":[{},{}],\"data_offsets\":[{begin},{end}]}}",
            entry.name, entry.dtype, entry.shape[0], entry.shape[1]
        ));
    }
    json.push('}');
    let chunks: Vec<(&str, Vec<u8>)> = entries
        .iter()
        .map(|entry| (entry.name, entry.bytes.clone()))
        .collect();
    safetensors_bytes(&json, &chunks)
}

/// 两层 3→2→2 合成模型夹具：输入、归档字节与串行参考输出；装载路线由各用例自选。
fn two_layer_archive_fixture() -> ([f32; 3], Vec<u8>, Vec<f32>) {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let w2 = [1.0, 1.0, 2.0, 0.0];
    let hidden = gemv_hidden(&x, &w1);
    let mut reference = vec![0.0; 2];
    nn::gemv_f32::forward(&hidden, &w2, &mut reference).expect("参考：第二层");
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

/// 把归档字节写进进程隔离的临时文件，返回路径；装配根用真实文件路径走打开→装载→前向。
fn write_archive_file(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = scratch_dir(tag).join("model.safetensors");
    std::fs::write(&path, bytes).expect("写入临时权重文件");
    path
}

/// 空头部归档：没有任何条目。
fn empty_archive() -> Vec<u8> {
    safetensors_bytes("{}", &[])
}

/// 注册内核冒烟测试的公共模板：建空执行器 → 参考实现算一遍 → bind → `run_one` → 对拍。
///
/// 调用形状 `kernel_smoke!(测试名, "内核名", |executor| { ... })`：闭包拿到执行器，绑好输入并
/// 返回 `(输入编号, 输出编号, 参考输出)`，输出长度取参考长度。宏只收拢样板，各用例的参考调用
/// 与断言原样保留。
macro_rules! kernel_smoke {
    ($(#[$meta:meta])* $name:ident, $op:literal, |$executor:ident| $body:block) => {
        $(#[$meta])*
        #[test]
        fn $name() {
            let mut executor = cpu();
            let (inputs, output, reference) =
                (|$executor: &mut CpuExecutor| $body)(&mut executor);
            let produced = run_one(&mut executor, $op, &inputs, output, reference.len());
            assert_close(&produced, &reference);
        }
    };
}

/// 走一次 `gemv_f32` 节点：`m` 行宽、`n` 列数、`rows` 行，返回执行器产出的整段输出。
fn run_gemv_rows(x: &[f32], w: &[f32], m: usize, n: usize, rows: usize) -> Vec<f32> {
    let mut executor = cpu();
    bind(&mut executor, 0, x);
    bind(&mut executor, 1, w);
    let output = 2;
    executor.bind(output, vec![0.0; n * rows]);
    let mut node = Node::new("gemv_f32", Launch::new([1, 1, 1], [n as u32, m as u32, 1]));
    node.bind_input(0);
    node.bind_input(1);
    node.bind_output(output);
    executor.run(&node).expect("行批执行");
    executor.get(output).expect("输出缓冲").to_vec()
}

/// 两层 F32 线性栈 `3 → 2 → 2`：返回栈、配套参数表与两份行主序权重，参考链与装配吃同一份权重。
fn two_layer_f32_fixture() -> (Stack, Params, [f32; 6], [f32; 4]) {
    // [2, 3] 行主序：行 0 取 x[0]，行 1 取 x[1]
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    // [2, 2] 行主序
    let w2 = [1.0, 1.0, 2.0, 0.0];
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 2, 2).expect("层"),
    ])
    .expect("栈");
    let params = params_of(&[("blk.0.weight", w1.to_vec()), ("blk.1.weight", w2.to_vec())]);
    (stack, params, w1, w2)
}

/// 查表行：行 `k` 的第 `d` 维为 `k + d·0.1`，便于逐行核对 gather 取的是整行。
fn embed_table(vocab: usize, dim: usize) -> Vec<f32> {
    (0..vocab * dim)
        .map(|idx| {
            let row = idx / dim;
            let col = idx % dim;
            row as f32 + col as f32 * 0.1
        })
        .collect()
}

/// 串行参考：单 token 查表 → 投影一行；权重 `[out, in]`，输出宽 `w.len()/dim`。
fn reference_row(table: &[f32], w: &[f32], token: u32, dim: usize) -> Vec<f32> {
    let mut embedded = vec![0.0; dim];
    nn::embed_f32::forward(
        tensor::View::one(&[token as f32]),
        tensor::View::one(table),
        &mut embedded,
    )
    .expect("参考查表");
    let mut out = vec![0.0; w.len() / dim];
    nn::gemv_f32::forward(&embedded, w, &mut out).expect("参考投影");
    out
}

/// 查表 → 条件记忆：主链宽度 `dim`，表槽数 `buckets`，尾部 `order`-gram 按 `gate` 加回。
fn memory_stack(
    vocab: usize,
    dim: usize,
    order: usize,
    seed: u64,
    buckets: usize,
    gate: f32,
) -> Stack {
    Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, dim, 1).expect("身份嵌入")),
        Box::new(MemoryBlock::new("mem", dim, order, seed, buckets, gate).expect("条件记忆")),
    ])
    .expect("两层栈")
}

/// 串行参考：主链查表一行；历史够阶时再按槽加回门控后的表行，与装配侧三节点同序。
///
/// 累加次序固定为「先主链、后记忆」：装配图里记忆节点读的是主链的**输出**，两者不可交换。
// 测试参考实现：几何与数据同为入参，拆结构体只会让调用点更长。
#[allow(clippy::too_many_arguments)]
fn memory_reference(
    token: u32,
    history: &[u32],
    table: &[f32],
    memory_table: &[f32],
    order: usize,
    seed: u64,
    buckets: usize,
    gate: f32,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; dim];
    nn::embed_f32::forward(
        tensor::View::one(&[token as f32]),
        tensor::View::one(table),
        &mut out,
    )
    .expect("参考查表");
    let Some(slot) = nn::slot_of(history, order, seed, buckets) else {
        return out;
    };
    for (at, value) in out.iter_mut().enumerate() {
        *value += gate * memory_table[slot * dim + at];
    }
    out
}

/// 投影权重 `[out, in]`：确定性小值矩阵，便于参考路径串行重算。
fn projection_matrix(out: usize, width_in: usize, seed: f32) -> Vec<f32> {
    (0..out * width_in)
        .map(|idx| seed + idx as f32 * 0.01)
        .collect()
}

/// int4 投影的打包权重 + 标度：`out` 行、每行 `in` 元素按 32 一组共享一条标度，码取确定性小值。
fn quant_projection(out: usize, in_width: usize, seed: i8) -> (Vec<u8>, Vec<f32>) {
    let codes: Vec<i8> = (0..out * in_width)
        .map(|idx| (seed + (idx as i8 % 16)).rem_euclid(16) - 8)
        .collect();
    let groups = in_width / 32;
    let scales: Vec<f32> = (0..out * groups)
        .map(|idx| 0.25 + (idx % 3) as f32 * 0.25)
        .collect();
    (pack_codes(4, &codes), scales)
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

#[path = "model_kernel_tests.rs"]
mod kernel;

#[path = "model_attention_tests.rs"]
mod attention;

#[path = "model_stack_tests.rs"]
mod stack;

#[path = "model_embed_tests.rs"]
mod embed;

#[path = "model_protocol_tests.rs"]
mod protocol;
