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
    let mut node = Node::new(op, Launch::new([1, 1, 1], [len as u32, 1, 1]));
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

    let hidden = {
        let mut h = vec![0.0; 2];
        op::gemv_f32::forward(&x, &w1, &mut h).expect("参考：第一层");
        h
    };
    let mut reference = vec![0.0; 2];
    op::gemv_f32::forward(&hidden, &w2, &mut reference).expect("参考：第二层");

    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 2, 2).expect("层"),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", w1.to_vec());
    params.insert("blk.1.weight", w2.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    assert_eq!(model.output_len(), 2);
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 描述表把编号映射回条目名与声明形状；激活缓冲没有条目名，形状只记单轴长度（D10 / ADR 0044）。
#[test]
fn descriptors_map_ids_back_to_entries_and_shapes() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 2, 2).expect("层"),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 6]);
    params.insert("blk.1.weight", vec![0.0; 4]);
    let model = CpuModel::new(&stack, &params).expect("装配");

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
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 5]); // 需要 3×2 = 6

    let error = CpuModel::new(&stack, &params).expect_err("长度不符必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 参数表缺条目：装配期按名报 `Missing`。
#[test]
fn a_missing_weight_is_rejected_at_assembly() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");

    let error = CpuModel::new(&stack, &Params::new()).expect_err("缺权重必须拒绝");
    assert_eq!(error.code(), Code::Missing);
}

/// 非 F32 编码（默认 int4）：P1 只接稠密浮点，装配期报 `Unsupported`。
#[test]
fn a_non_f32_layer_is_rejected_at_assembly() {
    let stack = Stack::new(vec![Linear::new("blk.0", 3, 2).expect("层")]).expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 6]);

    let error = CpuModel::new(&stack, &params).expect_err("非 F32 必须拒绝");
    assert_eq!(error.code(), Code::Unsupported);
}

/// 层间宽度链断裂：上一层输出宽度对不上本层输入宽度，装配期报 `Invalid`。
#[test]
fn a_width_chain_break_is_rejected_at_assembly() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
        Linear::with_format("blk.1", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 6]);
    params.insert("blk.1.weight", vec![0.0; 6]);

    let error = CpuModel::new(&stack, &params).expect_err("宽度链断裂必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 内存归档装载：头部推出层栈、Loader 取出条目字节、解码进驻留表，装配后 forward 与串行参考对拍。
#[test]
fn entries_from_an_in_memory_archive_load_into_params() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let w2 = [1.0, 1.0, 2.0, 0.0];

    let hidden = {
        let mut h = vec![0.0; 2];
        op::gemv_f32::forward(&x, &w1, &mut h).expect("参考：第一层");
        h
    };
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
    let archive = Archive::parse(&bytes).expect("归档合法");
    let stack = Stack::from_header(archive.header()).expect("层栈");
    let params = Params::from_stack(&stack, archive.header(), &archive).expect("装载");
    let mut model = CpuModel::new(&stack, &params).expect("装配");
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

    let hidden = {
        let mut h = vec![0.0; 2];
        op::gemv_f32::forward(&x, &w1, &mut h).expect("参考：投影");
        h
    };
    let mut reference = vec![0.0; 2];
    op::silu_f32::forward(&hidden, &mut reference).expect("参考：激活");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Silu::new("act", 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", w1.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    assert_eq!(model.output_len(), 2);
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 投影后接归一化：gamma 作为驻留权重整块装配，参考路径串行 `gemv → rmsnorm` 重算。
#[test]
fn a_projection_then_rmsnorm_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let gamma = [1.0, 1.0];

    let hidden = {
        let mut h = vec![0.0; 2];
        op::gemv_f32::forward(&x, &w1, &mut h).expect("参考：投影");
        h
    };
    let mut reference = vec![0.0; 2];
    op::rmsnorm_f32::forward(&hidden, &gamma, &mut reference).expect("参考：归一化");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(RmsNorm::new("norm.0", 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", w1.to_vec());
    params.insert("norm.0.weight", gamma.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    assert_eq!(model.output_len(), 2);
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 一元层宽度与上一层输出不等长：装配期报 `Invalid`。
#[test]
fn a_unary_width_chain_break_is_rejected_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Silu::new("act", 3).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 6]);

    let error = CpuModel::new(&stack, &params).expect_err("宽度链断裂必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 归一化缺驻留 gamma：装配期按名报 `Missing`。
#[test]
fn a_missing_rmsnorm_gamma_is_reported_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(RmsNorm::new("norm.0", 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 6]);

    let error = CpuModel::new(&stack, &params).expect_err("缺 gamma 必须拒绝");
    assert_eq!(error.code(), Code::Missing);
}

/// 逐元素二元层：主链输入 + 旁路激活两路接线，参考路径对旁路持有张量逐元素相加。
#[test]
fn a_binary_layer_takes_a_bypass_activation_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let hidden = {
        let mut h = vec![0.0; 2];
        op::gemv_f32::forward(&x, &w1, &mut h).expect("参考：投影");
        h
    };
    let mut reference = vec![0.0; 2];
    op::add_f32::forward(&hidden, &hidden, &mut reference).expect("参考：二元相加");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Add::new("res", 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", w1.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    assert_eq!(model.output_len(), 2);
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 逐元素乘法同族接线：参考路径对旁路持有张量逐元素相乘。
#[test]
fn a_binary_mul_layer_takes_a_bypass_activation_matches_the_reference() {
    let x = [1.0, 2.0, 3.0];
    let w1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let hidden = {
        let mut h = vec![0.0; 2];
        op::gemv_f32::forward(&x, &w1, &mut h).expect("参考：投影");
        h
    };
    let mut reference = vec![0.0; 2];
    op::mul_f32::forward(&hidden, &hidden, &mut reference).expect("参考：二元相乘");

    let stack = Stack::of(vec![
        Box::new(Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层")),
        Box::new(Mul::new("gate", 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", w1.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    let produced = model.forward(&x).expect("前向");

    assert_close(&produced, &reference);
}

/// 二元层前面没有可借用的旁路激活（首层即二元）：装配期报 `Invalid`，不静默走单输入。
#[test]
fn a_binary_layer_without_a_bypass_activation_is_rejected_at_assembly() {
    let stack = Stack::of(vec![Box::new(Add::new("res", 2).expect("层"))]).expect("栈");
    let error = CpuModel::new(&stack, &Params::new()).expect_err("无旁路必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
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

    let error = CpuModel::new(&stack, &params).expect_err("非 F32 必须拒绝");
    assert_eq!(error.code(), Code::Unsupported);
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
    let mut params = Params::new();
    params.insert("tok_emb.weight", table.clone());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    assert_eq!(model.output_len(), dim);
    let produced = model.forward(&idx).expect("前向");

    assert_close(&produced, &reference);
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

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, dim, 1).expect("层")),
        Box::new(Linear::with_format("proj", WeightFormat::F32, dim, 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", table.clone());
    params.insert("proj.weight", w.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    assert_eq!(model.output_len(), 2);
    let produced = model.forward(&idx).expect("前向");

    assert_close(&produced, &reference);
}

/// 查表缺驻留词表：装配期按名报 `Missing`。
#[test]
fn a_missing_embedding_table_is_reported_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", 8, 4, 1).expect("层"),
    )])
    .expect("栈");

    let error = CpuModel::new(&stack, &Params::new()).expect_err("缺词表必须拒绝");
    assert_eq!(error.code(), Code::Missing);
}

/// 词表元素数与 `vocab × dim` 不符：装配期报 `Invalid`。
#[test]
fn an_embedding_table_length_mismatch_is_rejected_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", 8, 4, 1).expect("层"),
    )])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", vec![0.0; 8 * 4 - 1]);

    let error = CpuModel::new(&stack, &params).expect_err("表长不符必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 多 token 查表需要矩阵主链通路，当前向量主链按 `Unsupported` 拒绝，留连续批刀落地。
#[test]
fn a_multi_token_embedding_is_not_assembled_yet() {
    let stack = Stack::of(vec![Box::new(
        Embedding::new("tok_emb", 8, 4, 3).expect("层"),
    )])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", embed_table(8, 4));

    let error = CpuModel::new(&stack, &params).expect_err("多 token 矩阵通路留后续");
    assert_eq!(error.code(), Code::Unsupported);
}

/// Predict 一步前向：序列末位 token 折成索引，查表→投影主链对拍张量级 forward。
#[test]
fn predict_logits_from_an_embedding_sequence_match_the_reference() {
    let (vocab, dim) = (8, 4);
    let table = embed_table(vocab, dim);
    let w = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let sequence = Sequence::new(0, &[2u32], 2, 16, Stream::new(0)).expect("序列");

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, dim, 1).expect("层")),
        Box::new(Linear::with_format("proj", WeightFormat::F32, dim, 2).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", table.clone());
    params.insert("proj.weight", w.to_vec());

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    let mut batch = plain_batch();
    let produced = model.logits(&sequence, &mut batch).expect("一步前向");
    let reference = model.forward(&[2.0]).expect("张量级参考");

    assert_close(&produced, &reference);
}

/// 投影首层拿不到隐向量：序列只持有 token id，Predict 通路按 `Unsupported` 拒绝。
#[test]
fn predict_with_a_projection_first_layer_is_not_assembled_yet() {
    let stack = Stack::new(vec![
        Linear::with_format("blk.0", WeightFormat::F32, 3, 2).expect("层"),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.weight", vec![0.0; 6]);

    let mut model = CpuModel::new(&stack, &params).expect("装配");
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

    let table = embed_table(vocab, hidden);
    let wq = projection_matrix(kv, hidden, 0.0);
    let wk = projection_matrix(kv, hidden, 1.0);
    let wv = projection_matrix(kv, hidden, 2.0);
    let wo = projection_matrix(hidden, kv, 3.0);

    let hist_k: Vec<f32> = (0..kv).map(|i| -0.3 + i as f32 * 0.05).collect();
    let hist_v: Vec<f32> = (0..kv).map(|i| 0.2 - i as f32 * 0.04).collect();
    let mut history_row = Vec::with_capacity(2 * kv);
    history_row.extend_from_slice(&hist_k);
    history_row.extend_from_slice(&hist_v);

    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", vocab, hidden, 1).expect("层")),
        Box::new(
            AttentionBlock::new(
                "blk.0",
                hidden,
                kv,
                kv,
                4,
                AttentionAlgo::Naive,
                WeightFormat::F32,
            )
            .expect("层"),
        ),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", table.clone());
    params.insert("blk.0.q.weight", wq.clone());
    params.insert("blk.0.k.weight", wk.clone());
    params.insert("blk.0.v.weight", wv.clone());
    params.insert("blk.0.o.weight", wo.clone());

    let pager = Pager::new(Cache::new(4, 4).expect("页池"), 2 * kv).expect("行宽");
    let mut batch = Batch::new(pager);
    let sequence = Sequence::new(0, &[2u32], 2, 4, Stream::new(0)).expect("序列");
    batch.admit(sequence.clone()).expect("准入");
    batch.write_kv(0, 0, &history_row).expect("种子历史");

    let mut model = CpuModel::new(&stack, &params).expect("装配");
    let produced = model.logits(&sequence, &mut batch).expect("一步 decode");

    let embedded = &table[2 * hidden..3 * hidden];
    let mut q = vec![0.0; kv];
    op::gemv_f32::forward(embedded, &wq, &mut q).expect("参考 q");
    let mut k = vec![0.0; kv];
    op::gemv_f32::forward(embedded, &wk, &mut k).expect("参考 k");
    let mut v = vec![0.0; kv];
    op::gemv_f32::forward(embedded, &wv, &mut v).expect("参考 v");
    // q/k 先按当前步位置过 RoPE 旋转，再进注意力；写回页行的 K 同样取旋转后一路。
    let pos = sequence.len();
    let mut q_rot = vec![0.0; kv];
    op::rope_f32::forward(pos, &q, &mut q_rot).expect("参考 rope q");
    let mut k_rot = vec![0.0; kv];
    op::rope_f32::forward(pos, &k, &mut k_rot).expect("参考 rope k");
    let mut attn = vec![0.0; kv];
    op::attention::decode_forward(&q_rot, &hist_k, &hist_v, &mut attn, 1, hist, kv)
        .expect("参考注意力");
    let mut o = vec![0.0; hidden];
    op::gemv_f32::forward(&attn, &wo, &mut o).expect("参考 o");
    let mut reference = vec![0.0; hidden];
    op::residual_add_f32::forward(&o, embedded, &mut reference).expect("参考残差");

    assert_close(&produced, &reference);

    let row = batch.kv(0, sequence.len()).expect("当前步页行");
    let (written_k, written_v) = row.split_at(kv);
    assert_close(written_k, &k_rot);
    assert_close(written_v, &v);
}

/// 块内一个投影条目缺失：装配期按名报 `Missing`。
#[test]
fn a_missing_block_projection_weight_is_reported_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        AttentionBlock::new("blk.0", 8, 8, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
            .expect("层"),
    )])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.k.weight", vec![0.0; 64]);
    params.insert("blk.0.v.weight", vec![0.0; 64]);
    params.insert("blk.0.o.weight", vec![0.0; 64]);

    let error = CpuModel::new(&stack, &params).expect_err("缺 q 权重必须拒绝");
    assert_eq!(error.code(), Code::Missing);
}

/// 上一层输出宽度与块隐藏宽度不符：装配期报 `Invalid`。
#[test]
fn a_block_width_chain_break_is_rejected_at_assembly() {
    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", 8, 4, 1).expect("层")),
        Box::new(
            AttentionBlock::new("blk.0", 8, 8, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
                .expect("层"),
        ),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", embed_table(8, 4));
    params.insert("blk.0.q.weight", vec![0.0; 64]);
    params.insert("blk.0.k.weight", vec![0.0; 64]);
    params.insert("blk.0.v.weight", vec![0.0; 64]);
    params.insert("blk.0.o.weight", vec![0.0; 64]);

    let error = CpuModel::new(&stack, &params).expect_err("宽度链断裂必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 块内权重元素数与 `in × out` 不符：装配期报 `Invalid`。
#[test]
fn a_block_weight_length_mismatch_is_rejected_at_assembly() {
    let stack = Stack::of(vec![Box::new(
        AttentionBlock::new("blk.0", 8, 8, 8, 4, AttentionAlgo::Naive, WeightFormat::F32)
            .expect("层"),
    )])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.q.weight", vec![0.0; 63]);
    params.insert("blk.0.k.weight", vec![0.0; 64]);
    params.insert("blk.0.v.weight", vec![0.0; 64]);
    params.insert("blk.0.o.weight", vec![0.0; 64]);

    let error = CpuModel::new(&stack, &params).expect_err("权重长度不符必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 多头展平（`kv_width ≠ head_dim`）：单头通路按 `Unsupported` 拒绝，多头切片留后续。
#[test]
fn a_multi_head_block_is_not_assembled_yet() {
    let stack = Stack::of(vec![Box::new(
        AttentionBlock::new(
            "blk.0",
            8,
            16,
            8,
            4,
            AttentionAlgo::Naive,
            WeightFormat::F32,
        )
        .expect("层"),
    )])
    .expect("栈");
    let mut params = Params::new();
    params.insert("blk.0.q.weight", vec![0.0; 16 * 8]);
    params.insert("blk.0.k.weight", vec![0.0; 16 * 8]);
    params.insert("blk.0.v.weight", vec![0.0; 16 * 8]);
    params.insert("blk.0.o.weight", vec![0.0; 8 * 16]);

    let error = CpuModel::new(&stack, &params).expect_err("多头通路留后续");
    assert_eq!(error.code(), Code::Unsupported);
}
