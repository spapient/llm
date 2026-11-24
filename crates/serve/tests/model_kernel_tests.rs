use super::*;

/// 量化 gemv 参考内核的函数指针形态：`x/w/scales → out`。
type QuantGemv = fn(tensor::View<'_, f32>, &[u8], tensor::View<'_, f32>, &mut [f32]) -> Result<()>;

/// 量化 gemv 对拍夹具：跑一遍参考内核、绑三路输入，返回输入编号、输出槽与参考缓冲。
fn quant_gemv_smoke(
    executor: &mut CpuExecutor,
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    forward: QuantGemv,
) -> (Vec<u32>, u32, Vec<f32>) {
    let mut reference = vec![0.0; 1];
    forward(
        tensor::View::one(x),
        w,
        tensor::View::one(scales),
        &mut reference,
    )
    .expect("参考");
    bind(executor, 0, x);
    bind_u8(executor, 1, w);
    bind(executor, 2, scales);
    (vec![0, 1, 2], 3, reference)
}

kernel_smoke!(
    gemv_runs_through_the_registered_kernel,
    "gemv_f32",
    |executor| {
        let x = [1.0, 2.0, 3.0];
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut reference = vec![0.0; 2];
        nn::gemv_f32::forward(tensor::View::one(&x), &weights, &mut reference).expect("参考");
        bind(executor, 0, &x);
        bind(executor, 1, &weights);
        (vec![0, 1], 2, reference)
    }
);

/// 行批投影：整块两行共享同一份权重，逐行 gemv 与串行参考逐行重算对拍。
#[test]
fn gemv_row_batch_runs_each_row_against_the_shared_weight() {
    let (m, n) = (3, 2);
    let w: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.5).collect();
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut reference = vec![0.0; n * 2];
    for row in 0..2 {
        let xs = &x[row * m..(row + 1) * m];
        nn::gemv_f32::forward(
            tensor::View::one(xs),
            &w,
            &mut reference[row * n..(row + 1) * n],
        )
        .expect("参考行");
    }

    let produced = run_gemv_rows(&x, &w, m, n, 2);

    assert_close(&produced, &reference);
}

/// 行批跨过微内核行块（`nn::fast::MIN_ROWS`）后换到 packed 入口：两侧各取一个批宽，否则这条分派
/// 只有一侧被覆盖——2 行走按列 tile，`MIN_ROWS + 1` 行走 8×8 寄存器分块那条。
#[test]
fn gemv_row_batch_switches_to_the_packed_kernel_at_the_row_block() {
    let (m, n) = (5, 7);
    let w: Vec<f32> = (0..n * m).map(|i| (i % 9) as f32 * 0.25 - 1.0).collect();
    for rows in [2, nn::fast::MIN_ROWS + 1] {
        let x: Vec<f32> = (0..rows * m).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
        let mut reference = vec![0.0; n * rows];
        for row in 0..rows {
            nn::gemv_f32::forward(
                tensor::View::one(&x[row * m..(row + 1) * m]),
                &w,
                &mut reference[row * n..][..n],
            )
            .expect("参考行");
        }

        let produced = run_gemv_rows(&x, &w, m, n, rows);

        assert_close(&produced, &reference);
    }
}

/// 行批几何不自洽：输入不是行宽整数倍时拒绝，不静默截位。
#[test]
fn gemv_row_batch_rejects_inconsistent_geometry() {
    let mut executor = cpu();
    bind(&mut executor, 0, &[1.0, 2.0]);
    bind(&mut executor, 1, &[0.0; 6]);
    let output = 2;
    executor.bind(output, Buf::F32(vec![0.0; 4]));
    let mut node = Node::new("gemv_f32", Launch::new([1, 1, 1], [2, 3, 1]));
    node.bind_input(0);
    node.bind_input(1);
    node.bind_output(output);

    assert!(executor.run(&node).is_err());
}

// 4 位量化乘加：`x: f32, w: u8, scales: f32 → y: f32`，与独立 `nn` 参考对拍。
kernel_smoke!(
    gemv_i4_runs_through_the_registered_kernel,
    "gemv_i4",
    |executor| {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = pack_codes(4, &[1, 2, 3, 4]);
        let scales = [0.5f32];
        quant_gemv_smoke(executor, &x, &w, &scales, nn::gemv_i4::forward)
    }
);

// 8 位量化乘加：权重逐元素 int8 位模式打包进 `u8`，与独立 `nn` 参考对拍。
kernel_smoke!(
    gemv_i8_runs_through_the_registered_kernel,
    "gemv_i8",
    |executor| {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w: Vec<u8> = [1i8, 2, 3, 4].iter().map(|value| *value as u8).collect();
        let scales = [0.5f32];
        quant_gemv_smoke(executor, &x, &w, &scales, nn::gemv_i8::forward)
    }
);

// 4 位反量化：打包字节每字节两个 nibble，scale/zero 取样本常量，与独立 `nn` 参考对拍。
kernel_smoke!(
    dequant_i4_runs_through_the_registered_kernel,
    "dequant_i4",
    |executor| {
        let q = [0x52u8];
        let mut reference = vec![0.0; 2];
        nn::dequant_i4::forward(
            nn::dequant_i4::SCALE,
            nn::dequant_i4::ZERO,
            &q,
            &mut reference,
        )
        .expect("参考");
        bind_u8(executor, 0, &q);
        (vec![0], 1, reference)
    }
);

// 8 位反量化：码点本身是 `f32`，scale/zero 取样本常量，与独立 `nn` 参考对拍。
kernel_smoke!(
    dequant_i8_runs_through_the_registered_kernel,
    "dequant_i8",
    |executor| {
        let q = [8.0f32, 12.0];
        let mut reference = vec![0.0; 2];
        nn::dequant_i8::forward(
            nn::dequant_i8::SCALE,
            nn::dequant_i8::ZERO,
            tensor::View::one(&q),
            &mut reference,
        )
        .expect("参考");
        bind(executor, 0, &q);
        (vec![0], 1, reference)
    }
);

// 2 位反量化：每字节四个码，scale/zero 取样本常量，与独立 `nn` 参考对拍。
kernel_smoke!(
    dequant_i2_runs_through_the_registered_kernel,
    "dequant_i2",
    |executor| {
        let q = [0x52u8];
        let mut reference = vec![0.0; 4];
        nn::dequant_i2::forward(
            nn::dequant_i2::SCALE,
            nn::dequant_i2::ZERO,
            &q,
            &mut reference,
        )
        .expect("参考");
        bind_u8(executor, 0, &q);
        (vec![0], 1, reference)
    }
);

// 2 位量化乘加：`x: f32, w: u8, scales: f32 → y: f32`，与独立 `nn` 参考对拍。
kernel_smoke!(
    gemv_i2_runs_through_the_registered_kernel,
    "gemv_i2",
    |executor| {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = pack_codes(2, &[1, -1, 0, 1]);
        let scales = [0.5f32];
        quant_gemv_smoke(executor, &x, &w, &scales, nn::gemv_i2::forward)
    }
);

/// 量化 GEMM 经注册 Program 对拍：A/B 打包字节、Y 浮点；几何由样本静态轴与缓冲长度反推。
///
/// 三条位宽 lane 共用同一套夹具：样本缓冲取 `nn::<lane>::sample`，参考走同一 `forward`，
/// 执行器拆两路 u8 输入、一路 f32 输出；打包规则完全来自 `nn`，夹具不重写。
fn gemm_quant_matches_reference(
    lane: &str,
    a: &[u8],
    b: &[u8],
    m: usize,
    n: usize,
    reference: &[f32],
) {
    let mut executor = cpu();
    bind_u8(&mut executor, 0, a);
    bind_u8(&mut executor, 1, b);
    let produced = run_one(&mut executor, lane, &[0, 1], 2, reference.len());
    assert_eq!(produced.len(), m * n);
    assert_close(&produced, reference);
}

/// int4 矩阵乘：样本 A/B 为打包 nibble，与独立 `nn` 参考对拍。
#[test]
fn gemm_i4_runs_through_the_registered_kernel() {
    let (m, n) = (2, 3);
    let sample = nn::gemm::sample::<nn::gemm_i4::GemmI4>(m, n);
    let a = sample.u8("a").expect("a");
    let b = sample.u8("b").expect("b");
    let mut reference = vec![0.0; m * n];
    nn::gemm_i4::forward(a, b, m, n, &mut reference).expect("参考");

    gemm_quant_matches_reference("gemm_i4", a, b, m, n, &reference);
}

/// int8 矩阵乘：样本 A/B 为逐元素字节，与独立 `nn` 参考对拍。
#[test]
fn gemm_i8_runs_through_the_registered_kernel() {
    let (m, n) = (2, 3);
    let sample = nn::gemm::sample::<nn::gemm_i8::GemmI8>(m, n);
    let a = sample.u8("a").expect("a");
    let b = sample.u8("b").expect("b");
    let mut reference = vec![0.0; m * n];
    nn::gemm_i8::forward(a, b, m, n, &mut reference).expect("参考");

    gemm_quant_matches_reference("gemm_i8", a, b, m, n, &reference);
}

/// int2 矩阵乘：样本 A/B 为打包 2 位码，与独立 `nn` 参考对拍。
#[test]
fn gemm_i2_runs_through_the_registered_kernel() {
    let (m, n) = (2, 3);
    let sample = nn::gemm::sample::<nn::gemm_i2::GemmI2>(m, n);
    let a = sample.u8("a").expect("a");
    let b = sample.u8("b").expect("b");
    let mut reference = vec![0.0; m * n];
    nn::gemm_i2::forward(a, b, m, n, &mut reference).expect("参考");

    gemm_quant_matches_reference("gemm_i2", a, b, m, n, &reference);
}

/// 量化权重槽误绑成浮点缓冲：执行期形态核对给 `Invalid`，不静默按字节解释。
#[test]
fn a_quantized_weight_bound_as_f32_is_rejected_by_the_kernel() {
    let mut executor = cpu();
    bind(&mut executor, 0, &[1.0; 4]);
    bind(&mut executor, 1, &[0.0; 2]);
    bind(&mut executor, 2, &[0.5]);
    executor.bind(3, Buf::F32(vec![0.0; 1]));

    let mut node = Node::new("gemv_i4", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0)
        .bind_input(1)
        .bind_input(2)
        .bind_output(3);

    let error = executor.run(&node).expect_err("权重形态不符必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

kernel_smoke!(
    rmsnorm_runs_through_the_registered_kernel,
    "rmsnorm_f32",
    |executor| {
        let x = [1.0, 2.0, 3.0];
        let gamma = [1.0, 1.0, 1.0];
        let mut reference = vec![0.0; 3];
        nn::rmsnorm_f32::forward(
            tensor::View::one(&x),
            tensor::View::one(&gamma),
            &mut reference,
        )
        .expect("参考");
        bind(executor, 0, &x);
        bind(executor, 1, &gamma);
        (vec![0, 1], 2, reference)
    }
);

#[test]
fn binary_elementwise_runs_through_the_registered_kernel() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0];

    let mut add_ref = vec![0.0; 2];
    nn::add_f32::forward(tensor::View::one(&a), tensor::View::one(&b), &mut add_ref).expect("参考");
    let mut mul_ref = vec![0.0; 2];
    nn::mul_f32::forward(tensor::View::one(&a), tensor::View::one(&b), &mut mul_ref).expect("参考");
    let mut residual_ref = vec![0.0; 2];
    nn::residual_add_f32::forward(
        tensor::View::one(&a),
        tensor::View::one(&b),
        &mut residual_ref,
    )
    .expect("参考");

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
    nn::silu_f32::forward(tensor::View::one(&a), &mut silu_ref).expect("参考");
    let mut gelu_ref = vec![0.0; 4];
    nn::gelu_f32::forward(tensor::View::one(&a), &mut gelu_ref).expect("参考");
    let mut scale_ref = vec![0.0; 4];
    nn::scale_f32::forward(tensor::View::one(&a), &mut scale_ref).expect("参考");

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
    nn::softmax_f32::forward(tensor::View::one(&x), &mut softmax_ref).expect("参考");
    let mut max_ref = vec![0.0; 1];
    nn::reduce_max_f32::forward(tensor::View::one(&x), &mut max_ref).expect("参考");
    let mut sum_ref = vec![0.0; 1];
    nn::reduce_sum_f32::forward(tensor::View::one(&x), &mut sum_ref).expect("参考");

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
    let sample = nn::conv::attention::sample(2, 8);
    let q = sample.f32("q").expect("q");
    let k = sample.f32("k").expect("k");
    let v = sample.f32("v").expect("v");
    let d = nn::conv::attention::MODEL.shape.1;
    let s = q.len() / d;

    let mut plain = vec![0.0; q.len()];
    nn::conv::attention::forward(
        tensor::View::matrix(s, d, q).expect("q 二维"),
        tensor::View::matrix(s, d, k).expect("k 二维"),
        tensor::View::matrix(s, d, v).expect("v 二维"),
        &mut plain,
    )
    .expect("朴素参考");
    let mut flash = vec![0.0; q.len()];
    nn::conv::attention::flash_forward(
        tensor::View::matrix(s, d, q).expect("q 二维"),
        tensor::View::matrix(s, d, k).expect("k 二维"),
        tensor::View::matrix(s, d, v).expect("v 二维"),
        &mut flash,
    )
    .expect("flash 参考");

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
    nn::add_f32::forward(tensor::View::one(&a), tensor::View::one(&b), &mut middle)
        .expect("参考：加和");
    let mut reference = vec![0.0; 2];
    nn::silu_f32::forward(tensor::View::one(&middle), &mut reference).expect("参考：激活");

    let mut executor = cpu();
    bind(&mut executor, 0, &a);
    bind(&mut executor, 1, &b);
    executor.bind(2, Buf::F32(vec![0.0; 2]));
    executor.bind(3, Buf::F32(vec![0.0; 2]));

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
    executor.bind(0, Buf::F32(vec![1.0, 2.0]));
    let mut node = Node::new("gemm_f32", Launch::new([1, 1, 1], [2, 1, 1]));
    node.bind_input(0).bind_output(1);

    let error = executor.run(&node).expect_err("未覆盖算子必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("gemm_f32"), "{error}");
}

/// decode 两条注意力内核经登记表跑通：与独立 `nn` decode 参考对拍。
#[test]
fn decode_attention_kernels_match_the_reference() {
    let sample = nn::conv::decode_attention::sample(1, 8);
    let q = sample.f32("q").expect("q").to_vec();
    let k = sample.f32("k").expect("k").to_vec();
    let v = sample.f32("v").expect("v").to_vec();
    let (d, hist) = (8, 4);

    let mut naive_reference = vec![0.0; d];
    nn::conv::attention::decode_forward(
        tensor::View::matrix(1, d, &q).expect("q 二维"),
        tensor::View::matrix(hist, d, &k).expect("k 二维"),
        tensor::View::matrix(hist, d, &v).expect("v 二维"),
        &mut naive_reference,
    )
    .expect("朴素参考");
    let mut flash_reference = vec![0.0; d];
    nn::conv::attention::decode_flash_forward(
        tensor::View::matrix(1, d, &q).expect("q 二维"),
        tensor::View::matrix(hist, d, &k).expect("k 二维"),
        tensor::View::matrix(hist, d, &v).expect("v 二维"),
        &mut flash_reference,
    )
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
    executor.bind(3, Buf::F32(vec![0.0; d]));

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

#[test]
fn a_row_batch_projection_equals_row_by_row_gemv() {
    // 权重 `[n, k]` = `[4, 5]`，3 行输入：多行走线性层分块核，必须与逐行点积同值。
    let (m, n, rows) = (5usize, 4usize, 3usize);
    let x: Vec<f32> = (0..rows * m)
        .map(|i| (i as f32 * 0.25 - 1.5).sin())
        .collect();
    let weights: Vec<f32> = (0..n * m).map(|i| (i as f32 * 0.5 + 0.25).cos()).collect();

    let mut batched = vec![0.0f32; rows * n];
    project_rows(&x, &weights, &mut batched, m, n).expect("行批投影");
    let mut row_by_row = vec![0.0f32; rows * n];
    for row in 0..rows {
        nn::gemv_f32::forward(
            tensor::View::one(&x[row * m..(row + 1) * m]),
            &weights,
            &mut row_by_row[row * n..(row + 1) * n],
        )
        .expect("单行投射");
    }
    assert_close(&batched, &row_by_row);

    // 一行仍走 `gemv`：decode 口径与直接调用逐位相同，不受分块核影响。
    let mut single = vec![0.0f32; n];
    project_rows(&x[..m], &weights, &mut single, m, n).expect("单行投影");
    let mut reference = vec![0.0f32; n];
    nn::gemv_f32::forward(tensor::View::one(&x[..m]), &weights, &mut reference).expect("gemv");
    assert_eq!(single, reference, "decode 口径必须与 gemv 逐位一致");
}
