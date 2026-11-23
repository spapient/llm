use super::*;

#[test]
fn flash_and_plain_attention_agree_and_rows_are_distributions() {
    let sample = sample(8, 8);
    let plain = reference(&sample).expect("朴素参考");
    let flash = flash_reference(&sample).expect("flash 参考");
    assert!(
        testkit::approx_eq(&plain, &flash, 1e-5),
        "两套算法必须一致：逐元素绝对容差 1e-5"
    );
    // 每个 query 行的注意力权重隐含归一化：输出对全 1 的 V 应得 1，这里直接核对确定性结果非 NaN。
    assert!(testkit::all_finite(&plain));

    let mut out = vec![0.0f32; 64];
    forward(
        sample.f32("q").unwrap(),
        sample.f32("k").unwrap(),
        sample.f32("v").unwrap(),
        &mut out,
        8,
        8,
    )
    .expect("朴素内核");
    assert_eq!(out, plain);

    let mut flash_out = vec![0.0f32; 64];
    flash_forward(
        sample.f32("q").unwrap(),
        sample.f32("k").unwrap(),
        sample.f32("v").unwrap(),
        &mut flash_out,
        8,
        8,
    )
    .expect("flash 内核");
    assert!(testkit::approx_eq(&flash_out, &flash, 1e-5));
}

/// 行主序确定性向量：`rows × d`，第 `i` 个元素取 `base + i·step`，便于逐行核对。
fn det_rows(rows: usize, d: usize, base: f32, step: f32) -> Vec<f32> {
    (0..rows * d).map(|idx| base + idx as f32 * step).collect()
}

#[test]
fn decode_flash_and_plain_agree_with_single_query_row() {
    let (q_rows, hist, d) = (1, 5, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    let mut plain = vec![0.0f32; q_rows * d];
    decode_forward(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素 decode");
    assert!(testkit::all_finite(&plain));

    let mut flash = vec![0.0f32; q_rows * d];
    decode_flash_forward(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash decode");
    assert!(
        testkit::approx_eq(&flash, &plain, 1e-5),
        "decode 朴素与 flash 必须一致：逐元素绝对容差 1e-5"
    );
}

#[test]
fn decode_supports_multiple_query_rows_against_history() {
    let (q_rows, hist, d) = (2, 4, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    let mut plain = vec![0.0f32; q_rows * d];
    decode_forward(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素 decode");
    let mut flash = vec![0.0f32; q_rows * d];
    decode_flash_forward(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash decode");

    assert_plain_flash(&plain, &flash);
}

#[test]
fn decode_rejects_buf_length_mismatch() {
    let (q_rows, hist, d) = (1, 5, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    // k 少一行：历史长度对不上。
    let mut out = vec![0.0f32; q_rows * d];
    assert!(decode_forward(&q, &k[..(hist - 1) * d], &v, &mut out, q_rows, hist, d).is_err());
    // 输出缓冲行数对不上。
    let mut short_out = vec![0.0f32; d - 1];
    assert!(decode_forward(&q, &k, &v, &mut short_out, q_rows, hist, d).is_err());
}

#[test]
fn decode_rejects_zero_head_width_or_empty_rows() {
    let empty: &[f32] = &[];
    let mut out = vec![];
    // d=0：√d 除零。
    assert!(decode_forward(empty, empty, empty, &mut out, 1, 5, 0).is_err());
    assert!(decode_flash_forward(empty, empty, empty, &mut out, 1, 5, 0).is_err());
    // 历史为空：softmax 分母为 0。
    let q = det_rows(1, 4, -0.5, 0.13);
    let mut hist_out = vec![0.0f32; 4];
    assert!(decode_forward(&q, empty, empty, &mut hist_out, 1, 0, 4).is_err());
    // 查询行为空。
    assert!(decode_forward(empty, &[0.0; 4], &[0.0; 4], &mut out, 0, 1, 4).is_err());
}

#[test]
fn attention_rejects_zero_head_width() {
    // d=0：三个输入与输出都是空缓冲，长度校验放行后由头宽守卫拦下，避免 √d 除零成 NaN。
    let empty: &[f32] = &[];
    let mut out = vec![];
    assert!(forward(empty, empty, empty, &mut out, 8, 0).is_err());
    assert!(flash_forward(empty, empty, empty, &mut out, 8, 0).is_err());
    assert!(reference(&sample(8, 0)).is_err(), "朴素参考同样守卫头宽");
    assert!(
        flash_reference(&sample(8, 0)).is_err(),
        "flash 参考同样守卫头宽"
    );
}

/// 无头 decode 的确定性 q/k/v：`q` 取 `q_rows` 行、`k`/`v` 各取 `hist` 行，同一条线性序列。
fn decode_qkv(q_rows: usize, hist: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    headed_qkv(1, q_rows, hist, d)
}

/// 多头 decode 的确定性 q/k/v：头优先块布局。
fn headed_qkv(
    heads: usize,
    q_rows: usize,
    hist: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        det_rows(heads * q_rows, d, -0.5, 0.13),
        det_rows(heads * hist, d, 0.4, -0.09),
        det_rows(heads * hist, d, -0.2, 0.11),
    )
}

/// plain/flash 两套 decode 的公共对拍：全有限 + 逐元素绝对容差 1e-5。
fn assert_plain_flash(plain: &[f32], flash: &[f32]) {
    assert!(testkit::all_finite(plain));
    assert!(testkit::approx_eq(flash, plain, 1e-5));
}

/// 常数 V 段校验：段 `seq` 全取 `1 + seq`，softmax 权重和为 1，输出应恒等于该段常数。
fn assert_constant_v(plain: &[f32], q_seq: usize, batch: usize) {
    for seq in 0..batch {
        let constant = 1.0 + seq as f32;
        let span = &plain[seq * q_seq..(seq + 1) * q_seq];
        assert!(
            span.iter().all(|value| (value - constant).abs() < 1e-5),
            "序列 {seq} 输出应恒等于本段 V 常数 {constant}，得到 {span:?}"
        );
    }
}

/// 常数 V 段构造：第 `seq` 段全取 `1 + seq`，供段间隔离与常数输出核对。
fn constant_v(spans: impl Iterator<Item = usize>) -> Vec<f32> {
    spans
        .enumerate()
        .flat_map(|(seq, len)| std::iter::repeat_n(1.0 + seq as f32, len))
        .collect()
}

#[test]
fn multi_head_decode_keeps_each_head_independent_and_plain_matches_flash() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let q = det_rows(heads * q_rows, d, -0.5, 0.13);
    let k = det_rows(heads * hist, d, 0.4, -0.09);
    // V 按头取常数：头 0 全 1、头 1 全 2。softmax 权重和为 1，常数 V 下头内输出恒等于该常数，
    // 借此核对两头结果互不串段——头 0 段不应出现 2，头 1 段不应出现 1。
    let v = constant_v(std::iter::repeat_n(hist * d, heads));

    let mut plain = vec![0.0; heads * q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素多头 decode");
    let mut flash = vec![0.0; heads * q_rows * d];
    decode_flash_forward_heads(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash 多头 decode");

    assert_plain_flash(&plain, &flash);
    for head in 0..heads {
        let constant = 1.0 + head as f32;
        let span = &plain[head * q_rows * d..(head + 1) * q_rows * d];
        assert!(
            span.iter().all(|value| (value - constant).abs() < 1e-5),
            "头 {head} 输出应恒等于本头 V 常数 {constant}，得到 {span:?}"
        );
    }
}

#[test]
fn one_head_entry_matches_the_single_head_kernel() {
    let (q_rows, hist, d) = (2, 4, 4);
    let (q, k, v) = decode_qkv(q_rows, hist, d);

    let mut single = vec![0.0; q_rows * d];
    decode_forward(&q, &k, &v, &mut single, q_rows, hist, d).expect("单头朴素");
    let mut multi = vec![0.0; q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut multi, q_rows, hist, d).expect("单头经多头入口");
    assert_eq!(single, multi, "heads=1 必须与单头内核逐元素等价");
}

#[test]
fn multi_head_rejects_non_multiple_buf_length() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);
    let mut out = vec![0.0; heads * q_rows * d];
    // q 只给一个头的长度，而 k/v 是两头：头块倍数校验应拦下。
    assert!(decode_forward_heads(&q[..q_rows * d], &k, &v, &mut out, q_rows, hist, d).is_err());
    // q 长度不是 q_rows·d 的整数倍：无法推出整头，直接拒绝。
    let mut ragged = vec![0.0; q.len() - 1];
    assert!(decode_forward_heads(&q[..q.len() - 1], &k, &v, &mut ragged, q_rows, hist, d).is_err());
}

#[test]
fn batch_decode_keeps_each_sequence_independent_and_plain_matches_flash() {
    let (q_rows, hist, heads, d, batch) = (1, 4, 2, 3, 2);
    let q_seq = heads * q_rows * d;
    let kv_seq = heads * hist * d;
    let mut q = Vec::new();
    let mut k = Vec::new();
    for seq in 0..batch {
        q.extend(det_rows(heads * q_rows, d, -0.5 + seq as f32, 0.13));
        k.extend(det_rows(heads * hist, d, 0.4 + seq as f32, -0.09));
    }
    // V 按序列段取常数：段 0 全 1、段 1 全 2。softmax 权重和为 1，常数 V 下段输出恒等于
    // 该段常数，借此核对两条序列结果互不串段。
    let v = constant_v(std::iter::repeat_n(kv_seq, batch));

    let mut plain = vec![0.0; batch * q_seq];
    decode_forward_batch(&q, &k, &v, &mut plain, q_rows, hist, d, heads, batch)
        .expect("朴素批量 decode");
    let mut flash = vec![0.0; batch * q_seq];
    decode_flash_forward_batch(&q, &k, &v, &mut flash, q_rows, hist, d, heads, batch)
        .expect("flash 批量 decode");

    assert_plain_flash(&plain, &flash);
    assert_constant_v(&plain, q_seq, batch);
}

/// 单段经批量入口必须与多头入口逐元素等价；`batch_entry` 给出这一侧的跑法。
fn single_batch_matches_multi_head(
    (q_rows, hist, heads, d): (usize, usize, usize, usize),
    batch_entry: impl Fn((usize, usize, usize, usize), &[f32], &[f32], &[f32], &mut [f32]),
) {
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);
    let mut multi = vec![0.0; heads * q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut multi, q_rows, hist, d).expect("多头朴素");
    let mut batch = vec![0.0; heads * q_rows * d];
    batch_entry((q_rows, hist, heads, d), &q, &k, &v, &mut batch);
    assert_eq!(multi, batch, "batch=1 必须与多头入口逐元素等价");
}

#[test]
fn batch_one_entry_matches_the_multi_head_kernel() {
    single_batch_matches_multi_head((2, 4, 2, 3), |(q_rows, hist, heads, d), q, k, v, out| {
        decode_forward_batch(q, k, v, out, q_rows, hist, d, heads, 1).expect("单段经批量入口");
    });
}

#[test]
fn batch_rejects_non_multiple_length_and_zero_batch() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let (q, k, v) = headed_qkv(heads, q_rows, hist, d);
    let mut out = vec![0.0; heads * q_rows * d];
    // batch=2 但 q/k/v 只给一段：批块倍数校验应拦下。
    assert!(decode_forward_batch(&q, &k, &v, &mut out, q_rows, hist, d, heads, 2).is_err());
    // batch=0：序列数必须非零。
    assert!(decode_forward_batch(&q, &k, &v, &mut out, q_rows, hist, d, heads, 0).is_err());
    // q 长度不是批段整数倍：无法推出整段，直接拒绝。
    let mut ragged = vec![0.0; q.len() - 1];
    assert!(
        decode_forward_batch(
            &q[..q.len() - 1],
            &k,
            &v,
            &mut ragged,
            q_rows,
            hist,
            d,
            heads,
            1
        )
        .is_err()
    );
}

#[test]
fn batch_var_keeps_mixed_history_lengths_independent_and_plain_matches_flash() {
    let (q_rows, heads, d) = (1, 2, 3);
    let hist = [3usize, 5];
    let batch = hist.len();
    let q_seq = heads * q_rows * d;
    let mut q = Vec::new();
    let mut k = Vec::new();
    for (seq, &h) in hist.iter().enumerate() {
        q.extend(det_rows(heads * q_rows, d, -0.5 + seq as f32, 0.13));
        k.extend(det_rows(heads * h, d, 0.4 + seq as f32, -0.09));
    }
    // V 按序列段取常数：段 0 全 1、段 1 全 2。softmax 权重和为 1，常数 V 下段输出恒等于
    // 该段常数，借此核对两条不同历史长度的序列结果互不串段。
    let v = constant_v(hist.iter().map(|&h| heads * h * d));

    let mut plain = vec![0.0; batch * q_seq];
    decode_forward_batch_var(&q, &k, &v, &mut plain, q_rows, &hist, d, heads)
        .expect("朴素变长批量 decode");
    let mut flash = vec![0.0; batch * q_seq];
    decode_flash_forward_batch_var(&q, &k, &v, &mut flash, q_rows, &hist, d, heads)
        .expect("flash 变长批量 decode");

    assert_plain_flash(&plain, &flash);
    assert_constant_v(&plain, q_seq, batch);
}

#[test]
fn batch_var_one_entry_matches_the_multi_head_kernel() {
    single_batch_matches_multi_head((1, 4, 2, 3), |(q_rows, hist, heads, d), q, k, v, out| {
        decode_forward_batch_var(q, k, v, out, q_rows, &[hist], d, heads).expect("变长入口单段");
    });
}

#[test]
fn batch_var_rejects_mismatched_kv_length_empty_or_zero_history() {
    let (q_rows, heads, d) = (1, 2, 3);
    let hist = [3usize, 5];
    let q: Vec<_> = (0..heads * q_rows * d * hist.len()).map(|_| 0.1).collect();
    let k: Vec<_> = (0..heads * hist.iter().sum::<usize>() * d)
        .map(|_| 0.2)
        .collect();
    let v: Vec<_> = k.clone();
    let mut out = vec![0.0; q.len()];

    // K/V 总长与逐段历史之和不符：游标切段守不住，直接拒绝。
    assert!(
        decode_forward_batch_var(&q, &k[..k.len() - 1], &v, &mut out, q_rows, &hist, d, heads)
            .is_err()
    );
    // 空历史表：序列数必须非零。
    assert!(decode_forward_batch_var(&q, &k, &v, &mut out, q_rows, &[], d, heads).is_err());
    // 某段历史为 0：段内行数守卫拦下。
    assert!(decode_forward_batch_var(&q, &k, &v, &mut out, q_rows, &[3, 0], d, heads).is_err());
}
