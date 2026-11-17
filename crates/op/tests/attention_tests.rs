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
    (0..rows * d)
        .map(|index| base + index as f32 * step)
        .collect()
}

#[test]
fn decode_flash_and_plain_agree_with_single_query_row() {
    let (q_rows, hist, d) = (1, 5, 4);
    let q = det_rows(q_rows, d, -0.5, 0.13);
    let k = det_rows(hist, d, 0.4, -0.09);
    let v = det_rows(hist, d, -0.2, 0.11);

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
    let q = det_rows(q_rows, d, -0.5, 0.13);
    let k = det_rows(hist, d, 0.4, -0.09);
    let v = det_rows(hist, d, -0.2, 0.11);

    let mut plain = vec![0.0f32; q_rows * d];
    decode_forward(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素 decode");
    let mut flash = vec![0.0f32; q_rows * d];
    decode_flash_forward(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash decode");

    assert!(testkit::all_finite(&plain));
    assert!(testkit::approx_eq(&flash, &plain, 1e-5));
}

#[test]
fn decode_rejects_buffer_length_mismatch() {
    let (q_rows, hist, d) = (1, 5, 4);
    let q = det_rows(q_rows, d, -0.5, 0.13);
    let k = det_rows(hist, d, 0.4, -0.09);
    let v = det_rows(hist, d, -0.2, 0.11);

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

/// 头优先块布局：第 `h` 头的 `q`/`out` 是一段连续 `rows·d`，`k`/`v` 是一段连续 `hist·d`。
fn det_headed(heads: usize, rows: usize, d: usize, base: f32, step: f32) -> Vec<f32> {
    (0..heads * rows * d)
        .map(|index| base + index as f32 * step)
        .collect()
}

#[test]
fn multi_head_decode_keeps_each_head_independent_and_plain_matches_flash() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let q = det_headed(heads, q_rows, d, -0.5, 0.13);
    let k = det_headed(heads, hist, d, 0.4, -0.09);
    // V 按头取常数：头 0 全 1、头 1 全 2。softmax 权重和为 1，常数 V 下头内输出恒等于该常数，
    // 借此核对两头结果互不串段——头 0 段不应出现 2，头 1 段不应出现 1。
    let mut v = vec![0.0; heads * hist * d];
    for head in 0..heads {
        let constant = 1.0 + head as f32;
        for value in v[head * hist * d..(head + 1) * hist * d].iter_mut() {
            *value = constant;
        }
    }

    let mut plain = vec![0.0; heads * q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut plain, q_rows, hist, d).expect("朴素多头 decode");
    let mut flash = vec![0.0; heads * q_rows * d];
    decode_flash_forward_heads(&q, &k, &v, &mut flash, q_rows, hist, d).expect("flash 多头 decode");

    assert!(testkit::all_finite(&plain));
    assert!(testkit::approx_eq(&flash, &plain, 1e-5));
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
    let q = det_rows(q_rows, d, -0.5, 0.13);
    let k = det_rows(hist, d, 0.4, -0.09);
    let v = det_rows(hist, d, -0.2, 0.11);

    let mut single = vec![0.0; q_rows * d];
    decode_forward(&q, &k, &v, &mut single, q_rows, hist, d).expect("单头朴素");
    let mut multi = vec![0.0; q_rows * d];
    decode_forward_heads(&q, &k, &v, &mut multi, q_rows, hist, d).expect("单头经多头入口");
    assert_eq!(single, multi, "heads=1 必须与单头内核逐元素等价");
}

#[test]
fn multi_head_rejects_non_multiple_buffer_length() {
    let (q_rows, hist, heads, d) = (1, 4, 2, 3);
    let q = det_headed(heads, q_rows, d, -0.5, 0.13);
    let k = det_headed(heads, hist, d, 0.4, -0.09);
    let v = det_headed(heads, hist, d, -0.2, 0.11);
    let mut out = vec![0.0; heads * q_rows * d];
    // q 只给一个头的长度，而 k/v 是两头：头块倍数校验应拦下。
    assert!(decode_forward_heads(&q[..q_rows * d], &k, &v, &mut out, q_rows, hist, d).is_err());
    // q 长度不是 q_rows·d 的整数倍：无法推出整头，直接拒绝。
    let mut ragged = vec![0.0; q.len() - 1];
    assert!(decode_forward_heads(&q[..q.len() - 1], &k, &v, &mut ragged, q_rows, hist, d).is_err());
}
