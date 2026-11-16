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
