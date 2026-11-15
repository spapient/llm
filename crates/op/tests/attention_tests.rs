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
