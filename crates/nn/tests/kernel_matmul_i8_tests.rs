/// 确定性 i8 样本：覆盖正负与零，(len, salt) 固定即序列固定。
fn sample(len: usize, salt: i32) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let x = (i as i32).wrapping_mul(31).wrapping_add(salt);
            (x % 64 - 32) as i8
        })
        .collect()
}

fn assert_tiled_matches(m: usize, n: usize, k: usize) {
    let a = sample(m * k, 7);
    let b = sample(k * n, 11);
    let (expected, actual) = run_both::<i8>(&a, &b, m, n, k);
    assert_eq!(actual, expected, "i8 tile 必须与标量参考逐位一致");
}

#[test]
fn tiled_matches_scalar_on_standard_shape() {
    assert_tiled_matches(256, 256, 4096);
}

#[test]
fn tiled_matches_scalar_on_tile_multiples() {
    assert_tiled_matches(128, 64, 96);
}

#[test]
fn tiled_matches_scalar_on_ragged_edges() {
    assert_tiled_matches(100, 70, 4093);
}

#[test]
fn tiled_matches_scalar_on_trivial_shapes() {
    assert_tiled_matches(1, 1, 1);
    assert_tiled_matches(1, 8, 3);
    assert_tiled_matches(8, 1, 3);
}

#[test]
fn empty_contraction_yields_zeros() {
    let mut out = vec![1i32; 4];
    forward_tiled::<i8>(
        tensor::View::matrix(2, 0, &[]).expect("空收缩 a"),
        tensor::View::matrix(0, 2, &[]).expect("空收缩 b"),
        &mut out,
    )
    .expect("k=0 合法");
    assert_eq!(out, vec![0i32; 4], "空收缩轴输出必须全零");
}

#[test]
fn forward_rejects_mismatched_bufs() {
    let a = vec![0i8; 8];
    let b = vec![0i8; 8];
    let mut out = vec![0i32; 8];
    let result = (|| {
        let av = tensor::View::matrix(3, 3, &a)?;
        let bv = tensor::View::matrix(3, 3, &b)?;
        forward_scalar(av, bv, &mut out)
    })();
    assert!(result.is_err(), "尺寸不符必须报错");
}
