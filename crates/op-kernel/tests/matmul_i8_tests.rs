use super::*;

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
    let mut expected = vec![0i32; m * n];
    let mut actual = vec![0i32; m * n];
    forward_scalar(&a, &b, m, n, k, &mut expected).expect("参考可计算");
    forward_tiled(&a, &b, m, n, k, &mut actual).expect("tile 可计算");
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
    forward_tiled(&[], &[], 2, 2, 0, &mut out).expect("k=0 合法");
    assert_eq!(out, vec![0i32; 4], "空收缩轴输出必须全零");
}

#[test]
fn forward_rejects_mismatched_buffers() {
    let a = vec![0i8; 8];
    let b = vec![0i8; 8];
    let mut out = vec![0i32; 8];
    assert!(
        forward_scalar(&a, &b, 3, 3, 3, &mut out).is_err(),
        "尺寸不符必须报错"
    );
}
