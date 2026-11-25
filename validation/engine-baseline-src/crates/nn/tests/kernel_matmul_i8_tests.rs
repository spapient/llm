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
    let (expected, actual) = run_both::<i8, _, _>(
        &a,
        &b,
        m,
        n,
        k,
        forward_scalar,
        forward_tiled,
    );
    assert_eq!(actual, expected, "i8 tile 必须与标量参考逐位一致");
}

// 每个表项仍是独立 `#[test]`；形状表与 f32 档共用，本行只选 i8 的逐位判据。
tiled_shape_cases!(assert_tiled_matches);

#[test]
fn const_generic_tiles_cover_ragged_regions() {
    let (m, n, k) = (7, 19, 33);
    let a = sample(m * k, 7);
    let b = sample(k * n, 11);
    let mut expected = vec![0i32; m * n];
    matmul_scalar(&a, &b, m, n, k, &mut expected);
    macro_rules! check {
        ($rows:literal, $cols:literal) => {
            let mut actual = vec![0i32; m * n];
            blocked::<$rows, $cols>(&a, &b, m, n, k, &mut actual);
            assert_eq!(actual, expected, "瓦片 {}×{}", $rows, $cols);
        };
    }
    check!(1, 1);
    check!(2, 3);
    check!(4, 16);
    check!(8, 32);
}

#[test]
fn empty_contraction_yields_zeros() {
    let mut out = vec![1i32; 4];
    forward_tiled(
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
