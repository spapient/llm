//! `execution` 的形状与回读判据用例。
//!
//! 移出源文件是为了让生产文件留在静态门的行数上限内；`super::*` 仍是 `execution` 那层。

use super::*;

fn sample(m: usize, n: usize) -> Sample {
    // `check_sample` 只读 m/n：空缓冲与空标量足以驱动形状判据，不引入任何硬件或夹具。
    Sample {
        m,
        n,
        scalars: Vec::new(),
        bufs: Vec::new(),
    }
}

#[test]
fn check_sample_accepts_a_shape_on_the_group_grid() {
    check_sample(4, &sample(8, 5)).expect("8 是 group 4 的正整数倍");
    check_sample(1, &sample(1, 1)).expect("group=1 时任意正 m 都在网格上");
}

#[test]
fn check_sample_rejects_zero_or_off_grid_shapes() {
    for bad in [sample(0, 5), sample(5, 0), sample(7, 5)] {
        assert_eq!(
            check_sample(4, &bad).unwrap_err().code(),
            Code::Invalid,
            "形状 {}x{} 必须被拒",
            bad.m,
            bad.n
        );
    }
}

#[test]
fn check_parity_rejects_length_mismatch_and_drift() {
    assert_eq!(
        check_parity(&[1.0], &[1.0, 2.0], 0.1).unwrap_err().code(),
        Code::Invalid,
        "长度不一致先于误差判定"
    );
    assert_eq!(
        check_parity(&[1.0, 2.0], &[1.0, 3.0], 0.1)
            .unwrap_err()
            .code(),
        Code::Invalid,
        "相对误差超容差"
    );
}

#[test]
fn check_parity_reports_the_error_within_tolerance() {
    assert_eq!(check_parity(&[1.0], &[1.0], 0.1).expect("相等"), 0.0);
    // 样本取 1/16 网格上的数：`1.0625f32` 与 `1.0f32` 的差在 f32 里精确等于 0.0625，
    // 于是"返回的是实测误差而不是布尔"可以用 `==` 判。写成 `1.05` 会让 `|1.05f32 - 1.0|`
    // 与十进制 0.05 差 4.8e-8，任何 1e-9 级断言都会假失败——那是样本选错，不是实现有漂移。
    let err = check_parity(&[1.0625], &[1.0], 0.1).expect("6.25% 误差在 0.1 容差内");
    assert_eq!(err, 0.0625, "返回实测误差而不是只给布尔");
}
