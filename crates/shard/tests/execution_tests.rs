//! `execution` 的派发几何与回读判据用例：目录取内置声明、样本取该声明自己的语义表，
//! 于是「网格怎么来」只由声明与样本决定，测试不另造形状。
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

/// 内置目录里的一条真实声明与它自己的样本：派发几何只由声明与样本推出，测试不另造形状。
fn decl(name: &str) -> (&'static OpDecl, Sample) {
    // 目录是进程级只读数据：解析一次就够了，每条用例各重建一遍只是白花时间。
    static CATALOGUE: std::sync::OnceLock<Catalogue> = std::sync::OnceLock::new();
    let catalogue = CATALOGUE.get_or_init(|| Catalogue::try_builtin().expect("内置数据可解析"));
    let op = catalogue.op(name).expect("目录里有这条声明");
    let model = nn::registry::model(name).expect("有语义表");
    let sample = (model.sample)(8, 4);
    (op, sample)
}

#[test]
fn a_pointwise_launch_is_one_item_per_element_in_thread_blocks() {
    let (op, sample) = decl("scale_f32");
    let launch = launch(op, &sample).expect("逐点可派发");
    assert_eq!(launch.kind, Lower::Pointwise);
    // 逐点：工作项 = 长度轴上的元素数（签名里唯一的标量），成组发射。
    let len = sample.scalar("N").expect("样本带 `N`");
    assert_eq!(launch.items, [usize::try_from(len).unwrap(), 1, 1]);
    assert_eq!(launch.group, POINT_THREADS);
}

#[test]
fn a_chunked_family_is_one_item_per_block_and_never_grouped() {
    let (op, sample) = decl("rmsnorm_f32");
    let launch = launch(op, &sample).expect("分块可派发");
    let Lower::ChunkRmsnorm { chunk } = launch.kind else {
        panic!("rmsnorm 应降成分块归一化，实为 {:?}", launch.kind);
    };
    // 块数是长度轴的向上取整：末块不满也算一块，否则尾部元素没人算。
    assert_eq!(launch.items, [sample.m.div_ceil(chunk), 1, 1]);
    // 一块一线程：组内再塞线程会让同一块被多算几遍。
    assert_eq!(launch.group, 1);
}

#[test]
fn a_declared_grid_becomes_the_item_count_axis_by_axis() {
    let (op, sample) = decl("gemv_i8");
    let launch = launch(op, &sample).expect("网格可派发");
    let Lower::Loop(grid) = &launch.kind else {
        panic!("gemv 应降成网格，实为 {:?}", launch.kind);
    };
    // gemv 的网格是 `dims = ["n:N"]`（一维、按输出行拆线程）：工作项数就是该标量的取值，
    // 其余维补 1。"按行拆"因此不是每条线程算整遍，而是每行一条线程。
    assert_eq!(grid.dims.len(), 1, "gemv 是一维网格");
    let rows = sample.scalar("N").expect("样本带 `N`");
    assert_eq!(launch.items, [usize::try_from(rows).unwrap(), 1, 1]);
    assert_eq!(launch.group, 1);
}

#[test]
fn a_serial_fallback_launches_exactly_one_item() {
    // 零维网格是「原样跑一遍」：不给 0（网格为零既不执行也不报错），只给一格。
    assert_eq!(pad(&[]), [1, 1, 1]);
    assert_eq!(along(0), [1, 1, 1]);
    assert_eq!(along(5), [5, 1, 1]);
    assert_eq!(pad(&[4, 2]), [4, 2, 1]);
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
