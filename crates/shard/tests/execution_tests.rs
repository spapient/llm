//! `execution` 的形状与回读判据用例。
//!
//! 移出源文件是为了让生产文件留在静态门的行数上限内；`super::*` 仍是 `execution` 那层。

use super::*;

/// 独立标量参考入口；执行断言与表驱动用例共用这一签名。
type ReferenceFn = fn(&Sample) -> diagnostic::Result<Vec<f32>>;

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

/// 走真实 C 编译执行，并额外对拍朴素参考，避免 flash 内核与参考共享同一错误。
fn assert_cpu_matches(op: &str, sample: &Sample, reference: ReferenceFn) {
    let root = std::env::temp_dir().join(format!(
        "cutile-wide-attention-{op}-{}-{}",
        sample.n,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let catalogue = Catalogue::try_builtin().expect("内置算子与后端档案");
    let execution = execute(&catalogue, &Dirs::under(&root), op, "cpu", sample)
        .unwrap_or_else(|error| panic!("{op} D={} 编译执行失败：{error}", sample.n));
    let expected = reference(sample).expect("独立朴素参考");
    let error = cutile::reference::max_rel_err(&execution.output, &expected);
    assert!(
        error <= execution.tolerance,
        "{op} D={} 与朴素参考不符：max_rel_err {error:.3e} > {:.0e}",
        sample.n,
        execution.tolerance
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn flash_attention_supports_head_widths_beyond_the_accumulator_block() {
    for d in [1, 64, 65, 128] {
        let sample = nn::conv::attention::sample(2, d);
        assert_cpu_matches(
            "flash_attention_f32",
            &sample,
            nn::conv::attention::reference,
        );
    }
}

#[test]
fn decode_flash_attention_supports_head_widths_beyond_the_accumulator_block() {
    for d in [1, 64, 65, 128] {
        let sample = nn::conv::decode_attention::sample(2, d);
        assert_cpu_matches(
            "decode_flash_attention_f32",
            &sample,
            nn::conv::decode_attention::reference,
        );
    }
}

#[test]
fn spatial_kernels_match_their_default_models() {
    let cases: [(&str, Sample, ReferenceFn); 5] = [
        (
            "conv2d_f32",
            nn::conv::conv2d::sample(16, 3),
            nn::conv::conv2d::reference,
        ),
        (
            "adaptive_pool_f32",
            nn::conv::adaptive_pool::sample(64, 64),
            nn::conv::adaptive_pool::reference,
        ),
        (
            "avg_pool2d_f32",
            nn::conv::avg_pool2d::sample(64, 64),
            nn::conv::avg_pool2d::reference,
        ),
        (
            "max_pool2d_f32",
            nn::conv::max_pool2d::sample(64, 64),
            nn::conv::max_pool2d::reference,
        ),
        (
            "interpolate_f32",
            nn::conv::interpolate::sample(64, 64),
            nn::conv::interpolate::reference,
        ),
    ];
    for (name, sample, reference) in cases {
        assert_cpu_matches(name, &sample, reference);
    }
}

/// 改声明常量后走同一条真实发射流水线；参考由调用方独立计算，不读默认模型。
fn assert_configured_cpu_matches(
    op_name: &str,
    overrides: &[(&str, &str)],
    sample: &Sample,
    expected: &[f32],
) {
    let root = std::env::temp_dir().join(format!(
        "cutile-configured-{op_name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = Dirs::under(root);
    let catalogue = Catalogue::try_builtin().expect("内置算子与后端档案");
    let mut op = catalogue.op(op_name).expect("注册算子").clone();
    for (key, value) in overrides {
        let constant = op
            .consts
            .iter_mut()
            .find(|(name, _)| name == key)
            .unwrap_or_else(|| panic!("{op_name} 没有配置项 {key}"));
        constant.1 = (*value).to_string();
    }
    let profile = catalogue.resolve("cpu").expect("CPU 档案");
    let run = established(&op, profile, &dirs).expect("建立 CPU 执行证据");
    let path = run.negotiate().expect("协商执行路径");
    check_sample(u64::from(run.model.group.max(1)), sample).expect("样本形状");
    let prepared = run
        .prepare(&path, &dirs, sample.m, sample.n)
        .expect("按修改后的常量发射");
    let harness = prepared.harness(&op, sample, 1).expect("执行夹具");
    let tool = crate::toolchain::detect();
    let (outcome, _) = build::execute(profile, tool.as_ref(), &prepared.key, &harness, &dirs)
        .unwrap_or_else(|error| panic!("{op_name} 配置化内核编译执行失败：{error}"));
    check_parity(
        &outcome.bits,
        expected,
        cutile::reference::tolerance(run.model.dtype),
    )
    .unwrap_or_else(|error| panic!("{op_name} 配置化对拍失败：{error}"));
    let _ = std::fs::remove_dir_all(dirs.cache().parent().expect("独立测试根"));
}

fn spatial_sample(input_len: usize, output_len: usize) -> Sample {
    Sample::f32_io(
        output_len,
        output_len,
        input_len,
        output_len,
        vec![("N", output_len as i64)],
    )
}

const RECTANGULAR_POOL: &[(&str, &str)] = &[
    ("H", "3"),
    ("W", "5"),
    ("OH", "2"),
    ("OW", "3"),
    ("C", "2"),
    ("BATCH", "2"),
    ("KH", "2"),
    ("KW", "3"),
    ("SH", "2"),
    ("SW", "2"),
    ("PH", "1"),
    ("PW", "1"),
];

#[test]
fn configured_conv2d_supports_rectangular_strided_batches() {
    let mut sample = spatial_sample(2 * 3 * 5 * 2, 2 * 2 * 3 * 2);
    sample.n = 2;
    sample.scalars = vec![("N", 2)];
    sample.bufs.push(prof::model::Buf {
        name: "weight",
        values: prof::model::Values::F32(
            (0..2 * 3 * 2 * 2)
                .map(|idx| 0.05 + (idx % 7) as f32 * 0.03)
                .collect(),
        ),
    });
    sample.bufs.push(prof::model::Buf {
        name: "bias",
        values: prof::model::Values::F32(vec![0.4; 2]),
    });
    let geom = nn::conv::ConvGeom {
        kh: 2,
        kw: 3,
        sh: 2,
        sw: 2,
        ph: 1,
        pw: 1,
    };
    let mut expected = vec![0.0; 24];
    nn::conv::Conv2d::new(2, 2, geom)
        .forward_f32(
            sample.f32("x").expect("输入"),
            sample.f32("weight").expect("权重"),
            sample.f32("bias").expect("偏置"),
            2,
            (3, 5),
            &mut expected,
        )
        .expect("通用卷积参考");
    const CONFIG: &[(&str, &str)] = &[
        ("H", "3"),
        ("W", "5"),
        ("CIN", "2"),
        ("COUT", "2"),
        ("KH", "2"),
        ("KW", "3"),
        ("SH", "2"),
        ("SW", "2"),
        ("PH", "1"),
        ("PW", "1"),
    ];
    assert_configured_cpu_matches("conv2d_f32", CONFIG, &sample, &expected);
}

#[test]
fn configured_adaptive_pool_supports_uneven_nhwc_windows() {
    let sample = spatial_sample(60, 24);
    let mut expected = vec![0.0; 24];
    nn::conv::AdaptiveAvgPool2d { oh: 2, ow: 3 }
        .forward(sample.f32("x").expect("输入"), 2, 3, 5, 2, &mut expected)
        .expect("通用自适应池化参考");
    assert_configured_cpu_matches("adaptive_pool_f32", RECTANGULAR_POOL, &sample, &expected);
}

#[test]
fn configured_pool2d_supports_padding_strides_and_both_reductions() {
    let sample = spatial_sample(60, 24);
    let geom = nn::conv::ConvGeom {
        kh: 2,
        kw: 3,
        sh: 2,
        sw: 2,
        ph: 1,
        pw: 1,
    };
    let mut expected = vec![0.0; 24];
    nn::conv::AvgPool2d {
        geom,
        count_include_pad: true,
    }
    .forward(sample.f32("x").expect("输入"), 2, 3, 5, 2, &mut expected)
    .expect("通用平均池化参考");
    let mut average = RECTANGULAR_POOL.to_vec();
    average.push(("INCLUDE_PAD", "1"));
    assert_configured_cpu_matches("avg_pool2d_f32", &average, &sample, &expected);

    nn::conv::MaxPool2d { geom }
        .forward(sample.f32("x").expect("输入"), 2, 3, 5, 2, &mut expected)
        .expect("通用最大池化参考");
    assert_configured_cpu_matches("max_pool2d_f32", RECTANGULAR_POOL, &sample, &expected);
}

#[test]
fn configured_interpolate_supports_single_pixel_axes_and_nhwc() {
    const CONFIG: &[(&str, &str)] = &[
        ("H", "1"),
        ("W", "3"),
        ("OH", "3"),
        ("OW", "5"),
        ("C", "2"),
        ("BATCH", "2"),
    ];
    let sample = spatial_sample(12, 60);
    let mut expected = vec![0.0; 60];
    nn::conv::Interpolate {
        sampler: nn::conv::Sampler::Linear,
        oh: 3,
        ow: 5,
    }
    .forward(sample.f32("x").expect("输入"), 2, 1, 3, 2, &mut expected)
    .expect("通用插值参考");
    assert_configured_cpu_matches("interpolate_f32", CONFIG, &sample, &expected);
}
