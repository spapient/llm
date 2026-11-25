use super::*;
use crate::fixtures::{catalogue, profile};
use cutile::builtin::Catalogue;

#[test]
fn shape_validation_matches_the_operator_constraints() {
    let group = u64::from(model_for("gemv_i4").expect("有模型").group);
    assert_eq!(parse_shape("4096x4096", group).unwrap(), (4096, 4096));
    assert!(parse_shape("4096", group).is_err());
    assert!(parse_shape("1000x8", group).is_err());
    assert!(parse_shape("0x8", group).is_err());
}

#[test]
fn the_shape_matrix_follows_the_operator_model() {
    let model = model_for("gemv_i4").expect("有模型");
    assert_eq!(verify_shapes(&model), vec![(32, 1), (64, 3), (4096, 4096)]);
    // 换一个生产形状，矩阵跟着变：审计里没有第二个方阵。
    let other = nn::model::Model {
        shape: (128, 64),
        ..model
    };
    assert_eq!(verify_shapes(&other), vec![(32, 1), (64, 3), (128, 64)]);
}

#[test]
fn the_model_group_matches_the_declaration() {
    // 声明里的 `needs.quant.group` 是能力请求，模型里的 `group` 是样本与代价的取值；
    // 两者漂移会让 verify 在一组原语支撑不了的组大小上"通过"。
    let catalogue = catalogue();
    let declared = catalogue
        .op("gemv_i4")
        .expect("内置算子")
        .needs
        .iter()
        .filter(|need| need.prim == prof::atom::Prim::Quant)
        .flat_map(|need| need.attrs.iter())
        .find(|attr| attr.key() == prof::atom::AttrKey::Group)
        .and_then(|attr| attr.values().first())
        .expect("声明里有 quant.group")
        .parse::<u32>()
        .expect("组大小是整数");
    assert_eq!(declared, model_for("gemv_i4").expect("有模型").group);
}

// ---- execution_tests.rs ----

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
    let catalogue = catalogue();
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

/// 注册声明 → 能力协商 → Program IR → C float4 → 编译执行的完整生产链。
#[test]
fn registered_f32x4_kernel_executes_and_matches_scalar_lanes() {
    let root = testkit::ScratchDir::new("shard-f32x4-pipeline");
    let catalogue = catalogue();
    let model = nn::registry::model("add_f32x4").expect("向量算子已注册");
    let sample = (model.sample)(8, 1);
    assert!(
        sample
            .bufs
            .iter()
            .all(|buf| matches!(buf.values, nn::model::Values::F32x4(_)))
    );
    let expected: Vec<f32> = sample
        .f32("a")
        .expect("向量输入 a")
        .iter()
        .zip(sample.f32("b").expect("向量输入 b"))
        .map(|(left, right)| left + right)
        .collect();
    let execution = execute(
        &catalogue,
        &Dirs::under(root.path()),
        "add_f32x4",
        "cpu",
        &sample,
    )
    .expect("四车道生产链编译执行");
    assert_eq!(execution.output, expected);
    assert_eq!(execution.expected, expected);
    assert_eq!(execution.output.len(), 8);
    assert_eq!(execution.max_rel_err, 0.0);

    let invalid = (model.sample)(6, 1);
    let error = execute(
        &catalogue,
        &Dirs::under(root.path()),
        "add_f32x4",
        "cpu",
        &invalid,
    )
    .expect_err("非标量整组输入必须在编译前拒绝");
    assert_eq!(error.code(), Code::Invalid);
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
    let catalogue = catalogue();
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

// ---- gate_tests.rs ----

use crate::toolchain;
use cutile::prof::Emit;

/// OpenCL 前端表行：拒绝用例要拿它的参数过一遍 OpenCL C 不支持的构造。
fn opencl() -> &'static LanguageFront {
    config()
        .gates
        .iter()
        .find(|entry| entry.emit == Emit::Opencl)
        .expect("OpenCL 语法前端已登记")
}

#[test]
fn a_witness_its_own_language_rejects_is_a_defect() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let Some(toolchain) = toolchain::detect() else {
        return;
    };
    if clang().is_none() {
        eprintln!("skip: 未探测到 clang（`-x cl` 语法检查需要它），本用例未运行");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutile-language-gate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("临时目录");
    let source = dir.join("witness.cl");
    // 函数指针是 OpenCL C 明确排除的构造（标准与两个 clang 前端都拒绝）；`double` 依赖
    // FP64 扩展与前端默认配置，不同构建可能收下，不能当稳定的拒绝夹具。同一份源码在 host
    // C99 里合法，下面先跑 host 编译器把这个前提钉住。
    std::fs::write(
        &source,
        "int add(int a, int b) { return a + b; }\nint witness_fnptr(int (*fp)(int, int)) { return fp(1, 2); }\n",
    )
    .expect("写见证");

    let gate = opencl();
    let arguments: Vec<&str> = gate.check.iter().map(String::as_str).collect();
    let rejected = parse_one(gate, &arguments, &source).expect_err("OpenCL C 不允许函数指针");
    assert!(rejected.to_string().contains("witness.cl"), "{rejected}");
    assert!(rejected.to_string().contains("-x cl"), "{rejected}");

    let host = dir.join("witness.c");
    std::fs::copy(&source, &host).expect("复制");
    let status = std::process::Command::new(&toolchain.command)
        .args(["-std=c99", "-fsyntax-only"])
        .arg(&host)
        .status()
        .expect("跑 host C 编译器");
    assert!(status.success(), "前提：同一份源码在 host C 里合法");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 昇腾前端的三条判据缺一条，解析的就不是同一门语言：`-xcce` 决定设备语言、`dav-c220` 决定
/// 核芯（`id = "910b"`）、`-fsyntax-only` 决定只解析不产出。
#[test]
fn the_ascend_front_is_ccec_for_the_910b_core() {
    let entry = config()
        .gates
        .iter()
        .find(|entry| entry.emit == Emit::Ascendc)
        .expect("AscendC 语法前端已登记");
    assert_eq!(
        entry.tool,
        vec!["ccec".to_string()],
        "昇腾的前端是 CANN 的 ccec"
    );
    for flag in ["-xcce", "--cce-aicore-algorithm=dav-c220", "-fsyntax-only"] {
        assert!(
            entry.check.iter().any(|item| item.as_str() == flag),
            "缺 `{flag}`：{:?}",
            entry.check
        );
    }
}

/// 前端是档案声明的那一个，不是宿主 C 编译器：本机没有 `ccec` 时门只能是 `Absent`（且指出缺的
/// 是哪个程序），拿 `cc` 去解析只会把「没装 CANN」报成一条 C 语法错误——那是 `Rejected`。
#[test]
fn a_missing_declared_front_is_absent_not_rejected() {
    if toolchain::probe("ccec").is_some() {
        eprintln!("skip: 本机装了 `ccec`，缺席分支无从构造");
        return;
    }
    let ascend = config()
        .gates
        .iter()
        .find(|entry| entry.emit == Emit::Ascendc)
        .expect("AscendC 语法前端已登记");
    assert!(front(ascend).is_none(), "`ccec` 不在本机时解析器就不在场");
    let profile = profile("ascend");
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-gate-ascend"));
    match language_gate(&profile, &dirs) {
        LanguageGate::Absent { reason } => {
            assert!(reason.contains("ccec"), "缺席要点名缺哪个程序：{reason}");
            assert!(reason.contains("不在本机"), "{reason}");
        }
        other => panic!("缺 `ccec` 时门不该是 `{other}`"),
    }
}

// ---- 运行时配置校验 ----

use crate::config::RuntimeConfig;
use std::time::Duration;

#[test]
fn defaults_are_valid_and_cover_every_required_front() {
    let config = RuntimeConfig::default();
    config.validate().expect("默认配置必须合法");
    let languages: Vec<_> = config.gates.iter().map(|gate| gate.emit).collect();
    assert_eq!(
        languages,
        [
            Emit::Cuda,
            Emit::Opencl,
            Emit::C,
            Emit::Metal,
            Emit::Ascendc
        ]
    );
}

#[test]
fn a_front_constructor_accepts_owned_and_borrowed_strings() {
    let borrowed = LanguageFront::new(Emit::C, ["clang"], ["-fsyntax-only"]);
    let owned = LanguageFront::new(
        Emit::C,
        [String::from("clang")],
        [String::from("-fsyntax-only")],
    );
    assert_eq!(borrowed, owned);
}

#[test]
fn malformed_fronts_are_rejected_before_they_can_idx_an_empty_command() {
    for gate in [
        LanguageFront::new(Emit::C, [] as [&str; 0], ["-fsyntax-only"]),
        LanguageFront::new(Emit::C, ["  "], ["-fsyntax-only"]),
        LanguageFront::new(Emit::C, ["clang"], [] as [&str; 0]),
    ] {
        let config = RuntimeConfig {
            gates: vec![gate],
            ..RuntimeConfig::default()
        };
        assert_eq!(config.validate().unwrap_err().code(), Code::Invalid);
    }
}

#[test]
fn duplicate_languages_are_rejected_instead_of_using_the_first_row_silently() {
    let mut config = RuntimeConfig::default();
    let duplicate = config.gates[0].clone();
    config.gates.push(duplicate);
    assert!(config.validate().unwrap_err().to_string().contains("重复"));
}

#[test]
fn every_invalid_threshold_and_zero_lock_wait_is_rejected() {
    for threshold in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
        let config = RuntimeConfig {
            l3_threshold: threshold,
            ..RuntimeConfig::default()
        };
        assert_eq!(config.validate().unwrap_err().code(), Code::Invalid);
    }
    let config = RuntimeConfig {
        lock_wait: Duration::ZERO,
        ..RuntimeConfig::default()
    };
    assert_eq!(config.validate().unwrap_err().code(), Code::Invalid);
}

// ---- fragments_tests.rs ----

use cutile::{decl, emit, substitute};

/// 一台设备全部算子 × 分块组合的最终源文件：`{算子}_{序号}` → 文本。
///
/// 走的是执行层同一条路（`substitute::render` 拼片段层 → `emit::assemble` 补 preamble），
/// 所以这里解析的文本就是最终落盘的那份。协商判据也与生产链路同源：设备接不住的档
/// （如 CPU 前端不具备的 f16/bf16）不发射到它的语法前端，只对协商可行的算子做收下校验。
fn units_of(catalogue: &Catalogue, profile: &Profile, dirs: &Dirs) -> Vec<(String, String)> {
    let mut units = Vec::new();
    for op in catalogue.ops() {
        let specialized = prof::catalogue::for_device(op, &profile.meta.vendor).expect("后端模板");
        if prof::sign::is_native_template(&specialized.template, &op.name) {
            eprintln!(
                "native: `{}/{}` 由目标工具链专项验收",
                profile.meta.id, op.name
            );
            continue;
        }
        let Ok(path) = cutile::cap::negotiate(op, profile) else {
            continue;
        };
        let combos = decl::expand(op).expect("tile 候选");
        for (idx, combo) in combos.iter().enumerate() {
            match crate::build::source(op, profile, &path, combo, dirs) {
                Ok((source, _)) => units.push((format!("{}_{idx}", op.name), source.text)),
                Err(error) if error.to_string().contains("no registered native float4") => {
                    eprintln!(
                        "unsupported: `{}/{}` 尚未登记 float4 lowering",
                        profile.meta.id, op.name
                    );
                }
                Err(error) => panic!("{} 的真实生产发射失败：{error}", op.name),
            }
        }
    }
    units
}

/// 片段层只有 `witness.*` 过门，正文从来没被任何编译器看过——声明里写 `barrier`、正文是一句
/// “留给 P3”，门一样绿。这条用例把判据钉在正文上：片段拼进翻译单元之后，这台设备**自己声明的
/// 那门语言**必须收下它，否则片段就是占位。
#[test]
fn every_fragment_composes_into_a_translation_unit_its_language_accepts() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let catalogue = catalogue();
    let dir = std::env::temp_dir().join(format!("cutile-fragments-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("临时目录");
    let mut languages = 0;
    let mut units = 0;
    for profile in catalogue.profiles() {
        let Some(spec) = config()
            .gates
            .iter()
            .find(|entry| entry.emit == profile.meta.emit)
        else {
            eprintln!(
                "skip: `{}` 的 `{}` 没有登记语法前端，它的片段未被解析",
                profile.meta.id, profile.meta.emit
            );
            continue;
        };
        if front(spec).is_none() {
            eprintln!(
                "skip: 本机没有 `{}`，`{}` 的片段未被解析",
                spec.tool.join(" "),
                profile.meta.id
            );
            continue;
        }
        let arguments = args_for(spec, profile);
        let dirs = Dirs::under(dir.join(format!("{}-build", profile.meta.id)));
        for (name, text) in units_of(&catalogue, profile, &dirs) {
            let path = dir.join(format!(
                "{}-{name}.{}",
                profile.meta.id,
                profile.meta.emit.ext()
            ));
            std::fs::write(&path, &text).expect("写翻译单元");
            if let Err(error) = parse_one(spec, &arguments, &path) {
                let _ = std::fs::remove_dir_all(&dir);
                panic!(
                    "`{}` 的 `{}` 收不下 `{name}`：{error}",
                    profile.meta.id, spec.emit
                );
            }
            units += 1;
        }
        languages += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(languages > 0, "没有任何设备走到解析：这条门等于没开");
    eprintln!("{languages} 门语言 / {units} 份翻译单元");
}

/// 反向验证：正文改坏之后判据必须变红，否则上一条的 green 不说明门在看正文。
#[test]
fn a_fragment_body_the_language_rejects_is_a_defect() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let catalogue = catalogue();
    let profile = catalogue.profile("intel").expect("intel 档案");
    let spec = config()
        .gates
        .iter()
        .find(|entry| entry.emit == profile.meta.emit)
        .expect("OpenCL 语法前端已登记");
    if front(spec).is_none() {
        eprintln!("skip: 本机没有 `clang`");
        return;
    }
    let arguments = args_for(spec, profile);
    let honest =
        prof::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext()).expect("片段层");
    // `gemv_f32` 用到 `bar`：改坏它，拼进去的翻译单元就该被拒。
    let op = catalogue
        .ops()
        .iter()
        .find(|op| op.name == "gemv_f32")
        .expect("gemv_f32 已声明");
    let combo = decl::expand(op)
        .expect("tile 候选")
        .into_iter()
        .next()
        .expect("至少一组分块");
    let variant = op.variants.first().expect("降级链至少一档");
    let dir = std::env::temp_dir().join(format!("cutile-fragments-broken-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("临时目录");
    let parse = |fragments: &prof::FragmentSet, tag: &str| {
        let body = substitute::render(op, &combo, fragments).expect("片段拼得进去");
        let source = emit::assemble(profile, op, variant, &combo, &body);
        let path = dir.join(format!("{tag}.cl"));
        std::fs::write(&path, &source.text).expect("写翻译单元");
        parse_one(spec, &arguments, &path)
    };
    parse(&honest, "honest").expect("原样的片段必须过门");
    let broken = corrupted(&honest, prof::Prim::Dot);
    let rejected = parse(&broken, "broken").expect_err("改坏的正文必须被语言拒绝");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(rejected.to_string().contains("broken.cl"), "{rejected}");
}

#[cfg(windows)]
#[test]
fn every_negotiable_intel_translation_unit_builds_on_uhd750() {
    let device = prof::runtime::opencl::OpenClDevice::new().expect("必须探测到 UHD 750");
    let catalogue = catalogue();
    let profile = catalogue.profile("intel").expect("intel 档案");
    let dirs = Dirs::under(std::env::temp_dir().join(format!(
        "cutile-intel-build-{}",
        std::process::id()
    )));
    let units = units_of(&catalogue, profile, &dirs);
    assert!(!units.is_empty(), "UHD 750 至少要有可达的翻译单元");
    for (name, text) in &units {
        let source = format!("{text}\n__kernel void cutile_syntax_probe() {{}}\n");
        device
            .compile(&source, &profile.flags.join(" "), "cutile_syntax_probe")
            .unwrap_or_else(|error| panic!("UHD 750 拒绝 `{name}`：{error}"));
    }
    eprintln!("UHD 750 实际编译 {} 份翻译单元", units.len());
}

/// 复制一份片段层，把 `target` 的正文换成语言收不下的文本（声明行不动，拼进去的仍是一段片段）。
fn corrupted(base: &prof::FragmentSet, target: prof::Prim) -> prof::FragmentSet {
    let mut set = prof::FragmentSet::new(base.vendor(), base.ext());
    for prim in prof::Prim::ALL {
        for fragment in base.fragments_of(*prim) {
            let source = fragment.source();
            let text = if *prim == target {
                format!("{source}\nint cutile_broken_body = ;\n")
            } else {
                source.to_string()
            };
            set.insert(*prim, text).expect("签名声明没动，登记照旧");
        }
    }
    set
}

#[test]
fn autotune_is_explicitly_opt_in() {
    let config = RuntimeConfig::default();
    assert!(!config.autotune, "默认只走确定性启发式");
    let enabled = RuntimeConfig {
        autotune: true,
        ..config
    };
    assert!(enabled.validate().is_ok(), "显式开启不应改变其他配置约束");
}
