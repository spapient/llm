//! 档案层的单元测试：语料由 `build.rs` 在构建期生成，测试只改写生成的那棵树。
//!
//! 与 `prof.rs` 分开放，是为了让生产文件只含生产代码——两边的行数各自可读，也各自受门禁约束。

use super::schema::eval_expression;
use super::*;

/// 内置 CPU 档案：`devices/cpu.toml` 在构建期生成的树，运行期没有 TOML 文本。
fn cpu() -> &'static Table {
    crate::embedded("cpu").expect("内置 cpu 档案")
}

/// 在档案末尾追加一条供给：测试只写被考察的字段，其余按语料给全。
fn with_provide(table: &Table, pairs: Vec<(&str, Value)>) -> Table {
    let mut table = table.clone();
    table.push("provides", Value::table(pairs));
    table
}

/// 覆盖档案里的一个路径，其余按语料给全：测试只写被考察的那个字段。
fn edited(table: &Table, path: &str, value: Value) -> Table {
    let mut table = table.clone();
    table.set(path, value);
    table
}

/// 追加一条"只有文本字段"的供给：字段按 `键, 值, 键, 值…` 平铺。
fn provide(fields: &[&str]) -> Table {
    let pairs = fields.chunks(2).map(|pair| (pair[0], Value::text(pair[1])));
    with_provide(cpu(), pairs.collect())
}

/// `(prim, impl)` 那条供给在 `provides` 里的位置：按签名定位，不依赖语料顺序。
fn position(table: &Table, prim: &str, implementation: &str) -> usize {
    let provides = table.array("provides").expect("档案必须有 `provides`");
    let matches = |entry: &Value| match entry {
        Value::Table(entry) => {
            entry.string("prim") == Some(prim) && entry.string("impl") == Some(implementation)
        }
        _ => false,
    };
    provides.iter().position(matches).expect("供给存在")
}

/// 见证校验的临时目录：落地即建，`TempDir` 离开作用域时自动清理。
struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("prof-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("建临时目录");
    TempDir(dir)
}

#[test]
fn parses_builtin_cpu_profile() {
    let profile = Profile::parse(cpu()).expect("内置 CPU 档案必须可解析");
    assert_eq!(profile.meta.vendor, "cpu");
    assert_eq!(profile.meta.emit, Emit::C);
    assert_eq!(profile.physical_space(Space::Fast), Some("L2"));
    assert_eq!(profile.simt.group, 1);
    assert!(profile.providers(Prim::Dot).next().is_some());
}

/// 同一个签名字符串在两层里必须接受同一套写法。
///
/// `[needs]` 认 `m` / `bytes` / `extent` 三个名字，也认标量形式的 `dt` / `space`；档案层
/// 若只认 `m` 与数组，那么同一份签名换个合法写法就会**静默少一个约束**——`space` 甚至
/// 没有后续校验能把它捞回来（空集在协商里表现为"缺空间"，离原因很远）。
#[test]
fn a_provide_accepts_the_same_signature_spellings_as_a_need() {
    let table = provide(&[
        "prim",
        "load",
        "impl",
        "scalar_copy_alias",
        "bytes",
        "<=4096",
        "dt",
        "f32",
        "space",
        "FAST",
        "src",
        "GM",
        "dst",
        "FAST",
        "grade",
        "L0",
    ]);
    let profile = Profile::parse(&table).expect("档案层必须接受与 `[needs]` 相同的写法");
    let provide = profile.provides.last().expect("新加的供给");
    assert_eq!(provide.prim, Prim::Load);
    assert_eq!(
        provide.extent,
        Span::at_most(4096),
        "`bytes` 是主维度的别名"
    );
    assert!(provide.dt.contains(DType::F32), "标量 `dt` 不能被丢掉");
    assert!(
        provide.space.contains(Space::Fast),
        "标量 `space` 不能被静默跳过"
    );
    let value = |key: AttrKey| {
        provide
            .attrs
            .iter()
            .find(|attr| attr.key() == key)
            .and_then(|attr| attr.values().first())
            .map(String::as_str)
    };
    assert_eq!(value(AttrKey::Src), Some("GM"));
    assert_eq!(value(AttrKey::Dst), Some("FAST"));
}

/// 看不懂的写法要报错，不能当作"没写"。
#[test]
fn a_signature_field_with_the_wrong_shape_is_an_error() {
    for (name, value) in [("space", Value::int(3)), ("n", Value::int(16))] {
        let mut table = provide(&["prim", "dot", "impl", "scalar_fma", "grade", "L0"]);
        let appended = table.array("provides").expect("provides").len() - 1;
        table.set(&format!("provides[{appended}].{name}"), value);
        let error = Profile::parse(&table).expect_err(&format!("`{name}` 不能当成没写"));
        assert!(error.to_string().contains(name), "{error}");
    }
}

/// 一个维度只有一个名字：同时写 `m` 与 `bytes` 会让"哪个生效"取决于遍历顺序，
/// 而档案是数据，不该有顺序语义。
#[test]
fn two_names_for_one_dimension_are_an_error() {
    let table = provide(&[
        "prim",
        "load",
        "impl",
        "scalar_copy",
        "m",
        "<=4096",
        "bytes",
        "<=4096",
        "grade",
        "L0",
    ]);
    let error = Profile::parse(&table).expect_err("一个维度只能写一个名字");
    assert!(error.to_string().contains("只能写一个"), "{error}");
}

#[test]
fn builtin_cpu_profile_is_schema_valid() {
    let profile = Profile::parse(cpu()).expect("可解析");
    // 见证文件属于片段层（`crates/<device>`），本 crate 只校验 schema 与 `[units]`；
    // "见证真的存在"由 `cutile` 的 `Catalogue::validate` 负责。
    profile.validate(None).expect("CPU 档案必须自洽");
}

/// 属性键属于原语自己的签名：`sat` 是 `cvt` 的参数，`load` 写上它只会让档案看起来
/// 更具体——协商只比对需求里出现的键，这条永远读不到。
#[test]
fn an_attr_key_must_belong_to_its_primitive() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    let load = profile
        .provides
        .iter_mut()
        .find(|provide| provide.prim == Prim::Load)
        .expect("cpu 有 load 供给");
    load.attrs.push(Attr::new(AttrKey::Sat, ["off"]));
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors.iter().any(|error| matches!(
            error,
            ProfileError::Unknown { field, value, .. }
                if field == "provides[load].sat" && value == "sat"
        )),
        "{errors:?}"
    );
}

#[test]
fn missing_emit_is_rejected() {
    let error = Profile::parse(&edited(cpu(), "meta.emit", Value::text("nope"))).unwrap_err();
    assert!(matches!(error, ProfileError::Unknown { .. }), "{error}");
}

#[test]
fn l2_without_witness_is_rejected() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    for provide in &mut profile.provides {
        provide.witness = None;
    }
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors
            .iter()
            .all(|e| matches!(e, ProfileError::Witness { .. }))
    );
}

/// 协商双证据口径：档案等级达到 L2+ 且形状覆盖才满足；L0/L1 即使形状覆盖也被拒，不许降级成
/// "形状对上就算有"。见证存在性由 [`l2_without_witness_is_rejected`] 与 L3 用例守住。
#[test]
fn negotiate_rejects_unverified_suppliers_even_when_covering() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    let prim = profile.provides[0].prim;
    let need = Need::of(prim);

    profile.provides[0].grade = Grade::L2;
    assert!(profile.satisfies(&need), "L2 且形状覆盖应当满足");

    for grade in [Grade::L0, Grade::L1] {
        profile.provides[0].grade = grade;
        assert!(
            !profile.satisfies(&need),
            "grade {grade} 形状虽覆盖，未验证也不许满足"
        );
    }
}

/// L3 与 L2 同属已验证等级：缺见证同样在校验期报 `Witness`，不许只对 L2 守见证口径。
#[test]
fn l3_without_witness_is_rejected_by_validation() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    for provide in &mut profile.provides {
        provide.grade = Grade::L3;
        provide.witness = None;
    }
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, ProfileError::Witness { .. })),
        "L3 同样必须带见证：{errors:?}"
    );
}

#[test]
fn a_peak_must_equal_its_own_formula() {
    // 108.8e9 是旧档案里的手抄值：与 8*2.6e9*8*2 差了三倍，装载期必须报出来。
    let table = edited(cpu(), "peak.flops_f32.value", Value::real(108.8e9));
    let error = Profile::parse(&table).expect_err("value 与 src 不一致必须是错误");
    assert!(error.to_string().contains("不一致"), "{error}");
}

#[test]
fn a_peak_formula_with_a_unit_name_is_rejected() {
    let formula = "8 cores * 2.6 GHz * 8 lanes * 2 flop";
    let table = edited(cpu(), "peak.flops_f32.src", Value::text(formula));
    let error = Profile::parse(&table).expect_err("带单位的推导式无法核验");
    assert!(error.to_string().contains("求值失败"), "{error}");
}

#[test]
fn formulas_are_four_operations_with_parentheses() {
    assert!((eval_expression("2*3+4").expect("可求值") - 10.0).abs() < 1e-9);
    assert!((eval_expression("(1+2)*3/4").expect("可求值") - 2.25).abs() < 1e-9);
    assert!((eval_expression("2*4096/8*8e9").expect("可求值") - 8.192e12).abs() < 1.0);
    assert!(eval_expression("1/0").is_err());
    assert!(eval_expression("8*").is_err());
    assert!(eval_expression("(1").is_err());
    assert!(eval_expression("2 3").is_err());
}

/// 推导式边界：空串、纯空白、单符号、嵌套括号、重复运算符与越界写法都必须带说明拒绝。
#[test]
fn formulas_reject_saturated_and_degenerate_inputs_with_a_message() {
    let cases: &[&str] = &[
        "", "   ", "+", "(", ")", "**", "1..2", "abc", "1+", "((1+2))", "2/3/4",
    ];
    let expect_ok = ["((1+2))", "2/3/4"];
    for (index, text) in cases.iter().enumerate() {
        match eval_expression(text) {
            Ok(_) => assert!(expect_ok.contains(text), "用例 {index} `{text}` 不应通过"),
            Err(error) => {
                assert_eq!(error.code(), Code::Invalid, "用例 {index}");
                assert!(
                    !error.message().trim().is_empty(),
                    "用例 {index}：拒绝必须有话说"
                );
            }
        }
    }
}

/// 合法推导式：字面量、嵌套括号、左结合加减与乘除优先级链式求值都对得上手算。
#[test]
fn formulas_evaluate_priority_chains_and_parenthesized_groups() {
    let cases: &[(&str, f64)] = &[
        ("0", 0.0),
        ("1.5", 1.5),
        ("2*3+4", 10.0),
        ("2+3*4", 14.0),
        ("(2+3)*4", 20.0),
        ("(1+2)*(3+4)", 21.0),
        ("10-6+2", 6.0),
        ("10-(6+2)", 2.0),
        ("8*4096/8", 4096.0),
    ];
    for (index, (text, expected)) in cases.iter().enumerate() {
        let actual = eval_expression(text).expect("用例 {index} `{text}` 应可求值");
        assert!(
            (actual - expected).abs() < 1e-9,
            "用例 {index} `{text}`：{actual} != {expected}"
        );
    }
}

/// 拒绝用例表驱动：除零、括号不闭合/不成对、运算符续接、连续运算符、非法字符与悬空括号，
/// 每条都必须带非空说明消息；非法字面量也按位置报出。
#[test]
fn formulas_reject_each_degenerate_shape_with_a_message() {
    let cases: &[&str] = &[
        "1/0", "(1+2", "1+2)", "1+2)*3", "8*", "1+2-", "*3", "**2", "1++2", "1--2", "1*/2", "2 3",
        "1a2", "3GHz", ".", "1.2.3", "e5", "1e",
    ];
    for (index, text) in cases.iter().enumerate() {
        let error = eval_expression(text).expect_err("用例 {index} `{text}` 应当拒绝");
        assert_eq!(error.code(), Code::Invalid, "用例 {index} `{text}`");
        assert!(
            !error.message().trim().is_empty(),
            "用例 {index} `{text}`：拒绝必须带说明"
        );
    }
}

/// 饱和字面量：极大指数与相邻大数乘积按 f64 折叠（可溢出成 inf），既不 panic 也不静默截断。
#[test]
fn formulas_handle_saturated_large_literals_without_panic() {
    let cases: &[&str] = &[
        "1e308",
        "1e308*1e308",
        "0.0",
        "0*1e308",
        "1e-308+1e-308",
        "4294967295",
    ];
    for (index, text) in cases.iter().enumerate() {
        let value = eval_expression(text).expect("用例 {index} `{text}` 应可求值");
        assert!(
            value.is_finite() || value.is_infinite(),
            "用例 {index}：不应是 NaN"
        );
    }
}

/// 区间解析四种写法与退化边界：空串、非数字、上界越上、空白两侧、不限区间的往返。
#[test]
fn span_parse_covers_boundary_and_degenerate_forms() {
    let cases: &[(&str, Result<Span, ()>)] = &[
        ("16", Ok(Span::exact(16))),
        ("16..64", Ok(Span::range(16, 64).unwrap())),
        ("<=512", Ok(Span::at_most(512))),
        ("32..", Ok(Span::at_least(32))),
        ("  16  ", Ok(Span::exact(16))),
        ("0", Ok(Span::exact(0))),
        ("4294967295", Ok(Span::exact(u32::MAX))),
        ("64..16", Err(())),
        ("abc", Err(())),
        ("", Err(())),
        ("..", Err(())),
        ("16..4.5", Err(())),
    ];
    for (index, (text, expected)) in cases.iter().enumerate() {
        let parsed = Span::parse(text);
        match expected {
            Ok(span) => assert_eq!(parsed.as_ref(), Ok(span), "用例 {index} `{text}`"),
            Err(()) => assert!(parsed.is_err(), "用例 {index} `{text}` 应当拒绝"),
        }
    }
    let any = Span::any();
    assert!(any.is_any());
    assert!(any.covers(Span::exact(u32::MAX)));
}

/// 闭集解析：空串、大小写漂移、截断词与重复成员都不命中；全量成员名字往返一致。
#[test]
fn closed_set_parse_rejects_boundary_and_foreign_names() {
    let foreign: &[&str] = &["", " ", "MMA", "dot ", "tma_bulk", "f32 ", "FAST ", "L4"];
    for name in foreign {
        assert!(Prim::parse(name).is_none(), "`{name}` 不应命中原语闭集");
        assert!(Space::parse(name).is_none(), "`{name}` 不应命中存储闭集");
        assert!(Grade::parse(name).is_none(), "`{name}` 不应命中等级闭集");
        assert!(
            DType::parse_name(name).is_none(),
            "`{name}` 不应命中类型闭集"
        );
    }
    for prim in Prim::ALL.iter().copied() {
        assert_eq!(Prim::parse(prim.as_str()), Some(prim));
    }
    for space in Space::ALL.iter().copied() {
        assert_eq!(Space::parse(space.as_str()), Some(space));
    }
}

#[test]
fn a_witness_is_file_plus_function_plus_expectation() {
    let profile = Profile::parse(cpu()).expect("可解析");
    let witness = profile
        .providers(Prim::Quant)
        .find_map(|provide| provide.witness.as_ref())
        .expect("quant 的见证");
    assert_eq!(witness.file, "cpu/witness.c");
    assert_eq!(witness.func, "witness_quant");
    assert_eq!(witness.expect, vec![-4.0, 3.5]);
}

#[test]
fn a_witness_path_without_a_function_is_rejected() {
    let quant = position(cpu(), "quant", "unpack");
    let witness = Value::text("cpu/witness.c");
    let table = edited(cpu(), &format!("provides[{quant}].witness"), witness);
    let error = Profile::parse(&table).expect_err("只有路径的见证必须是错误");
    assert!(error.to_string().contains("内联表"), "{error}");
}

#[test]
fn an_empty_expectation_is_rejected() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    profile.provides[0]
        .witness
        .as_mut()
        .expect("见证")
        .expect
        .clear();
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("期望输出")),
        "{errors:?}"
    );
}

#[test]
fn the_witness_function_must_exist_in_the_file() {
    let dir = temp_dir("witness");
    let vendor = dir.0.join("cpu");
    std::fs::create_dir_all(&vendor).expect("建 vendor 目录");
    std::fs::write(
        vendor.join("witness.c"),
        "int witness_dot(float *out) { out[0] = 70.0f; return 1; }\n",
    )
    .expect("写见证");
    let mut profile = Profile::parse(cpu()).expect("可解析");
    profile.provides[0].witness.as_mut().expect("见证").func = "witness_absent".to_string();
    let errors = profile.validate_in(&dir.0).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("witness_absent")),
        "{errors:?}"
    );
}

/// 见证文件不在是 `missing`，不是"档案写错"：两者在脚本里该分开处理。
#[test]
fn a_missing_witness_file_is_missing() {
    let dir = temp_dir("witness-missing");
    let profile = Profile::parse(cpu()).expect("可解析");
    let errors = profile.validate_in(&dir.0).unwrap_err();
    assert!(
        errors.iter().any(|error| error.code() == Code::Missing),
        "见证文件不在应是 `missing`：{errors:?}"
    );
}

/// 见证可以先落成数据：等级还是 L0/L1 时，声明过的文件与函数名同样要校验，
/// 否则命名写错会一直潜伏到升 L2 才暴露。
#[test]
fn a_declared_witness_is_validated_below_l2() {
    let dir = temp_dir("witness-l1");
    let vendor = dir.0.join("nvidia");
    std::fs::create_dir_all(&vendor).expect("建 vendor 目录");
    std::fs::write(
        vendor.join("witness.cu"),
        "int witness_dot(float *out) { out[0] = 70.0f; return 1; }\n",
    )
    .expect("写见证");
    let mut profile =
        Profile::parse(crate::embedded("nvidia").expect("内置 nvidia 档案")).expect("可解析");
    // 只留一条声明，其余见证置空（L1 允许没有见证），这样错误只指向被测的那条。
    for provide in &mut profile.provides {
        if provide.prim != Prim::Dot {
            provide.witness = None;
        }
    }
    let dot = profile
        .provides
        .iter_mut()
        .find(|provide| provide.prim == Prim::Dot)
        .expect("dot 供给");
    assert_eq!(dot.grade, Grade::L1, "前提：这条签名还没到 L2");
    dot.witness.as_mut().expect("见证").func = "witness_absent".to_string();
    let errors = profile.validate_in(&dir.0).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("witness_absent")),
        "{errors:?}"
    );
}

/// 两条签名不能共用一份见证：执行通过的结论必须能归属到具体的那一条。
///
/// 见证证明的是"这条签名写的是什么算法"，共用入口会让"每条 provides 都有见证"退化成
/// "每条都指向同一份证据"——`mmad` 与 `scalar_fma` 正是两种算法。
#[test]
fn a_witness_may_serve_only_one_signature() {
    let mut profile = Profile::parse(ascend()).expect("可解析");
    let quant = profile
        .provides
        .iter()
        .position(|provide| provide.prim == Prim::Quant)
        .expect("quant 供给");
    let mut second = profile.provides[quant].clone();
    second.impl_name = "scalar_dequant_copy".to_string();
    profile.provides.push(second);
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors.iter().any(|error| matches!(
            error,
            ProfileError::Witness { detail, .. } if detail.contains("只能证明一条签名")
        )),
        "{errors:?}"
    );
}

#[test]
fn fingerprint_rules_match_versions_and_sets() {
    assert!(MatchRule::parse("10.0..10.3").matches("10.2"));
    assert!(!MatchRule::parse("10.0..10.3").matches("9.0"));
    assert!(MatchRule::parse("contains: B200|GB200").matches("NVIDIA B200"));
    assert!(!MatchRule::parse("contains: B200|GB200").matches("H100"));
    assert!(MatchRule::parse("x86_64{avx2,avx512}").matches("x86_64+avx2+avx512"));
    assert!(!MatchRule::parse("x86_64{avx2,avx512}").matches("x86_64+avx2"));
    assert!(MatchRule::parse("sm90").matches("sm90"));
    assert!(MatchRule::parse("x86_64|aarch64").matches("aarch64"));
    assert!(!MatchRule::parse("x86_64|aarch64").matches("riscv64"));
}

#[test]
fn profile_matches_detected_fingerprint() {
    let profile = Profile::parse(cpu()).expect("可解析");
    let arch = std::env::consts::ARCH;
    let detected = Detected::new()
        .with("algorithm", arch)
        .with("features", format!("{arch}+avx2"));
    assert!(profile.matches(&detected), "{detected}");
    assert!(!profile.matches(&Detected::new().with("algorithm", "sparc")));
}

#[test]
fn only_verified_provides_satisfy_needs() {
    let profile = Profile::parse(cpu()).expect("可解析");
    let need = Need::of(Prim::Atomic);
    // CPU 档案把 atomic 声明在 L0：未验证视为不具备。
    assert!(!profile.satisfies(&need));
    assert!(profile.providers(Prim::Atomic).next().is_some());
}

/// 子组内交换/广播以"有第二条 lane"为前提：单 lane 设备声明它只会造出自欺的证据。
///
/// `[simt]` 此前完全不进能力比对，warp 宽度契约因此没有校验入口；这条把它变成装载期
/// 判据，而不是档案作者的自觉。
#[test]
fn a_single_lane_device_may_not_declare_a_lane_exchange() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    assert_eq!(profile.simt.group, 1, "前提：CPU 是单 lane");
    let template = profile.provides[0].clone();
    let mut shuffle = template;
    shuffle.prim = Prim::Shuffle;
    shuffle.impl_name = "fake_shuffle".to_string();
    shuffle.dt = DTypeSet::of([DType::F32]);
    profile.provides.push(shuffle);
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("`[simt].group > 1`")),
        "{errors:?}"
    );
}

/// 内置昇腾档案：执行单元与"哪一层只有谁能访问"全在数据里。
fn ascend() -> &'static Table {
    crate::embedded("ascend").expect("内置 ascend 档案")
}

fn ascend_profile() -> Profile {
    let profile = Profile::parse(ascend()).expect("内置 ascend 档案必须可解析");
    profile.validate(None).expect("内置 ascend 档案必须自洽");
    profile
}

fn exec(values: &[&str]) -> Attr {
    Attr::new(AttrKey::Exec, values.iter().copied())
}

#[test]
fn parses_builtin_ascend_profile_with_units() {
    let profile = ascend_profile();
    assert_eq!(profile.meta.vendor, "ascend");
    assert_eq!(profile.meta.emit, Emit::Ascendc);
    assert_eq!(profile.physical_space(Space::Fast), Some("L1"));
    assert_eq!(profile.physical_space(Space::Reg), Some("L0C"));

    let cube = profile
        .units
        .iter()
        .find(|unit| unit.name == "cube")
        .expect("cube 单元");
    assert!(cube.space.contains(Space::Fast));
    assert!(cube.space.contains(Space::Reg));
    assert!(!cube.space.contains(Space::Gm), "Cube 访问不到 GM");

    let dot = profile
        .providers(Prim::Dot)
        .find(|provide| provide.attrs.iter().any(|attr| attr.key() == AttrKey::Exec))
        .expect("dot 供给");
    let exec = dot
        .attrs
        .iter()
        .find(|attr| attr.key() == AttrKey::Exec)
        .expect("dot 必须声明执行单元");
    assert_eq!(exec.values(), ["cube".to_string()]);
}

#[test]
fn exec_must_name_a_declared_unit() {
    let cube = position(ascend(), "dot", "mmad");
    let exec = Value::array(vec![Value::text("cube0")]);
    let table = edited(ascend(), &format!("provides[{cube}].exec"), exec);
    let profile = Profile::parse(&table).expect("解析期只做闭集校验");
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors.iter().any(|error| matches!(
            error,
            ProfileError::Unknown { value, .. } if value == "cube0"
        )),
        "{errors:?}"
    );
}

#[test]
fn a_unit_cannot_reach_a_space_it_does_not_have() {
    let mmad = position(ascend(), "dot", "mmad");
    let space = Value::array(vec![
        Value::text("FAST"),
        Value::text("REG"),
        Value::text("GM"),
    ]);
    let table = edited(ascend(), &format!("provides[{mmad}].space"), space);
    let profile = Profile::parse(&table).expect("`GM` 是闭集内的存储名");
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors.iter().any(|error| matches!(
            error,
            ProfileError::BadValue { field, detail }
                if field == "provides[dot].space" && detail.contains("GM")
        )),
        "{errors:?}"
    );
}

#[test]
fn one_primitive_can_have_several_signatures_but_not_the_same_impl_twice() {
    let mut profile = Profile::parse(cpu()).expect("可解析");
    let mut extra = profile.provides[0].clone();
    extra.grade = Grade::L1;
    extra.witness = None;
    // 换一个 impl 名就是另一条签名：同一原语下允许并存。
    extra.impl_name = "scalar_i4_wide".to_string();
    profile.provides.push(extra.clone());
    profile
        .validate(None)
        .expect("不同 impl 的两条签名可以并存");
    // 同名 = 无法区分谁实现了什么。
    profile.provides.push(extra);
    let errors = profile.validate(None).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("重名")),
        "{errors:?}"
    );
}

#[test]
fn a_need_can_demand_an_execution_unit() {
    let profile = ascend_profile();
    let dots: Vec<&Provide> = profile.providers(Prim::Dot).collect();
    assert!(
        dots.iter()
            .any(|dot| dot.covers(&Need::of(Prim::Dot).with_attr(exec(&["cube"]))))
    );
    // 档案里没有任何一条 dot 由 vector 实现：需求不能被"另一条签名"顶替。
    let vector = Need::of(Prim::Dot).with_attr(exec(&["vector"]));
    assert!(dots.iter().all(|dot| !dot.covers(&vector)));
    assert!(dots.iter().all(|dot| dot.diff(&vector).len() == 1));
    // 只有 L0：形状与单元都对得上，但仍然"不具备"。
    assert!(!profile.satisfies(&Need::of(Prim::Dot).with_attr(exec(&["cube"]))));
}

/// 一条 dequant 签名覆盖 i2 / i4 两种打包宽度：片段里位宽由 `$width` / `$cpb` / `$zero` 派生，
/// 所以不需要第二条签名。签名缺 i2 时 i2 系列 op 是"dt 缺 i2"的结构性缺席，永远协商不上——
/// 那不是证据等级问题，改 grade 也修不好。
#[test]
fn a_dequant_signature_covers_both_packed_widths() {
    let profile = ascend_profile();
    let dequant = profile
        .providers(Prim::Quant)
        .find(|provide| provide.impl_name == "scalar_dequant")
        .expect("P0 模板用的是 scalar_dequant");
    for dt in [DType::I2, DType::I4] {
        let need = Need::of(Prim::Quant).with_dt(DTypeSet::of([dt]));
        assert!(
            dequant.covers(&need),
            "scalar_dequant 缺 {dt:?}：{:?}",
            dequant.diff(&need)
        );
    }
}
