use super::*;
use diagnostic::Code;
use diagnostic::data::Value;

#[test]
fn every_builtin_entry_declares_an_operator() {
    for op in builtin() {
        assert!(!op.name().is_empty());
        assert!(op.declaration().table("op").is_some(), "{}", op.name());
        assert!(!op.template().is_empty(), "{}", op.name());
    }
}

/// 生成表按文件名升序（旧的 `op::load(dir)` 也这样排序）。
#[test]
fn declared_names_are_sorted() {
    let generated = names();
    assert!(
        generated.windows(2).all(|pair| pair[0] < pair[1]),
        "{generated:?}"
    );
    for name in generated {
        assert!(declared(name).is_some(), "{name} 有名字却没有表");
    }
}

/// 注册表与构建期生成的声明表必须逐条对上。
///
/// 少一条是"声明层有、注册表没有"——[`builtin`] 会静默跳过它，于是算子发布了却取不到；
/// 多一条是"注册表有、`ops/` 里没有这个文件"，`include_str!` 让它在编译期就挂。两个方向都判。
#[test]
fn the_registry_covers_the_generated_table_exactly() {
    let registered: Vec<&str> = REGISTRY.iter().map(|entry| entry.name).collect();
    assert_eq!(
        registered,
        names(),
        "注册表与 `ops/*.toml` 的算子集必须一一对应"
    );
    assert_eq!(builtin().len(), REGISTRY.len(), "每条注册都必须有声明");
}

/// 声明与语义同属一条注册：不该出现"内嵌里有、语义表没有"的算子。
#[test]
fn every_builtin_entry_has_a_model() {
    for op in builtin() {
        assert!(model(op.name()).is_some(), "{} 没有语义表", op.name());
    }
    assert!(model("no_such_op").is_none());
}

/// 测试要的是**量化那条**算子，不是"注册表里的第一条"。
///
/// 按位置取会在注册表从一条变两条的那天静默换人——而两条算子并不等价（稠密 f32 与 int4
/// 的代价差 8 倍）。名字写在这里，注册表增删都不会让这条测试去验另一个算子。
fn int4() -> Op {
    builtin_named("gemv_i4").expect("gemv_i4 已注册")
}

/// 在 `variants` 里按名字取一个变体表；测试改写声明时定位到具体变体。
fn variant_mut<'a>(declaration: &'a mut Table, name: &str) -> &'a mut Table {
    declaration
        .array_mut("variants")
        .expect("variants")
        .iter_mut()
        .find_map(|value| {
            let Value::Table(variant) = value else {
                return None;
            };
            (variant.string("name") == Some(name)).then_some(variant)
        })
        .expect("变体存在")
}

#[test]
fn the_builtin_template_only_uses_declared_primitives() {
    let op = int4();
    let declaration =
        decl::OpDecl::parse(op.declaration(), op.template()).expect("内置声明必须自洽");
    assert_eq!(declaration.name, op.name());
}

#[test]
fn a_signature_with_an_unsupported_buffer_type_is_rejected_at_load() {
    // ABI 的唯一事实源是 `.k` 的签名：把 `scales` 写成 `double *` 必须在装载期就失败，
    // 而不是等 `cc` 报 C 的指针类型不兼容——诊断要带算子名与形参名。
    let op = int4();
    let broken = op
        .template()
        .replace("const float *scales", "double *scales");
    let error = decl::OpDecl::parse(op.declaration(), &broken).expect_err("类型不在闭集内");
    let text = error.to_string();
    assert!(text.contains(op.name()), "{text}");
    assert!(text.contains("scales"), "{text}");
    assert!(text.contains("double"), "{text}");
}

#[test]
fn a_template_primitive_must_be_declared_in_needs() {
    let op = int4();
    let template = format!("{}\n@atomic(y, n, out);\n", op.template());
    let error = decl::OpDecl::parse(op.declaration(), &template)
        .expect_err("模板里的原语不在 needs 里必须报错");
    assert!(error.to_string().contains("atomic"), "{error}");
    assert!(error.to_string().contains(op.name()), "{error}");
}

/// 模板传的能力参数必须被 `[needs]` 允许：否则设备按声明验、代码按模板生成。
#[test]
fn a_call_argument_must_match_the_declared_need() {
    let op = int4();
    let broken = op.template().replace("sat: off", "sat: on");
    let error = decl::OpDecl::parse(op.declaration(), &broken).expect_err("模板与需求矛盾必须报错");
    let text = error.to_string();
    assert!(text.contains(op.name()), "{text}");
    assert!(text.contains("cvt"), "{text}");
    assert!(text.contains("sat"), "{text}");
    assert!(text.contains("on"), "{text}");
}

/// 闭集内的能力属性传了却没声明，等于协商永远不检查它。
#[test]
fn a_capability_argument_must_be_declared_in_needs() {
    let op = int4();
    let broken = op
        .template()
        .replace("sat: off", "sat: off, accuracy: approx");
    let error =
        decl::OpDecl::parse(op.declaration(), &broken).expect_err("未声明的能力参数必须报错");
    let text = error.to_string();
    assert!(text.contains(op.name()), "{text}");
    assert!(text.contains("cvt"), "{text}");
    assert!(text.contains("accuracy"), "{text}");
}

/// 闭集之外的命名参数是片段级提示，不构成能力声明，不能被这条判据误伤。
#[test]
fn a_fragment_only_named_argument_stays_legal() {
    let op = int4();
    let extra = op.template().replace("sat: off", "sat: off, along: 32");
    decl::OpDecl::parse(op.declaration(), &extra).expect("片段级命名参数不该被能力判据拒绝");
}

#[test]
fn every_variant_must_cover_the_shared_template() {
    let op = int4();
    // 把 `bar` 从算子级需求挪到只有 `pipe2` 的增量需求里：`direct` 仍然会 emit 出 `@bar`，
    // 所以装载期必须拒绝，而不是等协商选中 `direct` 才在 substitute 里发现片段缺失。
    let mut broken = op.declaration().clone();
    broken.table_mut("needs").expect("needs").remove("bar");
    variant_mut(&mut broken, "pipe2").set("extra_needs.bar", Value::table(vec![]));
    let error =
        decl::OpDecl::parse(&broken, op.template()).expect_err("变体没有覆盖共享模板必须报错");
    assert!(error.to_string().contains("direct"), "{error}");
    assert!(error.to_string().contains("bar"), "{error}");
}

/// 判据 1（cutile README §12.2）：首选变体必须把非 MMA 工作盖住，否则该换 S3/S4。
#[test]
fn the_preferred_variant_must_hide_its_non_mma_work() {
    let op = int4();
    // 占比降下来链仍然单调（0.10 → 0.05），所以先失败的只会是掩盖判据。
    let mut hidden = op.declaration().clone();
    variant_mut(&mut hidden, "pipe2").set("mma_share", Value::real(0.10));
    variant_mut(&mut hidden, "direct").set("mma_share", Value::real(0.05));
    assert_ne!(
        hidden.at("variants[0].mma_share"),
        op.declaration().at("variants[0].mma_share"),
        "声明里的占比写法变了"
    );
    let error = decl::OpDecl::parse(&hidden, op.template()).expect_err("首选变体掩盖不足必须报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("首选变体 `pipe2`"), "{error}");
    assert!(error.to_string().contains("45%"), "{error}");
}

/// 兜底变体的掩盖账超阈值是事实，不是声明错误：它存在的理由正是"设备缺原语时仍能跑"。
#[test]
fn a_falling_back_variant_may_miss_the_masking_bar() {
    let op = int4();
    let op = decl::OpDecl::parse(op.declaration(), op.template()).expect("内置声明应当通过");
    let direct = op.variant("direct").expect("兜底变体");
    let masking = sch::Masking::of(direct.mma_share, &direct.schedule);
    assert!(masking.claims_mma(), "{masking}");
    assert!(!masking.within_limit(), "{masking}");
    assert!(masking.to_string().contains("降级兜底"), "{masking}");
}
