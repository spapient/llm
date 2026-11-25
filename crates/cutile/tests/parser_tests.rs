use super::*;
use crate::atom::{Need, Provide, Span};
use crate::builtin::Catalogue;
use crate::kernel::{Arith, LocalType, ScalarType};
use crate::substitute::FragmentSet;

/// 全量内置模板都走 `.k → Program → printer`，没有文本回退。
#[test]
fn every_builtin_template_lowers_to_verified_ir() {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let profile = catalogue
        .profiles()
        .iter()
        .find(|profile| profile.meta.vendor == "cpu")
        .expect("CPU 档案");
    let fragments = prof::builtin("cpu", "c").expect("CPU 片段");
    let expected: usize = catalogue
        .ops()
        .iter()
        .map(|op| crate::decl::expand(op).expect("期望 tile 数").len())
        .sum();
    let mut lowered = 0;
    for original in catalogue.ops() {
        let specialized = prof::catalogue::for_device(original, "cpu").expect("CPU 算子");
        let op = specialized.as_ref();
        let variant = op.variants.first().expect("至少一个变体");
        // 语法覆盖夹具：只补类型化供给，不代表真实 CPU 获得这些能力。
        let fixture = syntax_profile(op, variant, profile, &fragments);
        for combo in crate::decl::expand(op).expect("tile 候选") {
            let atoms = AtomTable::new(op, variant, &fixture, &fragments)
                .unwrap_or_else(|error| panic!("{} 原子绑定失败：{error}", op.name));
            let program = parse(op, &combo, &atoms)
                .unwrap_or_else(|error| panic!("{} / {combo} 解析失败：{error}", op.name));
            program
                .verify()
                .unwrap_or_else(|error| panic!("{} / {combo} 校验失败：{error}", op.name));
            let expected_atoms =
                variant.needs(op).count() + variant.schedule.required_prims().len();
            assert_eq!(
                program.atoms().as_slice().len(),
                expected_atoms,
                "{} 原子数不符",
                op.name
            );
            let encoded = program.encode().expect("结构编码");
            assert_eq!(encoded, program.encode().expect("再次编码"), "编码必须稳定");
            let source = program
                .emit_k()
                .unwrap_or_else(|error| panic!("{} 发射失败：{error}", op.name));
            assert!(
                source.contains(&format!("void cutile_{}_kernel", op.name)),
                "{source}"
            );
            assert!(
                !source
                    .lines()
                    .any(|line| line.trim_start().starts_with('@')),
                "原子调用语句必须在 printer 展开：{source}"
            );
            lowered += 1;
        }
    }
    assert_eq!(lowered, expected, "每个内置模板 × tile 都必须进入 IR");
}

/// 向量类型按真实后端能力 lowering，不把同一个名字当成所有目标的完成证据。
#[test]
fn vector_program_uses_registered_target_lowerings_only() {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let profile = catalogue.profile("cpu").expect("CPU 档案");
    let fragments = prof::builtin("cpu", "c").expect("CPU 片段");
    let op = catalogue.op("add_f32x4").expect("向量算子");
    let combo = crate::decl::expand(op).expect("tile").remove(0);
    let variant = op.variants.first().expect("变体");
    let atoms = AtomTable::new(op, variant, profile, &fragments).expect("原子表");
    let program = parse(op, &combo, &atoms).expect("向量 Program IR");

    let c = program.emit(crate::prof::Emit::C).expect("C 向量扩展");
    assert!(c.contains("const float4 *a"));
    assert!(!c.contains("vector_size"), "typedef 归最终装配层");
    let source = crate::emit::assemble(profile, op, variant, &combo, &c).text;
    assert_eq!(source.matches("typedef float float4").count(), 1);
    let opencl = program
        .emit(crate::prof::Emit::Opencl)
        .expect("OpenCL 原生向量");
    assert!(opencl.contains("const float4 *a"));
    assert!(!opencl.contains("vector_size"));
    let error = program
        .emit(crate::prof::Emit::Cuda)
        .expect_err("CUDA 算术 lowering 未登记");
    assert!(error.to_string().contains("no registered native float4"));
}

#[test]
fn family_quantization_constants_are_available_to_templates() {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let op = catalogue.op("gemv_i2").expect("量化 GEMV");
    assert_eq!(op.const_value("cpb"), Some("4"));
    assert_eq!(op.const_value("width"), Some("2"));
    assert_eq!(op.const_value("zero"), Some("2"));
}

/// 活跃 `#error` 在 IR 构建期失败，不能流进后端源码。
#[test]
fn active_preprocessor_error_is_rejected_before_printing() {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let profile = catalogue
        .profiles()
        .iter()
        .find(|profile| profile.meta.vendor == "cpu")
        .expect("CPU 档案");
    let fragments = prof::builtin("cpu", "c").expect("CPU 片段");
    let mut op = catalogue.op("add_f32").expect("add").clone();
    op.template = format!("#if 1\n#error injected\n#endif\n{}", op.template);
    let combo = crate::decl::expand(&op).expect("tile").remove(0);
    let variant = op.variants.first().expect("变体");
    let atoms = AtomTable::new(&op, variant, profile, &fragments).expect("原子表");
    let error = parse(&op, &combo, &atoms).expect_err("活跃 error 必须失败");
    assert!(error.to_string().contains("injected"), "{error}");
}

/// 通用程序也经过同一常量折叠与死形参通道。
#[test]
fn program_optimizer_folds_constants_and_drops_unused_parameters() {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let profile = catalogue
        .profiles()
        .iter()
        .find(|profile| profile.meta.vendor == "cpu")
        .expect("CPU 档案");
    let fragments = prof::builtin("cpu", "c").expect("CPU 片段");
    let op = catalogue.op("add_f32").expect("add");
    let combo = crate::decl::expand(op).expect("tile").remove(0);
    let variant = op.variants.first().expect("变体");
    let atoms = AtomTable::new(op, variant, profile, &fragments).expect("原子表");
    let parsed = parse(op, &combo, &atoms).expect("解析");
    let folded = LocalDecl {
        name: "folded".to_string(),
        ty: LocalType {
            element: ScalarType::F32,
            pointers: 0,
            constant: false,
            array: None,
        },
        init: Some(Expr::Binary {
            op: Arith::Add,
            left: Box::new(Expr::F32(1.0f32.to_bits())),
            right: Box::new(Expr::F32(2.0f32.to_bits())),
        }),
    };
    let program = Program::new(
        "constant",
        parsed.params().to_vec(),
        parsed.atoms().clone(),
        vec![Statement::Decl(folded)],
    );
    let source = program.emit_k().expect("程序发射");
    assert!(source.contains("float folded = 3.0f"), "{source}");
    assert!(source.contains("void cutile_constant_kernel()"), "{source}");
}

/// 仅为语法覆盖构造合成能力：所有类型仍经真实 `AtomTable` 与 `Provide::diff` 选择。
fn syntax_profile(
    op: &OpDecl,
    variant: &crate::decl::Variant,
    base: &crate::prof::Profile,
    fragments: &FragmentSet,
) -> crate::prof::Profile {
    let mut profile = base.clone();
    let mut needs: Vec<Need> = variant.needs(op).cloned().collect();
    needs.extend(
        variant
            .schedule
            .required_prims()
            .iter()
            .copied()
            .map(Need::of),
    );
    let mut prims = Vec::new();
    for need in &needs {
        if !prims.contains(&need.prim) {
            prims.push(need.prim);
        }
    }
    profile
        .provides
        .retain(|provide| !prims.contains(&provide.prim));
    for prim in prims {
        let required: Vec<&Need> = needs.iter().filter(|need| need.prim == prim).collect();
        let implementation = fragments
            .fragment(prim)
            .map_or("noop", |fragment| fragment.implementation());
        profile.provides.push(Provide {
            prim,
            extent: Span::any(),
            n: Span::any(),
            k: Span::any(),
            dt: crate::atom::DTypeSet::of(required.iter().flat_map(|need| need.dt.iter())),
            space: crate::atom::SpaceSet::of(required.iter().flat_map(|need| need.space.iter())),
            attrs: required
                .iter()
                .flat_map(|need| need.attrs.clone())
                .collect(),
            impl_name: implementation.to_string(),
            grade: crate::atom::Grade::L2,
            witness: None,
        });
    }
    profile
}
