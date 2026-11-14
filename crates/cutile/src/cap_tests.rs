use crate::cap::*;
use crate::atom::{Attr, AttrKey, DType, DTypeSet, Grade, Need, Prim, Provide};
use crate::decl::{Axis, OpDecl, Skeleton, Variant};
use crate::prof::Profile;
use crate::sch::Schedule;
use diagnostic::data::Value;

fn op() -> OpDecl {
    let source = &op::builtin_named("gemv_i4").expect("内置");
    OpDecl::parse(source.declaration(), source.template()).expect("声明可解析")
}

fn cpu() -> Profile {
    Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案可解析")
}

fn nvidia() -> Profile {
    Profile::parse(prof::embedded("nvidia").expect("nvidia 档案")).expect("档案可解析")
}

/// cpu 档案去掉这些原语：用来演示"设备缺一条能力就降级"。
fn cpu_without(prims: &[Prim]) -> Profile {
    let mut profile = cpu();
    profile.provides.retain(|have| !prims.contains(&have.prim));
    profile
}

/// 一个模板只用 `dot` 的合成声明：内置 `gemv_i4` 的模板由所有变体共享，`load_async` /
/// `commit_wait` / `bar` 因此是算子级需求，演示不了"缺一个原语就降级"，差异只能落在 `extra_needs`。
fn downgrade_op() -> OpDecl {
    // 合成声明只需要一份可解析的签名：协商不读 ABI，但 `OpDecl` 的不变量要求它存在。
    let template = "void cutile_probe_kernel(float *y) { @dot(acc, a, b, k); }\n";
    OpDecl {
        name: "probe".to_string(),
        skeleton: Skeleton::Gemm,
        summary: "-".to_string(),
        axes: vec![Axis {
            name: "GMAX".to_string(),
            values: vec![32],
        }],
        needs: vec![Need::of(Prim::Dot)],
        layouts: Vec::new(),
        variants: vec![
            Variant {
                name: "pipe2".to_string(),
                order: 1,
                schedule: Schedule::parse("pipe<2>").expect("调度"),
                mma_share: 0.62,
                extra_needs: vec![Need::of(Prim::LoadAsync), Need::of(Prim::CommitWait)],
            },
            Variant {
                name: "direct".to_string(),
                order: 2,
                schedule: Schedule::parse("direct").expect("调度"),
                mma_share: 0.35,
                extra_needs: Vec::new(),
            },
        ],
        template: template.to_string(),
        signature: op::sig::Signature::parse("probe", template).expect("签名"),
    }
}

#[test]
fn cpu_hits_the_first_variant() {
    let path = negotiate(&op(), &cpu()).expect("CPU 应命中首档");
    assert_eq!(path.variant, "pipe2");
    assert_eq!(path.index, 0);
    assert!(path.schedule.uses_async());
}

#[test]
fn missing_evidence_blocks_every_variant() {
    let error = negotiate(&op(), &nvidia()).unwrap_err();
    assert!(error.to_string().contains("没有可行变体"));
    assert!(error.chain.iter().all(|step| !step.ok));
    assert!(error.to_string().contains("L2"), "证据不足：{error}");
}

#[test]
fn variant_level_needs_drive_downgrade() {
    // 去掉 load_async / commit_wait：pipe2 需要它们，direct 不需要 —— 这正是 extra_needs 的意义。
    let profile = cpu_without(&[Prim::LoadAsync, Prim::CommitWait]);
    let path = negotiate(&downgrade_op(), &profile).expect("应降级到 direct");
    assert_eq!(path.variant, "direct");
    assert!(!path.schedule.uses_async());
}

#[test]
fn diagnostics_name_the_missing_atom_and_the_unlock() {
    // dot 是算子级公共需求：缺它时整条链都不可行，诊断必须点名并给出解锁路径。
    let error = negotiate(&op(), &cpu_without(&[Prim::Dot])).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("dot"), "{text}");
    assert!(text.contains("建议："), "{text}");
    assert!(text.contains("L2 见证"), "{text}");
}

/// 解锁建议必须自洽：变体名与"还缺哪些原语"来自同一档。首选缺 `dot` 与 `load_async`、兜底只缺
/// `dot` 时，建议应指向兜底（第 2 档）且只列 `dot`；拼错会让人补完证之后发现仍然不可用。
#[test]
fn the_unlock_advice_names_the_variant_its_missing_atoms_belong_to() {
    let error =
        negotiate(&downgrade_op(), &cpu_without(&[Prim::Dot, Prim::LoadAsync])).unwrap_err();
    let unlock = error.unlock.as_ref().expect("应当给出解锁路径");
    assert_eq!(unlock.variant, "direct", "{unlock:?}");
    assert_eq!(unlock.missing, vec![Prim::Dot], "{unlock:?}");
    let text = error.to_string();
    assert!(text.contains("可用 direct（第 2/2 档"), "{text}");
    assert!(!text.contains("可用 pipe2"), "{text}");
}

/// 缺项按原语去重：调度与 `needs` 都要 `bar` 时建议只列一次。重复会让"补哪些原语"读成"补两条
/// 需求"，也会让"缺得最少"的挑选偏向需求条数多的变体。
#[test]
fn the_unlock_advice_lists_each_atom_once() {
    let mut op = downgrade_op();
    op.needs = vec![Need::of(Prim::Bar)];
    op.variants.truncate(1);
    let error = negotiate(&op, &cpu_without(&[Prim::Bar])).unwrap_err();
    let unlock = error.unlock.as_ref().expect("应当给出解锁路径");
    assert_eq!(unlock.missing, vec![Prim::Bar], "{unlock:?}");
    let text = error.to_string();
    let advice = text.split("建议：").nth(1).expect("有建议行");
    assert_eq!(advice.matches("`bar`").count(), 1, "{advice}");
}

#[test]
fn missing_async_atom_only_disables_the_pipelined_variant() {
    let op = downgrade_op();
    let profile = cpu_without(&[Prim::LoadAsync]);
    let path = negotiate(&op, &profile).expect("应降级到 direct");
    assert_eq!(path.variant, "direct");
    // 解释不复用协商结果：inspect 总是看完整条链，因此被跳过的 pipe2 也在。
    let checks = inspect(&op, &profile);
    let piped = &checks[0];
    assert_eq!(piped.name, "pipe2");
    assert!(!piped.failures.is_empty(), "pipe2 应报出缺失的 load_async");
}

/// P0 的每条需求都必须被某一条签名**结构性**覆盖，剩下的唯一障碍只能是证据等级——`dt`/`lanes`/`dst`
/// 对不上时补见证也没用，不能混在 L0/L1 里。
#[test]
fn every_p0_need_is_covered_by_some_signature_on_every_accelerator() {
    for vendor in ["nvidia", "intel", "ascend"] {
        let profile =
            Profile::parse(prof::embedded(vendor).expect("内置档案")).expect("档案可解析");
        let checks = inspect(&op(), &profile);
        assert!(!checks.is_empty(), "{vendor} 应当有降级链");
        for check in &checks {
            for failure in &check.failures {
                assert!(
                    failure.mismatches.is_empty(),
                    "{vendor}: {} 仍然不同构：{}",
                    failure.need.signature(),
                    failure.reason()
                );
                assert!(failure.blocked_by_grade, "{vendor}: 只应被证据等级挡住");
            }
        }
    }
}

#[test]
fn the_signature_that_covers_the_need_is_the_one_compared() {
    let profile = nvidia();
    let need = Need::of(Prim::Dot)
        .with_dt(DTypeSet::of([DType::F32]))
        .with_attr(Attr::new(AttrKey::Lanes, ["1"]));
    // 这条用例只验"签名比对挑中了哪条供给"，与证据等级无关：按档案自述取证据即可。
    let report = check_need(&profile, &need, &Proof::declared(&profile));
    assert!(report.mismatches.is_empty(), "{}", report.reason());
    assert_eq!(report.have.expect("nvidia 有 dot 供给").impl_name, "fma");
}

#[test]
fn inspect_reports_the_whole_chain_as_data() {
    let checks = inspect(&op(), &cpu());
    assert_eq!(checks.len(), 2);
    assert!(checks.iter().all(|check| check.failures.is_empty()));
    assert_eq!(checks[0].name, "pipe2");
    assert_eq!(checks[1].name, "direct");
}

/// 同一份签名在两层里换一种合法写法（标量/数组、`bytes`/`m`）必须仍然同构：两层解析器若无共同词汇，
/// "协商通过"与"档案声明"说的就不是一回事。这里直接拿协商用的覆盖关系当判据，两个方向都要成立。
#[test]
fn one_signature_written_in_either_layer_stays_the_same() {
    fn declare(load: &[(&str, Value)]) -> OpDecl {
        let source = op::builtin_named("gemv_i4").expect("内置");
        let mut table = source.declaration().clone();
        table.set("needs.load", Value::table(load.to_vec()));
        OpDecl::parse(&table, source.template()).expect("合成声明可解析")
    }

    fn provide(signature: &[(&str, Value)]) -> Provide {
        let mut fields = vec![
            ("prim", Value::text("load")),
            ("impl", Value::text("alias")),
        ];
        fields.extend_from_slice(signature);
        fields.push(("grade", Value::text("L0")));
        let mut table = prof::embedded("cpu").expect("cpu 档案").clone();
        table.push("provides", Value::table(fields));
        let mut profile = Profile::parse(&table).expect("合成档案可解析");
        profile.provides.pop().expect("新加的供给")
    }

    /// 声明里的 `load` 需求被这条供给覆盖了吗。
    fn assert_covers(op: &OpDecl, have: &Provide, note: &str) {
        let need = op
            .needs
            .iter()
            .find(|need| need.prim == Prim::Load)
            .expect("load 需求");
        assert!(have.covers(need), "{note}：{:?}", have.diff(need));
    }

    let scalar = vec![
        ("bytes", Value::text("<=4096")),
        ("dt", Value::text("f32")),
        ("space", Value::text("FAST")),
        ("src", Value::text("GM")),
        ("dst", Value::text("FAST")),
    ];
    let array = vec![
        ("m", Value::text("<=4096")),
        ("dt", Value::array(vec![Value::text("f32")])),
        ("space", Value::array(vec![Value::text("FAST")])),
        ("src", Value::array(vec![Value::text("GM")])),
        ("dst", Value::array(vec![Value::text("FAST")])),
    ];
    // 声明写标量、档案写数组与 `m`；反向：声明写数组、档案写标量。
    assert_covers(&declare(&scalar), &provide(&array), "档案写法覆盖声明");
    assert_covers(&declare(&array), &provide(&scalar), "声明写法覆盖档案");
}

#[test]
fn schedule_primitives_are_part_of_the_requirement() {
    // 只剩 direct 可用时 bar 不再被需要。
    let path = negotiate(&downgrade_op(), &cpu_without(&[Prim::Bar])).expect("应降级到 direct");
    assert_eq!(path.variant, "direct");
    // 再去掉公共需求里的 dot：两条链都不可行，诊断必须点名 bar（来自 pipe<2> 调度）。
    let error = negotiate(&downgrade_op(), &cpu_without(&[Prim::Bar, Prim::Dot])).unwrap_err();
    assert!(error.to_string().contains("bar"), "{error}");
}

/// 空证据不是"全都具备"：链会走完，每一档都不可行，原因写的是"欠一次执行"。
#[test]
fn an_unexecuted_l2_offer_does_not_negotiate() {
    let error = negotiate_with(&op(), &cpu(), &Proof::none()).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("没有这次执行的见证"), "{text}");
    assert!(!error.needs.is_empty(), "{error}");
    assert!(
        error.needs.iter().all(|need| !need.is_satisfied()),
        "空证据下一档都不能满足：{:?}",
        error.needs
    );
    // 它不是因为"档案没写"而失败：这些供给都写着 L2，缺的是执行。
    assert!(
        error
            .needs
            .iter()
            .all(|need| !need.blocked_by_grade && need.unproven),
        "{:?}",
        error.needs
    );
}

/// 档案自述是解释性报告的口径：同一份档案按声明协商照样命中首档。
#[test]
fn the_declared_reading_is_still_available_for_reports() {
    let profile = cpu();
    let declared = Proof::declared(&profile);
    let verified = profile
        .provides
        .iter()
        .filter(|have| have.grade.is_verified())
        .count();
    assert_eq!(declared.len(), verified, "每一条 grade ≥ L2 的供给都在里面");
    let path = negotiate_with(&op(), &profile, &declared).expect("按声明可行");
    assert!(path.is_head());
}

/// 两条覆盖同一需求的供给：先取这次真的跑过的那条，而不是等级写得漂亮的那条。
#[test]
fn a_proven_supply_wins_over_a_higher_grade_that_was_not_executed() {
    let op = op();
    let need = op
        .needs
        .iter()
        .find(|need| need.prim == Prim::Dot)
        .expect("dot 需求")
        .clone();
    let mut profile = cpu();
    // 一条从没被执行过的 `L3`：签名与 `scalar_f32` 逐项相同，只有等级写得更漂亮。
    let mut vendor = profile
        .providers(Prim::Dot)
        .next()
        .cloned()
        .expect("cpu 有 dot 供给");
    vendor.impl_name = "vendor_dot".to_string();
    vendor.grade = Grade::L3;
    profile.provides.push(vendor);

    let mut proof = Proof::none();
    for have in &profile.provides {
        if have.impl_name != "vendor_dot" {
            proof.insert(have.prim, &have.impl_name);
        }
    }
    let (chosen, _) = select(&profile, &need, &proof).expect("有供给");
    assert_eq!(
        chosen.impl_name, "scalar_f32",
        "不能拿没执行过的等级去挤掉跑过的实现"
    );

    // 反向：它真的跑过了，就按等级取它。
    let mut proven = proof.clone();
    proven.insert(Prim::Dot, "vendor_dot");
    let (chosen, _) = select(&profile, &need, &proven).expect("有供给");
    assert_eq!(chosen.impl_name, "vendor_dot");
}
