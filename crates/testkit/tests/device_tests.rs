use super::*;

const SIGNED: &str = "/* cutile-sign: load/scalar_copy */\n$0 = $1;";

/// 一台最小的探针：一条 `load/scalar_copy` 供给，见证源码里一个 `witness_load`。
const PROVIDES: [Provision<'_>; 1] = [Provision {
    prim: "load",
    impl_name: "scalar_copy",
    witness: Some(Witness {
        func: "witness_load",
        expect: &[1.0],
    }),
}];

fn probe<'a>(fragments: &'a [(&'a str, &'a str)], source: &'a str) -> Contract<'a> {
    Contract {
        id: "probe",
        ext: "c",
        dir: ".",
        p0: &["load"],
        fragments,
        prims: PRIMS_OF_PROBE,
        provides: &PROVIDES,
        epilogue: &[],
        epilogue_stem: "epilogue",
        source,
    }
}

const PRIMS_OF_PROBE: &[(&str, &str)] = &[("load", "witness_load")];

#[test]
fn a_signature_is_read_from_the_first_comment() {
    assert_eq!(sign(SIGNED).ok(), Some(("load", "scalar_copy")));
    assert!(sign("$0 = $1;").is_err(), "缺签名声明必须被发现");
    assert!(sign("/* cutile-sign: load */").is_err(), "少一段必须被发现");
}

#[test]
fn only_real_definitions_are_witnesses() {
    let source = "// 统一 ABI：`int witness_<name>(float *out)` 返回个数\n\
                  int witness_dot(float *out) { out[0] = 1.0f; return 1; }\n\
                  int helper(int x) { return x; }\n";
    assert_eq!(crate::evidence::definitions(source), ["witness_dot"]);
}

#[test]
fn a_well_formed_probe_passes_every_judgement() {
    let fragments = [("load", SIGNED)];
    let source = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n";
    let contract = probe(&fragments, source);
    assert_eq!(contract.check_p0_set(), Ok(()));
    assert_eq!(contract.check_signatures(), Ok(()));
    assert_eq!(contract.check_witness_entries(), Ok(()));
    assert_eq!(contract.check_evidence(), Ok(()));
}

#[test]
fn a_signature_the_archive_does_not_claim_is_a_defect() {
    let fragments = [("load", "/* cutile-sign: load/vectorized */\n$0 = $1;")];
    let defect = probe(&fragments, "").check_signatures();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("没有对应供给")),
        "无人认领的实现必须被发现：{defect:?}"
    );
}

#[test]
fn an_unclaimed_witness_is_a_defect() {
    let fragments = [("load", SIGNED)];
    let source = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n\
                  int witness_orphan(float *out) { out[0] = 1.0f; return 1; }\n";
    let defect = probe(&fragments, source).check_witness_entries();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("没有任何供给认领")),
        "没人认领的见证必须被发现：{defect:?}"
    );
}

#[test]
fn a_provision_without_any_evidence_is_a_defect() {
    let fragments = [("load", SIGNED)];
    let bare = [
        PROVIDES[0],
        Provision {
            prim: "mma",
            impl_name: "cube",
            witness: None,
        },
    ];
    let contract = Contract {
        provides: &bare,
        ..probe(&fragments, "")
    };
    let defect = contract.check_evidence();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("无凭无据")),
        "只有一行声明、正文与见证都缺的供给必须被发现：{defect:?}"
    );
}

#[test]
fn a_fragment_set_that_overruns_p0_is_a_defect() {
    let fragments = [("load", SIGNED), ("load", SIGNED)];
    let defect = probe(&fragments, "").check_p0_set();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("抢同一个原语")),
        "同一原语两份正文必须被发现：{defect:?}"
    );
}

/// 一条只关心"哪个形式、哪个入口"的写回条目。
fn claimed_epi<'a>(form: &'a str, func: &'a str) -> EpilogueEntry<'a> {
    EpilogueEntry {
        form,
        impl_name: "probe",
        witness: Witness {
            func,
            expect: &[1.0],
        },
    }
}

/// 写回见证与供给见证分属两个闭集，认领规则却一样：正文里有几个算法，档案里就得有几条写着它。
#[test]
fn an_epilogue_witness_that_no_entry_claims_is_a_defect() {
    const TWO: &str = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n\
                       int witness_epi_bias(float *out) { out[0] = 2.0f; return 1; }";
    let orphan = Contract {
        epilogue: &[],
        ..probe(&[("load", SIGNED)], TWO)
    };
    assert!(
        orphan
            .check_witness_entries()
            .is_err_and(|reason| reason.contains("witness_epi_bias")),
        "无供给认领的写回见证必须被发现"
    );
    let claimed = Contract {
        epilogue: &[claimed_epi("bias", "witness_epi_bias")],
        ..probe(&[("load", SIGNED)], TWO)
    };
    assert_eq!(claimed.check_witness_entries(), Ok(()));
}

/// 档案里写了条目、正文里没有那段函数，是另一半的漏。
#[test]
fn an_epilogue_entry_without_its_function_is_a_defect() {
    let ghost = Contract {
        epilogue: &[claimed_epi("bias", "witness_epi_absent")],
        ..probe(
            &[("load", SIGNED)],
            "int witness_load(float *out) { out[0] = 1.0f; return 1; }",
        )
    };
    assert!(
        ghost.check_witness_entries().is_err(),
        "只有条目没有正文的写回必须被发现"
    );
}

/// 写回见证也进执行器：同一段正文，期望值与它算的东西不一致时必须被判缺陷。
#[test]
fn an_epilogue_witness_that_computes_something_else_is_a_defect() {
    const SOURCE: &str = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n\
                          int witness_epi_bias(float *out) { out[0] = 1.0f; return 1; }";
    let entry = |expect: &'static [f32]| EpilogueEntry {
        form: "bias",
        impl_name: "column_broadcast",
        witness: Witness {
            func: "witness_epi_bias",
            expect,
        },
    };
    let honest = Contract {
        epilogue: &[entry(&[1.0])],
        ..probe(&[("load", SIGNED)], SOURCE)
    };
    assert_eq!(honest.check_witnesses(), Ok(()));
    let lying = Contract {
        epilogue: &[entry(&[2.0])],
        ..probe(&[("load", SIGNED)], SOURCE)
    };
    assert!(
        lying.check_witnesses().is_err(),
        "写回见证算的不是档案写的值，必须被发现"
    );
}
