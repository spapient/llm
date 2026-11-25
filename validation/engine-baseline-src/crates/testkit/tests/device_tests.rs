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
