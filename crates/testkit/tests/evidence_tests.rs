use super::*;
use crate::device::{Contract, Provision, Witness};

const SIGNED: &str = "/* cutile-sign: load/scalar_copy */\n$0 = $1;";
const SOURCE: &str = "int witness_load(float *out) { out[0] = 2.0f; return 1; }\n";

fn witness<'a>(expect: &'a [f32]) -> Provision<'a> {
    Provision {
        prim: "load",
        impl_name: "scalar_copy",
        witness: Some(Witness {
            func: "witness_load",
            expect,
        }),
    }
}

/// 一份最小探针：一条 `load/scalar_copy` 供给，签名、入口、见证都齐。
fn probe<'a>(provides: &'a [Provision<'a>], source: &'a str) -> Contract<'a> {
    Contract {
        id: "probe",
        ext: "c",
        dir: ".",
        p0: &["load"],
        fragments: &[("load", SIGNED)],
        prims: &[("load", "witness_load")],
        provides,
        epilogue_stem: "epilogue",
        source,
    }
}

#[test]
fn a_self_contained_witness_is_accepted_but_a_quoted_include_is_not() {
    let good = [witness(&[2.0])];
    assert_eq!(probe(&good, SOURCE).check_self_contained(), Ok(()));
    assert_eq!(
        probe(&good, "#include <math.h>\n").check_self_contained(),
        Ok(()),
        "尖括号里的标准头不破坏自足"
    );
    let defect = probe(&good, "#include \"preamble.h\"\n").check_self_contained();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("引号包含")),
        "引号包含在副本旁边解析不到，必须被发现：{defect:?}"
    );
}

#[test]
fn expectations_must_fit_the_reading_buf() {
    let good = [witness(&[2.0])];
    assert_eq!(probe(&good, SOURCE).check_expectations(), Ok(()));
    let overflow = [2.0; MAX_OUT + 1];
    let loud = [witness(&overflow)];
    let defect = probe(&loud, SOURCE).check_expectations();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("期望值")),
        "超过读数缓冲的期望值必须被发现：{defect:?}"
    );
}

#[test]
fn a_witness_that_computes_something_else_is_a_defect() {
    let good = [witness(&[2.0])];
    assert_eq!(probe(&good, SOURCE).check_witnesses(), Ok(()));
    let wrong = [witness(&[3.0])];
    let defect = probe(&wrong, SOURCE).check_witnesses();
    assert!(
        defect
            .as_ref()
            .is_err_and(|reason| reason.contains("不成立")),
        "算错值的见证必须被发现：{defect:?}"
    );
}
