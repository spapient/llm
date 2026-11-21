use super::*;

/// 夹具自身：算得对的见证回读到同值，算错的一条被相对误差抓到。
#[test]
fn a_host_witness_is_compiled_and_compared() {
    let source = "int witness_ok(float *out) { out[0] = 2.0f; return 1; }\n\
                  int witness_off(float *out) { out[0] = 2.5f; return 1; }\n";
    let calls = [
        Call {
            func: "witness_ok",
            expect: &[2.0],
        },
        Call {
            func: "witness_off",
            expect: &[2.0],
        },
    ];
    let Ok(readings) = run(source, "c", &calls) else {
        assert!(crate::gating::skip("本机没有可用的 C 编译器"));
        return;
    };
    assert_eq!(readings[0].count, 1);
    assert_eq!(readings[0].values, [2.0]);
    assert!(readings[0].max_rel_err <= 1e-6, "{:?}", readings[0]);
    assert!(readings[1].max_rel_err > 1e-3, "{:?}", readings[1]);
}

/// 少写一个值必须按个数判失败，而不是拿前几个值凑过去。
#[test]
fn a_count_mismatch_is_not_a_near_miss() {
    let source = "int witness_short(float *out) { out[0] = 1.0f; return 1; }\n";
    let calls = [Call {
        func: "witness_short",
        expect: &[1.0, 1.0],
    }];
    let Ok(readings) = run(source, "c", &calls) else {
        assert!(crate::gating::skip("本机没有可用的 C 编译器"));
        return;
    };
    assert_eq!(readings[0].count, 1);
    assert!(readings[0].max_rel_err.is_infinite());
}

/// 编译器不认的源码是错误，不是"没跑到"。
#[test]
fn a_witness_the_host_rejects_is_an_error() {
    let source = "int witness_broken(float *out) { out[0] = ; return 1; }\n";
    let calls = [Call {
        func: "witness_broken",
        expect: &[0.0],
    }];
    let Err(error) = run(source, "c", &calls) else {
        panic!("语法错误的见证必须编译失败");
    };
    assert!(error.contains("拒绝了见证源码"), "{error}");
}
