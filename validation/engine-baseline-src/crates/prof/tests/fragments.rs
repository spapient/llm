//! 片段层与档案的对账：六台设备各跑一遍同一套判据。
//!
//! 判据本身在 `testkit::device`，用例由 `testkit::device_suite!` 摊开；每台设备只是一行
//! `mod`，因为要摊开的判据相同、只有生成表不同。左边的表来自 `build.rs`，右边是
//! `crates/prof/src/<vendor>/<vendor>.toml` 声称具备的能力，中间的任何错位都是缺陷。

mod ascend {
    use llm_prof::ascend::*;
    testkit::device_suite!();
}

mod cpu {
    use llm_prof::cpu::*;
    testkit::device_suite!();
}

mod intel {
    use llm_prof::intel::*;
    testkit::device_suite!();

    /// 宿主 `cc` 编得过 `.cl` 只是因为它写的是标量 C；OpenCL 前端才是这份文件的真读者。
    #[test]
    fn the_witness_is_accepted_by_the_opencl_frontend() {
        let frontend = std::env::var_os("CLANG").unwrap_or_else(|| "clang".into());
        let witness = format!("{ROOT}/witness.{}", DEVICE.ext);
        let probe = std::process::Command::new(&frontend)
            .args(["-x", "cl", "-cl-std=CL2.0", "-fsyntax-only"])
            .arg(&witness)
            .output();
        let Ok(output) = probe else {
            testkit::skip(format!(
                "本机没有 `{}`：OpenCL 见证未过设备前端",
                frontend.to_string_lossy()
            ));
            return;
        };
        assert!(
            output.status.success(),
            "`{}` 拒绝了 {witness}：{}",
            frontend.to_string_lossy(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
}

mod metal {
    use llm_prof::metal::*;
    testkit::device_suite!();
}

mod musa {
    use llm_prof::musa::*;
    testkit::device_suite!();
}

mod nvidia {
    use llm_prof::nvidia::*;
    testkit::device_suite!();
}

mod nvidia_a3000 {
    use llm_prof::nvidia_a3000::*;
    testkit::device_suite!();
}

/// 片段集合的克隆只加一次引用计数；克隆之后再登记，改的是克隆自己（写时复制）。
#[test]
fn a_clone_keeps_its_own_inserts() {
    let mut base = llm_prof::FragmentSet::new("cpu", "c");
    base.insert(llm_prof::Prim::Load, fragment("load", "one"))
        .expect("登记首份片段");
    let mut copy = base.clone();
    copy.insert(llm_prof::Prim::Load, fragment("load", "two"))
        .expect("登记第二份片段");
    assert_eq!(
        base.fragments_of(llm_prof::Prim::Load).len(),
        1,
        "克隆不能共享后来的插入"
    );
    assert_eq!(copy.fragments_of(llm_prof::Prim::Load).len(), 2);
    assert_eq!(base.implementation(llm_prof::Prim::Load), Some("one"));
}

/// 一份合法片段：第一行是签名声明，后面是正文。
fn fragment(prim: &str, implementation: &str) -> String {
    format!(
        "/* {} {prim}/{implementation} */\nvoid {implementation}(void) {{}}\n",
        llm_prof::SIGN_MARKER
    )
}
