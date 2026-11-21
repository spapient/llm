//! `intel` 片段层与档案的对账：左边是 `build.rs` 编出来的三张表，右边是
//! `crates/prof/devices/intel.toml` 声称具备的能力，中间的任何错位都是缺陷。
//!
//! 判据本身在 `testkit::device`（六台设备共用一套），这里只负责把两张表摊成契约要的形状。
//! 额外一条：见证正文还要被 OpenCL 自己的前端接受——宿主 `cc` 编得过它，不等于
//! `-cl-std=CL2.0` 也认它。

use super::*;
use prof::prof::{Profile, builtin_profiles};
use testkit::device::{Contract, Provision, Witness};

/// P0 闭集：`crates/op/ops/*.k` 真的会 `@` 到的九个原语。屏障在这台设备上是真栅栏，
/// 但名字与另外八条一样不能缺——模板按名字取片段，缺一条就拼不出内核。
const P0: [&str; 9] = [
    "bar",
    "commit_wait",
    "cvt",
    "dot",
    "load",
    "load_async",
    "quant",
    "reduce",
    "store",
];

fn profile() -> Profile {
    builtin_profiles()
        .into_iter()
        .find(|profile| profile.meta.vendor == DEVICE.vendor)
        .unwrap_or_else(|| panic!("档案里没有厂商 `{}`", DEVICE.vendor))
}

/// 摊成同一份契约：片段正文与入口来自生成表，供给来自档案。`ext` 取 `DEVICE.ext`
/// 而不是档案的 `emit`——两处若不相等，由最后一个用例单独指出，其余用例只在一个口径上判定。
fn with_contract<T>(body: impl FnOnce(&Contract<'_>) -> T) -> T {
    let profile = profile();
    let fragments: Vec<(&str, &str)> = FRAGMENTS
        .iter()
        .map(|(prim, text)| (prim.as_str(), *text))
        .collect();
    let prims: Vec<(&str, &str)> = PRIMS
        .iter()
        .map(|(prim, func)| (prim.as_str(), *func))
        .collect();
    let provides: Vec<Provision<'_>> = profile
        .provides
        .iter()
        .map(|provide| Provision {
            prim: provide.prim.as_str(),
            impl_name: provide.impl_name.as_str(),
            witness: provide.witness.as_ref().map(|witness| Witness {
                func: witness.func.as_str(),
                expect: &witness.expect,
            }),
        })
        .collect();
    body(&Contract {
        id: &profile.meta.id,
        ext: DEVICE.ext,
        dir: ROOT,
        p0: &P0,
        fragments: &fragments,
        prims: &prims,
        provides: &provides,
        source: DEVICE.source,
    })
}

#[test]
fn the_fragment_layer_is_exactly_the_p0_set() {
    with_contract(|contract| contract.assert_p0_set());
}

#[test]
fn every_signature_is_a_profile_provide_that_has_a_witness() {
    with_contract(|contract| contract.assert_signatures());
}

#[test]
fn every_witness_function_is_claimed_by_exactly_one_entry() {
    with_contract(|contract| contract.assert_witness_entries());
}

#[test]
fn every_provision_rests_on_a_fragment_or_a_witness() {
    with_contract(|contract| contract.assert_evidence());
}

#[test]
fn the_fragment_directory_is_a_closed_set() {
    with_contract(|contract| contract.assert_closed_directory());
}

#[test]
fn every_expectation_fits_the_readback_buffer() {
    with_contract(|contract| contract.assert_expectations());
}

#[test]
fn the_witness_compiles_wherever_it_is_copied_to() {
    with_contract(|contract| contract.assert_self_contained());
}

#[test]
fn every_witness_computes_what_the_profile_expects() {
    with_contract(|contract| contract.assert_witnesses());
}

#[test]
fn the_device_row_is_the_profile_it_was_generated_from() {
    let profile = profile();
    assert_eq!(DEVICE.vendor, profile.meta.vendor, "设备名即档案的厂商键");
    assert_eq!(
        DEVICE.ext,
        profile.meta.emit.ext(),
        "片段扩展名由输出语言决定"
    );
    assert_eq!(DEVICE.emit, profile.meta.emit.as_str(), "输出语言名");
    assert_eq!(DEVICE.tool, profile.meta.tool, "档案声明的编译工具");
    assert_eq!(
        DEVICE.tool_ver, profile.meta.tool_ver,
        "登记的工具版本不能与档案漂移"
    );
    assert_eq!(
        DEVICE.prims, &PRIMS[..],
        "设备行里的原语入口必须就是生成表那份"
    );
}

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
        testkit::gating::skip(format!(
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
