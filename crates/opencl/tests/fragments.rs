//! `intel` 片段层与档案的对账：左边是 `build.rs` 编出来的三张表，右边是
//! `crates/prof/devices/intel.toml` 声称具备的能力，中间的任何错位都是缺陷。
//!
//! 判据本身在 `testkit::device`（六台设备共用一套），用例由 `testkit::device_suite!`
//! 摊开，这里只交出设备行与 P0 闭集。额外一条：见证正文还要被 OpenCL 自己的前端接受——
//! 宿主 `cc` 编得过它，不等于 `-cl-std=CL2.0` 也认它。

use super::*;

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

testkit::device_suite!(P0);

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
