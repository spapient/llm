use super::*;
use crate::toolchain;

/// OpenCL 前端表行：拒绝用例要拿它的参数过一遍裸 `double`。
fn opencl() -> &'static GateSpec {
    GATES
        .iter()
        .find(|entry| entry.emit == Emit::Opencl)
        .expect("OpenCL 语法前端已登记")
}

#[test]
fn a_witness_its_own_language_rejects_is_a_defect() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let Some(toolchain) = toolchain::detect() else {
        return;
    };
    if clang().is_none() {
        eprintln!("skip: 未探测到 clang（`-x cl` 语法检查需要它），本用例未运行");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutile-language-gate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("临时目录");
    let source = dir.join("witness.cl");
    std::fs::write(
        &source,
        "int witness_double(float *out) { double x = 2.0; out[0] = (float)x; return 1; }\n",
    )
    .expect("写见证");

    let rejected = parse_one(opencl(), &source).expect_err("OpenCL C 不允许裸 double");
    assert!(rejected.to_string().contains("witness.cl"), "{rejected}");
    assert!(rejected.to_string().contains("-x cl"), "{rejected}");

    let host = dir.join("witness.c");
    std::fs::copy(&source, &host).expect("复制");
    let status = std::process::Command::new(&toolchain.command)
        .args(["-std=c99", "-fsyntax-only"])
        .arg(&host)
        .status()
        .expect("跑 host C 编译器");
    assert!(status.success(), "前提：同一份源码在 host C 里合法");
    let _ = std::fs::remove_dir_all(&dir);
}
