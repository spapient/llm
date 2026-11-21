use super::*;
use crate::toolchain;

/// OpenCL 前端表行：拒绝用例要拿它的参数过一遍 OpenCL C 不支持的构造。
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
    // 函数指针是 OpenCL C 明确排除的构造（标准与两个 clang 前端都拒绝）；`double` 依赖
    // FP64 扩展与前端默认配置，不同构建可能收下，不能当稳定的拒绝夹具。同一份源码在 host
    // C99 里合法，下面先跑 host 编译器把这个前提钉住。
    std::fs::write(
        &source,
        "int add(int a, int b) { return a + b; }\nint witness_fnptr(int (*fp)(int, int)) { return fp(1, 2); }\n",
    )
    .expect("写见证");

    let rejected =
        parse_one(opencl(), opencl().check, &source).expect_err("OpenCL C 不允许函数指针");
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

/// 昇腾前端的三条判据缺一条，解析的就不是同一门语言：`-xcce` 决定设备语言、`dav-c220` 决定
/// 核芯（`id = "910b"`）、`-fsyntax-only` 决定只解析不产出。
#[test]
fn the_ascend_front_is_ccec_for_the_910b_core() {
    let entry = GATES
        .iter()
        .find(|entry| entry.emit == Emit::Ascendc)
        .expect("AscendC 语法前端已登记");
    assert_eq!(entry.tool, ["ccec"], "昇腾的前端是 CANN 的 ccec");
    for flag in ["-xcce", "--cce-aicore-algorithm=dav-c220", "-fsyntax-only"] {
        assert!(
            entry.check.contains(&flag),
            "缺 `{flag}`：{:?}",
            entry.check
        );
    }
}

/// 前端是档案声明的那一个，不是宿主 C 编译器：本机没有 `ccec` 时门只能是 `Absent`（且指出缺的
/// 是哪个程序），拿 `cc` 去解析只会把「没装 CANN」报成一条 C 语法错误——那是 `Rejected`。
#[test]
fn a_missing_declared_front_is_absent_not_rejected() {
    if toolchain::probe("ccec").is_some() {
        eprintln!("skip: 本机装了 `ccec`，缺席分支无从构造");
        return;
    }
    let ascend = GATES
        .iter()
        .find(|entry| entry.emit == Emit::Ascendc)
        .expect("AscendC 语法前端已登记");
    assert!(front(ascend).is_none(), "`ccec` 不在本机时解析器就不在场");
    let profile = cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("ascend")
        .cloned()
        .expect("ascend 档案");
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-gate-ascend"));
    match language_gate(&profile, &dirs) {
        LanguageGate::Absent { reason } => {
            assert!(reason.contains("ccec"), "缺席要点名缺哪个程序：{reason}");
            assert!(reason.contains("不在本机"), "{reason}");
        }
        other => panic!("缺 `ccec` 时门不该是 `{other}`"),
    }
}
