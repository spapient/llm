//! `cutile::device` 的用例：设备枚举与能力查询。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;
use crate::fixtures::catalogue;

/// 一条 mock 后端（SPIRV，本机必然缺席）：只用来验注册表的「去重 + 按 emit 分派」机制，不碰任何
/// 厂商工具链。真机 Metal / CUDA 后端在 `engine`（L6），其 Codegen / Driver 测试也搬到了那边。
struct Mock;

impl Codegen for Mock {
    fn emit(&self) -> Emit {
        Emit::Spirv
    }
    fn program(
        &self,
        _: &Profile,
        _: &Dirs,
        _: &[Selected<'_>],
        _: Scope,
    ) -> diagnostic::Result<String> {
        Ok(String::new())
    }
}

impl Driver for Mock {
    fn available(&self, _: &Profile) -> bool {
        false
    }
    fn run(
        &self,
        _: &Profile,
        _: &Dirs,
        _: &str,
        _: usize,
        _: Scope,
    ) -> diagnostic::Result<Readback> {
        Ok((Vec::new(), Vec::new()))
    }
    fn toolchain(&self) -> Option<crate::toolchain::Toolchain> {
        None
    }
}

impl Backend for Mock {}

static MOCK: Mock = Mock;

/// 生成 host C 参考后端的见证源码（设备文本 + 设备档案的 host 自检文本）。默认只断言非空；
/// `CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本写到 `/tmp/cutile-golden` 供人工比对。
/// Metal / CUDA 的设备文本由 `engine` 生成，不再从 cutile 的注册表里取。
#[test]
fn golden_capture() {
    let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
    let dir = std::path::PathBuf::from("/tmp/cutile-golden");
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-golden-work"));
    let catalogue = catalogue();

    let cpu = catalogue.profile("cpu").cloned().expect("cpu 档案");
    let front = spec(Emit::C).expect("c 前端");
    let selected = crate::witness::declared_witnesses(&cpu).expect("见证");
    let text = front
        .program(&cpu, &dirs, &selected, Scope::Device)
        .expect("可生成");
    assert!(!text.is_empty(), "cpu 生成文本不能为空");
    if capture {
        std::fs::create_dir_all(&dir).expect("目录");
        std::fs::write(dir.join("witness_c.txt"), text).expect("写");
    }

    let metal = catalogue.profile("metal").cloned().expect("metal 档案");
    let selected = crate::witness::declared_witnesses(&metal).expect("见证");
    let host = front
        .program(&metal, &dirs, &selected, Scope::Host)
        .expect("可生成");
    assert!(!host.is_empty(), "metal 档案的 host 自检文本不能为空");
    if capture {
        std::fs::create_dir_all(&dir).expect("目录");
        std::fs::write(dir.join("witness_metal_host.txt"), host).expect("写");
    }
}

#[test]
fn scratch_directories_do_not_collide() {
    let dirs = Dirs::under(std::env::temp_dir());
    let first = crate::runner::driver::scratch(&dirs, "gate");
    let second = crate::runner::driver::scratch(&dirs, "gate");
    assert_ne!(first, second);
    assert_eq!(first.parent(), Some(dirs.work()));
}

/// 执行后端注册表：宿主 C 参考后端种子在场、每个 `emit` 恰好一行、同 emit 重复注册被去重（
/// `spec(..)` 命中第一行，后一行静默失效），表不许被清空。
#[test]
fn specs_register_each_execution_frontend_once() {
    assert!(spec(Emit::C).is_some(), "宿主 C 参考后端必须种子注册");
    crate::device::register(&MOCK);
    crate::device::register(&MOCK);

    let mut seen = std::collections::BTreeSet::new();
    for entry in specs() {
        assert!(
            seen.insert(entry.emit()),
            "执行前端 `{}` 在注册表里重复登记：后一行会被 `spec(..)` 静默忽略",
            entry.emit().as_str()
        );
        assert!(
            spec(entry.emit()).is_some(),
            "表内 `{}` 的查找行与登记表分家",
            entry.emit().as_str()
        );
    }
    assert!(!seen.is_empty(), "执行前端注册表不得被清空");
}

#[test]
fn scratch_action_cleans_up_on_success_and_failure() {
    let root = std::env::temp_dir().join(format!("shard-scratch-cleanup-{}", std::process::id()));
    let dirs = Dirs::under(&root);
    let mut success_path = None;
    let value = crate::runner::driver::with_scratch(&dirs, "success", |dir| {
        success_path = Some(dir.to_path_buf());
        std::fs::create_dir_all(dir).expect("临时目录");
        Ok(7)
    });
    assert_eq!(value.expect("成功值"), 7);
    assert!(!success_path.expect("路径").exists(), "成功后必须清理");

    let mut failure_path = None;
    let result: diagnostic::Result<()> =
        crate::runner::driver::with_scratch(&dirs, "failure", |dir| {
            failure_path = Some(dir.to_path_buf());
            std::fs::create_dir_all(dir).expect("临时目录");
            Err(diagnostic::Error::new(
                diagnostic::Code::Invalid,
                "故意失败",
            ))
        });
    assert!(result.is_err());
    assert!(!failure_path.expect("路径").exists(), "失败后也必须清理");
    let _ = std::fs::remove_dir_all(root);
}
