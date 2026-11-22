// libfuzzer harness 入口：只在 `engine` feature 下编译（默认测试不拉 libFuzzer 运行时）。
// 每个 `src/fuzz_*.rs` 入口 `include!` 本文件：适配器由调用方传进 `fuzz_harness`，三契约的核对
// 只在 `Target::exercise` 一处，这里仅把边界上的违约转成 panic。
//
// libFuzzer 运行时只在 Unix 提供可链接入口；Windows/MSVC 下目标编一个空 `main` 桩，契约 sweep
// （`tests/parsers.rs`）仍在 Windows 上跑，只是没有覆盖引导运行时。
//
// 本文件被 15 个入口共享 include：`fuzz_target` 只在 harness 目标里被用到，按 include 语义
// 定向放开未用导入告警。

// `drivers.rs` 以 `crate::Target` 引用判据 trait：入口的 crate 根必须存在 `Target` 这个名字，
// 由这里补上；入口文件自身保持零导入。
use llm_fuzz::Target;

#[allow(unused_imports)]
#[cfg(all(unix, feature = "engine"))]
use libfuzzer_sys::fuzz_target;

/// 把一个 `Target` 适配接成 libFuzzer 入口；违约时 panic，由 libFuzzer 留下触发输入。
///
/// 契约库本身不 panic，终止进程只属于这一处 harness 边界：`exercise` 返回的 [`Finding`] 带着
/// 目标名、用例号与输入，panic 消息因此直接可读。trait 在闭包内按需引入，入口文件就不必再写
/// 一遍导入。
#[cfg(all(unix, feature = "engine"))]
#[allow(unused_macros)]
macro_rules! fuzz_harness {
    ($adapter:expr) => {
        fuzz_target!(|input: &[u8]| {
            use llm_fuzz::Target as _;
            if let Err(finding) = $adapter.exercise(input) {
                panic!("{finding}");
            }
        });
    };
}

/// 非 Unix 桩：没有 libFuzzer 运行时，目标只需要一个能链接的入口；适配器仍被引用以保持驱动
/// 签名与 Unix 侧一致。
#[cfg(all(not(unix), feature = "engine"))]
#[allow(unused_macros)]
macro_rules! fuzz_harness {
    ($adapter:expr) => {
        #[allow(dead_code)]
        fn main() {
            let _ = &$adapter;
        }
    };
}
