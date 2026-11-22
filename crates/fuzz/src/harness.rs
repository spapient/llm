// libfuzzer harness 入口：只在 `engine` feature 下编译（默认测试不拉 libFuzzer 运行时）。
// 每个 `src/fuzz_*.rs` 入口 `include!` 本文件：适配器由调用方传给 `fuzz_harness`；契约核对在
// `llm_fuzz::Target` 一处实现，这里只做边界上的违约转 panic。
//
// libFuzzer 运行时只在 Unix 提供可链接入口；Windows/MSVC 下 harness 目标编一个空 `main`
// 桩，契约 sweep（`tests/parsers.rs`）仍在 Windows 上跑，只是没有覆盖引导运行时。
//
// 本文件被 15 个 harness 入口共享 include：这两个 import 只在 harness 目标里被使用，非 harness
// 目标在 --all-features 下同样满足 cfg，因此按 include 语义定向放开未用导入告警。
#[allow(unused_imports)]
#[cfg(all(unix, feature = "engine"))]
use libfuzzer_sys::fuzz_target;
#[cfg(all(unix, feature = "engine"))]
use llm_fuzz::Target;

/// harness 边界：契约核对违约时把 [`Finding`] 转成 panic，交给 libFuzzer 留下触发输入。
///
/// 契约库本身不出现 panic（生产 `src` 零 panic 门）；终止进程这一步只属于 harness 入口，
/// 全部目标共用这一处转换，不再各写一遍 `if let Err(...) { panic!(...) }`。
#[cfg(all(unix, feature = "engine"))]
#[allow(dead_code)]
fn panic_on_finding<T: Target>(target: &T, data: &[u8]) {
    if let Err(finding) = target.exercise(data) {
        panic!("{finding}");
    }
}

/// 把一个 `Target` 适配接成 libFuzzer 入口；违约由 [`panic_on_finding`] 转成 panic。
#[cfg(all(unix, feature = "engine"))]
#[allow(unused_macros)]
macro_rules! fuzz_harness {
    ($adapter:expr) => {
        fuzz_target!(|data: &[u8]| panic_on_finding(&$adapter, data));
    };
}

/// 非 Unix 桩：没有 libFuzzer 运行时，目标只需要一个能链接的入口；适配器仍被引用以保持
/// 驱动签名与 Unix 侧一致。
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
