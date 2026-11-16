// 共用 libfuzzer harness 入口。
//
// 每个测试二进制 `include!` 这份定义：适配器由调用方传给 `fuzz_harness`；契约核对与违约转
// panic 的逻辑只在 `llm_fuzz::Target` 一处实现，harness 文件不再各抄一遍闭包。

use libfuzzer_sys::fuzz_target;
use llm_fuzz::Target;

/// 把一个 `Target` 适配接成 libFuzzer 入口；违约由 `exercise_or_panic` 转成 panic，交给
/// libFuzzer 留下触发输入作为复现样本。
macro_rules! fuzz_harness {
    ($adapter:expr) => {
        fuzz_target!(|data: &[u8]| $adapter.exercise_or_panic(data));
    };
}
