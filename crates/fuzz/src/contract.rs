//! 三条契约的核对入口：确定性遍历 [`sweep`] 与 libfuzzer harness 从这里取同一份判据，
//! 拒绝判据不会在两处悄悄漂移。

use crate::Target;
use crate::finding::{Behaviour, Finding};
use crate::seed::Seed;

/// 契约一：任何输入都必须被解析成结论，而不是打死宿主。
const PANIC: &str = "不 panic：任何输入都必须被解析成结论，而不是打死宿主";
/// 契约三：拒绝时必须给出非空消息。
const EMPTY_MESSAGE: &str = "拒绝时必须给出非空消息";
/// 契约二：同一输入跑两次必须给出同一个结论。
const STABILITY: &str = "同一输入必须给出同一个结论";

/// 跑一次并把 `Result` 归一成 [`Behaviour`]，让"接受/拒绝"的比较不依赖 `Error` 的内部结构。
///
/// panic 也在这里归一：入口是不受信任输入的边界，`catch_unwind` 是唯一能让"第 7 条输入打死
/// 进程"变成"第 7 条输入是一条证据"的手段。`AssertUnwindSafe` 的前提是 [`Target::run`] 在
/// panic 之后不会被观察到半截状态——它是 `&self` 的纯判定，没有跨用例的状态需要恢复。
pub fn judge(target: &dyn Target, input: &[u8]) -> Behaviour {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| target.run(input))) {
        Ok(Ok(())) => Behaviour::Accepted,
        Ok(Err(error)) => Behaviour::Rejected {
            code: error.code(),
            message: error.message().to_string(),
        },
        Err(payload) => Behaviour::Panicked {
            message: panic_message(&payload),
        },
    }
}

/// 核对单条用例的三条契约；违约返回第一条证据。
///
/// 顺序就是报告的优先级：先 panic（进程已死，其余判据没有意义），再空消息（拒绝了却说不清
/// 为什么），最后稳定性（同输入两次结论不一致）。harness 与 [`sweep`] 都走这一条，判据只有
/// 一处实现。
pub fn check(target: &dyn Target, case: u64, input: Vec<u8>) -> Result<(), Finding> {
    let name = target.name().to_string();
    let first = judge(target, &input);
    if let Behaviour::Panicked { .. } = first {
        return Err(Finding::new(&name, case, input, first, PANIC));
    }
    if let Behaviour::Rejected { message, .. } = &first
        && message.trim().is_empty()
    {
        return Err(Finding::new(&name, case, input, first, EMPTY_MESSAGE));
    }
    let second = judge(target, &input);
    if first != second {
        return Err(Finding::new(&name, case, input, second, STABILITY));
    }
    Ok(())
}

/// 跑 `cases` 个确定性用例；全部符合契约返回 `Ok(())`，否则返回第一条证据。
///
/// `cases` 为 0 时不做任何事：遍历范围由调用方决定，本函数不猜"默认跑多少个"。
pub fn sweep(target: &dyn Target, seed: &Seed, cases: u64) -> Result<(), Finding> {
    (0..cases).try_for_each(|case| check(target, case, seed.derive(case)))
}

/// 跑 `cases` 个确定性用例，收齐**全部**违约证据。
///
/// [`sweep`] 只带第一条证据便于快速定位；需要一份完整回归清单（CI 汇总、按契约分组统计）
/// 时用这条，遍历成本与 [`sweep`] 相同，只是不提前退出。
pub fn sweep_all(target: &dyn Target, seed: &Seed, cases: u64) -> Vec<Finding> {
    (0..cases)
        .filter_map(|case| check(target, case, seed.derive(case)).err())
        .collect()
}

/// 从 panic payload 取回说明：`panic!("…")` 给 `&str`，`panic!("{x}")` 给 `String`，
/// `panic_any` 可能给任意值。常见数值/字节载荷按 `Debug` 渲染；都取不到时留占位说明，
/// 不留空白。
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(bytes) = payload.downcast_ref::<&[u8]>() {
        return format!("{bytes:?}");
    }
    if let Some(value) = [
        debug_payload::<u8>(payload),
        debug_payload::<u32>(payload),
        debug_payload::<u64>(payload),
        debug_payload::<usize>(payload),
        debug_payload::<bool>(payload),
    ]
    .into_iter()
    .flatten()
    .next()
    {
        return value;
    }
    "panic payload 没有携带字符串说明".to_string()
}

/// 常见数值载荷按 `Debug` 取回；类型不符返回 `None`。
fn debug_payload<T: std::fmt::Debug + 'static>(
    payload: &Box<dyn std::any::Any + Send>,
) -> Option<String> {
    payload
        .downcast_ref::<T>()
        .map(|value| format!("{value:?}"))
}
