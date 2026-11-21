//! 打包面板的暂存区：按线程复用，用完即还。
//!
//! 打包出来的是「整段 B 面板」，`n = k = 4096` 时是 `4096 × 4096 × 4 B = 67 MB`。每次调用现分配，
//! 光 `calloc` 与首次触页就要几十毫秒，而 decode 与 prefill 都是**同一线程反复调**同一个几何，
//! 这块内存每步都用得上，却每步被丢掉一次。这里让每个调用线程留一块只增不减的缓冲：
//!
//! - **只按线程共享，不跨线程**：并发调用各占一个线程私有的缓冲，不存在两次调用同时写同一块；
//!   池里的 worker 只是拿到主线程的裸指针去写自己的分片，收回发生在 [`with_panels`] 返回之前。
//! - **只扩容，不缩小**：几何变了就扩容一次，之后小几何沿用它前面的部分。常驻量等于本线程见过的
//!   最大面板，与「每次分配再释放」的峰值一致（面板本来就要整块驻留）。
//! - **重入退化为临时缓冲**：若某个打包路径在一次借用期间又调 [`with_panels`]，`RefCell` 已经在
//!   借出状态，这里退化成一次性 `Vec`——宁可多分配一次，也不 panic。

use std::cell::RefCell;

thread_local! {
    /// 本线程的面板暂存区；长度只增不减。
    static PANELS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// 借一块长度 `len` 的 f32 暂存区（不足则扩容并清零新增部分），返回 `f` 的结果。
///
/// 清零只针对**新增**的那一段：面板的每个字节都会被打包阶段覆写，但列残尾之外的补零区
/// （`n % NR` 的补齐列）与 `edge` 的读路径依赖 0，复用旧内容时必须在扩容时清一次。
pub(super) fn with_panels<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PANELS.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < len {
                buf.resize(len, 0.0);
            }
            f(&mut buf[..len])
        }
        // 重入（同一次调用里又借一次）：这一次用临时缓冲，语义不变。
        Err(_) => f(&mut vec![0.0f32; len]),
    })
}

#[cfg(test)]
#[path = "../../tests/scratch_tests.rs"]
mod tests;
