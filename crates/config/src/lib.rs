//! 配置层的占位成员。
//!
//! 该 crate 由 `cargo new --lib crates/config` 生成，当前只有一个 `add` 占位函数；
//! 保留它是为了在 workspace 里给后续的"配置化能力协商"留位置（见 DESIGN §1.1 对
//! 旧工程 `llm.u` 的 `crates/config` 的引用）。真正实现时再按 `cutile` 的模式补齐。

/// 占位函数：验证 workspace 的元数据与 lint 继承是否生效。
///
/// # Examples
///
/// ```
/// assert_eq!(config::add(2, 2), 4);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(add(2, 2), 4);
    }
}
