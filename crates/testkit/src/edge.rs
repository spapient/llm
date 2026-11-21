//! 一条成员依赖边，以及成员名的规范化。
//!
//! 两侧事实源（`docs/ARCHITECTURE.md` 与 `cargo metadata`）用的都是**短名**（`op`，不是
//! `llm-op`），所以包名规范化与 [`Edge`] 同处：名字的写法只有这一个定义，改前缀不会漏掉一侧。

/// 成员之间的一条依赖边。
///
/// `Ord` 按 `(from, to, dev)` 排：门禁用集合差报告多出/缺失的边，输出顺序因此稳定可 diff。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Edge {
    /// 依赖方（下游）。
    pub(crate) from: String,
    /// 被依赖方。
    pub(crate) to: String,
    /// `+dev` 边：只服务测试与夹具，不进任何产品二进制。
    pub(crate) dev: bool,
}

/// 包名 → 成员短名：`llm-op` 与文档里的 `op` 指同一个成员。
pub(crate) fn short(name: &str) -> String {
    let name = name.trim();
    name.strip_prefix("llm-").unwrap_or(name).to_string()
}
