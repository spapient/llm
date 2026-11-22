//! 片段签名声明的纯字符串规则：`/* cutile-sign: <prim>/<impl> */`（ADR 0027 §5），
//! 与片段语料的三个保留词（标记、见证文件名、尾块文件名）。
//!
//! 同一条规则有两个调用点，而它们不能共用编译产物：本 crate 的 `build.rs` 在**构建期**拿它比对
//! 档案里的 `provides`（扫描语料目录要区分"片段 / 见证 / 尾块"），`Fragment::parse` 在**装载期**
//! 解析同一行。文件住在 `src/` 里的无依赖模块，构建脚本用 `#[path]` 把它编进来——于是构建期与
//! 装载期读的是同一份文本，而不是两份互相漂移的抄写。
//!
//! 本模块不认识原语闭集，也不构造诊断：闭集比对、`impl` 非空判定与错误格式化留给调用方。
//!
//! # 例子
//!
//! ```
//! use llm_prof::sign;
//!
//! let declared = sign::strip_marker("/* cutile-sign: dot/fma */").expect("这是声明");
//! assert_eq!(sign::split_slash(declared), Some(("dot", "fma")));
//! assert_eq!(sign::strip_marker("/* 别的注释 */"), None);
//! ```

/// 签名声明的标记：片段第一行 `/* cutile-sign: <prim>/<impl> */` 里的 `cutile-sign:`。
pub const MARKER: &str = "cutile-sign:";

/// 见证源码的文件名（不含扩展名）。
pub const WITNESS_STEM: &str = "witness";

/// 写回尾块片段的文件名（不含扩展名）：一份文本装六种形式。
///
/// 它的签名声明不带 `/`（`/* cutile-sign: epilogue */`）：原语片段的声明回答"这一段实现哪条
/// 签名"，而一份尾块文本实现六种形式，逐个形式写声明会让同一份文件出现六条。
pub const EPILOGUE_STEM: &str = "epilogue";

/// 剥掉 `/* */` 外壳与 [`MARKER`]，得到声明正文 `<prim>/<impl>`；形态不符返回 `None`。
pub fn strip_marker(first: &str) -> Option<&str> {
    first
        .trim()
        .strip_prefix("/*")?
        .strip_suffix("*/")
        .map(str::trim)?
        .strip_prefix(MARKER)
        .map(str::trim)
}

/// 把 `<prim>/<impl>` 按 `/` 切成两半；缺 `/` 返回 `None`。
pub fn split_slash(decl: &str) -> Option<(&str, &str)> {
    decl.split_once('/')
}
