//! 片段签名声明的纯字符串规则：`/* cutile-sign: <prim>/<impl> */`（ADR 0027 §5）。
//!
//! 同一条规则有两个调用点，而它们不能共用编译产物：设备 crate 的 `build.rs` 在构建期拿它比对
//! 档案里的 `provides`（`crate::codegen`），`prim::Fragment::parse` 在装载期解析同一行。规则住在
//! 这里——两边都依赖本层——而不是各写一份。
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

/// 签名声明的标记，与 `prim::SIGN_MARKER` 同一个字符串。
pub const MARKER: &str = "cutile-sign:";

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
