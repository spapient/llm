// 片段签名声明的纯字符串规则：crate 装载期与 `build.rs` 编译期经同一份文本共用，
// 解析规则只有一份事实源（ADR 0027 §5）。
//
// 本文件不认识原语闭集、也不构造诊断错误：闭集比对、`impl` 非空判定与错误格式化
// 留给各自调用方。`build.rs` 用 `include!("src/signature.rs")` 拉入这份源码，
// 不依赖正在构建的 crate 的编译产物。注释用普通行注释：被 include 进构建脚本后，
// 内层文档注释（`//!`）与项文档注释（`///`）在那边不合法。

// 签名声明标记，与 `lib.rs` 里对外暴露的 `SIGN_MARKER` 同一个字符串。
pub(crate) const MARKER: &str = "cutile-sign:";

// 剥掉 `/* */` 注释外壳与 `MARKER`，得到声明正文 `<prim>/<impl>`；形态不符返回 `None`。
pub(crate) fn strip_marker(first: &str) -> Option<&str> {
    first
        .trim()
        .strip_prefix("/*")?
        .strip_suffix("*/")
        .map(str::trim)?
        .strip_prefix(MARKER)
        .map(str::trim)
}

// 把 `<prim>/<impl>` 按 `/` 切成两半；缺 `/` 返回 `None`。
pub(crate) fn split_slash(decl: &str) -> Option<(&str, &str)> {
    decl.split_once('/')
}
