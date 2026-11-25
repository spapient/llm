//! 门禁诊断错误的共享构造点：各子模块只给站点短上下文，前缀在这里唯一拼出。

use diagnostic::{Code, Error};

/// 构造一条挂在 `tools::<module>::<ctx>` 站点下的 [`Code::Invalid`] 错误。
///
/// `module` 是子模块名（`bench` / `card` / `regress`），`ctx` 是出错的类型或函数短名；
/// 站点前缀集中在此，避免三处逐字重复同一套 `Error::new(..).with(..)`。
pub(crate) fn invalid(module: &str, ctx: &str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("tools::{module}::{ctx}"))
}
