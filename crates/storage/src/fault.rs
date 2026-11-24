//! "非法输入"的唯一构造点：上层只提供消息，发现的文件与行号由 `#[track_caller]` 自动标注。

use std::panic::Location;

use diagnostic::{Code, Error};

/// 构造 [`Code::Invalid`] 错误，并以调用点（`文件:行号`）作为上下文帧。
#[track_caller]
pub fn bad(message: impl AsRef<str>) -> Error {
    let site = Location::caller();
    Error::new(Code::Invalid, message.as_ref()).with(format!("{}:{}", site.file(), site.line()))
}

/// `Option` → `Result`：有值取之，无值按 [`bad`] 报错。
#[track_caller]
pub fn or_bad<T>(value: Option<T>, message: &str) -> diagnostic::Result<T> {
    value.ok_or_else(|| bad(message))
}
