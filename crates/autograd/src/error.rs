//! 本层的错误构造：类别只有两种，站点字符串由调用方给出。
//!
//! 反传调度里没有"缺能力"：`Unsupported` / `Missing` 都属于别的层（缺设备、缺声明），
//! 这里只会出现"调用顺序或声明写错"（`Invalid`）与"账本自身不一致"（`Internal`）。

use diagnostic::{Code, Error};

/// 调用顺序或声明写错了：引用不存在的变量、乱序结算、给不进图的节点要梯度。
pub(crate) fn invalid(site: &'static str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(site)
}

/// 本层自己的不变量被破坏：扇出计数已经减到 0 却还在收贡献。
pub(crate) fn internal(site: &'static str, message: impl Into<String>) -> Error {
    Error::new(Code::Internal, message).with(site)
}
