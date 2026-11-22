//! 闭集枚举的单一生成点：`#[repr(u8)]` 判别值与 `from_code` 都从这一份表出。
//!
//! 线上码位稳定是容器格式的一部分，因此每个变体的码位只写一次——枚举本身与它的
//! 反向解析不再各维护一份 `match`，未知码统一显式拒绝、不静默回退。

/// 生成一个带稳定判别值的闭集枚举，以及它的 `from_code`。
///
/// 每个变体带 `= 码位`；`from_code` 按判别值反向解析，未知码返回 [`crate::bad`] 的
/// `Invalid` 错误。变体上的文档注释原样保留。
macro_rules! closed_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:expr,
            )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant = $code,
            )+
        }

        impl $name {
            /// 从码位解析；未知码显式报错，不静默回退。
            pub(crate) fn from_code(code: u8) -> diagnostic::Result<Self> {
                match code {
                    $($code => Ok(Self::$variant),)+
                    other => Err(crate::bad(format!(
                        "未知 {} 码 {other}",
                        stringify!($name)
                    ))),
                }
            }
        }
    };
}
