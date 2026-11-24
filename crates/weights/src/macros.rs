//! 闭集枚举的单一生成点：`#[repr(u8)]` 判别值、码位访问器、反向解析与短名同出一处声明。
//!
//! 线上码位稳定是容器格式的一部分，因此每个变体的码位只写一次——枚举本身与它的
//! 反向解析不再各维护一份 `match`，未知码统一显式拒绝、不静默回退。带短名的枚举
//! 还把账本/文本里的字符串写法一并生成，短名与码位同源、不会漂移。
//!
//! 形态选择：
//! - 默认（可带 `=> "短名"`）：枚举 + `from_code`（未知码 `Invalid`），带短名时另生成 `as_str`；
//! - `@named`：只有枚举与 `as_str`，没有线上数值码的反向解析（账本列）；
//! - `@coded` / `@coded_unsupported`：另生成 `code()` 访问器，供线上读写双端使用；
//!   后者把未知码报成 `Unsupported`——读到本端不认识的格式特性，与"字节损坏"（`Invalid`）
//!   不是一回事。

/// 生成一个带稳定判别值的闭集枚举，以及它的 `from_code`；带 `=> "短名"` 时再生成 `as_str`。
///
/// 每个变体带 `= 码位`；`from_code` 按判别值反向解析，未知码返回 [`crate::bad`] 的
/// `Invalid` 错误。变体上的文档注释原样保留。首 token 选 `@named` / `@coded` /
/// `@coded_unsupported` 可换形态，见模块文档。
macro_rules! closed_enum {
    // 带短名：枚举 + from_code + as_str。
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal => $sname:literal,
            )+
        }
    ) => {
        closed_enum!(@str_only
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
            { $( $variant => $sname, )+ }
        );
        closed_enum!(@from_code $name { $( $code => $variant, )+ });
    };

    // 无短名：只有枚举与 from_code。
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal,
            )+
        }
    ) => {
        closed_enum!(@def
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
        closed_enum!(@from_code $name { $( $code => $variant, )+ });
    };

    // —— 内部臂：枚举定义 / 反向解析 / 字符串写法，各只写一份 ——
    (
        @def
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident = $code:literal, )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        $vis enum $name {
            $( $(#[$variant_meta])* $variant = $code, )+
        }
    };

    (
        @from_code
        $name:ident {
            $( $code:literal => $variant:ident, )+
        }
    ) => {
        impl $name {
            /// 从码位解析；未知码显式报错，不静默回退。
            pub(crate) fn from_code(code: u8) -> diagnostic::Result<Self> {
                match code {
                    $( $code => Ok(Self::$variant), )+
                    other => Err(crate::bad(format!(
                        "未知 {} 码 {other}",
                        stringify!($name)
                    ))),
                }
            }
        }
    };

    (
        @as_str
        $name:ident {
            $( $variant:ident => $sname:literal, )+
        }
    ) => {
        impl $name {
            /// 账本/线上的字符串写法（闭集，码位稳定）。
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $sname, )+
                }
            }
        }
    };

    // —— 内部臂：只带短名（枚举定义 + as_str），不带 from_code；供外层两个臂共用 ——
    (
        @str_only
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident = $code:literal, )+
        }
        { $( $vs:ident => $sname:literal, )+ }
    ) => {
        closed_enum!(@def
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
        closed_enum!(@as_str $name { $( $vs => $sname, )+ });
    };

    // 只带短名：没有线上数值码的反向解析，只有枚举定义与 `as_str`；调用首 token 写 `@named`。
    (
        @named
        $(#[$nm:meta])*
        $v:vis enum $e:ident {
            $( $(#[$vm:meta])* $var:ident = $c:literal => $sn:literal, )+
        }
    ) => {
        closed_enum!(@str_only
            $(#[$nm])*
            $v enum $e {
                $( $(#[$vm])* $var = $c, )+
            }
            { $( $var => $sn, )+ }
        );
    };

    // —— 读写双端形态：另生成 `code()`，写端不再 `as u8` 散写 ——

    // `@coded`：枚举 + code() + from_code（未知码 `Invalid`）。
    (
        @coded
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal,
            )+
        }
    ) => {
        closed_enum!(@def
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
        closed_enum!(@code $name);
        closed_enum!(@from_code $name { $( $code => $variant, )+ });
    };

    // `@coded_unsupported`：与 `@coded` 同形，未知码报 `Unsupported`。
    (
        @coded_unsupported
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $code:literal,
            )+
        }
    ) => {
        closed_enum!(@def
            $(#[$meta])*
            $vis enum $name {
                $( $(#[$variant_meta])* $variant = $code, )+
            }
        );
        closed_enum!(@code $name);
        closed_enum!(@from_code_unsupported $name { $( $code => $variant, )+ });
    };

    (
        @code $name:ident
    ) => {
        impl $name {
            /// 线上码位：与判别值同源，不存在第二份映射表。
            pub(crate) const fn code(self) -> u8 {
                self as u8
            }
        }
    };

    (
        @from_code_unsupported
        $name:ident {
            $( $code:literal => $variant:ident, )+
        }
    ) => {
        impl $name {
            /// 从码位解析；未知码按"本端不支持"显式报错，不静默回退。
            pub(crate) fn from_code(code: u8) -> diagnostic::Result<Self> {
                match code {
                    $( $code => Ok(Self::$variant), )+
                    other => Err(diagnostic::Error::new(
                        diagnostic::Code::Unsupported,
                        format!("未知 {} 码 {other}", stringify!($name)),
                    )),
                }
            }
        }
    };
}
