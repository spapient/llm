//! 编译期表驱动宏：闭集由一张表展开变体、`ALL` 与映射，同一个清单不再手抄多遍。
//!
//! 三类闭集按映射轴区分：
//!
//! - [`code_set!`]：只映射到 `u16` 编号（响应状态）；
//! - [`name_set!`]：只映射到稳定短名（生成结局）；
//! - [`code_name_set!`]：编号与短名双轴（帧类型），再由 `ALL` 派生 `from_code`。
//!
//! 表是唯一事实源：加变体只改表一行，`ALL`、`code`、`as_str`、`Display` 随之展开，
//! 不会出现"发端编得出去、收端认不出"的漏映射。

/// 编号闭集：变体、`ALL` 与 `code()` 由同一张表展开。
macro_rules! code_set {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $( $(#[$vmeta:meta])* $variant:ident => $code:expr ),+ $(,)?
    }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// 全部成员，供遍历与用例。
            $vis const ALL: &'static [Self] = &[ $(Self::$variant),+ ];

            /// 对应的编码。
            $vis fn code(self) -> u16 {
                match self { $(Self::$variant => $code),+ }
            }
        }
    };
}

/// 短名闭集：变体、`ALL`、`as_str()` 与 `Display` 由同一张表展开。
macro_rules! name_set {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $( $(#[$vmeta:meta])* $variant:ident => $text:literal ),+ $(,)?
    }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// 全部成员，供遍历与用例。
            $vis const ALL: &'static [Self] = &[ $(Self::$variant),+ ];

            /// 稳定的短名：写进日志与诊断，不随翻译变化。
            $vis fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

/// 双轴闭集：编号 + 短名；`from_code` 由 `ALL` 驱动，不再手写第二张解码表。
macro_rules! code_name_set {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $( $(#[$vmeta:meta])* $variant:ident => $code:expr, $text:literal ),+ $(,)?
    }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// 全部成员，供遍历与用例。
            $vis const ALL: &'static [Self] = &[ $(Self::$variant),+ ];

            /// 线上编码。
            $vis fn code(self) -> u16 {
                match self { $(Self::$variant => $code),+ }
            }

            /// 稳定的短名：写进日志与诊断，不随翻译变化（线上用的是 [`Self::code`]）。
            $vis fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }

            /// 由编码还原；未知编码返回 `None`。
            ///
            /// 唯一事实源是 [`Self::code`]：这里的查找由 [`Self::ALL`] 驱动。
            $vis fn from_code(code: u16) -> Option<Self> {
                Self::ALL.iter().find(|item| item.code() == code).copied()
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}
