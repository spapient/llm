//! 闭集枚举宏：把变体、名字表、`Display` 与固有 `parse` 收进同一张表。

/// 声明一个闭集枚举：变体、`ALL`、名字表、`Display` 与固有 `parse` 由同一张表展开，
/// 名字因此只有一份（与 `spec::atom::vocabulary!` 同构，多出「附加列」一臂）。
///
/// 变体顺序即 `ALL` 顺序、即位集下标（`idx = self as usize`）；`as_str` 按 `idx` 查同序名字表，
/// 展开的查找与渲染都是表驱动（ccn = 1）。带附加列的形式（`ext: Cuda => ("cuda", "cu")`）
/// 额外生成一个 `const fn ext(self) -> &'static str`，供 `Emit::ext` 这类"名字之外的列"用。
///
/// ```ignore
/// closed_enum! {
///     /// 输出语言闭集。
///     Emit {
///         ext: Cuda => ("cuda", "cu"), Metal => ("metal", "metal"),
///     }
///     Skeleton {
///         Elementwise => "elementwise", Reduce => "reduce",
///     }
/// }
/// ```
#[macro_export]
macro_rules! closed_enum {
    (
        $(#[$doc:meta])*
        $name:ident { $extra:ident: $($variant:ident => ($text:literal, $value:literal)),* $(,)? }
    ) => {
        $crate::closed_enum! { $(#[$doc])* $name { $($variant => $text),* } }
        impl $name {
            /// 附加列取值（按闭集顺序）。
            #[must_use]
            pub const fn $extra(self) -> &'static str {
                const VALUES: &[&str] = &[$($value),*];
                // `Closed::idx` 是 trait 方法，不能出现在 `const fn` 里；闭集是 fieldless enum，
                // 判别值就是 `ALL` 的下标，`as usize` 与 `idx()` 同值。
                VALUES[self as usize]
            }
        }
    };
    (
        $(#[$doc:meta])*
        $name:ident { $($variant:ident => $text:literal),* $(,)? }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        // 变体的文档就是它的跨层名字：名字表只有一份，文档不该是第二份手抄。
        pub enum $name { $(#[doc = $text] $variant,)* }

        impl $crate::atom::Closed for $name {
            const ALL: &'static [Self] = &[$(Self::$variant),*];
            fn as_str(self) -> &'static str {
                const NAMES: &[&str] = &[$($text),*];
                NAMES[self.idx()]
            }
            fn idx(self) -> usize {
                self as usize
            }
        }
        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(self.as_str())
            }
        }
        impl $name {
            /// 闭集全量（见 [`crate::atom::Closed::ALL`]）。
            pub const ALL: &'static [Self] = <Self as $crate::atom::Closed>::ALL;
            /// 跨层名字（见 [`crate::atom::Closed::as_str`]）。
            #[must_use]
            pub fn as_str(self) -> &'static str {
                <Self as $crate::atom::Closed>::as_str(self)
            }
            /// 按名字解析（见 [`crate::atom::Closed::parse`]）。
            #[must_use]
            pub fn parse(name: &str) -> Option<Self> {
                <Self as $crate::atom::Closed>::parse(name)
            }
        }
    };
}
