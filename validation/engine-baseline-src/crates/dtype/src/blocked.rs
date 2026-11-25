//! `qtype` / `mxtype` 共用的 **`const fn`** 访问器。

macro_rules! blocked_accessors {
    ($specs:ident, $spec:ty) => {
        /// 布局元数据。
        #[must_use]
        pub const fn spec(self) -> $spec {
            $specs[self as usize]
        }

        /// 规范小写名，用于档案、注解与诊断；与 `Format::parse_name` 双向。
        #[must_use]
        pub const fn name(self) -> &'static str {
            self.spec().name
        }

        /// 每块元素数。
        #[must_use]
        pub const fn block_len(self) -> u32 {
            self.spec().block_len
        }

        /// `numel` 个元素按本格式需要的块数（不足一块按整块计）。
        #[must_use]
        pub const fn block_count(self, numel: u64) -> u64 {
            numel.div_ceil(self.block_len() as u64)
        }
    };
}

pub(crate) use blocked_accessors;
