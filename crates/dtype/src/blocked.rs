//! `qtype` / `mxtype` 共用的 **`const fn`** 访问器。
//!
//! 两个模块的行表都按判别值排序，`name`、块长与块数这几项逐字相同。宏只负责 trait 表达不了
//! 的部分——`const fn`（stable 还没有 const trait）；「按名解析」的唯一实现在
//! [`crate::Format::parse_name`]，两个枚举的 inherent `parse_name` 只是转调。`block_bytes` 与
//! `bits` 的来源不同（`QType` 显式存 `ggml-common.h` 的字节数，`MxType` 由元素位宽派生），
//! 留在各自文件里。
//!
//! 表名与行类型必须由调用点显式给出：宏按名字查 `SPECS` 会让「这个模块必须有一张叫 `SPECS`
//! 的表」变成隐式约定，改名后错误出现在展开处而不是定义处。

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
