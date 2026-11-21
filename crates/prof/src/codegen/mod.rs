//! 构建期代码生成：一个设备目录 → 一份编译期表（`$OUT_DIR/tables.rs`）。
//!
//! 三件事分开住，谁都不读别人的内部：扫描与校验在 [`scan`]（为什么生成器住在本层、四条构建期
//! 校验、重跑条件的道理都写在那里），表文本在 `table`，设备镜像在 `image`。对外只有一行
//! [`device`]——设备 crate 的 `build.rs` 就调它。
//!
//! 本模块只在 `build` feature 下编译，运行期二进制里没有它。

mod image;
mod scan;
mod table;

pub use scan::device;
