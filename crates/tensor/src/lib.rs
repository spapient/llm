//! 形状描述：一个 [`Shape`] 记各轴长度，[`DType`] 定元素宽度。
//!
//! 本层只回答"数据长什么样、占几个字节"，不持有元素、不重排布局、不执行。两件事各只有一处定义：
//!
//! - **[`Shape`]**：各轴长度。`numel` 是饱和乘积（[`product`]），`element_bytes` 由
//!   [`DType::bytes_for`] 给出——亚字节精度按位向上取整，所以 `U4` 的字节数小于 `numel`。
//! - **[`DType`]**：元素类型闭集。唯一事实源在 L1 `dtype`（24 变体 + 能力分类），这里只再导出。
//!
//! **为什么没有 `Tensor`**：本仓的抽象单元是算子契约 + 设备 Fragment + 执行缝（ADR 0053），
//! 张量只做无执行的元数据描述。带 stride 的视图代数（broadcast / narrow / permute / flip /
//! squeeze / diagonal）曾在本 crate 实现 1247 行，但全仓零个计算路径使用它：51 处
//! `tensor::` 调用只取 `Shape` / `DType` / `Order`，唯一的 `Tensor` 调用点是 `cli status`
//! 打印一个硬编码的 2×2 矩阵。删掉它是"去死代码"，不是"能力减少"——真实调用面一行未动，
//! 逐算子去向与验证用例见 `docs/lld.md` v3.0.58 的能力映射表。
//!
//! **拥有**：`Shape`、`AxisLen`、`product` 与 `DType` 的再导出。
//! **不拥有**：元素类型闭集（L1 `dtype`）、步长与分区（算子契约 / DSL）、字节视图（`weights`
//! 描述符）、布局搬运（`storage`）、设备句柄（ADR 0025）。
//! **依赖**：`dtype`。**被依赖**：`storage`、`weights`、`nn`、`train`、`engine`、`autograd`、
//! `snapshot`、`testkit`、`benchmarks`、`cli`、`fuzz`。
//!
//! # 例子
//!
//! ```
//! use llm_tensor::{DType, Shape, product};
//!
//! let shape = Shape::new([2, 3]);
//! assert_eq!(shape.numel(), 6);
//! assert_eq!(shape.rank(), 2);
//! assert_eq!(shape.element_bytes(DType::F32), 24);
//! assert_eq!(shape.element_bytes(DType::U4), 3, "亚字节精度按位向上取整");
//! assert_eq!(product::<usize>(shape.dims()), 6);
//!
//! // 空形状是标量：0 个轴，1 个元素。
//! assert_eq!(Shape::new([]).numel(), 1);
//! ```

#![deny(unsafe_code)]

mod shape;

// 元素类型的唯一事实源在 L1 `dtype`：全精度 24 变体加能力分类，这里不再自定义枚举。
pub use dtype::DType;
pub use shape::{AxisLen, Shape, product};

#[cfg(test)]
#[path = "../tests/shape_tests.rs"]
mod shape_tests;
