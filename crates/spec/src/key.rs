//! 职责：内核登记键（ADR 0027 七元组）。

use crate::atom::*;

/// 一条内核实现的登记键（ADR 0027 §1 七元组）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    /// 原语闭集。
    pub prim: Prim,
    /// 实现名，与档案 `provides.impl_name` 同源（`scalar_f32` / `fma` / `dp4a` …）。
    pub impl_name: String,
    /// 元素类型；键取单点，不是位集。
    pub dtype: DType,
    /// 存储层级；键取单点。
    pub space: Space,
    /// 子组 / 向量宽度；标量实现记 `1`。
    pub lanes: u32,
    /// 指令集 / 后端登记标签（`cpu` / `neon` / `metal` / `cuda` …），不产生热路径分发。
    pub arch: String,
    /// 分块宽度（tile 尺寸 / 量化块宽）；标量实现记 `1`。
    pub block: u32,
}

impl KernelKey {
    /// 登记一条内核键；`lanes` / `block` 为 `0` 按标量归一成 `1`。
    pub fn new(
        prim: Prim,
        impl_name: impl Into<String>,
        dtype: DType,
        space: Space,
        lanes: u32,
        arch: impl Into<String>,
        block: u32,
    ) -> Self {
        Self {
            prim,
            impl_name: impl_name.into(),
            dtype,
            space,
            lanes: lanes.max(1),
            arch: arch.into(),
            block: block.max(1),
        }
    }

    /// 签名渲染：`dot{dt:f32, space:FAST, lanes:4, algorithm:cpu, block:16}/impl:scalar_f32`。
    pub fn signature(&self) -> String {
        format!(
            "{}{{dt:{}, space:{}, lanes:{}, algorithm:{}, block:{}}}/impl:{}",
            self.prim,
            self.dtype.name(),
            self.space.as_str(),
            self.lanes,
            self.arch,
            self.block,
            self.impl_name
        )
    }
}
