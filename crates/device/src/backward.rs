//! 设备侧反向核：逐点一元算子的导数描述与反向程序生成。
//!
//! 前向 lowering 走 `cutile` 目录声明；反向是新的一条设备通道：这里只拥有「一条反向核需要哪些
//! 操作数、对哪个保存值求导、后端方言源怎么拼」。梯度**图编排**（谁先谁后、梯度何时齐、激活
//! 何时留还）归 `autograd`，标量反向公式的权威实现仍在 `train`，本模块的设备核是同式的第二份
//! 执行体，与前向设备核复用同一纪律：机制层只给导数表达式，厂商语法差异落在后端分支。
//!
//! # 反向核操作数契约（四段定长槽位，顺序固定）
//!
//! | 槽位 | 角色 | 缓冲 | 含义 |
//! | --- | --- | --- | --- |
//! | 0 | 输入 | `x` | 前向输入激活（只读） |
//! | 1 | 输入 | `y` | 前向输出快照（只读；不读输出的算子仍占槽，保证槽位序闭集） |
//! | 2 | 输入 | `dout` | 上游梯度（只读） |
//! | 3 | 输出 | `dx` | 本输入的梯度（只写） |
//!
//! 逐点族一元素一线程，导数表达式在元素内展开，没有规约轴；梯度累加回参数槽由 `train` 侧完成，
//! 设备核只产出本节点对本输入的逐元素贡献。

use diagnostic::{Code, Error, Result};
use llm_optim::execution::Backend;

const SITE: &str = "device::backward";

/// 一条逐点一元算子的反向核描述：算子名、导数表达式、是否读回前向输出。
///
/// 导数表达式在元素下标 `i` 上展开，可引用 `x[i]`（输入）与 `y[i]`（保存的前向输出）；上游
/// 梯度 `dout[i]` 由内核骨架统一相乘，表达式只写对输入的偏导。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientEntry {
    op: &'static str,
    deriv: &'static str,
    saved_output: bool,
}

impl GradientEntry {
    /// 算子名（与前向注册表同名）。
    pub fn op(&self) -> &str {
        self.op
    }

    /// 偏导表达式（元素下标 `i`，引用 `x[i]` / `y[i]`）。
    pub fn derivative(&self) -> &str {
        self.deriv
    }

    /// 导数是否依赖保存的前向输出 `y`；为 `false` 时槽位 1 仍绑定但内核不读。
    pub fn reads_output(&self) -> bool {
        self.saved_output
    }
}

/// 已接设备反向核的逐点一元算子闭集；其余可微算子（`silu` / `gelu` / 归一化 / 注意力 / 卷积 /
/// 矩阵乘等）反向仍只走 `train` CPU 标量公式，设备通道待同式扩入本表。
pub const POINTWISE: &[GradientEntry] = &[
    GradientEntry {
        op: "relu_f32",
        deriv: "(x[i] > 0.0f ? 1.0f : 0.0f)",
        saved_output: false,
    },
    GradientEntry {
        op: "leaky_relu_f32",
        deriv: "(x[i] >= 0.0f ? 1.0f : 0.01f)",
        saved_output: false,
    },
    GradientEntry {
        op: "sigmoid_f32",
        deriv: "(y[i] * (1.0f - y[i]))",
        saved_output: true,
    },
    GradientEntry {
        op: "tanh_f32",
        deriv: "(1.0f - y[i] * y[i])",
        saved_output: true,
    },
    GradientEntry {
        op: "exp_f32",
        deriv: "y[i]",
        saved_output: true,
    },
];

/// 按算子名取反向核描述；未登记返回 `None`（该算子反向不在设备通道内）。
pub fn lookup(op: &str) -> Option<&'static GradientEntry> {
    POINTWISE.iter().find(|entry| entry.op == op)
}

/// 生成一条反向核的设备源。
///
/// # Errors
///
/// 目标后端尚未接入反向源生成时返回 [`Code::Unsupported`]；CUDA 通道目前只留缝，本机无 NVCC 与
/// 真机证据，不产出未验证的内核文本。
pub fn source(backend: Backend, entry: &GradientEntry, len: usize) -> Result<String> {
    match backend {
        Backend::Metal => Ok(metal_source(entry, len)),
        Backend::Cuda => Err(Error::new(
            Code::Unsupported,
            format!(
                "CUDA 反向核 `{}` 尚未生成：本机无 NVCC 与真机证据",
                entry.op
            ),
        )
        .with(SITE)),
    }
}

/// Metal 逐点反向核：一元素一线程，长度守卫跳过末组越界线程；骨架统一乘上游梯度。
/// 线程下标命名为 `i`，与导数表达式里的 `x[i]` / `y[i]` 对齐。
fn metal_source(entry: &GradientEntry, len: usize) -> String {
    let name = format!("{}_grad", entry.op);
    let deriv = entry.deriv;
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void {name}(\n\
         \x20    device const float* x [[buffer(0)]],\n\
         \x20    device const float* y [[buffer(1)]],\n\
         \x20    device const float* dout [[buffer(2)]],\n\
         \x20    device float* dx [[buffer(3)]],\n\
         \x20    uint i [[thread_position_in_grid]]) {{\n\
         \x20    if (i >= {len}) return;\n\
         \x20    dx[i] = dout[i] * {deriv};\n\
         }}"
    )
}
