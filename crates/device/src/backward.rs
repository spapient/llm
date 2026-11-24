//! 设备侧反向核：逐点一元算子的导数描述与反向程序生成。
//!
//! 前向 lowering 走 `cutile` 目录声明；反向是新的一条设备通道：这里只拥有「一条反向核需要哪些
//! 操作数、对哪个保存值求导、后端方言源怎么拼」。梯度**图编排**（谁先谁后、梯度何时齐、激活
//! 何时留还）归 `autograd`，标量反向公式的权威实现仍在 `train`，本模块的设备核是同式的第二份
//! 执行体，与前向设备核复用同一纪律：机制层只标导数公式身份，厂商语法与 intrinsic 命名落在
//! 后端发射器分支。
//!
//! # 反向核操作数契约（四段定长槽位，顺序固定）
//!
//! | 槽位 | 角色 | 缓冲 | 含义 |
//! --- | --- | --- | --- |
//! | 0 | 输入 | `x` | 前向输入激活（只读） |
//! | 1 | 输入 | `y` | 前向输出快照（只读；不读输出的算子仍占槽，保证槽位序闭集） |
//! | 2 | 输入 | `dout` | 上游梯度（只读） |
//! | 3 | 输出 | `dx` | 本输入的梯度（只写） |
//!
//! 逐点族一元素一线程，导数表达式在元素内展开，没有规约轴；梯度累加回参数槽由 `train` 侧完成，
//! 设备核只产出本节点对本输入的逐元素贡献。归一化 / 注意力 / 卷积 / 矩阵乘等带规约轴或多输出的
//! 算子反向不在本逐点契约内，走各自的多趟核通道。

use llm_optim::execution::Backend;

/// 闭集一元导数公式：只标身份，方言 intrinsic（`exp`/`erf` 命名、浮点后缀）由发射器落地，
/// 机制层不出现厂商字面量。公式口径与 `train` 标量反向一一对应。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Deriv {
    /// `max(x, 0)`：`x > 0 ? 1 : 0`（严格大于，负区斜率 0）。
    Relu,
    /// `x ≥ 0 ? x : 0.01·x`：正区斜率 1，负区斜率 [`LEAKY_SLOPE`]。
    LeakyRelu,
    /// `σ(x)`：`y·(1−y)`，读回前向输出 `y`。
    SigmoidOut,
    /// `tanh(x)`：`1−y²`，读回前向输出 `y`。
    TanhOut,
    /// `exp(x)`：`y`，读回前向输出 `y`。
    ExpOut,
    /// `x·σ(x)`：`σ + x·σ·(1−σ)`，对 `x` 现算 σ。
    Silu,
    /// `0.5·x·(1+erf(x/√2))`：`Φ(x) + x·φ(x)`，对 `x` 现算正态密度。
    Gelu,
}

/// LeakyReLU 负区斜率，与 `nn::leaky_relu_f32::SLOPE` 同值。
const LEAKY_SLOPE: f32 = 0.01;

impl Deriv {
    /// 该公式是否读回前向输出 `y`；为 `false` 时槽位 1 仍绑定但内核不读。
    fn reads_output(self) -> bool {
        matches!(self, Deriv::SigmoidOut | Deriv::TanhOut | Deriv::ExpOut)
    }
}

/// 一条逐点一元算子的反向核描述：算子名 + 导数公式身份。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientEntry {
    op: &'static str,
    deriv: Deriv,
}

impl GradientEntry {
    /// 算子名（与前向注册表同名）。
    pub fn op(&self) -> &str {
        self.op
    }

    /// 导数是否依赖保存的前向输出 `y`；为 `false` 时槽位 1 仍绑定但内核不读。
    pub fn reads_output(&self) -> bool {
        self.deriv.reads_output()
    }
}

/// 已接设备反向核的逐点一元算子闭集；其余可微算子（归一化 / 注意力 / 卷积 / 矩阵乘等）反向仍
/// 只走 `train` CPU 标量公式，设备通道待各自的多趟规约核扩入。
pub const POINTWISE: &[GradientEntry] = &[
    GradientEntry { op: "relu_f32", deriv: Deriv::Relu },
    GradientEntry { op: "leaky_relu_f32", deriv: Deriv::LeakyRelu },
    GradientEntry { op: "sigmoid_f32", deriv: Deriv::SigmoidOut },
    GradientEntry { op: "tanh_f32", deriv: Deriv::TanhOut },
    GradientEntry { op: "exp_f32", deriv: Deriv::ExpOut },
    GradientEntry { op: "silu_f32", deriv: Deriv::Silu },
    GradientEntry { op: "gelu_f32", deriv: Deriv::Gelu },
];

/// 按算子名取反向核描述；未登记返回 `None`（该算子反向不在设备通道内）。
pub fn lookup(op: &str) -> Option<&'static GradientEntry> {
    POINTWISE.iter().find(|entry| entry.op == op)
}

/// 生成一条逐点反向核的设备源（只含设备内核，不含主机装配；主机缓冲与回读由调用方 harness 拼）。
pub fn source(backend: Backend, entry: &GradientEntry, len: usize) -> String {
    match backend {
        Backend::Metal => metal_kernel(entry, len),
        Backend::Cuda => cuda_kernel(entry, len),
    }
}

/// 方言 intrinsic 命名：Metal 用 `exp`，CUDA 用 `expf`（MSL 无 `erf`，误差函数走 A&S 多项式）。
struct Intr {
    exp: &'static str,
}

const METAL_INTR: Intr = Intr { exp: "exp" };
const CUDA_INTR: Intr = Intr { exp: "expf" };

/// 元素体内核语句：先算公式需要的局部量，再统一乘上游梯度 `dout[i]`。
fn body_lines(deriv: Deriv, intr: &Intr) -> Vec<String> {
    let exp = intr.exp;
    match deriv {
        Deriv::Relu => vec!["dx[i] = dout[i] * (x[i] > 0.0f ? 1.0f : 0.0f);".into()],
        Deriv::LeakyRelu => vec![format!(
            "dx[i] = dout[i] * (x[i] >= 0.0f ? 1.0f : {LEAKY_SLOPE}f);"
        )],
        Deriv::SigmoidOut => vec!["dx[i] = dout[i] * (y[i] * (1.0f - y[i]));".into()],
        Deriv::TanhOut => vec!["dx[i] = dout[i] * (1.0f - y[i] * y[i]);".into()],
        Deriv::ExpOut => vec!["dx[i] = dout[i] * y[i];".into()],
        Deriv::Silu => vec![
            format!("float s = 1.0f / (1.0f + {exp}(-x[i]));"),
            "dx[i] = dout[i] * (s + x[i] * s * (1.0f - s));".into(),
        ],
        Deriv::Gelu => {
            // Φ(x) 里的 erf 走 A&S 7.1.26 多项式（|误差| < 1.5e-7），与 `nn::gelu_f32::erf` 同式。
            vec![
                "float az = abs(x[i] * 0.70710678f);".into(),
                "float t = 1.0f / (1.0f + 0.3275911f * az);".into(),
                "float series = t * (0.2548296f + t * (-0.28449672f + t * (1.4214138f + t * (-1.4531521f + t * 1.0614054f))));".into(),
                format!("float erf = (x[i] < 0.0f ? -1.0f : 1.0f) * (1.0f - series * {exp}(-az * az));"),
                "float phi = 0.5f * (1.0f + erf);".into(),
                format!("float pdf = x[i] * {exp}(-x[i] * x[i] * 0.5f) * 0.39894228f;"),
                "dx[i] = dout[i] * (phi + pdf);".into(),
            ]
        }
    }
}

/// Metal 逐点反向核：一元素一线程，长度守卫跳过末组越界线程；线程下标命名 `i`，与体内 `x[i]` 对齐。
fn metal_kernel(entry: &GradientEntry, len: usize) -> String {
    let name = format!("{}_grad", entry.op);
    let body = body_lines(entry.deriv, &METAL_INTR);
    let body = body
        .iter()
        .map(|line| format!("         \x20    {line}"))
        .collect::<Vec<_>>()
        .join("\n");
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
         {body}\n\
         }}"
    )
}

/// CUDA 逐点反向核：256 线程一块，线程下标由 `blockIdx`/`threadIdx` 装配；长度守卫同上。
fn cuda_kernel(entry: &GradientEntry, _len: usize) -> String {
    let name = format!("{}_grad", entry.op);
    let body = body_lines(entry.deriv, &CUDA_INTR);
    let body = body
        .iter()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "__global__ void {name}(\n\
         \x20    const float* x,\n\
         \x20    const float* y,\n\
         \x20    const float* dout,\n\
         \x20    float* dx,\n\
         \x20    int n) {{\n\
         \x20    int i = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20    if (i >= n) return;\n\
         {body}\n\
         }}"
    )
}
