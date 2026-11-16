//! CPU 推理内核桥：把 `op` 的 f32 标量参考适配成 engine 可登记的 [`Program`]。
//!
//! 落点裁决见 [`backend-runtime-design.md`](../../../docs/backend-runtime-design.md) §4：
//! 端点契约、路由与线格式保持模型无关；CPU 模型实现作为注入 [`Predict`](crate::Predict)
//! 的提供者收在本模块内，真设备后端仍按 ADR 0025 收在独立边界 crate，不回流这里。
//!
//! # 职责
//!
//! - 编译期锁定的算子表：表内每个名字都必须在 `op::REGISTRY` 内（`const` 子集断言）；
//!   表未覆盖的算子不登记，执行时由 [`CpuExecutor`](engine::exec::CpuExecutor) 按名拒绝，
//!   不留静默空入口。
//! - 适配函数只做拆切片 + 调 `op::<op>::forward`，算术不重写。模式复刻 `train::KERNELS`
//!   （`crates/train/src/kernel.rs`），但不带 train 的 `View`/`Lane` 适配层：engine 缓冲
//!   形态就是 `Vec<f32>` 切片。
//! - 层栈装配、驻留参数表与前向宿主收在 [`assemble`] 子模块，本文件只留内核适配表。
//!
//! # 不拥有
//!
//! 执行/调度/KV 簿记（`engine`）、算子声明与参考算术（`op`）、层形状语义（`nn`）、
//! 权重字节读取（`weights`）。本模块不出现设备厂商或后端类型的特判分支。
//!
//! # 范围
//!
//! 只接纯切片签名的 f32 算子；`rope_f32` 的位置标量、decode 变长历史几何随后续 KV 工作
//! 一起落地（设计文档 §5.1）。u8 lane 量化族延后 P3（§5.3）。

mod assemble;

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Launch, Program};
use op::REGISTRY;

pub use assemble::{CpuModel, Params};

const SITE: &str = "serve::model";

/// 注意力内核入口：`q/k/v → out`，序列长 `s` 与头宽 `d` 随几何传入。
type AttentionKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize) -> Result<()>;

/// 适配入口：拆好切片后调用算子参考实现。
type KernelFn = fn(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()>;

/// 表内一行：算子名（与 `op::REGISTRY` 同名）+ 缓冲级函数指针。
struct Entry {
    op: &'static str,
    run: KernelFn,
}

/// 算子适配宏：按 arity 分支展开拆切片 + 调 `op::<op>::forward`。
///
/// - `unary`：一进一出；
/// - `binary`：两进一出（两路是向量+向量，或向量/索引 + 驻留权重整块，算子签名相同）。
///
/// 适配函数名即算子名，注册名由 [`kernel_table!`] 从标识符派生，不再手抄第二份字符串。
macro_rules! kernel {
    ($($run:ident),* ; unary) => {$(
        fn $run(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
            let (Some(input), Some(output)) = (inputs.first(), outputs.first_mut()) else {
                return Err(Error::new(
                    Code::Invalid,
                    concat!(stringify!($run), " 需要一进一出"),
                )
                .with(SITE));
            };
            op::$run::forward(input, output)
        }
    )*};
    ($($run:ident),* ; binary) => {$(
        fn $run(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
            let (Some(left), Some(right), Some(output)) =
                (inputs.first(), inputs.get(1), outputs.first_mut())
            else {
                return Err(Error::new(
                    Code::Invalid,
                    concat!(stringify!($run), " 需要两进一出"),
                )
                .with(SITE));
            };
            op::$run::forward(left, right, output)
        }
    )*};
}

kernel!(
    gelu_f32,
    reduce_max_f32,
    reduce_sum_f32,
    scale_f32,
    silu_f32,
    softmax_f32; unary
);
kernel!(
    add_f32,
    mul_f32,
    residual_add_f32,
    rmsnorm_f32,
    embed_f32,
    gemv_f32; binary
);

/// 注意力三路适配：`q/k/v → out`；头宽 `d` 取算子静态语义表，序列长 `s` 由 `q` 长推出，
/// 形状不符交由算子参考校验。朴素与 Flash 同几何，只换内核算式。
fn attention_paths(
    inputs: &[&[f32]],
    outputs: &mut [&mut [f32]],
    run: AttentionKernel,
) -> Result<()> {
    let (Some(q), Some(k), Some(v), Some(out)) = (
        inputs.first(),
        inputs.get(1),
        inputs.get(2),
        outputs.first_mut(),
    ) else {
        return Err(Error::new(Code::Invalid, "注意力需要三进一出").with(SITE));
    };
    let d = op::attention::MODEL.shape.1;
    let s = q.len() / d;
    run(q, k, v, out, s, d)
}

fn attention_f32(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
    attention_paths(inputs, outputs, op::attention::forward)
}

fn flash_attention_f32(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
    attention_paths(inputs, outputs, op::attention::flash_forward)
}

/// 内核表展开：每条例子取适配函数标识符，算子名由 `stringify!` 派生，避免名字与函数名各写一份。
macro_rules! kernel_table {
    ($( $run:ident ),* $(,)?) => {
        &[ $( Entry { op: stringify!($run), run: $run } ),* ]
    };
}

/// CPU 内核表：按 `op::REGISTRY` 顺序排列的 f32 纯切片子集；长度由表本身数出。
const KERNELS: &[Entry] = kernel_table![
    add_f32,
    attention_f32,
    embed_f32,
    flash_attention_f32,
    gelu_f32,
    gemv_f32,
    mul_f32,
    reduce_max_f32,
    reduce_sum_f32,
    residual_add_f32,
    rmsnorm_f32,
    scale_f32,
    silu_f32,
    softmax_f32,
];

const _: () = all_in_registry(KERNELS, &REGISTRY);

/// 编译期断言：表内每个算子名都能在 `op::REGISTRY` 里找到（子集即可，不必等长）。
const fn all_in_registry(entries: &[Entry], registry: &[op::Registration]) {
    let mut index = 0;
    while index < entries.len() {
        assert!(
            name_located(entries[index].op, registry),
            "serve::model 内核表含注册表之外的算子"
        );
        index += 1;
    }
}

/// `const` 上下文里没有 `str` 比较，逐字节比名字。
const fn name_located(name: &str, registry: &[op::Registration]) -> bool {
    let mut index = 0;
    while index < registry.len() {
        if same_str(name, registry[index].name) {
            return true;
        }
        index += 1;
    }
    false
}

const fn same_str(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// 表条目对 engine [`Program`] 契约的适配：名字随表走，执行入口落到适配函数。
struct Kernel {
    entry: &'static Entry,
}

impl Program for Kernel {
    fn name(&self) -> &str {
        self.entry.op
    }

    fn run(&self, _launch: Launch, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        (self.entry.run)(inputs, outputs)
    }
}

/// 把表内全部 CPU 内核登记进执行器；重名由 [`CpuExecutor::register`] 拒绝。
///
/// 装配侧在构造 CPU 模型宿主时调用一次；之后执行器只收图与操作数编号，不再改内核表。
pub fn register_cpu(executor: &mut CpuExecutor) -> Result<()> {
    for entry in KERNELS {
        executor.register(Box::new(Kernel { entry }))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/model_tests.rs"]
mod tests;
