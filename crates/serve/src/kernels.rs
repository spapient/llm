//! CPU 内核登记：共享缓冲视图提取，并把适配函数表投影成 engine [`Program`]。
//!
//! 算术只住在 `nn`；稠密/量化与注意力/RoPE 的签名适配分属两个内联模块。本文件只拥有
//! 注册表、视图类型检查和 `Program` 桥，不拥有算子语义、设备分派或图接线。

use diagnostic::{Code, Error, Result};
use engine::exec::{Executor, Launch, Program, View, ViewMut};
use nn::kernel_table;

mod attention {
    //! 注意力与 RoPE 的签名适配：实例几何来自 `Launch` 或缓冲，算法体只在 `nn`。

    use super::{input_f32, output_f32};
    use diagnostic::{Code, Error, Result};
    use engine::exec::{Launch, View, ViewMut};

    /// prefill、单段 decode 与变长批段 decode 的三种 `nn` 入口。
    type AttentionKernel = fn(
        tensor::View<'_, f32>,
        tensor::View<'_, f32>,
        tensor::View<'_, f32>,
        &mut [f32],
    ) -> Result<()>;
    type DecodeKernel = fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()>;
    type BatchDecodeKernel =
        fn(&[f32], &[f32], &[f32], &mut [f32], usize, &[usize], usize, usize) -> Result<()>;
    type DecodeViews<'a, 'b> = (&'a [f32], &'a [f32], &'a [f32], &'b mut [f32]);

    /// 拆出 q/k/v 三路读入与一路写出；错误名指向实际适配族。
    fn qkv<'a, 'b>(
        name: &str,
        inputs: &'a [View<'a>],
        outputs: &'b mut [ViewMut<'_>],
    ) -> Result<DecodeViews<'a, 'b>> {
        Ok((
            input_f32(name, inputs, 0)?,
            input_f32(name, inputs, 1)?,
            input_f32(name, inputs, 2)?,
            output_f32(name, outputs, 0)?,
        ))
    }

    /// prefill 共享几何：头宽来自静态语义表，序列行数由 q 缓冲推出。
    fn attention_paths(
        inputs: &[View<'_>],
        outputs: &mut [ViewMut<'_>],
        run: AttentionKernel,
    ) -> Result<()> {
        let (q, k, v, out) = qkv("attention", inputs, outputs)?;
        let d = nn::conv::attention::MODEL.shape.1;
        let (q, k, v) = tensor::View::matrix_triple(q.len() / d, d, q, k, v)?;
        run(q, k, v, out)
    }

    /// prefill 表项：朴素与 Flash 只替换 `nn` 入口，适配骨架保持一份。
    macro_rules! attention_kernel {
        ($($run:ident => $forward:path),* $(,)?) => {$(
            pub(super) fn $run(_: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
                attention_paths(inputs, outputs, $forward)
            }
        )*};
    }

    attention_kernel!(
        attention_f32 => nn::conv::attention::forward,
        flash_attention_f32 => nn::conv::attention::flash_forward,
    );

    /// decode 公共路由：第四路非空时是 ragged 段长表，否则是单序列历史。
    fn decode_paths(
        launch: &Launch,
        inputs: &[View<'_>],
        outputs: &mut [ViewMut<'_>],
        single: DecodeKernel,
        batch: BatchDecodeKernel,
    ) -> Result<()> {
        let (q, k, v, out) = qkv("decode_attention", inputs, outputs)?;
        match inputs
            .get(3)
            .and_then(View::f32)
            .filter(|table| !table.is_empty())
        {
            Some(history) => decode_batch(launch, q, k, v, out, history, batch),
            None => decode_single(q, k, v, out, single),
        }
    }

    /// 单序列几何：头宽取静态语义表，查询行数与历史行数分别由缓冲长度推出。
    fn decode_single(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        single: DecodeKernel,
    ) -> Result<()> {
        let d = nn::conv::decode_attention::MODEL.shape.1;
        let heads = q.len() / d;
        single(
            q,
            k,
            v,
            out,
            q.len() / (heads * d),
            k.len() / (heads * d),
            d,
        )
    }

    /// 批段几何：段长表只在此处转换为 `nn` 的索引档，K/V 布局由批入口统一解释。
    fn decode_batch(
        launch: &Launch,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        history: &[f32],
        batch: BatchDecodeKernel,
    ) -> Result<()> {
        let head_dim = launch.block[1] as usize;
        if head_dim == 0 {
            return Err(Error::new(Code::Invalid, "decode 批段头宽为 0").with(super::SITE));
        }
        if history.is_empty() {
            return Err(Error::new(Code::Invalid, "decode 批段长表为空").with(super::SITE));
        }
        let heads = launch.block[0] as usize / head_dim;
        let q_rows = q.len() / (history.len() * heads * head_dim);
        let history: Vec<usize> = history.iter().map(|&rows| rows as usize).collect();
        batch(q, k, v, out, q_rows, &history, head_dim, heads)
    }

    /// decode 表项：朴素与 Flash 共享单段/批段路由和几何守卫。
    macro_rules! decode_kernel {
        ($($run:ident => $single:path, $batch:path),* $(,)?) => {$(
            pub(super) fn $run(launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
                decode_paths(&launch, inputs, outputs, $single, $batch)
            }
        )*};
    }

    decode_kernel!(
        decode_attention_f32 => nn::conv::attention::decode_forward_heads, nn::conv::attention::decode_forward_batch_var,
        decode_flash_attention_f32 => nn::conv::attention::decode_flash_forward_heads, nn::conv::attention::decode_flash_forward_batch_var,
    );

    /// RoPE 路由：批位置表优先，缺省从 `Launch` 标量槽读取单序列位置。
    pub(super) fn rope_f32(
        launch: Launch,
        inputs: &[View<'_>],
        outputs: &mut [ViewMut<'_>],
    ) -> Result<()> {
        let x = input_f32("rope_f32", inputs, 0)?;
        let out = output_f32("rope_f32", outputs, 0)?;
        let head_dim = launch.block[1] as usize;
        match inputs
            .get(1)
            .and_then(View::f32)
            .filter(|table| !table.is_empty())
        {
            Some(positions) => {
                nn::rope_f32::forward_batch_from_f32(positions, tensor::View::one(x), out, head_dim)
            }
            None => rope_single(&launch, x, out, head_dim),
        }
    }

    /// 单序列位置通道：缺位置即拒绝，不把未初始化标量当成位置零。
    fn rope_single(launch: &Launch, x: &[f32], out: &mut [f32], head_dim: usize) -> Result<()> {
        let position = launch
            .scalars()
            .rope_position()
            .ok_or_else(|| Error::new(Code::Invalid, "rope_f32 位置标量未填").with(super::SITE))?;
        if head_dim == 0 || head_dim == x.len() {
            nn::rope_f32::forward(position as usize, tensor::View::one(x), out)
        } else {
            nn::rope_f32::forward_heads(position as usize, tensor::View::one(x), out, head_dim)
        }
    }
}
#[path = "dense.rs"]
mod dense;
use attention::*;
use dense::*;

pub(crate) const SITE: &str = "serve::model";

/// 适配入口：拆好视图后调用算子实现；`Launch` 承载实例几何与标量槽。
type KernelFn = fn(Launch, &[View<'_>], &mut [ViewMut<'_>]) -> Result<()>;

/// 取第 `slot` 路输入为 `f32` 切片；编号缺失或形态不符给 `Invalid`。
fn input_f32<'a>(name: &str, inputs: &[View<'a>], slot: usize) -> Result<&'a [f32]> {
    inputs.get(slot).and_then(View::f32).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{name}` 第 {slot} 路输入不是浮点缓冲"),
        )
        .with(SITE)
    })
}

/// 取第 `slot` 路输出为可写 `f32` 切片；缺失或形态不符给 `Invalid`。
fn output_f32<'a>(
    name: &str,
    outputs: &'a mut [ViewMut<'_>],
    slot: usize,
) -> Result<&'a mut [f32]> {
    outputs
        .get_mut(slot)
        .and_then(ViewMut::f32_mut)
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("`{name}` 第 {slot} 路输出不是浮点缓冲"),
            )
            .with(SITE)
        })
}

/// 表内一行：算子名与 `nn::registry::REGISTRY` 同名，执行体是静态函数指针。
struct Entry {
    op: &'static str,
    run: KernelFn,
}

/// CPU 内核表：名字由适配函数标识符派生，避免名称与入口各写一份。
const KERNELS: &[Entry] = &kernel_table![
    add_f32,
    attention_f32,
    decode_attention_f32,
    decode_flash_attention_f32,
    dequant_i2,
    dequant_i4,
    dequant_i8,
    embed_f32,
    flash_attention_f32,
    gelu_f32,
    gemm_i2,
    gemm_i4,
    gemm_i8,
    gemv_f32,
    gemv_i2,
    gemv_i4,
    gemv_i8,
    mul_f32,
    reduce_max_f32,
    reduce_sum_f32,
    residual_add_f32,
    rmsnorm_f32,
    rope_f32,
    scale,
    silu_f32,
    softmax_f32,
];

const _: () = all_in_registry(KERNELS);

/// 编译期断言：表内每个算子都在唯一注册表内；复用上游 const 查询，不复制字节比较。
const fn all_in_registry(entries: &[Entry]) {
    let mut idx = 0;
    while idx < entries.len() {
        assert!(
            nn::registry::contains(entries[idx].op),
            "serve::model 内核表含注册表之外的算子"
        );
        idx += 1;
    }
}

/// 一条静态表项对 engine 动态 `Program` 契约的适配；动态性只留在后端边界。
struct Kernel {
    entry: &'static Entry,
}

impl Program for Kernel {
    fn name(&self) -> &str {
        self.entry.op
    }

    fn run(&self, launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        (self.entry.run)(launch, inputs, outputs)
    }
}

/// 把表内全部 CPU 内核登记进执行器；重名由 [`Executor::register_program`] 拒绝。
pub fn register_cpu<E: Executor>(executor: &mut E) -> Result<()> {
    for entry in KERNELS {
        executor.register_program(Box::new(Kernel { entry }))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/model_tests.rs"]
mod tests;
