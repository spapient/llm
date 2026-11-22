//! 声明 → 并行程序：把 cutile 推出的 [`Launch`] 装配成「设备源 + 按网格发射的 host 驱动」。

use diagnostic::{Code, Error, Result};
use llm_optim::backend::{Launch, Lower, Source};
use prof::decl::Grid;
use nn::model::Sample;
use nn::sig::Signature;

#[cfg(test)]
#[path = "../../testkit/tests/kernel.rs"]
mod cutile_x_kernel;

fn fold_of(func: &str) -> &'static str {
    if func.contains("max") { "max" } else { "sum" }
}

/// 「这台后端没有这个形态的并行程序」：调用方按它**显式回落**到语义门，不把缺实现说成已验证。
fn unsupported(vendor: &str, kind: &Lower) -> Error {
    Error::new(
        Code::Unsupported,
        format!("{vendor} 没有 `{kind:?}` 的并行程序"),
    )
    .with("engine::program")
}

#[cfg(test)]
#[path = "../tests/program.rs"]
mod tests;

/// CUDA 并行程序：按形态装配设备源与 host 驱动，整份可直接 `mcc` / `nvcc` 编译执行。
///
/// # Errors
///
/// 该形态在 CUDA 上没有并行实现（tile / flash / WMMA 仍走各自的生产态入口），或样本与签名不一致。
pub fn cuda(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<String> {
    match &launch.kind {
        Lower::Pointwise | Lower::Loop(_) => cuda_single(kernel, signature, sample, launch),
        Lower::ChunkReduce { chunk, func } => {
            cuda_reduce(kernel, signature, sample, launch, *chunk, func)
        }
        _ => cuda_segments(kernel, signature, sample, launch),
    }
}

/// 单源族：一份设备源 + 一份主机驱动，两族的差别只在几何怎么折算。
fn cuda_single(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<String> {
    let Lower::Loop(grid) = &launch.kind else {
        let device = crate::cuda::cuda_grid_kernel(kernel, signature, sample)?;
        // CUDA 的网格是**块数**：工作项按线程组大小折算，块内再放 `group` 个线程。
        let blocks = launch.items[0].div_ceil(launch.group);
        return crate::cuda::cuda_point_driver(&device, signature, sample, blocks, launch.group);
    };
    // 零维网格就是串行：语义门那一份单线程 harness 已是它的并行程序（1 个线程）。
    if grid.dims.is_empty() {
        return crate::cuda::cuda_harness(kernel, signature, sample);
    }
    let device = crate::cuda::cuda_loop_kernel(kernel, signature, sample, grid)?;
    crate::cuda::cuda_loop_driver(&device, signature, sample, &launch.items[..grid.dims.len()])
}

/// 归约族：单段设备源 + 主机折叠，折叠算子在 `fold_of` 一处派生。
fn cuda_reduce(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
    chunk: usize,
    func: &str,
) -> Result<String> {
    let device = crate::cuda::cuda_chunk_reduce(kernel, signature, sample, chunk, func)?;
    crate::cuda::cuda_chunk_reduce_driver(
        &device,
        signature,
        sample,
        launch.items[0],
        fold_of(func),
    )
}

/// 多段族：段数与主机折叠是一体的，所以每臂只调各自模块的合成入口，这里不再拼第二份段表。
fn cuda_segments(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<String> {
    match &launch.kind {
        Lower::ChunkRmsnorm { chunk } => {
            crate::norm::cuda_rmsnorm(kernel, signature, sample, *chunk)
        }
        Lower::ChunkSoftmax { chunk } => {
            crate::norm::cuda_softmax(kernel, signature, sample, *chunk)
        }
        Lower::ChunkScan { chunk, decay } => {
            crate::scan::cuda_scan(kernel, signature, sample, *chunk, *decay)
        }
        other => Err(unsupported("CUDA", other)),
    }
}

/// Metal 并行程序：单源族（逐点 / 循环注入）各有一份 `kernel_main`，分块族是多段源 + 主机折叠。
///
/// 分块族的段数与折叠是一体的，因此每臂只调 [`crate::metal::fold`] 里各自的合成入口，
/// 与 CUDA 侧的 `cuda_segments` 分法一致：这层只管「什么形态配哪份程序」。
///
/// # Errors
///
/// 该形态没有 Metal 并行实现（调用方据此回落到语义门），或样本与签名不一致。
pub fn metal(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<Source> {
    match metal_segments(kernel, signature, sample, launch) {
        Some(source) => source,
        None => metal_dispatch(kernel, signature, sample, launch),
    }
}

/// 分块族的四份合成入口；不属分块族时 `None`（交给单源族）。
fn metal_segments(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Option<Result<Source>> {
    let kind = &launch.kind;
    if let Lower::ChunkReduce { chunk, func } = kind {
        return Some(crate::metal::fold::reduce(
            kernel, signature, sample, *chunk, func,
        ));
    }
    if let Lower::ChunkRmsnorm { chunk } = kind {
        return Some(crate::metal::fold::rmsnorm(
            kernel, signature, sample, *chunk,
        ));
    }
    if let Lower::ChunkSoftmax { chunk } = kind {
        return Some(crate::metal::fold::softmax(
            kernel, signature, sample, *chunk,
        ));
    }
    if let Lower::ChunkScan { chunk, decay } = kind {
        return Some(crate::metal::fold::scan(
            kernel, signature, sample, *chunk, *decay,
        ));
    }
    None
}

/// 单源族：设备源 + 按网格发射的驱动。
fn metal_dispatch(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<Source> {
    let (device, block) = metal_device(kernel, signature, sample, launch)?;
    let host =
        crate::metal::metal_dispatch_driver(signature, sample, "kernel_main", launch.items, block)?;
    Ok(Source::Pair { device, host })
}

/// 设备源与线程组：`dispatchThreads` 收的是**总线程数**（不是块数），工作项直接当线程；
/// 循环注入是「一维一线程」，组内 1 个。逐点内核带长度守卫，末组多出的线程自己跳过。
fn metal_device(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<(String, [usize; 3])> {
    match &launch.kind {
        Lower::Pointwise => metal_pointwise(kernel, signature, sample, launch.group),
        Lower::Loop(grid) => metal_loop(kernel, signature, sample, grid),
        // 其余形态（分块族 / tile / flash / 波内特化）不是单源 Metal 程序：报「没有实现」而不是
        // 拿去按逐点 lowering 试一遍——那样签名校验会先报 `Invalid`，把「缺实现」说成「签名错」，
        // 调用方据此无法回落。
        other => Err(unsupported("Metal", other)),
    }
}

/// 逐点网格：工作项就是元素，一个元素一个线程，线程组为声明里的 `group`。
fn metal_pointwise(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    group: usize,
) -> Result<(String, [usize; 3])> {
    let device = crate::metal::metal_grid_kernel(kernel, signature, sample)?;
    Ok((device, [group, 1, 1]))
}

/// 循环注入：网格各维即线程形状（一维一线程），组内 1 个。
fn metal_loop(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<(String, [usize; 3])> {
    if grid.dims.is_empty() {
        return Err(Error::new(
            Code::Unsupported,
            "Metal 没有零维网格的并行程序".to_string(),
        )
        .with("engine::program"));
    }
    let device = crate::metal::metal_loop_kernel(kernel, signature, sample, grid)?;
    Ok((device, [1, 1, 1]))
}
