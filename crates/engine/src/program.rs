//! 声明 → 并行程序：把 cutile 推出的 [`Launch`] 装配成「设备源 + 按网格发射的 host 驱动」。
//!
//! 本模块只回答**怎么装配**，不回答「并行还是串行」——形态与派发几何都来自
//! `cutile::execution::launch`（唯一分派点）。按算子名在这里挑驱动会形成第二份分派，与 `[chunk]` /
//! `[grid]` 声明漂移：声明改了而这里没改，验证会拿错的形状去比，结论还照样是「通过」。
//!
//! 装配只做三件事：把同一份内核降成该形态要的段数、按段数拼上对应的 host 驱动、把 `launch.items`
//! 原样交给驱动发射。段数与驱动本来就是一体的（两段扫描 / 三段 softmax 各有一份主机折叠），
//! 因此这里的分派**只由形态决定**，不看 `m` / `n`，也不看算子名。

use cutile::backend::{Launch, Lower, Source};
use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::Signature;

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
        let device = crate::codegen::cuda_grid_kernel(kernel, signature, sample)?;
        // CUDA 的网格是**块数**：工作项按线程组大小折算，块内再放 `group` 个线程。
        let blocks = launch.items[0].div_ceil(launch.group);
        return crate::codegen::cuda_point_driver(&device, signature, sample, blocks, launch.group);
    };
    // 零维网格就是串行：语义门那一份单线程 harness 已是它的并行程序（1 个线程）。
    if grid.dims.is_empty() {
        return crate::codegen::cuda_harness(kernel, signature, sample);
    }
    let device = crate::codegen::cuda_loop_kernel(kernel, signature, sample, grid)?;
    crate::codegen::cuda_loop_driver(&device, signature, sample, &launch.items[..grid.dims.len()])
}

/// 归约族：单段设备源 + 主机折叠，折叠算子在 [`fold_of`] 一处派生。
fn cuda_reduce(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
    chunk: usize,
    func: &str,
) -> Result<String> {
    let device = crate::codegen::cuda_chunk_reduce(kernel, signature, sample, chunk, func)?;
    crate::codegen::cuda_chunk_reduce_driver(
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

/// Metal 并行程序：逐点与循环注入各有一份 `kernel_main`，按 `launch.items` 发射。
///
/// 分块族（归一化 / 扫描 / 归约）是多段源 + 主机折叠，Metal 侧的落点是各自的 `metal_*_program`，
/// 不走这条单源装配——这里显式拒绝，调用方据此回落到语义门，而不是拿错的段数硬拼。
///
/// # Errors
///
/// 该形态没有单源 Metal 并行实现，或样本与签名不一致。
pub fn metal(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<Source> {
    let (device, block) = metal_device(kernel, signature, sample, launch)?;
    let host = crate::codegen::metal_dispatch_driver(
        signature,
        sample,
        "kernel_main",
        launch.items,
        block,
    )?;
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
        Lower::Pointwise => {
            let device = crate::codegen::metal_grid_kernel(kernel, signature, sample)?;
            Ok((device, [launch.group, 1, 1]))
        }
        Lower::Loop(grid) => metal_loop_device(kernel, signature, sample, grid),
        // 其余形态（分块族 / tile / flash / 波内特化）不是单源 Metal 程序：报「没有实现」而不是
        // 拿去按逐点 lowering 试一遍——那样签名校验会先报 `Invalid`，把「缺实现」说成「签名错」，
        // 调用方据此无法回落。
        other => Err(unsupported("Metal", other)),
    }
}

fn metal_loop_device(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &op::decl::Grid,
) -> Result<(String, [usize; 3])> {
    if grid.dims.is_empty() {
        return Err(unsupported("Metal", &Lower::Loop(grid.clone())));
    }
    let device = crate::codegen::metal_loop_kernel(kernel, signature, sample, grid)?;
    Ok((device, [1, 1, 1]))
}

/// 主机折叠算子：`func` 由声明的 `reduce.op` 派生（`_cutile_max_f32` / `_cutile_sum_f32`），
/// 主机折叠读同一个字段，不另立一份算子名表。
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
