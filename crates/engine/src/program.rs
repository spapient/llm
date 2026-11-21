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
use op::decl::Grid;
use op::model::Sample;
use op::sig::Signature;

#[cfg(test)]
#[path = "../../testkit/src/kernel.rs"]
mod cutile_x_kernel;

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

/// 归约族：单段设备源 + 主机折叠，折叠算子在 [`fold_of`] 一处派生。
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
///
/// 零维网格没有可发射的线程形状——`dispatchThreads` 收的就是线程数，给 0 既不执行也不报错。
/// 这里报「没有并行实现」，让调用方显式回落到语义门，而不是拿一份不发射任何线程的驱动当通过。
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

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::execution::POINT_THREADS;
    use op::decl::Grid;
    use op::model::{Buffer, Values};

    const POINT_SOURCE: &str = "void cutile_scale_f32_kernel(const float *a, float *y, int N) {}\n";

    /// gemv 形签名与样本：与 `crates/testkit/src/kernel.rs` 共用一份取值。
    ///
    /// 用它是因为它的声明是**零维网格**（串行语义门那一档）且属可被拒绝的族，正好覆盖
    /// 「没有并行实现」的两条分支；逐点那条另用 [`pointwise`]，两者签名宽度不同。
    fn gemv() -> (Signature, Sample) {
        (cutile_x_kernel::signature(), cutile_x_kernel::sample())
    }

    /// 逐点签名与它自己的样本：签名里的缓冲名必须在样本里存在，否则装配先报“漂移”。
    fn pointwise() -> (Signature, Sample) {
        let signature = Signature::parse("scale_f32", POINT_SOURCE).expect("逐点签名");
        let sample = Sample {
            m: 4,
            n: 1,
            scalars: vec![("N", 4)],
            buffers: vec![
                Buffer {
                    name: "a",
                    values: Values::F32(vec![1.0; 4]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; 4]),
                },
            ],
        };
        (signature, sample)
    }

    fn launch(kind: Lower, items: [usize; 3], group: usize) -> Launch {
        Launch { kind, items, group }
    }

    #[test]
    fn a_pointwise_launch_folds_items_into_blocks() {
        let (signature, sample) = pointwise();
        let program = cuda(
            POINT_SOURCE,
            &signature,
            &sample,
            &launch(Lower::Pointwise, [1024, 1, 1], POINT_THREADS),
        )
        .expect("逐点可装配");
        assert!(
            program.contains("<<<4, 256>>>"),
            "1024 个工作项 / 每组 256 个 = 4 块：{program}"
        );
        // 末块不足一组仍要发射这一块：多余线程由内核自身的长度守卫跳过。
        let ragged = cuda(
            POINT_SOURCE,
            &signature,
            &sample,
            &launch(Lower::Pointwise, [1000, 1, 1], POINT_THREADS),
        )
        .expect("逐点可装配");
        assert!(ragged.contains("<<<4, 256>>>"), "{ragged}");
    }

    #[test]
    fn a_loop_launch_without_dims_is_the_serial_harness() {
        // 零维网格 = 串行：它的「并行程序」就是那一份单线程语义门，而不是另一份文本。
        let program = cuda(
            cutile_x_kernel::SOURCE,
            &gemv().0,
            &gemv().1,
            &launch(Lower::Loop(Grid { dims: Vec::new() }), [1, 1, 1], 1),
        )
        .expect("串行可装配");
        assert!(program.contains("cutile_launch<<<1, 1>>>"), "{program}");
    }

    #[test]
    fn metal_dispatches_one_thread_per_item() {
        let (signature, sample) = pointwise();
        let Source::Pair { device, host } = metal(
            POINT_SOURCE,
            &signature,
            &sample,
            &launch(Lower::Pointwise, [512, 1, 1], POINT_THREADS),
        )
        .expect("逐点可装配") else {
            panic!("Metal 并行程序是设备源 + 主机驱动两份");
        };
        assert!(device.contains("kernel void kernel_main("), "{device}");
        // `dispatchThreads` 收总线程数（不是块数）：工作项直接当线程，组内 `group` 个。
        assert!(
            host.contains("dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)"),
            "{host}"
        );
    }

    #[test]
    fn a_loop_launch_without_dims_has_no_metal_thread_shape() {
        let error = metal(
            cutile_x_kernel::SOURCE,
            &gemv().0,
            &gemv().1,
            &launch(Lower::Loop(Grid { dims: Vec::new() }), [1, 1, 1], 1),
        )
        .expect_err("零维网格在 Metal 上没有线程形状");
        assert_eq!(error.code(), Code::Unsupported, "{error}");
    }

    #[test]
    fn the_tile_families_report_unsupported_rather_than_a_wrong_source() {
        // 这几种形态的生产态入口各自存在（`engine::tiled` / `wmma` / flash），单源装配不认识它们。
        // 报 `Unsupported` 而不是硬拼一份源：调用方据此回落语义门，而不是把缺实现当成已验证。
        for kind in [
            Lower::TiledGemm {
                tile: 8,
                per_thread: 1,
            },
            Lower::TiledAttention { threads: 32 },
            Lower::SimdGroupGemm {
                row_frags: 1,
                col_frags: 1,
            },
            Lower::FlashAttention {
                threads: 32,
                block: 32,
            },
            Lower::WmmaGemm,
        ] {
            let launch = launch(kind, [1, 1, 1], 1);
            let cuda_error = cuda(cutile_x_kernel::SOURCE, &gemv().0, &gemv().1, &launch)
                .expect_err("CUDA 没有该形态的并行程序");
            assert_eq!(cuda_error.code(), Code::Unsupported, "{cuda_error}");
            let metal_error = metal(cutile_x_kernel::SOURCE, &gemv().0, &gemv().1, &launch)
                .expect_err("Metal 没有该形态的并行程序");
            assert_eq!(metal_error.code(), Code::Unsupported, "{metal_error}");
        }
    }

    #[test]
    fn a_block_family_on_metal_is_stages_plus_a_host_fold() {
        // 分块族在 Metal 侧的落点是 [`crate::metal::fold`]：段与折叠一体，不是「不支持」。
        let Source::Pair { device, host } = metal(
            cutile_x_kernel::SOURCE,
            &gemv().0,
            &gemv().1,
            &launch(
                Lower::ChunkReduce {
                    chunk: 4,
                    func: "_cutile_sum_f32".to_string(),
                },
                [4, 1, 1],
                1,
            ),
        )
        .expect("分块归约在 Metal 上是多段程序") else {
            panic!("Metal 分块程序是设备源 + 主机驱动两份");
        };
        assert!(device.contains("kernel void kernel_main("), "{device}");
        // 单段归约的折叠由主机做：部分结果留在输出缓冲里，按声明的 `_cutile_sum_f32` 求和。
        assert!(host.contains("acc += parts[g];"), "{host}");
    }

    #[test]
    fn the_host_fold_follows_the_declared_reduction() {
        assert_eq!(fold_of("_cutile_max_f32"), "max");
        assert_eq!(fold_of("_cutile_sum_f32"), "sum");
        // 没声明折叠算子时按求和：把“认不出”读成 max 会让归约结果静默变成窗口最大值。
        assert_eq!(fold_of("_cutile_unknown_f32"), "sum");
    }

    #[test]
    fn the_unsupported_error_names_the_vendor_and_the_kind() {
        let error = unsupported(
            "Metal",
            &Lower::FlashAttention {
                threads: 32,
                block: 32,
            },
        );
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("Metal"), "{error}");
        assert!(error.to_string().contains("FlashAttention"), "{error}");
    }
}
