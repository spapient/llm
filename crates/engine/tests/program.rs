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
        host.contains(
            "dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)"
        ),
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
