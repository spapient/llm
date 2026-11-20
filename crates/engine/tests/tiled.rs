use super::*;

/// `tensor_geometry` 对任意形状都必须产出矩阵乘单元装得下的几何：跨度在支持档位、规约块
/// 合法、B 片落在 32 KiB 线程组内存内，且 `M` 被行跨度整除（否则 `matmul2d` 整块读 A 会越界）。
#[test]
fn geometry_is_always_dispatchable() {
    let shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (512, 512, 4096),
        (1024, 1024, 4096),
        (2048, 2048, 4096),
        (128, 128, 96),
        (256, 512, 100),
        (64, 64, 1000),
    ];
    for (m, n, k) in shapes {
        let g = crate::tiled::tensor_geometry(m, n, k);
        assert!(g.spans_ok(), "{m}×{n}×{k} 跨度越界：{g:?}");
        assert!(g.block_ok(), "{m}×{n}×{k} 规约块/线程组数越界：{g:?}");
        assert!(
            g.fits_threadgroup_memory(),
            "{m}×{n}×{k} B 片超出线程组内存：{g:?}"
        );
        assert!(m.is_multiple_of(g.tile_m), "{m}×{n}×{k} M 未整块：{g:?}");
    }
}

/// 中小输出优先低占用率几何并让规约块整除 `K`：整除时生成的内核没有尾块分支，而形状小到
/// 内核只有零点几毫秒时这段开销的占比最高。
#[test]
fn small_output_prefers_exact_geometry() {
    let g = crate::tiled::tensor_geometry(1024, 1024, 1024);
    assert_eq!((g.tile_m, g.tile_n, g.simdgroups), (128, 64, 4));
    assert!(g.exact(1024), "规约块应整除 K：{g:?}");

    // `K` 不被任何可选档位整除时必须仍给出可派发几何，只是失去无尾块内核。
    let g = crate::tiled::tensor_geometry(1024, 1024, 1000);
    assert!(g.fits_threadgroup_memory(), "{g:?}");
    assert!(!g.exact(1000), "1000 无可选整除档位：{g:?}");
}

/// 大输出改取列跨度 128 配 8 个 simdgroup 换每块更高的 A/B 重用，规约块取装得下的最大档。
#[test]
fn large_output_prefers_reuse_geometry() {
    let g = crate::tiled::tensor_geometry(4096, 4096, 4096);
    assert_eq!((g.tile_m, g.tile_n, g.simdgroups), (128, 128, 8));
    assert_eq!(g.k_tile, 96, "应取装得下的最大规约块：{g:?}");
}
