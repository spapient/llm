//! 职责：张量核读的 Bt（B 的 half 转置）布局前置判据。

use super::*;

/// 张量核的布局前置 `Bt[j][i] = half(B[i][j])` 逐元素**位相等**。
///
/// 这条前置此前完全未被对拍：`tensor_stage` 跑了转置，但只把 GEMM 的输出与参考比，而大形状的
/// 参考又会被跳过（`误差 NaN`）。若转置错位一列，整批 GEMM 全错却可能仍然「误差很小」地自洽。
///
/// `100×70` 两个方向都不被 32 整除，`32×32` 分块的边界守卫与共享内存补零都被覆盖。
#[test]
fn transpose_half_layout_is_bit_exact_including_tail_tiles() {
    const K: usize = 100;
    const N: usize = 70;

    let Some((device, queue)) = bench_device("half 转置布局") else {
        return;
    };
    let mut rng = Rng::new(0x5eed_0000_0000_0007);
    // 落在 half 正规区间：避开次正规与溢出的舍入差异，使「位相等」是唯一正确的期望。
    let b: Vec<f32> = (0..K * N).map(|_| rng.half_normal()).collect();

    let source = upload(&device, &b, "B");
    let mut transposed = device.new_buf(N * K * 2).expect("Bt 缓冲");
    let msl = crate::tiled::metal_transpose_half(K, N).expect("转置 MSL 应能生成");
    let pipe = device.compile(&msl).expect("转置 MSL 应能编译");
    // 派发几何与 `tensor_stage` 完全一致：线程组 [⌈N/32⌉, ⌈K/32⌉]、组内 32×8。
    assert!(
        queue.execute(
            &pipe,
            &[&source, &transposed],
            [(N as u32).div_ceil(32) * 32, (K as u32).div_ceil(32) * 8, 1],
            [32, 8, 1]
        ),
        "转置派发应成功"
    );

    let bytes = transposed.as_mut_bytes();
    assert_eq!(bytes.len(), N * K * 2, "Bt 应按 half 分配 N·K 个元素");
    let mut checked = 0usize;
    for j in 0..N {
        for i in 0..K {
            let at = (j * K + i) * 2;
            let got = half_to_f32(u16::from_ne_bytes([bytes[at], bytes[at + 1]]));
            let want = half_precision(b[i * N + j]);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "Bt[{j}][{i}] 与 half(B[{i}][{j}]) 不一致：{got:.8e} ≠ {want:.8e}"
            );
            checked += 1;
        }
    }
    println!("half 转置 {K}×{N}：{checked} 个元素逐位相等（含尾块）");
}
