/// 与非转置入口逐位对拍：同一组 `(rows, n, k)` 下，把 `w` 转置成 NT 布局后两边结果必须相等
/// （累加次序完全相同，只有权重寻址方式不同）。
#[test]
fn tn_matches_transposed_nt_bit_patterns() {
    for (rows, n, k) in [(1, 8, 64), (7, 13, 33), (16, 24, 256), (23, 8, 300)] {
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 3) % 17) as f32 * 0.25 - 2.0)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 5) % 11) as f32 * 0.5 - 2.5)
            .collect();
        let mut wt = vec![0.0f32; k * n];
        for j in 0..n {
            for q in 0..k {
                wt[q * n + j] = w[j * k + q];
            }
        }
        let mut got = vec![0.0f32; rows * n];
        let mut want = vec![0.0f32; rows * n];
        super::gemm_f32_tn(
            tensor::View::matrix(rows, k, &x).expect("x 二维"),
            tensor::View::matrix(n, k, &w).expect("w 二维"),
            &mut got,
        )
        .expect("TN 几何合法");
        crate::fast::gemm_f32(&x, &wt, rows, n, k, &mut want);
        assert_eq!(got, want, "TN 与 NT（转置后）应当逐位相同：{rows}×{n}×{k}");
    }
}
