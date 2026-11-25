//! 网络级上界必须罩住**实测**误差，而且量在同一条内核跑出来的那份误差上。
//!
//! 两条用例问的是同一件事的两个尺度：一段权重上的界，与整批（多行输入）的界。后者证明的是
//! `√rows` 而不是 `rows`——逐行上界相同、行间独立，界按行数的平方根放大。

use weights::chain::{Chain, Rung};

use super::chain_measure::{distance, forward, frobenius, quantized, rung, scaled, two_layer};

#[test]
fn a_quantized_layer_stays_inside_the_l3_bound() {
    let (k, n, rows) = (32usize, 32usize, 1usize);
    let x = scaled(rows * k, 1, 1.0);
    let w1 = scaled(n * k, 2, 0.25);
    let w2 = scaled(n * n, 3, 0.25);
    let w2_hat = quantized(&w2);

    // 参考与压缩走同一个函数，只在第二层权重上分叉：差出来的就是量化造成的实测误差。
    let exact = two_layer(&x, &w1, &w2, rows, k, n);
    let compressed = two_layer(&x, &w1, &w2_hat, rows, k, n);

    // 段序即执行序：线性 → ReLU（精确，L = 1、b = 0、ε = 0）→ 量化后的线性。
    let relu = Rung::new(1.0, 1.0, 0.0, 1.0, 0.0).expect("ReLU 段");
    let chain = Chain::new(
        frobenius(&x),
        vec![rung(&w1, &w1, 1.0, 0.0), relu, rung(&w2, &w2_hat, 1.0, 0.0)],
    )
    .expect("链");
    let bound = chain.bracket().expect("界");

    let observed = distance(&exact, &compressed);
    assert!(observed > 0.0, "量化没造成任何差异，这条用例什么也没测到");
    assert!(
        observed <= bound.error(),
        "实测 {observed:.6e} 超出网络级上界 {:.6e}",
        bound.error()
    );
    // ρ 刻意取 Frobenius（`‖W‖₂ ≤ ‖W‖_F` 恒成立、无需先分解），所以界偏松是设计取舍而不是
    // 判据放宽：要紧的是它**必须罩住**实测值，并且是有限、非零的数。
    assert!(bound.error().is_finite() && bound.error() > observed);
    assert_eq!(chain.uncovered(), 0);
}

#[test]
fn a_batch_scales_with_the_square_root_of_its_rows() {
    let (k, n, rows) = (16usize, 16usize, 4usize);
    let x = scaled(rows * k, 4, 1.0);
    let w = scaled(n * k, 5, 0.25);
    let w_hat = quantized(&w);

    let exact = forward(&x, &w, rows, n, k);
    let compressed = forward(&x, &w_hat, rows, n, k);

    // 逐行上界相同、行间独立，所以整批的界是单条界的 √rows 倍（不是 rows 倍）。
    let widest = (0..rows)
        .map(|row| frobenius(&x[row * k..(row + 1) * k]))
        .fold(0.0f64, f64::max);
    let chain = Chain::new(widest, vec![rung(&w, &w_hat, 1.0, 0.0)]).expect("链");
    let bound = chain.bracket().expect("界").error() * (rows as f64).sqrt();

    let observed = distance(&exact, &compressed);
    assert!(observed > 0.0, "量化没造成任何差异，这条用例什么也没测到");
    assert!(
        observed <= bound,
        "实测 {observed:.6e} 超出整批界 {bound:.6e}"
    );
}
