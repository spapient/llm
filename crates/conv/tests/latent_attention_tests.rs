//! 潜空间注意力的语义锚与「内核 ↔ 参考」对拍。

use crate::attention::latent::{forward, reference, sample};

/// 一条样本的全部缓冲：`q`/`k` 宽 `W`，`v`/`y` 宽 `dc`。
fn buffers(sample: &crate::model::Sample) -> Vec<&[f32]> {
    ["q", "k", "v"]
        .iter()
        .map(|name| sample.f32(name).expect("样本带这三个缓冲"))
        .collect()
}

#[test]
fn zero_scores_degrade_to_uniform_average_of_the_compressed_values() {
    let s = sample(2, 8);
    let q = vec![0.0f32; 2 * 8];
    let bufs = buffers(&s);
    let (k, v) = (bufs[1], bufs[2]);
    let mut y = vec![0.0f32; 2 * 4];
    forward(&q, k, v, &mut y, 8).expect("长度自洽");
    // 全零查询 ⇒ 分数全等 ⇒ softmax 均匀 ⇒ 输出是压缩 v 的行均值。
    let mean: Vec<f32> = (0..4).map(|c| (v[c] + v[4 + c]) / 2.0).collect();
    for row in 0..2 {
        for c in 0..4 {
            assert!((y[row * 4 + c] - mean[c]).abs() < 1e-6, "行 {row} 列 {c}");
        }
    }
}

#[test]
fn a_single_key_value_row_passes_through_unchanged() {
    let q = vec![0.3f32, -0.7, 0.2, 0.1, 0.5, -0.5, 0.9, -0.2];
    let k = vec![0.4f32, 0.1, -0.3, 0.6, 0.2, -0.8, 0.1, 0.3];
    let v = vec![-1.5f32, 2.5, 0.25];
    let mut y = vec![0.0f32; 3];
    forward(&q, &k, &v, &mut y, 8).expect("长度自洽");
    // 只有一条键 ⇒ 权重恒为 1 ⇒ 输出就是那一行压缩值。
    assert_eq!(y, v);
}

#[test]
fn the_rope_tail_moves_the_scores() {
    let s = sample(2, 8);
    let bufs = buffers(&s);
    let (q, v) = (bufs[0], bufs[2]);
    let mut base = vec![0.0f32; 2 * 8];
    let mut moved = vec![0.0f32; 2 * 8];
    base.copy_from_slice(bufs[1]);
    moved.copy_from_slice(bufs[1]);
    // 只改**第二条键**的尾段（RoPE 列）：同一查询行的各头共享这段，改它必然改分数。
    for c in 4..8 {
        moved[8 + c] = base[8 + c] + 3.0;
    }
    let (mut ya, mut yb) = (vec![0.0f32; 8], vec![0.0f32; 8]);
    forward(q, &base, v, &mut ya, 8).expect("长度自洽");
    forward(q, &moved, v, &mut yb, 8).expect("长度自洽");
    assert!(
        ya.iter().zip(&yb).any(|(a, b)| (a - b).abs() > 1e-4),
        "尾段应当移动分数"
    );
}

#[test]
fn the_kernel_agrees_with_the_reference_on_the_declared_sample() {
    let s = sample(3, 8);
    let got = {
        let q = s.f32("q").expect("q");
        let k = s.f32("k").expect("k");
        let v = s.f32("v").expect("v");
        let mut y = vec![0.0f32; 3 * 4];
        forward(q, k, v, &mut y, 8).expect("长度自洽");
        y
    };
    let want = reference(&s).expect("参考可算");
    assert_eq!(got.len(), want.len());
    for (index, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-5, "第 {index} 项：{a} vs {b}");
    }
}

#[test]
fn the_output_width_may_be_narrower_than_the_read_width() {
    // R=3、T=3、W=10、dc=3：输出比读入窄，几何全部由长度给出，没有标量可依赖。
    let (r, t, w, dc) = (3usize, 3usize, 10usize, 3usize);
    let q: Vec<f32> = (0..r * w).map(|i| (i as f32 * 0.07).sin()).collect();
    let k: Vec<f32> = (0..t * w).map(|i| (i as f32 * 0.11).cos()).collect();
    let v: Vec<f32> = (0..t * dc).map(|i| i as f32 * 0.25 - 1.0).collect();
    let mut y = vec![0.0f32; r * dc];
    forward(&q, &k, &v, &mut y, w).expect("长度自洽");
    // 期望值手算：分数 → softmax → 加权。
    let scale = (w as f32).sqrt();
    for row in 0..r {
        let s: Vec<f32> = (0..t)
            .map(|j| (0..w).map(|c| q[row * w + c] * k[j * w + c]).sum::<f32>() / scale)
            .collect();
        let max = s.iter().copied().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = s.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = ex.iter().sum();
        for c in 0..dc {
            let want: f32 = (0..t).map(|j| ex[j] / sum * v[j * dc + c]).sum();
            assert!((y[row * dc + c] - want).abs() < 1e-5, "行 {row} 列 {c}");
        }
    }
}
