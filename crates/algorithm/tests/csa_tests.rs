//! `csa` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;
use diagnostic::Code;

/// 分块因子钉死：m' ≫ m。
#[test]
fn frozen_compress_factors() {
    assert_eq!(CSA_M, 4);
    assert_eq!(HCA_M, 128);
    const { assert!(HCA_M > CSA_M) };
}

/// 有效长 = ⌈n/m⌉。
#[test]
fn effective_len_ceil_div() {
    assert_eq!(effective_len(100, CSA_M), 25);
    assert_eq!(effective_len(100, HCA_M), 1);
    assert_eq!(effective_len(7, 4), 2);
}

/// 分块压缩按 m 均值池化，末块不足 m 也正确。
#[test]
fn compress_mean_pools_blocks() {
    // n=4, d=2, m=2 → 2 块。
    let seq = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let got = compress(&seq, 4, 2, 2);
    assert_eq!(got, vec![2.0, 3.0, 6.0, 7.0]);
    // 末块不足：n=3, m=2 → 2 块，末块单行。
    let tail = compress(&[1.0, 10.0, 3.0, 30.0, 5.0, 50.0], 3, 2, 2);
    assert_eq!(tail, vec![2.0, 20.0, 5.0, 50.0]);
}

/// 分块压缩 f64 同式（泛型复用：与 f32 不复制实现）。
#[test]
fn compress_f64_agrees() {
    let seq = vec![1.0f64, 2.0, 3.0, 4.0];
    assert_eq!(compress(&seq, 4, 1, 2), vec![1.5, 3.5]);
}

/// DSA 每行只保留 top-k。
#[test]
fn dsa_topk_keeps_largest_k() {
    let mut s = vec![1.0f32, 5.0, 3.0, 4.0, 2.0];
    dsa_topk(&mut s, 1, 5, 2);
    let finite: Vec<f32> = s.iter().copied().filter(|x| x.is_finite()).collect();
    assert_eq!(finite.len(), 2);
    assert!(finite.contains(&5.0) && finite.contains(&4.0));
}

/// 稠密 softmax 行和为 1。
#[test]
fn dense_softmax_sums_to_one() {
    let mut s = vec![1.0f32, 2.0, 3.0];
    dense_softmax(&mut s, 1, 3);
    assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

/// Sinkhorn 收敛到双随机，且 ‖B‖₂ ≤ 1。
#[test]
fn sinkhorn_reaches_doubly_stochastic() {
    let mut b = vec![1.0f32, 3.0, 4.0, 2.0];
    let ok = sinkhorn(&mut b, 2, 200, 1e-5);
    assert!(ok, "Sinkhorn 应收敛到双随机");
    assert!(spectral_norm(&b, 2, 100) <= 1.0 + 1e-3);
}

/// 幂迭代对单位阵给出谱范数 1。
#[test]
fn spectral_norm_of_identity_is_one() {
    let mut id = vec![0.0f32; 9];
    for i in 0..3 {
        id[i * 3 + i] = 1.0;
    }
    assert!((spectral_norm(&id, 3, 100) - 1.0).abs() < 1e-4);
}

/// 稠密因果注意力（对拍基准）：位置 `t` 看 `0..=t` 的全部原始行。
fn dense(q: &[f32], k: &[f32], v: &[f32], dim: usize, t: usize) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut scores: Vec<f32> = (0..=t)
        .map(|i| (0..dim).map(|j| q[j] * k[i * dim + j]).sum::<f32>() * scale)
        .collect();
    dense_softmax(&mut scores, 1, t + 1);
    (0..dim)
        .map(|j| (0..=t).map(|i| scores[i] * v[i * dim + j]).sum())
        .collect()
}

/// 一行确定性的 K/V 夹具（不用随机数：对拍要可复跑）。
fn fixture(seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let key = (0..seq * dim)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1)
        .collect();
    let value = (0..seq * dim)
        .map(|i| ((i * 5 + 1) % 13) as f32 * 0.2)
        .collect();
    (key, value)
}

/// 块数 = ⌈seq/m⌉ = `effective_len`，压缩表长度与块数一致。
#[test]
fn a_block_table_holds_ceil_seq_over_m_blocks() {
    let (key, _) = fixture(100, 1);
    let table = BlockTable::new(&key, &key, 100, 1, CSA_M, WIN).expect("合法表");
    assert_eq!(table.blocks(), effective_len(100, CSA_M));
    assert_eq!(table.compressed().0.len(), 100 / CSA_M);
    assert_eq!(table.compressed().1.len(), 100 / CSA_M);
    let hca = BlockTable::new(&key, &key, 100, 1, HCA_M, WIN).expect("合法表");
    assert_eq!(hca.blocks(), 1, "100 行按 128 压成 1 块");
}

/// m = win = 1 时可见槽恰好是 `0..=t`，输出与稠密因果注意力一致。
#[test]
fn tight_geometry_matches_dense_causal_attention() {
    let (seq, dim) = (6, 3);
    let (key, value) = fixture(seq, dim);
    let table = BlockTable::new(&key, &value, seq, dim, 1, 1).expect("m = win = 1");
    for t in 0..seq {
        let q: Vec<f32> = (0..dim).map(|j| ((t + 3 * j) % 5) as f32 * 0.3).collect();
        let got = table.attend(&q, t, None).expect("合法查询");
        let want = dense(&q, &key, &value, dim, t);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "t = {t}：{got:?} ≠ {want:?}");
        }
    }
}

/// 因果：`t` 之后的行怎么改都不影响 `t` 的输出；`t` 之前的一改就影响。
#[test]
fn future_tokens_cannot_reach_the_past() {
    let (seq, dim) = (8, 2);
    let (key, value) = fixture(seq, dim);
    let (mut tail_key, mut tail_value) = (key.clone(), value.clone());
    for x in tail_key[5 * dim..]
        .iter_mut()
        .chain(tail_value[5 * dim..].iter_mut())
    {
        *x *= 100.0;
    }
    let before = BlockTable::new(&key, &value, seq, dim, CSA_M, 2).expect("合法表");
    let after = BlockTable::new(&tail_key, &tail_value, seq, dim, CSA_M, 2).expect("合法表");
    let q = [1.0f32, -0.5];
    for t in 0..5 {
        assert_eq!(
            before.attend(&q, t, None).expect("合法查询"),
            after.attend(&q, t, None).expect("合法查询"),
            "t = {t} 看见了未来"
        );
    }
    let mut past = key.clone();
    past[0] = 42.0;
    let shifted = BlockTable::new(&past, &value, seq, dim, CSA_M, 2).expect("合法表");
    assert_ne!(
        before.attend(&q, 4, None).expect("合法查询"),
        shifted.attend(&q, 4, None).expect("合法查询"),
        "改第一行必须改变 t = 4 的输出，否则上一条是空断言"
    );
}

/// top-k 只留分数最大的槽：k = 1 的输出就是最优槽的 V。
#[test]
fn topk_keeps_only_the_best_slot() {
    let k = [1.0f32, 2.0, 3.0, 4.0];
    let v = [10.0f32, 20.0, 30.0, 40.0];
    let table = BlockTable::new(&k, &v, 4, 1, 1, 1).expect("合法表");
    assert_eq!(table.attend(&[1.0], 3, Some(1)).expect("合法查询"), [40.0]);
    let dense_out = table.attend(&[1.0], 3, None).expect("合法查询");
    assert!(
        dense_out[0] > 30.0 && dense_out[0] < 40.0,
        "稠密是真加权：{dense_out:?}"
    );
}

/// 几何不合法与查询越界都在入口失败。
#[test]
fn a_block_table_rejects_bad_geometry_and_queries() {
    let k = [1.0f32, 2.0, 3.0, 4.0];
    for (dim, m, win) in [(0, 1, 1), (2, 0, 1), (2, 1, 0)] {
        assert_eq!(
            BlockTable::new(&k, &k, 2, dim, m, win).unwrap_err().code(),
            Code::Invalid,
            "dim = {dim}、m = {m}、win = {win} 必须在构造期被拒"
        );
    }
    assert_eq!(
        BlockTable::new(&k, &k, 3, 2, 1, 1).unwrap_err().code(),
        Code::Invalid,
        "K/V 长度必须是 seq × dim"
    );
    let table = BlockTable::new(&k, &k, 2, 2, 1, 1).expect("合法表");
    assert_eq!(
        table.attend(&[1.0], 0, None).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        table.attend(&[1.0, 2.0], 2, None).unwrap_err().code(),
        Code::Invalid,
        "位置 2 已经越过 seq = 2"
    );
}
