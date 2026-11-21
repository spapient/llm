//! `codec` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;
use crate::Boundary;

/// 一张每槽都不同的表：池化与局部性要能区分不同输入，全零表会让它们永远相等。
fn filled<S: Scalar>(len: usize, dim: usize) -> Table<S> {
    let mut table = Table::new(len, dim);
    for k in 0..table.len() {
        let row: Vec<S> = (0..dim)
            .map(|d| S::from_f64(((k * dim + d) % 7) as f64 - 3.0))
            .collect();
        table.set(&[k as u32], 0, &row);
    }
    table
}

/// 测用的编码器：每槽都不同、种子固定。
fn encoder(len: usize, dim: usize, seed: u64) -> ByteEncoder<f32> {
    ByteEncoder::new(filled(len, dim), seed)
}

/// 只在这些下标上给高分：一条可复现的切分（`rise` 放到极大，只留全局阈值）。
fn scores(len: usize, boundaries: &[usize]) -> Vec<f32> {
    (0..len)
        .map(|i| if boundaries.contains(&i) { 6.0 } else { 0.1 })
        .collect()
}

/// 按 [`scores`] 的分数切分。
fn split(scores: &[f32]) -> Patches {
    Boundary::<f32>::Entropy {
        global: 5.0,
        rise: 1e9,
    }
    .split(scores)
    .expect("参数与分数都合法")
}

/// 窗口只取已有后缀：首字节六个窗口都是它自己，历史够长之后长度封顶在阶上。
#[test]
fn a_window_is_the_available_suffix() {
    let bytes = *b"abcdefghi";
    let first = windows(&bytes, 0);
    assert!(
        first.iter().all(|w| w.len() == 1),
        "首字节没有历史：六个窗口都只有它自己"
    );
    assert!(first.iter().all(|w| w.iter().all(|&b| b == b'a')));
    assert!(windows(&bytes, 2).iter().all(|w| w.len() == 3));
    let heights: Vec<usize> = windows(&bytes, 5).iter().map(|w| w.len()).collect();
    assert_eq!(heights, vec![3, 4, 5, 6, 6, 6], "阶 6 起被已有长度截住");
    for i in 0..bytes.len() {
        for window in windows(&bytes, i) {
            assert_eq!(window.last(), Some(&bytes[i]), "窗口只回看：每段都以它结尾");
        }
    }
}

/// 每个 patch 一行，行数就是全局 transformer 的遍数；空输入没有行。
#[test]
fn every_patch_gets_one_row() {
    let encoder = encoder(13, 4, 3);
    let bytes: Vec<u8> = (0..32_u8).map(|b| b.wrapping_add(64)).collect();
    let patches = split(&scores(bytes.len(), &[0, 8, 20]));
    let rows = encoder.encode(&bytes, &patches).expect("切分与输入同源");
    assert_eq!(rows.len(), patches.len() * encoder.dim());
    assert_eq!(patches.len(), 3);
    assert!(
        patches.len() < bytes.len(),
        "全局只跑 {} 遍，不是 {} 遍",
        patches.len(),
        bytes.len()
    );
    assert!(encoder.encode(&[], &split(&[])).expect("空输入").is_empty());
}

/// 前缀稳定：追加字节不改变已完成 patch 的表示——表示只看过去。
#[test]
fn a_prefix_leaves_finished_patches_alone() {
    let encoder = encoder(13, 4, 5);
    let bytes: Vec<u8> = (0..24_u8).map(|b| b.wrapping_mul(3)).collect();
    let full = split(&scores(bytes.len(), &[0, 8, 16]));
    let reference = encoder.encode(&bytes, &full).expect("同源");
    for cut in 0..=bytes.len() {
        let prefix = split(&scores(cut, &[0, 8, 16]));
        let rows = encoder.encode(&bytes[..cut], &prefix).expect("同源");
        compare_finished(&full, &reference, &rows, cut, encoder.dim());
    }
}

/// 在 `cut` 处收尾时，完整落在前缀里的那些 patch 必须逐位不变。
fn compare_finished(full: &Patches, reference: &[f32], rows: &[f32], cut: usize, dim: usize) {
    for (p, range) in full.ranges().enumerate() {
        if range.end > cut {
            continue;
        }
        let want = &reference[p * dim..(p + 1) * dim];
        let got = &rows[p * dim..(p + 1) * dim];
        assert_eq!(got, want, "截到 {cut} 时第 {p} 个 patch 的表示变了");
    }
}

/// 局部性：窗口是回看的，一个字节只改写自己与之后至多 `WIDEST − 1` 个字节的向量。
#[test]
fn a_changed_byte_moves_only_the_patches_it_reaches() {
    let encoder = encoder(17, 4, 9);
    let mut bytes: Vec<u8> = (0..32_u8).map(|b| b.wrapping_add(100)).collect();
    let patches = split(&scores(bytes.len(), &[0, 16]));
    let before = encoder.encode(&bytes, &patches).expect("同源");
    // 第 4 个字节的影响面是 `4..=11`，都留在第一个 patch（`0..16`）里。
    bytes[4] ^= 0xFF;
    let after = encoder.encode(&bytes, &patches).expect("同源");
    let dim = encoder.dim();
    assert_ne!(
        &before[..dim],
        &after[..dim],
        "改动的字节在第一个 patch 里，它的表示应当变"
    );
    assert_eq!(
        &before[dim..],
        &after[dim..],
        "第二个 patch 的字节没动：表示也不该动"
    );
}

/// 池化是 patch 内的加权和，不是取边界字节、也不是均匀平均。
#[test]
fn the_pool_is_an_attention_mix() {
    let encoder = encoder(5, 2, 1);
    let vectors = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0_f32; 2];
    encoder.pool(&vectors, &mut out);
    assert_ne!(out, vec![1.0, 1.0], "池化不是取边界字节");
    assert!(
        out.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "凸组合的每一维落在字节向量的极值之间：{out:?}"
    );
    assert!(
        out.iter().any(|&v| v > 0.0 && v < 1.0),
        "权重不是 0/1：{out:?}"
    );
}

/// 相同的字节向量池化回它自己：softmax 的权重和为 1，均匀内容不该被缩放。
#[test]
fn a_uniform_patch_pools_back_to_its_vector() {
    let encoder = encoder(5, 3, 2);
    let common = [0.25_f32, -1.5, 3.0];
    let vectors: Vec<f32> = common
        .iter()
        .cycle()
        .take(3 * common.len())
        .copied()
        .collect();
    let mut out = vec![0.0_f32; 3];
    encoder.pool(&vectors, &mut out);
    for (got, want) in out.iter().zip(common) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

/// 切分来自另一次输入就拒绝，而不是拿它去给这一次的字节分行。
#[test]
fn a_split_from_another_input_is_rejected() {
    let encoder = encoder(7, 2, 0);
    let other = split(&scores(5, &[0]));
    let error = encoder
        .encode(&[1, 2, 3, 4], &other)
        .expect_err("切分不是这次输入的");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("5"), "{error}");
}

/// 只让字节 `A` 那行有分量的投影：分布归一、贪心给出 `A`。
fn one_hot() -> ByteDecoder<f32> {
    let mut projection = vec![0.0_f32; BYTES];
    projection[b'A' as usize] = 2.0;
    ByteDecoder::new(1, &projection).expect("长度对得上")
}

/// 256-way 分布：通道数就是字节取值数，和为 1，贪心与分布一致。
#[test]
fn the_decoder_gives_one_distribution_over_every_byte_value() {
    let decoder = one_hot();
    let probs = decoder.probs(&[1.0]).expect("宽度对");
    assert_eq!(probs.len(), BYTES);
    let total: f32 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "分布归一：{total}");
    assert_eq!(decoder.byte(&[1.0]).expect("宽度对"), b'A');
    assert_eq!(
        decoder.logits(&[1.0]).expect("宽度对").len(),
        BYTES,
        "分数也是每字节一个"
    );
}

/// 并列时取最小字节值：同一份表示永远给出同一个字节。
#[test]
fn ties_go_to_the_smallest_byte_value() {
    let decoder = ByteDecoder::new(1, &vec![0.0_f32; BYTES]).expect("长度对得上");
    assert_eq!(decoder.byte(&[5.0]).expect("宽度对"), 0);
}

/// 几何在入口挡：宽度为零、投影长度不对、表示宽度不对，都报 `Invalid`。
#[test]
fn degenerate_geometry_fails_at_the_door() {
    assert_eq!(
        ByteDecoder::<f32>::new(0, &[])
            .expect_err("宽度为零")
            .code(),
        Code::Invalid
    );
    assert_eq!(
        ByteDecoder::<f32>::new(2, &[0.0; 3])
            .expect_err("长度不是 BYTES × width")
            .code(),
        Code::Invalid
    );
    let decoder = one_hot();
    for error in [
        decoder.logits(&[1.0, 2.0]).expect_err("宽了"),
        decoder.probs(&[]).expect_err("窄了"),
        decoder.byte(&[1.0, 2.0]).expect_err("宽了"),
    ] {
        assert_eq!(error.code(), Code::Invalid);
    }
}

/// `f64` 与 `f32` 是同一份算术，切分与表示只差精度。
#[test]
fn the_f64_encoder_agrees_with_the_f32_one() {
    let bytes: Vec<u8> = (0..16_u8).map(|b| b.wrapping_add(7)).collect();
    let patches = split(&scores(bytes.len(), &[0, 8]));
    let coarse = encoder(13, 3, 4).encode(&bytes, &patches).expect("同源");
    let fine = ByteEncoder::<f64>::new(filled(13, 3), 4)
        .encode(&bytes, &patches)
        .expect("同源");
    assert_eq!(coarse.len(), fine.len());
    for (a, b) in coarse.iter().zip(&fine) {
        assert!((f64::from(*a) - *b).abs() < 1e-6, "{a} vs {b}");
    }
}

/// 表与种子是读侧口令：桶数、宽度、种子都原样回显，写侧不在这层。
#[test]
fn the_encoder_reads_the_table_it_was_given() {
    let encoder = encoder(13, 4, 7);
    assert_eq!(encoder.buckets(), 13);
    assert_eq!(encoder.dim(), 4);
    assert_eq!(encoder.seed(), 7);
    assert_eq!(encoder.table().len(), 13);
}
