//! `mla` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;

/// 确定性有限取值：`[−0.5, 0.5)` 的整数派生物，避开 0 与 1 这类退化值。
fn values<S: Scalar>(n: usize, salt: usize) -> Vec<S> {
    (0..n)
        .map(|i| S::from_f64(((i * 37 + salt * 11) % 13) as f64 / 13.0 - 0.5))
        .collect()
}

/// 第 `row` 行按 `scale` 缩放的取值：用来造"未来行被改过"的对照组。
fn scaled<S: Scalar>(width: usize, row: usize, scale: f64) -> Vec<S> {
    values::<S>(width, row)
        .into_iter()
        .map(|v| v.mul(S::from_f64(scale)))
        .collect()
}

/// 2 头、`d_h = 3`、`d_c = 4`、`d_h^R = 2` 的小层：`n_h · d_h · d_c = 24`。
fn layer() -> Mla<f32> {
    Mla::new(2, 3, 4, 2, &values(24, 1), &values(24, 2)).expect("层")
}

/// 顺序追加 `seq` 行；每行的取值随行号变。泛型在 `S` 上，`f64` 用例不再抄一遍循环。
fn cache<S: Scalar>(mla: &Mla<S>, seq: usize) -> Latent<S> {
    let mut cache = mla.empty_cache();
    for t in 0..seq {
        cache
            .push(&scaled(mla.latent(), t, 1.0), &scaled(mla.rope(), t, 1.0))
            .expect("追加");
    }
    cache
}

/// V3 的行宽记账：`d_c + d_h^R = 576`，对稠密 `2 · n_h · d_h = 32768` 是 1.76%。
#[test]
fn the_v3_latent_row_is_a_fraction_of_the_dense_row() {
    assert_eq!(row_width(V3_LATENT, V3_ROPE), 576);
    assert_eq!(dense_width(V3_HEADS, V3_HEAD_DIM), 32_768);
    assert!((ratio(V3_HEADS, V3_HEAD_DIM, V3_LATENT, V3_ROPE) - 0.0176).abs() < 5e-5);
    // 61 层 × 576 个元素 × 2 字节 = 70272 字节/token，与 lld §5.8.1 的记账一致。
    assert_eq!(V3_LAYERS * row_width(V3_LATENT, V3_ROPE) * 2, 70_272);
    // 几何不成立时不 panic：记账函数只回答"省不省"，配置对不对由构造器判。
    assert!(ratio(0, 0, 4, 2).is_infinite());
}

/// 缓存里只有两段：潜在量与解耦 RoPE 键，每 token 正好 `d_c + d_h^R` 个元素。
#[test]
fn only_the_latent_and_the_rope_key_are_cached() {
    let mla = layer();
    let cache = cache(&mla, 5);
    assert_eq!(cache.seq(), 5);
    assert_eq!(cache.cache_elements(), 5 * (4 + 2));
    assert_eq!(cache.cache_elements(), 5 * mla.row_width());
    assert!(
        cache.cache_elements() < 5 * mla.dense_width(),
        "潜在行必须比同头的稠密 K/V 行窄"
    );
    // 行视图就是两段本身：没有 `k^{no-nope}`、也没有 `V` 的位置。
    let (latent, rope) = cache.row(2).expect("第 2 行");
    assert_eq!((latent.len(), rope.len()), (4, 2));
    assert_eq!(latent, scaled::<f32>(4, 2, 1.0).as_slice());
}

/// 吸收式与升维参考给出同一个数：这是"潜在缓存等价"的唯一可验证形态。
#[test]
fn absorb_matches_the_upref_reference() {
    let mla = layer();
    let cache = cache(&mla, 5);
    let q_nope = values::<f32>(6, 7);
    let q_rope = values::<f32>(4, 9);
    let absorbed = mla
        .attend(&cache, &q_nope, &q_rope, 4, Path::Absorb)
        .expect("吸收");
    let upref = mla
        .attend(&cache, &q_nope, &q_rope, 4, Path::Upref)
        .expect("升维");
    assert_eq!(absorbed.len(), 6);
    for (a, b) in absorbed.iter().zip(&upref) {
        assert!((a - b).abs() < 1e-5, "吸收式 {a} 与升维参考 {b} 不等");
    }
}

/// 同一份泛型实现在 `f64` 下同样等价：两条路径不因精度复制代码。
#[test]
fn the_two_paths_agree_in_f64_too() {
    let mla: Mla<f64> = Mla::new(2, 3, 4, 2, &values(24, 1), &values(24, 2)).expect("层");
    let cache = cache(&mla, 5);
    let q_nope: Vec<f64> = values(6, 7);
    let q_rope: Vec<f64> = values(4, 9);
    let absorbed = mla
        .attend(&cache, &q_nope, &q_rope, 3, Path::Absorb)
        .expect("吸收");
    let upref = mla
        .attend(&cache, &q_nope, &q_rope, 3, Path::Upref)
        .expect("升维");
    for (a, b) in absorbed.iter().zip(&upref) {
        assert!(
            (a - b).abs() < 1e-12,
            "f64 下吸收式 {a} 与升维参考 {b} 不等"
        );
    }
}

/// 因果性：`t` 之后的行怎么改都不影响 `t` 的输出，`t` 之前（含 `t`）的一改就影响。
#[test]
fn future_tokens_cannot_reach_the_past() {
    let mla = layer();
    let q_nope = values::<f32>(6, 7);
    let q_rope = values::<f32>(4, 9);
    let base = cache(&mla, 5);

    let mut future_changed = mla.empty_cache();
    for t in 0..5 {
        let scale = if t > 2 { 7.0 } else { 1.0 };
        future_changed
            .push(
                &scaled(mla.latent(), t, scale),
                &scaled(mla.rope(), t, scale),
            )
            .expect("追加");
    }
    let mut past_changed = mla.empty_cache();
    for t in 0..5 {
        let scale = if t == 0 { 3.0 } else { 1.0 };
        past_changed
            .push(
                &scaled(mla.latent(), t, scale),
                &scaled(mla.rope(), t, scale),
            )
            .expect("追加");
    }
    for path in [Path::Absorb, Path::Upref] {
        assert_eq!(
            mla.attend(&base, &q_nope, &q_rope, 2, path).expect("基线"),
            mla.attend(&future_changed, &q_nope, &q_rope, 2, path)
                .expect("未来改动"),
            "{} 路径读到了位置 2 之后的行",
            path.as_str()
        );
        assert_ne!(
            mla.attend(&base, &q_nope, &q_rope, 2, path).expect("基线"),
            mla.attend(&past_changed, &q_nope, &q_rope, 2, path)
                .expect("过去改动"),
            "{} 路径没有读到位置 0 的行",
            path.as_str()
        );
    }
}

/// 解耦 RoPE 键直接进分数：换掉 `q^R` 就换输出，不能只看 `c^KV`。
#[test]
fn the_rope_key_enters_the_score_directly() {
    let mla = layer();
    let cache = cache(&mla, 3);
    let q_nope = values::<f32>(6, 7);
    let q_rope = values::<f32>(4, 9);
    let zeros = vec![0.0f32; 4];
    assert_ne!(
        mla.attend(&cache, &q_nope, &q_rope, 2, Path::Absorb)
            .expect("有 RoPE 查询"),
        mla.attend(&cache, &q_nope, &zeros, 2, Path::Absorb)
            .expect("无 RoPE 查询")
    );
}

/// 位置与查询宽度都在入口校验：越界不截断、宽度不符不猜。
#[test]
fn a_query_or_position_outside_the_geometry_is_rejected() {
    let mla = layer();
    let cache = cache(&mla, 5);
    let q_nope = values::<f32>(6, 7);
    let q_rope = values::<f32>(4, 9);
    assert!(
        mla.attend(&cache, &q_nope, &q_rope, 5, Path::Absorb)
            .is_err()
    );
    assert!(
        mla.attend(&cache, &q_nope[..5], &q_rope, 4, Path::Absorb)
            .is_err()
    );
    assert!(
        mla.attend(&cache, &q_nope, &q_rope[..3], 4, Path::Absorb)
            .is_err()
    );
    assert!(
        mla.attend(&mla.empty_cache(), &q_nope, &q_rope, 0, Path::Absorb)
            .is_err(),
        "空缓存没有可寻址的位置"
    );
    let other = Mla::new(2, 3, 3, 2, &values(18, 1), &values(18, 2)).expect("另一层");
    let foreign = other.cache(&values(18, 1), &values(12, 2)).expect("缓存");
    assert!(
        mla.attend(&foreign, &q_nope, &q_rope, 3, Path::Absorb)
            .is_err(),
        "别的层的潜在缓存不能拿来算"
    );
}

/// 构造期把长度与几何一次判清：错的形状活到寻址期只会整体错位。
#[test]
fn construction_checks_every_length() {
    let key = values::<f32>(24, 1);
    let value = values::<f32>(24, 2);
    assert!(Mla::new(0, 3, 4, 2, &key, &value).is_err());
    assert!(Mla::new(2, 0, 4, 2, &key, &value).is_err());
    assert!(Mla::new(2, 3, 0, 2, &key, &value).is_err());
    assert!(Mla::new(2, 3, 4, 0, &key, &value).is_err());
    assert!(Mla::new(2, 3, 4, 2, &key[..23], &value).is_err());
    assert!(Mla::new(2, 3, 4, 2, &key, &values::<f32>(25, 2)).is_err());

    let mla = layer();
    assert!(mla.cache(&values::<f32>(23, 1), &values(4, 2)).is_err());
    assert!(mla.cache(&values::<f32>(24, 1), &values(3, 2)).is_err());
    // 两段各自都是整数倍，但算出的 seq 不同：6 行 vs 4 行。
    assert!(mla.cache(&values::<f32>(24, 1), &values(8, 2)).is_err());

    let mut cache = mla.empty_cache();
    assert!(cache.push(&values::<f32>(3, 1), &values(2, 1)).is_err());
    assert!(cache.push(&values::<f32>(4, 1), &values(1, 1)).is_err());
    assert_eq!(cache.seq(), 0, "失败的追加不许推进一半的缓存");
    cache.push(&values(4, 1), &values(2, 1)).expect("追加");
    assert_eq!(cache.seq(), 1);
}

/// 路径短名稳定：报告与参数表按这个名字对齐。
#[test]
fn paths_have_stable_names() {
    assert_eq!(Path::Absorb.as_str(), "absorb");
    assert_eq!(Path::Upref.as_str(), "upref");
}
