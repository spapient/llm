//! `axes` 模块的单元测试。

use super::*;



#[test]

fn raw_baseline_roundtrips() {

    let w: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 1.3).collect();

    let pick = search(&w, 8, 8).expect("搜索");

    assert_eq!(pick.axis, Axis::Raw);

    // 不可压缩数据走 store：只增加固定头与一个标记字节，有界常量，不被 rans 头部放大。

    assert!(pick.len() <= w.len() * 4 + HEAD + 1);

    // 核心：逐位往返。

    let got = decode(&pick.artifact).expect("解码");

    assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));

}



#[test]

fn dict_wins_on_low_cardinality() {

    // 200 个互不相同、尾数随机的 f32 周期排布：同一尺度网格上量化不着、64×64 上秩轴要给

    // 两万个字节的因子，只有字典帧（200 个数 + 每元素 1 字节）真正更小。

    let vals: Vec<f32> = (0..200u32)

        .map(|i| f32::from_bits(0x3f80_0000 | (i.wrapping_mul(2_654_435_761) & 0x007f_ffff)))

        .collect();

    let w: Vec<f32> = (0..4096).map(|i| vals[(i * 37) % 200]).collect();

    let pick = search(&w, 64, 64).expect("搜索");

    assert_eq!(pick.axis, Axis::Dict);

    assert!(pick.len() < w.len() * 4);

}



#[test]

fn sparse_roundtrips_zero_heavy() {

    let mut w = vec![0.0f32; 4096];

    for i in (0..4096).step_by(97) {

        w[i] = (i as f32).sqrt();

    }

    let pick = search(&w, 64, 64).expect("搜索");

    let got = decode(&pick.artifact).expect("解码");

    assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));

}



#[test]

fn quant_wins_on_exact_grid() {

    let w: Vec<f32> = (0..4096)

        .map(|i| {

            let code = ((i as u32).wrapping_mul(2654435761) >> 12) & 255;

            1.0 + code as f32 * 0.25

        })

        .collect();

    let pick = search(&w, 64, 64).expect("搜索");

    assert_eq!(pick.axis, Axis::Quant);

}



#[test]

fn rank_inexact_is_dropped_not_faked() {

    let w: Vec<f32> = (0..64)

        .map(|i| ((i as f32 * 1.7).cos() * 0.9) + 0.01)

        .collect();

    let pick = search(&w, 8, 8).expect("搜索");

    // 不逐位精确的低秩不得胜出；raw 兜底。

    assert_ne!(pick.axis, Axis::Rank);

}



/// 有谱隙的矩阵：秩轴产出载荷，载荷自带秩与残差、能逐位还原。

///

/// 这里不假设"秩轴一定在五轴里胜出"——显式残差是每个元素 4 字节，只有在矩阵**精确**低秩

/// （残差全零）时才比原字节便宜，那是结构性质而不是门控性质。门控性质只保证一件事：证书

/// 不过就没有这个候选。

#[test]

fn a_certified_matrix_yields_a_rank_payload_that_roundtrips() {

    // `w[i][j] = 64·i + j`：秩 2、值两两不同，且 f32 精确。

    let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

    let payload = rank_payload(&w, 64, 64).expect("谱隙存在，秩轴必须给出候选");

    let pick = frame(Axis::Rank, 64, 64, payload);

    let got = decode(&pick.artifact).expect("解码");

    assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));

}



/// 没有谱隙的矩阵不存在可信的秩：证书拒压，秩轴整轴不参与（不是压出一个近似值）。

#[test]

fn a_matrix_without_a_spectral_gap_skips_the_rank_axis() {

    let zeros = vec![0.0f32; 4096];

    assert!(

        rank_payload(&zeros, 64, 64).is_none(),

        "σ 全相等就没有可判秩"

    );

    assert_ne!(search(&zeros, 64, 64).expect("搜索").axis, Axis::Rank);

    // 单行/单列同理：`1 <= r < min(m,n)` 无解。

    let row: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();

    assert!(rank_payload(&row, 1, 64).is_none());

    assert_ne!(search(&row, 1, 64).expect("搜索").axis, Axis::Rank);

    let got = decode(&search(&zeros, 64, 64).expect("搜索").artifact).expect("解码");

    assert!(got.iter().all(|x| x.to_bits() == 0.0f32.to_bits()));

}



/// 证书随载荷落盘：不解码权重也能读到它，读到的就是写端那一张（不是重新算的）。

#[test]

fn the_rank_payload_carries_its_certificate() {

    let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

    let cert = ledger::certify(&w, 64, 64)

        .expect("认证")

        .expect("谱隙存在");

    let artifact = frame(Axis::Rank, 64, 64, rank_payload(&w, 64, 64).expect("候选")).artifact;

    let note = certificate(&artifact).expect("解析").expect("秩轴");

    assert_eq!(note, cert.note, "落盘的是认证那一次，不是读端重算的另一张");

    // 其余轴没有证书这一说：返回 `None` 而不是编一张空证书。

    let raw = frame(Axis::Raw, 64, 64, pack_bytes(&f32_bytes(&w))).artifact;

    assert!(certificate(&raw).expect("解析").is_none());

}



/// 证书被改写就解不出来：读端把截断误差当断言，而不是相信写端的声称。

///

/// 伪造的两处都落在「证书 + 载荷」两方对质能查出、而单看证书自身 `check` 查不出的位置：

/// 改写 `decomp`（`check` 只看它有限且与 `lossless` 一致），把 `stable` 翻成 `Fail`

/// （`check` 允许 `Fail`，但一条 `Fail` 的证书根本不该被落盘）。

#[test]

fn a_tampered_certificate_is_rejected_on_read() {

    let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

    let cert = ledger::certify(&w, 64, 64)

        .expect("认证")

        .expect("谱隙存在");

    let left = pack_bytes(&f64_bytes(&cert.left));

    let right = pack_bytes(&f64_bytes(&cert.right));

    let rest = pack_bytes(&f32_bytes(&cert.rest));

    let framed = |note: Vec<u8>| {

        let mut p = framed(&note);

        p.extend(framed(&left));

        p.extend(framed(&right));

        p.extend(rest.clone());

        frame(Axis::Rank, 64, 64, p).artifact

    };



    let honest = framed(cert.note.to_bytes());

    let got = decode(&honest).expect("原证书可读");

    assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));



    // 按字段伪造而不是改字节：`to_bytes` 的长度前缀是变长编码，偏移量不该在测试里写死。

    // 这张矩阵精确低秩，`decomp` 本就是 0（乘任何系数都不动），所以按加法伪造。

    let mut wrong_error = cert.note.clone();

    wrong_error.decomp += 1.0;

    assert!(

        decode(&framed(wrong_error.to_bytes())).is_err(),

        "截断误差被改写必须被拒"

    );



    // 稳定性翻成 `Fail`：`Fail` 本身合法，但一条没放行的证书不该出现在载荷里。

    let mut not_passed = cert.note.clone();

    not_passed.stable = ledger::Stable::Fail;

    assert!(

        decode(&framed(not_passed.to_bytes())).is_err(),

        "没放行的证书不能落盘"

    );

}



/// 旧布局（秩裸数字在最前）不再是合法载荷：第一段被读成证书长度，随后的字节读不成证书。

#[test]

fn the_rank_payload_requires_a_certificate() {

    let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

    let cert = ledger::certify(&w, 64, 64)

        .expect("认证")

        .expect("谱隙存在");

    // v3 布局：证书之后是一整张 `low`，它的长度前缀此刻被读成左因子长度。

    let low = pack_bytes(&f32_bytes(&cert.low));

    let mut old = framed(&cert.note.to_bytes());

    old.extend_from_slice(&(low.len() as u32).to_le_bytes());

    old.extend(low);

    old.extend(pack_bytes(&f32_bytes(&cert.rest)));

    assert!(decode(&frame(Axis::Rank, 64, 64, old).artifact).is_err());

}



#[test]

fn malformed_artifacts_are_errors() {

    assert!(decode(b"").is_err());

    assert!(decode(b"AX01").is_err());

    assert!(decode(b"BAD!000000000").is_err());

}



#[test]

fn bad_shape_is_an_error() {

    assert!(search(&[1.0], 3, 1).is_err());

}
