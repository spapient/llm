//! `settle` 模块的单元测试。
use super::*;

use crate::ledger;
use crate::seg::Segment;
use diagnostic::Code;

use crate::testutil::{f32_matrix, f32_segment};

use crate::writer::Writer;

use crate::{Quant, Wire};

/// 写入并 settle：`compress` 决定写侧是否尝试熵编码。
fn settled(segments: Vec<Segment>, compress: bool) -> Settlement {
    container(segments, compress).0
}

/// 落盘、记账，并把字节一起留下：核账必须比的是同一份字节。
fn container(segments: Vec<Segment>, compress: bool) -> (Settlement, Vec<u8>) {
    let mut writer = Writer::new();

    writer.compress(compress);

    for segment in segments {
        writer.push(segment).expect("入写");
    }

    let bytes = writer.finish(6).expect("落盘");

    let settlement = Settlement::settle(&bytes).expect("记账");

    (settlement, bytes)
}

/// 精确秩 2 的矩阵：两对伪随机因子外积相加，谱隙没有噪声撑腰，证书签得出。
///
/// 因子刻意不取线性序列——线性矩阵的重复度高，熵编码会压得比秩轴更小，写侧也就不会选中秩轴。
fn values_low_rank(m: usize, n: usize) -> Vec<f32> {
    let lane = |i: usize, salt: usize| ((i * 37 + salt * 11) % 97) as f32 - 48.0;

    (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);

            lane(r, 1) * lane(c, 3) + lane(r, 5) * lane(c, 7)
        })
        .collect()
}

/// 0/1 交替的 f32 流：重复度高，写侧会选中熵编码。
fn compressible(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i % 2) as f32).collect()
}

/// 只有一条时取出它：多条说明用例自己的构造错了。
fn only(settlement: &Settlement) -> &Settled {
    let [entry] = settlement.tensors() else {
        panic!("应当只有一条，实际 {}", settlement.tensors().len());
    };

    entry
}

#[test]
fn a_rank_compressed_tensor_carries_a_sound_certificate() {
    // 精确 rank-2 且边长足够：因子化存的是 `(m+n)·r` 个数，整张 `L` 是 `m·n` 个数，收益随

    // 边长增长——16×16 这种小矩阵上熵编码反而更小，写侧不选秩轴是本分之内的正确选择。

    let settlement = settled(
        vec![f32_matrix(
            "blk.0.attn_q.weight",
            &values_low_rank(128, 128),
            (128, 128),
        )],
        true,
    );

    let entry = only(&settlement);

    assert_eq!(entry.reason(), Reason::Axes, "恰好低秩的矩阵应走轴帧");

    assert_eq!(entry.cert(), Cert::Diagnostic, "秩轴段必须带自洽证书");

    assert!(entry.render().contains("\"cert\": \"l0_diagnostic\""));
}

#[test]
fn a_tensor_that_never_used_rank_reports_no_certificate() {
    // 满秩且含噪声：五种轴都不更小，落成 `store`/`rans`，本来就没有证书可言。

    let values: Vec<f32> = (0..64).map(|i| ((i * 37) % 101) as f32 * 0.5).collect();

    let settlement = settled(vec![f32_segment("blk.0.mlp.w", &values)], true);

    assert_eq!(only(&settlement).cert(), Cert::None, "没走秩轴就没有证书");

    assert_eq!(only(&settlement).cert().as_str(), "none");
}

#[test]
fn a_certificate_that_names_another_measurement_is_refused() {
    let cert = crate::ledger::certify(&values_low_rank(4, 4), 4, 4)
        .expect("可认证")
        .expect("有谱隙");

    assert_eq!(sound_cert(&cert.note), Cert::Diagnostic, "签发端本来就自洽");

    for tampered in [
        ledger::Note {
            method: ledger::EpsMethod::Candidate,

            ..cert.note.clone()
        },
        ledger::Note {
            bound: ledger::Bound::Candidate,

            stable: ledger::Stable::Open,

            equal: false,

            lossless: false,

            ..cert.note.clone()
        },
        ledger::Note {
            comp: 1e-9,

            ..cert.note
        },
    ] {
        assert_eq!(sound_cert(&tampered), Cert::Refused, "{tampered:?}");
    }
}

#[test]
fn a_refused_certificate_forbids_the_book_from_calling_it_compressed() {
    // 判决只看两件事：省了字节、证书不矛盾。缺后者，字节再小也不叫 `compress`。

    let settlement = settled(
        vec![f32_segment("blk.0.attn_q.weight", &values_low_rank(16, 16))],
        true,
    );

    let mut entry = only(&settlement).clone();

    assert_eq!(entry.decision(), "compress", "有证书且更小");

    entry.cert = Cert::Refused;

    assert_eq!(entry.decision(), "reject", "证书背不住就退回原样");
}

#[test]
fn certificate_column_names_are_stable() {
    let names: Vec<&str> = [Cert::None, Cert::Diagnostic, Cert::Refused]
        .into_iter()
        .map(Cert::as_str)
        .collect();

    assert_eq!(names, ["none", "l0_diagnostic", "refused"]);
}

#[test]
fn a_stored_tensor_reports_its_dense_length_and_no_gain() {
    let values: Vec<f32> = (1..=64).map(|i| i as f32).collect();

    let settlement = settled(vec![f32_segment("backbone.layers.3.mlp.w", &values)], false);

    let entry = only(&settlement);

    assert_eq!(entry.name(), "backbone.layers.3.mlp.w");

    assert_eq!(entry.layer(), Some(3), "层号取名字里第一个整数分量");

    assert_eq!(entry.raw_len(), 256, "64 个 f32");

    assert_eq!(entry.volume(), entry.raw_len(), "原样落盘");

    assert_eq!(entry.volume_ratio(), 1.0);

    assert_eq!(entry.decision(), "reject", "没省就是 reject");

    assert_eq!(entry.reason(), Reason::Raw);

    assert!(entry.bitwise() && entry.lossless());
}

#[test]
fn a_compressed_tensor_reports_the_measured_gain() {
    let settlement = settled(vec![f32_segment("w", &compressible(1024))], true);

    let entry = only(&settlement);

    assert_eq!(entry.reason(), Reason::Rans);

    assert_eq!(entry.decision(), "compress");

    assert!(entry.volume() < entry.raw_len());

    let measured = entry.volume() as f64 / 4096.0;

    assert!((entry.volume_ratio() - measured).abs() < 1e-12);

    assert!(entry.bitwise(), "解出来的长度必须等于逻辑长度");

    assert!(entry.lossless());
}

#[test]
fn a_tiny_tensor_is_passed_through_instead_of_coded() {
    let settlement = settled(vec![f32_segment("bias", &[1.0, 2.0, 3.0])], true);

    let entry = only(&settlement);

    assert_eq!(entry.raw_len(), 12);

    assert_eq!(entry.reason(), Reason::Small);

    assert_eq!(entry.decision(), "reject");

    assert!(entry.lossless(), "原样落盘仍然无损");
}

#[test]
fn a_deduplicated_tensor_occupies_no_bytes_and_says_so() {
    let values = compressible(512);

    let settlement = settled(
        vec![f32_segment("w", &values), f32_segment("w.tied", &values)],
        true,
    );

    assert_eq!(settlement.tensors().len(), 2);

    let tied = &settlement.tensors()[1];

    assert_eq!(tied.volume(), 0, "引用段自身零字节");

    assert_eq!(tied.volume_ratio(), 0.0);

    assert_eq!(tied.reason(), Reason::Deduplicated);

    assert_eq!(tied.decision(), "compress");

    assert_eq!(tied.raw_len(), 2048, "逻辑长度仍是这张量自己的");
}

#[test]
fn a_quantized_tensor_is_never_lossless_and_reports_real_bits() {
    let quant = Quant::new(1, 1, 0, false, Wire::F16, None).expect("方案");

    let segment = Segment::new("q", Wire::U8, vec![64], Some(quant), vec![0u8; 64]).expect("段");

    let settlement = settled(vec![segment], false);

    let entry = only(&settlement);

    assert_eq!(entry.quant(), Some((8, 0)), "64 字节 64 元素即 8 bit");

    assert!(!entry.lossless(), "带量化方案的条目不是无损");

    assert!(!settlement.lossless());
}

#[test]
fn entropy_is_measured_on_the_logical_bytes() {
    let settlement = settled(vec![f32_segment("zeros", &[0.0; 64])], false);

    let entry = only(&settlement);

    assert!(
        entry.entropy() < 1.0,
        "全零字节的熵接近 0：{}",
        entry.entropy()
    );
}

#[test]
fn an_empty_tensor_does_not_divide_by_zero() {
    let segment = Segment::new("empty", Wire::F32, vec![0], None, Vec::new()).expect("段");

    let settlement = settled(vec![segment], true);

    let entry = only(&settlement);

    assert_eq!(entry.raw_len(), 0);

    assert_eq!(entry.volume_ratio(), 1.0);

    assert_eq!(entry.entropy(), 0.0);

    assert_eq!(settlement.total_ratio(), 1.0);
}

#[test]
fn the_ledger_carries_every_contracted_key_in_a_stable_order() {
    let settlement = settled(vec![f32_segment("l.0.w", &compressible(256))], true);

    let text = settlement.render();

    assert_eq!(text, settlement.render(), "同一份账要给出同一份文本");

    assert!(text.starts_with("{\n  \"schema\": \"llm.cert.v1\",\n  \"file_len\": "));

    for key in [
        "name",
        "layer",
        "raw_len",
        "volume_bytes",
        "volume_ratio",
        "entropy_H",
        "quant_bits",
        "quant_group_size",
        "verify_bitwise_equal",
        "is_lossless",
        "decision",
        "codec_reason",
    ] {
        assert!(text.contains(&format!("\"{key}\": ")), "缺字段 {key}");
    }

    assert!(text.ends_with("  ]\n}\n"));
}

#[test]
fn a_name_without_a_layer_idx_maps_to_null() {
    let settlement = settled(vec![f32_segment("embed", &compressible(64))], true);

    assert_eq!(only(&settlement).layer(), None);

    assert!(settlement.render().contains("\"layer\": null"));
}

#[test]
fn the_ledger_refuses_bytes_that_are_not_a_container() {
    let error = Settlement::settle(&[0u8; 8]).expect_err("不是容器");

    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn a_cell_this_bytes_cannot_produce_is_rejected() {
    let (settlement, bytes) = container(vec![f32_segment("a", &compressible(256))], true);

    let text = settlement.render();

    // 条目里多一格：`note` 现算不出来，账本却在描述它。

    let forged = text.replace(
        "\"codec_reason\": ",
        "\"note\": \"ok\",\n      \"codec_reason\": ",
    );

    assert_ne!(forged, text, "必须真插入一格");

    let error = Settlement::verify(&forged, &bytes).expect_err("条目多一格");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("note"), "{error}");

    // 顶层多一格同理：整份账只有那三个整体字段。

    let forged = text.replacen(
        "{\n  \"schema\"",
        "{\n  \"author\": \"nobody\",\n  \"schema\"",
        1,
    );

    assert_ne!(forged, text, "必须真插入一格");

    let error = Settlement::verify(&forged, &bytes).expect_err("顶层多一格");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("author"), "{error}");
}

#[test]
fn the_book_reads_back_as_itself() {
    let (settlement, bytes) = container(vec![f32_segment("a", &compressible(256))], true);

    let text = settlement.render();

    let read = Settlement::verify(&text, &bytes).expect("原样读回必须通过");

    assert_eq!(read.render(), text, "读回的就是同一份账");

    assert_eq!(read.tensors().len(), settlement.tensors().len());
}

#[test]
fn one_altered_cell_is_rejected_and_named() {
    let (settlement, bytes) = container(vec![f32_segment("a", &compressible(256))], true);

    let forged = settlement
        .render()
        .replace("\"volume_ratio\": ", "\"volume_ratio\": 9");

    let error = Settlement::verify(&forged, &bytes).expect_err("改了一格");

    assert_eq!(error.code(), Code::Invalid);

    // 消息要点名出错的那一格与它的主人，否则「账本有问题」无从修起。

    let message = error.to_string();

    assert!(message.contains("volume_ratio"), "{message}");

    assert!(message.contains("a 的"), "{message}");
}

#[test]
fn a_decimal_written_another_way_is_not_the_same_book() {
    // 账本是同一份字节的规范产物：`1` 与 `1.000000` 不是同一格，哪怕数值相等。

    let values: Vec<f32> = (1..=64).map(|i| i as f32).collect();

    let (settlement, bytes) = container(vec![f32_segment("a", &values)], false);

    let text = settlement.render();

    let spelled = "\"volume_ratio\": 1.000000";

    assert!(text.contains(spelled), "原样落盘的体积比应当是 1.000000");

    let forged = text.replace(spelled, "\"volume_ratio\": 1");

    let error = Settlement::verify(&forged, &bytes).expect_err("写法不同");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("1.000000"), "{error}");
}

#[test]
fn a_book_missing_a_cell_is_incomplete_rather_than_lenient() {
    let (settlement, bytes) = container(vec![f32_segment("a", &compressible(256))], true);

    let text = settlement
        .render()
        .replace("      \"is_lossless\": true,\n", "")
        .replace("      \"is_lossless\": false,\n", "");

    let error = Settlement::verify(&text, &bytes).expect_err("缺一格");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("is_lossless"), "{error}");
}

#[test]
fn a_book_for_another_container_is_rejected() {
    let (one, _) = container(vec![f32_segment("a", &compressible(256))], true);

    let (_, two) = container(
        vec![
            f32_segment("a", &compressible(256)),
            f32_segment("b", &compressible(256)),
        ],
        true,
    );

    let error = Settlement::verify(&one.render(), &two).expect_err("条目数不符");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("2 段"), "{error}");
}

#[test]
fn the_ledger_agrees_with_the_container_metrics() {
    let values = compressible(1024);

    let mut writer = Writer::new();

    writer
        .push(f32_segment("a", &values))
        .expect("入写")
        .push(f32_segment("b", &values))
        .expect("入写");

    let bytes = writer.finish(6).expect("落盘");

    let settlement = Settlement::settle(&bytes).expect("记账");

    let archive = Archive::parse(&bytes).expect("打开");

    let metrics = archive.metrics().expect("度量");

    assert_eq!(settlement.lossless(), metrics.lossless());

    assert_eq!(settlement.file_len(), bytes.len());
}

#[test]
fn every_reason_has_a_stable_name() {
    let names: Vec<&str> = [
        Reason::Raw,
        Reason::Rans,
        Reason::BelowGain,
        Reason::Incompressible,
        Reason::Small,
        Reason::Mismatch,
        Reason::Axes,
        Reason::Deduplicated,
    ]
    .iter()
    .map(|reason| reason.as_str())
    .collect();

    assert_eq!(
        names,
        [
            "raw",
            "rans",
            "below_gain",
            "incompressible",
            "passthrough:small",
            "fallback:mismatch",
            "axes",
            "deduplicated",
        ]
    );
}

#[test]
fn strings_from_the_file_are_escaped() {
    assert_eq!(escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");

    assert_eq!(escape("\u{1}"), "\\u0001");
}
