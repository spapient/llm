//! `archive` 模块的单元测试。
use super::*;

use diagnostic::Code;
use std::borrow::Cow;

use crate::{Codec, Quant, Segment, Wire};

#[cfg(test)]
mod malformed_tests {

    use super::*;

    /// fuzz 实测崩溃产物：段计数被篡改为 `u32::MAX`，剩余字节远不够这么多描述符。
    /// 旧实现按计数直接 `with_capacity` 触发约 39 GiB 分配；现在计数先按剩余字节粗查。
    /// 稀疏非零段保持原始字节，其余正文显式零填充，避免无意义的长串十六进制零。
    fn fuzz_crash() -> [u8; 114] {
        let mut bytes = [0; 114];
        bytes[..30].copy_from_slice(&[
            0x2E, 0x75, 0x00, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x06, 0xff, 0xff, 0xff,
            0xff, 0x01, 0x01, 0x77, 0x14, 0x02, 0x02, 0x03, 0x00, 0x00, 0x18, 0xa2, 0x94, 0x87,
            0x9e, 0x0a,
        ]);
        bytes[48..51].copy_from_slice(&[0x80, 0x60, 0x64]);
        bytes[110..].copy_from_slice(&[0x80, 0x60, 0x64, 0x35]);
        bytes
    }

    #[test]
    fn a_count_exceeding_remaining_bytes_is_rejected_not_an_oom() {
        let error = Archive::parse(&fuzz_crash()).expect_err("谎报计数必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("规模"), "{error}");
    }
}

#[cfg(test)]
mod cow_tests {

    use super::*;

    use crate::seg::Segment;

    use crate::testutil::{archive, f32_segment};

    use crate::writer::Writer;

    testkit::test_cases! {
    /// 关闭压缩写出的段必然是 `Store`：`load` 必须返回 `Cow::Borrowed`，不复制逻辑字节。
    fn store_segment_loads_as_borrowed() {
        let mut w = Writer::new();

        w.compress(false);

        w.push(f32_segment("w", &[1.0, 2.0, 3.0, 4.0]))
            .expect("push");

        let bytes = w.finish(6).expect("字节");

        let src = Archive::parse(&bytes).expect("src");

        let seg = src.get("w").expect("store 段");

        let cow = src.load(seg).expect("store 加载");

        assert!(matches!(cow, Cow::Borrowed(_)), "Store 段必须借用");

        assert_eq!(&cow[..], &f32_segment("w", &[1.0, 2.0, 3.0, 4.0]).data);
    }

    /// 高偏斜 U8 段经 rANS 严格变小，写端选 `Rans`：`load` 必须返回 `Cow::Owned` 解压缓冲。
    fn rans_segment_loads_as_owned() {
        let mut sparse = vec![0u8; 4000];

        for i in (0..sparse.len()).step_by(10) {
            sparse[i] = (i % 7) as u8 + 1;
        }

        let seg = Segment::new("sparse", Wire::U8, vec![4000], None, sparse.clone()).expect("段");

        let bytes = archive(vec![seg]);

        let src = Archive::parse(&bytes).expect("src");

        let seg = src.get("sparse").expect("rans 段");

        assert_eq!(seg.codec(), Codec::Rans, "偏斜段应被压缩");

        let cow = src.load(seg).expect("rans 加载");

        assert!(matches!(cow, Cow::Owned(_)), "Rans 段必须解压为 Owned");

        assert_eq!(cow.len(), 4000);

        assert_eq!(&cow[..], &sparse[..]);
    }

    /// `store` 的帧长由容器自己声明，没有别的环节会拿它和描述符推出的逻辑长度比对：一段宣称
    /// 自己有 64 字节、实际只存了 4 字节的帧必须被 `load` 挡下，否则度量按描述符报
    /// `lossless=true`、`settle` 按往返报 `bitwise=false`，同一份字节两种结论。
    fn a_store_frame_shorter_than_its_descriptor_is_rejected() {
        let frame = [0u8; 4];

        let error = decode_frame(&frame, Codec::Store, 64).expect_err("短帧必须被挡下");

        assert_eq!(error.code(), Code::Invalid, "坏容器不是内部错误");
    }

    /// 长度一致时 `store` 仍是零拷贝借用——加核对不能把这条路径变成复制。
    fn a_store_frame_of_the_declared_length_still_borrows() {
        let frame = [7u8; 8];

        let cow = decode_frame(&frame, Codec::Store, 8).expect("长度一致");

        assert!(matches!(cow, Cow::Borrowed(_)), "Store 段必须借用");
    }
    }
}

#[cfg(test)]
mod ref_tests {

    use super::*;

    use crate::Codec;

    use crate::crc;

    use storage::leb;

    use crate::testutil::f32_segment;

    use crate::writer::{GLOBAL_CRC, MAGIC, SEG_CRC, VERSION};

    /// 手拼描述符条目：内联段带存储帧，引用段只带代表段下标。
    enum Raw {
        Inline(Segment, Vec<u8>),

        Ref(Segment, u32),
    }

    fn f16(name: &str) -> Segment {
        Segment::new(name, Wire::F16, vec![2], None, vec![0u8; 4]).expect("段")
    }

    /// 按 v2 布局手拼容器：align_log=0（无填充），两级 CRC 开启。
    fn encode_raw(raw: &[Raw]) -> Vec<u8> {
        let mut body = Vec::new();

        body.extend_from_slice(&MAGIC.to_le_bytes());

        body.extend_from_slice(&VERSION.to_le_bytes());

        body.extend_from_slice(&(GLOBAL_CRC | SEG_CRC).to_le_bytes());

        body.push(0);

        leb::write(&mut body, raw.len() as u32);

        let mut data = Vec::new();

        for item in raw {
            match item {
                Raw::Inline(seg, stored) => {
                    let seg_crc = crc::crc32(stored);

                    seg.encode_desc(&mut body, Codec::Store, stored.len() as u64, seg_crc)
                        .expect("内联描述符");

                    data.extend_from_slice(stored);
                }

                Raw::Ref(seg, rep) => seg.encode_ref_desc(&mut body, *rep).expect("引用描述符"),
            }
        }

        body.extend_from_slice(&data);

        let global = crc::crc32(&body);

        body.extend_from_slice(&global.to_le_bytes());

        body
    }

    testkit::test_cases! {
    fn equal_segments_share_one_inline_frame() {
        let a = f32_segment("a", &[1.0, 2.0]);

        let b = f32_segment("b", &[1.0, 2.0]);

        let bytes = crate::testutil::archive(vec![a.clone(), b]);

        let archive = Archive::parse(&bytes).expect("合法容器");

        assert_eq!(archive.segments().len(), 2);

        let sa = archive.get("a").expect("a");

        let sb = archive.get("b").expect("b");

        assert!(!sa.is_ref());

        assert!(sb.is_ref());

        assert_eq!(sb.rep_idx(), Some(0));

        assert_eq!(
            archive.load(sa).expect("a 字节"),
            archive.load(sb).expect("b 字节")
        );

        assert_eq!(archive.load(sb).expect("b 字节").as_ref(), &a.data[..]);

        let metrics = archive.metrics().expect("度量");

        assert_eq!(metrics.ref_hits(), 1);

        assert_eq!(metrics.ref_saved_bytes(), 8);
    }

    fn dangling_reference_is_rejected() {
        let raw = [
            Raw::Inline(f32_segment("a", &[1.0]), f32_segment("a", &[1.0]).data),
            Raw::Ref(f32_segment("b", &[1.0]), 5),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("悬空下标必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("越界"), "{error}");
    }

    fn self_reference_is_rejected() {
        let raw = [
            Raw::Inline(f32_segment("a", &[1.0]), f32_segment("a", &[1.0]).data),
            Raw::Ref(f32_segment("b", &[1.0]), 1),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("自引用必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("自引用"), "{error}");
    }

    fn reference_to_reference_is_rejected() {
        let raw = [
            Raw::Inline(f32_segment("a", &[1.0]), f32_segment("a", &[1.0]).data),
            Raw::Ref(f32_segment("b", &[1.0]), 0),
            Raw::Ref(f32_segment("c", &[1.0]), 1),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("代表段是引用必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("引用"), "{error}");
    }

    fn reference_cycle_is_rejected() {
        let raw = [
            Raw::Ref(f32_segment("a", &[1.0]), 1),
            Raw::Ref(f32_segment("b", &[1.0]), 0),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("引用成环必须报错");

        assert_eq!(error.code(), Code::Invalid);
    }

    fn mismatched_reference_metadata_is_rejected() {
        let raw = [
            Raw::Inline(
                f32_segment("a", &[1.0, 2.0]),
                f32_segment("a", &[1.0, 2.0]).data,
            ),
            Raw::Ref(f16("b"), 0),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("元数据不一致必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("元数据"), "{error}");
    }

    fn duplicate_names_are_rejected() {
        let raw = [
            Raw::Inline(f32_segment("w", &[1.0]), f32_segment("w", &[1.0]).data),
            Raw::Inline(f32_segment("w", &[2.0]), f32_segment("w", &[2.0]).data),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("重名必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("重复"), "{error}");
    }
    }
}

#[cfg(test)]
mod metrics_tests {

    use super::*;

    use crate::testutil::{archive, f32_segment};

    testkit::test_cases! {
    fn metrics_are_computed_from_live_bytes() {
        let segment = f32_segment("w", &[1.0, 2.0, 3.0]);

        let bytes = archive(vec![segment]);

        let archive = Archive::parse(&bytes).expect("合法容器");

        let metrics = archive.metrics().expect("度量");

        assert_eq!(metrics.entries(), 1);

        assert_eq!(metrics.params(), 3);

        assert_eq!(metrics.v_fp16(), 6);

        assert_eq!(metrics.v_artifact(), bytes.len());

        assert_eq!(metrics.ratio(), bytes.len() as f64 / 6.0);

        assert!(metrics.lossless());

        assert_eq!(metrics.ref_hits(), 0);

        assert_eq!(metrics.ref_saved_bytes(), 0);
    }

    fn a_quantized_segment_is_not_lossless() {
        let quant = Quant::new(1, 0, 32, false, Wire::F16, None).expect("方案");

        let segment =
            Segment::new("q", Wire::U8, vec![8, 64], Some(quant), vec![3u8; 512]).expect("量化段");

        let bytes = archive(vec![segment]);

        let archive = Archive::parse(&bytes).expect("合法容器");

        assert!(!archive.metrics().expect("度量").lossless());
    }
    }

    /// 按验收口径直造一份度量：`1/24` 门只有精确取值才谈得上边界，靠真实容器造不出来。
    fn metrics_of(params: u64, v_artifact: usize, lossless: bool) -> Metrics {
        let v_fp16 = params.saturating_mul(2);
        let ratio = if v_fp16 == 0 {
            0.0
        } else {
            v_artifact as f64 / v_fp16 as f64
        };
        Metrics {
            entries: usize::from(params > 0),
            params,
            v_artifact,
            v_fp16,
            ratio,
            lossless,
            ref_hits: 0,
            ref_saved_bytes: 0,
        }
    }

    testkit::test_cases! {
    fn ratio_param_divides_by_the_parameter_level_denominator() {
        let metrics = metrics_of(12, 3, true);

        assert_eq!(metrics.ratio_param(), 0.125);

        // 与 `ratio` 同分母：参考体积就是 `params × 2`，今天两条口径数值必然相等。
        assert_eq!(metrics.ratio_param(), metrics.ratio());
    }

    fn the_e8_gate_accepts_exactly_one_over_twenty_four_and_not_one_byte_more() {
        // 12 参数 ⇒ 分母 24 字节；1 字节正好落在门上，2 字节即出界。
        assert_eq!(metrics_of(12, 1, true).ratio_param(), crate::E8_RATIO);

        assert!(metrics_of(12, 1, true).e8_pass());

        assert!(!metrics_of(12, 2, true).e8_pass());
    }

    fn a_lossy_artifact_never_passes_the_e8_gate_however_small_it_is() {
        assert!(!metrics_of(12, 1, false).e8_pass());
    }

    fn an_empty_artifact_does_not_pass_the_e8_gate_vacuously() {
        let empty = metrics_of(0, 0, true);

        assert_eq!(empty.ratio_param(), 0.0);

        assert!(!empty.e8_pass());
    }

    /// 真实 checkpoint 的端到端 E8 度量：给 `E8_WITNESS=<file.safetensors>` 才跑。
    ///
    /// 走的是生产同一条路——导入 → 写 `.u` → 解析 → **整表解码** → 由落盘字节现算度量与裁决。
    /// 解码一步不省：`e8_pass` 是"账"，逐位还原才是"证"，两件一起打印才算一次 E8 运行。
    /// 没给变量就直接返回（这条是基准而非回归用例，不该在没有样本的机器上失败）。
    #[ignore = "需要真实 safetensors：E8_WITNESS=<路径>"]
    fn a_real_checkpoint_goes_through_the_e8_gate() {
        let Ok(source) = std::env::var("E8_WITNESS") else {
            return;
        };

        let segments = crate::import::segments_from_safetensors_file(std::path::Path::new(&source))
            .expect("导入");

        let out = std::env::temp_dir().join("llm-e8-witness.u");

        let mut writer = crate::Writer::new();

        for segment in segments {
            writer.push(segment).expect("装入");
        }

        writer.write_to_path(6, &out).expect("落盘");

        let bytes = std::fs::read(&out).expect("读回");

        std::fs::remove_file(&out).expect("清理临时容器");

        let archive = Archive::parse(&bytes).expect("合法容器");

        let mut decoded = 0usize;

        let mut refs = 0usize;

        let (mut axes, mut rans, mut store) = (0usize, 0usize, 0usize);

        for segment in archive.segments() {
            archive.load(segment).expect("逐位解码");

            decoded += 1;

            if segment.is_ref() {
                refs += 1;

                continue;
            }

            match segment.codec() {
                Codec::Axes => axes += 1,
                Codec::Rans => rans += 1,
                Codec::Store => store += 1,
            }
        }

        let metrics = archive.metrics().expect("度量");

        println!(
            "E8 witness {source}: entries={} decoded={} refs={refs} inline={} \
             V_artifact={} V_FP16={} ratio={:.6} ratio_param={:.6} lossless={} \
             inline_codecs[axes={axes} rans={rans} store={store}] e8_pass={}",
            metrics.entries(),
            decoded,
            metrics.entries() - refs,
            metrics.v_artifact(),
            metrics.v_fp16(),
            metrics.ratio(),
            metrics.ratio_param(),
            metrics.lossless(),
            metrics.e8_pass()
        );

        assert_eq!(decoded, metrics.entries(), "整表都要解码过");

        assert!(metrics.lossless(), "五轴选码与去重都必须逐位无损");
    }
    }
}

// ---- proof ----

/// 测试用一次性摘要，包装流式结构。
fn hash(data: &[u8]) -> Digest {
    let mut ctx = Sha256::new();

    ctx.update(data);

    ctx.finish()
}

fn hex(digest: Digest) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

testkit::test_cases! {
fn empty_vector() {
    assert_eq!(
        hex(hash(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

fn abc_vector() {
    assert_eq!(
        hex(hash(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

fn two_block_vector() {
    // 448 位（56 字节）恰好落在填充边界，外加多块长度，覆盖双块路径。

    let data = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

    assert_eq!(
        hex(hash(data)),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

fn streaming_equals_one_shot() {
    let data = vec![0x5au8; 200];

    let whole = hash(&data);

    let mut ctx = Sha256::new();

    let (chunks, remainder) = data.as_chunks::<7>();

    for chunk in chunks {
        ctx.update(chunk);
    }

    ctx.update(remainder);

    assert_eq!(ctx.finish(), whole);
}

fn distinct_inputs_distinct_hashes() {
    assert_ne!(hash(b"x"), hash(b"y"));
}
}

fn leaves(names: &[&str]) -> Vec<Digest> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| leaf(name, &[i as u8; 16]))
        .collect()
}

testkit::test_cases! {
fn every_idx_proves_against_root() {
    let names = ["a", "b", "c", "d", "e", "f", "g"];

    let tree = leaves(&names);

    let merkle_root = root(&tree);

    for (idx, digest) in tree.iter().enumerate() {
        let proof = prove(&tree, idx).expect("证明");

        assert!(
            verify_proof(digest, &proof, &merkle_root),
            "下标 {idx} 应验证通过"
        );

        // 证明里回填的必须是叶子原位，不是折叠到最后的那层下标（每层减半后恒为 0）。
        assert_eq!(proof.idx(), idx, "证明要记住叶子原位 {idx}");

        // 证明长度为 ceil(log2 n)。

        assert_eq!(proof.siblings.len(), 3);
    }
}

fn proof_rejects_tampered_leaf() {
    let tree = leaves(&["a", "b", "c", "d"]);

    let merkle_root = root(&tree);

    let proof = prove(&tree, 1).expect("证明");

    let fake = leaf("b", &[0xFF; 16]);

    assert!(!verify_proof(&fake, &proof, &merkle_root));
}

fn proof_rejects_wrong_root() {
    let tree = leaves(&["a", "b", "c"]);

    let other = leaves(&["x", "y", "z"]);

    let proof = prove(&tree, 0).expect("证明");

    assert!(!verify_proof(&tree[0], &proof, &root(&other)));
}

fn out_of_range_proof_is_none() {
    let tree = leaves(&["a", "b"]);

    assert!(prove(&tree, 5).is_none());
}

fn name_binding_blocks_swap() {
    // 同内容不同名的叶子哈希必须不同。

    assert_ne!(leaf("x", &[1; 8]), leaf("y", &[1; 8]));
}

fn empty_root_is_zero() {
    assert_eq!(root(&[]), [0u8; 32]);
}
}

// ---- lifecycle ----

use crate::gguf::Gguf;
use crate::import::{segments_from_gguf, segments_from_torch};
use crate::testutil::{archive, f32_segment, real_bin_bytes};
use crate::torch::Checkpoint;
use crate::writer::Writer;
use dtype::DType;

testkit::test_cases! {
fn release_strips_opt_and_marks_infer() {
    let bytes = archive(vec![
        f32_segment("w", &[1.0, 2.0]),
        f32_segment("opt.m.w", &[0.5, 0.5]),
    ]);
    let src = Archive::parse(&bytes).expect("src");

    let released = release(&src).expect("release");
    let names: Vec<&str> = released.iter().map(|s| s.name()).collect();
    assert!(names.contains(&"w"));
    assert!(!names.iter().any(|n| is_opt(n)));

    let released_bytes = archive(released);
    let released_archive = Archive::parse(&released_bytes).expect("x");
    assert_eq!(mode_of(&released_archive).expect("模式"), Mode::Infer);
}

fn missing_marker_defaults_infer() {
    let bytes = archive(vec![f32_segment("w", &[1.0])]);
    assert_eq!(
        mode_of(&Archive::parse(&bytes).expect("a")).expect("模式"),
        Mode::Infer
    );
}

fn train_marker_roundtrips() {
    let mut segments = vec![f32_segment("w", &[1.0])];
    segments.push(Mode::Train.to_segment().expect("标记"));
    let bytes = archive(segments);
    let parsed = Archive::parse(&bytes).expect("a");
    assert_eq!(mode_of(&parsed).expect("模式"), Mode::Train);
}

fn an_unknown_mode_code_is_rejected_not_fallen_back_to_infer() {
    let bad = Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![7]).expect("段");
    let bytes = archive(vec![f32_segment("w", &[1.0]), bad]);
    let parsed = Archive::parse(&bytes).expect("a");
    let error = mode_of(&parsed).expect_err("未知码必须报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("Mode"), "{error}");
}

fn xor_roundtrip_recovers_cur() {
    let base_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.0])]);
    let base = Archive::parse(&base_bytes).expect("base");
    let cur_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.5])]);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    let patch_bytes = archive(diff(&base, &cur).expect("diff"));
    let patch = Archive::parse(&patch_bytes).expect("patch");

    let joined = join(&base, &patch).expect("join");
    let expected = cur.load(&cur.segments()[0]).expect("cur 字节");
    assert_eq!(joined[0].data, &expected[..]);
}

fn small_change_patch_is_smaller() {
    let base_bytes = archive(vec![f32_segment("w", &[1.0; 4096])]);
    let base = Archive::parse(&base_bytes).expect("base");
    let mut changed = vec![1.0f32; 4096];
    changed[10] = 1.0001;
    let cur_bytes = archive(vec![f32_segment("w", &changed)]);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    let mut w = Writer::new();
    w.compress(true);
    for seg in diff(&base, &cur).expect("diff") {
        w.push(seg).expect("push");
    }
    assert!(w.finish(6).expect("patch").len() < cur_bytes.len());
}

fn missing_base_is_full_not_garbage() {
    let base_bytes = archive(vec![f32_segment("a", &[1.0])]);
    let base = Archive::parse(&base_bytes).expect("base");
    let cur_bytes = archive(vec![f32_segment("b", &[9.0])]);
    let cur = Archive::parse(&cur_bytes).expect("cur");

    let patch = diff(&base, &cur).expect("diff");
    assert_eq!(patch[0].name(), "b");
    assert_eq!(patch[0].data, f32_segment("b", &[9.0]).data);
}
}

// —— 格式 → 中性段流的桥接：`import::segments_from_*` 依赖本 crate 的 `Segment`，用例只能留在这里。
use crate::gguf_fixture as fixture;

testkit::test_cases! {
fn keeps_torch_tensors_as_neutral_segments() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let file = Checkpoint::parse(&raw).expect("解析");
    let segments = segments_from_torch(&file).expect("导入中性段流");
    assert_eq!(segments.len(), file.tensors().len());
    for (segment, tensor) in segments.iter().zip(file.tensors()) {
        assert_eq!(segment.name(), tensor.name());
        assert_eq!(segment.dtype(), tensor.dtype());
        assert_eq!(segment.shape(), tensor.shape());
        assert_eq!(segment.data, file.data(tensor).expect("字节"));
    }
}

fn propagates_unreadable_torch_checkpoint() {
    let error = Checkpoint::parse(b"definitely not a torch bin")
        .and_then(|file| segments_from_torch(&file));
    assert!(error.is_err(), "读不通的检查点必须显式失败");
}

fn maps_scalar_gguf_tensors_to_neutral_segments() {
    let raw = fixture::plain();
    let file = Gguf::parse(&raw).unwrap();
    let segments = segments_from_gguf(&file).expect("标量张量可导入");
    assert_eq!(segments.len(), 2);

    let f32 = &segments[0];
    assert_eq!(f32.name(), "f32.weight");
    assert_eq!(f32.dtype(), DType::F32);
    assert_eq!(f32.shape(), &[2]);
    assert_eq!(f32.data.len(), 8);

    let half = &segments[1];
    assert_eq!(half.name(), "half.weight");
    assert_eq!(half.dtype(), DType::F16);
    assert_eq!(half.shape(), &[2]);
    assert_eq!(half.data.len(), 4);
}

fn imports_every_block_kind_as_dequantized_f32() {
    let raw = fixture::standard();
    let file = Gguf::parse(&raw).unwrap();
    let segments = segments_from_gguf(&file).expect("块量化在导入时反量化");
    let values = |name: &str| -> Vec<f32> {
        let segment = segments.iter().find(|s| s.name() == name).expect("段");
        assert_eq!(segment.dtype(), DType::F32, "{name} 反量化后是 f32");
        segment
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    assert_eq!(
        values("q8.weight"),
        (0..32).map(|i| i as f32 - 16.0).collect::<Vec<f32>>()
    );
    // 夹具的块量化是 d=1.0、min=0、scales 全 1、qs 全 0x21，数值按 GGML 规范手推：
    // Q4_K 每 64 值是 32 个 1.0 后接 32 个 2.0；Q6_K 每 128 值是 64 个 -31.0 后接 64 个 -30.0。
    for group in values("q4.weight").chunks(64) {
        assert!(group[..32].iter().all(|&v| v == 1.0), "Q4_K 低位");
        assert!(group[32..].iter().all(|&v| v == 2.0), "Q4_K 高位");
    }
    for group in values("q6.weight").chunks(128) {
        assert!(group[..64].iter().all(|&v| v == -31.0), "Q6_K 低位");
        assert!(group[64..].iter().all(|&v| v == -30.0), "Q6_K 高位");
    }
    // Q5_K 的 qh 全 0x5A：组 0/1 是 1.0/18.0，组 2/3 是 17.0/2.0。
    for (g, chunk) in values("q5.weight").chunks(64).enumerate() {
        let (low, high) = if g < 2 { (1.0, 18.0) } else { (17.0, 2.0) };
        let (a, b) = chunk.split_at(32);
        assert!(a.iter().all(|&v| v == low), "Q5_K 组 {g} 低位");
        assert!(b.iter().all(|&v| v == high), "Q5_K 组 {g} 高位");
    }
}
}
