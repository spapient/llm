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
    /// 夹具按 v2 `.u` 魔数与版本写头，正文沿用原崩溃字节。
    const FUZZ_CRASH: [u8; 114] = [
        0x2E, 0x75, 0x00, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x06, 0xff, 0xff, 0xff, 0xff,
        0x01, 0x01, 0x77, 0x14, 0x02, 0x02, 0x03, 0x00, 0x00, 0x18, 0xa2, 0x94, 0x87, 0x9e, 0x0a,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x80, 0x60, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x60, 0x64, 0x35,
    ];

    #[test]
    fn a_count_exceeding_remaining_bytes_is_rejected_not_an_oom() {
        let error = Archive::parse(&FUZZ_CRASH).expect_err("谎报计数必须报错");

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

    /// 关闭压缩写出的段必然是 `Store`：`load` 必须返回 `Cow::Borrowed`，不复制逻辑字节。
    #[test]
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
    #[test]
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
    #[test]
    fn a_store_frame_shorter_than_its_descriptor_is_rejected() {
        let frame = [0u8; 4];

        let error = decode_frame(&frame, Codec::Store, 64).expect_err("短帧必须被挡下");

        assert_eq!(error.code(), Code::Invalid, "坏容器不是内部错误");
    }

    /// 长度一致时 `store` 仍是零拷贝借用——加核对不能把这条路径变成复制。
    #[test]
    fn a_store_frame_of_the_declared_length_still_borrows() {
        let frame = [7u8; 8];

        let cow = decode_frame(&frame, Codec::Store, 8).expect("长度一致");

        assert!(matches!(cow, Cow::Borrowed(_)), "Store 段必须借用");
    }
}

#[cfg(test)]
mod ref_tests {

    use super::*;

    use crate::Codec;

    use crate::crc;

    use wire::leb;

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

    #[test]
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

        assert_eq!(sb.rep_index(), Some(0));

        assert_eq!(
            archive.load(sa).expect("a 字节"),
            archive.load(sb).expect("b 字节")
        );

        assert_eq!(archive.load(sb).expect("b 字节").as_ref(), &a.data[..]);

        let metrics = archive.metrics().expect("度量");

        assert_eq!(metrics.ref_hits(), 1);

        assert_eq!(metrics.ref_saved_bytes(), 8);
    }

    #[test]
    fn dangling_reference_is_rejected() {
        let raw = [
            Raw::Inline(f32_segment("a", &[1.0]), f32_segment("a", &[1.0]).data),
            Raw::Ref(f32_segment("b", &[1.0]), 5),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("悬空下标必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("越界"), "{error}");
    }

    #[test]
    fn self_reference_is_rejected() {
        let raw = [
            Raw::Inline(f32_segment("a", &[1.0]), f32_segment("a", &[1.0]).data),
            Raw::Ref(f32_segment("b", &[1.0]), 1),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("自引用必须报错");

        assert_eq!(error.code(), Code::Invalid);

        assert!(error.to_string().contains("自引用"), "{error}");
    }

    #[test]
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

    #[test]
    fn reference_cycle_is_rejected() {
        let raw = [
            Raw::Ref(f32_segment("a", &[1.0]), 1),
            Raw::Ref(f32_segment("b", &[1.0]), 0),
        ];

        let error = Archive::parse(&encode_raw(&raw)).expect_err("引用成环必须报错");

        assert_eq!(error.code(), Code::Invalid);
    }

    #[test]
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

    #[test]
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

#[cfg(test)]
mod metrics_tests {

    use super::*;

    use crate::testutil::{archive, f32_segment};

    #[test]
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

    #[test]
    fn a_quantized_segment_is_not_lossless() {
        let quant = Quant::new(1, 0, 32, false, Wire::F16, None).expect("方案");

        let segment =
            Segment::new("q", Wire::U8, vec![8, 64], Some(quant), vec![3u8; 512]).expect("量化段");

        let bytes = archive(vec![segment]);

        let archive = Archive::parse(&bytes).expect("合法容器");

        assert!(!archive.metrics().expect("度量").lossless());
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

    #[test]
    fn ratio_param_divides_by_the_parameter_level_denominator() {
        let metrics = metrics_of(12, 3, true);

        assert_eq!(metrics.ratio_param(), 0.125);

        // 与 `ratio` 同分母：参考体积就是 `params × 2`，今天两条口径数值必然相等。
        assert_eq!(metrics.ratio_param(), metrics.ratio());
    }

    #[test]
    fn the_e8_gate_accepts_exactly_one_over_twenty_four_and_not_one_byte_more() {
        // 12 参数 ⇒ 分母 24 字节；1 字节正好落在门上，2 字节即出界。
        assert_eq!(metrics_of(12, 1, true).ratio_param(), crate::E8_RATIO);

        assert!(metrics_of(12, 1, true).e8_pass());

        assert!(!metrics_of(12, 2, true).e8_pass());
    }

    #[test]
    fn a_lossy_artifact_never_passes_the_e8_gate_however_small_it_is() {
        assert!(!metrics_of(12, 1, false).e8_pass());
    }

    #[test]
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
    #[test]
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
