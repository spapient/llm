#![deny(clippy::pedantic)]
//! dtype 的契约总表：MX 规范对照、闭集身份、块几何、可选互操作与随机属性。
//!
//! 五个面各自只断言一类事实，合并后仍按段落保持分层：每段前的注释是该段自己的说明。
//! 改一张行表只会让一处红灯：规范金标、跨闭集身份、几何性质、feature 门后的互操作、
//! 不受信输入的属性骨架互不代劳。

use core::str::FromStr;

use llm_dtype::mxtype::Scale;
use llm_dtype::{
    BlockFormat, DType, Element, Format, Layout, MxType, QType, UnknownCode, UnknownName,
};

// ---------------------------------------------------------------------------
// MX 规范对照
// ---------------------------------------------------------------------------
//
// MX 布局的规范对照：OCP MX v1.1 与 NVFP4 厂商布局的字面块。
//
// [`MxType`] 还没有解码器消费者，块几何过去只是「照规范推 + 自测」。这里把规范表本身
// 搬成金标：块字节与期望数值都逐字写死，不由本文件的辅助函数「先编码再解码」推出来——
// 编解码同源的话，打包顺序写反也照样自圆其说。块长、每块字节、元素位宽、半字节顺序或
// E8M0/E4M3 的偏置只要漂移，长度或数值就会先对不上。
//
// 两种块共用一张用例表与一套断言：它们的差别只有块长、块字节与 scale 编码，逐项抄两遍
// 只会让「新增一个 MX 格式忘了补断言」变得看不出来。
//
// 仍未覆盖：真实模型的 GGUF 权重字节流。那份对照要等 `weights` 接上 MXFP4 反量化，
// 所以本段是**规范对照**，不是解码实现的验收。

/// OCP MX v1.1 的 E2M1 数值（按 4 位码去掉符号位后的下标索引）。
const E2M1_MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// OCP MXFP4 的一块：`0x7F` 是 E8M0 的 `2^0`，其后 16 字节负载（低半字节在前）。
const MXFP4_BLOCK: [u8; 17] = [
    0x7F, 0x90, 0xB2, 0xD4, 0xF6, 0x90, 0xB2, 0xD4, 0xF6, 0x90, 0xB2, 0xD4, 0xF6, 0x90, 0xB2, 0xD4,
    0xF6,
];

/// 上面那一块的 32 个元素：0、-0.5、1、-1.5、2、-3、4、-6 四轮（逐项手写，不由打包推）。
const MXFP4_VALUES: [f32; 32] = [
    0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0, 0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0, 0.0,
    -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0, 0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0,
];

/// NVFP4 的微块：`0x38` 是 E4M3 的 `1.0`，其后 8 字节负载；每张量的 FP32 二级 scale 在块外。
const NVFP4_BLOCK: [u8; 9] = [0x38, 0x90, 0xB2, 0xD4, 0xF6, 0x90, 0xB2, 0xD4, 0xF6];

/// 上面那个微块的 16 个元素（与 OCP 块同一批码，重复两轮）。
const NVFP4_VALUES: [f32; 16] = [
    0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0, 0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0,
];

/// 一条规范金标：格式身份、块几何与逐项期望值都写成字面量。
struct Case {
    /// 被测格式。
    format: MxType,
    /// 规范规定的每块元素数。
    block_len: u32,
    /// 规范规定的每块字节数（负载 + 1 字节 scale）。
    block_bytes: u32,
    /// 块内元素格式。
    element: Element,
    /// 规范里的一块字节流，首字节是 scale。
    block: &'static [u8],
    /// 这一块解码后的逐项期望值。
    values: &'static [f32],
}

/// OCP 的 32 元素块与 NVFP4 的 16 元素微块。
const CASES: [Case; 2] = [
    Case {
        format: MxType::MxFp4E2M1,
        block_len: 32,
        block_bytes: 17,
        element: Element::E2M1,
        block: &MXFP4_BLOCK,
        values: &MXFP4_VALUES,
    },
    Case {
        format: MxType::NvFp4,
        block_len: 16,
        block_bytes: 9,
        element: Element::E2M1,
        block: &NVFP4_BLOCK,
        values: &NVFP4_VALUES,
    },
];

/// E8M0 共享指数：`2^(value - 127)`；`0xFF` 是 NaN，本向量不使用。
fn e8m0_scale(value: u8) -> f32 {
    assert_ne!(value, u8::MAX, "0xFF 是 NaN，不是可比较的 scale");
    2f32.powi(i32::from(value) - 127)
}

/// E4M3（OCP MX v1.1，指数偏置 7）微块 scale：本向量只用 `1.0`，这里按位域完整解一遍。
fn e4m3_scale(value: u8) -> f32 {
    assert_eq!(value & 0x80, 0, "本向量不接受负 scale");
    let exponent = i32::from((value >> 3) & 0x0F);
    let mantissa = f32::from(value & 0x07) / 8.0;
    2f32.powi(exponent - 7) * (1.0 + mantissa)
}

/// 按格式声明的 scale 种类解首字节：把 NVFP4 的 E4M3 当 E8M0（或反过来）数值立刻对不上，
/// 所以 `scale` 这一列也被算了一遍。
fn block_scale(format: MxType, byte: u8) -> f32 {
    match format.scale() {
        Scale::E8M0 => e8m0_scale(byte),
        Scale::E4M3 => e4m3_scale(byte),
    }
}

/// 按 E2M1 表解负载：每字节先低半字节后高半字节，4 位码的低 3 位是数值、最高位是符号。
fn decode_payload(scale: f32, payload: &[u8]) -> Vec<f32> {
    let mut values = Vec::with_capacity(payload.len() * 2);
    for byte in payload {
        for nibble in [byte & 0x0F, byte >> 4] {
            let magnitude = E2M1_MAGNITUDE[usize::from(nibble & 0x07)];
            values.push(if nibble & 0x08 == 0 {
                scale * magnitude
            } else {
                -scale * magnitude
            });
        }
    }
    values
}

#[test]
fn mx_blocks_match_the_specification() {
    for case in &CASES {
        let format = case.format;
        assert_eq!(format.element(), case.element, "{format:?} 的元素格式");
        assert_eq!(format.block_len(), case.block_len, "{format:?} 的块长");
        assert_eq!(
            format.block_bytes(),
            case.block_bytes,
            "{format:?} 的每块字节"
        );

        // 字面块与字面期望值：长度、块字节、半字节顺序、数值表四项里任一项漂移都在这里对不上。
        assert_eq!(
            u64::try_from(case.block.len()).expect("块长远小于 u64 上限"),
            format.bytes_for(u64::from(case.block_len)),
            "{format:?} 的字面块长度与 `bytes_for` 不符"
        );
        assert_eq!(
            case.values.len(),
            usize::try_from(case.block_len).expect("块长放得进 usize"),
            "{format:?} 的期望值个数必须等于块长"
        );
        assert_eq!(
            decode_payload(block_scale(format, case.block[0]), &case.block[1..]).as_slice(),
            case.values,
            "{format:?} 的解码值与规范表不符"
        );
    }
}

// ---------------------------------------------------------------------------
// 闭集身份
// ---------------------------------------------------------------------------
//
// 身份契约：四个格式闭集共用一套读法——判别值即下标、规范名唯一且能双向解析、
// `Display` 与 `FromStr` 读同一张行表，并且 trait 视角与 `const fn` 视角指向同一行。
//
// 这里只断言**跨闭集共有**的身份契约；每个闭集表里的字面事实（位宽、线上 code、块字节、
// scale、编码族）由 `src/*.rs` 的内联行表逐条锁定。分层的目的是让每个事实只有一个断言点：
// 改一张表只会让一处红灯，而不是两个文件各写一遍同一个期望值。

/// 闭集身份：下标、规范名、`Display`、`FromStr` 与按名解析必须指向同一行。
///
/// 不需要 `F: Display` 这个额外 bound：[`Format`] 已经把 `Display` 算作闭集契约的一部分。
fn assert_identity<F>(unknown: &str)
where
    F: Format + FromStr,
{
    for (idx, format) in F::ALL.iter().enumerate() {
        assert_eq!(format.idx(), idx, "{format:?} 的判别值必须等于行表下标");
        assert_eq!(
            format.name().parse::<F>().ok(),
            Some(*format),
            "{format:?} 的 `FromStr` 与规范名分家"
        );
        assert_eq!(
            Format::parse_name(format.name()),
            Some(*format),
            "{format:?} 的规范名不能往返"
        );
        assert_eq!(
            format!("{format}"),
            format.name(),
            "{format:?} 的 `Display` 与规范名分家"
        );
    }
    assert!(unknown.parse::<F>().is_err(), "`{unknown}` 不在闭集里");
    assert_eq!(F::parse_name(unknown), None, "`{unknown}` 不能被解析");

    let mut names: Vec<&str> = F::ALL.iter().map(|format| format.name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        F::ALL.len(),
        "规范名必须唯一（解析不能有歧义）"
    );
}

/// 泛型 trait 视角与 inherent `const fn` 视角必须读出同一份事实：按名解析只能走 trait，
/// 常量求值只能走 inherent，两条路读的必须是同一张行表。
///
/// 两个视角没法写成泛型断言：inherent 方法与 trait 方法同名，泛型上下文里只会解析到 trait。
/// 所以按具体类型展开——新增一个闭集在这里加一行，多一个块格式加一个 `block`。
macro_rules! assert_one_table {
    ($format:ty) => {
        assert_one_table!($format, @identity);
    };
    ($format:ty, block) => {
        assert_one_table!($format, @identity);
        for format in <$format>::ALL {
            assert_eq!(
                BlockFormat::block_len(*format),
                <$format>::block_len(*format),
                "{format:?} 的块长两个视角分家"
            );
            assert_eq!(
                BlockFormat::block_bytes(*format),
                <$format>::block_bytes(*format),
                "{format:?} 的块字节两个视角分家"
            );
            assert_eq!(
                BlockFormat::block_count(*format, 300),
                <$format>::block_count(*format, 300),
                "{format:?} 的块数两个视角分家"
            );
            assert_eq!(
                BlockFormat::bytes_for(*format, 300),
                <$format>::bytes_for(*format, 300),
                "{format:?} 的字节数两个视角分家"
            );
        }
    };
    ($format:ty, @identity) => {
        for format in <$format>::ALL {
            assert_eq!(
                Format::idx(*format),
                *format as usize,
                "{format:?} 的判别值必须等于行表下标"
            );
            assert_eq!(
                Format::name(*format),
                <$format>::name(*format),
                "{format:?} 的规范名两个视角分家"
            );
            assert_eq!(
                Format::bits(*format),
                <$format>::bits(*format),
                "{format:?} 的位宽两个视角分家"
            );
            // `DType::parse_name(..)` 这类路径调用不导入 `Format`，是 `annotation` / `prof` /
            // `weights` 实际用的入口；inherent 只转发，不能读另一张表。
            assert_eq!(
                <$format>::parse_name(<$format>::name(*format)),
                Some(*format),
                "{format:?} 的按名解析两个视角分家"
            );
        }
    };
}

#[test]
fn every_closed_set_obeys_the_identity_contract() {
    // 四个闭集跑同一套断言：新增一个元素、一个 MX 格式或一个标量类型不会只补到其中一处。
    assert_identity::<DType>("i7");
    assert_identity::<QType>("q4_9");
    assert_identity::<MxType>("mxfp3");
    assert_identity::<Element>("e1m1");
}

#[test]
fn the_trait_view_forwards_to_the_const_accessors() {
    // 身份与块几何是同一份事实的两个视角：trait 只转发，不能另写一份公式。
    assert_one_table!(DType);
    assert_one_table!(QType, block);
    assert_one_table!(MxType, block);
    assert_one_table!(Element);
}

#[test]
fn parse_errors_say_what_went_wrong_without_allocating() {
    // 错误类型也是公开 API：描述固定，不因为输入是外部字符串就分配、也不夹带原文。
    assert_eq!(UnknownName.to_string(), "unknown format name");
    assert_eq!(UnknownCode { code: 0 }.to_string(), "unknown dtype code 0");
    assert_eq!(
        UnknownCode { code: u16::MAX }.to_string(),
        "unknown dtype code 65535"
    );
}

// ---------------------------------------------------------------------------
// 块几何
// ---------------------------------------------------------------------------
//
// 块几何契约：块数向上取整、字节数随元素数单调不减且在溢出时饱和，标量与块两条手臂
// 共用同一份 [`Layout`]。
//
// 断言只依赖公开的几何方法与 `ALL`，不重算实现里的公式：单调性与饱和是**性质**，
// 具体字节数由 `src/*.rs` 的行表逐格式锁定。公式改写成等价形式时测试不应误报。
//
// trait 视角与 `const fn` 视角的对拍（含块几何）在上面「闭集身份」段：同一份事实只有
// 一个断言点，这里不再重复一遍。

/// 块格式：整块向上取整，`bytes_for` 永远是「块数 × 每块字节」。
fn assert_whole_blocks<F: BlockFormat>() {
    for format in F::ALL {
        let len = u64::from(format.block_len());
        assert!(len > 0, "{format:?} 的块长不能为 0");
        assert_eq!(format.block_count(0), 0, "{format:?} 的空输入不占块");
        assert_eq!(format.block_count(len), 1, "{format:?} 的恰好一块");
        assert_eq!(format.block_count(len + 1), 2, "{format:?} 的跨块补齐");
        assert_eq!(format.bytes_for(0), 0, "{format:?} 的空输入不占字节");
        assert_eq!(
            format.bytes_for(len),
            u64::from(format.block_bytes()),
            "{format:?} 的整块字节"
        );
        assert_eq!(
            format.bytes_for(len - 1),
            u64::from(format.block_bytes()),
            "{format:?} 的不足一块按整块计"
        );
        assert_eq!(
            format.bytes_for(len * 3),
            3 * u64::from(format.block_bytes()),
            "{format:?} 的三块字节"
        );
    }
}

/// 字节数随元素数单调不减：抽样扫描 0..=4096。
///
/// 块格式与标量共用这一份性质，不按轴手抄两遍。
fn assert_monotonic(bytes_of: impl Fn(u64) -> u64, label: &str) {
    let mut previous = 0;
    for numel in (0..=4096u64).step_by(37) {
        let bytes = bytes_of(numel);
        assert!(
            bytes >= previous,
            "{label} 在 {numel} 个元素处字节数倒退：{bytes} < {previous}"
        );
        previous = bytes;
    }
}

/// 块格式：单调，且不受信任的 `numel` 到顶时饱和而不是回绕成小值。
fn assert_block_bytes<F: BlockFormat>() {
    for format in F::ALL {
        assert_monotonic(|numel| format.bytes_for(numel), format.name());
        // 饱和值不能小于半量级，更不能回绕成十几字节。
        let huge = format.bytes_for(u64::MAX);
        assert!(
            huge >= format.bytes_for(u64::MAX / 2),
            "{format:?} 溢出未饱和"
        );
        assert!(
            huge >= u64::from(format.block_bytes()),
            "{format:?} 溢出回绕"
        );
    }
}

#[test]
fn block_formats_count_whole_blocks() {
    assert_whole_blocks::<QType>();
    assert_whole_blocks::<MxType>();
}

#[test]
fn byte_counts_are_monotonic_and_saturate() {
    assert_block_bytes::<QType>();
    assert_block_bytes::<MxType>();
    // 标量轴同样单调、同样饱和：全部 24 种都跑一遍，而不只抽一个亚字节类型。
    for dtype in DType::ALL {
        assert_monotonic(|numel| dtype.bytes_for(numel), dtype.name());
        assert_eq!(
            dtype.bytes_for(u64::MAX),
            u64::MAX.div_ceil(8),
            "{dtype:?} 的溢出必须饱和到同一个上界"
        );
    }
}

#[test]
fn layout_carries_the_geometry_of_both_arms() {
    let scalar: Layout<QType> = Layout::Scalar(DType::F16);
    assert_eq!((scalar.elems(), scalar.span()), (1, 2));
    assert_eq!(scalar.bytes_for(4), 8);

    let block: Layout<QType> = Layout::Block(QType::Q4_K);
    assert_eq!((block.elems(), block.span()), (256, 144));
    assert_eq!(block.bytes_for(256), 144);
    assert_eq!(block.bytes_for(257), 288, "多一个元素也要多一整块");

    // 同一个泛型布局也能装 MX 格式：接 MXFP4/NVFP4 解码时不必另写一份 `elems`/`span`。
    let micro: Layout<MxType> = Layout::Block(MxType::NvFp4);
    assert_eq!((micro.elems(), micro.span()), (16, 9));
    assert_eq!(micro.bytes_for(64), 36);

    // 亚字节标量：`span()` 是「一个元素占几个字节」，与按位打包的 `bytes_for` 不是同一件事，
    // 所以标量手臂不实现 `BlockFormat`。这条断言把那个差别固定下来，避免有人按 `span × n` 算。
    let packed: Layout<QType> = Layout::Scalar(DType::U4);
    assert_eq!((packed.elems(), packed.span()), (1, 1));
    assert_eq!(packed.bytes_for(8), 4, "8 个 U4 只占 4 字节");
    assert_eq!(
        packed.bytes_for(u64::MAX),
        u64::MAX.div_ceil(8),
        "饱和后按位取整"
    );
}

// ---------------------------------------------------------------------------
// 可选互操作
// ---------------------------------------------------------------------------
//
// 可选互操作：`serde` 的线上名字与 `bytemuck` 的受检字节视图。
//
// 两者都只在对应 feature 打开时编译；默认依赖集仍是零第三方 crate。名字必须与 `Display`
// 逐字一致——配置、元数据与诊断里出现的是同一个词，不能有两套拼写。
//
// 断言写在泛型 helper 里而不是每个闭集抄一遍：`serde`/`bytemuck` 对四个闭集的要求完全一样，
// 新增一个闭集只是多一行调用。

#[cfg(feature = "serde")]
mod serde_wire_format {
    use llm_dtype::{DType, Element, Format, MxType, QType};
    use serde::Serialize;
    use serde::de::DeserializeOwned;

    /// 序列化成 JSON 后必须等于 `Display` 的规范名、能原样读回，表外名字必须失败。
    fn assert_json_names<F>(unknown: &str)
    where
        F: Format + Serialize + DeserializeOwned,
    {
        for format in F::ALL {
            let text = serde_json::to_string(format).expect("格式身份必须可序列化");
            assert_eq!(
                text,
                format!("\"{format}\""),
                "{format:?} 的线上名字与 `Display` 分家"
            );
            assert_eq!(
                serde_json::from_str::<F>(&text).ok(),
                Some(*format),
                "{format:?} 不能从 JSON 读回"
            );
        }
        assert!(
            serde_json::from_str::<F>(&format!("\"{unknown}\"")).is_err(),
            "`{unknown}` 不在闭集里，反序列化必须失败而不是落回某个默认值"
        );
    }

    #[test]
    fn names_round_trip_through_json() {
        assert_json_names::<DType>("i7");
        assert_json_names::<QType>("q4_9");
        assert_json_names::<MxType>("mxfp3");
        assert_json_names::<Element>("e1m1");
    }
}

#[cfg(feature = "bytemuck")]
mod byte_view {
    use llm_dtype::{DType, Element, Format, MxType, QType};

    /// 写视图是一字节判别值（即行表下标），受检读视图能把它原样读回；闭集之外的每个字节
    /// 都是非法位模式。
    fn assert_single_byte<F>()
    where
        F: Format + bytemuck::NoUninit + bytemuck::checked::CheckedBitPattern + Copy,
    {
        for format in F::ALL {
            let bytes = bytemuck::bytes_of(format);
            let code = u8::try_from(format.idx()).expect("闭集远小于 256 个变体");
            assert_eq!(bytes, &[code], "{format:?} 写出的不是一字节判别值");
            assert_eq!(
                bytemuck::checked::try_from_bytes::<F>(bytes).ok().copied(),
                Some(*format),
                "{format:?} 不能从自己的字节读回"
            );
        }

        // 逐字节穷举 256 个位模式：抽样只测 0xFF 时，「除了 0xFF 全放行」的实现照样能过。
        for code in u8::MIN..=u8::MAX {
            let parsed = bytemuck::checked::try_from_bytes::<F>(&[code])
                .ok()
                .copied();
            match parsed {
                Some(format) => {
                    assert_eq!(
                        Format::idx(format),
                        usize::from(code),
                        "{code} 读成了别的格式"
                    );
                    assert!(
                        usize::from(code) < F::ALL.len(),
                        "{code} 在闭集之外却被接受"
                    );
                }
                None => assert!(usize::from(code) >= F::ALL.len(), "{code} 在闭集之内却被拒"),
            }
        }
    }

    #[test]
    fn every_format_round_trips_through_its_single_byte() {
        assert_single_byte::<DType>();
        assert_single_byte::<QType>();
        assert_single_byte::<MxType>();
        assert_single_byte::<Element>();
    }
}

// ---------------------------------------------------------------------------
// 随机属性
// ---------------------------------------------------------------------------
//
// 属性骨架：随机字符串要么命中闭集并按规范名往返，要么被拒绝；解析路径不得 panic。
//
// 闭集内每个变体的下标/规范名/`Display`/`FromStr` 双向契约由上面的「闭集身份」段逐条锁定，
// 这里只覆盖不受信输入分布：随机拼出的名字即使命中，也必须能从规范名解析回同一个变体。

#[test]
fn random_names_either_round_trip_or_are_rejected() {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
    let mut rng = testkit::Rng::new(0xD1_7E);
    let charset_len = u32::try_from(CHARSET.len()).expect("字符集长度远小于 u32 上限");
    // 用例串行生成：下一个字符依赖上一次 PRNG 状态；不变量是命中即往返、未命中即 `None`。
    for _ in 0..128 {
        let len = rng.below(12) as usize;
        let text: String = (0..len)
            .map(|_| CHARSET[rng.below(charset_len) as usize] as char)
            .collect();
        if let Some(value) = DType::parse_name(&text) {
            assert_eq!(value.name(), text, "命中的名字必须按规范名往返");
            assert_eq!(DType::parse_name(value.name()), Some(value));
        }
    }
}
