//! MX 布局的规范对照：OCP MX v1.1 与 NVFP4 厂商布各一个手写块。
//!
//! [`MxType`] 还没有解码器消费者，块几何过去只是「照规范推 + 自测」。这里把规范表本身
//! 搬成金标：块字节与期望数值都逐字写死，不由本文件的辅助函数「先编码再解码」推出来——
//! 编解码同源的话，打包顺序写反也照样自圆其说。块长、每块字节、元素位宽、半字节顺序或
//! E8M0/E4M3 的偏置只要漂移，长度或数值就会先对不上。
//!
//! 仍未覆盖：真实模型的 GGUF 权重字节流。那份对照要等 `weights` 接上 MXFP4 反量化，
//! 所以本文件是**规范对照**，不是解码实现的验收。

use llm_dtype::{Element, MxType};

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
fn an_ocp_mxfp4_block_matches_the_specification_table() {
    let format = MxType::MxFp4E2M1;
    assert_eq!(format.element(), Element::E2M1);
    assert_eq!(format.block_len(), 32, "OCP MX 固定块 32 元素");
    assert_eq!(format.block_bytes(), 17, "16 字节负载 + 1 字节 E8M0 scale");

    // 字面块与字面期望值：长度、块字节、半字节顺序、数值表四项里任一项漂移都在这里对不上。
    assert_eq!(
        u64::try_from(MXFP4_BLOCK.len()).expect("块长远小于 u64 上限"),
        format.bytes_for(32),
        "OCP 块必须正好是 `bytes_for(32)` 字节"
    );
    assert_eq!(e8m0_scale(MXFP4_BLOCK[0]), 1.0, "0x7F 是 E8M0 的 2^0");
    assert_eq!(
        decode_payload(e8m0_scale(MXFP4_BLOCK[0]), &MXFP4_BLOCK[1..]).as_slice(),
        &MXFP4_VALUES[..]
    );
}

#[test]
fn an_nvfp4_micro_block_matches_the_vendor_layout() {
    let format = MxType::NvFp4;
    assert_eq!(format.element(), Element::E2M1);
    assert_eq!(format.block_len(), 16, "NVFP4 的原生微块是 16 元素");
    assert_eq!(
        format.block_bytes(),
        9,
        "8 字节负载 + 1 字节 E4M3 微块 scale"
    );

    // E4M3 的 `1.0` 是 `0x38`（指数 7 减偏置 7、尾数 0）；每张量的 FP32 二级 scale 在块外。
    assert_eq!(e4m3_scale(NVFP4_BLOCK[0]), 1.0);
    assert_eq!(
        u64::try_from(NVFP4_BLOCK.len()).expect("块长远小于 u64 上限"),
        format.bytes_for(16)
    );
    assert_eq!(
        decode_payload(e4m3_scale(NVFP4_BLOCK[0]), &NVFP4_BLOCK[1..]).as_slice(),
        &NVFP4_VALUES[..]
    );
}
