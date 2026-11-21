//! 职责：GGUF 字节夹具——**独立写字节，不引用被测常量**，所以它同时是格式的对照口径。
//!
//! `weights` 的导入桥接用例（`tests/import_bridge.rs`）以跨 crate `#[path]` 复用本文件，
//! 于是"标量导入 / 块量化显式失败"两条判据只有一份夹具。
const MAGIC: [u8; 4] = *b"GGUF";

const VERSION: u32 = 3;

const ALIGN: u64 = 32;

const T_F32: u32 = 0;

const T_F16: u32 = 1;

const T_Q8_0: u32 = 8;

const T_Q4_K: u32 = 12;

const T_Q6_K: u32 = 14;

fn text(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());

    out.extend_from_slice(value.as_bytes());
}

fn word(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn wide(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn key(out: &mut Vec<u8>, name: &str, kind: u32) {
    text(out, name);

    word(out, kind);
}

/// 拼一个 GGUF：四条元数据 + 若干张量，正文按 `ALIGN` 逐个对齐。
pub(crate) fn assemble(pieces: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
    let mut head = Vec::new();

    head.extend_from_slice(&MAGIC);

    word(&mut head, VERSION);

    wide(&mut head, pieces.len() as u64);

    wide(&mut head, 4);

    key(&mut head, "general.name", 8);

    text(&mut head, "fixture");

    key(&mut head, "general.architecture", 8);

    text(&mut head, "llama");

    key(&mut head, "general.alignment", 4);

    word(&mut head, ALIGN as u32);

    key(&mut head, "llama.block_count", 4);

    word(&mut head, 1);

    let mut offset = 0u64;

    let mut infos = Vec::new();

    let mut body = Vec::new();

    for (name, shape, kind, bytes) in pieces {
        offset = offset.div_ceil(ALIGN) * ALIGN;

        text(&mut infos, name);

        word(&mut infos, shape.len() as u32);

        for dim in shape {
            wide(&mut infos, *dim);
        }

        word(&mut infos, *kind);

        wide(&mut infos, offset);

        body.extend(std::iter::repeat_n(0, offset as usize - body.len()));

        body.extend_from_slice(bytes);

        offset += bytes.len() as u64;
    }

    head.extend(infos);

    let fill = (ALIGN - head.len() as u64 % ALIGN) % ALIGN;

    head.extend(std::iter::repeat_n(0, fill as usize));

    head.extend(body);

    head
}

/// 全类型夹具：F32/F16 标量与 Q8_0/Q4_K/Q6_K 块量化各一。
pub(crate) fn standard() -> Vec<u8> {
    let f32s: Vec<u8> = [1.0f32, -0.5, 3.25, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();

    let halves: Vec<u8> = [0x3800u16, 0xC000, 0, 0x3E00]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();

    let mut q8 = 0x3C00u16.to_le_bytes().to_vec();

    q8.extend((0..32).map(|i| (i as i8) - 16).map(|q| q as u8));

    let mut q4 = 0x3C00u16.to_le_bytes().to_vec();

    q4.extend(0u16.to_le_bytes());

    q4.extend([1u8; 12]);

    q4.extend([0x21u8; 128]);

    let mut q6 = vec![0x21u8; 128];

    q6.extend([0u8; 64]);

    q6.extend([1u8; 16]);

    q6.extend(0x3C00u16.to_le_bytes());

    assemble(&[
        ("f32.weight", vec![2, 2], T_F32, f32s),
        ("half.weight", vec![4], T_F16, halves),
        ("q8.weight", vec![32], T_Q8_0, q8),
        ("q4.weight", vec![256], T_Q4_K, q4),
        ("q6.weight", vec![256], T_Q6_K, q6),
    ])
}

/// 只有标量 F32/F16 的最小夹具：供导入通路的成功路径对拍。
pub(crate) fn plain() -> Vec<u8> {
    let f32s: Vec<u8> = [1.0f32, -0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();

    let halves: Vec<u8> = [0x3800u16, 0xC000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();

    assemble(&[
        ("f32.weight", vec![2], T_F32, f32s),
        ("half.weight", vec![2], T_F16, halves),
    ])
}
