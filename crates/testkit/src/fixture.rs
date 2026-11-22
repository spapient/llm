//! 格式夹具：safetensors 文件字节与量化码打包，供各成员拼测试输入。

/// 按 safetensors 布局拼一份完整文件：8 字节小端头长 + 头部 JSON + 依序拼接的数据段。
#[must_use]
pub fn safetensors_bytes(json: &str, data: &[(&str, Vec<u8>)]) -> Vec<u8> {
    for (name, _) in data {
        debug_assert!(json.contains(name), "头部 JSON 必须声明数据段 `{name}`");
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    for (_, chunk) in data {
        bytes.extend_from_slice(chunk);
    }
    bytes
}

/// 把 `bits` 位有符号码按低 `bits` 位打包：`zero = 2^(bits-1)` 平移，低 `bits` 位落在字节内。
#[must_use]
pub fn pack_codes(bits: usize, values: &[i8]) -> Vec<u8> {
    let per_byte = 8 / bits;
    let zero = 1i16 << (bits - 1);
    let mask = u8::MAX >> (8 - bits);
    values
        .chunks(per_byte)
        .map(|group| {
            (0..per_byte).fold(0u8, |byte, slot| {
                let code = (i16::from(group.get(slot).copied().unwrap_or(0)) + zero) as u8 & mask;
                byte | (code << (slot * bits))
            })
        })
        .collect()
}
