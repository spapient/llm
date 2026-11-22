//! `snapshot` 测试共用的夹具：恢复事实、条目表、往返对拍与手工拼字节的助手。

use super::*;
use std::path::Path;
use tensor::Shape;
use weights::Entry;
pub(crate) fn facts() -> Facts {
    Facts::new(7, 2, 2, 3, 3).expect("恢复事实")
}

/// 空权重头部（快照侧对拍常用）。
pub(crate) fn empty_header() -> Header {
    Header::new(Vec::new(), 0).expect("空头部")
}

/// 累积梯度逐槽位级对拍：f32 经小端往返不能只差一位。
pub(crate) fn assert_accum_bits(snapshot: &Snapshot, restored: &Snapshot) {
    let before = snapshot.accum().slots();
    let after = restored.accum().slots();
    assert_eq!(before.len(), after.len(), "槽位数必须一致");
    for (idx, (want, got)) in before.iter().zip(after).enumerate() {
        let want_bits: Vec<u32> = want.iter().map(|value| value.to_bits()).collect();
        let got_bits: Vec<u32> = got.iter().map(|value| value.to_bits()).collect();
        assert_eq!(got_bits, want_bits, "第 {idx} 槽累积梯度逐位一致");
    }
}

/// 目录下的文件名清单（写盘后"不留残骸"对账用）。
pub(crate) fn dir_entry_names(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .expect("目录")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect()
}

/// 写盘后对账：文件长度 = 承诺长度、读回逐对象相等，返回目录文件名清单供残骸断言。
pub(crate) fn assert_written_snapshot(dir: &Path, path: &Path, snapshot: &Snapshot) -> Vec<String> {
    let meta = std::fs::metadata(path).expect("文件存在");
    assert_eq!(meta.len() as usize, snapshot.encoded_len().expect("长度"));
    assert_eq!(&Snapshot::read(path).expect("读"), snapshot);
    dir_entry_names(dir)
}

/// 手改样本里的重名条目必须被条目表校验拦下。
pub(crate) fn reject_duplicate_name(bytes: Vec<u8>) {
    let error = Snapshot::decode(&bytes).expect_err("重名");
    assert!(error.to_string().contains("重复"), "{error}");
}

pub(crate) fn snapshot() -> Snapshot {
    let weights = Header::new(
        vec![
            Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([2, 3])).at(64),
            Entry::new("blk.0.scales", DType::U4, Shape::new([4])).at(88),
        ],
        64,
    )
    .expect("头部");
    Snapshot::new(facts(), weights)
}

/// 用任意正文重新封上校验和：只有测试会造"CRC 正确但正文有问题"的文件。
pub(crate) fn seal(payload: &[u8]) -> Vec<u8> {
    let mut bytes = payload.to_vec();
    bytes.extend_from_slice(&weights::crc::crc32(payload).to_le_bytes());
    bytes
}

pub(crate) fn push_varint(bytes: &mut Vec<u8>, value: u64) {
    varint(bytes, value);
}

pub(crate) fn payload_of(snapshot: &Snapshot) -> Vec<u8> {
    let bytes = snapshot.encode().expect("编码");
    bytes[..bytes.len() - Format::FOOTER_BYTES].to_vec()
}

pub(crate) fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// 一整套承诺：长度是算准的、写进缓冲区的字节与 `encode` 一致、缓冲区多出来的部分不被碰、
/// 解码回来逐项相同、再编码逐字节相同。
pub(crate) fn assert_round_trip(snapshot: &Snapshot) {
    let bytes = snapshot.encode().expect("编码");
    assert_eq!(
        bytes.len(),
        snapshot.encoded_len().expect("长度"),
        "承诺的长度就是写出的长度"
    );

    let mut buf = vec![0xAAu8; snapshot.encoded_len().expect("长度") + 16];
    let written = snapshot.encode_into(&mut buf).expect("缓冲够大");
    assert_eq!(written, snapshot.encoded_len().expect("长度"));
    assert_eq!(&buf[..written], bytes.as_slice());
    assert!(
        buf[written..].iter().all(|byte| *byte == 0xAA),
        "缓冲区多出来的部分不能被碰"
    );

    let restored = Snapshot::decode(&bytes).expect("解码");
    assert_eq!(&restored, snapshot);
    assert_eq!(
        restored.encode().expect("编码"),
        bytes,
        "同一份快照只有一种写法"
    );
}

/// 手工拼一份正文：用来造编码器**永远不会**产生的字节（两条目同名、共享长度写得比实际
/// 少、布局明明相同却不写重复）。解码器的判据必须比编码器更强，所以测试要能绕过编码器
/// 直接说话。
///
/// 条目是 `(共享前缀, 共享后缀, 中段, 偏移, 布局标签)`；标签为 `k > 0` 时后面跟类型 `F32`
/// 与 `k - 1` 个值为 1 的维度。
pub(crate) fn handcrafted(entries: &[(u64, u64, &str, usize, u64)]) -> Vec<u8> {
    let mut payload = Format::MAGIC.to_vec();
    payload.extend_from_slice(&Format::VERSION.to_le_bytes());
    for value in [7, 1, 1, 1, 1, 64] {
        push_varint(&mut payload, value);
    }
    push_varint(&mut payload, entries.len() as u64);
    let mut previous = 0usize;
    for (front, back, middle, offset, layout) in entries {
        push_varint(&mut payload, *front);
        push_varint(&mut payload, *back);
        push_varint(&mut payload, middle.len() as u64);
        payload.extend_from_slice(middle.as_bytes());
        push_varint(&mut payload, *layout);
        if *layout > 0 {
            payload.push(tag(DType::F32).expect("f32 在快照标签闭集内"));
            for _ in 0..*layout - 1 {
                push_varint(&mut payload, 1);
            }
        }
        push_varint(&mut payload, (*offset - previous) as u64);
        previous = *offset;
    }
    // 矩状态段：0 个槽 + 步数 0（旧手改样本没有矩状态）。
    push_varint(&mut payload, 0);
    push_varint(&mut payload, 0);
    // 累积梯度段：窗口批数 0、0 个槽。
    push_varint(&mut payload, 0);
    push_varint(&mut payload, 0);
    // 权重绑定段：0 对绑定。
    push_varint(&mut payload, 0);
    seal(&payload)
}
