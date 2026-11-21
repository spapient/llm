//! 端到端用例的共享夹具：派生二进制的入口、合成 safetensors 的写法。
//!
//! 两个端到端目标（[`crate::` 无法跨目标共享）都 `#[path = "support.rs"] mod support;`，
//! 因此夹具只有这一份：同一个"好权重"不会在两个文件里各长一遍。

use assert_cmd::Command;
use std::path::PathBuf;

pub fn llm() -> Command {
    Command::cargo_bin("llm").expect("二进制在测试构建里必须存在")
}

/// 写一份合成 safetensors：两个秩 2 条目 + 一条秩 1 条目（故意留一条未覆盖）。
///
/// 数据段全零：示例只验证"链路取到了字节"，数值含义由库层用例负责。
pub fn write_weights(tag: &str, header: &str, data_bytes: usize) -> PathBuf {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + data_bytes, 0);
    let path =
        std::env::temp_dir().join(format!("llm-cli-{}-{tag}.safetensors", std::process::id()));
    std::fs::write(&path, &bytes).expect("临时权重可写");
    path
}

pub fn good_weights(tag: &str) -> (PathBuf, String) {
    let header = r#"{"blk.0.attn_q.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
                    "blk.0.norm.weight": {"dtype": "F32", "shape": [4], "data_offsets": [64, 80]}}"#;
    let path = write_weights(tag, header, 80);
    (path, header.to_string())
}
