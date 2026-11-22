//! 职责：外部权重格式的只读适配层——把别人的文件格式读成「名字 + 类型 + 形状 + 字节区间」，
//! 不参与自有容器、量化方案与证书判定。
//!
//! 文件名即格式名或唯一机制名，限定词下沉为同名目录（`gguf/quant`）：`json`、`zip`、
//! `pickle`、`torch`、`gguf`，共享机制 `cursor`。唯一的测试夹具是文件尾的 `torch_sample`
//! （只负责本机 torch `.bin` 样本）——它内联在此，`crate::torch_sample` 路径不变。

#![deny(unsafe_code)]

pub(crate) mod cursor;
pub mod gguf;
pub mod json;
pub(crate) mod pickle;
pub mod torch;
pub(crate) mod zip;
mod quant;

/// 测试夹具：本机 torch `.bin` 样本；其余夹具仍留在 `weights`。
#[cfg(test)]
mod torch_sample {
    /// 读取本机 torch `.bin` 样本；样本缺失返回 `None`，测试跳过而不是伪造字节。
    pub(crate) fn real_bin_bytes() -> Option<Vec<u8>> {
        std::fs::read("/tmp/pt/model.bin")
            .ok()
            .or_else(|| testkit::skip_none("torch 样本缺失"))
    }
}
