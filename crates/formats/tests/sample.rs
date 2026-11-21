//! 测试共享夹具：本层只需本机 torch `.bin` 样本，其余夹具仍留在 `weights`。
/// 读取本机 torch `.bin` 样本；样本缺失返回 `None`，测试跳过而不是伪造字节。
pub(crate) fn real_bin_bytes() -> Option<Vec<u8>> {
    std::fs::read("/tmp/pt/model.bin")
        .ok()
        .or_else(|| testkit::skip_none("torch 样本缺失"))
}
