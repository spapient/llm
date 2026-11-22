//! [`Buf`]：一段拥有的扁平字节缓冲。

/// 一段拥有的扁平字节缓冲：CPU 后端的不透明句柄。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buf {
    bytes: Vec<u8>,
}

impl Buf {
    /// 由拥有的字节向量构造；长度即缓冲容量。
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// 只读字节。
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// 可写字节。
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// 字节长度。
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// 拆回拥有的字节向量（归还分配器前取出数据用）。
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
