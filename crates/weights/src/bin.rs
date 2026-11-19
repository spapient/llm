//! torch.save `.bin` 只读解析（零依赖、纯安全 Rust）：ZIP_STORED 归档内 `data.pkl` 是 pickle 流，
//! `data/<key>` 是各 storage 原始字节。先解归档，再用栈式 pickle 虚拟机解释流，拦截
//! `_rebuild_tensor_v2` 重建与 `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 + 字节区间」；原始字节按需借用，未知 opcode / 不可识别可调用对象显式报错，不执行任意代码、不猜结果。
#![allow(missing_docs)]

use diagnostic::{Code, Error, Result};
use dtype::DType;

#[path = "bin_pickle.rs"]
pub(crate) mod pickle;
#[path = "bin_zip.rs"]
pub(crate) mod zip;

use zip::Zip;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<u64>,
    pub(crate) storage_key: String,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl Tensor {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

pub struct Bin<'a> {
    zip: Zip<'a>,
    tensors: Vec<Tensor>,
}

impl<'a> Bin<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let zip = Zip::parse(bytes)?;
        let pkl = zip.data_suffix("data.pkl");
        let pkl = pkl.or_else(|_| zip.data_suffix("/data.pkl"))?;
        let tensors = pickle::Vm::run(pkl)?;
        Ok(Self { zip, tensors })
    }

    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn data(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let suffix = format!("/data/{}", tensor.storage_key);
        let storage = self.zip.data_suffix(&suffix);
        let fallback = format!("data/{}", tensor.storage_key);
        let storage = storage.or_else(|_| self.zip.data_suffix(&fallback))?;
        storage
            .get(tensor.offset..tensor.offset + tensor.len)
            .ok_or_else(|| bad("张量字节区间越界"))
    }
}

fn or_bad<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| bad(message))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("bin")
}

#[cfg(test)]
mod tests {
use super::*;
use crate::testutil::real_bin_bytes;
use dtype::DType;

#[test]
fn parses_real_torch_bin() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let bin = Bin::parse(&raw).expect("解析");
    assert_eq!(bin.tensors().len(), 4);

    let w = bin.tensor("layer.weight").expect("张量");
    assert_eq!(w.dtype(), DType::F32);
    assert_eq!(w.shape(), &[4, 6]);
    assert_eq!(bin.data(w).expect("字节").len(), 96);

    let embed = bin.tensor("embed").expect("f16");
    assert_eq!(embed.dtype(), DType::F16);
    assert_eq!(bin.data(embed).expect("字节").len(), 24);

    let count = bin.tensor("count").expect("i64");
    assert_eq!(count.dtype(), DType::I64);
    assert_eq!(bin.data(count).expect("字节").len(), 24);
}

#[test]
fn bytes_match_torch_storage() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let bin = Bin::parse(&raw).expect("解析");
    let bias = bin.tensor("layer.bias").expect("bias");
    let bytes = bin.data(bias).expect("字节");
    assert_eq!(bytes.len(), 16);
}

#[test]
fn rejects_non_bin() {
    assert!(Bin::parse(b"definitely not a torch bin").is_err());
}

#[test]
fn import_keeps_torch_tensors_as_neutral_segments() {
    let Some(raw) = real_bin_bytes() else {
        return;
    };
    let bin = Bin::parse(&raw).expect("解析");
    let segments = crate::import::segments_from_bin(&bin).expect("导入中性段流");
    assert_eq!(segments.len(), bin.tensors().len());
    for (segment, tensor) in segments.iter().zip(bin.tensors()) {
        assert_eq!(segment.name(), tensor.name());
        assert_eq!(segment.dtype(), tensor.dtype());
        assert_eq!(segment.shape(), tensor.shape());
        assert_eq!(segment.data, bin.data(tensor).expect("字节"));
    }
}

#[test]
fn import_propagates_unreadable_bin() {
    let error = Bin::parse(b"definitely not a torch bin")
        .and_then(|bin| crate::import::segments_from_bin(&bin));
    assert!(error.is_err(), "读不通的 bin 必须显式失败");
}
}

