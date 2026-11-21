//! torch.save `.bin` 只读解析（零依赖、纯安全 Rust）：ZIP_STORED 归档内 `data.pkl` 是 pickle 流，
//! `data/<key>` 是各 storage 原始字节。先解归档，再用栈式 pickle 虚拟机解释流，拦截
//! `_rebuild_tensor_v2` 重建与 `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 + 字节区间」；原始字节按需借用，未知 opcode / 不可识别可调用对象显式报错，不执行任意代码、不猜结果。
#![allow(missing_docs)]

use wire::bad;
use diagnostic::Result;
use dtype::DType;

use crate::zip::Zip;

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
        let tensors = crate::pickle::Vm::run(pkl)?;
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

#[cfg(test)]
#[path = "../tests/bin.rs"]
mod tests;
