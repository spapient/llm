//! 职责：torch.save 检查点（`.bin`）的只读解析——`data.pkl` 是 pickle 流，`data/<key>` 是各
//! storage 的原始字节。
//!
//! 先解归档（`zip::Zip`），再用栈式 pickle 虚拟机解释流，拦截 `_rebuild_tensor_v2` 重建与
//! `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 + 字节区间」；原始字节按需借用，
//! 未知 opcode 与不可识别的可调用对象一律显式报错，不执行任意代码、不猜结果。

use diagnostic::Result;
use dtype::DType;

use crate::zip::Zip;

/// 张量目录条目：名字、元素类型、形状，加存储区在归档内的字节区间。
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
    /// 张量名，取自 `_rebuild_tensor_v2` 所在字典的键。
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素类型，由 storage 类名决定。
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状，按书写轴序。
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// 一个已解析的检查点：归档句柄 + 张量目录。
pub struct Checkpoint<'a> {
    zip: Zip<'a>,
    tensors: Vec<Tensor>,
}

impl<'a> Checkpoint<'a> {
    /// 解析归档并跑完 pickle 流；`data.pkl` 的两种条目名都试。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let zip = Zip::parse(bytes)?;
        let pkl = zip.data_suffix("data.pkl");
        let pkl = pkl.or_else(|_| zip.data_suffix("/data.pkl"))?;
        let tensors = crate::pickle::Vm::run(pkl)?;
        Ok(Self { zip, tensors })
    }

    /// 张量目录，按 pickle 流里出现的先后顺序。
    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// 按名字查张量；没有该名字返回 `None`。
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 张量字节：`/data/<key>` 与 `data/<key>` 两种条目名都试；区间越界由共享的
    /// `cursor::slice` 判据统一拒——偏移相加必须先查溢出，否则绕回小值
    /// 会让越界区间变成「空的成功切片」。
    pub fn data(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let key = &tensor.storage_key;
        let storage = self
            .zip
            .data_suffix(&format!("/data/{key}"))
            .or_else(|_| self.zip.data_suffix(&format!("data/{key}")))?;
        crate::cursor::slice(storage, tensor.offset, tensor.len)
    }
}

#[cfg(test)]
#[path = "../tests/torch.rs"]
mod tests;
