//! 驻留权重表：装载期一次性解码 F32 条目，生成循环内不再读文件。
//!
//! 形状与条目名描述归 [`crate::descriptors`](crate::descriptors)；这里只持有
//! `条目名 → f32 整块`，不解释编号含义。

use std::collections::BTreeMap;

use diagnostic::{Code, Error, Result};
use nn::{AttentionBlock, Contract, Stack, WEIGHT_SUFFIX};
use weights::{Entry as WeightEntry, Header, Loader, Wire, decode_f32};

use crate::kernels::SITE;

/// 驻留权重表：条目名 → f32 整块。装载期一次性解码进来，生成循环内不再读文件。
#[derive(Debug, Clone, Default)]
pub struct Params {
    entries: BTreeMap<String, Vec<f32>>,
}

impl Params {
    /// 空参数表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一条权重整块；同名条目覆盖旧值（装配期唯一性由 [`Stack`] 层名检查守住）。
    pub fn insert(&mut self, entry: impl Into<String>, weights: Vec<f32>) -> &mut Self {
        self.entries.insert(entry.into(), weights);
        self
    }

    /// 按条目名取权重整块。
    pub fn get(&self, entry: &str) -> Option<&[f32]> {
        self.entries.get(entry).map(Vec::as_slice)
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否一条权重都没有。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按层栈与权重头部，经读取后端把 F32 条目解码进驻留表；非 F32 条目不填块，
    /// 留给装配期按格式拒绝。
    pub fn from_stack(stack: &Stack, header: &Header, loader: &impl Loader) -> Result<Self> {
        let mut params = Self::new();
        for layer in stack.layers() {
            Self::insert_layer(&mut params, layer.as_ref(), header, loader)?;
        }
        Ok(params)
    }

    /// 装载单层：注意力块走 q/k/v/o 四条子条目，其余层走单条目。
    fn insert_layer(
        params: &mut Self,
        layer: &dyn Contract,
        header: &Header,
        loader: &impl Loader,
    ) -> Result<()> {
        if let Some(block) = layer.attention_block() {
            return Self::load_block(block, header, loader, params);
        }
        let entry = Self::entry_for(layer, header)?;
        let Some(weight) = Self::load_f32(entry, loader)? else {
            return Ok(());
        };
        params.insert(entry.name().to_string(), weight);
        Ok(())
    }

    /// 解码注意力块的 q/k/v/o 四个投影条目；非 F32 条目跳过，装配期按格式拒绝。
    fn load_block(
        block: &AttentionBlock,
        header: &Header,
        loader: &impl Loader,
        params: &mut Self,
    ) -> Result<()> {
        for level in ["q", "k", "v", "o"] {
            let name = format!("{}.{level}{WEIGHT_SUFFIX}", block.name());
            let Some(entry) = header.get(&name) else {
                return Err(
                    Error::new(Code::Missing, format!("权重头部缺条目 `{name}`")).with(SITE),
                );
            };
            let Some(weight) = Self::load_f32(entry, loader)? else {
                continue;
            };
            params.insert(name, weight);
        }
        Ok(())
    }

    /// 取层名对应的权重条目；头部缺条目报 [`Code::Missing`]。
    fn entry_for<'a>(layer: &dyn Contract, header: &'a Header) -> Result<&'a WeightEntry> {
        let name = format!("{}{WEIGHT_SUFFIX}", layer.name());
        header
            .get(&name)
            .ok_or_else(|| Error::new(Code::Missing, format!("权重头部缺条目 `{name}`")).with(SITE))
    }

    /// 解码一条 F32 条目；其余编码返回 `None`，不在装载期解包。
    fn load_f32(entry: &WeightEntry, loader: &impl Loader) -> Result<Option<Vec<f32>>> {
        if entry.dtype() != Wire::F32 {
            return Ok(None);
        }
        let bytes = loader.load(entry)?;
        decode_f32(&bytes).map(Some)
    }
}
