//! 驻留权重表：装载期一次性解码 F32 条目，生成循环内不再读文件。
//!
//! 形状与条目名描述归 [`crate::descriptors`](crate::descriptors)；这里只持有
//! `条目名 → f32 整块`，不解释编号含义。

use std::collections::BTreeMap;

use diagnostic::{Code, Error, Result};
use nn::{AttentionBlock, Contract, Stack, WEIGHT_SUFFIX};
use weights::{Entry as WeightEntry, Header, Loader, Reader, Wire, decode_f32};

use crate::kernels::SITE;

/// 条目装载后端的最小契约：把一条 F32 条目解包进驻留 `f32` 缓冲；非 F32 返回 `None`，
/// 留给装配期按格式拒绝。
///
/// 两种物理后端经薄适配实现本契约——[`Loader`] 拷字节、[`Reader`] 借字节，族别分流与
/// 注意力块四路条目只写一份，不在后端之间复制装配规则。借用适配直接把映射/归档切片喂给
/// 解包，不额外拷贝一次字节缓冲。
trait WeightSource {
    /// 解码条目；非 F32 返回 `Ok(None)`。
    fn f32(&self, entry: &WeightEntry) -> Result<Option<Vec<f32>>>;
}

/// 拷贝后端适配：seek/read 把条目字节读进调用方缓冲。
struct LoaderSource<'a>(&'a dyn Loader);

impl WeightSource for LoaderSource<'_> {
    fn f32(&self, entry: &WeightEntry) -> Result<Option<Vec<f32>>> {
        decode_entry(entry, &self.0.load(entry)?)
    }
}

/// 借用后端适配：条目字节从归档/映射区借出，解包时不经过 `Vec<u8>` 中转。
struct ReaderSource<'a>(&'a dyn Reader);

impl WeightSource for ReaderSource<'_> {
    fn f32(&self, entry: &WeightEntry) -> Result<Option<Vec<f32>>> {
        decode_entry(entry, self.0.read(entry)?)
    }
}

/// 共用的类型判据与解包：非 F32 跳过，F32 整段解包成驻留缓冲。
fn decode_entry(entry: &WeightEntry, bytes: &[u8]) -> Result<Option<Vec<f32>>> {
    if entry.dtype() != Wire::F32 {
        return Ok(None);
    }
    decode_f32(bytes).map(Some)
}

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

    /// 按层栈与权重头部，经拷贝后端（`File`）把 F32 条目解码进驻留表；非 F32 条目不填块，
    /// 留给装配期按格式拒绝。
    pub fn from_stack(stack: &Stack, header: &Header, loader: &impl Loader) -> Result<Self> {
        Self::assemble(stack, header, &LoaderSource(loader))
    }

    /// 按层栈与权重头部，经借用后端（内存归档、只读映射）把 F32 条目解码进驻留表；
    /// 条目字节不经过一次拷贝，解包后的驻留缓冲仍由本表拥有。
    pub fn from_reader_stack(
        stack: &Stack,
        header: &Header,
        reader: &impl Reader,
    ) -> Result<Self> {
        Self::assemble(stack, header, &ReaderSource(reader))
    }

    /// 装配主循环：族别分流与条目解析对两种后端一致，差别只在字节来源由 [`WeightSource`] 适配。
    fn assemble<S: WeightSource>(stack: &Stack, header: &Header, source: &S) -> Result<Self> {
        let mut params = Self::new();
        for layer in stack.layers() {
            Self::insert_layer(&mut params, layer.as_ref(), header, source)?;
        }
        Ok(params)
    }

    /// 装载单层：注意力块走 q/k/v/o 四条子条目，其余层走单条目。
    fn insert_layer<S: WeightSource>(
        params: &mut Self,
        layer: &dyn Contract,
        header: &Header,
        source: &S,
    ) -> Result<()> {
        if let Some(block) = layer.attention_block() {
            return Self::load_block(block, header, source, params);
        }
        let entry = Self::entry_for(layer, header)?;
        let Some(weight) = source.f32(entry)? else {
            return Ok(());
        };
        params.insert(entry.name().to_string(), weight);
        Ok(())
    }

    /// 解码注意力块的 q/k/v/o 四个投影条目；非 F32 条目跳过，装配期按格式拒绝。
    fn load_block<S: WeightSource>(
        block: &AttentionBlock,
        header: &Header,
        source: &S,
        params: &mut Self,
    ) -> Result<()> {
        for level in ["q", "k", "v", "o"] {
            let name = format!("{}.{level}{WEIGHT_SUFFIX}", block.name());
            let Some(entry) = header.get(&name) else {
                return Err(
                    Error::new(Code::Missing, format!("权重头部缺条目 `{name}`")).with(SITE),
                );
            };
            let Some(weight) = source.f32(entry)? else {
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
}
