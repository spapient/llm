//! 驻留权重表：装载期一次性解码条目，生成循环内不再读文件。
//!
//! 形状与条目名描述归 [`crate::descriptors`](crate::descriptors)；这里只持有
//! `条目名 → 权重块`。权重块有两种：稠密 `f32`（[`Block::Dense`]）与量化打包字节
//! （[`Block::Packed`]）；量化标度仍是 f32，作为独立条目按标度条目名登记，
//! [`Params::scale_entry`] 记录权重条目到标度条目的对应，装配侧据此把两条绑到同一节点。

use std::collections::BTreeMap;

use diagnostic::{Code, Error, Result};
use nn::{AttentionBlock, Contract, Stack, WEIGHT_SUFFIX};
use weights::{Entry as WeightEntry, Header, Loader, Quantized, Reader, Wire, decode_f32};

use crate::kernels::SITE;

/// 驻留权重块：稠密 f32，或量化打包字节（i4 每字节两码、i8 每字节一码）。
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// 稠密 f32 整块。
    Dense(Vec<f32>),
    /// 量化打包字节整块。
    Packed(Vec<u8>),
}

impl Block {
    /// 块内元素/字节数：稠密是 f32 元素数，打包是字节数。
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Packed(bytes) => bytes.len(),
        }
    }

    /// 稠密 f32 视图；打包返回 `None`。
    pub fn dense(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(values) => Some(values),
            Self::Packed(_) => None,
        }
    }

    /// 打包字节视图；稠密返回 `None`。
    pub fn packed(&self) -> Option<&[u8]> {
        match self {
            Self::Packed(bytes) => Some(bytes),
            Self::Dense(_) => None,
        }
    }
}

/// 条目装载后端的最小契约：把一条 F32 条目解包进驻留 `f32` 缓冲（非 F32 返回 `None`），
/// 或取一条条目的原始字节（量化打包权重的存储形态就是字节，不做数值解码）。
///
/// 两种物理后端经薄适配实现本契约——[`Loader`] 拷字节、[`Reader`] 借字节，族别分流与
/// 注意力块四路条目只写一份，不在后端之间复制装配规则。
trait WeightSource {
    /// 解码条目；非 F32 返回 `Ok(None)`。
    fn f32(&self, entry: &WeightEntry) -> Result<Option<Vec<f32>>>;
    /// 条目原始字节。
    fn bytes(&self, entry: &WeightEntry) -> Result<Vec<u8>>;
}

/// 拷贝后端适配：seek/read 把条目字节读进调用方缓冲。
struct LoaderSource<'a>(&'a dyn Loader);

impl WeightSource for LoaderSource<'_> {
    fn f32(&self, entry: &WeightEntry) -> Result<Option<Vec<f32>>> {
        decode_entry(entry, &self.0.load(entry)?)
    }

    fn bytes(&self, entry: &WeightEntry) -> Result<Vec<u8>> {
        self.0.load(entry)
    }
}

/// 借用后端适配：条目字节从归档/映射区借出，解包时不经过 `Vec<u8>` 中转。
struct ReaderSource<'a>(&'a dyn Reader);

impl WeightSource for ReaderSource<'_> {
    fn f32(&self, entry: &WeightEntry) -> Result<Option<Vec<f32>>> {
        decode_entry(entry, self.0.read(entry)?)
    }

    fn bytes(&self, entry: &WeightEntry) -> Result<Vec<u8>> {
        Ok(self.0.read(entry)?.to_vec())
    }
}

/// 共用的类型判据与解包：非 F32 跳过，F32 整段解包成驻留缓冲。
fn decode_entry(entry: &WeightEntry, bytes: &[u8]) -> Result<Option<Vec<f32>>> {
    if entry.dtype() != Wire::F32 {
        return Ok(None);
    }
    decode_f32(bytes).map(Some)
}

/// 驻留权重表：条目名 → 权重块，另记量化权重条目到其标度条目的对应。
#[derive(Debug, Clone, Default)]
pub struct Params {
    entries: BTreeMap<String, Block>,
    /// 量化权重条目名 → 标度条目名。
    scales: BTreeMap<String, String>,
}

impl Params {
    /// 空参数表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一条稠密权重整块；同名条目覆盖旧值（装配期唯一性由 [`Stack`] 层名检查守住）。
    pub fn insert(&mut self, entry: impl Into<String>, weights: Vec<f32>) -> &mut Self {
        self.entries.insert(entry.into(), Block::Dense(weights));
        self
    }

    /// 登记一条量化打包字节整块；同名条目覆盖旧值。
    pub fn insert_packed(&mut self, entry: impl Into<String>, packed: Vec<u8>) -> &mut Self {
        self.entries.insert(entry.into(), Block::Packed(packed));
        self
    }

    /// 登记一条量化权重整块：打包字节 + 标度各占一条目，并记录两者对应。
    pub fn insert_quantized(
        &mut self,
        entry: impl Into<String>,
        packed: Vec<u8>,
        scale_entry: impl Into<String>,
        scales: Vec<f32>,
    ) -> &mut Self {
        let entry = entry.into();
        let scale_entry = scale_entry.into();
        self.insert_packed(entry.clone(), packed);
        self.insert(scale_entry.clone(), scales);
        self.scales.insert(entry, scale_entry);
        self
    }

    /// 按条目名取权重块。
    pub fn get(&self, entry: &str) -> Option<&Block> {
        self.entries.get(entry)
    }

    /// 按条目名取稠密 f32 整块；打包条目返回 `None`。
    pub fn dense(&self, entry: &str) -> Option<&[f32]> {
        self.get(entry).and_then(Block::dense)
    }

    /// 按量化权重条目名取它的标度条目名；非量化返回 `None`。
    pub fn scale_entry(&self, entry: &str) -> Option<&str> {
        self.scales.get(entry).map(String::as_str)
    }

    /// 条目数（含标度条目）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否一条权重都没有。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按层栈与权重头部，经拷贝后端（`File`）把条目解码进驻留表；量化权重装成
    /// `Block::Packed` + 标度条目，非 F32 稠密条目不填块，留给装配期按格式拒绝。
    pub fn from_stack(stack: &Stack, header: &Header, loader: &impl Loader) -> Result<Self> {
        Self::assemble(stack, header, &LoaderSource(loader))
    }

    /// 按层栈与权重头部，经借用后端（内存归档、只读映射）把条目解码进驻留表；
    /// 条目字节不经过一次拷贝，解包后的驻留缓冲仍由本表拥有。
    pub fn from_reader_stack(stack: &Stack, header: &Header, reader: &impl Reader) -> Result<Self> {
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
        let name = format!("{}{WEIGHT_SUFFIX}", layer.name());
        Self::insert_named(params, header, &name, source)
    }

    /// 解码注意力块的 q/k/v/o 四个投影条目；量化条目装成打包块 + 标度，非 F32 稠密条目跳过。
    fn load_block<S: WeightSource>(
        block: &AttentionBlock,
        header: &Header,
        source: &S,
        params: &mut Self,
    ) -> Result<()> {
        for level in ["q", "k", "v", "o"] {
            let name = format!("{}.{level}{WEIGHT_SUFFIX}", block.name());
            Self::insert_named(params, header, &name, source)?;
        }
        Ok(())
    }

    /// 按条目名取条目解码进驻留表；头部缺条目报 [`Code::Missing`]，非 F32 稠密条目跳过不填。
    /// 量化条目（有标度伙伴）装成 [`Block::Packed`]（权重原始字节）外加一条 f32 标度条目，
    /// 并记录两者对应。
    fn insert_named<S: WeightSource>(
        params: &mut Self,
        header: &Header,
        name: &str,
        source: &S,
    ) -> Result<()> {
        let entry = header.get(name).ok_or_else(|| {
            Error::new(Code::Missing, format!("权重头部缺条目 `{name}`")).with(SITE)
        })?;
        match header.quantization(name)? {
            Some(quantized) => Self::load_quantized(params, name, entry, quantized, source),
            None => Self::load_dense(params, name, entry, source),
        }
    }

    /// 量化条目：打包字节取原始字节、标度解包成 f32，两条按对应关系入表。
    fn load_quantized<S: WeightSource>(
        params: &mut Self,
        name: &str,
        entry: &WeightEntry,
        quantized: Quantized<'_>,
        source: &S,
    ) -> Result<()> {
        let packed = source.bytes(entry)?;
        let scales = source.f32(quantized.scales())?.ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("量化权重 `{name}` 的标度条目不是 F32"),
            )
            .with(SITE)
        })?;
        params.insert_quantized(name, packed, quantized.scales_name(), scales);
        Ok(())
    }

    /// 稠密条目：F32 解包入表，非 F32 跳过不填（装配期按格式拒绝）。
    fn load_dense<S: WeightSource>(
        params: &mut Self,
        name: &str,
        entry: &WeightEntry,
        source: &S,
    ) -> Result<()> {
        let Some(weight) = source.f32(entry)? else {
            return Ok(());
        };
        params.insert(name.to_string(), weight);
        Ok(())
    }
}

pub mod descriptors {
    //! 编号 ↔ 形状/条目名描述表（D10 / ADR 0044）：只留 serve 装配侧。
    //!
    //! engine [`Bindings`](engine::exec::Bindings) 仍是纯 `u32 → Vec<f32>`，不解释编号含义，
    //! 也不加只读形状访问器；形状与条目名只在装配期由层声明推出，写进这张表供装配校验与诊断。

    use diagnostic::{Code, Error, Result};

    use crate::kernels::SITE;
    use crate::params::Params;

    /// 编号 ↔ 形状/条目名描述表条目。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct BindingDesc {
        /// 不透明绑定编号（与 engine `Bindings` 的键一致）。
        pub(crate) id: u32,
        /// 驻留权重条目名；激活缓冲为 `None`。
        pub(crate) entry: Option<String>,
        /// 声明形状（轴顺序与层视图一致）；激活只记一条轴长度，历史占位记 `[0]`。
        pub(crate) shape: Vec<usize>,
    }

    /// 装配期自洽：描述表编号与权重/激活台账一一对应，形状积（权重）或单轴长度（激活）相符。
    ///
    /// 描述表是装配侧事实；引擎不读这张表，也不解释编号含义。
    pub(crate) fn check_all(
        descriptors: &[BindingDesc],
        params: &Params,
        activations: &[(u32, usize)],
    ) -> Result<()> {
        for desc in descriptors {
            check_one(desc, params, activations)?;
        }
        Ok(())
    }

    fn check_one(desc: &BindingDesc, params: &Params, activations: &[(u32, usize)]) -> Result<()> {
        match &desc.entry {
            Some(entry) => check_weight(desc, entry, params),
            None => check_activation(desc, activations),
        }
    }

    /// 权重条目：参数表须有条目，形状积与权重整块长度相符。
    fn check_weight(desc: &BindingDesc, entry: &str, params: &Params) -> Result<()> {
        let Some(block) = params.get(entry) else {
            return Err(
                Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE),
            );
        };
        let product: usize = desc.shape.iter().product();
        if product != block.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "描述表条目 `{entry}` 形状积 {product} 与权重长度 {} 不符",
                    block.len()
                ),
            )
            .with(SITE));
        }
        Ok(())
    }

    /// 激活条目：编号须在激活台账内，单轴长度与台账相符；历史占位 `[0]` 由台账长度 0 对上。
    fn check_activation(desc: &BindingDesc, activations: &[(u32, usize)]) -> Result<()> {
        let Some(&len) = activations
            .iter()
            .find_map(|(id, activation_len)| (*id == desc.id).then_some(activation_len))
        else {
            return Err(
                Error::new(Code::Internal, format!("激活编号 {} 没有台账记录", desc.id)).with(SITE),
            );
        };
        if desc.shape.first().copied() != Some(len) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "激活编号 {} 形状 {:?} 与台账长度 {len} 不符",
                    desc.id, desc.shape
                ),
            )
            .with(SITE));
        }
        Ok(())
    }
}

pub mod vocab;
