//! CPU 模型宿主：装配结果 + 驻留执行器，承载张量级前向与 [`Predict`](crate::Predict) 路线 B 入口。
//!
//! 图装配与台账留在 [`assembly`](crate::assembly)；本模块只负责每步取数、历史 K/V 装载与当前步
//! K/V 写回，不另存模型侧 KV 账。

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Device};
use engine::{Batch, Sequence};
use nn::Stack;
use std::path::Path;

use crate::assembly::{
    Assembly, BlockBinding, LayerKind, Params, assemble, bind_activations, bind_weights,
    submit_and_fetch,
};
use crate::kernels::SITE;

/// CPU 模型宿主：驻留权重 + 装配好的图 + 登记了内核的执行器。
#[derive(Debug)]
pub struct CpuModel {
    executor: CpuExecutor,
    assembly: Assembly,
}

impl CpuModel {
    /// 由层栈与驻留参数表装配；形状、宽度、权重缺失都在构造期报出。
    pub fn new(stack: &Stack, params: &Params) -> Result<Self> {
        let mut executor = CpuExecutor::new(Device::new("cpu", "generic"));
        crate::kernels::register_cpu(&mut executor)?;
        let assembly = assemble(stack, params)?;
        bind_weights(&mut executor, &assembly, params)?;
        bind_activations(&mut executor, &assembly);

        Ok(Self { executor, assembly })
    }

    /// 从权重文件就地装配：打开头部 → 推出层栈 → 装载驻留参数表 → 装配图与内核执行器。
    ///
    /// 字节来源按后端类别在类型上可见分流（ADR 0036 §1），不立平行 `Source` 聚合 trait：
    /// - Unix + `mmap` feature：[`weights::MmapSource`] 实现 [`weights::Reader`]，条目字节从映射区
    ///   借用、不经过 `Vec<u8>` 中转，走 [`Params::from_reader_stack`]；
    /// - 其余平台或未开 feature：[`weights::safetensors::File`] 实现 [`weights::Loader`]，按条目
    ///   seek+read 拷进缓冲，走 [`Params::from_stack`]。
    ///
    /// 两条路径的栈、参数与模型构造一致，差别只在字节来源；非 Unix 不提供假装映射的空实现。
    ///
    /// # Errors
    ///
    /// 文件打不开或头部校验失败、层栈推出失败、参数缺条目或类型/形状不符、装配期宽度链断裂，
    /// 都沿既有诊断类别上抛。
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        #[cfg(all(feature = "mmap", unix))]
        {
            Self::from_mmap(path.as_ref())
        }
        #[cfg(not(all(feature = "mmap", unix)))]
        {
            Self::from_file(path.as_ref())
        }
    }

    /// 映射借用路径：条目字节从只读映射区借出，不经过 `Vec<u8>` 中转。
    #[cfg(all(feature = "mmap", unix))]
    fn from_mmap(path: &Path) -> Result<Self> {
        let source = weights::MmapSource::open(path)?;
        let stack = Stack::from_header(source.header())?;
        let params = Params::from_reader_stack(&stack, source.header(), &source)?;
        Self::new(&stack, &params)
    }

    /// 文件拷贝路径：按条目 seek+read 拷进驻留缓冲；非 Unix 或未开 mmap feature 走此路。
    #[cfg(not(all(feature = "mmap", unix)))]
    fn from_file(path: &Path) -> Result<Self> {
        let file = weights::safetensors::File::open(path)?;
        let stack = Stack::from_header(file.header())?;
        let params = Params::from_stack(&stack, file.header(), &file)?;
        Self::new(&stack, &params)
    }

    /// 一步张量级前向；输入长度必须等于首层输入宽度，返回末层激活。
    ///
    /// 权重在构造期驻留一次；每步只覆写输入与逐层激活缓冲，不重新分配。
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.assembly.input_len {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "输入 {} 维，模型首层要 {} 维",
                    input.len(),
                    self.assembly.input_len
                ),
            )
            .with(SITE));
        }
        self.executor.bind(self.assembly.input, input.to_vec());
        // 张量级前向没有序列位置：rope 位置取 0（角度为 0 的恒等旋转）。
        self.assembly.set_rope_positions(0);
        submit_and_fetch(&mut self.executor, &self.assembly)
    }

    /// 模型输出宽度（末层输出）。
    pub fn output_len(&self) -> usize {
        self.assembly.output_len
    }

    /// 编号 ↔ 形状/条目名描述表（只读，D10 / ADR 0044）：形状事实留在装配侧，引擎不解释编号含义。
    /// 仅装配自检与测试读取；生产前向不读这张表。
    #[cfg(test)]
    pub(crate) fn descriptors(&self) -> &[crate::assembly::BindingDesc] {
        self.assembly.descriptors()
    }

    /// 按首层族别把序列当前位置 token 折成一步前向的输入缓冲。
    ///
    /// 查表层首层收一个索引（decode 一步取上下文末位 token）；投影/点wise/归一化/注意力块首层需要的是
    /// 隐向量，而序列只持有 token id，给不出向量，当前通路拒绝，向量主链随连续批刀落地。
    fn predict_input(&self, sequence: &Sequence) -> Result<Vec<f32>> {
        self.predict_input_batch(std::slice::from_ref(sequence))
    }

    /// 批量收集各序列末位 token，按准入序折成整批索引缓冲；非查表层首层走向量主链占位。
    fn predict_input_batch(&self, sequences: &[Sequence]) -> Result<Vec<f32>> {
        if self.assembly.input_kind != LayerKind::Embedding {
            // 向量主链：投影/点wise/归一化/注意力首层收隐向量，序列只持 token id，给不出真实嵌入；
            // 按设计稿在 serve 层造零向量占位（真实嵌入由外部/后续通路供给），先让非查表层首层可跑。
            return Ok(vec![0.0; self.assembly.input_len * sequences.len()]);
        }
        sequences
            .iter()
            .map(|sequence| {
                sequence
                    .context()
                    .last()
                    .copied()
                    .ok_or_else(|| {
                        Error::new(Code::Invalid, "序列为空：没有可前向的 token").with(SITE)
                    })
                    .map(|token| token as f32)
            })
            .collect()
    }

    /// 把输入与各逐层激活缓冲重绑到 `batch` 份单序列段长；驻留权重不乘批。
    ///
    /// 前向每步覆写输入、op 逐层覆写激活，无跨步残留；历史 K/V 块缓冲由 [`load_history`](Self::load_history)
    /// 单独重绑，不在本台账内。
    fn resize_for_batch(&mut self, batch: usize) {
        self.executor.bind(
            self.assembly.input,
            vec![0.0; self.assembly.input_len * batch],
        );
        for (id, len) in &self.assembly.activations {
            self.executor.bind(*id, vec![0.0; len * batch]);
        }
    }

    /// 无 K/V 块的整批前向：一次图提交，输出整块按单序列输出宽拆回各行；行序对齐准入序。
    fn batch_forward(&mut self, sequences: &[Sequence]) -> Result<Vec<Vec<f32>>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let batch = sequences.len();
        let input = self.predict_input_batch(sequences)?;
        self.resize_for_batch(batch);
        self.executor.bind(self.assembly.input, input);
        // 无块链没有 rope 节点：位置取 0（角度为 0 的恒等旋转）。
        self.assembly.set_rope_positions(0);
        let flat = submit_and_fetch(&mut self.executor, &self.assembly)?;
        self.split_batch_output(flat, batch)
    }

    /// 单序列历史装填：批历史装填的单段特例（段游标 0、一段历史），复用同一段紧排布局。
    fn load_history(&mut self, sequence: &Sequence, batch: &Batch) -> Result<()> {
        self.load_history_batch(std::slice::from_ref(sequence), batch)
    }

    /// 单序列当前步 K/V 写回：批写回路径的单段特例，行写回各自页行。
    fn write_current_kv(&mut self, sequence: &Sequence, batch: &mut Batch) -> Result<()> {
        self.write_current_kv_batch(std::slice::from_ref(sequence), batch)
    }

    /// 批历史装填：逐块把各段历史 K/V 按 §7 段紧排拼进一块缓冲；段间游标累加。
    /// 历史位置顺序即页行顺序，是串行依赖，保留逐段逐行循环；行内头段整块 `copy_from_slice`。
    fn load_history_batch(&mut self, sequences: &[Sequence], batch: &Batch) -> Result<()> {
        // 块清单是纯数据，先快照成自有副本再进入可变借用；块绑定不含缓冲所有权。
        let blocks = self.assembly.blocks.clone();
        for block in blocks {
            self.load_block_history(block, sequences, batch)?;
        }
        Ok(())
    }

    /// 单块批历史装填：段紧排拼进该块历史缓冲；段内复用头优先块重排。
    fn load_block_history(
        &mut self,
        block: BlockBinding,
        sequences: &[Sequence],
        batch: &Batch,
    ) -> Result<()> {
        let kv = block.kv_width;
        let head_dim = block.head_dim;
        let hist_total = sequences.iter().map(|sequence| sequence.len() * kv).sum();
        let mut hist_k = self
            .executor
            .bind(block.hist_k, Vec::new())
            .unwrap_or_default();
        let mut hist_v = self
            .executor
            .bind(block.hist_v, Vec::new())
            .unwrap_or_default();
        hist_k.resize(hist_total, 0.0);
        hist_v.resize(hist_total, 0.0);
        let mut cursor = 0;
        for sequence in sequences {
            let hist = sequence.len();
            let id = sequence.id();
            for token in 0..hist {
                place_history_row(
                    &mut hist_k,
                    &mut hist_v,
                    batch,
                    id,
                    token,
                    kv,
                    head_dim,
                    hist,
                    cursor,
                )?;
            }
            cursor += hist * kv;
        }
        self.executor.bind(block.hist_k, hist_k);
        self.executor.bind(block.hist_v, hist_v);
        Ok(())
    }

    /// 回填批形态两张共享表：逐序列当前位置写进 rope 位置表，历史行数写进 attention 段长表。
    /// 两表元素同为序列当前长度，语义分属位置通道与段长通道，分别绑各自编号。
    fn fill_batch_tables(&mut self, sequences: &[Sequence]) {
        let lengths: Vec<f32> = sequences.iter().map(|s| s.len() as f32).collect();
        if let Some(id) = self.assembly.rope_pos_table {
            self.executor.bind(id, lengths.clone());
        }
        if let Some(id) = self.assembly.hist_len_table {
            self.executor.bind(id, lengths);
        }
    }

    /// 批当前步写回：逐块把各段第 `seq` 行 K/V 拼成页行，写回各自序列当前位置。
    fn write_current_kv_batch(&mut self, sequences: &[Sequence], batch: &mut Batch) -> Result<()> {
        let blocks = self.assembly.blocks.clone();
        for block in blocks {
            self.write_block_kv(block, sequences, batch)?;
        }
        Ok(())
    }

    /// 单块批写回：取该块激活缓冲各段 K/V 行，拼 `[k|v]` 页行写回页账。
    fn write_block_kv(
        &mut self,
        block: BlockBinding,
        sequences: &[Sequence],
        batch: &mut Batch,
    ) -> Result<()> {
        let kv = block.kv_width;
        let k = bound_slice(&self.executor, block.k, "K")?;
        let v = bound_slice(&self.executor, block.v, "V")?;
        for (seq, sequence) in sequences.iter().enumerate() {
            let start = seq * kv;
            let mut row = Vec::with_capacity(2 * kv);
            row.extend_from_slice(&k[start..start + kv]);
            row.extend_from_slice(&v[start..start + kv]);
            batch.write_kv(sequence.id(), sequence.len(), &row)?;
        }
        Ok(())
    }

    /// 含块链整批前向：批表回填、历史段紧排装填、一次提交、按段写回页行，输出按宽拆行。
    fn forward_block_batch(
        &mut self,
        sequences: &[Sequence],
        batch: &mut Batch,
    ) -> Result<Vec<Vec<f32>>> {
        let batch_n = sequences.len();
        let input = self.predict_input_batch(sequences)?;
        self.resize_for_batch(batch_n);
        self.executor.bind(self.assembly.input, input);
        self.fill_batch_tables(sequences);
        self.load_history_batch(sequences, batch)?;
        let flat = submit_and_fetch(&mut self.executor, &self.assembly)?;
        self.write_current_kv_batch(sequences, batch)?;
        self.split_batch_output(flat, batch_n)
    }

    /// 整批输出按单序列输出宽拆回各行；总长与 `batch×width` 不符归内部错误。
    fn split_batch_output(&self, flat: Vec<f32>, batch_n: usize) -> Result<Vec<Vec<f32>>> {
        let width = self.assembly.output_len;
        if flat.len() != batch_n * width {
            return Err(Error::new(
                Code::Internal,
                format!(
                    "整批输出长度 {len} 与 {batch_n}×{width} 不符",
                    len = flat.len()
                ),
            )
            .with(SITE));
        }
        Ok(flat.chunks(width).map(|row| row.to_vec()).collect())
    }
}

/// 读出一页历史行，校验行宽后按头优先块写进段缓冲；`cursor` 为批段基址（单序列为 0）。
#[allow(clippy::too_many_arguments)]
fn place_history_row(
    hist_k: &mut [f32],
    hist_v: &mut [f32],
    batch: &Batch,
    id: u64,
    token: usize,
    kv: usize,
    head_dim: usize,
    hist: usize,
    cursor: usize,
) -> Result<()> {
    let row = batch.kv(id, token)?;
    let expected = 2 * kv;
    if row.len() != expected {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "KV 行宽 {} 与块 KV 宽度 {kv} 推出的行宽 {expected}",
                row.len()
            ),
        )
        .with(SITE));
    }
    reorder_history_row(hist_k, hist_v, row, kv, head_dim, hist, token, cursor);
    Ok(())
}

/// 把一行页行 `[k(head,d) | v(head,d)]` 按头优先块 `[head, token, d]` 写进历史缓冲；
/// `cursor` 为批段基址（单序列为 0），段内布局按本段行数 `hist` 排。heads=1 时排列与行优先
/// 一致。页行到块的重排是布局搬运，头段整块 `copy_from_slice`。
#[allow(clippy::too_many_arguments)]
fn reorder_history_row(
    hist_k: &mut [f32],
    hist_v: &mut [f32],
    row: &[f32],
    kv: usize,
    head_dim: usize,
    hist: usize,
    token: usize,
    cursor: usize,
) {
    let heads = kv / head_dim;
    let (k, v) = row.split_at(kv);
    for head in 0..heads {
        let seg = head * head_dim;
        let start = cursor + head * hist * head_dim + token * head_dim;
        hist_k[start..start + head_dim].copy_from_slice(&k[seg..seg + head_dim]);
        hist_v[start..start + head_dim].copy_from_slice(&v[seg..seg + head_dim]);
    }
}

/// 取回已预绑的投影缓冲；编号在装配期绑定，取回失败归内部错误。
fn bound_slice<'a>(executor: &'a CpuExecutor, id: u32, what: &str) -> Result<&'a [f32]> {
    executor.get(id).ok_or_else(|| {
        Error::new(Code::Internal, format!("{what} 投影缓冲已预绑，取回失败")).with(SITE)
    })
}

impl crate::Predict for CpuModel {
    fn logits(&mut self, sequence: &Sequence, batch: &mut Batch) -> Result<Vec<f32>> {
        let input = self.predict_input(sequence)?;
        self.resize_for_batch(1);
        self.executor.bind(self.assembly.input, input);
        self.assembly.set_rope_positions(sequence.len() as u32);
        self.load_history(sequence, batch)?;
        let logits = submit_and_fetch(&mut self.executor, &self.assembly)?;
        self.write_current_kv(sequence, batch)?;
        Ok(logits)
    }

    fn logits_batch(
        &mut self,
        sequences: &[engine::Sequence],
        batch: &mut Batch,
    ) -> Result<Vec<Vec<f32>>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        // 无块链走无 K/V 整批前向；含块链整批一次提交，块链前向细节在 [`forward_block_batch`]。
        if self.assembly.blocks.is_empty() {
            return self.batch_forward(sequences);
        }
        self.forward_block_batch(sequences, batch)
    }
}
