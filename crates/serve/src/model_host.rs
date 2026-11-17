//! CPU 模型宿主：装配结果 + 驻留执行器，承载张量级前向与 [`Predict`](crate::Predict) 路线 B 入口。
//!
//! 图装配与台账留在 [`assembly`](crate::assembly)；本模块只负责每步取数、历史 K/V 装载与当前步
//! K/V 写回，不另存模型侧 KV 账。

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Device};
use engine::{Batch, Sequence};
use nn::Stack;

use crate::assembly::{
    Assembly, LayerKind, Params, assemble, bind_activations, bind_weights, submit_and_fetch,
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

    /// 批量收集各序列末位 token，按准入序折成整批索引缓冲；非查表层首层拒绝。
    fn predict_input_batch(&self, sequences: &[Sequence]) -> Result<Vec<f32>> {
        if self.assembly.input_kind != LayerKind::Embedding {
            return Err(Error::new(
                Code::Unsupported,
                "首层不是查表层：序列只持有 token id，无法提供隐向量输入；向量主链通路随连续批落地",
            )
            .with(SITE));
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
        let width = self.assembly.output_len;
        if flat.len() != batch * width {
            return Err(Error::new(
                Code::Internal,
                format!(
                    "整批输出长度 {len} 与 {batch}×{width} 不符",
                    len = flat.len()
                ),
            )
            .with(SITE));
        }
        Ok(flat.chunks(width).map(|row| row.to_vec()).collect())
    }

    /// 从页账把各块已缓存的历史 K/V 整行读出，按块拼接进历史缓冲；每步重绑，不另存模型侧 KV 账。
    ///
    /// 历史位置顺序即页行顺序，是串行依赖，保留逐行循环；每行 `[k|v]` 整段切片切分后整块
    /// `copy_from_slice` 拼进缓冲，不写逐元素拷贝。上一轮同编号缓冲经 `bind` 返回后接住复用容量，
    /// 历史单调变长，多数步次不再堆分配；首次没有旧缓冲时新建。
    fn load_history(&mut self, sequence: &Sequence, batch: &Batch) -> Result<()> {
        let id = sequence.id();
        let hist = sequence.len();
        for block in &self.assembly.blocks {
            let kv = block.kv_width;
            let head_dim = block.head_dim;
            let len = hist * kv;
            // 用占位空 Vec 换出旧历史块，填好后再绑回执行器。
            let mut hist_k = self
                .executor
                .bind(block.hist_k, Vec::new())
                .unwrap_or_default();
            let mut hist_v = self
                .executor
                .bind(block.hist_v, Vec::new())
                .unwrap_or_default();
            hist_k.resize(len, 0.0);
            hist_v.resize(len, 0.0);
            for token in 0..hist {
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
                reorder_history_row(&mut hist_k, &mut hist_v, row, kv, head_dim, hist, token);
            }
            self.executor.bind(block.hist_k, hist_k);
            self.executor.bind(block.hist_v, hist_v);
        }
        Ok(())
    }

    /// 前向后取回各块当前步 K/V 投影结果，拼成 `[k|v]` 页行写回当前位置；页账归调度器，模型不另存。
    fn write_current_kv(&mut self, sequence: &Sequence, batch: &mut Batch) -> Result<()> {
        let id = sequence.id();
        let position = sequence.len();
        for block in &self.assembly.blocks {
            let kv = block.kv_width;
            let k = bound_slice(&self.executor, block.k, "K")?;
            let v = bound_slice(&self.executor, block.v, "V")?;
            let mut row = Vec::with_capacity(2 * kv);
            row.extend_from_slice(k);
            row.extend_from_slice(v);
            batch.write_kv(id, position, &row)?;
        }
        Ok(())
    }
}

/// 把一行页行 `[k(head,d) | v(head,d)]` 按头优先块 `[head, token, d]` 写进历史缓冲；
/// heads=1 时排列与行优先一致。页行到块的重排是布局搬运，头段整块 `copy_from_slice`。
fn reorder_history_row(
    hist_k: &mut [f32],
    hist_v: &mut [f32],
    row: &[f32],
    kv: usize,
    head_dim: usize,
    hist: usize,
    token: usize,
) {
    let heads = kv / head_dim;
    let (k, v) = row.split_at(kv);
    for head in 0..heads {
        let seg = head * head_dim;
        let start = head * hist * head_dim + token * head_dim;
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
        // 含注意力块的链整批 K/V 搬运尚未接线：回退逐条前向，保持与单条路径等价。
        if self.assembly.blocks.is_empty() {
            self.batch_forward(sequences)
        } else {
            sequences
                .iter()
                .map(|sequence| self.logits(sequence, batch))
                .collect()
        }
    }
}
