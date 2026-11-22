//! `hybrid` 的层栈装配：K3 配比（每 3 个记忆层接 1 个混合层）与整栈一步。

use super::*;

/// 按 K3 配比堆起来的层栈：每 [`MEMORY_PER_ATTENTION`] 个记忆层后接 1 个混合层。
#[derive(Debug, Clone, PartialEq)]
pub struct Tower<S: Scalar = f32> {
    layers: Vec<Gear<S>>,
    width: usize,
    streams: usize,
}

impl<S: Scalar> Tower<S> {
    /// 按配比装配：`memory.len()` 必须是 `MEMORY_PER_ATTENTION × attention.len()`。
    ///
    /// 记忆层按给出的次序落在每组的前三格，混合层落在第四格；层序只由这里决定一次。
    ///
    /// # Errors
    ///
    /// - 混合层一个都没有（没有宽度可依，也没有注意力可跑）；
    /// - 记忆层个数不是 `3 × 混合层个数`（配比不是注释而是构造条件）；
    /// - 任一记忆层的读出宽 `d_v` 不等于混合层宽度（读数要加回主路，宽度不齐就是两张不同的表）；
    /// - 混合层之间宽度或流条数不一致（同一栈里各层几何必须同源）。
    pub fn new(attention: Vec<Hybrid<S>>, memory: Vec<crate::kda::Kda>) -> Result<Self> {
        let (width, streams) = geometry(&attention)?;
        check_ratio(attention.len(), memory.len())?;
        check_memory_width(&memory, width)?;
        let layers = interleave(attention, memory);
        Ok(Self {
            layers,
            width,
            streams,
        })
    }

    accessors! {
        /// 本栈宽度（= 混合层的 `n_h · d_h` = 记忆层读出宽 = 主路宽）。
        width -> usize = width,
        /// 残差流条数 `M`。
        streams -> usize = streams,
    }

    /// 层序的形态视图：`Memory` 与 `Attend` 的序列。
    pub fn order(&self) -> impl Iterator<Item = Kind> + '_ {
        self.layers.iter().map(|layer| match layer {
            Gear::Memory(_) => Kind::Memory,
            Gear::Attend(_) => Kind::Attend,
        })
    }

    /// 层数。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 装配后的初始状态：每个记忆层一份零记忆。
    pub fn state(&self) -> State<S> {
        State {
            memories: self
                .layers
                .iter()
                .filter_map(|layer| match layer {
                    Gear::Memory(kda) => Some(kda.memory::<S>()),
                    Gear::Attend(_) => None,
                })
                .collect(),
        }
    }

    /// 本 token 的查询铺成 `M` 条流：初值就是同一个值，混合之后才分开。
    ///
    /// # Errors
    ///
    /// `hidden` 宽度不是本栈宽度时返回 `Code::Invalid`。
    pub fn seed(&self, hidden: &[S]) -> Result<Vec<Vec<S>>> {
        if hidden.len() != self.width {
            return err!("初值宽度是 {}，本栈宽度是 {}", hidden.len(), self.width);
        }
        Ok(vec![hidden.to_vec(); self.streams])
    }

    /// 一次推进：按层序把记忆层与混合层各走一遍，末尾一次性提交。
    ///
    /// 记忆层先把本 token 写进**该层自己**的记忆，再读——与 [`crate::kda::Kda::read`] 的口径一致：
    /// 单个 token 写进去、立刻读得出来。读数只加回主路（第 0 条流），理由与 [`Hybrid::step`] 相同：
    /// 同一份增量加给每条流会让多流退化成单流。
    ///
    /// 缓存在整次推进里**只读**：位置取 `cache.seq() - 1`，先 `push` 再 `step` 由调用方保证。
    ///
    /// # Errors
    ///
    /// 流几何不符、记忆输入条数与记忆层不等、任一记忆层的门/键/值/查询几何或取值越界、查询几何不符、
    /// 缓存为空——都返回 `Code::Invalid`，且 `streams` 与 `state` **都没有被改动**。
    pub fn step(
        &self,
        streams: &mut [Vec<S>],
        state: &mut State<S>,
        cache: &Latent<S>,
        query: &Query<'_, S>,
        memories: &[MemoryStep<'_, S>],
        path: Path,
    ) -> Result<()> {
        cursor(cache)?;
        let mut staged = streams.to_vec();
        self.check_streams(&staged)?;
        let mut staged_state = state.clone();
        let mut walk = Walk {
            streams: &mut staged,
            state: &mut staged_state,
            cache,
            query,
            memories,
            path,
            used: 0,
        };
        walk.run(&self.layers)?;
        check_count(walk.used, memories.len())?;
        commit(streams, staged);
        *state = staged_state;
        Ok(())
    }

    /// 流条数与各流宽度必须正好是本栈的：多一条会在混流里被当成第 `M+1` 条流，窄一格会让
    /// [`accumulate`] 静默只加重叠部分。
    fn check_streams(&self, streams: &[Vec<S>]) -> Result<()> {
        if streams.len() != self.streams {
            return err!("流条数是 {}，本栈是 {}", streams.len(), self.streams);
        }
        match streams.iter().position(|row| row.len() != self.width) {
            Some(bad) => err!(
                "第 {bad} 条流宽度是 {}，本栈宽度是 {}",
                streams[bad].len(),
                self.width
            ),
            None => Ok(()),
        }
    }
}
