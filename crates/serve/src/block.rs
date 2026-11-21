//! 注意力块装配：单头几何/宽度链/四投影权重校验，拆成 q/k/v/o + decode 注意力 + 残差子节点。

use diagnostic::{Code, Error, Result};
use engine::exec::Launch;
use nn::{AttentionAlgo, AttentionBlock, Contract, Linear, WEIGHT_SUFFIX};

use crate::assembly::NodeSink;
use crate::kernels::SITE;
use crate::params::Params;
use crate::plan::{LayerKind, LayerPlan, check_weight, linear_plan};

/// 一个注意力块接线记录：当前步 K/V 取回拼页行；历史 K/V 缓冲每步重绑；旁路即块输入编号。
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockBinding {
    pub(crate) k: u32,
    pub(crate) v: u32,
    pub(crate) hist_k: u32,
    pub(crate) hist_v: u32,
    /// K/V 整段宽度（`heads · head_dim`），页行按它切 K/V 半行。
    pub(crate) kv_width: usize,
    /// 单头头宽；历史与激活按头分段，头数 = `kv_width / head_dim`。
    pub(crate) head_dim: usize,
}

/// 块内投影条目名：`{块名}.{level}.weight`。
fn block_entry(block: &AttentionBlock, level: &str) -> String {
    format!("{}.{level}{WEIGHT_SUFFIX}", block.name())
}

/// decode 算子名映射：块内算法沿用 nn 绑定的 prefill 族别，装配侧切成对应 decode 变体。
fn decode_operator(algo: AttentionAlgo) -> &'static str {
    match algo {
        AttentionAlgo::Naive => "decode_attention_f32",
        AttentionAlgo::Flash => "decode_flash_attention_f32",
    }
}

/// 几何与宽度链衔接：`kv_width` 是 `heads · head_dim`（整除由 nn 构造校验）；块输入对齐上一层输出宽度。
fn check_head(block: &AttentionBlock, prev_out: Option<usize>) -> Result<()> {
    let hidden = block.hidden();
    if let Some(previous) = prev_out
        && previous != hidden
    {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "注意力块 `{}` 输入宽度 {hidden} 与上一层输出宽度 {previous} 不符",
                block.name()
            ),
        )
        .with(SITE));
    }
    Ok(())
}

/// 块内四个投影的装配计划：q/k/v/o 各一份，身份/算子名/权重/标度形状齐全；权重校验在
/// [`block_plans`] 内一次走完。宽度链不在这里逐条核——q/k/v 取 hidden、o 取 kv 由块几何构造期保证。
struct BlockPlans {
    q: LayerPlan,
    k: LayerPlan,
    v: LayerPlan,
    o: LayerPlan,
}

/// 块内一条投影计划：名字取 `{块}.{level}`、条目取 `{块}.{level}.weight`，计划与权重校验走
/// 共享 [`crate::plan::linear_plan`] / [`crate::plan::check_weight`]，不各写一套。
fn projection_plan_for(
    block: &AttentionBlock,
    level: &str,
    linear: &Linear,
    params: &Params,
) -> Result<LayerPlan> {
    let entry = block_entry(block, level);
    let plan = linear_plan(&format!("{}.{level}", block.name()), &entry, linear, params)?;
    check_weight(&plan, params)?;
    Ok(plan)
}

/// 四路投影计划：逐条构造并校验条目与可选标度。
fn block_plans(block: &AttentionBlock, params: &Params) -> Result<BlockPlans> {
    Ok(BlockPlans {
        q: projection_plan_for(block, "q", block.q(), params)?,
        k: projection_plan_for(block, "k", block.k(), params)?,
        v: projection_plan_for(block, "v", block.v(), params)?,
        o: projection_plan_for(block, "o", block.out(), params)?,
    })
}

/// q/k/v 三路投影共享块输入扇出；返回三个投影输出编号。
fn wire_qkv(sink: &mut NodeSink<'_>, prev: u32, plans: &BlockPlans) -> Result<(u32, u32, u32)> {
    let q = sink.push_node(prev, &plans.q, None)?;
    let k = sink.push_node(prev, &plans.k, None)?;
    let v = sink.push_node(prev, &plans.v, None)?;
    Ok((q, k, v))
}

/// RoPE 子节点：对一路投影向量按位置做旋转；单序列位置走标量槽，由宿主每步回填；
/// 批形态逐序列位置走独立输入位置表（ADR 0043 §5），两通道不在同节点并存。
/// 头宽走 Launch 几何第二维（ADR 0043 §3：静态几何不进标量槽），单头时等于整段宽度。返回旋转后编号。
fn wire_rope(
    sink: &mut NodeSink<'_>,
    input: u32,
    width: usize,
    head_dim: usize,
    pos_table: u32,
) -> Result<u32> {
    // 第二路为批位置表（ADR 0043 §5）；单序列走标量槽时该表编号仍登记，宿主按形态回填。
    let output = sink.emit(
        "rope_f32",
        Launch::new([1, 1, 1], [width as u32, head_dim as u32, 1]),
        &[input, pos_table],
        width,
    )?;
    sink.descriptors.push(crate::descriptors::BindingDesc {
        id: output,
        entry: None,
        shape: vec![width],
    });
    sink.rope.push(output);
    Ok(output)
}

/// decode 注意力节点：q + 历史 K + 历史 V → 注意力输出；返回 `(hist_k, hist_v, attn)` 编号。
fn wire_attention(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    q_slot: u32,
    hist_len_table: u32,
) -> Result<(u32, u32, u32)> {
    let kv = block.kv_width();
    let hist_k = sink.alloc_table()?;
    let hist_v = sink.alloc_table()?;
    let attn_slot = sink.emit(
        decode_operator(block.algo()),
        Launch::new([1, 1, 1], [kv as u32, block.head_dim() as u32, 1]),
        // 第四路为 ragged 段长表（ADR 0046 §8）；单序列缺省回退单序列 heads 入口。
        &[q_slot, hist_k, hist_v, hist_len_table],
        kv,
    )?;
    sink.descriptors.push(crate::descriptors::BindingDesc {
        id: attn_slot,
        entry: None,
        shape: vec![kv],
    });
    Ok((hist_k, hist_v, attn_slot))
}

/// o 投影回 hidden，再接残差旁路（o 输出 + 块输入）；返回块输出编号。
fn wire_output(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    attn_slot: u32,
    plans: &BlockPlans,
) -> Result<u32> {
    let hidden = block.hidden();
    let pre_residual = sink.push_node(attn_slot, &plans.o, None)?;
    sink.emit(
        "residual_add_f32",
        Launch::new([1, 1, 1], [hidden as u32, 1, 1]),
        &[pre_residual, prev],
        hidden,
    )
}

/// 把注意力块拆成子节点接进图；返回块输出激活编号。
fn push_block(
    sink: &mut NodeSink<'_>,
    prev: u32,
    block: &AttentionBlock,
    plans: &BlockPlans,
) -> Result<u32> {
    let (pos_table, len_table) = sink.ensure_tables()?;
    let (k_slot, v_slot, hist_k, hist_v, out_slot) =
        wire_block_graph(sink, block, prev, plans, pos_table, len_table)?;
    sink.blocks.push(BlockBinding {
        k: k_slot,
        v: v_slot,
        hist_k,
        hist_v,
        kv_width: block.kv_width(),
        head_dim: block.head_dim(),
    });
    Ok(out_slot)
}

/// 块内子节点接线：q/k/v 投影 + 双 RoPE + decode 注意力 + o 投影残差；返回 `(k, v, hist_k, hist_v, out)`。
#[allow(clippy::too_many_arguments)]
fn wire_block_graph(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    plans: &BlockPlans,
    pos_table: u32,
    len_table: u32,
) -> Result<(u32, u32, u32, u32, u32)> {
    let (q_slot, k_slot, v_slot) = wire_qkv_rope(sink, block, prev, plans, pos_table)?;
    let (hist_k, hist_v, attn_slot) = wire_attention(sink, block, q_slot, len_table)?;
    let out_slot = wire_output(sink, block, prev, attn_slot, plans)?;
    Ok((k_slot, v_slot, hist_k, hist_v, out_slot))
}

/// q/k/v 投影后，q/k 各插一路 RoPE 旋转再进注意力；写回页行的 K 取旋转后的一路。
fn wire_qkv_rope(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    plans: &BlockPlans,
    pos_table: u32,
) -> Result<(u32, u32, u32)> {
    let (q_slot, k_slot, v_slot) = wire_qkv(sink, prev, plans)?;
    let head_dim = block.head_dim();
    let q_slot = wire_rope(sink, q_slot, block.kv_width(), head_dim, pos_table)?;
    let k_slot = wire_rope(sink, k_slot, block.kv_width(), head_dim, pos_table)?;
    Ok((q_slot, k_slot, v_slot))
}

/// 注意力块层：几何/权重校验后拆成 q/k/v/o + decode 注意力 + 残差子节点。
pub(crate) fn advance_block(
    sink: &mut NodeSink<'_>,
    prev: u32,
    prev_out: Option<usize>,
    block: &AttentionBlock,
    params: &Params,
) -> Result<(u32, usize, usize, LayerKind)> {
    check_head(block, prev_out)?;
    let plans = block_plans(block, params)?;
    let next = push_block(sink, prev, block, &plans)?;
    let hidden = block.hidden();
    Ok((next, hidden, hidden, LayerKind::Projection))
}
