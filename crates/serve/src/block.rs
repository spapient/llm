//! 注意力块装配：单头几何/宽度链/四投影权重校验，拆成 q/k/v/o + decode 注意力 + 残差子节点。
//!
//! 块内算子名沿用 nn 绑定的 prefill 族别，装配侧切成对应 decode 变体；历史 K/V 缓冲编号与
//! `kv_width` 记进 [`BlockBinding`]，运行时由模型宿主按它装载历史、取回当前步写回页行。

use diagnostic::{Code, Error, Result};
use engine::exec::{Launch, Node};
use nn::{AttentionAlgo, AttentionBlock, Contract, Linear, WEIGHT_SUFFIX};

use crate::assembly::NodeSink;
use crate::kernels::SITE;
use crate::params::Params;
use crate::plan::{
    LayerKind, check_weight_entry, plan_operator, projection_shapes, validate_quantized_shape,
};

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

/// 注意力块装配校验：单头几何、宽度链衔接、四个投影条目齐备且长度相符。
fn check_block(block: &AttentionBlock, prev_out: Option<usize>, params: &Params) -> Result<()> {
    check_head(block, prev_out)?;
    check_block_weights(block, params)
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

/// 四个投影条目：条目齐备、长度与格式相符。量化投影按打包字节 + 标度条数核对，与普通
/// [`crate::plan::check_weight_entry`] 同一份判据；`plan_operator` 的格式校验在 [`block_operators`] 一次走完。
fn check_block_weights(block: &AttentionBlock, params: &Params) -> Result<()> {
    for (level, linear) in [
        ("q", block.q()),
        ("k", block.k()),
        ("v", block.v()),
        ("o", block.out()),
    ] {
        check_projection(block, level, linear, params)?;
    }
    Ok(())
}

/// 一条块内投影：整除校验后，按格式取打包字节/标度条数，再用共享判据核对条目与可选标度。
fn check_projection(
    block: &AttentionBlock,
    level: &str,
    linear: &Linear,
    params: &Params,
) -> Result<()> {
    let entry = block_entry(block, level);
    validate_quantized_shape(linear.format(), linear.in_features())?;
    let (weight_len, _, scales_len, _) =
        projection_shapes(linear.format(), linear.in_features(), linear.out_features());
    let scale_entry = params.scale_entry(&entry);
    check_weight_entry(params, &entry, weight_len, scale_entry, scales_len)
}

/// 块内四个投影的算子名与可选标度条目名；格式校验在 `plan_operator` 内一次完成，
/// 标度条目名从参数表取回（稠密投影为 `None`）。
struct ProjectionOps {
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
    q_scale: Option<String>,
    k_scale: Option<String>,
    v_scale: Option<String>,
    o_scale: Option<String>,
}

/// 量化投影的标度条目名：稠密投影无标度，返回 `None`。
fn scale_entry_of(block: &AttentionBlock, params: &Params, level: &str) -> Option<String> {
    params.scale_entry(&block_entry(block, level)).map(str::to_owned)
}

fn block_operators(block: &AttentionBlock, params: &Params) -> Result<ProjectionOps> {
    Ok(ProjectionOps {
        q: plan_operator(&format!("{}.q", block.name()), block.q())?,
        k: plan_operator(&format!("{}.k", block.name()), block.k())?,
        v: plan_operator(&format!("{}.v", block.name()), block.v())?,
        o: plan_operator(&format!("{}.o", block.name()), block.out())?,
        q_scale: scale_entry_of(block, params, "q"),
        k_scale: scale_entry_of(block, params, "k"),
        v_scale: scale_entry_of(block, params, "v"),
        o_scale: scale_entry_of(block, params, "o"),
    })
}

/// q/k/v 三路投影共享块输入扇出；返回三个投影输出编号。
fn wire_qkv(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    ops: &ProjectionOps,
) -> Result<(u32, u32, u32)> {
    let q = sink.push_projection(prev, block.q(), &block_entry(block, "q"), ops.q, ops.q_scale.as_deref())?;
    let k = sink.push_projection(prev, block.k(), &block_entry(block, "k"), ops.k, ops.k_scale.as_deref())?;
    let v = sink.push_projection(prev, block.v(), &block_entry(block, "v"), ops.v, ops.v_scale.as_deref())?;
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
    let output = sink.slots.activation()?;
    let mut node = Node::new(
        "rope_f32".to_string(),
        Launch::new([1, 1, 1], [width as u32, head_dim as u32, 1]),
    );
    node.bind_input(input);
    // 第二路为批位置表（ADR 0043 §5）；单序列走标量槽时该表编号仍登记，宿主按形态回填。
    node.bind_input(pos_table);
    node.bind_output(output);
    sink.graph.push(node);
    sink.activations.push((output, width));
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
    let hist_k = sink.slots.activation()?;
    let hist_v = sink.slots.activation()?;
    sink.activations.push((hist_k, 0));
    sink.activations.push((hist_v, 0));
    sink.descriptors.push(crate::descriptors::BindingDesc {
        id: hist_k,
        entry: None,
        shape: vec![0],
    });
    sink.descriptors.push(crate::descriptors::BindingDesc {
        id: hist_v,
        entry: None,
        shape: vec![0],
    });
    let attn_slot = sink.slots.activation()?;
    let mut attention = Node::new(
        decode_operator(block.algo()).to_string(),
        Launch::new([1, 1, 1], [kv as u32, block.head_dim() as u32, 1]),
    );
    attention.bind_input(q_slot);
    attention.bind_input(hist_k);
    attention.bind_input(hist_v);
    // 第四路为 ragged 段长表（ADR 0046 §8）；单序列缺省回退单序列 heads 入口。
    attention.bind_input(hist_len_table);
    attention.bind_output(attn_slot);
    sink.graph.push(attention);
    sink.activations.push((attn_slot, kv));
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
    ops: &ProjectionOps,
) -> Result<u32> {
    let hidden = block.hidden();
    let pre_residual = sink.push_projection(
        attn_slot,
        block.out(),
        &block_entry(block, "o"),
        ops.o,
        ops.o_scale.as_deref(),
    )?;
    let out_slot = sink.slots.activation()?;
    let mut residual = Node::new(
        "residual_add_f32".to_string(),
        Launch::new([1, 1, 1], [hidden as u32, 1, 1]),
    );
    residual.bind_input(pre_residual);
    residual.bind_input(prev);
    residual.bind_output(out_slot);
    sink.graph.push(residual);
    sink.activations.push((out_slot, hidden));
    Ok(out_slot)
}

/// 把注意力块拆成子节点接进图；返回块输出激活编号。
fn push_block(
    sink: &mut NodeSink<'_>,
    prev: u32,
    block: &AttentionBlock,
    params: &Params,
) -> Result<u32> {
    let ops = block_operators(block, params)?;
    let (pos_table, len_table) = sink.ensure_tables()?;
    let (k_slot, v_slot, hist_k, hist_v, out_slot) =
        wire_block_graph(sink, block, prev, &ops, pos_table, len_table)?;
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
    ops: &ProjectionOps,
    pos_table: u32,
    len_table: u32,
) -> Result<(u32, u32, u32, u32, u32)> {
    let (q_slot, k_slot, v_slot) = wire_qkv_rope(sink, block, prev, ops, pos_table)?;
    let (hist_k, hist_v, attn_slot) = wire_attention(sink, block, q_slot, len_table)?;
    let out_slot = wire_output(sink, block, prev, attn_slot, ops)?;
    Ok((k_slot, v_slot, hist_k, hist_v, out_slot))
}

/// q/k/v 投影后，q/k 各插一路 RoPE 旋转再进注意力；写回页行的 K 取旋转后的一路。
fn wire_qkv_rope(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    ops: &ProjectionOps,
    pos_table: u32,
) -> Result<(u32, u32, u32)> {
    let (q_slot, k_slot, v_slot) = wire_qkv(sink, block, prev, ops)?;
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
    check_block(block, prev_out, params)?;
    let next = push_block(sink, prev, block, params)?;
    let hidden = block.hidden();
    Ok((next, hidden, hidden, LayerKind::Projection))
}
