//! 私有图拓扑 IR：只描述本引擎自有算子集，不做通用 IR。
//!
//! 闭集算子避免了 ONNX 式兼容全部框架的算子地狱。图是单文件内的紧凑二进制块，以保留张量名
//! `GRAPH_NAME` 存放，复用容器块表/CRC，可静态读取做融合提示。
//!
//! 相比旧版的三处增强：
//! 1. 整数走 [`crate::leb`] 的 LEB128 而非固定 `u32`，小图更紧凑，且只有一种字节形态；
//! 2. 每条输入边必须指向**已存在**的更早节点——构造与解码都强制，于是引用合法、无自引用、
//!    无环（边只能从后向前指）一次性成立，不需要额外的环检测；
//! 3. 输出名非空、输入边数不超过 `MAX_IN`。

use diagnostic::{Code, Error, Result};

use crate::dtype::Wire;
use crate::leb::{self, Cursor};
use crate::seg::Segment;
use crate::sink::Sink;

/// 图块在容器内的保留张量名。
pub const GRAPH_NAME: &str = "llm.graph";

/// 单节点输入边上限。
const MAX_IN: usize = 8;

/// 闭集算子：仅引擎自有，码位稳定，不随外部框架扩张。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// 矩阵乘。
    Gemm,
    /// RMS 归一化。
    Rms,
    /// Adam 更新。
    Adam,
    /// 扫描主体（SSM scan）。
    Scan,
    /// 扫描定长修正。
    ScanFix,
    /// 扫描点积。
    ScanDot,
    /// 扫描发射。
    ScanEmit,
    /// 注意力。
    Attention,
}

/// 闭集算子的码位顺序：[`Op::code`] 给下标，[`Op::read`] 收回下标。
///
/// 码位表只此一份：两边都读它，“写出来的码能读回来”才是结构性事实而不是测试巧合。
const CODES: [Op; 8] = [
    Op::Gemm,
    Op::Rms,
    Op::Adam,
    Op::Scan,
    Op::ScanFix,
    Op::ScanDot,
    Op::ScanEmit,
    Op::Attention,
];

impl Op {
    /// 稳定数字码位：线上格式只认数字，不认变体名。
    fn code(self) -> u8 {
        CODES.iter().position(|&op| op == self).unwrap_or(0) as u8
    }

    /// 码位 → 算子；闭集之外显式报错。
    fn read(code: u8) -> Result<Self> {
        CODES.get(code as usize).copied().ok_or_else(|| {
            Error::new(Code::Invalid, format!("ir: 未知算子码 {code}")).with("ir::Op")
        })
    }
}

/// 图节点：闭集算子 + 输入边（节点 id）+ 输出张量名 + 融合提示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    op: Op,
    inputs: Vec<u32>,
    out: String,
    fused: bool,
}

impl Node {
    /// 节点算子。
    #[must_use]
    pub fn op(&self) -> Op {
        self.op
    }

    /// 输入边（节点 id），按声明顺序。
    #[must_use]
    pub fn inputs(&self) -> &[u32] {
        &self.inputs
    }

    /// 输出张量名。
    #[must_use]
    pub fn out(&self) -> &str {
        &self.out
    }

    /// 是否允许与相邻节点合并发射。
    #[must_use]
    pub fn fused(&self) -> bool {
        self.fused
    }
}

/// 私有拓扑：节点按下标 id 引用，边只能指向更早节点。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// 空图。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 节点数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否空图。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 全部节点，下标即节点 id。
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 按 id 取节点。
    #[must_use]
    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.get(id as usize)
    }

    /// 可融合节点数：静态融合的候选规模。
    #[must_use]
    pub fn fusible(&self) -> usize {
        self.nodes.iter().filter(|node| node.fused).count()
    }

    /// 追加节点，返回其 id。
    ///
    /// 校验：输出名非空、输入边数不超过 `MAX_IN`、每条输入边指向已存在节点（id 小于当前
    /// 节点数）。最后一条同时杜绝自引用与环——新 id 还不在合法范围内，且边只能指向更早节点。
    pub fn add(&mut self, op: Op, inputs: &[u32], out: &str, fused: bool) -> Result<u32> {
        if out.is_empty() {
            return Err(bad("ir: 输出名不能为空"));
        }
        let id = self.nodes.len() as u32;
        check_edges(inputs.len())?;
        check_into(inputs, id)?;
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            out: out.to_string(),
            fused,
        });
        Ok(id)
    }

    /// 序列化为 LEB128 紧凑字节，写入任意 Sink（算长度与写出同路径）。
    pub(crate) fn encode(&self, out: &mut impl Sink) {
        leb::write_u32(out, self.nodes.len() as u32);
        for node in &self.nodes {
            out.put(&[node.op.code()]);
            leb::write_u32(out, node.inputs.len() as u32);
            for input in &node.inputs {
                leb::write_u32(out, *input);
            }
            out.put(&[u8::from(node.fused)]);
            let name = node.out.as_bytes();
            leb::write_u64(out, name.len() as u64);
            out.put(name);
        }
    }

    /// 序列化为 `Vec<u8>`。
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// 把图编码成可写进容器的段：保留名 `GRAPH_NAME`、`U8` 单轴、原始字节不量化。
    ///
    /// 形状取字节数，因此段自洽校验天然成立；经 `Writer::push` 落盘后，读端可再用
    /// [`Graph::decode`] 还原。
    pub fn to_segment(&self) -> Result<Segment> {
        let data = self.to_bytes();
        Segment::new(GRAPH_NAME, Wire::U8, vec![data.len() as u64], None, data)
    }

    /// 从字节解析：未知码、截断、非法 UTF-8 与悬空/自指边都显式报错。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);
        let count = cursor.read_u32()?;
        let mut graph = Self::new();
        for _ in 0..count {
            let node = decode_node(&mut cursor)?;
            check_into(&node.inputs, graph.nodes.len() as u32)?;
            graph.nodes.push(node);
        }
        Ok(graph)
    }
}

/// 构造一个"输入不合法"错误。
fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("ir")
}

/// 单节点的输入边上限：`MAX_IN` 是声明预算，构造与解码两个入口共用同一句文案。
fn check_edges(count: usize) -> Result<()> {
    if count <= MAX_IN {
        return Ok(());
    }
    Err(bad(&format!("ir: 输入边 {count} 条超过上限 {MAX_IN}")))
}

/// 每条输入边必须指向已存在的更早节点：新节点的 id 恰好是当前节点数，
/// 所以这条判据同时挡掉自引用与前向边（不需要额外的环检测）。
fn check_into(inputs: &[u32], limit: u32) -> Result<()> {
    for input in inputs {
        if *input >= limit {
            return Err(bad(&format!(
                "ir: 输入边指向不存在的节点 {input}（当前只有 {limit} 个节点）"
            )));
        }
    }
    Ok(())
}

/// 一条节点记录：算子码与输入边先读，再是融合位与输出名。
fn decode_node(cursor: &mut Cursor<'_>) -> Result<Node> {
    let (op, inputs) = edge_list(cursor)?;
    let fused = cursor.read_u8()? == 1;
    let out = label(cursor)?;
    Ok(Node {
        op,
        inputs,
        out,
        fused,
    })
}

/// 节点开头：算子码与输入边（边数先过上限，再逐条读，不为非法条数分配）。
fn edge_list(cursor: &mut Cursor<'_>) -> Result<(Op, Vec<u32>)> {
    let op = op_at(cursor)?;
    let n_in = cursor.read_u32()? as usize;
    check_edges(n_in)?;
    Ok((op, edges(cursor, n_in)?))
}

/// 算子码：先读一个字节再解闭集，两段都可能失败，所以合成一个入口。
fn op_at(cursor: &mut Cursor<'_>) -> Result<Op> {
    Op::read(cursor.read_u8()?)
}

/// 输入边列表；条数已由调用方校验，这里只逐条读。
fn edges(cursor: &mut Cursor<'_>, count: usize) -> Result<Vec<u32>> {
    let mut edges = Vec::with_capacity(count);
    for _ in 0..count {
        edges.push(cursor.read_u32()?);
    }
    Ok(edges)
}

/// 输出名：LEB128 长度前缀 + 原始字节，必须是合法 UTF-8。
fn label(cursor: &mut Cursor<'_>) -> Result<String> {
    let len = cursor.read_u64()? as usize;
    let raw = cursor.take(len)?;
    core::str::from_utf8(raw)
        .map_err(|_| bad("ir: 输出名不是合法 UTF-8"))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_get_sequential_ids_and_earlier_edges() {
        let mut graph = Graph::new();
        let a = graph.add(Op::Gemm, &[], "a", true).expect("首节点");
        let b = graph.add(Op::Rms, &[a], "b", true).expect("引用首节点");
        assert_eq!([a, b], [0, 1]);
        assert_eq!(graph.fusible(), 2);
        assert_eq!(graph.node(0).unwrap().op(), Op::Gemm);
    }

    #[test]
    fn dangling_self_and_forward_edges_are_rejected() {
        let mut graph = Graph::new();
        // 指向不存在的前向节点（也覆盖自引用，因为新 id 就是 0）。
        assert!(graph.add(Op::Gemm, &[0], "x", false).is_err());
        graph.add(Op::Gemm, &[], "x", false).expect("先建 0");
        // 第二个节点引用 id 2（不存在）。
        assert!(graph.add(Op::Rms, &[2], "y", false).is_err());
        // 空名与超量边。
        assert!(graph.add(Op::Rms, &[], "", false).is_err());
        let many = [0u32; 9];
        assert!(graph.add(Op::Rms, &many, "z", false).is_err());
    }

    #[test]
    fn a_graph_round_trips_byte_for_byte() {
        let mut graph = Graph::new();
        let a = graph.add(Op::Gemm, &[], "blk.0.attn_q", true).expect("a");
        graph.add(Op::Scan, &[a], "blk.0.scan", true).expect("b");
        let bytes = graph.to_bytes();
        let back = Graph::decode(&bytes).expect("解码合法");
        assert_eq!(back, graph);
    }

    #[test]
    fn graph_round_trips_through_the_container() {
        let mut graph = Graph::new();
        let a = graph.add(Op::Gemm, &[], "a", true).expect("a");
        graph.add(Op::Scan, &[a], "b", true).expect("b");

        let mut writer = crate::writer::Writer::new();
        writer
            .push(graph.to_segment().expect("段"))
            .expect("入容器");
        let bytes = writer.finish(6).expect("落盘");

        let archive = crate::reader::Archive::parse(&bytes).expect("容器合法");
        let stored = archive.get(GRAPH_NAME).expect("图块存在");
        let back = Graph::decode(archive.read(stored)).expect("解码图");
        assert_eq!(back, graph);
    }

    #[test]
    fn truncated_and_unknown_op_bytes_are_errors() {
        let mut graph = Graph::new();
        graph.add(Op::Gemm, &[], "a", false).expect("a");
        let bytes = graph.to_bytes();
        assert!(Graph::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes.clone();
        bad[4] = 99; // 闭集之外的算子码（op 码位于节点起点）
        assert!(Graph::decode(&bad).is_err());
    }
}
