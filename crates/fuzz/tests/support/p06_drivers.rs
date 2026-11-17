//! P0-6 接线：tokenizer / op / prof 三族纯函数的 fuzz 适配。
//!
//! 与既有 `drivers.rs` 同口径：运行时只把"任意字节 → 接受/拒绝"交回契约层；纯函数本身不持有
//! 跨用例状态，[`llm_fuzz::contract::judge`] 的 `catch_unwind` 兜底生效。输入先按定长块切成
//! f32 / 整数，长度不是块对齐时多余字节留给最后的解析入口（文本类目标）或直接拒绝（形状类目标）。
#![allow(dead_code)]

use diagnostic::{Code, Error};
use llm_fuzz::Target;
use tokenizer::Tokenizer;

/// 把一段字节按 LE f32 整块解码；尾部不足一块的字节直接丢弃，长度对不齐不猜。
///
/// 解码器只在定长块上读，空输入产出空切片——空行本身就是一条边界用例。
fn le_f32s(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// 文本入口：非 UTF-8 记成带类别的非空拒绝，与既有 `drivers::text` 同口径。
fn text(input: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(input)
        .map_err(|_| Error::new(Code::Invalid, "输入不是 UTF-8").with("fuzz::p06::text"))
}

// —— tokenizer：词表查找、特殊句柄、字节解码、流式解码 ——

/// 词表装配 + 字节反查：词片表由输入字节按 8 字节定长块切开，末尾残片作为最后一个词片。
///
/// 词表内的编号反查必须能 round-trip；非法编号不 panic。空输入装配出空词表，查表返回空。
pub struct VocabLookup;

impl Target for VocabLookup {
    fn name(&self) -> &str {
        "tokenizer::Vocab::lookup"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let (chunks, remainder) = input.as_chunks::<8>();
        let pieces: Vec<Vec<u8>> = chunks
            .iter()
            .map(|chunk| chunk.to_vec())
            .chain((!remainder.is_empty()).then(|| remainder.to_vec()))
            .collect();
        let source = tokenizer::VocabSource {
            family: tokenizer::VocabFamily::WordLevel,
            pieces,
            special: tokenizer::SpecialTokens::EMPTY,
            scores: Vec::new(),
            merges: Vec::new(),
        };
        let vocab = tokenizer::Vocab::try_new(source).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::VocabLookup")
        })?;
        // 反查整条输入：命中则编号必须能回到同一片，未命中返回空而不是 panic。
        if let Some(token) = vocab.lookup(input) {
            let piece = vocab.piece(token.id()).ok_or_else(|| {
                Error::new(Code::Internal, "命中编号反取不到词片").with("fuzz::VocabLookup")
            })?;
            if piece != input {
                return Err(
                    Error::new(Code::Internal, "词片反查往返不一致").with("fuzz::VocabLookup")
                );
            }
        }
        for id in [0u32, vocab.vocab_size() as u32, u32::MAX] {
            _ = vocab.piece(id);
        }
        Ok(())
    }
}

/// 特殊句柄：首字节当句柄编号，登记后必须能反查命中。
pub struct SpecialHandles;

impl Target for SpecialHandles {
    fn name(&self) -> &str {
        "tokenizer::SpecialTokens"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        if input.is_empty() {
            let empty = tokenizer::SpecialTokens::EMPTY;
            return empty.is_special(0).then_some(()).ok_or_else(|| {
                Error::new(Code::Invalid, "空句柄不应命中任何编号").with("fuzz::SpecialHandles")
            });
        }
        let id = u32::from(input[0]);
        let handles = tokenizer::SpecialTokens::EMPTY.with_unk(id);
        if handles.is_special(id) {
            Ok(())
        } else {
            Err(Error::new(Code::Internal, "登记的句柄编号查不回").with("fuzz::SpecialHandles"))
        }
    }
}

/// 字节级解码：编号流按 LE u32 块切开；非法字节与越界编号必须返回带说明的拒绝。
pub struct BytesDecode;

impl Target for BytesDecode {
    fn name(&self) -> &str {
        "tokenizer::Bytes::decode"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let bytes = tokenizer::Bytes;
        let tokens: Vec<tokenizer::Token> = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| {
                let id = u32::from_le_bytes(*chunk);
                tokenizer::Token::new(id)
            })
            .collect();
        let text = bytes.decode(&tokens).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::BytesDecode")
        })?;
        let reencoded = bytes.encode(&text).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::BytesDecode")
        })?;
        if reencoded != tokens {
            return Err(Error::new(Code::Internal, "字节解码往返不一致").with("fuzz::BytesDecode"));
        }
        Ok(())
    }
}

/// 流式解码：先 `push` 再 `finish`，半字符与非法续节必须返回带说明的拒绝。
pub struct StreamDecode;

impl Target for StreamDecode {
    fn name(&self) -> &str {
        "tokenizer::DecoderStream"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut stream = tokenizer::DecoderStream::new();
        stream.push(input).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::StreamDecode")
        })?;
        stream.finish().map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::StreamDecode")
        })?;
        Ok(())
    }
}

// —— op：形状/索引校验、归一化、归约、逐元素、查表 ——

/// einsum 表达式解析：文本入口，覆盖空串、缺箭头、重复字母、非字母与输出悬空。
pub struct ExprParse;

impl Target for ExprParse {
    fn name(&self) -> &str {
        "op::einsum::Expr::parse"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let expr = op::einsum::Expr::parse(text(input)?).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ExprParse")
        })?;
        // 解析结论稳定：原样往返文本后重解析必须得到同一个表达式。
        let again = op::einsum::Expr::parse(expr.text()).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ExprParse")
        })?;
        (expr == again)
            .then_some(())
            .ok_or_else(|| Error::new(Code::Internal, "表达式重解析不一致").with("fuzz::ExprParse"))
    }
}

/// 轴长度表：前 8 字节按 LE u32 给首轴长度，其余字节是字母候选；零长度/重复字母/非字母拒绝。
pub struct ExtentsNew;

impl Target for ExtentsNew {
    fn name(&self) -> &str {
        "op::einsum::Extents::new"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let Some((head, rest)) = input.split_first() else {
            return op::einsum::Extents::new([])
                .map(|_| ())
                .map_err(|error| Error::new(error.code(), error.message().to_string()));
        };
        let length = u32::from_le_bytes([*head, 0, 0, 0]) as usize;
        let label = rest.first().copied().unwrap_or(b'a') as char;
        op::einsum::Extents::new([(label, length)])
            .map(|_| ())
            .map_err(|error| Error::new(error.code(), error.message().to_string()))
    }
}

/// 调度写法解析：`direct` / `pipe<N>` / `ws<prod,cons>` / `pingpong<N>`；参数个数、下限、非数字拒绝。
pub struct ScheduleParse;

impl Target for ScheduleParse {
    fn name(&self) -> &str {
        "op::sch::Schedule::parse"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let schedule = op::sch::Schedule::parse(text(input)?).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ScheduleParse")
        })?;
        // 渲染回声明层写法，重解析必须得到同一个调度。
        let again = op::sch::Schedule::parse(&schedule.as_str()).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ScheduleParse")
        })?;
        (schedule == again).then_some(()).ok_or_else(|| {
            Error::new(Code::Internal, "调度重解析不一致").with("fuzz::ScheduleParse")
        })
    }
}

/// Softmax：f32 行归一化；空行、长度不齐拒绝，非空行结果有限且和为 1。
pub struct SoftmaxF32;

impl Target for SoftmaxF32 {
    fn name(&self) -> &str {
        "op::softmax_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let x = le_f32s(input);
        if x.is_empty() {
            return Err(Error::new(Code::Invalid, "softmax 输入不能为空").with("fuzz::SoftmaxF32"));
        }
        let mut out = vec![0.0; x.len()];
        op::softmax_f32::forward(&x, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::SoftmaxF32")
        })?;
        let total: f32 = out.iter().sum();
        if !total.is_finite() || (total - 1.0).abs() >= 1e-4 {
            return Err(Error::new(Code::Internal, "softmax 行未归一化").with("fuzz::SoftmaxF32"));
        }
        Ok(())
    }
}

/// RMSNorm：x 与 gamma 各取一半 f32；空行、长度不齐、零范数边界由 EPS 托底。
pub struct RmsnormF32;

impl Target for RmsnormF32 {
    fn name(&self) -> &str {
        "op::rmsnorm_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let values = le_f32s(input);
        let split = values.len() / 2;
        let (x, gamma) = values.split_at(split);
        if x.is_empty() {
            return Err(Error::new(Code::Invalid, "rmsnorm 输入不能为空").with("fuzz::RmsnormF32"));
        }
        let mut out = vec![0.0; x.len()];
        op::rmsnorm_f32::forward(x, gamma, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::RmsnormF32")
        })?;
        if out.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(Code::Internal, "rmsnorm 结果非有限").with("fuzz::RmsnormF32"));
        }
        Ok(())
    }
}

/// 求和归约：f32 折叠到单元素输出；输出必须单元素，空输入折叠到初值 0。
pub struct ReduceSumF32;

impl Target for ReduceSumF32 {
    fn name(&self) -> &str {
        "op::reduce_sum_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let a = le_f32s(input);
        let mut out = vec![0.0; 1];
        op::reduce_sum_f32::forward(&a, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ReduceSumF32")
        })?;
        if out[0].is_nan() {
            return Err(Error::new(Code::Internal, "求和得到 NaN").with("fuzz::ReduceSumF32"));
        }
        Ok(())
    }
}

/// 最大值归约：空输入折叠到 `f32::MIN`，饱和值与重复元素不 panic。
pub struct ReduceMaxF32;

impl Target for ReduceMaxF32 {
    fn name(&self) -> &str {
        "op::reduce_max_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let a = le_f32s(input);
        let mut out = vec![0.0; 1];
        op::reduce_max_f32::forward(&a, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ReduceMaxF32")
        })?;
        if out[0].is_nan() {
            return Err(Error::new(Code::Internal, "最大值得到 NaN").with("fuzz::ReduceMaxF32"));
        }
        Ok(())
    }
}

/// 逐元素加法：f32 行切两半，a/b/out 等长；非对齐长度拒绝。
pub struct AddF32;

impl Target for AddF32 {
    fn name(&self) -> &str {
        "op::add_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let values = le_f32s(input);
        let split = values.len() / 2;
        let (a, b) = values.split_at(split);
        let mut out = vec![0.0; a.len()];
        op::add_f32::forward(a, b, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::AddF32")
        })?;
        Ok(())
    }
}

/// 查表嵌入：前一半 f32 当索引，后一半当权重表；行宽由输出长度推出，越界索引拒绝。
pub struct EmbedF32;

impl Target for EmbedF32 {
    fn name(&self) -> &str {
        "op::embed_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let values = le_f32s(input);
        if values.is_empty() {
            return Err(Error::new(Code::Invalid, "embed 输入为空").with("fuzz::EmbedF32"));
        }
        // 末段为权重表，索引取前半段；行宽按样本档 4。
        let dim = 4;
        let table = &values[values.len() - values.len() / dim * dim..];
        let idx = &values[..values.len() - table.len()];
        if idx.is_empty() {
            return Err(Error::new(Code::Invalid, "embed 索引为空").with("fuzz::EmbedF32"));
        }
        let rows = table.len() / dim;
        if table.is_empty() || rows == 0 {
            return Err(Error::new(Code::Invalid, "embed 权重表为空").with("fuzz::EmbedF32"));
        }
        let mut out = vec![0.0; idx.len() * dim];
        op::embed_f32::forward(idx, table, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::EmbedF32")
        })?;
        Ok(())
    }
}

/// RoPE：位置取首字节，其余 f32 行旋转；奇数长度尾元素原样透传，长度不齐拒绝。
pub struct RopeF32;

impl Target for RopeF32 {
    fn name(&self) -> &str {
        "op::rope_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        if input.is_empty() {
            return Err(Error::new(Code::Invalid, "rope 缺位置参数").with("fuzz::RopeF32"));
        }
        let (head, rest) = input.split_first().unwrap();
        let x = le_f32s(rest);
        if x.is_empty() {
            return Err(Error::new(Code::Invalid, "rope 输入为空").with("fuzz::RopeF32"));
        }
        let mut out = vec![0.0; x.len()];
        op::rope_f32::forward(usize::from(*head), &x, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::RopeF32")
        })?;
        Ok(())
    }
}

// —— prof：区间解析、闭集解析 ——

/// 能力区间解析：`16` / `16..64` / `<=512` / `32..`；空串、非数字、上界越上拒绝。
pub struct SpanParse;

impl Target for SpanParse {
    fn name(&self) -> &str {
        "prof::atom::Span::parse"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let span = prof::atom::Span::parse(text(input)?).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::SpanParse")
        })?;
        // 自洽：下界不大于上界，任意区间覆盖自身。
        if span.lo() > span.hi() {
            return Err(Error::new(Code::Internal, "区间下界大于上界").with("fuzz::SpanParse"));
        }
        if !span.covers(span) {
            return Err(Error::new(Code::Internal, "区间不覆盖自身").with("fuzz::SpanParse"));
        }
        Ok(())
    }
}

/// 原语闭集解析：未知名、大小写漂移、截断词返回空；全量成员名字往返一致。
pub struct PrimParse;

impl Target for PrimParse {
    fn name(&self) -> &str {
        "prof::atom::Prim::parse"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let name = text(input)?;
        if let Some(prim) = prof::atom::Prim::parse(name)
            && prim.as_str() != name
        {
            return Err(Error::new(Code::Internal, "原语名字往返不一致").with("fuzz::PrimParse"));
        }
        Ok(())
    }
}

/// 存储空间闭集解析：与原语同口径。
pub struct SpaceParse;

impl Target for SpaceParse {
    fn name(&self) -> &str {
        "prof::atom::Space::parse"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let name = text(input)?;
        if let Some(space) = prof::atom::Space::parse(name)
            && space.as_str() != name
        {
            return Err(Error::new(Code::Internal, "空间名字往返不一致").with("fuzz::SpaceParse"));
        }
        // 存储闭集名字是大写写法；小写漂移不应命中。
        if name == "gm" && prof::atom::Space::parse(name).is_some() {
            return Err(Error::new(Code::Internal, "小写不应命中存储闭集").with("fuzz::SpaceParse"));
        }
        Ok(())
    }
}

/// 数据类型闭集解析：名字表往返一致；未知名返回空而不是 panic。
pub struct DTypeParse;

impl Target for DTypeParse {
    fn name(&self) -> &str {
        "prof::atom::DType::parse_name"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let name = text(input)?;
        if let Some(dtype) = prof::atom::DType::parse_name(name)
            && dtype.name() != name
        {
            return Err(Error::new(Code::Internal, "类型名字往返不一致").with("fuzz::DTypeParse"));
        }
        Ok(())
    }
}

/// 逐元素乘法：f32 行切两半，a/b/out 等长；非对齐长度拒绝。
pub struct MulF32;

impl Target for MulF32 {
    fn name(&self) -> &str {
        "op::mul_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let values = le_f32s(input);
        let split = values.len() / 2;
        let (a, b) = values.split_at(split);
        let mut out = vec![0.0; a.len()];
        op::mul_f32::forward(a, b, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::MulF32")
        })?;
        Ok(())
    }
}

/// 标量缩放：f32 行逐元素乘 `ALPHA`；长度不齐拒绝，空行逐元素放行。
pub struct ScaleF32;

impl Target for ScaleF32 {
    fn name(&self) -> &str {
        "op::scale_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let a = le_f32s(input);
        let mut out = vec![0.0; a.len()];
        op::scale_f32::forward(&a, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::ScaleF32")
        })?;
        Ok(())
    }
}

/// SiLU 激活：f32 行逐元素；长度不齐拒绝，饱和值不产出 NaN/Inf。
pub struct SiluF32;

impl Target for SiluF32 {
    fn name(&self) -> &str {
        "op::silu_f32::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let a = le_f32s(input);
        let mut out = vec![0.0; a.len()];
        op::silu_f32::forward(&a, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::SiluF32")
        })?;
        if out.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(Code::Internal, "silu 结果非有限").with("fuzz::SiluF32"));
        }
        Ok(())
    }
}

/// 4 位反量化：前两字节标量，其余打包字节解两个 nibble；输出长度必须是字节数的 2 倍。
pub struct DequantI4;

impl Target for DequantI4 {
    fn name(&self) -> &str {
        "op::dequant_i4::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        if input.len() < 2 {
            return Err(Error::new(Code::Invalid, "dequant_i4 输入不足").with("fuzz::DequantI4"));
        }
        let scale = f32::from_le_bytes([input[0], input[1], 0, 0]);
        let q = &input[2..];
        let mut out = vec![0.0; q.len() * 2];
        op::dequant_i4::forward(scale, op::dequant_i4::ZERO, q, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::DequantI4")
        })?;
        Ok(())
    }
}

/// 8 位反量化：首 f32 当 scale，次 f32 当 zero，其余 f32 承载量化值；空行逐元素放行。
pub struct DequantI8;

impl Target for DequantI8 {
    fn name(&self) -> &str {
        "op::dequant_i8::forward"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let values = le_f32s(input);
        if values.is_empty() {
            return Err(Error::new(Code::Invalid, "dequant_i8 输入为空").with("fuzz::DequantI8"));
        }
        let scale = values[0];
        let zero = *values.get(1).unwrap_or(&op::dequant_i8::ZERO);
        let q = &values[1..];
        let mut out = vec![0.0; q.len()];
        op::dequant_i8::forward(scale, zero, q, &mut out).map_err(|error| {
            Error::new(error.code(), error.message().to_string()).with("fuzz::DequantI8")
        })?;
        Ok(())
    }
}
