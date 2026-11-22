//! 驱动侧适配：把真实解析器与领域 oracle 接成 [`crate::Target`]。
//!
//! 这份定义是**唯一出处**：确定性遍历（`tests/parsers.rs` 的 [`crate::sweep`]）与 libfuzzer
//! 模糊入口（`src/fuzz_*.rs`）都从这里取同一组入口。两边不各写一份判据，避免文本/字节边界的
//! 处理在两处悄悄漂移。
//!
//! 前半段是解析器契约：任意字节 → 接受/拒绝，解析器本身在 `cli` / `weights` / `storage` /
//! `serve::proto` / `wire`，不变量本身在 `wire`。后半段是**语义 oracle**：CRC 分块续算、
//! LEB 规范往返、形状算术、能力协商产物、分布与 Philox 不变量——它们同样吃任意字节，但把数学
//! 与协议承诺做成断言，这是纯"不 panic"覆盖不到的一层深度。
//!
//! 每个 harness 入口只取其中一个适配，未被该入口引用的项是有意保留的整组定义。

use std::sync::OnceLock;

use cutile::builtin::Catalogue;
use diagnostic::{Code, Error};
use wire::sink::Sink;

use crate::Target;

/// 文本解析器的公共外壳：非 UTF-8 不是"解析失败"，而是"这份输入不适用于文本入口"。
///
/// 仍然是一个合法的拒绝（带类别的非空消息），因此不违反 [`Target`] 三契约；把它区分出来，
/// 是为了之后加字节入口时不必再想一遍。
pub fn text(input: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(input)
        .map_err(|_| Error::new(Code::Invalid, "输入不是 UTF-8").with("fuzz::drivers"))
}

/// 命令行：空白切词，再交给唯一的文法。
pub struct Invocation;

impl Target for Invocation {
    fn name(&self) -> &str {
        "cli::Invocation"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let words: Vec<String> = text(input)?
            .split_whitespace()
            .map(str::to_string)
            .collect();
        cli::Invocation::parse(&words).map(|_| ())
    }
}

/// 词法解析：文本入口，覆盖未闭合分组/注释/字符串与饱和标识符等边角。
pub struct Lex;

impl Target for Lex {
    fn name(&self) -> &str {
        "tools::lex::parse"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        tool::parse(text(input)?).map(|_| ())
    }
}

/// safetensors 头部里的 JSON。
pub struct Json;

impl Target for Json {
    fn name(&self) -> &str {
        "weights::json"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::json::parse(text(input)?).map(|_| ())
    }
}

/// safetensors 整体：8 字节头长 + JSON 头 + 数据段。
pub struct Safetensors;

impl Target for Safetensors {
    fn name(&self) -> &str {
        "weights::safetensors::Archive"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::safetensors::Archive::parse(input).map(|_| ())
    }
}

/// 自有 `.u` 容器 v2：描述符表 + 对齐数据段 + 两级 CRC 尾部。
///
/// 与 safetensors 是两条独立的布局路径：段名唯一性、区间不重叠、两级校验和都在这条入口上
/// 暴露给任意字节。
pub struct UContainer;

impl Target for UContainer {
    fn name(&self) -> &str {
        "weights::Archive"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::Archive::parse(input).map(|_| ())
    }
}

/// 训练检查点：魔数 + 版本 + 事实 + 条目表 + CRC-32 尾部。
///
/// `decode` 内部先跑 `verify`，所以这一条同时覆盖"校验和在错误路径上不 panic"与
/// "正文解析只在校验通过后才开始"两件事。
pub struct Snapshot;

impl Target for Snapshot {
    fn name(&self) -> &str {
        "storage::Snapshot::decode"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        weights::Snapshot::decode(input).map(|_| ())
    }
}

/// 服务端帧解码：任意字节流的"能切出几帧、剩下的算不算坏"。
///
/// 逐帧取尽再 `finish`：只调 `push` 会漏掉"半帧断开"那条路径，而它正是协议最容易犯的
/// 安静错误（把半个请求当成完整的）。
pub struct Frames;

impl Target for Frames {
    fn name(&self) -> &str {
        "serve::proto::Decoder"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut decoder = serve::proto::Decoder::new();
        decoder.push(input)?;
        while decoder.next_frame()?.is_some() {}
        decoder.finish()
    }
}

/// HTTP/1.1 P0 请求头：任意字节只能得到稳定拒绝或已解析的请求行和长度。
pub struct HttpHead;

impl Target for HttpHead {
    fn name(&self) -> &str {
        "serve::http::head"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        serve::fuzz_http_head(input)
    }
}

/// LEB128 变长整数游标：一段字节里读一个规范无符号整数。
///
/// 非规范编码（前导零字节、超 `u64` 容量、第 10 字节越界）与字节不足都返回错误；游标本身
/// 不要求消费完整段，尾部多余字节留给流式上层判断，这里只验"读一个数不 panic、结论稳定"。
pub struct Leb;

impl Target for Leb {
    fn name(&self) -> &str {
        "wire::leb::Cursor"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut cursor = wire::leb::Cursor::new(input);
        let _: u64 = cursor.read()?;
        Ok(())
    }
}

/// 线格式的密封与计量：同一批字节经三种 `Sink` 必须得到互相一致的结论。
///
/// 这一条不拒绝任何输入（`Sink` 没有"格式错"这种状态），它验的是**不变量**：密封算出的
/// 校验和等于 `crc32`、计量长度等于实际写出长度、切片接收器在容量够时置位标志且不越界。
/// 三条里任何一条在某个字节位置翻转，都会让下游的长度承诺与校验承诺对不上。
pub struct Sealed;

impl Target for Sealed {
    fn name(&self) -> &str {
        "wire::sink::Sealed"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut written: Vec<u8> = Vec::new();
        let checksum = {
            let mut sealed = wire::sink::Sealed::new(&mut written);
            sealed.put(input);
            sealed.checksum()
        };
        let mut counted = wire::sink::Count::default();
        counted.put(input);
        let mut slice = vec![0u8; input.len()];
        let mut buf = wire::sink::SliceSink::new(&mut slice);
        buf.put(input);
        let inconsistent = [
            checksum != wire::crc::crc32(input),
            counted.0 != input.len(),
            written.as_slice() != input,
            buf.overflow(),
            buf.written() != input.len(),
        ]
        .contains(&true);
        if inconsistent {
            return Err(
                Error::new(Code::Internal, "三种接收器对同一批字节给出了不一致的结论")
                    .with("fuzz::drivers::Sealed"),
            );
        }
        Ok(())
    }
}

/// 不变量被破坏：测试侧 oracle 统一用这一条出口，契约层仍按「拒绝要有话说」处理。
fn inconsistent(what: &str) -> Error {
    Error::new(Code::Internal, format!("不变量被破坏：{what}")).with("fuzz::drivers")
}

/// CRC-32 的分块等价与逐位参考对拍。
///
/// `update` 承诺「跨块状态可续」，`crc32` 承诺「一次算完等于分段算」。查表下标、初值与
/// 终值取反任一处写错，长文的分段校验就会和整段校验分叉——而这正是容器完整性尾部的依据。
pub struct CrcChunk;

impl Target for CrcChunk {
    fn name(&self) -> &str {
        "wire::crc"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let whole = wire::crc::crc32(input);
        if whole != naive_crc32(input) {
            return Err(inconsistent("crc32 与逐位参考实现不一致"));
        }
        // 切点由长度与首字节共同决定：覆盖中点与任意位置，避免只测一种分段。
        let mid = input.len() / 2;
        let head = input
            .first()
            .map_or(0, |byte| usize::from(*byte) % (input.len() + 1));
        for split in [mid, head] {
            let (left, right) = input.split_at(split);
            let chained = !wire::crc::update(wire::crc::update(u32::MAX, left), right);
            if chained != whole {
                return Err(inconsistent("crc32 分段续算与整段结果不一致"));
            }
        }
        Ok(())
    }
}

/// 逐位 CRC-32（IEEE 802.3）：测试侧的独立参考，不进生产库。
fn naive_crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// LEB128 的规范往返：解码成功的前缀必须能被重新编码成同一串字节。
///
/// 任意 `u64` 写出再读回必须恒等且取最短规范长度；反过来，一段原始字节只要能按规范解出
/// 一个值，重编码就必须逐字复现被消费的前缀——解码器拒绝非规范形，所以这条等价于「解码
/// 与编码互逆」，任何一处宽度/移位写错都会在此暴露。
pub struct LebRoundtrip;

impl Target for LebRoundtrip {
    fn name(&self) -> &str {
        "wire::leb::roundtrip"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let mut word = [0u8; 8];
        let n = input.len().min(8);
        word[..n].copy_from_slice(&input[..n]);
        let value = u64::from_le_bytes(word);
        let mut canonical = Vec::new();
        wire::leb::write(&mut canonical, value);
        if canonical.len() != leb_len(value) {
            return Err(inconsistent("LEB128 写出的不是最短规范长度"));
        }
        let mut cursor = wire::leb::Cursor::new(&canonical);
        let decoded: u64 = cursor.read()?;
        if decoded != value || !cursor.remaining().is_empty() {
            return Err(inconsistent("LEB128 写→读不恒等或未读完"));
        }
        canonical_prefix_matches(input)
    }
}

/// 规范解码后的重编码必须逐字复现被消费的前缀；解不出数就无事可验。
fn canonical_prefix_matches(input: &[u8]) -> Result<(), Error> {
    let mut raw = wire::leb::Cursor::new(input);
    let Ok(prefix) = raw.read::<u64>() else {
        return Ok(());
    };
    let consumed = &input[..raw.pos()];
    let mut again = Vec::new();
    wire::leb::write(&mut again, prefix);
    if again.as_slice() != consumed {
        return Err(inconsistent("LEB128 规范解码后的重编码与原前缀不一致"));
    }
    Ok(())
}

/// 规范 LEB128 的最短字节数：`0` 占 1 字节，其余按 7 位一组向上取整。
fn leb_len(value: u64) -> usize {
    if value == 0 {
        1
    } else {
        (64 - value.leading_zeros() as usize).div_ceil(7)
    }
}

/// 形状算术：`numel` 的饱和乘积与 `element_bytes` 的位级取整都对独立参考。
///
/// 轴长可能来自不受信任的权重文件头，乘积溢出必须夹到上限而不是回绕；亚字节类型
/// （`U1`/`U4`/`F4`）又要求字节数按位向上取整。两条都在这里与一份 `u128` 参考比对。
pub struct ShapeArith;

impl Target for ShapeArith {
    fn name(&self) -> &str {
        "tensor::Shape"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let dims = dims_of(input);
        let shape = tensor::Shape::new(dims.clone());
        let expected = product_of(&dims);
        if shape.numel() != expected {
            return Err(inconsistent("Shape::numel 与饱和乘积参考不一致"));
        }
        if shape.rank() != dims.len() {
            return Err(inconsistent("Shape::rank 与轴长个数不一致"));
        }
        let dtype = dtype_of(input);
        if shape.element_bytes(dtype) != bytes_of(expected, dtype) {
            return Err(inconsistent("Shape::element_bytes 与位级取整参考不一致"));
        }
        Ok(())
    }
}

/// 轴长：每两字节一个小端 `u16`，最多取 8 个轴；末位缺字节按 0 补。
fn dims_of(input: &[u8]) -> Vec<usize> {
    (0..input.len().min(16))
        .step_by(2)
        .map(|at| {
            let lo = input.get(at).copied().unwrap_or(0);
            let hi = input.get(at + 1).copied().unwrap_or(0);
            usize::from(u16::from_le_bytes([lo, hi]))
        })
        .collect()
}

/// 饱和乘积：用 `u128` 计算再夹到 `usize`，与生产侧 `usize::saturating_mul` 互为独立参考。
fn product_of(dims: &[usize]) -> usize {
    let product = dims
        .iter()
        .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128));
    usize::try_from(product).unwrap_or(usize::MAX)
}

/// 由输入末端挑一个元素类型；空输入取第一个（`Bool`）。
fn dtype_of(input: &[u8]) -> tensor::DType {
    let all = tensor::DType::ALL;
    let idx = input
        .last()
        .map_or(0, |byte| usize::from(*byte) % all.len());
    all[idx]
}

/// 位级取整参考：先按 `u64` 饱和乘位宽，再向上取整到字节。
fn bytes_of(numel: usize, dtype: tensor::DType) -> usize {
    let total = (numel as u64).saturating_mul(u64::from(dtype.bits()));
    usize::try_from(total.div_ceil(8)).unwrap_or(usize::MAX)
}

/// 内核协商的产物契约：命中的变体必须真实存在于算子的降级链里。
///
/// 能力不足返回 `Err` 是合法结果（多数档案缺某条原语）；真正要守的是「成功时给出的变体
/// 名与在链里的下标可回指到声明的 `variants`」——否则下游按名字取调度/布局会取空。
pub struct CutileNegotiate;

impl Target for CutileNegotiate {
    fn name(&self) -> &str {
        "cutile::cap::negotiate"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        // 内置声明/档案表编译期内嵌，进程内只解析一次：避免每迭代重扫，压低吞吐。
        static CATALOGUE: OnceLock<Option<Catalogue>> = OnceLock::new();
        let Some(catalogue) = CATALOGUE.get_or_init(|| Catalogue::try_builtin().ok()) else {
            return Err(Error::new(Code::Missing, "内置算子声明或档案表无法解析")
                .with("fuzz::drivers::CutileNegotiate"));
        };
        let ops = catalogue.ops();
        let profiles = catalogue.profiles();
        if ops.is_empty() || profiles.is_empty() {
            return Err(inconsistent("内置算子声明或档案表为空"));
        }
        let op = &ops[usize::from(input.first().copied().unwrap_or(0)) % ops.len()];
        let profile = &profiles[usize::from(input.get(1).copied().unwrap_or(0)) % profiles.len()];
        if let Ok(path) = cutile::cap::negotiate(op, profile) {
            let points_back = op
                .variants
                .get(path.idx)
                .is_some_and(|variant| variant.name == path.variant);
            if !points_back {
                return Err(inconsistent("协商命中的变体名/下标无法回指到算子的降级链"));
            }
        }
        Ok(())
    }
}

/// 分布闭集的语义不变量：参数非法必须拒绝，合法时样本落在承诺区间内，同一 `seed` 序列确定。
///
/// 与解析器的「不 panic」不同，这里验的是**数学承诺**：`Uniform` 的样本不越界、`Bernoulli`
/// 的端点概率、`Normal` 的有限性，以及 Philox 计数器流的位级确定性。
pub struct DistSampler;

impl Target for DistSampler {
    fn name(&self) -> &str {
        "rng::dist"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        let words = words_of(params_of(input));
        let seed = u64::from(words[0]) | (u64::from(words[1]) << 32);
        philox_is_deterministic(seed)?;
        uniform_in_range(seed, words)?;
        bernoulli_endpoints(seed, words)?;
        normal_is_finite(seed, words)
    }
}

/// 前 16 字节小端补零成参数块：不足的部分按 0 补，避免长度决定分支。
fn params_of(input: &[u8]) -> [u8; 16] {
    let mut params = [0u8; 16];
    let n = input.len().min(16);
    params[..n].copy_from_slice(&input[..n]);
    params
}

/// 参数块切四个小端 `u32`。
fn words_of(params: [u8; 16]) -> [u32; 4] {
    std::array::from_fn(|i| {
        u32::from_le_bytes([
            params[i * 4],
            params[i * 4 + 1],
            params[i * 4 + 2],
            params[i * 4 + 3],
        ])
    })
}

/// Philox 确定性：同一 seed 起两条流，`u32`/`f32` 逐位一致，且 `next_f32` 落在 [0,1)。
fn philox_is_deterministic(seed: u64) -> Result<(), Error> {
    let mut left = rng::Rng::new(seed);
    let mut right = rng::Rng::new(seed);
    for _ in 0..16 {
        if left.next_u32() != right.next_u32() {
            return Err(inconsistent("同 seed 的两条 Philox 流 u32 序列分叉"));
        }
        let (a, b) = (left.next_f32(), right.next_f32());
        if a.to_bits() != b.to_bits() {
            return Err(inconsistent("同 seed 的两条 Philox 流 f32 序列分叉"));
        }
        if !(0.0..1.0).contains(&a) {
            return Err(inconsistent("next_f32 越出 [0,1)"));
        }
    }
    Ok(())
}

/// 均匀分布：参数合法时样本必须落在闭区间内（浮点取整下上界按闭区间判）。
fn uniform_in_range(seed: u64, words: [u32; 4]) -> Result<(), Error> {
    let (lo, hi) = (f32::from_bits(words[1]), f32::from_bits(words[2]));
    let Ok(uniform) = rng::Uniform::new(lo, hi) else {
        return Ok(());
    };
    let span_finite = (hi - lo).is_finite();
    let mut rng = rng::Rng::new(seed);
    for _ in 0..64 {
        let sample = uniform.sample(&mut rng);
        if outside(span_finite, sample, lo, hi) {
            return Err(inconsistent("Uniform 样本越出 [lo,hi] 或非有限"));
        }
    }
    Ok(())
}

/// 样本是否越界或非有限（只在区间有限时成立）。
fn outside(span_finite: bool, sample: f32, lo: f32, hi: f32) -> bool {
    span_finite && (!sample.is_finite() || sample < lo || sample > hi)
}

/// 伯努利端点：p=0 恒 false、p=1 恒 true。
fn bernoulli_endpoints(seed: u64, words: [u32; 4]) -> Result<(), Error> {
    let p = f32::from_bits(words[3]);
    let Ok(bernoulli) = rng::Bernoulli::new(p) else {
        return Ok(());
    };
    let mut rng = rng::Rng::new(seed);
    for _ in 0..64 {
        if hits_wrong_endpoint(bernoulli.sample(&mut rng), p) {
            return Err(inconsistent("Bernoulli 端点概率抽到了反例"));
        }
    }
    Ok(())
}

/// p=0 抽到 true 或 p=1 抽到 false，都是端点承诺被破坏。
fn hits_wrong_endpoint(hit: bool, p: f32) -> bool {
    (p == 0.0 && hit) || (p == 1.0 && !hit)
}

/// 正态分布：均值/标准差有限时样本必须有限（Box-Muller 半径有界）。
fn normal_is_finite(seed: u64, words: [u32; 4]) -> Result<(), Error> {
    let mean = f32::from_bits(words[0]);
    let std = 1.0 + (words[1] & 0x3FF) as f32 * 0.001;
    let Ok(normal) = rng::Normal::new(mean, std) else {
        return Ok(());
    };
    let mut rng = rng::Rng::new(seed);
    for _ in 0..64 {
        if !normal.sample(&mut rng).is_finite() {
            return Err(inconsistent("Normal 样本非有限"));
        }
    }
    Ok(())
}
