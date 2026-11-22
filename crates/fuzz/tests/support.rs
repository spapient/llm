// fuzz 支持：驱动适配、libfuzzer harness 入口与合法种子（原 `support/` 下三份合并）。
//
// 消费方与用法：
// - `parsers.rs` / `gen_corpus.rs`：`#[path = "support.rs"] mod support;`，按子模块取用；
// - 9 个 `fuzz_*.rs`：`include!("support.rs")`，模块与 harness 入口都落在测试二进制根部。
//
// harness 段只在 `engine` feature 下编译：libFuzzer 运行时是可选依赖，默认 `cargo test`
// 也会把本文件编进 `parsers` / `gen_corpus`，这两处不需要也不应该拉入运行时。

pub mod drivers {
    //! 驱动侧适配：把真实解析器接成 [`llm_fuzz::Target`]。
    //!
    //! 这份定义是**唯一出处**：确定性遍历（`tests/parsers.rs` 的 [`llm_fuzz::sweep`]）与
    //! libfuzzer 模糊 harness（`tests/fuzz_*.rs`）都从这里取同一组入口。两边不各写一份判据，避免
    //! 文本/字节边界的处理在两处悄悄漂移。
    //!
    //! 运行时只把"任意字节 → 接受/拒绝"的结果交回契约层；解析器本身在 `cli` / `weights` /
    //! `snapshot` / `serve::proto` / `wire`，不变量本身在 `wire`。
    //!
    //! 每个 harness 例子只取其中一个适配，未被该消费方引用的项是有意保留的整组定义，不报 dead_code。
    #![allow(dead_code)]

    use std::sync::OnceLock;

    use cutile::builtin::Catalogue;
    use diagnostic::{Code, Error};
    use llm_fuzz::Target;
    use wire::sink::Sink;

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
            let mut raw = wire::leb::Cursor::new(input);
            if let Ok(prefix) = raw.read::<u64>() {
                let consumed = &input[..raw.pos()];
                let mut again = Vec::new();
                wire::leb::write(&mut again, prefix);
                if again.as_slice() != consumed {
                    return Err(inconsistent("LEB128 规范解码后的重编码与原前缀不一致"));
                }
            }
            Ok(())
        }
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
            let profile =
                &profiles[usize::from(input.get(1).copied().unwrap_or(0)) % profiles.len()];
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
            let mut params = [0u8; 16];
            let n = input.len().min(16);
            params[..n].copy_from_slice(&input[..n]);
            let words: [u32; 4] = std::array::from_fn(|i| {
                u32::from_le_bytes([
                    params[i * 4],
                    params[i * 4 + 1],
                    params[i * 4 + 2],
                    params[i * 4 + 3],
                ])
            });
            let seed = u64::from(words[0]) | (u64::from(words[1]) << 32);

            // ① Philox 确定性：同一 seed 起两条流，逐位必须一致；`next_f32` 必须落在 [0,1)。
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

            // ② 均匀分布：参数合法时样本必须落在闭区间内（浮点取整下上界按闭区间判）。
            let lo = f32::from_bits(words[1]);
            let hi = f32::from_bits(words[2]);
            if let Ok(uniform) = rng::Uniform::new(lo, hi) {
                let span_finite = (hi - lo).is_finite();
                let mut rng = rng::Rng::new(seed);
                for _ in 0..64 {
                    let sample = uniform.sample(&mut rng);
                    if span_finite && (!sample.is_finite() || sample < lo || sample > hi) {
                        return Err(inconsistent("Uniform 样本越出 [lo,hi] 或非有限"));
                    }
                }
            }

            // ③ 伯努利端点：p=0 恒 false、p=1 恒 true。
            let p = f32::from_bits(words[3]);
            if let Ok(bernoulli) = rng::Bernoulli::new(p) {
                let mut rng = rng::Rng::new(seed);
                for _ in 0..64 {
                    let hit = bernoulli.sample(&mut rng);
                    if p == 0.0 && hit {
                        return Err(inconsistent("Bernoulli p=0 抽到了 true"));
                    }
                    if p == 1.0 && !hit {
                        return Err(inconsistent("Bernoulli p=1 抽到了 false"));
                    }
                }
            }

            // ④ 正态分布：均值/标准差有限时样本必须有限（Box-Muller 半径有界）。
            let mean = f32::from_bits(words[0]);
            let std = 1.0 + (words[1] & 0x3FF) as f32 * 0.001;
            if let Ok(normal) = rng::Normal::new(mean, std) {
                let mut rng = rng::Rng::new(seed);
                for _ in 0..64 {
                    if !normal.sample(&mut rng).is_finite() {
                        return Err(inconsistent("Normal 样本非有限"));
                    }
                }
            }
            Ok(())
        }
    }
}

pub mod seeds {
    //! 合法范例构造：sweep 遍历与 libFuzzer 种子语料共用同一份，格式演进只改一处。
    //!
    //! 二进制格式一律经库编码，不手抄字节：手抄版本一旦格式演进，就会悄悄变成"合法种子其实
    //! 不合法"。范例同时被两处消费：`parsers.rs` 的确定性扰动遍历，与 `gen_corpus.rs` 写到
    //! `corpus/<目标>/seed.bin` 的覆盖引导语料。
    //!
    //! 本模块被 `parsers.rs` 与 `gen_corpus.rs` 分别 include：单个目标只用到其中一部分构造，
    //! 未被该目标引用的项是有意保留的共享定义，不报 dead_code。
    #![allow(dead_code)]

    /// 一份合法的命令行：默认命令 + 一个带取值的选项。
    pub const ARGS: &[u8] = b"status --weights model.safetensors";

    /// 一份合法的 safetensors 头 JSON：一个 2×3 的 f32 条目，正好 24 字节数据。
    pub const HEADER: &str = r#"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;

    /// 由 [`HEADER`] 拼出完整文件字节（数据段全零，`parse` 不看内容只认长度）。
    pub fn safetensors() -> Vec<u8> {
        let mut file = (HEADER.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(HEADER.as_bytes());
        file.resize(file.len() + 24, 0);
        file
    }

    /// 一份合法的检查点：事实五个数取最小合法值，一个 2×3 的 f32 条目。
    pub fn snapshot() -> Vec<u8> {
        let facts = storage::Facts::new(1, 1, 1, 1, 1).expect("事实合法");
        let entry = weights::Entry::new("w", tensor::DType::F32, tensor::Shape::new([2, 3])).at(8);
        let header = weights::Header::new(vec![entry], 8).expect("条目表合法");
        weights::Snapshot::new(facts, header)
            .encode()
            .expect("编码合法检查点")
    }

    /// 一份合法的自有 `.u` 容器 v2：一个 2×3 的 f32 段，24 字节数据，64 字节对齐。
    pub fn u_container() -> Vec<u8> {
        let segment =
            weights::Segment::new("w", tensor::DType::F32, vec![2, 3], None, vec![0u8; 24])
                .expect("段合法");
        weights::Writer::new()
            .push(segment)
            .expect("段不重名")
            .finish(6)
            .expect("编码合法容器")
    }

    /// 一个合法的 LEB128 编码：300 的变长字节。
    pub fn leb() -> Vec<u8> {
        let mut out = Vec::new();
        wire::leb::write(&mut out, 300u64);
        out
    }

    /// 一条合法的帧：`Request` + 载荷 `hi`，含头尾校验。
    pub fn frame() -> Vec<u8> {
        serve::proto::Envelope::new(
            serve::proto::Version::current(),
            serve::proto::Kind::Request,
            b"hi".to_vec(),
        )
        .frame()
        .expect("编码合法帧")
        .to_vec()
    }

    /// 与目标无关的边界输入：空串、零字节、上界字节。
    ///
    /// 合法范例之外，覆盖引导还应从最短长度与极值字节起步；这些样本对所有解析器都适用，由
    /// 生成器随范例一起写进每个目标的语料目录。
    pub fn edges() -> [(&'static str, Vec<u8>); 3] {
        [
            ("edge-empty", Vec::new()),
            ("edge-null", vec![0x00]),
            ("edge-max", vec![0xFF]),
        ]
    }

    /// 一份合法的词法输入：含函数体、局部绑定与字面量，分组与字符串都闭合。
    pub const LEX: &[u8] = b"fn f() { let x = 1; }";

    /// harness 目标名 → 合法范例；顺序与 CI 的 `FUZZ_TARGETS`、`drivers.rs` 的驱动一一对应。
    pub fn exemplars() -> [(&'static str, Vec<u8>); 15] {
        [
            ("fuzz_cli", ARGS.to_vec()),
            ("fuzz_weights_json", HEADER.as_bytes().to_vec()),
            ("fuzz_weights_safetensors", safetensors()),
            ("fuzz_weights_archive", u_container()),
            ("fuzz_train_snapshot", snapshot()),
            ("fuzz_serve_frames", frame()),
            (
                "fuzz_http_head",
                b"POST /v1/echo HTTP/1.1\r\nContent-Length: 2\r\n\r\n".to_vec(),
            ),
            ("fuzz_wire_sealed", frame()),
            ("fuzz_wire_leb", leb()),
            ("fuzz_lex", LEX.to_vec()),
            // 语义 oracle 的种子：任意合法输入即可触发不变量断言（`run` 恒返回 Ok）。
            ("fuzz_crc", b"llm crc seed".to_vec()),
            ("fuzz_leb_roundtrip", leb()),
            ("fuzz_shape", vec![2, 0, 3, 0, 4, 0, 5, 0]),
            ("fuzz_cutile_negotiate", vec![0, 0]),
            ("fuzz_dist", vec![0u8; 16]),
        ]
    }
}

// libfuzzer harness 入口：只在 `engine` feature 下编译（默认测试不拉 libFuzzer 运行时）。
// 每个测试二进制 `include!` 本文件：适配器由调用方传给 `fuzz_harness`；契约核对在
// `llm_fuzz::Target` 一处实现，这里只做边界上的违约转 panic。
//
// libFuzzer 运行时只在 Unix 提供可链接入口；Windows/MSVC 下 harness 目标编一个空 `main`
// 桩，契约 sweep（`tests/parsers.rs`）仍在 Windows 上跑，只是没有覆盖引导运行时。

// 本文件被 15 个 harness 与 parsers/gen_corpus 共享 include：这两个 import 只在 harness
// 目标里被使用，非 harness 目标（gen_corpus/parsers）在 --all-features 下同样满足 cfg，
// 因此按 include 语义定向放开未用导入告警。
#[allow(unused_imports)]
#[cfg(all(unix, feature = "engine"))]
use libfuzzer_sys::fuzz_target;
#[cfg(all(unix, feature = "engine"))]
use llm_fuzz::Target;

/// harness 边界：契约核对违约时把 [`Finding`] 转成 panic，交给 libFuzzer 留下触发输入。
///
/// 契约库本身不出现 panic（生产 `src` 零 panic 门）；终止进程这一步只属于 harness 入口，
/// 全部目标共用这一处转换，不再各写一遍 `if let Err(...) { panic!(...) }`。
#[cfg(all(unix, feature = "engine"))]
#[allow(dead_code)]
fn panic_on_finding<T: Target>(target: &T, data: &[u8]) {
    if let Err(finding) = target.exercise(data) {
        panic!("{finding}");
    }
}

/// 把一个 `Target` 适配接成 libFuzzer 入口；违约由 [`panic_on_finding`] 转成 panic。
#[cfg(all(unix, feature = "engine"))]
#[allow(unused_macros)]
macro_rules! fuzz_harness {
    ($adapter:expr) => {
        fuzz_target!(|data: &[u8]| panic_on_finding(&$adapter, data));
    };
}

/// 非 Unix 桩：没有 libFuzzer 运行时，目标只需要一个能链接的入口；适配器仍被引用以保持
/// 驱动签名与 Unix 侧一致。
#[cfg(all(not(unix), feature = "engine"))]
#[allow(unused_macros)]
macro_rules! fuzz_harness {
    ($adapter:expr) => {
        #[allow(dead_code)]
        fn main() {
            let _ = &$adapter;
        }
    };
}
