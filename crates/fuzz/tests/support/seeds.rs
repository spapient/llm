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
    let facts = train::snapshot::Facts::new(1, 1, 1, 1, 1).expect("事实合法");
    let entry = weights::Entry::new("w", tensor::DType::F32, tensor::Shape::new([2, 3])).at(8);
    let header = weights::Header::new(vec![entry], 8).expect("条目表合法");
    train::snapshot::Snapshot::new(facts, header)
        .encode()
        .expect("编码合法检查点")
}

/// 一份合法的 LLPK 容器：一个 2×3 的 f32 段，24 字节数据，64 字节对齐。
pub fn llpk() -> Vec<u8> {
    let segment = weights::Segment::new("w", tensor::DType::F32, vec![2, 3], None, vec![0u8; 24])
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

/// P0-6 范例：词表边界——首字节起一个合法 ASCII 词片，其余字节给出重复/续节边界。
pub fn p06_vocab() -> Vec<u8> {
    b"abracadabra".to_vec()
}

/// P0-6 范例：调度写法 `ws<2,2>` 是合法闭集成员。
pub const P06_SCHEDULE: &[u8] = b"ws<2,2>";

/// P0-6 范例：einsum 表达式 `nm,m->n`。
pub const P06_EXPR: &[u8] = b"nm,m->n";

/// P0-6 范例：能力区间 `16..64`。
pub const P06_SPAN: &[u8] = b"16..64";

/// P0-6 范例：原语闭集名字 `mma`。
pub const P06_PRIM: &[u8] = b"mma";

/// P0-6 范例：存储空间闭集名字 `GM`。
pub const P06_SPACE: &[u8] = b"GM";

/// P0-6 范例：数据类型名字 `F32`。
pub const P06_DTYPE: &[u8] = b"F32";

/// P0-6 范例：两个等长 f32 行（加法/乘法/归约共用），`[0.25, 0.5, 0.75, 1.0]`。
pub fn p06_f32_pair() -> Vec<u8> {
    let a = [0.25f32, 0.5, 0.75, 1.0];
    let mut out = Vec::new();
    for value in a {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// P0-6 范例：softmax 一行 f32。
pub fn p06_f32_row() -> Vec<u8> {
    p06_f32_pair()
}

/// P0-6 范例：embedding 索引行 + 权重表（行宽 4，两行共 8 个表元素）。
pub fn p06_embed() -> Vec<u8> {
    let idx = [0.0f32, 1.0];
    let table = [
        0.1f32, 0.2, 0.3, 0.4, // 第 0 行
        0.5, 0.6, 0.7, 0.8, // 第 1 行
    ];
    let mut out = Vec::new();
    for value in idx.iter().chain(table.iter()) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// P0-6 范例：反量化打包字节（4 位 nibble 表）。
pub const P06_DEQUANT_I4: &[u8] = &[0x00, 0x00, 0x12, 0x34];

/// P0-6 范例：8 位反量化的 scale/zero/量化值 f32 块。
pub fn p06_dequant_i8() -> Vec<u8> {
    let values = [0.25f32, 8.0, 0.0, 8.0, 15.0];
    let mut out = Vec::new();
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// 一份合法的词法输入：含函数体、局部绑定与字面量，分组与字符串都闭合。
pub const LEX: &[u8] = b"fn f() { let x = 1; }";

/// harness 目标名 → 合法范例；顺序与 CI 的 `FUZZ_TARGETS`、`drivers.rs` 的驱动一一对应。
pub fn exemplars() -> [(&'static str, Vec<u8>); 9] {
    [
        ("fuzz_cli", ARGS.to_vec()),
        ("fuzz_weights_json", HEADER.as_bytes().to_vec()),
        ("fuzz_weights_safetensors", safetensors()),
        ("fuzz_weights_archive", llpk()),
        ("fuzz_train_snapshot", snapshot()),
        ("fuzz_serve_frames", frame()),
        ("fuzz_wire_sealed", frame()),
        ("fuzz_wire_leb", leb()),
        ("fuzz_lex", LEX.to_vec()),
    ]
}
