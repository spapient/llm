//! 合法范例构造：登记表与种子语料共用同一份，格式演进只改一处。
//!
//! 二进制格式一律经库编码，不手抄字节：手抄版本一旦格式演进，就会悄悄变成"合法种子其实
//! 不合法"。范例由 `drivers::registry` 配对到适配器，边界样本由 `drivers::edges` 写到语料目录。
//!
//! 本模块被 `drivers.rs` 以 `#[path]` 包含：单个目标只用到其中一部分构造，未被该目标引用
//! 的项是有意保留的共享定义，不报 dead_code。
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
    storage::leb::write(&mut out, 300u64);
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
