//! 确定性用例生成：一份合法范例 + 用例序号 → 跨平台可复现的扰动字节串。
//!
//! 生成器只做整数混合，不接随机源、不读时钟、不碰全局状态：同一个 `(seed, case)` 在任何
//! 机器、任何工具链版本下都得到同一串字节，失败输入才能原样复现。变异调度覆盖三类边角：
//! 位/字节翻转、边界字节替换与整数 ±1 扰动、长度增减（插入、删除、截断、追加、区段复制）
//! ——真实缺陷几乎总长在这些位置，而不是与范例毫无关系的纯噪声。

/// 确定性种子：同一个种子永远派生同一批用例，失败才能复现。
///
/// 持有一份**合法范例**字节：遍历从它的边角上长出错误路径，而不是与真实输入毫无关系的噪声。
/// 种子本身是否被目标接受，由集成测试单独核对（见 `tests/parsers.rs`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed(Vec<u8>);

impl Seed {
    /// 由初始字节构造；空种子是合法输入（"空输入"本身就是一道用例）。
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    /// 初始字节。
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// 初始字节数。
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// 是否为空种子。
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// 64 位确定性流（SplitMix64）：状态回卷后做两次乘加终混，低 32 位与高 32 位都有充分
/// 雪崩。类型不对外公开：外部只通过 [`Seed::derive`] 取字节，不暴露流的形状。
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    /// 由初始状态构造；状态为 0 也合法（常量增量会把它推离零点）。
    fn new(state: u64) -> Self {
        Self { state }
    }

    /// 取下一个 64 位值。
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ z >> 31
    }

    /// 取 `[0, bound)` 内的确定性下标；`bound` 为 0 时返回 0（调用方据此判空）。
    ///
    /// 用乘移（`u64 × bound` 取高 64 位）而不是取模：小边界下取模会让余数偏小的下标
    /// 概率略高，乘移在整个区间上均匀，且不需要拒绝采样回退流。
    fn below(&mut self, bound: usize) -> usize {
        match bound {
            0 => 0,
            bound => ((self.next_u64() as u128 * bound as u128) >> 64) as usize,
        }
    }

    /// 取一个随机字节。
    fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }
}

/// 范例字节的 64 位指纹：FNV-1a 逐字节折叠，再过 SplitMix 终混。
///
/// 旧实现把范例字节留在状态之外，改一个字节只影响该字节自身的异或掩码；这里让每一位输入都
/// 参与状态混合，范例的一位翻转会扩散到整条派生流。
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    let mut z = hash;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ z >> 31
}

/// 格式边角上反复出现的危险字节：零、上界、符号位边界、ASCII 空白与 UTF-8 续节/首节。
const BOUNDARY: [u8; 12] = [
    0x00, 0x7F, 0x80, 0xFF, 0x01, 0x7E, 0xFE, 0x20, 0x0A, 0x0D, 0xBF, 0xC0,
];

/// 单步变异数量；用例内每一步都从这张表里抽。新增一步只改这里与 [`mutate`] 的匹配。
const STEPS: usize = 9;

impl Seed {
    /// 派生第 `index` 个用例：范例的一份确定性扰动。
    ///
    /// 输出长度不再钉死在范例长度上：删除/截断能收出空串与前缀片段，插入/追加/复制能长出
    /// 更长的串。长度上限由变异步本身约束（追加 ≤ 4 字节、区段复制不超过范例长度），遍历成本
    /// 可预期。
    pub fn derive(&self, index: u64) -> Vec<u8> {
        let mut rng = Rng::new(fingerprint(&self.0) ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut out = self.0.clone();
        if out.is_empty() {
            // 空范例没有可删可改的位置：从流里造 0..=2 个边界字节，仍然覆盖空串、单字节与
            // 双字节这三个最短边界。
            let len = rng.below(3);
            return (0..len)
                .map(|_| BOUNDARY[rng.below(BOUNDARY.len())])
                .collect();
        }
        mutate(&mut out, &mut rng);
        out
    }
}

/// 对 `out` 做 1..=3 步复合变异；约一成用例原样保留，保住合法路径覆盖。
///
/// 每一步都在空/单字节边界上退化安全，不会 panic；截断收出空串后，后续位置类步骤直接跳过。
fn mutate(out: &mut Vec<u8>, rng: &mut Rng) {
    if rng.below(10) == 0 {
        return;
    }
    let steps = 1 + rng.below(3);
    for _ in 0..steps {
        match rng.below(STEPS) {
            0 => flip_bit(out, rng),
            1 => substitute_boundary(out, rng),
            2 => replace_byte(out, rng),
            3 => nudge_byte(out, rng),
            4 => insert_byte(out, rng),
            5 => delete_byte(out, rng),
            6 => truncate(out, rng),
            7 => append_bytes(out, rng),
            _ => duplicate_region(out, rng),
        }
    }
}

/// 翻转一个随机位；空串无可选位置，跳过。
fn flip_bit(out: &mut [u8], rng: &mut Rng) {
    if out.is_empty() {
        return;
    }
    let pos = rng.below(out.len());
    out[pos] ^= 1 << rng.below(8);
}

/// 把一个位置换成流里的随机字节；空串跳过。
fn replace_byte(out: &mut [u8], rng: &mut Rng) {
    if out.is_empty() {
        return;
    }
    let pos = rng.below(out.len());
    out[pos] = rng.byte();
}

/// 把一个位置换成表格里的边界字节；空串跳过。
fn substitute_boundary(out: &mut [u8], rng: &mut Rng) {
    if out.is_empty() {
        return;
    }
    let pos = rng.below(out.len());
    out[pos] = BOUNDARY[rng.below(BOUNDARY.len())];
}

/// 一个字节做环绕 +1 / -1：长度、偏移、计数值这类整数字段的差一错误走这一步；空串跳过。
fn nudge_byte(out: &mut [u8], rng: &mut Rng) {
    if out.is_empty() {
        return;
    }
    let pos = rng.below(out.len());
    // -1 用环绕加 255 表达，保持单字节运算。
    let delta = if rng.below(2) == 0 { 1u8 } else { u8::MAX };
    out[pos] = out[pos].wrapping_add(delta);
}

/// 在随机位置插入一个字节（边界字节与随机字节各半）。
fn insert_byte(out: &mut Vec<u8>, rng: &mut Rng) {
    let pos = rng.below(out.len() + 1);
    let byte = pick(rng);
    out.insert(pos, byte);
}

/// 删除一个位置；长度 ≤ 1 时保留，不做没有意义的删除。
fn delete_byte(out: &mut Vec<u8>, rng: &mut Rng) {
    if out.len() <= 1 {
        return;
    }
    let pos = rng.below(out.len());
    out.remove(pos);
}

/// 截断到随机前缀长度（含 0：整条变空输入）。
fn truncate(out: &mut Vec<u8>, rng: &mut Rng) {
    let keep = rng.below(out.len() + 1);
    out.truncate(keep);
}

/// 尾部追加 1..=4 个字节；先在小数组里填好，再整块 `extend_from_slice`，不逐个 push。
fn append_bytes(out: &mut Vec<u8>, rng: &mut Rng) {
    let count = 1 + rng.below(4);
    let mut bytes = [0u8; 4];
    for slot in &mut bytes[..count] {
        *slot = pick(rng);
    }
    out.extend_from_slice(&bytes[..count]);
}

/// 复制一段既有字节插到随机位置：长串重复与跨区拼接走这一步；空串跳过。
fn duplicate_region(out: &mut Vec<u8>, rng: &mut Rng) {
    if out.is_empty() {
        return;
    }
    let start = rng.below(out.len());
    let rest = out.len() - start;
    let end = start + 1 + rng.below(rest);
    let region = out[start..end].to_vec();
    let pos = rng.below(out.len() + 1);
    out.splice(pos..pos, region);
}

/// 一半取边界字节，一半取流随机字节。
fn pick(rng: &mut Rng) -> u8 {
    match rng.below(2) {
        0 => BOUNDARY[rng.below(BOUNDARY.len())],
        _ => rng.byte(),
    }
}
