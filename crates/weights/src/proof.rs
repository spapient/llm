//! 完整性证明：零依赖 SHA-256 摘要 + 逐张量 Merkle 树与包含证明。
//!
//! 哈希实现 FIPS 180-4（消息填充、64 轮压缩、消息调度），输出 32 字节定长摘要；
//! 正确性由标准向量（空输入、`"abc"`、双块长度）对拍钉住。Merkle 叶子绑定张量名，
//! 单张量包含证明 O(log n) 可离线验证。

/// 摘要长度（字节）。
pub const HASH_LEN: usize = 32;

/// 摘要字节。
pub type Digest = [u8; HASH_LEN];

/// 初始哈希值（FIPS 180-4 §5.3.3）。
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// 轮常量（前 32 位素数小数部分的立方根）。
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e376c08,
    0x2748774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc70208,
    0x90befffa,
    0xa4506ceb,
    0xbef9a3f7,
    0xc67178f2,
];

/// 流式哈希上下文：可增量吸收任意长度，最后取摘要。
#[derive(Clone)]
pub struct Sha256 {
    /// 当前链变量。
    state: [u32; 8],
    /// 未凑满 64 字节的缓冲。
    buffer: [u8; 64],
    /// 缓冲中已用字节。
    fill: usize,
    /// 已处理总长度（字节），用于最终填充。
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// 新的哈希上下文。
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: H0,
            buffer: [0u8; 64],
            fill: 0,
            total: 0,
        }
    }

    /// 增量吸收一段字节。
    pub fn update(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len() as u64);
        let mut input = bytes;

        // 先填满上一块的残余。
        if self.fill > 0 {
            let need = 64 - self.fill;
            let take = need.min(input.len());
            self.buffer[self.fill..self.fill + take].copy_from_slice(&input[..take]);
            self.fill += take;
            input = &input[take..];
            if self.fill == 64 {
                let block = self.buffer;
                compress(&mut self.state, &block);
                self.fill = 0;
            }
        }

        // 完整块直接处理，不经过缓冲。
        while input.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&input[..64]);
            compress(&mut self.state, &block);
            input = &input[64..];
        }

        // 记下尾部不足一块的部分。
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.fill = input.len();
        }
    }

    /// 结束并产出摘要（消耗 self）。
    #[must_use]
    pub fn finish(mut self) -> Digest {
        let bit_len = self.total.wrapping_mul(8);
        // 附加 0x80。
        self.buffer[self.fill] = 0x80;
        self.fill += 1;
        // 剩余空间放不下 8 字节长度时，先补零压一块再开新块。
        if self.fill > 56 {
            for slot in &mut self.buffer[self.fill..] {
                *slot = 0;
            }
            let block = self.buffer;
            compress(&mut self.state, &block);
            self.fill = 0;
        }
        for slot in &mut self.buffer[self.fill..56] {
            *slot = 0;
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        compress(&mut self.state, &block);

        let mut out = [0u8; HASH_LEN];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// 压缩一个 64 字节块，更新链变量。
fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().enumerate().take(16) {
        let at = i * 4;
        *word = u32::from_be_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for (i, &k) in K.iter().enumerate() {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(k)
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// 叶子哈希：域前缀 0x00 + 名字长度 + 名字 + 张量规范字节。
#[must_use]
pub(crate) fn leaf(name: &str, data: &[u8]) -> Digest {
    let mut ctx = Sha256::new();
    ctx.update(&[0x00]);
    ctx.update(&(name.len() as u64).to_le_bytes());
    ctx.update(name.as_bytes());
    ctx.update(data);
    ctx.finish()
}

/// 纯字节摘要：域前缀 0x02，不绑名字；供写端 REF 去重的代表表做哈希键。
///
/// 哈希只做快速分流，不等于复用判据：命中后仍要逐位比对元数据与字节，碰撞只错过复用，
/// 不会错配（见 `writer` 的代表表）。
#[must_use]
pub(crate) fn hash_bytes(data: &[u8]) -> Digest {
    let mut ctx = Sha256::new();
    ctx.update(&[0x02]);
    ctx.update(data);
    ctx.finish()
}

/// 内部节点：域前缀 0x01，与叶子区分。
fn pair(left: &Digest, right: &Digest) -> Digest {
    let mut ctx = Sha256::new();
    ctx.update(&[0x01]);
    ctx.update(left);
    ctx.update(right);
    ctx.finish()
}

/// 奇数层复制末节点补齐成对；`root` 与 `prove` 共用这一份补奇规则，避免两处各写一遍。
fn pad(level: &mut Vec<Digest>) {
    if !level.len().is_multiple_of(2)
        && let Some(last) = level.last().copied()
    {
        level.push(last);
    }
}

/// 补齐后的一层两两 `pair`，折出下一层。
fn fold(level: Vec<Digest>) -> Vec<Digest> {
    level
        .chunks(2)
        .map(|chunk| pair(&chunk[0], &chunk[1]))
        .collect()
}

/// 由叶子列表计算 Merkle 根；空列表返回全零（调用方应拒绝空账本）。
#[must_use]
pub(crate) fn root(leaves: &[Digest]) -> Digest {
    if leaves.is_empty() {
        return [0u8; HASH_LEN];
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        pad(&mut level);
        level = fold(level);
    }
    level[0]
}

/// 单张量包含证明：叶子下标 + 自底向上的兄弟节点序列。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Proof {
    /// 叶子在原列表的下标。
    index: usize,
    /// 兄弟节点，按从叶子到根的顺序。
    siblings: Vec<Digest>,
}

impl Proof {
    /// 叶子在原列表的下标。
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// 兄弟节点，按从叶子到根的顺序。
    #[must_use]
    pub fn siblings(&self) -> &[Digest] {
        &self.siblings
    }
}

/// 生成下标的包含证明；下标越界返回 None。
#[must_use]
pub(crate) fn prove(leaves: &[Digest], index: usize) -> Option<Proof> {
    if index >= leaves.len() {
        return None;
    }
    let mut level = leaves.to_vec();
    let mut idx = index;
    let mut siblings = Vec::new();
    while level.len() > 1 {
        pad(&mut level);
        let sibling = idx ^ 1;
        siblings.push(level[sibling]);
        level = fold(level);
        idx /= 2;
    }
    Some(Proof { index, siblings })
}

/// 仅用一个叶子哈希、包含证明和根验证归属，不接触其他张量。
#[must_use]
pub(crate) fn verify_proof(leaf_hash: &Digest, proof: &Proof, merkle_root: &Digest) -> bool {
    let mut node = *leaf_hash;
    let mut idx = proof.index;
    for sibling in &proof.siblings {
        node = if idx.is_multiple_of(2) {
            pair(&node, sibling)
        } else {
            pair(sibling, &node)
        };
        idx /= 2;
    }
    node == *merkle_root
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用一次性摘要，包装流式结构。
    fn hash(data: &[u8]) -> Digest {
        let mut ctx = Sha256::new();
        ctx.update(data);
        ctx.finish()
    }

    fn hex(digest: Digest) -> String {
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn empty_vector() {
        assert_eq!(
            hex(hash(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn abc_vector() {
        assert_eq!(
            hex(hash(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn two_block_vector() {
        // 448 位（56 字节）恰好落在填充边界，外加多块长度，覆盖双块路径。
        let data = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(hash(data)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn streaming_equals_one_shot() {
        let data = vec![0x5au8; 200];
        let whole = hash(&data);
        let mut ctx = Sha256::new();
        let (chunks, remainder) = data.as_chunks::<7>();
        for chunk in chunks {
            ctx.update(chunk);
        }
        ctx.update(remainder);
        assert_eq!(ctx.finish(), whole);
    }

    #[test]
    fn distinct_inputs_distinct_hashes() {
        assert_ne!(hash(b"x"), hash(b"y"));
    }

    fn leaves(names: &[&str]) -> Vec<Digest> {
        names
            .iter()
            .enumerate()
            .map(|(i, name)| leaf(name, &[i as u8; 16]))
            .collect()
    }

    #[test]
    fn every_index_proves_against_root() {
        let names = ["a", "b", "c", "d", "e", "f", "g"];
        let tree = leaves(&names);
        let merkle_root = root(&tree);
        for (index, digest) in tree.iter().enumerate() {
            let proof = prove(&tree, index).expect("证明");
            assert!(
                verify_proof(digest, &proof, &merkle_root),
                "下标 {index} 应验证通过"
            );
            // 证明长度为 ceil(log2 n)。
            assert_eq!(proof.siblings.len(), 3);
        }
    }

    #[test]
    fn proof_rejects_tampered_leaf() {
        let tree = leaves(&["a", "b", "c", "d"]);
        let merkle_root = root(&tree);
        let proof = prove(&tree, 1).expect("证明");
        let fake = leaf("b", &[0xFF; 16]);
        assert!(!verify_proof(&fake, &proof, &merkle_root));
    }

    #[test]
    fn proof_rejects_wrong_root() {
        let tree = leaves(&["a", "b", "c"]);
        let other = leaves(&["x", "y", "z"]);
        let proof = prove(&tree, 0).expect("证明");
        assert!(!verify_proof(&tree[0], &proof, &root(&other)));
    }

    #[test]
    fn out_of_range_proof_is_none() {
        let tree = leaves(&["a", "b"]);
        assert!(prove(&tree, 5).is_none());
    }

    #[test]
    fn name_binding_blocks_swap() {
        // 同内容不同名的叶子哈希必须不同。
        assert_ne!(leaf("x", &[1; 8]), leaf("y", &[1; 8]));
    }

    #[test]
    fn empty_root_is_zero() {
        assert_eq!(root(&[]), [0u8; 32]);
    }
}
