//! 逐张量 Merkle 证明：叶子绑定张量名，单张量包含证明 O(log n) 可离线验证。
//!
//! 防第二原像攻击的三条固定规则：
//! - 叶子与内部节点用不同域前缀（`0x00` / `0x01`），否则内部节点可被伪造成叶子；
//! - 叶子输入绑定张量名（含长度，防截断混淆），同内容的两个张量不可互换；
//! - 奇数层复制最后一个节点向上，验证侧规则一致，不引入歧义。

use crate::sha256::{self, Digest};

/// 叶子哈希：域前缀 0x00 + 名字长度 + 名字 + 张量规范字节。
#[must_use]
pub(crate) fn leaf(name: &str, data: &[u8]) -> Digest {
    let mut ctx = sha256::Sha256::new();
    ctx.update(&[0x00]);
    ctx.update(&(name.len() as u64).to_le_bytes());
    ctx.update(name.as_bytes());
    ctx.update(data);
    ctx.finish()
}

/// 内部节点：域前缀 0x01，与叶子区分。
fn pair(left: &Digest, right: &Digest) -> Digest {
    let mut ctx = sha256::Sha256::new();
    ctx.update(&[0x01]);
    ctx.update(left);
    ctx.update(right);
    ctx.finish()
}

/// 由叶子列表计算 Merkle 根；空列表返回全零（调用方应拒绝空账本）。
#[must_use]
pub(crate) fn root(leaves: &[Digest]) -> Digest {
    if leaves.is_empty() {
        return [0u8; sha256::HASH_LEN];
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        if !level.len().is_multiple_of(2)
            && let Some(last) = level.last().copied()
        {
            level.push(last);
        }
        level = level
            .chunks(2)
            .map(|chunk| pair(&chunk[0], &chunk[1]))
            .collect();
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
        if !level.len().is_multiple_of(2)
            && let Some(last) = level.last().copied()
        {
            level.push(last);
        }
        let sibling = idx ^ 1;
        siblings.push(level[sibling]);
        level = level
            .chunks(2)
            .map(|chunk| pair(&chunk[0], &chunk[1]))
            .collect();
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
