//! `proof` 模块的单元测试。
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
fn every_idx_proves_against_root() {
    let names = ["a", "b", "c", "d", "e", "f", "g"];

    let tree = leaves(&names);

    let merkle_root = root(&tree);

    for (idx, digest) in tree.iter().enumerate() {
        let proof = prove(&tree, idx).expect("证明");

        assert!(
            verify_proof(digest, &proof, &merkle_root),
            "下标 {idx} 应验证通过"
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
