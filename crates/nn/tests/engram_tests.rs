use super::*;

/// 数学验证 40：N=3、K=8 ⇒ 16 槽/token。
#[test]
fn slots_match_n3_k8() {
    assert_eq!(slots(3, 8), 16);
    assert_eq!(slots(4, 8), 24);
    assert_eq!(slots(2, 1), 1);
}

/// φ 是确定性的：同输入必同输出。
#[test]
fn phi_is_deterministic() {
    let g = [7u32, 11, 13];
    assert_eq!(phi(&g, 0), phi(&g, 0));
    assert_ne!(phi(&g, 0), phi(&g, 1), "不同种子应散开");
}

/// ρ 分配构成单位剖分：ρ·p + (1−ρ)·p = p。
#[test]
fn allocate_partitions_sparse() {
    for rho in [0.0f32, 0.25, 0.743, 1.0] {
        let (moe, engram) = allocate(rho, 100.0);
        assert!((moe + engram - 100.0).abs() < 1e-6);
        assert!((moe - rho * 100.0).abs() < 1e-6);
        assert!((engram - (1.0 - rho) * 100.0).abs() < 1e-6);
    }
}

/// 表长提升到素数。
#[test]
fn next_prime_is_prime_and_rounds_up() {
    assert_eq!(next_prime(1), 2);
    assert_eq!(next_prime(10), 11);
    assert_eq!(next_prime(11), 11);
    assert!(is_prime(next_prime(9973)));
}

/// 同 N-gram 落到同槽，读写往返一致。
#[test]
fn table_get_set_roundtrip() {
    let mut t = Table::<f32>::new(10, 4);
    assert!(t.len() >= 10 && is_prime(t.len()));
    let g = [3u32, 1, 4];
    t.set(&g, 7, &[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(t.get(&g, 7), &[0.1, 0.2, 0.3, 0.4]);
}

/// f64 表复用同一份实现（泛型不复制）。
#[test]
fn table_f64_reuses_impl() {
    let mut t = Table::<f64>::new(10, 2);
    let g = [1u32, 1, 2];
    t.set(&g, 3, &[0.5, 1.5]);
    assert_eq!(t.get(&g, 3), &[0.5, 1.5]);
}

/// 槽数与表长无关（O(1) 查找，§5.8.2）。
#[test]
fn slot_is_independent_of_table_length() {
    // `slots` 只依赖 N、K，不接收表长 M；不同表长对同 N-gram 的槽下标由 φ mod M 决定，
    // 单次查找复杂度 O(N)（哈希 N-gram），与 M 无关。
    let small = Table::<f32>::new(16, 1);
    let large = Table::<f32>::new(16_384, 1);
    let g = [5u32, 9];
    assert!(small.slot(&g, 0) < small.len());
    assert!(large.slot(&g, 0) < large.len());
}

/// 历史不足 `order` 时不加记忆（不是加零向量）。
#[test]
fn a_short_history_adds_nothing() {
    let bypass = Bypass::new(Table::<f32>::new(11, 2), 7, 3, 1.0).expect("合法旁路");
    let mut hidden = [1.0f32, 2.0];
    assert!(!bypass.apply(&mut hidden, &[1, 2]).expect("合法宽度"));
    assert_eq!(hidden, [1.0, 2.0]);
    assert!(bypass.recall(&[1, 2]).is_none());
    assert!(bypass.recall(&[]).is_none());
}

/// 记忆按 `gate` 缩放进残差；查的是**尾部** N-gram，历史里更早的 token 不算。
#[test]
fn gate_scales_the_memory_residual() {
    let mut table = Table::<f32>::new(11, 2);
    table.set(&[1, 2, 3], 7, &[2.0, -4.0]);
    let half = Bypass::new(table.clone(), 7, 3, 0.5).expect("合法旁路");
    let mut hidden = [1.0f32, 2.0];
    assert!(half.apply(&mut hidden, &[9, 1, 2, 3]).expect("合法宽度"));
    assert_eq!(hidden, [2.0, 0.0], "9 不在尾部三元组里");
    let off = Bypass::new(table, 7, 3, 0.0).expect("合法旁路");
    let mut untouched = [1.0f32, 2.0];
    assert!(off.apply(&mut untouched, &[1, 2, 3]).expect("合法宽度"));
    assert_eq!(untouched, [1.0, 2.0], "gate = 0 时残差为零");
}

/// 装配视图的参数格与 [`Bypass`] 同一套判据，另加一条下标通道上界。
#[test]
fn a_memory_block_rejects_params_outside_the_lattice() {
    assert_eq!(
        MemoryBlock::new("mem", 0, 2, 1, 8, 1.0).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        MemoryBlock::new("mem", 4, 2, 1, 0, 1.0).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        MemoryBlock::new("mem", 4, 2, 1, MAX_BUCKETS + 1, 1.0)
            .unwrap_err()
            .code(),
        Code::Invalid,
        "超过 2^24 的槽号在 f32 下标通道上会被静默合并"
    );
    assert_eq!(
        MemoryBlock::new("mem", 4, 1, 1, 8, 1.0).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        MemoryBlock::new("mem", 4, 2, 1, 8, f32::NAN)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    let block = MemoryBlock::new("mem", 4, 2, 0x5eed, 1 << 24, 0.5).expect("上界本身合法");
    assert_eq!(block.width(), 4);
    assert_eq!(block.order(), 2);
    assert_eq!(block.seed(), 0x5eed);
    assert_eq!(block.buckets(), 1 << 24);
    assert_eq!(block.gate(), 0.5);
}

/// 阶 < 2 与嵌入宽度不符都在入口失败。
#[test]
fn a_bypass_rejects_bad_order_and_width() {
    let bad = Bypass::new(Table::<f32>::new(11, 2), 7, 1, 1.0);
    assert_eq!(bad.unwrap_err().code(), Code::Invalid, "1-gram 没有上下文");
    let bypass = Bypass::new(Table::<f32>::new(11, 2), 7, 2, 0.25).expect("合法旁路");
    assert_eq!(bypass.order(), 2);
    assert_eq!(bypass.gate(), 0.25);
    assert_eq!(bypass.table().dim(), 2);
    assert_eq!(
        bypass.apply(&mut [0.0f32; 3], &[1, 2]).unwrap_err().code(),
        Code::Invalid
    );
}
