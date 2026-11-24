//! `snapshot` 的往返与完整性：编码对称、位级对拍、校验和、截断与版本判据。

use super::snapshot_support::*;
#[test]
fn an_unsupported_dtype_is_an_error_not_a_panic() {
    // 元素类型标签是闭集：权重头部来自文件，闭集外的类型必须在编码前就变成错误。
    let weights = Header::new(
        vec![Entry::new("blk.0.embd", DType::U8, Shape::new([2])).at(64)],
        64,
    )
    .expect("头部");
    let snapshot = Snapshot::new(facts(), weights);
    let error = snapshot
        .encoded_len()
        .expect_err("闭集外的元素类型必须报错");
    assert!(error.to_string().contains("U8"), "{error}");
    assert!(snapshot.encode().is_err(), "编码与测长必须同一条判据");
}

#[test]
fn a_snapshot_round_trips_through_its_bytes() {
    let snapshot = snapshot();
    assert_round_trip(&snapshot);
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");
    assert_eq!(restored.facts(), facts());
    assert_eq!(restored.step(), 7);
    assert_eq!(restored.weights().entries().len(), 2);
    assert_eq!(restored.weights().entries()[1].name(), "blk.0.scales");
}

/// 变长整数的每个宽度边界都要对上：错一档就是半个检查点。
#[test]
fn optimizer_moments_round_trip_bit_for_bit() {
    let slots = vec![
        SlotMoments::new(vec![1.0, -0.5], vec![2.0, 0.25]).expect("一阶/二阶等长"),
        SlotMoments::new(vec![3.25], vec![0.125]).expect("一阶/二阶等长"),
    ];
    let moments = Moments::new(slots, 7);
    let snapshot = Snapshot::new_with_moments(facts(), empty_header(), moments);
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");

    let restored_moments = restored.moments();
    assert_eq!(restored_moments.step(), 7, "优化器步数必须恢复");
    let before = snapshot.moments().slots();
    let after = restored_moments.slots();
    assert_eq!(before.len(), after.len(), "槽位数必须一致");
    for (idx, (want, got)) in before.iter().zip(after).enumerate() {
        // 位级比较：f32 经小端字节往返不能只差一位。
        let want_m: Vec<u32> = want.m().iter().map(|value| value.to_bits()).collect();
        let got_m: Vec<u32> = got.m().iter().map(|value| value.to_bits()).collect();
        let want_v: Vec<u32> = want.v().iter().map(|value| value.to_bits()).collect();
        let got_v: Vec<u32> = got.v().iter().map(|value| value.to_bits()).collect();
        assert_eq!(got_m, want_m, "第 {idx} 槽一阶矩逐位一致");
        assert_eq!(got_v, want_v, "第 {idx} 槽二阶矩逐位一致");
    }
}

#[test]
fn accumulated_gradients_round_trip_bit_for_bit() {
    let accum = GradAccum::new(vec![vec![0.5, -0.25], vec![1.5]], 2);
    let snapshot = Snapshot::new_with_state(facts(), empty_header(), Moments::empty(), accum);
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");

    let got = restored.accum();
    assert_eq!(got.count(), 2, "窗口批数必须恢复");
    assert_accum_bits(&snapshot, &restored);
}

/// 窗口内多次 micro-batch 累加后的梯度总和、批数与绑定表一起往返：逐位对梯度、逐对对绑定。
#[test]
fn accumulated_window_sums_count_and_ties_round_trip_bit_for_bit() {
    // 三批逐批累加：源槽 1.0 + 0.5 + 0.25 = 1.75；回填槽 2.0 + 1.0 + 0.5 = 3.5。
    let accum = GradAccum::new(vec![vec![1.75, -0.75], vec![3.5]], 3);
    let ties = Ties::new(vec![(0, 1)]);
    let snapshot = Snapshot::new_with_all(facts(), empty_header(), Moments::empty(), accum, ties);
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");

    let got = restored.accum();
    assert_eq!(got.count(), 3, "窗口批数必须恢复");
    assert_accum_bits(&snapshot, &restored);
    assert_eq!(
        restored.ties().pairs(),
        snapshot.ties().pairs(),
        "绑定关系逐对一致"
    );
}

/// 绑定表逐对往返：源槽/回填槽下标在字节里按 varint 成对出现，顺序不被重排。
#[test]
fn tie_bindings_round_trip_pair_for_pair() {
    let ties = Ties::new(vec![(0, 2), (1, 3)]);
    let snapshot = Snapshot::new_with_all(
        facts(),
        empty_header(),
        Moments::empty(),
        GradAccum::empty(),
        ties,
    );
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");
    assert_eq!(restored.ties().pairs(), &[(0, 2), (1, 3)], "绑定对逐对一致");
}

/// 空累积窗口往返：零批、零梯度槽、零绑定对；快照整体与解码回的对象逐字段相等。
#[test]
fn an_empty_accumulation_window_round_trips_without_state() {
    let snapshot = Snapshot::new_with_state(
        facts(),
        empty_header(),
        Moments::empty(),
        GradAccum::empty(),
    );
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");
    assert_eq!(restored.accum().count(), 0, "空窗口批数为 0");
    assert!(restored.accum().slots().is_empty(), "空窗口没有累积槽");
    assert!(restored.ties().pairs().is_empty(), "空绑定表");
    assert_eq!(&restored, &snapshot);
}

#[test]
fn a_slot_with_uneven_moments_is_rejected_before_encoding() {
    let error = SlotMoments::new(vec![1.0], vec![1.0, 2.0]).expect_err("长度不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("长度不一致"), "{error}");
}

#[test]
fn the_length_promise_holds_at_every_varint_boundary() {
    let steps = [0u64, 1, 127, 128, 16_383, 16_384, 4_294_967_295, u64::MAX];
    for (idx, step) in steps.into_iter().enumerate() {
        let offset = 64 + idx * 65_536;
        let header = Header::new(
            vec![Entry::new("w", DType::F32, Shape::new(vec![1, 127, 128, 16_384])).at(offset)],
            64,
        )
        .expect("头部");
        let size = idx + 1;
        let facts = Facts::new(step, size, size, size, size).expect("事实");
        assert_round_trip(&Snapshot::new(facts, header));
    }
}

#[test]
fn a_snapshot_without_entries_still_round_trips() {
    let header = empty_header();
    assert_round_trip(&Snapshot::new(facts(), header));
}

/// 名字长度按**字节**算：按字符算就会在解码时切进 UTF-8 中间。
#[test]
fn a_unicode_name_is_measured_in_bytes() {
    let header = Header::new(
        vec![Entry::new("层.0.注意_权重", DType::BF16, Shape::new([3])).at(64)],
        64,
    )
    .expect("头部");
    let snapshot = Snapshot::new(facts(), header);
    assert_round_trip(&snapshot);
    let name = "层.0.注意_权重".as_bytes();
    assert!(
        snapshot
            .encode()
            .expect("编码")
            .windows(name.len())
            .any(|window| window == name),
        "名字原样进字节"
    );
}

/// 恢复前的第二道关：文件都是好的，但配的是另一次训练的权重。
///
/// 这一种失败在字节层面完全看不出来——它不会在读取时炸，而是在训练若干步之后才表现成
/// loss 不降。所以每一条差异都要被逐个拦住，而不是只比一个总数。
#[test]
fn check_weights_reports_the_first_entry_that_differs() {
    let snapshot = snapshot();
    let same = snapshot.weights().clone();
    snapshot.check_weights(&same).expect("同一份表");

    let two = || {
        vec![
            Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([2, 3])).at(64),
            Entry::new("blk.0.scales", DType::U4, Shape::new([4])).at(88),
        ]
    };
    let edited = |name: &str, dtype: DType, shape: [usize; 2], offset: usize| {
        Header::new(
            vec![
                Entry::new(name, dtype, Shape::new(shape)).at(offset),
                two()[1].clone(),
            ],
            64,
        )
        .expect("头部")
    };
    let cases: [(Header, [&str; 2]); 6] = [
        (Header::new(two(), 60).expect("头部"), ["数据段起点", "60"]),
        (
            Header::new(vec![two()[0].clone()], 64).expect("头部"),
            ["引用 2 个", "有 1 个"],
        ),
        (
            edited("blk.0.attn_k.weight", DType::F16, [2, 3], 64),
            ["attn_k", "attn_q"],
        ),
        (
            edited("blk.0.attn_q.weight", DType::BF16, [2, 3], 64),
            ["元素类型", "BF16"],
        ),
        (
            edited("blk.0.attn_q.weight", DType::F16, [3, 3], 64),
            ["形状", "[3, 3]"],
        ),
        (
            edited("blk.0.attn_q.weight", DType::F16, [2, 3], 72),
            ["偏移", "72"],
        ),
    ];
    for (weights, wants) in cases {
        let error = snapshot.check_weights(&weights).expect_err(wants[0]);
        assert_eq!(error.code(), Code::Invalid);
        for want in wants {
            assert!(error.to_string().contains(want), "{want}: {error}");
        }
    }
}

/// 检查点最贵的失败不是"读不出来"，而是"读出来了但不是当初写的那份"。校验和把每一种
/// 一个比特的改动都变成解码失败，这条测试把整个文件逐位翻一遍。
#[test]
fn every_flipped_bit_is_rejected() {
    let bytes = snapshot().encode().expect("编码");
    for offset in 0..bytes.len() {
        for bit in 0..8 {
            let mut broken = bytes.clone();
            broken[offset] ^= 1 << bit;
            assert!(
                Snapshot::decode(&broken).is_err(),
                "第 {offset} 字节第 {bit} 位翻转后被接受了"
            );
        }
    }
}

/// 校验和本身也要能被独立复算：表是编译期生成的，另写一份逐位定义来对。
#[test]
fn the_checksum_matches_its_bitwise_definition() {
    assert_eq!(
        wire::crc::crc32(b"123456789"),
        0xCBF4_3926,
        "IEEE 802.3 的标准检验值"
    );

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let bytes: Vec<u8> = (0..2048).map(|_| (lcg(&mut state) >> 33) as u8).collect();
    let mut crc = !0u32;
    for byte in &bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    assert_eq!(wire::crc::crc32(&bytes), !crc);
}

#[test]
fn a_foreign_magic_is_rejected() {
    let mut bytes = snapshot().encode().expect("编码");
    bytes[0] = b'X';
    let error = Snapshot::decode(&bytes).expect_err("魔数不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("LLMC"), "{error}");
}

/// 版本错误要说清是哪一边旧：读端旧要重存，写端旧要升级。
#[test]
fn another_version_says_which_side_is_old() {
    let mut newer = snapshot().encode().expect("编码");
    newer[4..6].copy_from_slice(&(Format::VERSION + 1).to_le_bytes());
    let error = Snapshot::decode(&newer).expect_err("更新");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("新"), "{error}");

    let mut older = snapshot().encode().expect("编码");
    older[4..6].copy_from_slice(&1u16.to_le_bytes());
    let error = Snapshot::decode(&older).expect_err("更旧");
    assert!(error.to_string().contains("旧"), "{error}");
}

#[test]
fn a_truncated_file_is_a_checksum_mismatch_not_a_half_snapshot() {
    let bytes = snapshot().encode().expect("编码");
    let error = Snapshot::decode(&bytes[..bytes.len() - 3]).expect_err("截断");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("校验和"), "{error}");

    let error = Snapshot::decode(&[]).expect_err("连文件头都没有");
    assert!(error.to_string().contains("截断"), "{error}");
}

/// 巡检、备份、下载后自检只想知道"这份字节能自证吗"：不必建出一整张条目表。
#[test]
fn verify_checks_the_bytes_without_building_the_snapshot() {
    let bytes = snapshot().encode().expect("编码");
    Snapshot::verify(&bytes).expect("完好");

    let mut changed = bytes.clone();
    let last = changed.len() - 1;
    changed[last] ^= 0x80;
    let error = Snapshot::verify(&changed).expect_err("校验和被改");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("校验和"), "{error}");

    // 比文件头还短：这不是"校验和对不上"，而是少了一截。
    let error = Snapshot::verify(&bytes[..Format::HEADER_BYTES]).expect_err("只有一个头");
    assert!(error.to_string().contains("截断"), "{error}");

    let mut foreign = bytes;
    foreign[0] = b'X';
    assert!(Snapshot::verify(&foreign).is_err(), "魔数不符也要拦下");
}

/// 校验和正确但正文不够：此时只有游标能说清缺在哪个偏移。
#[test]
fn a_sealed_but_short_payload_names_the_offset() {
    let bytes = snapshot().encode().expect("编码");
    let payload = &bytes[..Format::HEADER_BYTES + 1];
    let error = Snapshot::decode(&seal(payload)).expect_err("正文不够");
    let offset = Format::HEADER_BYTES + 1;
    assert!(
        error.to_string().contains(&format!("偏移 {offset}")),
        "{error}"
    );
}
