//! `snapshot` 的测试：往返、截断、篡改与"长度即承诺"的逐条对拍。

use super::*;
use tensor::Shape;
use weights::Entry;

fn facts() -> Facts {
    Facts::new(7, 2, 2, 3, 3).expect("恢复事实")
}

fn snapshot() -> Snapshot {
    let weights = Header::new(
        vec![
            Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([2, 3])).at(64),
            Entry::new("blk.0.scales", DType::U4, Shape::new([4])).at(88),
        ],
        64,
    )
    .expect("头部");
    Snapshot::new(facts(), weights)
}

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

/// 用任意正文重新封上校验和：只有测试会造"CRC 正确但正文有问题"的文件。
fn seal(payload: &[u8]) -> Vec<u8> {
    let mut bytes = payload.to_vec();
    bytes.extend_from_slice(&weights::crc::crc32(payload).to_le_bytes());
    bytes
}

fn push_varint(bytes: &mut Vec<u8>, value: u64) {
    varint(bytes, value);
}

fn payload_of(snapshot: &Snapshot) -> Vec<u8> {
    let bytes = snapshot.encode().expect("编码");
    bytes[..bytes.len() - Format::FOOTER_BYTES].to_vec()
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// 一整套承诺：长度是算准的、写进缓冲区的字节与 `encode` 一致、缓冲区多出来的部分不被碰、
/// 解码回来逐项相同、再编码逐字节相同。
fn assert_round_trip(snapshot: &Snapshot) {
    let bytes = snapshot.encode().expect("编码");
    assert_eq!(
        bytes.len(),
        snapshot.encoded_len().expect("长度"),
        "承诺的长度就是写出的长度"
    );

    let mut buffer = vec![0xAAu8; snapshot.encoded_len().expect("长度") + 16];
    let written = snapshot.encode_into(&mut buffer).expect("缓冲够大");
    assert_eq!(written, snapshot.encoded_len().expect("长度"));
    assert_eq!(&buffer[..written], bytes.as_slice());
    assert!(
        buffer[written..].iter().all(|byte| *byte == 0xAA),
        "缓冲区多出来的部分不能被碰"
    );

    let restored = Snapshot::decode(&bytes).expect("解码");
    assert_eq!(&restored, snapshot);
    assert_eq!(
        restored.encode().expect("编码"),
        bytes,
        "同一份快照只有一种写法"
    );
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
    let snapshot = Snapshot::new_with_moments(
        facts(),
        Header::new(Vec::new(), 0).expect("空头部"),
        moments,
    );
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");

    let restored_moments = restored.moments();
    assert_eq!(restored_moments.step(), 7, "优化器步数必须恢复");
    let before = snapshot.moments().slots();
    let after = restored_moments.slots();
    assert_eq!(before.len(), after.len(), "槽位数必须一致");
    for (index, (want, got)) in before.iter().zip(after).enumerate() {
        // 位级比较：f32 经小端字节往返不能只差一位。
        let want_m: Vec<u32> = want.m().iter().map(|value| value.to_bits()).collect();
        let got_m: Vec<u32> = got.m().iter().map(|value| value.to_bits()).collect();
        let want_v: Vec<u32> = want.v().iter().map(|value| value.to_bits()).collect();
        let got_v: Vec<u32> = got.v().iter().map(|value| value.to_bits()).collect();
        assert_eq!(got_m, want_m, "第 {index} 槽一阶矩逐位一致");
        assert_eq!(got_v, want_v, "第 {index} 槽二阶矩逐位一致");
    }
}

#[test]
fn accumulated_gradients_round_trip_bit_for_bit() {
    let accum = GradAccum::new(vec![vec![0.5, -0.25], vec![1.5]], 2);
    let snapshot = Snapshot::new_with_state(
        facts(),
        Header::new(Vec::new(), 0).expect("空头部"),
        Moments::empty(),
        accum,
    );
    let restored = Snapshot::decode(&snapshot.encode().expect("编码")).expect("解码");

    let got = restored.accum();
    assert_eq!(got.count(), 2, "窗口批数必须恢复");
    let before = snapshot.accum().slots();
    let after = got.slots();
    assert_eq!(before.len(), after.len(), "槽位数必须一致");
    for (index, (want, got)) in before.iter().zip(after).enumerate() {
        let want_bits: Vec<u32> = want.iter().map(|value| value.to_bits()).collect();
        let got_bits: Vec<u32> = got.iter().map(|value| value.to_bits()).collect();
        assert_eq!(got_bits, want_bits, "第 {index} 槽累积梯度逐位一致");
    }
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
    for (index, step) in steps.into_iter().enumerate() {
        let offset = 64 + index * 65_536;
        let header = Header::new(
            vec![Entry::new("w", DType::F32, Shape::new(vec![1, 127, 128, 16_384])).at(offset)],
            64,
        )
        .expect("头部");
        let size = index + 1;
        let facts = Facts::new(step, size, size, size, size).expect("事实");
        assert_round_trip(&Snapshot::new(facts, header));
    }
}

#[test]
fn a_snapshot_without_entries_still_round_trips() {
    let header = Header::new(Vec::new(), 0).expect("空头部");
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
        weights::crc::crc32(b"123456789"),
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
    assert_eq!(weights::crc::crc32(&bytes), !crc);
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

    let mut foreign = bytes.clone();
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

/// 手工拼一份正文：用来造编码器**永远不会**产生的字节（两条目同名、共享长度写得比实际
/// 少、布局明明相同却不写重复）。解码器的判据必须比编码器更强，所以测试要能绕过编码器
/// 直接说话。
///
/// 条目是 `(共享前缀, 共享后缀, 中段, 偏移, 布局标签)`；标签为 `k > 0` 时后面跟类型 `F32`
/// 与 `k - 1` 个值为 1 的维度。
fn handcrafted(entries: &[(u64, u64, &str, usize, u64)]) -> Vec<u8> {
    let mut payload = Format::MAGIC.to_vec();
    payload.extend_from_slice(&Format::VERSION.to_le_bytes());
    for value in [7, 1, 1, 1, 1, 64] {
        push_varint(&mut payload, value);
    }
    push_varint(&mut payload, entries.len() as u64);
    let mut previous = 0usize;
    for (front, back, middle, offset, layout) in entries {
        push_varint(&mut payload, *front);
        push_varint(&mut payload, *back);
        push_varint(&mut payload, middle.len() as u64);
        payload.extend_from_slice(middle.as_bytes());
        push_varint(&mut payload, *layout);
        if *layout > 0 {
            payload.push(tag(DType::F32).expect("f32 在快照标签闭集内"));
            for _ in 0..*layout - 1 {
                push_varint(&mut payload, 1);
            }
        }
        push_varint(&mut payload, (*offset - previous) as u64);
        previous = *offset;
    }
    // 矩状态段：0 个槽 + 步数 0（旧手改样本没有矩状态）。
    push_varint(&mut payload, 0);
    push_varint(&mut payload, 0);
    // 累积梯度段：窗口批数 0、0 个槽。
    push_varint(&mut payload, 0);
    push_varint(&mut payload, 0);
    seal(&payload)
}

/// 校验和挡得住"被改动"，挡不住"拿着正确工具重封了一个坏检查点"：条目表的判据必须在
/// 解码期也跑一遍，否则手改过的检查点会等到恢复后的第 N 步才炸。
#[test]
fn a_sealed_checkpoint_still_has_its_entry_table_validated() {
    // 名字都写全（共享长度 0）、布局按规则写、校验和也对，不合法的只剩"名字重复"。
    let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (3, 0, "", 80, 0)]);
    let error = Snapshot::decode(&bytes).expect_err("重名");
    assert!(error.to_string().contains("重复"), "{error}");
}

/// 共享是按**字节**算的：两条名字的公共前缀可以正好切在一个 UTF-8 字符中间。`块.0.注意_é`
/// 与 `块.0.注意_è` 只差最后一个字节，共享长度落在那个两字节字符的正中间——拼回来的整串
/// 仍然是合法 UTF-8，所以判 UTF-8 必须放在**拼接之后**，不能对每一段单独判。
#[test]
fn a_shared_prefix_may_cut_through_a_character() {
    let header = Header::new(
        vec![
            Entry::new("块.0.注意_é", DType::F32, Shape::new([1])).at(64),
            Entry::new("块.0.注意_è", DType::F32, Shape::new([1])).at(80),
        ],
        64,
    )
    .expect("头部");
    let snapshot = Snapshot::new(facts(), header);
    assert_round_trip(&snapshot);

    // 判据落在**名字块**上而不是整份文件：整份文件还含事实、布局、偏移与 CRC，
    // 拿文件总长去比名字字节总和永远不成立。
    let bytes = snapshot.encode().expect("编码");
    let payload = &bytes[..bytes.len() - Format::FOOTER_BYTES];
    let mut cursor = Cursor::new(payload, 0).expect("游标定位");
    cursor.take(4 + 2).expect("魔数与版本");
    for _ in 0..6 {
        // 五个恢复事实 + 权重数据段起点。
        cursor.varint().expect("头部事实");
    }
    assert_eq!(cursor.count("条目数").expect("条目数"), 2);

    // 第一条：名字全写，随后跳过布局与偏移。
    let (f1, b1, m1) = (
        cursor.count("前缀").expect("前缀"),
        cursor.count("后缀").expect("后缀"),
        cursor.count("中段").expect("中段长"),
    );
    let previous = cursor.take(m1).expect("第一条名字").to_vec();
    assert_eq!((f1, b1, m1), (0, 0, previous.len()));
    let layout = cursor.count("布局").expect("布局");
    cursor.u8().expect("类型标签");
    for _ in 0..layout - 1 {
        cursor.varint().expect("形状维度");
    }
    cursor.varint().expect("偏移");

    // 第二条：记录名字块实际占用的字节区间，再按前后缀拼回整串。
    let start = cursor.offset;
    let front = cursor.count("前缀").expect("前缀");
    let back = cursor.count("后缀").expect("后缀");
    let middle_len = cursor.count("中段").expect("中段长");
    let middle = cursor.take(middle_len).expect("第二条名字中段");
    let block_bytes = cursor.offset - start;

    let mut rebuilt = previous[..front].to_vec();
    rebuilt.extend_from_slice(middle);
    rebuilt.extend_from_slice(&previous[previous.len() - back..]);

    assert_eq!(front, 14, "公共前缀按字节算，切在两字节字符的中间字节");
    assert_eq!(back, 0);
    assert_eq!(middle_len, 1, "第二条名字只该写一个中段字节");
    assert_eq!(
        rebuilt,
        "块.0.注意_è".as_bytes(),
        "拼接后才判 UTF-8，整串合法"
    );
    let full_block = 3 + rebuilt.len();
    assert!(
        block_bytes < full_block,
        "共享后名字块 {block_bytes} 字节，全写要 {full_block} 字节"
    );
}

/// 共享长度只有一种写法：同一份快照不能既"全写"又"共享"，否则 `encode(decode(b)) == b`
/// 不成立，校验和也就不再等于"这份字节就是这个检查点"。
#[test]
fn a_non_maximal_name_share_is_rejected() {
    // `aaa` → `aab`：公共前缀是 2，写成 1（自留 `ab` 当中段）同样能拼出 `aab`。
    let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (1, 0, "ab", 80, 0)]);
    let error = Snapshot::decode(&bytes).expect_err("前缀少写");
    assert!(error.to_string().contains("最长"), "{error}");

    // 后缀同理：`aaa` → `baa` 的公共后缀是 2，只写 1 就必须多带一个 `a`。
    let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (0, 1, "ba", 80, 0)]);
    let error = Snapshot::decode(&bytes).expect_err("后缀少写");
    assert!(error.to_string().contains("最长"), "{error}");

    // 写满就过，拼回来的名字与直写完全一样。
    let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (2, 0, "c", 80, 0)]);
    assert_eq!(
        Snapshot::decode(&bytes)
            .expect("最长写法")
            .weights()
            .entries()[1]
            .name(),
        "aac"
    );
    // 空中段也合法：`aaa` 后面还是 `aaa` 时前后缀就已拼满（重名由 `Header` 拦下）。
    let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (3, 0, "", 80, 0)]);
    let error = Snapshot::decode(&bytes).expect_err("重名");
    assert!(error.to_string().contains("重复"), "{error}");
}

/// 布局同样只有一种写法：与上一条同类型同形状就必须写标签 `0`，否则同一份快照可以既有
/// "重复"写法又有"重写"写法。第一个条目没有上一条，标签 `0` 无从引用。
#[test]
fn a_repeated_layout_must_be_written_as_a_repeat() {
    let bytes = handcrafted(&[(0, 0, "a", 64, 2), (0, 0, "b", 80, 2)]);
    let error = Snapshot::decode(&bytes).expect_err("同布局重写");
    assert!(error.to_string().contains("布局"), "{error}");

    let bytes = handcrafted(&[(0, 0, "a", 64, 0), (0, 0, "b", 80, 0)]);
    let error = Snapshot::decode(&bytes).expect_err("第一个条目");
    assert!(error.to_string().contains("第一个条目"), "{error}");

    // 标签 2 = 秩 1 的形状 `[1]`：两条都这么写就合法，条目各自拿到同一个形状。
    let bytes = handcrafted(&[(0, 0, "a", 64, 2), (0, 0, "b", 80, 0)]);
    let restored = Snapshot::decode(&bytes).expect("重复布局");
    assert_eq!(restored.weights().entries()[1].shape().dims(), [1]);
    assert_eq!(restored.weights().entries()[1].dtype(), DType::F32);
}

#[test]
fn trailing_bytes_after_the_entry_table_are_rejected() {
    let mut payload = payload_of(&snapshot());
    payload.push(0);
    let error = Snapshot::decode(&seal(&payload)).expect_err("尾部多余");
    assert!(error.to_string().contains("尾部多出 1 字节"), "{error}");
}

/// 一个值只能有一种写法，否则"同一份快照只有一种字节形态"就不成立。
#[test]
fn a_non_canonical_integer_is_rejected() {
    let mut payload = payload_of(&snapshot());
    assert_eq!(
        payload[Format::HEADER_BYTES],
        7,
        "步号 7 的正常写法是单字节"
    );
    payload[Format::HEADER_BYTES] = 0x87;
    payload.insert(Format::HEADER_BYTES + 1, 0x00);
    let error = Snapshot::decode(&seal(&payload)).expect_err("非规范");
    assert!(error.to_string().contains("规范"), "{error}");
}

#[test]
fn an_integer_wider_than_u64_is_rejected() {
    let mut payload = snapshot().encode().expect("编码")[..Format::HEADER_BYTES].to_vec();
    payload.extend_from_slice(&[0xFF; 9]);
    payload.push(0x02);
    let error = Snapshot::decode(&seal(&payload)).expect_err("超宽");
    assert!(error.to_string().contains("u64"), "{error}");
}

/// 先对账再分配：伪造的条目数不能换来一次大分配。
#[test]
fn an_entry_count_beyond_the_bytes_is_rejected_before_allocating() {
    let mut payload = snapshot().encode().expect("编码")[..Format::HEADER_BYTES].to_vec();
    for value in [0u64, 1, 1, 1, 1, 0] {
        push_varint(&mut payload, value);
    }
    push_varint(&mut payload, 1_000_000);
    let error = Snapshot::decode(&seal(&payload)).expect_err("计数超限");
    assert!(error.to_string().contains("权重条目"), "{error}");
}

#[test]
fn a_checkpoint_without_layers_is_rejected() {
    for (world, layers, rows, cols) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        let error = Facts::new(0, world, layers, rows, cols).expect_err("零值");
        assert_eq!(error.code(), Code::Invalid);
    }
}

#[test]
fn a_short_buffer_is_refused_not_silently_filled() {
    let snapshot = snapshot();
    let mut buffer = vec![0u8; snapshot.encoded_len().expect("长度") - 1];
    let error = snapshot.encode_into(&mut buffer).expect_err("放不下");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("需要"), "{error}");
}

/// 随机条目表（名字、秩、维度、间隔各不同）压一遍长度与往返承诺。
#[test]
fn random_snapshots_keep_the_length_promise_and_the_bytes() {
    let mut state = 0x2545_F491_4F6C_DD1D;
    for round in 0..200u64 {
        let count = (lcg(&mut state) % 8) as usize;
        let mut entries = Vec::new();
        let mut offset = 64usize;
        for index in 0..count {
            let rank = (lcg(&mut state) % 4) as usize;
            let dims: Vec<usize> = (0..rank)
                .map(|_| (lcg(&mut state) % 4096) as usize)
                .collect();
            let bytes = dims.iter().product::<usize>() * 4;
            offset += (lcg(&mut state) % 512) as usize;
            entries.push(
                Entry::new(
                    format!("blk.{round}.w{index}"),
                    DType::F32,
                    Shape::new(dims),
                )
                .at(offset),
            );
            offset += bytes.max(1);
        }
        let header = Header::new(entries, 64).expect("头部");
        let size = |state: &mut u64| 1 + (lcg(state) % 64) as usize;
        let facts = Facts::new(
            lcg(&mut state),
            size(&mut state),
            size(&mut state),
            size(&mut state),
            size(&mut state),
        )
        .expect("事实");
        assert_round_trip(&Snapshot::new(facts, header));
    }
}

/// 写盘是边编码边写的：超过写缓冲的检查点必须逐字节对得上——这一条走的是 `BufWriter`
/// 真正的刷写路径，而不是"先编进一块内存再拷贝"。
///
/// 顺带锁住名字的前后缀共享：同一种名字写 4096 遍，字节数必须**远小于**把每个名字整串写下
/// 去的体积，否则共享在某一轮优化里被悄悄关掉，只有文件变大而测试仍是绿的。
#[test]
fn a_snapshot_larger_than_the_write_buffer_lands_byte_for_byte() {
    let mut entries = Vec::new();
    let mut offset = 4096usize;
    let mut plain = 0usize;
    for layer in 0..4096 {
        let name = format!("blk.{layer}.attn_qkv.weight");
        plain += name.len();
        let entry = Entry::new(name, DType::F16, Shape::new([64, 64])).at(offset);
        offset += entry.len();
        entries.push(entry);
    }
    let header = Header::new(entries, 4096).expect("头部");
    let snapshot = Snapshot::new(facts(), header);
    assert!(
        snapshot.encoded_len().expect("长度") > WRITE_BUFFER,
        "这一条必须真的跨过写缓冲"
    );
    assert!(
        snapshot.encoded_len().expect("长度") < plain,
        "名字共享必须省下字节：{} 字节 vs 名字整串写下的 {plain} 字节",
        snapshot.encoded_len().expect("长度")
    );

    let dir = testkit::scratch_clean("stream");
    let path = dir.join("step-7.llmc");
    snapshot.write(&path).expect("写");
    let meta = std::fs::metadata(&path).expect("文件存在");
    assert_eq!(meta.len() as usize, snapshot.encoded_len().expect("长度"));
    assert_eq!(
        std::fs::read(&path).expect("字节"),
        snapshot.encode().expect("编码")
    );
    assert_eq!(Snapshot::read(&path).expect("读"), snapshot);

    let names: Vec<String> = std::fs::read_dir(&dir)
        .expect("目录")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["step-7.llmc"], "跨缓冲刷写也不留临时文件");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_then_read_restores_the_snapshot_and_leaves_no_residue() {
    let dir = testkit::scratch_clean("write");
    let path = dir.join("step-7.llmc");
    let snapshot = snapshot();
    snapshot.write(&path).expect("写");

    let meta = std::fs::metadata(&path).expect("文件存在");
    assert_eq!(
        meta.len() as usize,
        snapshot.encoded_len().expect("长度"),
        "写下的字节数就是承诺的字节数"
    );
    assert_eq!(Snapshot::read(&path).expect("读"), snapshot);

    let names: Vec<String> = std::fs::read_dir(&dir)
        .expect("目录")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["step-7.llmc"], "成功后不留临时文件");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_checkpoint_is_missing_and_a_failed_write_is_io() {
    let dir = testkit::scratch_clean("io");
    let error = Snapshot::read(&dir.join("absent.llmc")).expect_err("不存在");
    assert_eq!(error.code(), Code::Missing);

    let error = snapshot()
        .write(&dir.join("no-such-dir/step.llmc"))
        .expect_err("目录不存在");
    assert_eq!(error.code(), Code::Io);
    assert!(error.to_string().contains("写入"), "{error}");
    let names: Vec<String> = std::fs::read_dir(&dir)
        .expect("目录")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(names.is_empty(), "失败不留残骸：{names:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupted_file_names_its_path() {
    let dir = testkit::scratch_clean("corrupt");
    let path = dir.join("step-7.llmc");
    snapshot().write(&path).expect("写");

    let mut bytes = std::fs::read(&path).expect("读");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("覆盖");

    let error = Snapshot::read(&path).expect_err("校验和不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("step-7.llmc"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 版本 3 布局的固定字节黄金向量：往返测试只能证明"编码与解码对称"，布局若被对称地改动它
/// 察觉不到；黄金向量把当前字段顺序、类型标签与压缩写法钉死。任何不是格式版本递增的布局漂移，
/// 都会在这里逐字节失败。
#[test]
fn a_golden_vector_pins_the_version_four_layout() {
    const GOLDEN: [u8; 29] = [
        0x4C, 0x4C, 0x4D, 0x43, // `LLMC`
        0x04, 0x00, // 版本 4（小端 u16）
        0x01, 0x01, 0x01, 0x01, 0x01, // step=1、world=1、layers=1、rows=1、cols=1
        0x00, // data_start=0
        0x01, // 条目数 1
        0x00, 0x00, 0x01, 0x77, // 名：前 0 后 0，中段 1 字节 `w`
        0x02, 0x00, 0x01, // 布局：秩+1=2，F32 标签 00，秩 1
        0x00, // 偏移增量 0
        0x00, 0x00, // 矩状态：0 槽、步数 0
        0x00, 0x00, // 累积梯度：窗口 0 批、0 槽
        0xFF, 0xF0, 0x7A, 0xE0, // CRC-32（小端）
    ];

    let facts = Facts::new(1, 1, 1, 1, 1).expect("事实");
    let header =
        Header::new(vec![Entry::new("w", DType::F32, Shape::new([1])).at(0)], 0).expect("头部");
    let snapshot = Snapshot::new(facts, header);

    let bytes = snapshot.encode().expect("编码");
    assert_eq!(
        bytes.as_slice(),
        &GOLDEN,
        "布局改动必须先递增格式版本并更新黄金向量"
    );

    Snapshot::verify(&GOLDEN).expect("黄金向量自带校验和");
    let restored = Snapshot::decode(&GOLDEN).expect("解码黄金向量");
    assert_eq!(restored, snapshot);
}
