//! `snapshot` 的规范编码与落盘：解码器判据强于编码器、手工拼字节、黄金向量与写盘对账。

use super::snapshot_support::*;
/// 校验和挡得住"被改动"，挡不住"拿着正确工具重封了一个坏检查点"：条目表的判据必须在
/// 解码期也跑一遍，否则手改过的检查点会等到恢复后的第 N 步才炸。
#[test]
fn a_sealed_checkpoint_still_has_its_entry_table_validated() {
    // 名字都写全（共享长度 0）、布局按规则写、校验和也对，不合法的只剩"名字重复"。
    reject_duplicate_name(handcrafted(&[(0, 0, "aaa", 64, 2), (3, 0, "", 80, 0)]));
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
    reject_duplicate_name(handcrafted(&[(0, 0, "aaa", 64, 2), (3, 0, "", 80, 0)]));
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
fn a_short_buf_is_refused_not_silently_filled() {
    let snapshot = snapshot();
    let mut buf = vec![0u8; snapshot.encoded_len().expect("长度") - 1];
    let error = snapshot.encode_into(&mut buf).expect_err("放不下");
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
        for idx in 0..count {
            let rank = (lcg(&mut state) % 4) as usize;
            let dims: Vec<usize> = (0..rank)
                .map(|_| (lcg(&mut state) % 4096) as usize)
                .collect();
            let bytes = dims.iter().product::<usize>() * 4;
            offset += (lcg(&mut state) % 512) as usize;
            entries.push(
                Entry::new(format!("blk.{round}.w{idx}"), DType::F32, Shape::new(dims)).at(offset),
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
fn a_snapshot_larger_than_the_write_buf_lands_byte_for_byte() {
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
        snapshot.encoded_len().expect("长度") > WRITE_BUF,
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
    assert_eq!(
        std::fs::read(&path).expect("字节"),
        snapshot.encode().expect("编码")
    );
    let names = assert_written_snapshot(&dir, &path, &snapshot);
    assert_eq!(names, ["step-7.llmc"], "跨缓冲刷写也不留临时文件");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_then_read_restores_the_snapshot_and_leaves_no_residue() {
    let dir = testkit::scratch_clean("write");
    let path = dir.join("step-7.llmc");
    let snapshot = snapshot();
    snapshot.write(&path).expect("写");

    let names = assert_written_snapshot(&dir, &path, &snapshot);
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
    let names = dir_entry_names(&dir);
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

/// 版本 5 布局的固定字节黄金向量：往返测试只能证明"编码与解码对称"，布局若被对称地改动它
/// 察觉不到；黄金向量把当前字段顺序、类型标签与压缩写法钉死。任何不是格式版本递增的布局漂移，
/// 都会在这里逐字节失败。
#[test]
fn a_golden_vector_pins_the_version_five_layout() {
    const GOLDEN: [u8; 30] = [
        0x4C, 0x4C, 0x4D, 0x43, // `LLMC`
        0x05, 0x00, // 版本 5（小端 u16）
        0x01, 0x01, 0x01, 0x01, 0x01, // step=1、world=1、layers=1、rows=1、cols=1
        0x00, // data_start=0
        0x01, // 条目数 1
        0x00, 0x00, 0x01, 0x77, // 名：前 0 后 0，中段 1 字节 `w`
        0x02, 0x00, 0x01, // 布局：秩+1=2，F32 标签 00，秩 1
        0x00, // 偏移增量 0
        0x00, 0x00, // 矩状态：0 槽、步数 0
        0x00, 0x00, // 累积梯度：窗口 0 批、0 槽
        0x00, // 权重绑定：0 对
        0x30, 0x1E, 0xC8, 0xE8, // CRC-32（小端）
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
