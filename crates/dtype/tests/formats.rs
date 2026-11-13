//! 泛型契约：`Format` / `BlockFormat` / `Layout<B>` 是四个格式闭集的唯一读法。
//!
//! 这里用同一批泛型断言跑遍 `DType`、`QType`、`MxType` 与 `Element`——「加一种格式」不必在每个
//! 地方各补一份几何测试，也不必各写一遍「判别值即下标、名字往返、块数向上取整」。同时把 trait
//! 视角与各枚举的 inherent `const fn` 逐项对拍：两条路径读同一张 `SPECS`，一方漂移即失败。

use llm_dtype::{BlockFormat, DType, Element, Format, Layout, MxType, QType};

/// 任何闭集：判别值即下标、名字唯一且能往返。
fn assert_closed_set<F: Format>() {
    assert!(
        F::ALL.len() <= usize::from(u8::MAX) + 1,
        "判别值是 u8，闭集最多 256 个成员"
    );
    for (index, format) in F::ALL.iter().enumerate() {
        assert_eq!(format.index(), index, "{format:?} 的判别值必须等于行表下标");
        assert_eq!(
            F::parse_name(format.name()),
            Some(*format),
            "{format:?} 的规范名不能往返"
        );
    }
    let mut names: Vec<&str> = F::ALL.iter().map(|format| format.name()).collect();
    let total = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), total, "规范名必须唯一（解析不能有歧义）");
}

/// 标准 trait：`FromStr` 必须转调同一张行表，表外名字必须失败而不是落回某个默认值。
fn assert_from_str<F>(unknown: &str)
where
    F: Format + core::str::FromStr + PartialEq + core::fmt::Debug,
{
    for format in F::ALL {
        assert_eq!(
            format.name().parse::<F>().ok().as_ref(),
            Some(format),
            "{format:?} 的 `FromStr` 与规范名分家"
        );
    }
    assert!(unknown.parse::<F>().is_err(), "`{unknown}` 不在闭集里");
}

/// 块格式：块数向上取整、整块字节等于 `块数 × 每块字节`，且两条视角必须一致。
fn assert_block_format<F: BlockFormat>() {
    assert_closed_set::<F>();
    for format in F::ALL {
        let len = u64::from(format.block_len());
        assert!(len > 0, "{format:?} 的块长不能为 0");
        assert_eq!(format.block_count(0), 0, "{format:?} 的空输入不占块");
        assert_eq!(format.block_count(len), 1, "{format:?} 的恰好一块");
        assert_eq!(format.block_count(len + 1), 2, "{format:?} 的跨块补齐");
        assert_eq!(format.bytes_for(0), 0, "{format:?} 的空输入不占字节");
        assert_eq!(
            format.bytes_for(len),
            u64::from(format.block_bytes()),
            "{format:?} 的整块字节"
        );
        assert_eq!(
            format.bytes_for(len - 1),
            u64::from(format.block_bytes()),
            "{format:?} 的不足一块按整块计"
        );
        assert_eq!(
            format.bytes_for(len * 3),
            3 * u64::from(format.block_bytes()),
            "{format:?} 的三块字节"
        );
        // 饱和而不是回绕：不受信任的 `numel` 最多把字节数报成同一量级，绝不能变小学生。
        assert!(
            format.bytes_for(u64::MAX) >= format.bytes_for(u64::MAX / 2),
            "{format:?} 的字节数在溢出时必须饱和"
        );
    }
}

#[test]
fn block_tables_share_one_contract() {
    assert_block_format::<QType>();
    assert_block_format::<MxType>();
}

#[test]
fn every_closed_set_obeys_the_identity_contract() {
    // 四个闭集走同一套断言：加一个元素、一个 MX 格式或一个标量类型不会只在其中一处补测试。
    assert_closed_set::<Element>();
}

#[test]
fn from_str_forwards_to_the_name_table() {
    // 四个闭集各测一次：`Display` 与 `FromStr` 是同一张表的两个方向，中间不能再来一份清单。
    assert_from_str::<DType>("i7");
    assert_from_str::<QType>("q4_9");
    assert_from_str::<MxType>("mxfp3");
    assert_from_str::<Element>("e1m1");
}

/// 字节数对元素数**单调不减**，且极大输入饱和而不是回绕成小值。
fn assert_bytes_are_monotonic<F: BlockFormat>() {
    for format in F::ALL {
        let mut previous = 0;
        for numel in (0..=4096u64).step_by(37) {
            let bytes = format.bytes_for(numel);
            assert!(
                bytes >= previous,
                "{format:?} 在 {numel} 个元素处字节数倒退：{bytes} < {previous}"
            );
            previous = bytes;
        }
        // 不受信任的 `numel` 到顶时，饱和值不能小于半量级，更不能回绕成十几字节。
        let huge = format.bytes_for(u64::MAX);
        assert!(
            huge >= format.bytes_for(u64::MAX / 2),
            "{format:?} 溢出未饱和"
        );
        assert!(
            huge >= u64::from(format.block_bytes()),
            "{format:?} 溢出回绕"
        );
    }
}

#[test]
fn byte_counts_are_monotonic_and_saturate() {
    assert_bytes_are_monotonic::<QType>();
    assert_bytes_are_monotonic::<MxType>();
    // 标量轴：同样单调、同样饱和（亚字节类型不例外）。
    let mut previous = 0;
    for numel in (0..=4096u64).step_by(37) {
        let bytes = DType::F4.bytes_for(numel);
        assert!(bytes >= previous);
        previous = bytes;
    }
    // 乘法先饱和到 `u64::MAX`，再按位向上取整；关键是不回绕成小数。
    assert_eq!(DType::F4.bytes_for(u64::MAX), u64::MAX.div_ceil(8));
}

#[test]
fn the_scalar_table_obeys_the_same_identity_contract() {
    assert_closed_set::<DType>();
    // 亚字节类型按位跨元素打包：8 个元素正好 `bits` 字节（`U4` 为 4、`Bool` 为 1）。
    for dtype in DType::ALL {
        assert_eq!(u64::from(dtype.bits()), dtype.bytes_for(8), "{dtype:?}");
    }
}

#[test]
fn the_trait_view_matches_the_const_accessors() {
    // trait 是泛型视角、inherent `const fn` 是常量视角；同一张 `SPECS` 读出同一个答案。
    for qtype in QType::ALL {
        assert_eq!(Format::index(*qtype), *qtype as usize);
        assert_eq!(Format::name(*qtype), qtype.name());
        assert_eq!(Format::bits(*qtype), qtype.bits());
        assert_eq!(BlockFormat::block_len(*qtype), qtype.block_len());
        assert_eq!(BlockFormat::block_bytes(*qtype), qtype.block_bytes());
        assert_eq!(
            BlockFormat::block_count(*qtype, 300),
            qtype.block_count(300)
        );
        assert_eq!(BlockFormat::bytes_for(*qtype, 300), qtype.bytes_for(300));
    }
    for mx in MxType::ALL {
        assert_eq!(Format::index(*mx), *mx as usize);
        assert_eq!(Format::name(*mx), mx.name());
        assert_eq!(Format::bits(*mx), mx.bits());
        assert_eq!(BlockFormat::block_len(*mx), mx.block_len());
        assert_eq!(BlockFormat::block_bytes(*mx), mx.block_bytes());
        assert_eq!(BlockFormat::block_count(*mx, 300), mx.block_count(300));
        assert_eq!(BlockFormat::bytes_for(*mx, 300), mx.bytes_for(300));
    }
    for dtype in DType::ALL {
        assert_eq!(Format::index(*dtype), *dtype as usize);
        assert_eq!(Format::name(*dtype), dtype.name());
        assert_eq!(Format::bits(*dtype), dtype.bits());
    }
    for element in Element::ALL {
        assert_eq!(Format::index(*element), *element as usize);
        assert_eq!(Format::name(*element), element.name());
        assert_eq!(Format::bits(*element), element.bits());
    }
}

#[test]
fn layouts_carry_the_same_geometry_for_both_arms() {
    let scalar: Layout<QType> = Layout::Scalar(DType::F16);
    assert_eq!(scalar.elems(), 1);
    assert_eq!(scalar.span(), 2);
    assert_eq!(scalar.bytes_for(4), 8);

    let block: Layout<QType> = Layout::Block(QType::Q4_K);
    assert_eq!(block.elems(), 256);
    assert_eq!(block.span(), 144);
    assert_eq!(block.bytes_for(256), 144);
    assert_eq!(block.bytes_for(257), 288, "多一个元素也要多一整块");

    // 同一个泛型布局也能装 MX 格式：接 MXFP4/NVFP4 解码时不必另写一份 `elems`/`span`。
    let micro: Layout<MxType> = Layout::Block(MxType::NvFp4);
    assert_eq!((micro.elems(), micro.span()), (16, 9));
    assert_eq!(micro.bytes_for(64), 36);

    // 亚字节标量：`span()` 是「一个元素占几个字节」，与按位打包的 `bytes_for` 不是同一件事，
    // 所以标量手臂不实现 `BlockFormat`。这条断言把那个差别固定下来，避免有人按 `span × n` 算。
    let packed: Layout<QType> = Layout::Scalar(DType::U4);
    assert_eq!((packed.elems(), packed.span()), (1, 1));
    assert_eq!(packed.bytes_for(8), 4, "8 个 U4 只占 4 字节");
    assert_eq!(
        packed.bytes_for(u64::MAX),
        u64::MAX.div_ceil(8),
        "饱和后按位取整"
    );
}
