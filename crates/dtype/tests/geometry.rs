//! 块几何契约：块数向上取整、字节数随元素数单调不减且在溢出时饱和，标量与块两条手臂
//! 共用同一份 [`Layout`]。
//!
//! 断言只依赖公开的几何方法与 `ALL`，不重算实现里的公式：单调性与饱和是**性质**，
//! 具体字节数由 `src/*.rs` 的行表逐格式锁定。公式改写成等价形式时测试不应误报。

use llm_dtype::{BlockFormat, DType, Layout, MxType, QType};

/// 块格式：整块向上取整，`bytes_for` 永远是「块数 × 每块字节」。
fn assert_whole_blocks<F: BlockFormat>() {
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
    }
}

/// 字节数单调不减；极大输入饱和而不是回绕成小值。
fn assert_monotonic_and_saturating<F: BlockFormat>() {
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
fn block_formats_count_whole_blocks() {
    assert_whole_blocks::<QType>();
    assert_whole_blocks::<MxType>();
}

#[test]
fn the_block_trait_forwards_to_the_const_accessors() {
    // trait 只转发；两条路径必须给出同一份几何。
    for qtype in QType::ALL {
        assert_eq!(BlockFormat::block_len(*qtype), qtype.block_len());
        assert_eq!(BlockFormat::block_bytes(*qtype), qtype.block_bytes());
        assert_eq!(
            BlockFormat::block_count(*qtype, 300),
            qtype.block_count(300)
        );
        assert_eq!(BlockFormat::bytes_for(*qtype, 300), qtype.bytes_for(300));
    }
    for mx in MxType::ALL {
        assert_eq!(BlockFormat::block_len(*mx), mx.block_len());
        assert_eq!(BlockFormat::block_bytes(*mx), mx.block_bytes());
        assert_eq!(BlockFormat::block_count(*mx, 300), mx.block_count(300));
        assert_eq!(BlockFormat::bytes_for(*mx, 300), mx.bytes_for(300));
    }
}

#[test]
fn byte_counts_are_monotonic_and_saturate() {
    assert_monotonic_and_saturating::<QType>();
    assert_monotonic_and_saturating::<MxType>();
    // 标量轴同样单调、同样饱和：全部 24 种都跑一遍，而不只抽一个亚字节类型。
    for dtype in DType::ALL {
        let mut previous = 0;
        for numel in (0..=4096u64).step_by(37) {
            let bytes = dtype.bytes_for(numel);
            assert!(bytes >= previous, "{dtype:?} 在 {numel} 个元素处字节数倒退");
            previous = bytes;
        }
        assert_eq!(
            dtype.bytes_for(u64::MAX),
            u64::MAX.div_ceil(8),
            "{dtype:?} 的溢出必须饱和到同一个上界"
        );
    }
}

#[test]
fn sub_byte_scalars_pack_across_element_boundaries() {
    // 字面值而不是用 `bits` 重算公式：`U4` 两个元素共用一个字节，`Bool`/`U1` 按位打包，
    // 整字节类型不额外补齐。
    assert_eq!((DType::U1.bytes_for(8), DType::U1.bytes_for(9)), (1, 2));
    assert_eq!((DType::U4.bytes_for(1), DType::U4.bytes_for(2)), (1, 1));
    assert_eq!(DType::U4.bytes_for(3), 2);
    assert_eq!((DType::I2.bytes_for(3), DType::I2.bytes_for(4)), (1, 1));
    assert_eq!((DType::F4.bytes_for(1), DType::F4.bytes_for(8)), (1, 4));
    assert_eq!((DType::Bool.bytes_for(8), DType::Bool.bytes_for(9)), (1, 2));
    assert_eq!(DType::U8.bytes_for(1), 1);
    assert_eq!(DType::F32.bytes_for(3), 12);
    assert_eq!(DType::F64.bytes_for(2), 16);
}

#[test]
fn layout_carries_the_geometry_of_both_arms() {
    let scalar: Layout<QType> = Layout::Scalar(DType::F16);
    assert_eq!((scalar.elems(), scalar.span()), (1, 2));
    assert_eq!(scalar.bytes_for(4), 8);

    let block: Layout<QType> = Layout::Block(QType::Q4_K);
    assert_eq!((block.elems(), block.span()), (256, 144));
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
