//! proptest 属性层：与确定性 sweep / libfuzzer 互补的随机生成 + 收缩。
//!
//! libfuzzer 靠覆盖反馈探索单条输入，sweep 靠在合法种子边角上扰动；两者都很难系统性遍历
//! "形状 × 元素类型"、"任意字节 × 任意切点" 这类组合空间。这一层用 proptest 直接生成结构化
//! 输入，并用收缩给出最小反例，把 Burn 式属性测试作为第三道保险叠在契约与模糊之上。
//!
//! 与 `drivers` 的关系：`drivers` 把同一批不变量做成吃任意字节的 `Target`，这里把它们写成
//! 带类型的属性。两份判据不共享实现是有意的——独立实现才可能互相发现错。

use proptest::prelude::*;

proptest! {
    #![proptest_config(testkit::prop::config())]

    /// LEB128 写→读对任意 `u64` 恒等，且只消费写出的全部字节。
    #[test]
    fn leb_roundtrip(value: u64) {
        let mut out = Vec::new();
        storage::leb::write(&mut out, value);
        let mut cursor = storage::leb::Cursor::new(&out);
        let decoded: u64 = cursor.read().expect("规范编码必须可解");
        prop_assert_eq!(decoded, value);
        prop_assert!(cursor.remaining().is_empty());
    }

    /// CRC-32 分段续算与整段一致：任意字节、任意切点。
    #[test]
    fn crc_split_equals_whole(bytes: Vec<u8>) {
        let whole = storage::crc::crc32(&bytes);
        for split in [0, bytes.len() / 2, bytes.len()] {
            let (left, right) = bytes.split_at(split);
            let chained = !storage::crc::update(storage::crc::update(u32::MAX, left), right);
            prop_assert_eq!(chained, whole);
        }
    }

    /// 形状算术：`numel` 是轴长的饱和乘积，`element_bytes` 按位向上取整。
    #[test]
    fn shape_arith_matches_reference(
        dims in prop::collection::vec(any::<u16>(), 0..=8),
        dtype in prop::sample::select(tensor::DType::ALL),
    ) {
        let dims: Vec<usize> = dims.into_iter().map(usize::from).collect();
        let shape = tensor::Shape::new(dims.clone());

        let expected_numel = dims
            .iter()
            .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128));
        let expected_numel = usize::try_from(expected_numel).unwrap_or(usize::MAX);
        prop_assert_eq!(shape.numel(), expected_numel);

        let total_bits = (expected_numel as u64).saturating_mul(u64::from(dtype.bits()));
        let expected_bytes = usize::try_from(total_bits.div_ceil(8)).unwrap_or(usize::MAX);
        prop_assert_eq!(shape.element_bytes(dtype), expected_bytes);
    }
}

/// 元素类型 code 是 `1..=24` 的闭集：往返恒等、位宽为正、`0` 不分配。
#[test]
fn dtype_codes_are_a_closed_set() {
    for &dtype in tensor::DType::ALL {
        let code = dtype.code();
        assert_ne!(code, 0, "code 0 保留不分配");
        assert_eq!(tensor::DType::from_code(code), Some(dtype));
        assert!(dtype.bits() > 0, "{dtype:?} 的位宽应当为正");
    }
    assert_eq!(tensor::DType::from_code(0), None);
}

/// 能力协商：内置声明 × 档案的全笛卡尔积里，凡成功者都必须回指到真实变体。
///
/// 这条不靠随机：内置表是编译期内嵌的有限集合，穷举才是完整覆盖。
#[test]
fn negotiate_keeps_variant_contract_over_builtin_catalogue() {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置声明/档案表必须可解析");
    assert!(!catalogue.ops().is_empty(), "内置算子声明不应为空");
    assert!(!catalogue.profiles().is_empty(), "内置档案不应为空");
    for op in catalogue.ops() {
        for profile in catalogue.profiles() {
            if let Ok(path) = cutile::cap::negotiate(op, profile) {
                let hit = op
                    .variants
                    .get(path.idx)
                    .is_some_and(|variant| variant.name == path.variant);
                assert!(
                    hit,
                    "算子 `{}` × 档案 `{}`：协商变体 `{}` (idx {}) 无法回指",
                    op.name, profile.meta.vendor, path.variant, path.idx,
                );
            }
        }
    }
}
