//! 可选互操作：`serde` 的线上名字与 `bytemuck` 的受检字节视图。
//!
//! 两者都只在对应 feature 打开时编译；默认依赖集仍是零第三方 crate。名字必须与 `Display`
//! 逐字一致——配置、元数据与诊断里出现的是同一个词，不能有两套拼写。
//!
//! 断言写在泛型 helper 里而不是每个闭集抄一遍：`serde`/`bytemuck` 对四个闭集的要求完全一样，
//! 新增一个闭集只是多一行调用。

#[cfg(feature = "serde")]
mod serde_wire_format {
    use llm_dtype::{DType, Element, Format, MxType, QType};
    use serde::Serialize;
    use serde::de::DeserializeOwned;

    /// 序列化成 JSON 后必须等于 `Display` 的规范名、能原样读回，表外名字必须失败。
    fn assert_json_names<F>(unknown: &str)
    where
        F: Format + Serialize + DeserializeOwned,
    {
        for format in F::ALL {
            let text = serde_json::to_string(format).expect("格式身份必须可序列化");
            assert_eq!(
                text,
                format!("\"{format}\""),
                "{format:?} 的线上名字与 `Display` 分家"
            );
            assert_eq!(
                serde_json::from_str::<F>(&text).ok(),
                Some(*format),
                "{format:?} 不能从 JSON 读回"
            );
        }
        assert!(
            serde_json::from_str::<F>(&format!("\"{unknown}\"")).is_err(),
            "`{unknown}` 不在闭集里，反序列化必须失败而不是落回某个默认值"
        );
    }

    #[test]
    fn names_round_trip_through_json() {
        assert_json_names::<DType>("i7");
        assert_json_names::<QType>("q4_9");
        assert_json_names::<MxType>("mxfp3");
        assert_json_names::<Element>("e1m1");
    }
}

#[cfg(feature = "bytemuck")]
mod byte_view {
    use llm_dtype::{DType, Element, Format, MxType, QType};

    /// 写视图是一字节判别值（即行表下标），受检读视图能把它原样读回；闭集之外的每个字节
    /// 都是非法位模式。
    fn assert_single_byte<F>()
    where
        F: Format + bytemuck::NoUninit + bytemuck::checked::CheckedBitPattern + Copy,
    {
        for format in F::ALL {
            let bytes = bytemuck::bytes_of(format);
            let code = u8::try_from(format.index()).expect("闭集远小于 256 个变体");
            assert_eq!(bytes, &[code], "{format:?} 写出的不是一字节判别值");
            assert_eq!(
                bytemuck::checked::try_from_bytes::<F>(bytes).ok().copied(),
                Some(*format),
                "{format:?} 不能从自己的字节读回"
            );
        }

        // 逐字节穷举 256 个位模式：抽样只测 0xFF 时，「除了 0xFF 全放行」的实现照样能过。
        for code in u8::MIN..=u8::MAX {
            let parsed = bytemuck::checked::try_from_bytes::<F>(&[code])
                .ok()
                .copied();
            match parsed {
                Some(format) => {
                    assert_eq!(
                        Format::index(format),
                        usize::from(code),
                        "{code} 读成了别的格式"
                    );
                    assert!(
                        usize::from(code) < F::ALL.len(),
                        "{code} 在闭集之外却被接受"
                    );
                }
                None => assert!(usize::from(code) >= F::ALL.len(), "{code} 在闭集之内却被拒"),
            }
        }
    }

    #[test]
    fn every_format_round_trips_through_its_single_byte() {
        assert_single_byte::<DType>();
        assert_single_byte::<QType>();
        assert_single_byte::<MxType>();
        assert_single_byte::<Element>();
    }
}
