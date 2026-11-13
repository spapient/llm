//! 可选互操作：`serde` 的线上名字与 `bytemuck` 的受检字节视图。
//!
//! 两者都只在对应 feature 打开时编译；默认依赖集仍是零第三方 crate。名字必须与 `Display`
//! 逐字一致——配置、元数据与诊断里出现的是同一个词，不能有两套拼写。

#[cfg(feature = "serde")]
mod serde_wire_format {
    use llm_dtype::{DType, Element, MxType, QType};

    /// 序列化成 JSON 字符串后必须等于 `Display` 的规范名，且能原样读回。
    #[test]
    fn names_round_trip_through_json() {
        for qtype in QType::ALL {
            let text = serde_json::to_string(qtype).expect("格式身份必须可序列化");
            assert_eq!(text, format!("\"{qtype}\""));
            assert_eq!(serde_json::from_str::<QType>(&text).ok(), Some(*qtype));
        }
        for mx in MxType::ALL {
            let text = serde_json::to_string(mx).expect("格式身份必须可序列化");
            assert_eq!(text, format!("\"{mx}\""));
            assert_eq!(serde_json::from_str::<MxType>(&text).ok(), Some(*mx));
        }
        for dtype in DType::ALL {
            let text = serde_json::to_string(dtype).expect("格式身份必须可序列化");
            assert_eq!(text, format!("\"{dtype}\""));
            assert_eq!(serde_json::from_str::<DType>(&text).ok(), Some(*dtype));
        }
        for element in Element::ALL {
            let text = serde_json::to_string(element).expect("格式身份必须可序列化");
            assert_eq!(text, format!("\"{element}\""));
            assert_eq!(serde_json::from_str::<Element>(&text).ok(), Some(*element));
        }
    }

    /// 表里没有的名字必须在反序列化时失败，而不是落回某个默认值。
    #[test]
    fn unknown_names_are_rejected() {
        assert!(serde_json::from_str::<QType>("\"q4_9\"").is_err());
        assert!(serde_json::from_str::<DType>("\"i7\"").is_err());
        assert!(serde_json::from_str::<MxType>("\"mxfp3\"").is_err());
        assert!(serde_json::from_str::<Element>("\"e1m1\"").is_err());
    }
}

#[cfg(feature = "bytemuck")]
mod byte_view {
    use llm_dtype::{DType, Element, Format, MxType, QType};

    /// 写视图是一字节判别值，受检读视图能把它原样读回来。
    #[test]
    fn every_format_round_trips_through_its_single_byte() {
        for dtype in DType::ALL {
            let bytes = bytemuck::bytes_of(dtype);
            assert_eq!(bytes, &[*dtype as u8]);
            assert_eq!(
                bytemuck::checked::try_from_bytes::<DType>(bytes).ok(),
                Some(dtype)
            );
        }
        for qtype in QType::ALL {
            let bytes = bytemuck::bytes_of(qtype);
            assert_eq!(bytes, &[*qtype as u8]);
            assert_eq!(
                bytemuck::checked::try_from_bytes::<QType>(bytes).ok(),
                Some(qtype)
            );
        }
        for mx in MxType::ALL {
            let bytes = bytemuck::bytes_of(mx);
            assert_eq!(bytes, &[*mx as u8]);
            assert_eq!(
                bytemuck::checked::try_from_bytes::<MxType>(bytes).ok(),
                Some(mx)
            );
        }
        for element in Element::ALL {
            let bytes = bytemuck::bytes_of(element);
            assert_eq!(bytes, &[*element as u8]);
            assert_eq!(
                bytemuck::checked::try_from_bytes::<Element>(bytes).ok(),
                Some(element)
            );
        }
    }

    /// 闭集之外的每个字节都是非法位模式：受检读取必须失败，而不是造出一个不存在的格式。
    #[test]
    fn every_byte_outside_the_closed_set_is_rejected() {
        // 逐字节穷举 256 个位模式：抽样只测 0xFF 时，「除了 0xFF 全放行」的实现照样能过。
        fn assert_all_codes<F>()
        where
            F: Format + bytemuck::checked::CheckedBitPattern + Copy,
        {
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

        assert_all_codes::<DType>();
        assert_all_codes::<QType>();
        assert_all_codes::<MxType>();
        assert_all_codes::<Element>();
    }
}
