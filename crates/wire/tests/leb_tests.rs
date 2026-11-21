    use super::*;

    #[test]
    fn integers_round_trip_across_boundaries() {
        for value in [0u64, 1, 127, 128, 16_383, 16_384, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            write(&mut buf, value);
            assert_eq!(Cursor::new(&buf).read::<u64>().expect("规范"), value);
        }
    }

    #[test]
    fn every_unsigned_width_encodes() {
        let mut buf = Vec::new();
        write(&mut buf, 7u8);
        write(&mut buf, 300u16);
        write(&mut buf, 40_000u32);
        write(&mut buf, 9_000_000_000u64);
        write(&mut buf, 5usize);
        let mut cur = Cursor::new(&buf);
        assert_eq!(cur.read::<u64>().expect("u8"), 7);
        assert_eq!(cur.read::<u64>().expect("u16"), 300);
        assert_eq!(cur.read::<u32>().expect("u32"), 40_000);
        assert_eq!(cur.read::<u64>().expect("u64"), 9_000_000_000);
        assert_eq!(cur.read::<u64>().expect("usize"), 5);
    }

    #[test]
    fn a_padded_zero_is_rejected() {
        let error = Cursor::new(&[0x80, 0x00])
            .read::<u64>()
            .expect_err("非规范");
        assert_eq!(error.code(), Code::Invalid);
    }

    #[test]
    fn an_overflowing_sequence_is_rejected() {
        let mut seq = vec![0x80u8; 9];
        seq.push(0x02); // 第 10 字节用了第 2 位，超出 u64
        assert!(Cursor::new(&seq).read::<u64>().is_err());
        assert!(Cursor::new(&[0x80u8; 11]).read::<u64>().is_err());
    }

    #[test]
    fn taking_past_the_end_is_an_error() {
        let mut cursor = Cursor::new(&[1, 2, 3]);
        assert_eq!(cursor.take(2).expect("可取"), &[1, 2]);
        assert_eq!(cursor.pos(), 2);
        assert!(cursor.take(2).is_err());
    }

    /// 属性骨架：随机 `u64` 编码后必须规范解码回原值。
    ///
    /// 用例串行生成——下一个输入依赖上一次 PRNG 状态，无法预生成批量；不变量是往返一致与
    /// 字节数上界，边界值（0、127、128、u32::MAX、u64::MAX）由 `integers_round_trip_across_boundaries`
    /// 单独锁定，这里只覆盖中间分布。
    #[test]
    fn random_u64_values_round_trip_through_canonical_encoding() {
        let mut rng = testkit::Rng::new(0x1E_B12E);
        for _ in 0..64 {
            let value = ((rng.below(u32::MAX) as u64) << 32) | rng.below(u32::MAX) as u64;
            let mut buf = Vec::new();
            write(&mut buf, value);
            assert!(buf.len() <= 10, "u64 的 LEB128 最多 10 字节");
            let decoded: u64 = Cursor::new(&buf).read().expect("随机值必为规范编码");
            assert_eq!(decoded, value);
        }
    }
