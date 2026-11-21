    use super::*;

    #[test]
    fn the_table_agrees_with_the_bitwise_definition() {
        testkit::assert_table_matches_bitwise(crc32);
    }

    #[test]
    fn chunked_feeds_match_one_shot() {
        let data: Vec<u8> = (0u8..=250).collect();
        let whole = crc32(&data);
        let mut state = !0u32;
        for chunk in data.chunks(7) {
            state = update(state, chunk);
        }
        assert_eq!(!state, whole);
    }
