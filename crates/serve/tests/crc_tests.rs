use super::*;

#[test]
fn the_table_agrees_with_the_bitwise_definition() {
    testkit::assert_table_matches_bitwise(crc32);
}
