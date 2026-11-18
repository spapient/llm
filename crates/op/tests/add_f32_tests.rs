use super::*;

crate::elementwise::test_support::elementwise_binary_tests!(Add, forward);

#[test]
fn mismatched_lengths_are_rejected() {
    let mut out = vec![0.0; 2];
    assert!(forward(&[1.0, 2.0], &[1.0], &mut out).is_err());
}
