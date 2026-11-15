use super::*;

#[test]
fn a_group_must_be_sorted_and_unique() {
    let group = Group::new([0, 2, 4]).expect("升序");
    assert_eq!(group.ranks(), [0, 2, 4]);
    assert_eq!(group.size(), 3);
    assert_eq!(group.index_of(4), Some(2));
    assert!(!group.contains(1));
    assert_eq!(Group::new([]).unwrap_err().code(), Code::Invalid);
    assert_eq!(Group::new([2, 1]).unwrap_err().code(), Code::Invalid);
    assert_eq!(Group::new([1, 1]).unwrap_err().code(), Code::Invalid);
}
