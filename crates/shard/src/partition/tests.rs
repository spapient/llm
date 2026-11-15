use super::*;

#[test]
fn a_partition_requires_an_even_split() {
    let partition = Partition::new(0, 4).expect("切分");
    assert_eq!(partition.slice(16, 0), Ok((0, 4)));
    assert_eq!(partition.slice(16, 3), Ok((12, 4)));
    assert_eq!(partition.slice(15, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(partition.slice(16, 4).unwrap_err().code(), Code::Invalid);
    assert_eq!(Partition::new(0, 0).unwrap_err().code(), Code::Invalid);
}
