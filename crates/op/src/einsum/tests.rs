use super::*;

#[test]
fn a_closed_set_reports_every_width() {
    let widths: Vec<u32> = Bits::ALL.iter().map(|bits| bits.bits()).collect();
    assert_eq!(widths, [4, 8, 16, 32, 64]);
}

#[test]
fn packed_widths_are_rounded_up_to_whole_bytes() {
    assert_eq!(Bits::Four.bytes_for(3), 2, "3 个 int4 = 1.5 字节 → 2");
    assert_eq!(Bits::Four.bytes_for(0), 0);
    assert_eq!(Bits::ThirtyTwo.bytes_for(3), 12);
}

#[test]
fn the_zero_cost_is_the_identity_of_adding() {
    let cost = Cost::new(2.0, 8, 4);
    assert_eq!(cost.plus(Cost::ZERO), cost);
    assert_eq!(cost.bytes(), 12, "合计是派生量，不是口径本身");
    assert_eq!(
        Cost::new(1.0, 8, 4).plus(Cost::new(1.0, 8, 4)),
        Cost::new(2.0, 16, 8)
    );
}
