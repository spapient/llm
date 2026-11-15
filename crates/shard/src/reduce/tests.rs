use super::*;

#[test]
fn reduction_kinds_are_a_closed_set() {
    let names: Vec<&str> = Reduce::ALL.iter().map(|kind| kind.as_str()).collect();
    assert_eq!(names, ["sum", "max", "min", "prod"]);
}
