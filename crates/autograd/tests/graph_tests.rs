use super::Requirement::{Backward, Grad, None};
use super::Role;

testkit::test_cases! {
fn only_two_none_inputs_merge_to_none() {
    let cases = [
        (None, None, None),
        (None, Backward, Backward),
        (None, Grad, Backward),
        (Backward, None, Backward),
        (Backward, Backward, Backward),
        (Backward, Grad, Backward),
        (Grad, None, Backward),
        (Grad, Backward, Backward),
        (Grad, Grad, Backward),
    ];
    for (left, right, expected) in cases {
        assert_eq!(left.merged(right), expected, "{left:?} 与 {right:?}");
    }
}

fn merging_is_commutative_and_idempotent() {
    for left in [None, Backward, Grad] {
        for right in [None, Backward, Grad] {
            assert_eq!(left.merged(right), right.merged(left));
        }
        assert_eq!(
            left.merged(left),
            if left.is_none() { None } else { Backward }
        );
    }
}

fn role_is_derived_from_requirement_and_leaf_status() {
    let cases = [
        (None, true, Role::Constant),
        (Grad, true, Role::Parameter),
        (Backward, true, Role::Constant),
        (None, false, Role::Activation),
        (Backward, false, Role::Activation),
        (Grad, false, Role::Activation),
    ];
    for (requirement, is_leaf, expected) in cases {
        assert_eq!(Role::of(requirement, is_leaf), expected);
    }
    assert!(Role::Parameter.is_parameter());
    assert!(!Role::Constant.is_parameter() && !Role::Activation.is_parameter());
    assert_eq!(
        [
            Role::Constant.as_str(),
            Role::Parameter.as_str(),
            Role::Activation.as_str()
        ],
        ["constant", "parameter", "activation"]
    );
}

fn the_predicates_partition_the_three_variants() {
    assert!(None.is_none() && !None.requires() && !None.is_endpoint());
    assert!(!Backward.is_none() && Backward.requires() && !Backward.is_endpoint());
    assert!(!Grad.is_none() && Grad.requires() && Grad.is_endpoint());
    assert_eq!(
        [None.name(), Backward.name(), Grad.name()],
        ["None", "Backward", "Grad"]
    );
}
}
