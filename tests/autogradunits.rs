//! Relocated unit tests for the `autograd` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/autograd/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/decision-records/blackboxtesting.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/autograd/src/arena.rs`.
mod arena {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::autoggraderror;
    use ::autograd::graph;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn leaf_and_constant_are_recorded() {
        let mut g = graph::new();
        assert!(g.isempty());
        let a = g.leaf(build(vec![2], vec![1.0, 2.0]), true);
        let b = g.constant(build(vec![1], vec![7.0]));
        assert_eq!(g.getlen(), 2);
        assert!(g.requiresgrad(a).unwrap());
        assert!(!g.requiresgrad(b).unwrap());
        assert!(g.getgrad(a).is_none());
    }

    #[test]
    fn unknown_handles_are_rejected() {
        let g = graph::<f32>::new();
        assert_eq!(
            g.requiresgrad(::autograd::variableid::new(0)).map(|_| ()),
            Err(autoggraderror::unknownvariable { id: 0 })
        );
        assert!(g.getvalue(::autograd::variableid::new(9)).is_none());
    }

    #[test]
    fn detach_copies_the_value_without_a_producer() {
        let mut g = graph::new();
        let a = g.leaf(build(vec![2], vec![1.0, 2.0]), true);
        let d = g.detach(a).unwrap();
        assert!(!g.requiresgrad(d).unwrap());
        assert_eq!(g.getvalue(d).unwrap().getdata(), &[1.0, 2.0]);
    }

    #[test]
    fn root_handle_is_distinct_from_its_graph_scope() {
        // Regression guard: ids are graph-local indices in creation order.
        let mut g = graph::<f32>::new();
        let first = g.leaf(build(vec![1], vec![1.0]), false);
        let second = g.leaf(build(vec![1], vec![2.0]), false);
        assert_eq!(first.getindex() + 1, second.getindex());
        assert_eq!(g.get(second).unwrap().getid(), second);
    }
}

/// Tests relocated from `crates/autograd/src/backward.rs`.
mod backward {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::autoggraderror;
    use ::autograd::graph;
    use ::autograd::variableid;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn backward_on_a_leaf_seeds_its_gradient() {
        let mut g = graph::new();
        let a = g.leaf(build(vec![2], vec![1.0, 2.0]), true);
        g.backward(a).unwrap();
        assert_eq!(g.getgrad(a).unwrap().getdata(), &[1.0, 1.0]);
    }

    #[test]
    fn backward_on_a_constant_is_a_noop() {
        let mut g = graph::new();
        let a = g.constant(build(vec![1], vec![1.0]));
        g.backward(a).unwrap();
        assert!(g.getgrad(a).is_none());
    }

    #[test]
    fn seed_shape_mismatch_is_rejected() {
        let mut g = graph::new();
        let a = g.leaf(build(vec![2], vec![1.0, 2.0]), true);
        assert_eq!(
            g.backwardwith(a, Some(build(vec![3], vec![1.0, 1.0, 1.0])))
                .map(|_| ()),
            Err(autoggraderror::unsupportedshape {
                op: "backward",
                dims: vec![3],
            })
        );
    }

    #[test]
    fn unknown_output_is_rejected() {
        let mut g = graph::<f32>::new();
        assert_eq!(
            g.backward(variableid::new(5)).map(|_| ()),
            Err(autoggraderror::unknownvariable { id: 5 })
        );
    }
}

/// Tests relocated from `crates/autograd/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::autoggraderror;
    use ::tensor::tensorerror;
    use api::backenderror;

    #[test]
    fn converts_from_backenderror() {
        let e: autoggraderror = backenderror::sizeoverflow.into();
        assert_eq!(e, autoggraderror::backend(backenderror::sizeoverflow));
        assert_eq!(e.to_string(), "size overflow");
    }

    #[test]
    fn converts_from_tensorerror() {
        let e: autoggraderror = tensorerror::emptyshape.into();
        assert_eq!(e.to_string(), "a shape must have at least one dimension");
    }

    #[test]
    fn unknown_variable_reports_the_index() {
        let e = autoggraderror::unknownvariable { id: 7 };
        assert_eq!(e.to_string(), "variable id 7 is not present in the graph");
        assert_ne!(e, autoggraderror::unknownvariable { id: 8 });
    }

    #[test]
    fn shape_variant_carries_the_operation() {
        let e = autoggraderror::unsupportedshape {
            op: "matmul",
            dims: vec![3],
        };
        assert_eq!(e.to_string(), "operation matmul does not support shape [3]");
    }
}

/// Tests relocated from `crates/autograd/src/gradient.rs`.
mod gradient {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::autoggraderror;
    use ::autograd::graph;
    use ::autograd::variableid;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn zeroing_one_gradient_leaves_others_untouched() {
        let mut g = graph::new();
        let a = g.leaf(build(vec![1], vec![1.0]), true);
        let b = g.leaf(build(vec![1], vec![2.0]), true);
        g.backwardwith(a, Some(build(vec![1], vec![10.0]))).unwrap();
        g.backwardwith(b, Some(build(vec![1], vec![20.0]))).unwrap();
        g.zerograd(a).unwrap();
        assert!(g.getgrad(a).is_none());
        assert_eq!(g.getgrad(b).unwrap().getdata(), &[20.0]);
    }

    #[test]
    fn clearing_all_gradients_empties_the_tape_grades() {
        let mut g = graph::new();
        let a = g.leaf(build(vec![1], vec![1.0]), true);
        g.backwardwith(a, Some(build(vec![1], vec![10.0]))).unwrap();
        g.cleargradients();
        assert!(g.getgrad(a).is_none());
        assert_eq!(g.getlen(), 1);
    }

    #[test]
    fn zeroing_an_unknown_handle_is_rejected() {
        let mut g = graph::<f32>::new();
        assert_eq!(
            g.zerograd(variableid::new(4)).map(|_| ()),
            Err(autoggraderror::unknownvariable { id: 4 })
        );
    }
}

/// Tests relocated from `crates/autograd/src/node.rs`.
mod node {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::op;
    use ::autograd::variableid;

    #[test]
    fn handle_reports_its_index() {
        assert_eq!(variableid::new(3).getindex(), 3);
        assert_eq!(variableid::new(3), variableid::new(3));
        assert_ne!(variableid::new(3), variableid::new(4));
    }

    #[test]
    fn leaf_has_no_inputs() {
        assert!(op::leaf.isleaf());
        assert!(op::leaf.getinputs().is_empty());
        assert_eq!(op::leaf.getname(), "leaf");
    }

    #[test]
    fn binary_operations_report_two_inputs_in_order() {
        let a = variableid::new(0);
        let b = variableid::new(1);
        assert_eq!(op::matmul { lhs: a, rhs: b }.getinputs(), vec![a, b]);
        assert_eq!(op::matmul { lhs: a, rhs: b }.getname(), "matmul");
        assert_eq!(op::add { lhs: a, rhs: b }.getinputs(), vec![a, b]);
        assert_eq!(op::mul { lhs: a, rhs: b }.getinputs(), vec![a, b]);
    }

    #[test]
    fn unary_operations_report_one_input() {
        let a = variableid::new(2);
        for producer in [
            op::softmax { input: a },
            op::relu { input: a },
            op::silu { input: a },
            op::gelutanh { input: a },
        ] {
            assert_eq!(producer.getinputs(), vec![a]);
            assert!(!producer.isleaf());
        }
        assert_eq!(
            op::rmsnorm {
                input: a,
                weight: variableid::new(3),
                eps: 1.0e-5,
            }
            .getinputs(),
            vec![a, variableid::new(3)]
        );
    }
}

/// Tests relocated from `crates/autograd/src/nograd.rs`.
mod nograd {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::graph;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn scope_disables_and_restores_recording() {
        let mut g = graph::<f32>::new();
        assert!(g.recording());
        {
            let guard = g.nograd();
            assert!(!guard.recording());
        }
        assert!(g.recording());
    }

    #[test]
    fn deref_exposes_the_tape() {
        let mut g = graph::<f32>::new();
        let a = g.leaf(build(vec![1], vec![1.0]), true);
        {
            let mut guard = g.nograd();
            let b = guard.constant(build(vec![1], vec![2.0]));
            assert_eq!(guard.getvalue(b).unwrap().getdata(), &[2.0]);
        }
        assert!(g.getvalue(a).is_some());
    }
}

/// Tests relocated from `crates/autograd/src/ops/activation.rs`.
mod ops_activation {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::graph;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn gelu_matches_known_constants() {
        let mut g = graph::new();
        let positive = g.leaf(build(vec![1], vec![1.0]), false);
        let out = g.gelutanh(positive).unwrap();
        assert!((g.getvalue(out).unwrap().getdata()[0] - 0.841_192).abs() < 1e-6);

        let negative = g.leaf(build(vec![1], vec![-1.0]), false);
        let out = g.gelutanh(negative).unwrap();
        assert!((g.getvalue(out).unwrap().getdata()[0] + 0.158_808).abs() < 1e-6);
    }
}

/// Tests relocated from `crates/autograd/src/variables.rs`.
mod variables {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::autograd::graph;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn reports_identity_value_and_producer() {
        let mut g = graph::new();
        let id = g.leaf(build(vec![2], vec![1.0, 2.0]), true);
        let v = g.get(id).unwrap();
        assert_eq!(v.getid(), id);
        assert_eq!(v.getvalue().getdata(), &[1.0, 2.0]);
        assert!(v.requiresgrad());
        assert!(v.getproducer().isleaf());
        assert!(v.getgrad().is_none());
    }

    #[test]
    fn gradient_is_absent_until_written() {
        let mut g = graph::new();
        let id = g.leaf(build(vec![1], vec![3.0]), true);
        assert!(g.get(id).unwrap().getgrad().is_none());
        g.backward(id).unwrap();
        assert_eq!(g.getgrad(id).unwrap().getdata(), &[1.0]);
        g.zerograd(id).unwrap();
        assert!(g.getgrad(id).is_none());
    }
}
