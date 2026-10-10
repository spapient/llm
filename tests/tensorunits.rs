//! Relocated unit tests for the `tensor` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/tensor/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/decision-records/blackboxtesting.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/tensor/src/dtypes.rs`.
mod dtypes {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::dtype;
    use lib::precision;

    #[test]
    fn sizeinbytes_matches_storage_width() {
        assert_eq!(dtype::f32.sizeinbytes(), 4);
        assert_eq!(dtype::f64.sizeinbytes(), 8);
        assert_eq!(dtype::f16.sizeinbytes(), 2);
        assert_eq!(dtype::bf16.sizeinbytes(), 2);
        assert_eq!(dtype::i8.sizeinbytes(), 1);
        assert_eq!(dtype::u8.sizeinbytes(), 1);
        assert_eq!(dtype::i32.sizeinbytes(), 4);
        assert_eq!(dtype::i64.sizeinbytes(), 8);
    }

    #[test]
    fn from_precision_maps_each_precision() {
        assert_eq!(dtype::from(precision::f32), dtype::f32);
        assert_eq!(dtype::from(precision::f16), dtype::f16);
        assert_eq!(dtype::from(precision::bf16), dtype::bf16);
    }

    #[test]
    fn from_precision_preserves_storage_width() {
        for p in [precision::f32, precision::f16, precision::bf16] {
            assert_eq!(dtype::from(p).sizeinbytes(), p.sizeinbytes());
        }
    }

    #[test]
    fn f64_is_storage_only_and_not_a_compute_precision() {
        for p in [precision::f32, precision::f16, precision::bf16] {
            assert_ne!(dtype::from(p), dtype::f64);
        }
    }
}

/// Tests relocated from `crates/tensor/src/elements.rs`.
mod elements {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::dtype;
    use ::tensor::element;

    #[test]
    fn dtype_constants_match_storage_type() {
        assert_eq!(<f32 as element>::DTYPE, dtype::f32);
        assert_eq!(<f64 as element>::DTYPE, dtype::f64);
        assert_eq!(<i8 as element>::DTYPE, dtype::i8);
        assert_eq!(<u8 as element>::DTYPE, dtype::u8);
        assert_eq!(<i32 as element>::DTYPE, dtype::i32);
        assert_eq!(<i64 as element>::DTYPE, dtype::i64);
    }

    fn assertelementwidth<T: element>() {
        assert_eq!(
            <T as element>::DTYPE.sizeinbytes(),
            core::mem::size_of::<T>()
        );
    }

    #[test]
    fn dtype_width_matches_native_size() {
        assertelementwidth::<f32>();
        assertelementwidth::<f64>();
        assertelementwidth::<i8>();
        assertelementwidth::<u8>();
        assertelementwidth::<i32>();
        assertelementwidth::<i64>();
    }

    #[test]
    fn element_is_copy() {
        fn assertcopy<T: element>() {}
        assertcopy::<f32>();
        assertcopy::<f64>();
        assertcopy::<i8>();
        assertcopy::<u8>();
        assertcopy::<i32>();
        assertcopy::<i64>();
    }
}

/// Tests relocated from `crates/tensor/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::dtype;
    use ::tensor::tensorerror;

    #[test]
    fn variants_are_distinguishable_by_value() {
        assert_ne!(
            tensorerror::emptyshape,
            tensorerror::elementcountoverflow { dims: vec![1] }
        );
        assert_ne!(
            tensorerror::lengthmismatch {
                expected: 2,
                actual: 1,
            },
            tensorerror::lengthmismatch {
                expected: 3,
                actual: 1,
            }
        );
    }

    #[test]
    fn display_reports_carried_values() {
        let e = tensorerror::elementcountlimitexceeded {
            requested: 12,
            limit: 8,
        };
        assert_eq!(e.to_string(), "requested element count 12 exceeds limit 8");
        let d = tensorerror::dtypemismatch { dtype: dtype::f32 };
        assert_eq!(d.to_string(), "dtype mismatch: expected f32");
    }
}

/// Tests relocated from `crates/tensor/src/layouts.rs`.
mod layouts {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::layout;
    use ::tensor::shape;

    #[test]
    fn contiguous_strides_are_row_major() {
        let l = layout::contiguous(&shape::new(vec![2, 3]).unwrap());
        assert_eq!(l.getstrides(), &[3, 1]);
        assert_eq!(l.getoffset(), 0);
        assert!(l.iscontiguous());
    }

    #[test]
    fn contiguous_three_dim_strides() {
        let l = layout::contiguous(&shape::new(vec![2, 3, 4]).unwrap());
        assert_eq!(l.getstrides(), &[12, 4, 1]);
    }

    #[test]
    fn contiguous_layout_is_valid_for_its_shape() {
        let s = shape::new(vec![2, 3, 4]).unwrap();
        let l = layout::contiguous(&s);
        assert!(l.isvalidfor(&s));
    }

    #[test]
    fn rank_mismatch_is_invalid() {
        let s = shape::new(vec![2, 3]).unwrap();
        let l = layout::contiguous(&shape::new(vec![2, 3, 4]).unwrap());
        assert!(!l.isvalidfor(&s));
    }

    #[test]
    fn padded_layout_is_not_valid_for_shape() {
        let s = shape::new(vec![2, 3]).unwrap();
        let l = layout::contiguous(&shape::new(vec![2, 6]).unwrap());
        assert_eq!(l.getstrides(), &[6, 1]);
        assert!(l.iscontiguous());
        assert!(!l.isvalidfor(&s));
    }

    #[test]
    fn zero_dimension_layout_is_contiguous_and_valid() {
        let s = shape::new(vec![2, 0, 3]).unwrap();
        let l = layout::contiguous(&s);
        assert_eq!(l.getstrides(), &[0, 3, 1]);
        assert!(l.iscontiguous());
        assert!(l.isvalidfor(&s));
    }
}

/// Tests relocated from `crates/tensor/src/shapes.rs`.
mod shapes {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::shape;
    use ::tensor::tensorerror;
    use lib::limits;
    use proptest::collection;
    use proptest::prelude::*;

    #[test]
    fn new_rejects_empty_dimensions() {
        assert!(matches!(shape::new(vec![]), Err(tensorerror::emptyshape)));
    }

    #[test]
    fn rank_and_dims_are_reported() {
        let s = shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.getrank(), 3);
        assert_eq!(s.getdims(), &[2, 3, 4]);
    }

    #[test]
    fn single_element_shape_is_not_empty() {
        let s = shape::new(vec![1]).unwrap();
        assert_eq!(s.getrank(), 1);
        assert!(!s.isempty());
        assert_eq!(s.getelementcount(), Ok(1));
    }

    #[test]
    fn zero_dimension_is_empty() {
        let s = shape::new(vec![2, 0, 3]).unwrap();
        assert!(s.isempty());
        assert_eq!(s.getelementcount(), Ok(0));
    }

    #[test]
    fn elementcount_is_the_product_of_dimensions() {
        let s = shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.getelementcount(), Ok(24));
    }

    #[test]
    fn elementcount_overflow_is_reported() {
        let s = shape::new(vec![usize::MAX, 2]).unwrap();
        assert_eq!(
            s.getelementcount(),
            Err(tensorerror::elementcountoverflow {
                dims: vec![usize::MAX, 2],
            })
        );
    }

    #[test]
    fn checklimits_accepts_bound_and_rejects_above() {
        let lim = limits::new(8, 64, 16, 4);
        let within = shape::new(vec![2, 3]).unwrap();
        assert_eq!(within.checklimits(&lim), Ok(6));
        let over = shape::new(vec![4, 3]).unwrap();
        assert_eq!(
            over.checklimits(&lim),
            Err(tensorerror::elementcountlimitexceeded {
                requested: 12,
                limit: 8,
            })
        );
    }

    #[test]
    fn checklimits_propagates_overflow() {
        let lim = limits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX);
        let s = shape::new(vec![usize::MAX, 2]).unwrap();
        assert!(matches!(
            s.checklimits(&lim),
            Err(tensorerror::elementcountoverflow { .. })
        ));
    }

    fn referencecount(dims: &[usize]) -> Option<usize> {
        dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
    }

    proptest! {
        #[test]
        fn elementcount_agrees_with_checked_arithmetic(
            dims in collection::vec(any::<usize>(), 1..=6)
        ) {
            let s = shape::new(dims.clone()).unwrap();
            let want = referencecount(&dims);
            match (s.getelementcount(), want) {
                (Ok(got), Some(expected)) => prop_assert_eq!(got, expected),
                (Err(tensorerror::elementcountoverflow { .. }), None) => {}
                (got, expected) => prop_assert!(
                    false,
                    "count mismatch: {:?} vs {:?}",
                    got,
                    expected
                ),
            }
        }
    }
}

/// Tests relocated from `crates/tensor/src/tensors.rs`.
mod tensors {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::dtype;
    use ::tensor::shape;
    use ::tensor::tensor;
    use ::tensor::tensorerror;

    #[test]
    fn new_accepts_matching_length() {
        let t = tensor::new(shape::new(vec![2, 3]).unwrap(), vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(t.getlen(), 6);
        assert!(!t.isempty());
        assert_eq!(t.getdtype(), dtype::i32);
        assert_eq!(t.getdata(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(t.getshape().getdims(), &[2, 3]);
        assert_eq!(t.getlayout().getstrides(), &[3, 1]);
    }

    #[test]
    fn new_rejects_length_mismatch() {
        assert_eq!(
            tensor::new(shape::new(vec![2, 3]).unwrap(), vec![1, 2, 3]).map(|_| ()),
            Err(tensorerror::lengthmismatch {
                expected: 6,
                actual: 3,
            })
        );
    }

    #[test]
    fn empty_tensor_is_allowed() {
        let t = tensor::<f32>::new(shape::new(vec![0]).unwrap(), vec![]).unwrap();
        assert!(t.isempty());
        assert_eq!(t.getlen(), 0);
        assert_eq!(t.getdtype(), dtype::f32);
    }

    #[test]
    fn mutable_access_updates_data() {
        let mut t = tensor::new(shape::new(vec![3]).unwrap(), vec![1, 2, 3]).unwrap();
        t.getdatamut()[1] = 9;
        assert_eq!(t.getdata(), &[1, 9, 3]);
    }

    #[test]
    fn view_borrows_tensor_data() {
        let t = tensor::new(shape::new(vec![2, 2]).unwrap(), vec![1, 2, 3, 4]).unwrap();
        let v = t.getview();
        assert_eq!(v.getlen(), 4);
        assert_eq!(v.getdata(), &[1, 2, 3, 4]);
        assert_eq!(v.getshape().getdims(), &[2, 2]);
        assert_eq!(v.getlayout().getstrides(), &[2, 1]);
    }
}

/// Tests relocated from `crates/tensor/src/views.rs`.
mod views {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::tensor::shape;
    use ::tensor::tensor;

    #[test]
    fn toowned_copies_viewed_data() {
        let t = tensor::new(shape::new(vec![2, 3]).unwrap(), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let v = t.getview();
        let owned = v.toowned().unwrap();
        assert_eq!(owned.getdata(), t.getdata());
        assert_eq!(owned.getshape().getdims(), t.getshape().getdims());
        assert_eq!(owned.getlayout().getstrides(), t.getlayout().getstrides());
    }

    #[test]
    fn view_is_copy() {
        let t = tensor::new(shape::new(vec![2]).unwrap(), vec![7, 8]).unwrap();
        let v = t.getview();
        let w = v;
        assert_eq!(v.getdata(), w.getdata());
    }
}
