//! Cross-crate contract tests for `tensor` through its public API.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tensor::{dtype, element, layout, shape, tensor, tensorerror};
use lib::{limits, precision};

/// The precision-to-dtype mapping, the storage width of each dtype, and the
/// `element` type-level dtype constants must all agree.
#[test]
fn dtype_maps_from_precision_and_element_constants_agree() {
    assert_eq!(dtype::from(precision::f32), dtype::f32);
    assert_eq!(dtype::from(precision::f16), dtype::f16);
    assert_eq!(dtype::from(precision::bf16), dtype::bf16);
    assert_eq!(dtype::f32.sizeinbytes(), 4);
    assert_eq!(dtype::f64.sizeinbytes(), 8);
    assert_eq!(dtype::f16.sizeinbytes(), 2);
    assert_eq!(dtype::i64.sizeinbytes(), 8);
    assert_eq!(<f32 as element>::DTYPE, dtype::f32);
    assert_eq!(<f64 as element>::DTYPE, dtype::f64);
    assert_eq!(<i8 as element>::DTYPE, dtype::i8);
    assert_eq!(<u8 as element>::DTYPE, dtype::u8);
    assert_eq!(<i32 as element>::DTYPE, dtype::i32);
    assert_eq!(<i64 as element>::DTYPE, dtype::i64);
}

#[test]
fn f64_tensor_roundtrips_through_view() {
    let s = shape::new(vec![2, 2]).unwrap();
    let t = tensor::new(s, vec![0.5_f64, -1.0, 2.25, 4.0]).unwrap();
    assert_eq!(t.getdtype(), dtype::f64);
    assert_eq!(t.getview().getdata(), &[0.5, -1.0, 2.25, 4.0]);
    assert_eq!(t.getview().toowned().unwrap().getdata(), t.getdata());
}

#[test]
fn tensor_roundtrip_through_view() {
    let s = shape::new(vec![2, 3]).unwrap();
    let t = tensor::new(s, vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(t.getshape().getdims(), &[2, 3]);
    assert_eq!(t.getdtype(), dtype::i32);
    assert_eq!(t.getlen(), 6);
    assert!(!t.isempty());

    let v = t.getview();
    assert_eq!(v.getlen(), 6);
    assert_eq!(v.getdata(), &[1, 2, 3, 4, 5, 6]);

    let owned = v.toowned().unwrap();
    assert_eq!(owned.getdata(), t.getdata());
}

#[test]
fn contiguous_layout_is_valid_for_shape() {
    let s = shape::new(vec![2, 3, 4]).unwrap();
    let l = layout::contiguous(&s);
    assert_eq!(l.getstrides(), &[12, 4, 1]);
    assert_eq!(l.getoffset(), 0);
    assert!(l.iscontiguous());
    assert!(l.isvalidfor(&s));
}

#[test]
fn empty_shape_is_rejected() {
    assert!(matches!(shape::new(vec![]), Err(tensorerror::emptyshape)));
}

#[test]
fn element_count_overflow_is_reported() {
    let s = shape::new(vec![usize::MAX, 2]).unwrap();
    assert_eq!(
        s.getelementcount(),
        Err(tensorerror::elementcountoverflow {
            dims: vec![usize::MAX, 2],
        })
    );
}

#[test]
fn length_mismatch_is_reported() {
    let s = shape::new(vec![2, 3]).unwrap();
    assert_eq!(
        tensor::new(s, vec![1, 2, 3]).map(|_| ()),
        Err(tensorerror::lengthmismatch {
            expected: 6,
            actual: 3,
        })
    );
}

#[test]
fn checklimits_maps_limiterror_and_accepts_bound() {
    let lim = limits::new(8, 64, 16, 4);
    let over = shape::new(vec![4, 3]).unwrap();
    assert_eq!(
        over.checklimits(&lim),
        Err(tensorerror::elementcountlimitexceeded {
            requested: 12,
            limit: 8,
        })
    );
    let within = shape::new(vec![2, 3]).unwrap();
    assert_eq!(within.checklimits(&lim), Ok(6));
}
