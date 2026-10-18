//! Relocated unit tests for the `tensor` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/tensor/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/tensor/src/dtypes.rs`.
mod dtypes {
    use ::tensor::dtype;
    use lib::precision;

    /// The storage width of each dtype and the precision-to-dtype mapping are
    /// two views of the same table and must agree.
    #[test]
    fn dtype_storage_width_and_precision_mapping_agree() {
        assert_eq!(dtype::f32.sizeinbytes(), 4);
        assert_eq!(dtype::f64.sizeinbytes(), 8);
        assert_eq!(dtype::f16.sizeinbytes(), 2);
        assert_eq!(dtype::bf16.sizeinbytes(), 2);
        assert_eq!(dtype::i8.sizeinbytes(), 1);
        assert_eq!(dtype::u8.sizeinbytes(), 1);
        assert_eq!(dtype::i32.sizeinbytes(), 4);
        assert_eq!(dtype::i64.sizeinbytes(), 8);
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

/// Tests for the half-precision element implementations.
mod halfelements {
    use ::tensor::dtype;
    use ::tensor::element;
    use ::tensor::{bf16, f16};

    #[test]
    fn half_dtype_constants_match_storage_type() {
        assert_eq!(<f16 as element>::DTYPE, dtype::f16);
        assert_eq!(<bf16 as element>::DTYPE, dtype::bf16);
    }

    #[test]
    fn half_width_matches_native_size() {
        assert_eq!(
            <f16 as element>::DTYPE.sizeinbytes(),
            core::mem::size_of::<f16>()
        );
        assert_eq!(
            <bf16 as element>::DTYPE.sizeinbytes(),
            core::mem::size_of::<bf16>()
        );
    }

    #[test]
    fn half_elements_are_copy() {
        fn assertcopy<T: element>() {}
        assertcopy::<f16>();
        assertcopy::<bf16>();
    }

    #[test]
    fn f16_roundtrips_exactly_representable_values() {
        for value in [0.0f32, -0.0, 1.0, -1.0, 0.5, -2.25, 65504.0, -65504.0] {
            let half = f16::from_f32(value);
            let back = half.to_f32();
            assert_eq!(back.to_bits(), value.to_bits(), "roundtrip {value}");
        }
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        // 1.0 is 0x3c00; the next f16 is 1 + 2^-10 = 0x3c01, so the midpoint
        // is 1 + 2^-11 (even on the low side) and must round down; the point
        // just above the midpoint must round up.
        let midpoint = 1.0f32 + 2f32.powi(-11);
        assert_eq!(f16::from_f32(midpoint).to_bits(), 0x3c00);
        let just_above = 1.0f32 + 2f32.powi(-11) + 2f32.powi(-20);
        assert_eq!(f16::from_f32(just_above).to_bits(), 0x3c01);
        // 1.5 is exact; -1.5 keeps the sign.
        assert_eq!(f16::from_f32(1.5).to_bits(), 0x3e00);
        assert_eq!(f16::from_f32(-1.5).to_bits(), 0xbe00);
    }

    #[test]
    fn f16_handles_subnormals_and_underflow() {
        // Smallest f16 subnormal is 2^-24.
        let smallest = 2f32.powi(-24);
        assert_eq!(f16::from_f32(smallest).to_bits(), 0x0001);
        // Below half of the smallest subnormal rounds to zero.
        assert_eq!(f16::from_f32(smallest / 2.0).to_bits(), 0x0000);
        // Largest subnormal decodes back to its exact value.
        let largest = f16::from_bits(0x03ff);
        assert_eq!(largest.to_f32(), 0x03ff as f32 * 2f32.powi(-24));
        assert_eq!(largest.to_f32().to_bits(), 0x387fc000);
    }

    #[test]
    fn f16_handles_overflow_and_special_values() {
        // Above 65504 the midpoint between 65504 and 65536 rounds to infinity.
        assert_eq!(f16::from_f32(65504.0).to_bits(), 0x7bff);
        assert_eq!(f16::from_f32(65520.0).to_bits(), 0x7c00);
        assert_eq!(f16::from_f32(-65520.0).to_bits(), 0xfc00);
        assert_eq!(f16::from_f32(1e10).to_bits(), 0x7c00);
        // Signed zero and infinities survive.
        assert_eq!(f16::from_f32(f32::INFINITY).to_bits(), 0x7c00);
        assert_eq!(f16::from_f32(f32::NEG_INFINITY).to_bits(), 0xfc00);
        // NaN keeps a nonzero payload and a quiet bit.
        let nan = f16::from_f32(f32::NAN);
        assert_eq!(nan.to_bits() & 0x7c00, 0x7c00);
        assert_ne!(nan.to_bits() & 0x03ff, 0);
    }

    #[test]
    fn bf16_roundtrips_and_rounds() {
        for value in [0.0f32, -0.0, 1.0, -1.0, 0.5, 2.25] {
            let half = bf16::from_f32(value);
            assert_eq!(half.to_f32(), value, "roundtrip {value}");
        }
        // 1.0 + 2^-16 sits at the bf16 midpoint and must round down (lsb even).
        assert_eq!(bf16::from_f32(1.0 + 2f32.powi(-16)).to_bits(), 0x3f80);
        // 1.0 + 2^-8 is also a midpoint (between 0x3f80 and 0x3f81) and rounds
        // down to the even value; just above the midpoint rounds up.
        assert_eq!(bf16::from_f32(1.0 + 2f32.powi(-8)).to_bits(), 0x3f80);
        assert_eq!(
            bf16::from_f32(1.0 + 2f32.powi(-7) + 2f32.powi(-16)).to_bits(),
            0x3f81
        );
        // 65504 is not representable in bf16 (needs 12 mantissa bits); it
        // rounds to the next binade, 65536 (0x4780).
        assert_eq!(bf16::from_f32(65504.0).to_bits(), 0x4780);
        assert_eq!(bf16::from_f32(65504.0).to_f32(), 65536.0);
        // 1e38 is finite in bf16 (biased exponent 253 < 255, so not infinity).
        assert_ne!(bf16::from_f32(1e38).to_bits() & 0x7f80, 0x7f80);
        // Infinities and NaN survive.
        assert_eq!(bf16::from_f32(f32::INFINITY).to_bits(), 0x7f80);
        assert_eq!(bf16::from_f32(f32::NEG_INFINITY).to_bits(), 0xff80);
        let nan = bf16::from_f32(f32::NAN);
        assert_eq!(nan.to_bits() & 0x7f80, 0x7f80);
        assert_ne!(nan.to_bits() & 0x007f, 0);
    }
}

/// Tests for the byte-to-tensor conversion primitives.
mod bytetensor {
    use ::tensor::dtype;
    use ::tensor::layout;
    use ::tensor::shape;
    use ::tensor::tensor;
    use ::tensor::tensorerror;
    use ::tensor::{bf16, f16};

    fn bytesfrom<T: ::tensor::byteable>(values: &[T]) -> Vec<u8> {
        ::tensor::bytesof(values)
    }

    #[test]
    fn owned_f32_roundtrips_through_bytes() {
        let s = shape::new(vec![2, 3]).unwrap();
        let t = tensor::new(s.clone(), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rebuilt: tensor<f32> = tensor::tryfrombytes(s, &t.getbytes()).unwrap();
        assert_eq!(rebuilt.getdata(), t.getdata());
        assert_eq!(rebuilt.getdtype(), dtype::f32);
        assert_eq!(rebuilt.getshape().getdims(), &[2, 3]);
    }

    #[test]
    fn owned_f16_roundtrips_through_bytes() {
        let s = shape::new(vec![2, 2]).unwrap();
        let values = vec![
            f16::from_f32(1.5),
            f16::from_f32(-2.0),
            f16::from_f32(0.25),
            f16::from_f32(4.0),
        ];
        let t = tensor::new(s.clone(), values).unwrap();
        let rebuilt: tensor<f16> = tensor::tryfrombytes(s, &t.getbytes()).unwrap();
        assert_eq!(rebuilt.getdata(), t.getdata());
        assert_eq!(rebuilt.getdtype(), dtype::f16);
    }

    #[test]
    fn owned_bf16_roundtrips_through_bytes() {
        let s = shape::new(vec![2]).unwrap();
        let values = vec![bf16::from_f32(1.0), bf16::from_f32(-0.5)];
        let t = tensor::new(s.clone(), values).unwrap();
        let rebuilt: tensor<bf16> = tensor::tryfrombytes(s, &t.getbytes()).unwrap();
        assert_eq!(rebuilt.getdata(), t.getdata());
        assert_eq!(rebuilt.getdtype(), dtype::bf16);
    }

    #[test]
    fn owned_frombytes_accepts_unaligned_sources() {
        let s = shape::new(vec![4]).unwrap();
        let mut bytes = vec![0xAA, 0xBB];
        bytes.extend_from_slice(&::tensor::bytesof(&[1.0f32, 2.0, 3.0, 4.0]));
        let t: tensor<f32> = tensor::tryfrombytes(s, &bytes[2..]).unwrap();
        assert_eq!(t.getdata(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn owned_frombytes_rejects_length_mismatch() {
        let s = shape::new(vec![2, 3]).unwrap();
        let bytes = vec![0u8; 5];
        assert_eq!(
            tensor::<f32>::tryfrombytes(s, &bytes).map(|_| ()),
            Err(tensorerror::lengthmismatch {
                expected: 24,
                actual: 5,
            })
        );
    }

    #[test]
    fn owned_frombytes_propagates_shape_overflow() {
        let s = shape::new(vec![usize::MAX, 2]).unwrap();
        let bytes = vec![0u8; 8];
        assert!(matches!(
            tensor::<u8>::tryfrombytes(s, &bytes),
            Err(tensorerror::elementcountoverflow { .. })
        ));
    }

    #[test]
    fn owned_frombytes_rejects_layout_not_valid_for_shape() {
        let s = shape::new(vec![2, 3]).unwrap();
        let other = shape::new(vec![6]).unwrap();
        let l = layout::contiguous(&other);
        let bytes = bytesfrom(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            tensor::<f32>::tryfrombyteswithlayout(s, l, &bytes).map(|_| ()),
            Err(tensorerror::invalidlayout)
        );
    }

    #[test]
    fn empty_tensor_yields_an_empty_view() {
        let s = shape::new(vec![0]).unwrap();
        let t = tensor::new(s, Vec::<f32>::new()).unwrap();
        assert_eq!(t.getview().getlen(), 0);
    }
}

mod byteview {
    /// A stack buffer with a guaranteed 16-byte alignment, so the alignment
    /// tests below do not depend on allocator placement of `Vec<u8>` (which
    /// only guarantees alignment 1).
    #[repr(align(16))]
    struct Align16([u8; 64]);

    #[test]
    fn zero_copy_view_matches_frombytes_and_preserves_address() {
        let data: Vec<f32> = vec![1.0, -2.5, 3.125, f32::INFINITY, f32::NAN, 0.0];
        let encoded = tensor::bytesof(&data);
        let mut backing = Align16([0u8; 64]);
        backing.0[..encoded.len()].copy_from_slice(&encoded);
        let bytes = &backing.0[..encoded.len()];
        let view = tensor::trybyteview::<f32>(bytes).unwrap();
        // Values match the copying conversion.
        let copied = tensor::frombytes::<f32>(bytes).unwrap();
        assert_eq!(view.len(), copied.len());
        for (a, b) in view.iter().zip(copied.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert_eq!(*a, *b);
            }
        }
        // Zero copy: the view points at the source bytes, not a copy.
        assert_eq!(
            view.as_ptr() as usize,
            bytes.as_ptr() as usize,
            "byteview must alias the source bytes"
        );
    }

    #[test]
    fn rejects_non_multiple_byte_length() {
        let bytes = vec![0u8; 5];
        assert!(matches!(
            tensor::trybyteview::<f32>(&bytes),
            Err(tensor::tensorerror::byteconversion)
        ));
    }

    #[test]
    fn rejects_misaligned_bytes() {
        let backing = Align16([0u8; 64]);
        // Offset by one breaks the 4-byte alignment of f32 (16k + 1 is not a
        // multiple of 4, regardless of the 16-byte-aligned base); the 4-byte
        // length keeps the length check from firing first.
        let misaligned = &backing.0[1..5];
        assert!(matches!(
            tensor::trybyteview::<f32>(misaligned),
            Err(tensor::tensorerror::viewmisaligned { align: 4 })
        ));
    }

    #[test]
    fn accepts_empty_slice() {
        let view = tensor::trybyteview::<f32>(&[]).unwrap();
        assert!(view.is_empty());
    }

    #[test]
    fn i8_view_accepts_any_alignment() {
        // i8 has alignment 1, so even the offset slice is a valid view.
        let bytes = [1u8, 2, 3, 4, 5, 6];
        let view = tensor::trybyteview::<i8>(&bytes[1..]).unwrap();
        assert_eq!(view, &[2, 3, 4, 5, 6]);
    }
}
