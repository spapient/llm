use crate::dtype;

/// A primitive type that can be stored as a tensor element.
///
/// `element` is implemented for every primitive type with a native Rust
/// representation: `f32`, `f64`, `i8`, `u8`, `i32`, and `i64`, plus the
/// half-precision types [`half::f16`] and [`half::bf16`] provided by the
/// `half` crate. The half-precision types are storage-only: no backend
/// currently declares their execution.
///
/// [`element::DTYPE`] is total over the implemented types, and each
/// implementation's [`dtype::sizeinbytes`] MUST equal `size_of::<Self>()`. The
/// `dtype` taxonomy is a superset of [`precision`](lib::precision): integer
/// storage types, `dtype::f64`, `dtype::f16`, and `dtype::bf16` are
/// storage-only and have no compute precision, so their values may be held in a
/// tensor even though no backend declares their execution.
pub trait element: Copy + 'static {
    /// The storage data type of this element.
    const DTYPE: dtype;
}

impl element for f32 {
    const DTYPE: dtype = dtype::f32;
}

impl element for f64 {
    const DTYPE: dtype = dtype::f64;
}

impl element for i8 {
    const DTYPE: dtype = dtype::i8;
}

impl element for u8 {
    const DTYPE: dtype = dtype::u8;
}

impl element for i32 {
    const DTYPE: dtype = dtype::i32;
}

impl element for i64 {
    const DTYPE: dtype = dtype::i64;
}

impl element for half::f16 {
    const DTYPE: dtype = dtype::f16;
}

impl element for half::bf16 {
    const DTYPE: dtype = dtype::bf16;
}
