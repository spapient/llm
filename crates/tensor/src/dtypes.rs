use lib::precision;

/// The storage data type of a tensor element.
///
/// `dtype` is the complete storage taxonomy: every primitive layout a tensor
/// buffer can hold. It is deliberately a strict superset of the set of
/// computation precisions in [`precision`]. [`From<precision>`](dtype) embeds
/// the compute precisions into this taxonomy, but the reverse does not hold:
/// integer storage types have no compute precision, and [`dtype::f64`] is a
/// storage-only type with no corresponding [`precision`].
///
/// A `dtype` states how a value is laid out in memory. It does not imply that
/// any backend can execute it; the capabilities a backend declares are the only
/// authority for execution support.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum dtype {
    /// 32-bit IEEE-754 binary floating point.
    f32,
    /// 64-bit IEEE-754 binary floating point.
    ///
    /// Storage-only: no backend declares `f64` execution, and it has no
    /// [`precision`] counterpart. It exists so that tensors can hold
    /// double-precision reference data and accumulation buffers.
    f64,
    /// 16-bit IEEE-754 binary floating point.
    f16,
    /// 16-bit bfloat16 floating point.
    bf16,
    /// 8-bit signed integer.
    i8,
    /// 8-bit unsigned integer.
    u8,
    /// 32-bit signed integer.
    i32,
    /// 64-bit signed integer.
    i64,
}

impl dtype {
    /// Returns the storage size of one value of this `dtype`, in bytes.
    #[must_use]
    pub const fn sizeinbytes(self) -> usize {
        match self {
            Self::f32 => 4,
            Self::f64 => 8,
            Self::f16 => 2,
            Self::bf16 => 2,
            Self::i8 => 1,
            Self::u8 => 1,
            Self::i32 => 4,
            Self::i64 => 8,
        }
    }
}

impl From<precision> for dtype {
    fn from(p: precision) -> Self {
        match p {
            precision::f32 => Self::f32,
            precision::f16 => Self::f16,
            precision::bf16 => Self::bf16,
        }
    }
}
