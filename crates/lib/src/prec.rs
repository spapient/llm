/// Describes the floating-point precision used by a computation.
///
/// `precision` names a compute format only. It does not imply that a backend
/// supports it; capabilities declared by a backend are the authority for
/// support. The storage taxonomy is a strict superset of this set: integer
/// storage and storage-only `f64` tensors have no compute precision here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum precision {
    /// 32-bit IEEE-754 binary floating point.
    f32,
    /// 16-bit IEEE-754 binary floating point.
    f16,
    /// 16-bit bfloat16 floating point.
    bf16,
}

impl precision {
    /// Returns the storage size of one value of this precision, in bytes.
    #[must_use]
    pub const fn sizeinbytes(self) -> usize {
        match self {
            Self::f32 => 4,
            Self::f16 => 2,
            Self::bf16 => 2,
        }
    }
}
