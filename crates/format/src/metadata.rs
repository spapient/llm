use lib::limits;
use tensor::{dtype, shape};

use crate::error::formaterror;

/// Maps a storage [`dtype`] to the canonical safetensors dtype token.
///
/// The mapping is total over [`dtype`]: every storage type this workspace
/// models has a safetensors token, including the storage-only `f64`, `f16`,
/// and `bf16` types. Producing a token does not assert that any backend can
/// execute the dtype.
#[must_use]
pub fn dtypetoken(value: dtype) -> &'static str {
    match value {
        dtype::f32 => "F32",
        dtype::f64 => "F64",
        dtype::f16 => "F16",
        dtype::bf16 => "BF16",
        dtype::i8 => "I8",
        dtype::u8 => "U8",
        dtype::i32 => "I32",
        dtype::i64 => "I64",
    }
}

/// Parses a safetensors dtype token into a storage [`dtype`].
///
/// Returns `None` for any token outside the supported set. This explicitly
/// rejects boolean, `i16`, and the various `u16`/`u32`/`u64` tokens rather than
/// silently widening or truncating them.
#[must_use]
pub fn tokendtype(token: &str) -> Option<dtype> {
    match token {
        "F32" => Some(dtype::f32),
        "F64" => Some(dtype::f64),
        "F16" => Some(dtype::f16),
        "BF16" => Some(dtype::bf16),
        "I8" => Some(dtype::i8),
        "U8" => Some(dtype::u8),
        "I32" => Some(dtype::i32),
        "I64" => Some(dtype::i64),
        _ => None,
    }
}

/// Metadata for one tensor stored in an external artifact.
///
/// A value records the tensor's name, storage [`dtype`], logical [`shape`],
/// element count, and the half-open byte range `[begin, end)` it occupies
/// within the artifact's data section. It carries no element data and no
/// request-scoped state.
///
/// Values are only produced by validating constructors inside this crate, so
/// every field is consistent by construction: `end - begin` equals the element
/// count multiplied by the dtype width, and the range lies inside the data
/// section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct tensormetadata {
    name: String,
    dtype: dtype,
    shape: shape,
    begin: u64,
    end: u64,
    count: usize,
}

impl tensormetadata {
    /// Validates a raw safetensors entry into a `tensormetadata`.
    ///
    /// The dtype token is resolved to a storage type, the dimensions are
    /// checked for element-count overflow and against the configured element
    /// budget, and the byte range is checked to lie inside `datalength` and to
    /// match the dtype width exactly.
    ///
    /// # Errors
    ///
    /// Returns a [`formaterror`] describing the first failed check, so malformed
    /// or truncated entries never panic and never allocate based on an
    /// unvalidated size.
    pub(crate) fn new(
        name: String,
        dtypetoken: &str,
        dims: Vec<usize>,
        begin: u64,
        end: u64,
        datalength: u64,
        lim: &limits,
    ) -> Result<Self, formaterror> {
        let dtype = tokendtype(dtypetoken).ok_or_else(|| formaterror::unsupporteddtype {
            name: name.clone(),
            dtype: dtypetoken.to_string(),
        })?;
        if dims.is_empty() {
            return Err(formaterror::emptyshape { name });
        }
        let s = shape::new(dims).map_err(|_| formaterror::emptyshape { name: name.clone() })?;
        let count = s
            .getelementcount()
            .map_err(|_| formaterror::shapeoverflow { name: name.clone() })?;
        lim.checkelements(count)
            .map_err(|_| formaterror::elementlimitexceeded {
                name: name.clone(),
                requested: count,
                limit: lim.getmaxelements(),
            })?;
        let expected = (count as u64)
            .checked_mul(dtype.sizeinbytes() as u64)
            .ok_or_else(|| formaterror::shapeoverflow { name: name.clone() })?;
        if end < begin || end > datalength {
            return Err(formaterror::offsetsoutofrange {
                name,
                begin,
                end,
                datalength,
            });
        }
        let actual = end - begin;
        if actual != expected {
            return Err(formaterror::bytelengthmismatch {
                name,
                expected,
                actual,
            });
        }
        Ok(Self {
            name,
            dtype,
            shape: s,
            begin,
            end,
            count,
        })
    }

    /// Returns the tensor name.
    #[must_use]
    pub fn getname(&self) -> &str {
        &self.name
    }

    /// Returns the storage data type.
    #[must_use]
    pub fn getdtype(&self) -> dtype {
        self.dtype
    }

    /// Returns the logical shape.
    #[must_use]
    pub fn getshape(&self) -> &shape {
        &self.shape
    }

    /// Returns the inclusive start offset within the data section.
    #[must_use]
    pub fn getbegin(&self) -> u64 {
        self.begin
    }

    /// Returns the exclusive end offset within the data section.
    #[must_use]
    pub fn getend(&self) -> u64 {
        self.end
    }

    /// Returns the number of bytes the tensor occupies.
    #[must_use]
    pub fn getnbytes(&self) -> u64 {
        self.end - self.begin
    }

    /// Returns the number of logical elements.
    #[must_use]
    pub fn getelementcount(&self) -> usize {
        self.count
    }
}
