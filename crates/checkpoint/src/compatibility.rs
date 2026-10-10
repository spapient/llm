use format::dtypetoken;
use tensor::{dtype, shape};

use crate::error::checkpointerror;
use crate::manifests::checkpoint;

/// The expected name, dtype, and shape of one tensor.
///
/// A caller builds the expected tensor set from a model or training
/// configuration and passes it to [`checkcompat`]. The specification carries no
/// byte range: compatibility is a structural comparison of name, dtype, and
/// shape only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct tensorspec {
    name: String,
    dtype: dtype,
    shape: shape,
}

impl tensorspec {
    /// Creates an expectation for one tensor.
    #[must_use]
    pub fn new(name: String, value: dtype, dims: shape) -> Self {
        Self {
            name,
            dtype: value,
            shape: dims,
        }
    }

    /// Returns the expected tensor name.
    #[must_use]
    pub fn getname(&self) -> &str {
        &self.name
    }

    /// Returns the expected dtype.
    #[must_use]
    pub fn getdtype(&self) -> dtype {
        self.dtype
    }

    /// Returns the expected shape.
    #[must_use]
    pub fn getshape(&self) -> &shape {
        &self.shape
    }
}

/// Verifies that a checkpoint contains every expected tensor with a matching
/// dtype and shape.
///
/// Extra tensors in the checkpoint are permitted; the check is a lower bound on
/// the checkpoint's contents. This is a structural comparison only; it reads no
/// tensor bytes and performs no I/O, so it never returns a recoverable I/O
/// error.
///
/// # Errors
///
/// Returns [`checkpointerror::missingtensor`],
/// [`checkpointerror::dtypemismatch`], or
/// [`checkpointerror::shapemismatch`], each of which
/// [`checkpointerror::isincompatibility`] classifies as a structural
/// incompatibility.
pub fn checkcompat(ckpt: &checkpoint, expected: &[tensorspec]) -> Result<(), checkpointerror> {
    for spec in expected {
        let entry =
            ckpt.gettensor(spec.getname())
                .ok_or_else(|| checkpointerror::missingtensor {
                    name: spec.name.clone(),
                })?;
        if entry.getdtype() != spec.dtype {
            return Err(checkpointerror::dtypemismatch {
                name: spec.name.clone(),
                expected: dtypetoken(spec.dtype).to_string(),
                actual: dtypetoken(entry.getdtype()).to_string(),
            });
        }
        if entry.getshape().getdims() != spec.shape.getdims() {
            return Err(checkpointerror::shapemismatch {
                name: spec.name.clone(),
                expected: spec.shape.getdims().to_vec(),
                actual: entry.getshape().getdims().to_vec(),
            });
        }
    }
    Ok(())
}

/// Verifies that a checkpoint's version equals the supported version.
///
/// # Errors
///
/// Returns [`checkpointerror::unsupportedversion`], which
/// [`checkpointerror::isincompatibility`] classifies as a structural
/// incompatibility.
pub fn checkversion(ckpt: &checkpoint, supported: u32) -> Result<(), checkpointerror> {
    let found = ckpt.getmanifest().getversion();
    if found != supported {
        return Err(checkpointerror::unsupportedversion { found, supported });
    }
    Ok(())
}
