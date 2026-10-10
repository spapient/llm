use ::tensor::{shape, tensor};

use crate::error::opterror;

/// One named `f32` tensor inside an [`optimizerstate`] snapshot.
///
/// A `statetensor` is the optimizer's unit of persistence. It carries a stable
/// name, the element shape, and the element data, so it maps directly onto a
/// checkpoint tensor without the optimizer crate depending on the checkpoint
/// format. Per the workspace convention, values are serialized as
/// little-endian `f32`; [`Self::tobytes`] and [`Self::frombytes`] are exact
/// inverses for finite and non-finite values alike.
#[derive(Clone, Debug, PartialEq)]
pub struct statetensor {
    name: String,
    shape: shape,
    data: Vec<f32>,
}

impl statetensor {
    /// Creates a state tensor from a name, shape, and element data.
    ///
    /// # Errors
    ///
    /// Returns a [`tensorerror`](::tensor::tensorerror) when the shape's
    /// element count overflows, or [`opterror::invalidstate`] when the data
    /// length differs from the shape's element count.
    pub fn new(name: impl Into<String>, dims: shape, data: Vec<f32>) -> Result<Self, opterror> {
        let expected = dims.getelementcount()?;
        if data.len() != expected {
            return Err(opterror::state(format!(
                "state tensor has {} elements but shape {dims:?} requires {expected}",
                data.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape: dims,
            data,
        })
    }

    /// Returns the tensor name.
    #[must_use]
    pub fn getname(&self) -> &str {
        &self.name
    }

    /// Returns the element shape.
    #[must_use]
    pub fn getshape(&self) -> &shape {
        &self.shape
    }

    /// Returns the element data.
    #[must_use]
    pub fn getdata(&self) -> &[f32] {
        &self.data
    }

    /// Encodes the elements as contiguous little-endian `f32` bytes.
    #[must_use]
    pub fn tobytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * core::mem::size_of::<f32>());
        for value in &self.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Decodes a state tensor from little-endian `f32` bytes.
    ///
    /// # Errors
    ///
    /// Returns a [`tensorerror`](::tensor::tensorerror) when the shape's
    /// element count overflows, or [`opterror::invalidstate`] when the byte
    /// length does not equal the element count times the `f32` width.
    pub fn frombytes(name: impl Into<String>, dims: shape, bytes: &[u8]) -> Result<Self, opterror> {
        let count = dims.getelementcount()?;
        let width = core::mem::size_of::<f32>();
        let expected = count
            .checked_mul(width)
            .ok_or_else(|| opterror::state("state tensor byte length overflows usize"))?;
        if bytes.len() != expected {
            return Err(opterror::state(format!(
                "state tensor has {} bytes but shape {dims:?} requires {expected}",
                bytes.len()
            )));
        }
        let mut data = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(width) {
            let raw: [u8; 4] = chunk
                .try_into()
                .map_err(|_| opterror::state("state tensor byte chunk is not four bytes"))?;
            data.push(f32::from_le_bytes(raw));
        }
        Ok(Self {
            name: name.into(),
            shape: dims,
            data,
        })
    }

    /// Builds a state tensor by copying an existing tensor's shape and data.
    pub(crate) fn fromtensor(name: impl Into<String>, value: &tensor<f32>) -> Self {
        Self {
            name: name.into(),
            shape: value.getshape().clone(),
            data: value.getdata().to_vec(),
        }
    }

    /// Copies this state tensor back into an owned tensor.
    ///
    /// # Errors
    ///
    /// Returns a [`tensorerror`](::tensor::tensorerror) if the stored data
    /// length disagrees with the shape, which the constructor invariant
    /// prevents.
    pub(crate) fn totensor(&self) -> Result<tensor<f32>, opterror> {
        Ok(tensor::new(self.shape.clone(), self.data.clone())?)
    }
}

/// A serializable snapshot of an optimizer's mutable state.
///
/// The snapshot is deliberately separate from any model architecture: it holds
/// the completed step count, the current learning rate, the number of tracked
/// parameters, and the per-parameter moment tensors. It maps onto a checkpoint
/// through [`statetensor`] values, so a training layer can persist and restore
/// it without giving `optim` a dependency on the checkpoint format. The
/// training state that owns `step`/`epoch` remains the checkpoint crate's
/// `trainingstate`; the two are synchronized by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct optimizerstate {
    step: u64,
    lr: f32,
    params: usize,
    tensors: Vec<statetensor>,
}

impl optimizerstate {
    /// Creates a snapshot from its scalars and moment tensors.
    #[must_use]
    pub fn new(step: u64, lr: f32, params: usize, tensors: Vec<statetensor>) -> Self {
        Self {
            step,
            lr,
            params,
            tensors,
        }
    }

    /// Returns the number of completed optimizer steps.
    #[must_use]
    pub const fn getstep(&self) -> u64 {
        self.step
    }

    /// Returns the learning rate in effect when the snapshot was taken.
    #[must_use]
    pub const fn getlr(&self) -> f32 {
        self.lr
    }

    /// Returns the number of parameters the snapshot describes.
    #[must_use]
    pub const fn getparamcount(&self) -> usize {
        self.params
    }

    /// Returns the moment tensors in the order the optimizer emits them.
    #[must_use]
    pub fn gettensors(&self) -> &[statetensor] {
        &self.tensors
    }
}
