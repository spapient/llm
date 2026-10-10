use ::tensor::tensor;
use api::compute;

use crate::dictionary::statedict;
use crate::error::nnerror;
use crate::layers::{expandrows, leadingcount, reshape};
use crate::modules::module;
use crate::parameters::parameter;

/// A dense linear projection `y = x * W + b`.
///
/// The weight is stored as `[in_features, out_features]` and the optional bias
/// as `[out_features]`. Storing the weight in that order lets the forward pass
/// use `api::compute::matmul` directly, with no transpose and no extra
/// allocation. Leading batch dimensions of the input are preserved: an input of
/// shape `[..., in_features]` produces `[..., out_features]`.
#[derive(Clone, Debug, PartialEq)]
pub struct linear {
    weight: parameter,
    bias: Option<parameter>,
}

impl linear {
    /// Builds a linear layer from a weight and an optional bias.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] unless the weight is rank two and, if
    /// present, the bias is rank one with length equal to the output features.
    pub fn new(weight: parameter, bias: Option<parameter>) -> Result<Self, nnerror> {
        let wd = weight.getshape().getdims();
        if wd.len() != 2 {
            return Err(nnerror::shapemismatch {
                layer: "linear",
                expected: vec![wd.first().copied().unwrap_or(0), 0],
                actual: wd.to_vec(),
            });
        }
        let out = wd[1];
        if let Some(b) = &bias {
            let bd = b.getshape().getdims();
            if bd.len() != 1 || bd[0] != out {
                return Err(nnerror::shapemismatch {
                    layer: "linear",
                    expected: vec![out],
                    actual: bd.to_vec(),
                });
            }
        }
        Ok(Self { weight, bias })
    }

    /// Returns the input feature count.
    #[must_use]
    pub fn getinfeatures(&self) -> usize {
        self.weight.getshape().getdims()[0]
    }

    /// Returns the output feature count.
    #[must_use]
    pub fn getoutfeatures(&self) -> usize {
        self.weight.getshape().getdims()[1]
    }

    /// Returns the weight parameter.
    #[must_use]
    pub fn getweight(&self) -> &parameter {
        &self.weight
    }

    /// Returns the bias parameter, if present.
    #[must_use]
    pub fn getbias(&self) -> Option<&parameter> {
        self.bias.as_ref()
    }

    /// Applies the projection to `input` of shape `[..., in_features]`.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] when the last dimension is not the
    /// input feature count, or forwards the provider's error.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        input: &tensor<f32>,
    ) -> Result<tensor<f32>, nnerror> {
        let dims = input.getshape().getdims();
        let in_features = self.getinfeatures();
        let out_features = self.getoutfeatures();
        let last = dims[dims.len() - 1];
        if last != in_features {
            let mut expected = dims.to_vec();
            expected[dims.len() - 1] = in_features;
            return Err(nnerror::shapemismatch {
                layer: "linear",
                expected,
                actual: dims.to_vec(),
            });
        }
        let rows = leadingcount(&dims[..dims.len() - 1])?;
        let flat = reshape(input.getdata().to_vec(), vec![rows, in_features])?;
        let mut projected = engine.matmul(&flat.getview(), &self.weight.getvalue().getview())?;
        if let Some(b) = &self.bias {
            let expanded = expandrows(b.getvalue().getdata(), rows)?;
            projected = engine.add(&projected.getview(), &expanded.getview())?;
        }
        let mut outdims = dims[..dims.len() - 1].to_vec();
        outdims.push(out_features);
        reshape(projected.getdata().to_vec(), outdims)
    }
}

impl module for linear {
    fn getname(&self) -> &'static str {
        "linear"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        dict.add(prefix, self.weight.clone())?;
        if let Some(b) = &self.bias {
            dict.add(prefix, b.clone())?;
        }
        Ok(())
    }
}
