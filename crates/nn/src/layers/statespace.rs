use ::tensor::tensor;
use api::compute;

use crate::dictionary::statedict;
use crate::error::nnerror;
use crate::modules::module;
use crate::parameters::parameter;

/// A non-selective state-space layer `h_t = A h_{t-1} + B x_t`,
/// `y_t = C h_t + D x_t` with diagonal `A`, per channel, elementwise over the
/// state width.
///
/// The parameters are `a`/`b`/`c` (`[d, N]`) and `skip` (`[d]`). The forward
/// dispatches one [`compute::scan`], returning the `[n, d]` output and the
/// `[d, N]` final state; the spectral precondition `|a| <= 1` is enforced by the
/// backend's scan, not here. This is the `arch: trap` base, and the same layer
/// any linear-recurrence model mounts onto.
#[derive(Clone, Debug, PartialEq)]
pub struct ssm {
    a: parameter,
    b: parameter,
    c: parameter,
    skip: parameter,
}

impl ssm {
    /// Builds a state-space layer from its four parameters.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] unless `a`/`b`/`c` are rank two with
    /// equal shape and `skip` is rank one with length equal to the channel count.
    pub fn new(a: parameter, b: parameter, c: parameter, skip: parameter) -> Result<Self, nnerror> {
        let ad = a.getshape().getdims();
        let bd = b.getshape().getdims();
        let cd = c.getshape().getdims();
        if ad.len() != 2 || bd.len() != 2 || cd.len() != 2 {
            return Err(nnerror::shapemismatch {
                layer: "ssm",
                expected: vec![0, 0],
                actual: ad.to_vec(),
            });
        }
        if bd != ad || cd != ad {
            return Err(nnerror::shapemismatch {
                layer: "ssm",
                expected: ad.to_vec(),
                actual: bd.to_vec(),
            });
        }
        let skipd = skip.getshape().getdims();
        if skipd.len() != 1 || skipd[0] != ad[0] {
            return Err(nnerror::shapemismatch {
                layer: "ssm",
                expected: vec![ad[0]],
                actual: skipd.to_vec(),
            });
        }
        Ok(Self { a, b, c, skip })
    }

    /// Returns the channel count.
    #[must_use]
    pub fn getchannels(&self) -> usize {
        self.a.getshape().getdims()[0]
    }

    /// Returns the state width.
    #[must_use]
    pub fn getstatewidth(&self) -> usize {
        self.a.getshape().getdims()[1]
    }

    /// Returns the recurrence kernel parameter.
    #[must_use]
    pub fn geta(&self) -> &parameter {
        &self.a
    }

    /// Applies the scan to `input` (`[n, d]`) from `state` (`[d, N]`), returning
    /// the `[n, d]` output and the `[d, N]` final state.
    ///
    /// # Errors
    ///
    /// Forwards the backend's shape or spectral error.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        input: &tensor<f32>,
        state: &tensor<f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), nnerror> {
        Ok(engine.scan(
            &input.getview(),
            &state.getview(),
            &self.a.getvalue().getview(),
            &self.b.getvalue().getview(),
            &self.c.getvalue().getview(),
            &self.skip.getvalue().getview(),
        )?)
    }
}

impl module for ssm {
    fn getname(&self) -> &'static str {
        "ssm"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        dict.add(prefix, self.a.clone())?;
        dict.add(prefix, self.b.clone())?;
        dict.add(prefix, self.c.clone())?;
        dict.add(prefix, self.skip.clone())?;
        Ok(())
    }
}
