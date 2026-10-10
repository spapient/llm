use ::tensor::tensor;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::ops::{self, oneslike};

impl graph<f32> {
    /// Runs one reverse pass from `output` with an all-ones seed.
    ///
    /// This seeds `dOutput/dOutput = 1`, which is the correct starting point
    /// when the output is a scalar loss or when the caller only needs the
    /// unweighted gradients.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `output` is not on the
    /// tape, or an operator error if the tape was built inconsistently.
    pub fn backward(&mut self, output: variableid) -> Result<(), autoggraderror> {
        self.backwardwith(output, None)
    }

    /// Runs one reverse pass from `output` with an explicit seed.
    ///
    /// The seed (the upstream gradient, often written `dY`) must have the same
    /// shape as the output value. Passing `None` uses an all-ones seed. When
    /// `output` does not require a gradient the call is a no-op, so an
    /// inference-only graph needs no special casing.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] when `output` is not on the
    /// tape, [`autoggraderror::unsupportedshape`] when the seed shape differs
    /// from the output shape, or an operator error if the tape was built
    /// inconsistently.
    pub fn backwardwith(
        &mut self,
        output: variableid,
        seed: Option<tensor<f32>>,
    ) -> Result<(), autoggraderror> {
        let node = self.check(output)?;
        if !node.requiresgrad() {
            return Ok(());
        }
        let value = node.getvalue().clone();
        let seed = match seed {
            Some(given) => {
                if given.getshape().getdims() != value.getshape().getdims() {
                    return Err(autoggraderror::unsupportedshape {
                        op: "backward",
                        dims: given.getshape().getdims().to_vec(),
                    });
                }
                given
            }
            None => oneslike(&value)?,
        };
        self.setgrad(output, seed)?;

        let top = output.getindex();
        for index in (0..=top).rev() {
            let id = variableid::new(index);
            let Some(node) = self.get(id) else {
                continue;
            };
            if !node.requiresgrad() {
                continue;
            }
            let Some(gy) = node.getgrad().cloned() else {
                continue;
            };
            let producer = node.getproducer();
            let value = node.getvalue().clone();
            match producer {
                op::leaf => {}
                op::matmul { lhs, rhs } => ops::matmul::backward(self, lhs, rhs, &gy)?,
                op::add { lhs, rhs } => ops::elementwise::backwardadd(self, lhs, rhs, &gy)?,
                op::mul { lhs, rhs } => ops::elementwise::backwardmul(self, lhs, rhs, &gy)?,
                op::rmsnorm { input, weight, eps } => {
                    ops::normalization::backward(self, input, weight, eps, &gy)?;
                }
                op::softmax { input } => ops::softmax::backward(self, input, &gy, &value)?,
                op::relu { input } => ops::activation::backwardrelu(self, input, &gy)?,
                op::silu { input } => ops::activation::backwardsilu(self, input, &gy)?,
                op::gelutanh { input } => ops::activation::backwardgelutanh(self, input, &gy)?,
            }
        }
        Ok(())
    }
}
