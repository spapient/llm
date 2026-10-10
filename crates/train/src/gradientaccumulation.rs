use tensor::tensor;

use crate::error::trainerror;

/// A fixed-capacity buffer that averages gradients over a window of micro-batches.
///
/// Gradient accumulation trades memory for an effective batch size: instead of
/// one optimizer update per micro-batch, [`Self::accumulate`] sums the gradients
/// of `capacity` micro-batches and [`Self::take`] hands back their arithmetic
/// mean, scaling each element by `1 / count`. Because every micro-batch enters
/// the sum with equal weight, a window of ragged micro-batches still produces a
/// balanced gradient.
///
/// The shapes are frozen at construction from the parameter set, so a
/// mismatched gradient is rejected before any element is added
/// (`gradientcountmismatch`/`gradientshapemismatch`) and the buffer is never
/// left partially updated. Adding to a full buffer is also rejected
/// (`accumulatorfull`); a caller that observes an error and cannot continue must
/// call [`Self::reset`] so no half-window survives into the next attempt.
#[derive(Clone, Debug)]
pub struct gradientaccumulator {
    accum: Vec<tensor<f32>>,
    count: usize,
    capacity: usize,
}

impl gradientaccumulator {
    /// Builds a zeroed accumulator matching `params`.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::emptyparameters`] when `params` is empty and
    /// [`trainerror::invalidaccumulation`] when `capacity` is zero.
    pub fn new(params: &[tensor<f32>], capacity: usize) -> Result<Self, trainerror> {
        if capacity == 0 {
            return Err(trainerror::invalidaccumulation { found: 0 });
        }
        if params.is_empty() {
            return Err(trainerror::emptyparameters);
        }
        let accum = params
            .iter()
            .map(|param| {
                tensor::new(
                    param.getshape().clone(),
                    vec![0.0_f32; param.getdata().len()],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            accum,
            count: 0,
            capacity,
        })
    }

    /// Returns the number of micro-batches the window averages.
    #[must_use]
    pub const fn getcapacity(&self) -> usize {
        self.capacity
    }

    /// Returns how many micro-batches have been accumulated so far.
    #[must_use]
    pub const fn getcount(&self) -> usize {
        self.count
    }

    /// Returns `true` when no micro-batch is buffered.
    #[must_use]
    pub const fn isempty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` when the window holds a full set of micro-batches.
    #[must_use]
    pub const fn isfull(&self) -> bool {
        self.count >= self.capacity
    }

    /// Adds one micro-batch's gradients to the window.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::accumulatorfull`] when the window already holds
    /// `capacity` micro-batches, [`trainerror::gradientcountmismatch`] when
    /// `grads` has a different length than the parameter set, and
    /// [`trainerror::gradientshapemismatch`] when a gradient's shape differs
    /// from its parameter's shape. On any error the buffered sum is unchanged.
    pub fn accumulate(&mut self, grads: &[tensor<f32>]) -> Result<(), trainerror> {
        if self.isfull() {
            return Err(trainerror::accumulatorfull);
        }
        self.validate(grads)?;
        for (accum, grad) in self.accum.iter_mut().zip(grads.iter()) {
            for (slot, value) in accum.getdatamut().iter_mut().zip(grad.getdata().iter()) {
                *slot += *value;
            }
        }
        self.count += 1;
        Ok(())
    }

    /// Returns the mean gradient of the buffered window and empties it.
    ///
    /// The sum is scaled in place by `1 / count` before being moved out, so the
    /// returned tensors are freshly owned and the buffer is left zeroed and
    /// ready for the next window.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::invalidstate`] when the window is empty.
    pub fn take(&mut self) -> Result<Vec<tensor<f32>>, trainerror> {
        if self.count == 0 {
            return Err(trainerror::state(
                "no micro-batch is accumulated to take a gradient from",
            ));
        }
        let scale = 1.0 / self.count as f32;
        let mut out = Vec::with_capacity(self.accum.len());
        for accum in self.accum.iter_mut() {
            for value in accum.getdatamut().iter_mut() {
                *value *= scale;
            }
            let zeros = tensor::new(
                accum.getshape().clone(),
                vec![0.0_f32; accum.getdata().len()],
            )?;
            out.push(core::mem::replace(accum, zeros));
        }
        self.count = 0;
        Ok(out)
    }

    /// Discards any buffered micro-batches and zeroes the window.
    pub fn reset(&mut self) {
        for accum in self.accum.iter_mut() {
            for value in accum.getdatamut().iter_mut() {
                *value = 0.0;
            }
        }
        self.count = 0;
    }

    fn validate(&self, grads: &[tensor<f32>]) -> Result<(), trainerror> {
        if grads.len() != self.accum.len() {
            return Err(trainerror::gradientcountmismatch {
                expected: self.accum.len(),
                actual: grads.len(),
            });
        }
        for (index, (accum, grad)) in self.accum.iter().zip(grads.iter()).enumerate() {
            if accum.getshape().getdims() != grad.getshape().getdims() {
                return Err(trainerror::gradientshapemismatch {
                    index,
                    param: accum.getshape().getdims().to_vec(),
                    grad: grad.getshape().getdims().to_vec(),
                });
            }
        }
        Ok(())
    }
}
