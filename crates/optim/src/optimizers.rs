use ::tensor::tensor;

use crate::error::opterror;
use crate::state::{optimizerstate, statetensor};

/// The contract every update rule implements.
///
/// An optimizer owns only its own mutable state (moment buffers and a step
/// counter); it never owns model architecture and never stores request-scoped
/// data. [`Self::update`] receives the model's parameters and their gradients as
/// two parallel borrowed slices, in a caller-defined and stable order. The
/// optimizer keys its per-parameter state by that position, so the same order
/// must be used for every step and for [`Self::restore`].
///
/// The interface is decoupled from `autograd`: a gradient is any
/// `tensor<f32>`, whether it came from a reverse pass or from another source.
/// This keeps `optim` independent of the graph representation.
pub trait optimizer {
    /// Returns the stable optimizer name (`"adamw"` or `"sgd"`).
    fn getname(&self) -> &'static str;

    /// Returns the learning rate in effect for the next update.
    fn getlr(&self) -> f32;

    /// Sets the learning rate in effect for the next update.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `lr` is not finite or
    /// is negative.
    fn setlr(&mut self, lr: f32) -> Result<(), opterror>;

    /// Returns the number of completed updates.
    fn getstep(&self) -> u64;

    /// Applies one update to `params` using `grads`.
    ///
    /// `params` and `grads` must have the same length and, pairwise, the same
    /// shape. When a clipping configuration is set on the optimizer, the
    /// gradients are clipped in place before the update is applied.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::mismatchedcount`] when the slice lengths differ,
    /// [`opterror::shapemismatch`] when a gradient's shape differs from its
    /// parameter's shape, or [`opterror::invalidstate`] when already-initialized
    /// state does not match the supplied parameters. Validation completes before
    /// any tensor is mutated, so a rejected call leaves parameters, gradients,
    /// and state unchanged.
    fn update(
        &mut self,
        params: &mut [tensor<f32>],
        grads: &mut [tensor<f32>],
    ) -> Result<(), opterror>;

    /// Returns a snapshot of the optimizer's mutable state.
    fn getstate(&self) -> optimizerstate;

    /// Restores the optimizer's mutable state from a snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidstate`] when the snapshot's tensor count,
    /// names, element counts, or learning rate are inconsistent.
    fn restore(&mut self, state: &optimizerstate) -> Result<(), opterror>;
}

/// Rejects parameter/gradient slice lengths that differ.
pub(crate) fn checkcount(params: &[tensor<f32>], grads: &[tensor<f32>]) -> Result<(), opterror> {
    if params.len() != grads.len() {
        return Err(opterror::mismatchedcount {
            params: params.len(),
            grads: grads.len(),
        });
    }
    Ok(())
}

/// Rejects any gradient whose shape differs from its parameter's shape.
pub(crate) fn checkshapes(params: &[tensor<f32>], grads: &[tensor<f32>]) -> Result<(), opterror> {
    for (index, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
        if param.getshape().getdims() != grad.getshape().getdims() {
            return Err(opterror::shapemismatch {
                index,
                param: param.getshape().getdims().to_vec(),
                grad: grad.getshape().getdims().to_vec(),
            });
        }
    }
    Ok(())
}

/// Rejects a moment-buffer set that does not match the supplied parameters.
pub(crate) fn checkslots(
    slots: &[tensor<f32>],
    params: &[tensor<f32>],
    label: &str,
) -> Result<(), opterror> {
    if slots.len() != params.len() {
        return Err(opterror::state(format!(
            "{label} holds {} slots but {} parameters were supplied",
            slots.len(),
            params.len()
        )));
    }
    for (index, (slot, param)) in slots.iter().zip(params.iter()).enumerate() {
        if slot.getshape().getdims() != param.getshape().getdims() {
            return Err(opterror::state(format!(
                "{label} slot {index} has shape {:?} but its parameter has shape {:?}",
                slot.getshape().getdims(),
                param.getshape().getdims()
            )));
        }
    }
    Ok(())
}

/// Rejects a learning rate that is not finite and non-negative.
pub(crate) fn checklr(lr: f32) -> Result<(), opterror> {
    if !lr.is_finite() || lr < 0.0 {
        return Err(opterror::hyperparameter(
            "lr",
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Rejects a value that is not finite and non-negative.
pub(crate) fn checknonnegative(field: &'static str, value: f32) -> Result<(), opterror> {
    if !value.is_finite() || value < 0.0 {
        return Err(opterror::hyperparameter(
            field,
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Builds a zero-filled tensor with the same shape as `like`.
pub(crate) fn zerolike(like: &tensor<f32>) -> Result<tensor<f32>, opterror> {
    let count = like.getshape().getelementcount()?;
    Ok(tensor::new(like.getshape().clone(), vec![0.0; count])?)
}

/// Parses the numeric suffix of a slot name such as `m.3` under `prefix`.
pub(crate) fn slotindex(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse::<usize>().ok()
}

/// Decodes one state tensor into a fixed-size slot vector, rejecting a bad
/// index or a duplicate.
pub(crate) fn setslot(
    slots: &mut [Option<tensor<f32>>],
    index: usize,
    entry: &statetensor,
    label: &str,
) -> Result<(), opterror> {
    let slot = slots
        .get_mut(index)
        .ok_or_else(|| opterror::state(format!("{label} index {index} is out of range")))?;
    if slot.is_some() {
        return Err(opterror::state(format!("duplicate {label} slot {index}")));
    }
    *slot = Some(entry.totensor()?);
    Ok(())
}

/// Converts a fully populated slot vector, rejecting any missing entry.
pub(crate) fn collectslots(
    slots: Vec<Option<tensor<f32>>>,
    label: &str,
) -> Result<Vec<tensor<f32>>, opterror> {
    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.ok_or_else(|| opterror::state(format!("missing {label} slot {index}")))
        })
        .collect()
}
