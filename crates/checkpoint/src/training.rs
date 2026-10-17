use std::collections::BTreeMap;

use lib::{jsonnumber, jsonvalue};

use crate::error::checkpointerror;

/// A snapshot of training progress stored in a checkpoint.
///
/// The state is deliberately minimal and owned by this crate: it records the
/// completed optimizer step and epoch so a training run can resume. Optimizer
/// parameter state is not modeled here; that belongs to the optimizer crate and
/// is persisted as ordinary tensors through the same checkpoint container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct trainingstate {
    step: u64,
    epoch: u64,
}

impl trainingstate {
    /// Creates a training-state snapshot.
    #[must_use]
    pub const fn new(step: u64, epoch: u64) -> Self {
        Self { step, epoch }
    }

    /// Returns the number of completed optimizer steps.
    #[must_use]
    pub const fn getstep(self) -> u64 {
        self.step
    }

    /// Returns the number of completed epochs.
    #[must_use]
    pub const fn getepoch(self) -> u64 {
        self.epoch
    }

    /// Serializes the state to a JSON object value.
    #[must_use]
    pub(crate) fn tojson(self) -> jsonvalue {
        let mut object = BTreeMap::new();
        object.insert(
            "step".to_string(),
            jsonvalue::number(jsonnumber::integer(self.step as i64)),
        );
        object.insert(
            "epoch".to_string(),
            jsonvalue::number(jsonnumber::integer(self.epoch as i64)),
        );
        jsonvalue::object(object)
    }

    /// Parses the state from a JSON object value.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] when the value is not a
    /// `{"step":N,"epoch":N}` object with exact integer fields.
    pub(crate) fn fromjson(value: &jsonvalue) -> Result<Self, checkpointerror> {
        let object = value
            .getobject()
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "training state is not a JSON object".to_string(),
            })?;
        let step = object
            .get("step")
            .and_then(jsonvalue::getinteger)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "training state 'step' is not an exact non-negative integer".to_string(),
            })?;
        let epoch = object
            .get("epoch")
            .and_then(jsonvalue::getinteger)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "training state 'epoch' is not an exact non-negative integer".to_string(),
            })?;
        Ok(Self { step, epoch })
    }
}
