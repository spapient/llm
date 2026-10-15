use ::tensor::{tensor, tensorview};

use crate::backenderror;

/// An opaque handle to a step recorded on a [`submission`].
///
/// A `stepref` is produced by the `record*` methods and consumed either as an
/// operand of a later step ([`operand::result`]) or as a requested output of
/// [`submission::apply`]. The wrapped index is private, so the only way to
/// obtain a `stepref` is to record the step it names; a caller cannot invent a
/// reference to a step it did not declare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct stepref(usize);

impl stepref {
    /// Wraps a step index as a handle.
    ///
    /// Only a `submission` implementation records steps, so a caller should
    /// obtain a `stepref` from a `record*` method rather than construct one;
    /// implementations use this to name the step they just recorded.
    #[must_use]
    pub fn fromindex(index: usize) -> Self {
        Self(index)
    }

    /// Returns the wrapped step index.
    ///
    /// Intended for `submission` implementations to resolve
    /// [`operand::result`] and validate [`submission::apply`] outputs against
    /// the steps they recorded.
    #[must_use]
    pub fn getindex(&self) -> usize {
        self.0
    }
}

/// The source of one operand in a recorded step.
///
/// An operand is either an input the caller provides outright
/// ([`operand::input`]) or the result of an earlier recorded step
/// ([`operand::result`]). This mirrors the two ownership directions a step can
/// read from: data owned by the caller, or data owned by the submission's
/// intermediate state.
#[derive(Clone, Copy, Debug)]
pub enum operand<'a> {
    /// A caller-provided input view.
    input(tensorview<'a, f32>),
    /// The result of a previously recorded step.
    result(stepref),
}

/// An accumulative submission boundary over the [`compute`](crate::compute)
/// operator set.
///
/// `submission` lets a caller record a chain of steps and execute them with one
/// submit instead of one synchronous round trip per operator. The frozen
/// [`compute`](crate::compute) contract is unchanged and remains the default
/// path; a backend implements this contract once, alongside `compute`, reusing
/// the same operator set and error type. The CPU backend records by replaying
/// `compute` in order, so the three backends agree on semantics while each
/// submits in the way its hardware rewards (docs/design.md §21.13).
///
/// Each `record*` method names the operator it records, mirrors the matching
/// [`compute`](crate::compute) method's operands and attributes, and returns a
/// [`stepref`] to its result. Steps execute in declaration order, and a step may
/// reference the result of any earlier step. [`apply`](Self::apply) consumes the
/// submission, executes every recorded step, and returns only the declared
/// outputs, in the order requested.
///
/// The submission owns all of the request's intermediate buffers and its single
/// command submission; it never owns model weights and never shares mutable
/// state between requests. Declaring a step validates every operand shape and
/// the result against the backend budget through [`checkbudget`](crate::checkbudget),
/// so a declaration failure performs no device work.
///
/// # Errors
///
/// A `record*` method returns the same [`backenderror`] variant its matching
/// [`compute`](crate::compute) method would for a rejected shape, index, or
/// budget. [`apply`](Self::apply) returns [`backenderror::indexoutofrange`] when
/// an `outputs` entry names a step that was never recorded.
pub trait submission {
    /// Records matrix multiplication, mirroring [`compute::matmul`](crate::compute::matmul).
    fn recordmatmul(&mut self, lhs: operand<'_>, rhs: operand<'_>)
    -> Result<stepref, backenderror>;

    /// Records elementwise addition, mirroring [`compute::add`](crate::compute::add).
    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror>;

    /// Records elementwise multiplication, mirroring [`compute::mul`](crate::compute::mul).
    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror>;

    /// Records RMS normalization, mirroring [`compute::rmsnorm`](crate::compute::rmsnorm).
    fn recordrmsnorm(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror>;

    /// Records last-axis softmax, mirroring [`compute::softmax`](crate::compute::softmax).
    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror>;

    /// Records the sigmoid linear unit, mirroring [`compute::silu`](crate::compute::silu).
    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror>;

    /// Records a row gather, mirroring [`compute::embedding`](crate::compute::embedding).
    fn recordembedding(
        &mut self,
        table: operand<'_>,
        indices: &[usize],
    ) -> Result<stepref, backenderror>;

    /// Records rotary position embedding, mirroring [`compute::rope`](crate::compute::rope).
    fn recordrope(
        &mut self,
        input: operand<'_>,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror>;

    /// Records grouped-query attention, mirroring [`compute::attention`](crate::compute::attention).
    fn recordattention(
        &mut self,
        query: operand<'_>,
        key: operand<'_>,
        value: operand<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<stepref, backenderror>;

    /// Executes every recorded step and returns the declared outputs.
    ///
    /// The returned tensors match `outputs` in order and count. Consuming `self`
    /// makes the submission single-use: after `apply` there is nothing left to
    /// record into or re-execute, and dropping the submission before `apply`
    /// releases every intermediate buffer it owns.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::indexoutofrange`] when an `outputs` entry does
    /// not name a recorded step.
    fn apply(self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror>;
}
