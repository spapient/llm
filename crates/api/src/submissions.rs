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

/// A cached history and a new operand of an incremental attention step.
#[derive(Clone, Copy, Debug)]
pub struct operandpair<'a> {
    /// The accumulated cached operand.
    pub cached: operand<'a>,
    /// The new operand.
    pub new: operand<'a>,
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

    /// Records fused RMS normalization plus matrix multiplication, mirroring
    /// [`compute::rmsnormmatmul`](crate::compute::rmsnormmatmul).
    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
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

    /// Records absolute-position rotary embedding, mirroring
    /// [`compute::rope_at`](crate::compute::rope_at).
    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
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

    /// Records incremental attention, mirroring
    /// [`compute::attention_cached`](crate::compute::attention_cached).
    ///
    /// `query` is the new token's operand; `key` and `value` each pair the
    /// accumulated history with the new operand.
    fn recordattention_cached(
        &mut self,
        query: operand<'_>,
        key: operandpair<'_>,
        value: operandpair<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<stepref, backenderror>;

    /// Records a full semigroup scan, mirroring
    /// [`compute::scan`](crate::compute::scan).
    fn recordscan(
        &mut self,
        input: operand<'_>,
        init: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<stepref, backenderror>;

    /// Records a single-step incremental scan, mirroring
    /// [`compute::scan_cached`](crate::compute::scan_cached).
    fn recordscan_cached(
        &mut self,
        input: operand<'_>,
        state: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
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

/// The recording surface shared by single-use [`submission`]s and persistent
/// [`session`]s.
///
/// A model records its forward pass against this trait, so the same chain can
/// be executed by a single-use submission or by a request-scoped session that
/// keeps pinned inputs resident across decode steps. Every [`submission`]
/// implements this trait through a blanket implementation, so existing callers
/// need no change.
pub trait recorder {
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

    /// Records fused RMS normalization plus matrix multiplication, mirroring
    /// [`compute::rmsnormmatmul`](crate::compute::rmsnormmatmul).
    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
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

    /// Records absolute-position rotary embedding, mirroring
    /// [`compute::rope_at`](crate::compute::rope_at).
    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
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

    /// Records incremental attention, mirroring
    /// [`compute::attention_cached`](crate::compute::attention_cached).
    fn recordattention_cached(
        &mut self,
        query: operand<'_>,
        key: operandpair<'_>,
        value: operandpair<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<stepref, backenderror>;

    /// Records a full semigroup scan, mirroring
    /// [`compute::scan`](crate::compute::scan). The state output is not exposed;
    /// use [`recordscan_pair`](Self::recordscan_pair) when both are needed.
    fn recordscan(
        &mut self,
        input: operand<'_>,
        init: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<stepref, backenderror>;

    /// Records a single-step incremental scan, mirroring
    /// [`compute::scan_cached`](crate::compute::scan_cached).
    fn recordscan_cached(
        &mut self,
        input: operand<'_>,
        state: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<stepref, backenderror>;

    /// Pins an input operand for the lifetime of the recorder.
    ///
    /// A single-use submission ignores the request (its inputs are uploaded for
    /// that submission anyway). A persistent session uploads the operand once
    /// and reuses the device allocation across later steps, which is how model
    /// weights stop being re-uploaded on every decode token.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the upload fails.
    fn pin(&mut self, view: tensorview<'_, f32>) -> Result<(), backenderror> {
        let _ = view;
        Ok(())
    }

    /// Records a full semigroup scan and returns both the output and the final
    /// state.
    ///
    /// The default reports `unsupportedoperation`; a backend that can expose
    /// two device slots overrides it. This is the decode-chain form: the state
    /// handle can feed the next step's [`recordscan_cached`](Self::recordscan_cached).
    ///
    /// # Errors
    ///
    /// Returns the same shape/index/budget errors as the matching `compute`
    /// method, or `unsupportedoperation` when the backend cannot expose both
    /// outputs.
    fn recordscan_pair(
        &mut self,
        input: operand<'_>,
        init: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<(stepref, stepref), backenderror> {
        let _ = (input, init, a, b, c, skip);
        Err(backenderror::unsupportedoperation { op: "scan" })
    }

    /// Records a single-step incremental scan and returns both the output and
    /// the new state.
    ///
    /// The default reports `unsupportedoperation`; a backend that can expose
    /// two device slots overrides it. The state handle can feed the next
    /// decode step's [`recordscan_cached_pair`](Self::recordscan_cached_pair).
    ///
    /// # Errors
    ///
    /// Returns the same shape/index/budget errors as the matching `compute`
    /// method, or `unsupportedoperation` when the backend cannot expose both
    /// outputs.
    fn recordscan_cached_pair(
        &mut self,
        input: operand<'_>,
        state: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<(stepref, stepref), backenderror> {
        let _ = (input, state, a, b, c, skip);
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }
}

impl<T: submission> recorder for T {
    fn recordmatmul(
        &mut self,
        lhs: operand<'_>,
        rhs: operand<'_>,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordmatmul(self, lhs, rhs)
    }

    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        <T as submission>::recordadd(self, lhs, rhs)
    }

    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        <T as submission>::recordmul(self, lhs, rhs)
    }

    fn recordrmsnorm(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordrmsnorm(self, input, weight, eps)
    }

    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordrmsnormmatmul(self, input, weight, rhs, eps)
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        <T as submission>::recordsoftmax(self, input)
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        <T as submission>::recordsilu(self, input)
    }

    fn recordembedding(
        &mut self,
        table: operand<'_>,
        indices: &[usize],
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordembedding(self, table, indices)
    }

    fn recordrope(
        &mut self,
        input: operand<'_>,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordrope(self, input, heads, base)
    }

    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordrope_at(self, input, position, heads, base)
    }

    fn recordattention(
        &mut self,
        query: operand<'_>,
        key: operand<'_>,
        value: operand<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordattention(self, query, key, value, heads, kvheads, causal)
    }

    fn recordattention_cached(
        &mut self,
        query: operand<'_>,
        key: operandpair<'_>,
        value: operandpair<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordattention_cached(self, query, key, value, heads, kvheads, causal)
    }

    fn recordscan(
        &mut self,
        input: operand<'_>,
        init: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordscan(self, input, init, a, b, c, skip)
    }

    fn recordscan_cached(
        &mut self,
        input: operand<'_>,
        state: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<stepref, backenderror> {
        <T as submission>::recordscan_cached(self, input, state, a, b, c, skip)
    }
}

/// A persistent recording session.
///
/// Unlike a single-use [`submission`], a session keeps its pinned inputs
/// resident across [`apply`](Self::apply) calls, so a decode loop can record one
/// step per token without re-uploading the model weights. [`reset`](Self::reset)
/// releases every retained buffer.
pub trait session: recorder {
    /// Executes every step recorded since the last apply or reset and returns
    /// the declared outputs.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::indexoutofrange`] when an `outputs` entry does
    /// not name a recorded step.
    fn apply(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror>;

    /// Releases every retained buffer and clears the recorded steps.
    fn reset(&mut self);
}
