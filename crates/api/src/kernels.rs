use ::tensor::{element, tensor, tensorview};

use crate::backenderror;

/// A cached history and a new row of an incremental attention operand.
pub struct kvpair<'a> {
    /// The accumulated `[pos, width]` history.
    pub cached: tensorview<'a, f32>,
    /// The new `[1, width]` row.
    pub new: tensorview<'a, f32>,
}

/// A kernel contract over two operands of the same element type.
///
/// `binarykernel` extends the frozen unary [`kernel`](crate::kernel) contract
/// additively for operators such as matrix multiplication and elementwise
/// addition or multiplication. It is deliberately narrow: it reports a name and
/// applies itself to two borrowed operand views, returning a newly owned tensor
/// or a [`backenderror`]. An implementation MUST reject operands whose shapes
/// are incompatible for its operation rather than panicking.
pub trait binarykernel<T: element> {
    /// Returns the stable kernel name.
    fn getname(&self) -> &'static str;

    /// Applies the kernel to `lhs` and `rhs`.
    ///
    /// # Errors
    ///
    /// Returns a [`backenderror`] when the element dtype is unsupported, the
    /// operand shapes are incompatible with the operation, an element count
    /// overflows, or execution fails.
    fn apply(
        &self,
        lhs: &tensorview<'_, T>,
        rhs: &tensorview<'_, T>,
    ) -> Result<tensor<T>, backenderror>;
}

/// A kernel contract over an input and a weight operand of the same element
/// type.
///
/// `weightedkernel` covers normalization-style operators that scale each input
/// row by a learned weight, for example RMS normalization. The weight is a
/// separate borrowed view so its shape can be validated against the input.
pub trait weightedkernel<T: element> {
    /// Returns the stable kernel name.
    fn getname(&self) -> &'static str;

    /// Applies the kernel to `input` using `weight`.
    ///
    /// # Errors
    ///
    /// Returns a [`backenderror`] when the element dtype is unsupported, the
    /// weight shape does not match the input, an element count overflows, or
    /// execution fails.
    fn apply(
        &self,
        input: &tensorview<'_, T>,
        weight: &tensorview<'_, T>,
    ) -> Result<tensor<T>, backenderror>;
}

/// A kernel contract that gathers rows from a table by index.
///
/// `gatherkernel` covers embedding-style lookups: the first axis of `table` is
/// the row index space and every index MUST be within it. The output stacks the
/// selected rows in index order.
pub trait gatherkernel<T: element> {
    /// Returns the stable kernel name.
    fn getname(&self) -> &'static str;

    /// Gathers the rows named by `indices` from `table`.
    ///
    /// # Errors
    ///
    /// Returns a [`backenderror`] when the element dtype is unsupported, an
    /// index is outside the table's row range, an element count overflows, or
    /// execution fails.
    fn apply(
        &self,
        table: &tensorview<'_, T>,
        indices: &[usize],
    ) -> Result<tensor<T>, backenderror>;
}

/// The f32 primitive operator set consumed by the differentiation and neural
/// network layers.
///
/// `compute` is the narrow composition boundary between the workflow layers
/// (`autograd`, `nn`, `models`) and a concrete backend. It exposes exactly the
/// operators those layers compose, in terms of owned `f32` tensors, so those
/// layers never depend on a concrete backend. A backend implements this trait
/// once; the dependency graph keeps the edge from the backend to `api` only.
///
/// All methods are `f32`-only and return a [`backenderror`] on failure. Methods
/// that take attributes (an epsilon, a head count, a rotary base, a causal
/// flag) accept them as explicit parameters so no per-model configuration is
/// hidden inside a kernel instance.
///
/// Every method MUST validate its inputs and reject incompatible shapes rather
/// than panicking.
pub trait compute {
    /// Matrix multiplication.
    ///
    /// `lhs` and `rhs` MUST have equal rank `r >= 2` with equal leading batch
    /// dimensions; the last two dimensions are `[m, k]` and `[k, n]`. The
    /// result has shape `[batch..., m, n]`.
    ///
    /// The backend budget applies to the result as well as to the operands: a
    /// result whose element count or byte size exceeds the declared bound is
    /// rejected before the backend allocates it. An implementation MUST NOT
    /// accept a call whose result it cannot hold within its own budget.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] for incompatible shapes,
    /// [`backenderror::resourceexhausted`] when a budget rejects an operand or
    /// the result, or another [`backenderror`] when execution fails.
    fn matmul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror>;

    /// Elementwise addition of two equal-shaped tensors.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] when the shapes differ.
    fn add(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror>;

    /// Elementwise multiplication of two equal-shaped tensors.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] when the shapes differ.
    fn mul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror>;

    /// RMS normalization over the last axis, scaled by `weight`.
    ///
    /// `weight` MUST have the same element count as the last axis of `input`.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] when the weight does not match
    /// the last axis.
    fn rmsnorm(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        eps: f32,
    ) -> Result<tensor<f32>, backenderror>;

    /// Numerically stable softmax over the last axis.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::unsupportedoperation`] when the last axis is
    /// empty.
    fn softmax(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror>;

    /// Sigmoid linear unit, elementwise.
    fn silu(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror>;

    /// Gathers rows from `table` by `indices`.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::indexoutofrange`] when an index is outside the
    /// table's first axis.
    fn embedding(
        &self,
        table: &tensorview<'_, f32>,
        indices: &[usize],
    ) -> Result<tensor<f32>, backenderror>;

    /// Rotary position embedding.
    ///
    /// `input` has shape `[..., seq, heads * headdim]`; the second-to-last axis
    /// is the position axis and `headdim` MUST be even. Each contiguous
    /// `headdim` slice of the last axis is rotated by the position-dependent
    /// angle derived from `base`.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] when the rank is below two, the
    /// last axis is not divisible by `heads`, or `headdim` is not even.
    fn rope(
        &self,
        input: &tensorview<'_, f32>,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror>;

    /// Rotary position embedding of a single `[1, heads * headdim]` token at the
    /// absolute `position`.
    ///
    /// This is the decode-step form of [`rope`](Self::rope): the position axis
    /// is not read from the input, so a one-token query or key can be rotated at
    /// its absolute context position without materialising the padded history.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] when the input is not a single
    /// row, the last axis is not divisible by `heads`, or `headdim` is not even.
    fn rope_at(
        &self,
        input: &tensorview<'_, f32>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror>;

    /// Multi-head scaled dot-product attention with grouped-query key/value
    /// heads.
    ///
    /// `query` has shape `[seq, heads * headdim]`; `key` and `value` have shape
    /// `[seq, kvheads * headdim]`. `heads` MUST be a positive multiple of
    /// `kvheads`. When `causal` is set, position `s` attends only to positions
    /// `t <= s`.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] for inconsistent shapes or head
    /// counts.
    fn attention(
        &self,
        query: &tensorview<'_, f32>,
        key: &tensorview<'_, f32>,
        value: &tensorview<'_, f32>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<tensor<f32>, backenderror>;

    /// Incremental multi-head attention for a single new token.
    ///
    /// `query` holds one token (shape `[1, heads * headdim]`). Each of `key` and
    /// `value` pairs the accumulated `[pos, kvwidth]` history with the new
    /// `[1, kvwidth]` row; the effective operand is the cached rows followed by
    /// the new row, so the query attends over `pos + 1` positions under the
    /// causal mask. This is the decode-step shape [`attention`](Self::attention)
    /// supports with a shorter query, expressed as a pair so the cache and the
    /// new token stay in their own address spaces.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] for inconsistent shapes or head
    /// counts.
    fn attention_cached(
        &self,
        query: &tensorview<'_, f32>,
        key: kvpair<'_>,
        value: kvpair<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<tensor<f32>, backenderror>;
}
