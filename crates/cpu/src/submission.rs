use ::tensor::{tensor, tensorview};
use api::{backenderror, compute, kvpair, operand, operandpair, stepref, submission};

/// A CPU submission that records steps by replaying [`compute`] in order.
///
/// `cpusubmission` is the reference implementation of the accumulative
/// [`submission`](api::submission) contract. It owns no execution logic beyond
/// what its [`compute`] provider already implements: each `record*` call
/// resolves its operands, invokes the matching `compute` method immediately, and
/// stores the result, so a submission reproduces the same chain a caller would
/// otherwise drive one operator at a time. The backend is owned by the
/// submission and never shared across requests.
///
/// This is deliberately the naive path: it performs exactly as many compute
/// calls as the recorded chain, because the CPU has no per-call device round
/// trip to amortize. It exists so every backend agrees on `submission`
/// semantics and so the contract can be tested without a GPU.
pub struct cpusubmission<C: compute> {
    backend: C,
    results: Vec<tensor<f32>>,
}

impl<C: compute> cpusubmission<C> {
    /// Creates a submission over `backend`.
    #[must_use]
    pub fn new(backend: C) -> Self {
        Self {
            backend,
            results: Vec::new(),
        }
    }

    /// Resolves an operand to a view.
    ///
    /// A caller-provided input passes through unchanged; a [`operand::result`]
    /// borrows the matching recorded step's tensor. A step index that does not
    /// name a recorded step is rejected rather than panicking.
    fn view<'a>(&'a self, op: operand<'a>) -> Result<tensorview<'a, f32>, backenderror> {
        match op {
            operand::input(view) => Ok(view),
            operand::result(reference) => self
                .results
                .get(reference.getindex())
                .map(|t| t.getview())
                .ok_or_else(|| backenderror::indexoutofrange {
                    index: reference.getindex(),
                    len: self.results.len(),
                }),
        }
    }

    /// Stores a computed result and returns its handle.
    fn push(&mut self, out: tensor<f32>) -> stepref {
        self.results.push(out);
        stepref::fromindex(self.results.len() - 1)
    }
}

impl<C: compute> submission for cpusubmission<C> {
    fn recordmatmul(
        &mut self,
        lhs: operand<'_>,
        rhs: operand<'_>,
    ) -> Result<stepref, backenderror> {
        let out = {
            let lhs = self.view(lhs)?;
            let rhs = self.view(rhs)?;
            self.backend.matmul(&lhs, &rhs)?
        };
        Ok(self.push(out))
    }

    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        let out = {
            let lhs = self.view(lhs)?;
            let rhs = self.view(rhs)?;
            self.backend.add(&lhs, &rhs)?
        };
        Ok(self.push(out))
    }

    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        let out = {
            let lhs = self.view(lhs)?;
            let rhs = self.view(rhs)?;
            self.backend.mul(&lhs, &rhs)?
        };
        Ok(self.push(out))
    }

    fn recordrmsnorm(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        let out = {
            let input = self.view(input)?;
            let weight = self.view(weight)?;
            self.backend.rmsnorm(&input, &weight, eps)?
        };
        Ok(self.push(out))
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let out = {
            let input = self.view(input)?;
            self.backend.softmax(&input)?
        };
        Ok(self.push(out))
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let out = {
            let input = self.view(input)?;
            self.backend.silu(&input)?
        };
        Ok(self.push(out))
    }

    fn recordembedding(
        &mut self,
        table: operand<'_>,
        indices: &[usize],
    ) -> Result<stepref, backenderror> {
        let out = {
            let table = self.view(table)?;
            self.backend.embedding(&table, indices)?
        };
        Ok(self.push(out))
    }

    fn recordrope(
        &mut self,
        input: operand<'_>,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        let out = {
            let input = self.view(input)?;
            self.backend.rope(&input, heads, base)?
        };
        Ok(self.push(out))
    }

    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        let out = {
            let input = self.view(input)?;
            self.backend.rope_at(&input, position, heads, base)?
        };
        Ok(self.push(out))
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
        let out = {
            let query = self.view(query)?;
            let key = self.view(key)?;
            let value = self.view(value)?;
            self.backend
                .attention(&query, &key, &value, heads, kvheads, causal)?
        };
        Ok(self.push(out))
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
        let out = {
            let query = self.view(query)?;
            let cachedkey = self.view(key.cached)?;
            let cachedvalue = self.view(value.cached)?;
            let newkey = self.view(key.new)?;
            let newvalue = self.view(value.new)?;
            self.backend.attention_cached(
                &query,
                kvpair {
                    cached: cachedkey,
                    new: newkey,
                },
                kvpair {
                    cached: cachedvalue,
                    new: newvalue,
                },
                heads,
                kvheads,
                causal,
            )?
        };
        Ok(self.push(out))
    }

    fn apply(self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        outputs
            .iter()
            .map(|reference| {
                self.results
                    .get(reference.getindex())
                    .cloned()
                    .ok_or_else(|| backenderror::indexoutofrange {
                        index: reference.getindex(),
                        len: self.results.len(),
                    })
            })
            .collect()
    }
}
