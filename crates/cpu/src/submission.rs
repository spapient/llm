use ::tensor::{shape, tensor, tensorview};
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
    /// Caller data registered through `recordpersist`, keyed by
    /// `(pointer, length)`: the CPU reference reuses the same result slot for
    /// the same registered buffer, mirroring the engine's device-side
    /// persistent table, so a device cache that hands the same host buffer
    /// back every step keeps one snapshot.
    persistent: std::collections::HashMap<(usize, usize), usize>,
}

impl<C: compute> cpusubmission<C> {
    /// Creates a submission over `backend`.
    #[must_use]
    pub fn new(backend: C) -> Self {
        Self {
            backend,
            results: Vec::new(),
            persistent: std::collections::HashMap::new(),
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

    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        let out = {
            let input = self.view(input)?;
            let weight = self.view(weight)?;
            let rhs = self.view(rhs)?;
            self.backend.rmsnormmatmul(&input, &weight, &rhs, eps)?
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

    fn recordscan(
        &mut self,
        _input: operand<'_>,
        _init: operand<'_>,
        _a: operand<'_>,
        _b: operand<'_>,
        _c: operand<'_>,
        _skip: operand<'_>,
    ) -> Result<stepref, backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan" })
    }

    fn recordscan_cached(
        &mut self,
        _input: operand<'_>,
        _state: operand<'_>,
        _a: operand<'_>,
        _b: operand<'_>,
        _c: operand<'_>,
        _skip: operand<'_>,
    ) -> Result<stepref, backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }

    fn recordpersist(&mut self, data: &[f32], shape: shape) -> Result<stepref, backenderror> {
        let count = shape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        if data.len() != count {
            return Err(backenderror::invalidparameter { op: "persist" });
        }
        let key = (data.as_ptr() as usize, data.len());
        if let Some(&index) = self.persistent.get(&key) {
            return Ok(stepref::fromindex(index));
        }
        let t = tensor::new(shape, data.to_vec())
            .map_err(|_| backenderror::invalidparameter { op: "persist" })?;
        let index = self.push(t).getindex();
        self.persistent.insert(key, index);
        Ok(stepref::fromindex(index))
    }

    fn recordappend(
        &mut self,
        cached: operand<'_>,
        new: operand<'_>,
        position: usize,
    ) -> Result<(), backenderror> {
        let index = match cached {
            operand::result(reference) => reference.getindex(),
            operand::input(_) => {
                return Err(backenderror::invalidparameter { op: "append" });
            }
        };
        let newdata: Vec<f32> = self.view(new)?.getdata().to_vec();
        let width = newdata.len();
        if width == 0 {
            return Err(backenderror::invalidparameter { op: "append" });
        }
        let len = self.results.len();
        let cached_t = self
            .results
            .get_mut(index)
            .ok_or(backenderror::indexoutofrange { index, len })?;
        let start = position
            .checked_mul(width)
            .ok_or(backenderror::sizeoverflow)?;
        let end = start.checked_add(width).ok_or(backenderror::sizeoverflow)?;
        if end > cached_t.getlen() {
            return Err(backenderror::invalidparameter { op: "append" });
        }
        cached_t.getdatamut()[start..end].copy_from_slice(&newdata);
        Ok(())
    }

    fn recordview(&mut self, base: stepref, shape: shape) -> Result<stepref, backenderror> {
        let index = base.getindex();
        let t = self
            .results
            .get(index)
            .ok_or(backenderror::indexoutofrange {
                index,
                len: self.results.len(),
            })?;
        let count = shape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        if count > t.getlen() {
            return Err(backenderror::invalidparameter { op: "view" });
        }
        let sub = tensor::new(shape, t.getdata()[..count].to_vec())
            .map_err(|_| backenderror::invalidparameter { op: "view" })?;
        Ok(self.push(sub))
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
