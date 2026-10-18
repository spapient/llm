//! Reconstructing model weights from a snapshot.
//!
//! `checkpoint` owns model persistence (AGENTS.md §3.1), so restoring a
//! snapshot into a model is this crate's responsibility, not `models`'.
//! [`loadllamaweights`] reads every weight the Llama architecture declares —
//! using the same names [`llamaweights::collect`] writes — and rebuilds the
//! validated weight tree.
//!
//! [`weightview`] is the zero-copy consumer channel: for a raw, page-aligned
//! tensor the snapshot bytes alias the memory map directly (no element-wise
//! decode, no owned buffer), and a consumer such as a GPU weight-upload path
//! can hand the resulting [`tensorview`] straight to `api::operand::input`.
//! Tensors that need decoding (sparse, codec, transform, or a non-contiguous
//! layout) fall back to an owned materialization with the same view surface.

use std::borrow::Cow;

use ::nn::{embedding, linear, mlp, parameter, rmsnorm};
use ::tensor::{layout, shape, tensor, tensorview};
use models::{llamaconfig, llamalayerweights, llamaweights};

use crate::{checkpointerror, checkpointreader};

/// A loaded weight in either its zero-copy or its materialized form.
///
/// `borrowed` aliases the snapshot's memory map for raw tensors: the shape and
/// layout are the configuration-derived values (copied once — they are a few
/// words), while `data` is the mmap slice itself, so no element buffer is ever
/// allocated for the common case. `owned` carries a decoded fallback.
#[derive(Debug)]
pub enum weightview<'a> {
    /// A zero-copy alias of a raw, page-aligned snapshot tensor.
    borrowed {
        /// The configuration-derived shape.
        shape: shape,
        /// The contiguous layout for `shape`.
        layout: layout,
        /// The memory-mapped element slice.
        data: &'a [f32],
    },
    /// A decoded fallback (sparse, codec, transform, or non-contiguous).
    owned(tensor<f32>),
}

impl<'a> weightview<'a> {
    /// Returns a view over this weight, whichever storage it uses.
    ///
    /// The view borrows the `weightview` itself, so a consumer must apply it
    /// within the borrow of this value — the natural pattern for a GPU upload
    /// (record into a submission, then submit).
    #[must_use]
    pub fn getview(&self) -> tensorview<'_, f32> {
        match self {
            Self::borrowed {
                shape,
                layout,
                data,
            } => tensorview::fromparts(shape, layout, data),
            Self::owned(t) => t.getview(),
        }
    }
}

/// One weight the Llama architecture declares, with the shape it must have.
struct weightspec {
    name: String,
    dims: Vec<usize>,
}

/// The full Llama weight inventory: every name [`llamaweights::collect`]
/// writes, in collect order, with the configuration-derived shape.
fn weightspecs(config: &llamaconfig) -> Vec<weightspec> {
    let mut specs = vec![weightspec {
        name: "embedding.weight".to_string(),
        dims: vec![config.getvocab(), config.getdim()],
    }];
    for i in 0..config.getnlayers() {
        let prefix = format!("layers.{i}");
        specs.extend([
            weightspec {
                name: format!("{prefix}.attn_norm.weight"),
                dims: vec![config.getdim()],
            },
            weightspec {
                name: format!("{prefix}.wq.weight"),
                dims: vec![config.getdim(), config.getdim()],
            },
            weightspec {
                name: format!("{prefix}.wk.weight"),
                dims: vec![config.getdim(), config.getkvwidth()],
            },
            weightspec {
                name: format!("{prefix}.wv.weight"),
                dims: vec![config.getdim(), config.getkvwidth()],
            },
            weightspec {
                name: format!("{prefix}.wo.weight"),
                dims: vec![config.getdim(), config.getdim()],
            },
            weightspec {
                name: format!("{prefix}.ffn_norm.weight"),
                dims: vec![config.getdim()],
            },
            weightspec {
                name: format!("{prefix}.ffn.gate.weight"),
                dims: vec![config.getdim(), config.gethidden()],
            },
            weightspec {
                name: format!("{prefix}.ffn.up.weight"),
                dims: vec![config.getdim(), config.gethidden()],
            },
            weightspec {
                name: format!("{prefix}.ffn.down.weight"),
                dims: vec![config.gethidden(), config.getdim()],
            },
        ]);
    }
    specs.extend([
        weightspec {
            name: "norm.weight".to_string(),
            dims: vec![config.getdim()],
        },
        weightspec {
            name: "output.weight".to_string(),
            dims: vec![config.getdim(), config.getvocab()],
        },
    ]);
    specs
}

/// Validates every declared weight's snapshot shape, returning the first
/// mismatch as [`checkpointerror::shapemismatch`].
fn validate(reader: &checkpointreader, specs: &[weightspec]) -> Result<(), checkpointerror> {
    for spec in specs {
        let actual =
            reader
                .getentryshape(&spec.name)
                .ok_or_else(|| checkpointerror::missingtensor {
                    name: spec.name.clone(),
                })?;
        if actual.getdims() != spec.dims.as_slice() {
            return Err(checkpointerror::shapemismatch {
                name: spec.name.clone(),
                expected: spec.dims.clone(),
                actual: actual.getdims().to_vec(),
            });
        }
    }
    Ok(())
}

/// Reads one weight as an owned tensor.
fn readowned(reader: &checkpointreader, spec: &weightspec) -> Result<tensor<f32>, checkpointerror> {
    let s = shape::new(spec.dims.clone()).map_err(checkpointerror::tensor)?;
    let data = reader.readtensorview::<f32>(&spec.name)?.ok_or_else(|| {
        checkpointerror::missingtensor {
            name: spec.name.clone(),
        }
    })?;
    tensor::new(s, data.into_owned()).map_err(checkpointerror::tensor)
}

/// Rebuilds [`llamaweights`] for `config` from the tensors in `reader`.
///
/// Every declared weight must exist with the configuration's dtype and shape;
/// a missing tensor is [`checkpointerror::missingtensor`], a shape conflict is
/// [`checkpointerror::shapemismatch`], and any model-level inconsistency is
/// [`checkpointerror::snapshotmismatch`].
///
/// # Errors
///
/// Returns the snapshot or reconstruction errors described above.
pub fn loadllamaweights(
    reader: &checkpointreader,
    config: &llamaconfig,
) -> Result<llamaweights, checkpointerror> {
    let specs = weightspecs(config);
    validate(reader, &specs)?;

    let embedding = embedding::new(parameter::new("weight", readowned(reader, &specs[0])?))
        .map_err(|e| checkpointerror::snapshotmismatch {
            detail: e.to_string(),
        })?;

    let mut layers = Vec::with_capacity(config.getnlayers());
    let mut idx = 1;
    for _ in 0..config.getnlayers() {
        let attn_norm = rmsnorm::new(
            parameter::new("weight", readowned(reader, &specs[idx])?),
            config.getrmseps(),
        )
        .map_err(|e| checkpointerror::snapshotmismatch {
            detail: e.to_string(),
        })?;
        idx += 1;
        let mklinear = |i: &usize| {
            linear::new(
                parameter::new("weight", readowned(reader, &specs[*i])?),
                None,
            )
            .map_err(|e: nn::nnerror| checkpointerror::snapshotmismatch {
                detail: e.to_string(),
            })
        };
        let wq = mklinear(&idx)?;
        idx += 1;
        let wk = mklinear(&idx)?;
        idx += 1;
        let wv = mklinear(&idx)?;
        idx += 1;
        let wo = mklinear(&idx)?;
        idx += 1;
        let ffn_norm = rmsnorm::new(
            parameter::new("weight", readowned(reader, &specs[idx])?),
            config.getrmseps(),
        )
        .map_err(|e| checkpointerror::snapshotmismatch {
            detail: e.to_string(),
        })?;
        idx += 1;
        let gate = mklinear(&idx)?;
        idx += 1;
        let up = mklinear(&idx)?;
        idx += 1;
        let down = mklinear(&idx)?;
        idx += 1;

        let layer = llamalayerweights::new(
            config,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            mlp::new(gate, up, down).map_err(|e| checkpointerror::snapshotmismatch {
                detail: e.to_string(),
            })?,
        )
        .map_err(|e| checkpointerror::snapshotmismatch {
            detail: e.to_string(),
        })?;
        layers.push(layer);
    }

    let norm = rmsnorm::new(
        parameter::new("weight", readowned(reader, &specs[idx])?),
        config.getrmseps(),
    )
    .map_err(|e| checkpointerror::snapshotmismatch {
        detail: e.to_string(),
    })?;
    idx += 1;
    let output = linear::new(
        parameter::new("weight", readowned(reader, &specs[idx])?),
        None,
    )
    .map_err(|e| checkpointerror::snapshotmismatch {
        detail: e.to_string(),
    })?;

    llamaweights::new(config, embedding, layers, norm, output).map_err(|e| {
        checkpointerror::snapshotmismatch {
            detail: e.to_string(),
        }
    })
}

/// Loads every weight as a view, zero-copy where the snapshot allows it.
///
/// The returned list follows the same inventory as [`loadllamaweights`], in
/// collect order. A weight whose snapshot entry is a raw, page-aligned f32
/// tensor yields [`weightview::borrowed`] aliasing the memory map; any other
/// encoding yields [`weightview::owned`]. Consumers (for example a GPU
/// weight-upload pipeline) can feed each view straight into
/// `api::operand::input`, skipping the owned materialization entirely.
///
/// # Errors
///
/// Returns the same snapshot errors as [`loadllamaweights`].
pub fn loadweightviews<'a>(
    reader: &'a checkpointreader,
    config: &llamaconfig,
) -> Result<Vec<(String, weightview<'a>)>, checkpointerror> {
    let specs = weightspecs(config);
    validate(reader, &specs)?;

    let mut out = Vec::with_capacity(specs.len());
    for spec in &specs {
        let s = shape::new(spec.dims.clone()).map_err(checkpointerror::tensor)?;
        let l = layout::contiguous(&s);
        let data = reader.readtensorview::<f32>(&spec.name)?.ok_or_else(|| {
            checkpointerror::missingtensor {
                name: spec.name.clone(),
            }
        })?;
        let view = match data {
            Cow::Borrowed(slice) => weightview::borrowed {
                shape: s,
                layout: l,
                data: slice,
            },
            Cow::Owned(v) => {
                let t = tensor::new(s, v).map_err(checkpointerror::tensor)?;
                weightview::owned(t)
            }
        };
        out.push((spec.name.clone(), view));
    }
    Ok(out)
}
