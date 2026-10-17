use lib::{jsonvalue, parse};
use tensor::dtype;

use crate::error::formaterror;

/// The raw `config.json` schema for a Llama-family causal language model.
///
/// Unknown fields are ignored so that vendor-specific additions do not fail
/// parsing; every field this crate relies on is validated explicitly afterwards.

/// A validated Llama-family model configuration.
///
/// Only the fields this workspace models are retained; the raw JSON may carry
/// additional keys. Every numeric field is non-zero and, where applicable,
/// finite and positive; attention-head counts are consistent with the hidden
/// size and with grouped-query attention. There is no unchecked public
/// constructor: values are produced only by [`Self::parse`].
#[derive(Clone, Debug, PartialEq)]
pub struct modelconfig {
    architectures: Vec<String>,
    hiddensize: usize,
    intermediatesize: usize,
    attentionheads: usize,
    keyvalueheads: usize,
    hiddenlayers: usize,
    maxpositions: usize,
    rmsnormeps: f64,
    vocabsize: usize,
    ropetheta: f64,
    torchdtype: Option<dtype>,
    tiewordembeddings: Option<bool>,
}

impl modelconfig {
    /// Parses and validates a `config.json` byte buffer.
    ///
    /// The bytes are treated as untrusted. The JSON is parsed into the expected
    /// schema; the model type must be `"llama"`; every size must be non-zero;
    /// `hidden_size` must be divisible by `num_attention_heads`; the
    /// key/value-head count (defaulting to the attention-head count) must
    /// divide it; and `rms_norm_eps` and `rope_theta` must be finite and
    /// positive. An optional `torch_dtype` is mapped to a storage [`dtype`] or
    /// rejected.
    ///
    /// # Errors
    ///
    /// Returns a [`formaterror`] for malformed JSON, an unsupported model type
    /// or torch dtype, or an invalid field value.
    pub fn parse(bytes: &[u8]) -> Result<Self, formaterror> {
        let root = parse(bytes).map_err(|e| formaterror::invalidjson {
            line: e.getline(),
            column: e.getcolumn(),
            message: e.to_string(),
        })?;
        let object = root.getobject().ok_or_else(|| formaterror::invalidjson {
            line: 1,
            column: 1,
            message: "config.json is not a JSON object".to_string(),
        })?;
        let modeltype = stringfield(object, "model_type")?;
        if modeltype != "llama" {
            return Err(formaterror::unsupportedmodeltype { found: modeltype });
        }
        let hiddensize = sizefield(object, "hidden_size")?;
        let intermediatesize = sizefield(object, "intermediate_size")?;
        let attentionheads = sizefield(object, "num_attention_heads")?;
        let hiddenlayers = sizefield(object, "num_hidden_layers")?;
        let maxpositions = sizefield(object, "max_position_embeddings")?;
        let vocabsize = sizefield(object, "vocab_size")?;
        positive("hidden_size", hiddensize)?;
        positive("intermediate_size", intermediatesize)?;
        positive("num_attention_heads", attentionheads)?;
        positive("num_hidden_layers", hiddenlayers)?;
        positive("max_position_embeddings", maxpositions)?;
        positive("vocab_size", vocabsize)?;
        let keyvalueheads =
            optionalsizefield(object, "num_key_value_heads")?.unwrap_or(attentionheads);
        positive("num_key_value_heads", keyvalueheads)?;
        if !hiddensize.is_multiple_of(attentionheads) {
            return Err(formaterror::invalidvalue {
                field: "hidden_size",
                reason: "not divisible by num_attention_heads".to_string(),
            });
        }
        if !attentionheads.is_multiple_of(keyvalueheads) {
            return Err(formaterror::invalidvalue {
                field: "num_attention_heads",
                reason: "not divisible by num_key_value_heads".to_string(),
            });
        }
        let rmsnormeps = realfield(object, "rms_norm_eps")?;
        let ropetheta = realfield(object, "rope_theta")?;
        finite_positive("rms_norm_eps", rmsnormeps)?;
        finite_positive("rope_theta", ropetheta)?;
        let architectures = object
            .get("architectures")
            .map(|value| {
                value
                    .getarray()
                    .map(|items| {
                        items
                            .iter()
                            .map(|item| {
                                item.getstr().map(str::to_string).ok_or_else(|| {
                                    formaterror::invalidvalue {
                                        field: "architectures",
                                        reason: "entry is not a string".to_string(),
                                    }
                                })
                            })
                            .collect::<Result<Vec<String>, formaterror>>()
                    })
                    .unwrap_or_else(|| {
                        Err(formaterror::invalidvalue {
                            field: "architectures",
                            reason: "not an array".to_string(),
                        })
                    })
            })
            .transpose()?
            .unwrap_or_default();
        let torchdtype = object
            .get("torch_dtype")
            .map(|value| {
                let token = value.getstr().ok_or_else(|| formaterror::invalidvalue {
                    field: "torch_dtype",
                    reason: "not a string".to_string(),
                })?;
                parsetorchdtype(token)
            })
            .transpose()?;
        let tiewordembeddings = object
            .get("tie_word_embeddings")
            .map(|value| {
                value.getbool().ok_or_else(|| formaterror::invalidvalue {
                    field: "tie_word_embeddings",
                    reason: "not a boolean".to_string(),
                })
            })
            .transpose()?;
        Ok(Self {
            architectures,
            hiddensize,
            intermediatesize,
            attentionheads,
            keyvalueheads,
            hiddenlayers,
            maxpositions,
            rmsnormeps,
            vocabsize,
            ropetheta,
            torchdtype,
            tiewordembeddings,
        })
    }

    /// Returns the declared architecture names, possibly empty.
    #[must_use]
    pub fn getarchitectures(&self) -> &[String] {
        &self.architectures
    }

    /// Returns the hidden size.
    #[must_use]
    pub fn gethiddensize(&self) -> usize {
        self.hiddensize
    }

    /// Returns the feed-forward intermediate size.
    #[must_use]
    pub fn getintermediatesize(&self) -> usize {
        self.intermediatesize
    }

    /// Returns the number of query attention heads.
    #[must_use]
    pub fn getattentionheads(&self) -> usize {
        self.attentionheads
    }

    /// Returns the number of key/value heads.
    #[must_use]
    pub fn getkeyvalueheads(&self) -> usize {
        self.keyvalueheads
    }

    /// Returns the number of transformer blocks.
    #[must_use]
    pub fn gethiddenlayers(&self) -> usize {
        self.hiddenlayers
    }

    /// Returns the maximum supported sequence length.
    #[must_use]
    pub fn getmaxpositions(&self) -> usize {
        self.maxpositions
    }

    /// Returns the RMS-normalization epsilon.
    #[must_use]
    pub fn getrmsnormeps(&self) -> f64 {
        self.rmsnormeps
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub fn getvocabsize(&self) -> usize {
        self.vocabsize
    }

    /// Returns the RoPE base period.
    #[must_use]
    pub fn getropetheta(&self) -> f64 {
        self.ropetheta
    }

    /// Returns the per-head dimension, `hidden_size / num_attention_heads`.
    #[must_use]
    pub fn getheadsize(&self) -> usize {
        self.hiddensize / self.attentionheads
    }

    /// Returns the query heads per key/value head group.
    #[must_use]
    pub fn getkeyvaluegroups(&self) -> usize {
        self.attentionheads / self.keyvalueheads
    }

    /// Returns the declared torch dtype, when present.
    #[must_use]
    pub fn gettorchdtype(&self) -> Option<dtype> {
        self.torchdtype
    }

    /// Returns the tied-embedding declaration, when present.
    #[must_use]
    pub fn gettiewordembeddings(&self) -> Option<bool> {
        self.tiewordembeddings
    }
}

/// Rejects a zero-valued size field.
fn positive(field: &'static str, value: usize) -> Result<(), formaterror> {
    if value == 0 {
        return Err(formaterror::invalidvalue {
            field,
            reason: "must be non-zero".to_string(),
        });
    }
    Ok(())
}

/// Extracts a required string field.
fn stringfield(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &'static str,
) -> Result<String, formaterror> {
    object.get(field).ok_or_else(|| missingfield(field))?;
    object
        .get(field)
        .and_then(jsonvalue::getstr)
        .map(str::to_string)
        .ok_or_else(|| formaterror::invalidvalue {
            field,
            reason: "not a string".to_string(),
        })
}

/// Extracts a required non-negative integer size field.
fn sizefield(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &'static str,
) -> Result<usize, formaterror> {
    object.get(field).ok_or_else(|| missingfield(field))?;
    object
        .get(field)
        .and_then(jsonvalue::getunsigned)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| formaterror::invalidvalue {
            field,
            reason: "not a non-negative integer".to_string(),
        })
}

/// Extracts an optional non-negative integer size field.
fn optionalsizefield(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &'static str,
) -> Result<Option<usize>, formaterror> {
    object
        .get(field)
        .map(|value| {
            value
                .getunsigned()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| formaterror::invalidvalue {
                    field,
                    reason: "not a non-negative integer".to_string(),
                })
        })
        .transpose()
}

/// Extracts a required real field.
fn realfield(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &'static str,
) -> Result<f64, formaterror> {
    object.get(field).ok_or_else(|| missingfield(field))?;
    object
        .get(field)
        .and_then(jsonvalue::getreal)
        .ok_or_else(|| formaterror::invalidvalue {
            field,
            reason: "not a number".to_string(),
        })
}

/// Builds the JSON-level error for a missing required field, matching the
/// previous deserializer behaviour of reporting schema gaps as invalid JSON.
fn missingfield(field: &'static str) -> formaterror {
    formaterror::invalidjson {
        line: 1,
        column: 1,
        message: format!("missing required field {field:?}"),
    }
}

/// Rejects a non-finite or non-positive floating-point field.
fn finite_positive(field: &'static str, value: f64) -> Result<(), formaterror> {
    if !value.is_finite() || value <= 0.0 {
        return Err(formaterror::invalidvalue {
            field,
            reason: "must be finite and positive".to_string(),
        });
    }
    Ok(())
}

/// Maps a torch dtype name to a storage [`dtype`].
fn parsetorchdtype(token: &str) -> Result<dtype, formaterror> {
    match token {
        "float32" | "f32" => Ok(dtype::f32),
        "float16" | "f16" | "half" => Ok(dtype::f16),
        "bfloat16" | "bf16" => Ok(dtype::bf16),
        "float64" | "f64" | "double" => Ok(dtype::f64),
        _ => Err(formaterror::unsupportedtorchdtype {
            found: token.to_string(),
        }),
    }
}
