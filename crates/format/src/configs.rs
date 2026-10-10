use serde::Deserialize;
use tensor::dtype;

use crate::error::formaterror;

/// The raw `config.json` schema for a Llama-family causal language model.
///
/// Unknown fields are ignored so that vendor-specific additions do not fail
/// parsing; every field this crate relies on is validated explicitly afterwards.
#[derive(Deserialize)]
struct rawconfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(rename = "model_type")]
    modeltype: String,
    #[serde(rename = "hidden_size")]
    hiddensize: usize,
    #[serde(rename = "intermediate_size")]
    intermediatesize: usize,
    #[serde(rename = "num_attention_heads")]
    attentionheads: usize,
    #[serde(rename = "num_key_value_heads", default)]
    keyvalueheads: Option<usize>,
    #[serde(rename = "num_hidden_layers")]
    hiddenlayers: usize,
    #[serde(rename = "max_position_embeddings")]
    maxpositions: usize,
    #[serde(rename = "rms_norm_eps")]
    rmsnormeps: f64,
    #[serde(rename = "vocab_size")]
    vocabsize: usize,
    #[serde(rename = "rope_theta")]
    ropetheta: f64,
    #[serde(rename = "torch_dtype", default)]
    torchdtype: Option<String>,
    #[serde(rename = "tie_word_embeddings", default)]
    tiewordembeddings: Option<bool>,
}

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
        let raw: rawconfig =
            serde_json::from_slice(bytes).map_err(|e| formaterror::invalidjson {
                line: e.line(),
                column: e.column(),
                message: e.to_string(),
            })?;
        if raw.modeltype != "llama" {
            return Err(formaterror::unsupportedmodeltype {
                found: raw.modeltype,
            });
        }
        positive("hidden_size", raw.hiddensize)?;
        positive("intermediate_size", raw.intermediatesize)?;
        positive("num_attention_heads", raw.attentionheads)?;
        positive("num_hidden_layers", raw.hiddenlayers)?;
        positive("max_position_embeddings", raw.maxpositions)?;
        positive("vocab_size", raw.vocabsize)?;
        let keyvalueheads = raw.keyvalueheads.unwrap_or(raw.attentionheads);
        positive("num_key_value_heads", keyvalueheads)?;
        if !raw.hiddensize.is_multiple_of(raw.attentionheads) {
            return Err(formaterror::invalidvalue {
                field: "hidden_size",
                reason: "not divisible by num_attention_heads".to_string(),
            });
        }
        if !raw.attentionheads.is_multiple_of(keyvalueheads) {
            return Err(formaterror::invalidvalue {
                field: "num_attention_heads",
                reason: "not divisible by num_key_value_heads".to_string(),
            });
        }
        finite_positive("rms_norm_eps", raw.rmsnormeps)?;
        finite_positive("rope_theta", raw.ropetheta)?;
        let torchdtype = match raw.torchdtype {
            None => None,
            Some(token) => Some(parsetorchdtype(&token)?),
        };
        Ok(Self {
            architectures: raw.architectures,
            hiddensize: raw.hiddensize,
            intermediatesize: raw.intermediatesize,
            attentionheads: raw.attentionheads,
            keyvalueheads,
            hiddenlayers: raw.hiddenlayers,
            maxpositions: raw.maxpositions,
            rmsnormeps: raw.rmsnormeps,
            vocabsize: raw.vocabsize,
            ropetheta: raw.ropetheta,
            torchdtype,
            tiewordembeddings: raw.tiewordembeddings,
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
