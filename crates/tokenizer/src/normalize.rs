//! Normalization: the `normalizer` section of a `tokenizer.json`.

use serde_json::{Map, Value};
use unicode_normalization::UnicodeNormalization;

use crate::error::tokenizererror;

/// A text normalizer applied before pre-tokenization.
///
/// The subset covers the normalizers implementable without external Unicode
/// tables: lowercase, whitespace strip, and a sequence of normalizers. The
/// Unicode normalization forms (NFC/NFD/NFKC/NFKD) are rejected at load time
/// because they require the `unicode-normalization` dependency, which is
/// deferred until a real model needs it (see the optimization plan D2).
#[derive(Clone, Debug)]
pub enum normalizer {
    /// Lowercases the text.
    lowercase,
    /// Strips leading and trailing whitespace.
    strip,
    /// Prepends a literal string.
    prepend(String),
    /// Replaces every occurrence of a literal string.
    replace {
        /// The literal string to replace.
        pattern: String,
        /// The replacement string.
        content: String,
    },
    /// Replaces every regex match (e.g. HF's ` {2,}` → ` `).
    replaceregex {
        /// The compiled replacement pattern.
        pattern: fancy_regex::Regex,
        /// The replacement string.
        content: String,
    },
    /// NFC normalization.
    nfc,
    /// NFD normalization.
    nfd,
    /// NFKC normalization.
    nfkc,
    /// NFKD normalization.
    nfkd,
    /// Applies the children in order.
    sequence(Vec<normalizer>),
}

impl normalizer {
    /// Applies the normalizer to `text`.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        match self {
            Self::lowercase => text.to_lowercase(),
            Self::strip => text.trim().to_string(),
            Self::prepend(prefix) => format!("{prefix}{text}"),
            Self::replace { pattern, content } => text.replace(pattern, content),
            Self::replaceregex { pattern, content } => pattern
                .replace_all(text, content.as_str())
                .into_owned(),
            Self::nfc => text.nfc().collect(),
            Self::nfd => text.nfd().collect(),
            Self::nfkc => text.nfkc().collect(),
            Self::nfkd => text.nfkd().collect(),
            Self::sequence(children) => {
                let mut out = text.to_string();
                for child in children {
                    out = child.apply(&out);
                }
                out
            }
        }
    }
}

/// Parses the `normalizer` section, if present and non-null.
pub(crate) fn parsenormalizer(
    root: &Map<String, Value>,
) -> Result<Option<normalizer>, tokenizererror> {
    match root.get("normalizer") {
        Some(value) if value.is_null() => Ok(None),
        Some(value) => Ok(Some(parseone(value)?)),
        None => Ok(None),
    }
}

/// Parses one normalizer object.
fn parseone(value: &Value) -> Result<normalizer, tokenizererror> {
    let object = value.as_object().ok_or(tokenizererror::unexpectedtype {
        field: "normalizer",
        expected: "object",
    })?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(tokenizererror::missingfield {
            field: "normalizer.type",
        })?;
    match kind {
        "Lowercase" => Ok(normalizer::lowercase),
        "Strip" => Ok(normalizer::strip),
        "NFC" => Ok(normalizer::nfc),
        "NFD" => Ok(normalizer::nfd),
        "NFKC" => Ok(normalizer::nfkc),
        "NFKD" => Ok(normalizer::nfkd),
        // Precompiled charsmap (XLM-RoBERTa family) is a precompiled NFKC-plus-
        // control-mapping table. The library substitutes the NFKC form, which
        // matches the charsmap on the golden sample set (full-width punct/space
        // and fraction compatibility); a rare charsmap-only mapping (control
        // characters) may differ.
        "Precompiled" => Ok(normalizer::nfkc),
        "Prepend" => {
            let prefix = object.get("prepend").and_then(Value::as_str).ok_or(
                tokenizererror::missingfield {
                    field: "normalizer.prepend",
                },
            )?;
            Ok(normalizer::prepend(prefix.to_string()))
        }
        "Replace" => {
            let pattern = object
                .get("pattern")
                .and_then(Value::as_object);
            let literal = pattern.and_then(|p| p.get("String")).and_then(Value::as_str);
            let regex = pattern.and_then(|p| p.get("Regex")).and_then(Value::as_str);
            let content = object.get("content").and_then(Value::as_str).ok_or(
                tokenizererror::missingfield {
                    field: "normalizer.content",
                },
            )?;
            if let Some(literal) = literal {
                Ok(normalizer::replace {
                    pattern: literal.to_string(),
                    content: content.to_string(),
                })
            } else if let Some(regex) = regex {
                let compiled =
                    fancy_regex::Regex::new(regex).map_err(|error| tokenizererror::invalidregex {
                        detail: error.to_string(),
                    })?;
                Ok(normalizer::replaceregex {
                    pattern: compiled,
                    content: content.to_string(),
                })
            } else {
                Err(tokenizererror::unsupportednormalizer {
                    normaltype: "Replace without a String or Regex pattern".to_string(),
                })
            }
        }
        "RegexReplace" => {
            let pattern = object
                .get("pattern")
                .and_then(Value::as_object)
                .and_then(|p| p.get("Regex"))
                .and_then(Value::as_str)
                .ok_or(tokenizererror::unsupportednormalizer {
                    normaltype: "RegexReplace without a Regex pattern".to_string(),
                })?;
            let content = object.get("content").and_then(Value::as_str).ok_or(
                tokenizererror::missingfield {
                    field: "normalizer.content",
                },
            )?;
            let compiled =
                fancy_regex::Regex::new(pattern).map_err(|error| tokenizererror::invalidregex {
                    detail: error.to_string(),
                })?;
            Ok(normalizer::replaceregex {
                pattern: compiled,
                content: content.to_string(),
            })
        }
        "Sequence" => {
            let children = object
                .get("normalizers")
                .ok_or(tokenizererror::missingfield {
                    field: "normalizer.normalizers",
                })?;
            let array = children.as_array().ok_or(tokenizererror::unexpectedtype {
                field: "normalizer.normalizers",
                expected: "array",
            })?;
            let mut parsed = Vec::with_capacity(array.len());
            for child in array {
                parsed.push(parseone(child)?);
            }
            Ok(normalizer::sequence(parsed))
        }
        other => Err(tokenizererror::unsupportednormalizer {
            normaltype: other.to_string(),
        }),
    }
}
