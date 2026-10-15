//! TemplateProcessing post-processing: wraps encoded sequences with special
//! tokens (BOS/EOS) per the `post_processor` section of a `tokenizer.json`.

use lib::tokenid;
use serde_json::{Map, Value};

use crate::error::tokenizererror;
use crate::vocab::vocabulary;

/// One step of a post-processing template.
#[derive(Clone, Debug, PartialEq, Eq)]
enum poststep {
    /// A literal special token, resolved to its id at load time.
    special(tokenid),
    /// The first encoded sequence.
    sequencea,
    /// The second encoded sequence.
    sequenceb,
}

/// A `TemplateProcessing` post-processor.
///
/// The `single` template wraps one encoded sequence; the `pair` template wraps
/// two. Each template is an ordered list of special tokens and sequence
/// placeholders. Type ids are ignored: this tokenizer produces only token ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct postprocessor {
    single: Vec<poststep>,
    pair: Vec<poststep>,
}

impl postprocessor {
    /// Applies the single-sequence template to `ids`.
    #[must_use]
    pub fn apply(&self, ids: &[tokenid]) -> Vec<tokenid> {
        render(&self.single, ids, None)
    }

    /// Applies the pair template to `ids` and `pair`.
    #[must_use]
    pub fn applypair(&self, ids: &[tokenid], pair: &[tokenid]) -> Vec<tokenid> {
        render(&self.pair, ids, Some(pair))
    }
}

/// Renders `template`, splicing in `ids` for `A` and `pair` for `B`.
fn render(template: &[poststep], ids: &[tokenid], pair: Option<&[tokenid]>) -> Vec<tokenid> {
    let mut out = Vec::new();
    for step in template {
        match step {
            poststep::special(id) => out.push(*id),
            poststep::sequencea => out.extend_from_slice(ids),
            poststep::sequenceb => {
                if let Some(second) = pair {
                    out.extend_from_slice(second);
                }
            }
        }
    }
    out
}

/// Parses the `post_processor` section, if present.
pub(crate) fn parsepostprocessor(
    root: &Map<String, Value>,
    vocab: &vocabulary,
) -> Result<Option<postprocessor>, tokenizererror> {
    let Some(value) = root.get("post_processor") else {
        return Ok(None);
    };
    let object = value.as_object().ok_or(tokenizererror::unexpectedtype {
        field: "post_processor",
        expected: "object",
    })?;
    let posttype =
        object
            .get("type")
            .and_then(Value::as_str)
            .ok_or(tokenizererror::missingfield {
                field: "post_processor.type",
            })?;
    match posttype {
        // ByteLevel post-processing inserts the BOS token; it only affects
        // `add_special_tokens=true`, so it is accepted as a no-op here (the
        // raw encoding used by alignment is unaffected).
        "ByteLevel" => Ok(None),
        "TemplateProcessing" => {
            let single =
                parsetemplate(object.get("single"), vocab, "post_processor.single", false)?;
            let pair = parsetemplate(object.get("pair"), vocab, "post_processor.pair", true)?;
            Ok(Some(postprocessor { single, pair }))
        }
        other => Err(tokenizererror::unsupportedpostprocessor {
            posttype: other.to_string(),
        }),
    }
}

/// Parses a template array. `allowb` gates whether a `B` sequence placeholder
/// is permitted (the single template must not reference the second sequence).
fn parsetemplate(
    value: Option<&Value>,
    vocab: &vocabulary,
    field: &'static str,
    allowb: bool,
) -> Result<Vec<poststep>, tokenizererror> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let array = value.as_array().ok_or(tokenizererror::unexpectedtype {
        field,
        expected: "array",
    })?;
    let mut steps = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        steps.push(parsestep(item, vocab, allowb, index)?);
    }
    Ok(steps)
}

/// Parses one template entry.
fn parsestep(
    item: &Value,
    vocab: &vocabulary,
    allowb: bool,
    index: usize,
) -> Result<poststep, tokenizererror> {
    let object = item
        .as_object()
        .ok_or(tokenizererror::malformedpostprocessor { index })?;
    if let Some(special) = object.get("SpecialToken") {
        let content = special
            .as_object()
            .and_then(|obj| obj.get("id"))
            .and_then(Value::as_str)
            .ok_or(tokenizererror::malformedpostprocessor { index })?;
        let id = vocab.getid(content).ok_or(tokenizererror::unknowntoken {
            token: content.to_string(),
        })?;
        return Ok(poststep::special(id));
    }
    if let Some(sequence) = object.get("Sequence") {
        let id = sequence
            .as_object()
            .and_then(|obj| obj.get("id"))
            .and_then(Value::as_str)
            .ok_or(tokenizererror::malformedpostprocessor { index })?;
        return match id {
            "A" => Ok(poststep::sequencea),
            "B" if allowb => Ok(poststep::sequenceb),
            _ => Err(tokenizererror::malformedpostprocessor { index }),
        };
    }
    Err(tokenizererror::malformedpostprocessor { index })
}
