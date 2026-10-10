use lib::tokenid;
use serde_json::{Map, Value};

use crate::chat::{chatmessage, chattemplate};
use crate::decode::decodetokens;
use crate::encode::{encodeplain, encodespecial};
use crate::error::tokenizererror;
use crate::merges::bytepair;
use crate::special::specialtokens;
use crate::vocab::vocabulary;

/// A byte-pair-encoding tokenizer loaded from a HuggingFace `tokenizer.json`
/// subset.
///
/// Supported model type is `"BPE"`. The loaded subset covers `model.vocab`,
/// `model.merges`, `model.byte_fallback`, `model.unk_token`, and
/// `added_tokens`. Unknown fields are ignored. Pre-tokenization, the GPT-2
/// byte-to-unicode alphabet, and Jinja chat-template evaluation are not
/// implemented; see `docs/design.md` for the exact subset
/// and its limitations.
#[derive(Clone, Debug)]
pub struct tokenizer {
    vocabulary: vocabulary,
    bytepair: bytepair,
    specialtokens: specialtokens,
    chattemplate: Option<chattemplate>,
    bytefallback: bool,
    unk: Option<tokenid>,
}

impl tokenizer {
    /// Builds a tokenizer from a `tokenizer.json` document.
    ///
    /// # Errors
    ///
    /// Returns a [`tokenizererror`] when the input is not valid JSON, the root
    /// is not an object, a required field is missing or mistyped, the model
    /// type is not `BPE`, a merge or added-token record is malformed, a token
    /// id is out of range or conflicting, or the configured unknown token is
    /// absent from the vocabulary.
    pub fn fromjson(json: &str) -> Result<Self, tokenizererror> {
        let root: Value =
            serde_json::from_str(json).map_err(|error| tokenizererror::invalidjson {
                line: error.line(),
                column: error.column(),
            })?;
        let rootobject = root.as_object().ok_or(tokenizererror::rootnotobject)?;

        let model = parsemodel(rootobject)?;
        let vocabobject = parsevocabobject(model)?;
        let merges = match model.get("merges") {
            Some(value) => parsemerges(value)?,
            None => Vec::new(),
        };
        let bytefallback = parsebytefallback(model)?;
        let unkcontent = parseunkcontent(model)?;

        let mut entries = parsevocab(vocabobject)?;
        let added = match rootobject.get("added_tokens") {
            Some(value) => parseaddedtokens(value)?,
            None => Vec::new(),
        };
        for (content, id, _) in &added {
            entries.push((content.clone(), *id));
        }
        let vocab = vocabulary::fromentries(entries)?;
        let specials = specialtokens::new(
            added
                .iter()
                .filter(|(_, _, special)| *special)
                .map(|(content, id, _)| (content.clone(), *id))
                .collect(),
        );
        let bpe = bytepair::frommerges(&merges)?;
        let unk = match unkcontent {
            Some(content) => Some(
                vocab
                    .getid(&content)
                    .ok_or(tokenizererror::unknowntoken { token: content })?,
            ),
            None => None,
        };
        Ok(Self {
            vocabulary: vocab,
            bytepair: bpe,
            specialtokens: specials,
            chattemplate: None,
            bytefallback,
            unk,
        })
    }

    /// Returns the loaded vocabulary.
    #[must_use]
    pub fn getvocabulary(&self) -> &vocabulary {
        &self.vocabulary
    }
    /// Returns the loaded byte-pair merge table.
    #[must_use]
    pub fn getbytepair(&self) -> &bytepair {
        &self.bytepair
    }

    /// Returns the loaded special-token set.
    #[must_use]
    pub fn getspecialtokens(&self) -> &specialtokens {
        &self.specialtokens
    }

    /// Returns the installed chat template, if any.
    #[must_use]
    pub fn getchattemplate(&self) -> Option<&chattemplate> {
        self.chattemplate.as_ref()
    }

    /// Installs `template`, replacing any previous template.
    pub fn setchattemplate(&mut self, template: chattemplate) {
        self.chattemplate = Some(template);
    }

    /// Encodes `text` as ordinary text, without interpreting special-token
    /// literals.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unknowntoken`] when a symbol has no vocabulary
    /// entry, no complete byte fallback, and no configured unknown token.
    pub fn encode(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        encodeplain(
            &self.vocabulary,
            &self.bytepair,
            self.bytefallback,
            self.unk,
            text,
        )
    }

    /// Encodes `text`, recognizing special-token literals as structural markers.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::encode`] for the non-special segments.
    pub fn encodespecial(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        encodespecial(
            &self.vocabulary,
            &self.bytepair,
            self.bytefallback,
            self.unk,
            &self.specialtokens,
            text,
        )
    }

    /// Decodes token ids to a string.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unknowntokenid`] for an id absent from the
    /// vocabulary and [`tokenizererror::invalidutf8`] when byte-fallback tokens
    /// do not reconstruct valid UTF-8.
    pub fn decode(&self, ids: &[tokenid]) -> Result<String, tokenizererror> {
        decodetokens(&self.vocabulary, ids)
    }

    /// Renders and encodes a chat conversation.
    ///
    /// The installed template's beginning-of-sequence token (if any) is emitted
    /// first, then the rendered messages are encoded with
    /// [`Self::encodespecial`]. The template's end-of-sequence token is exposed
    /// through [`chattemplate::geteos`] for the caller to append if needed.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::nochattemplate`] when no template is installed,
    /// [`tokenizererror::unsupportedrole`] when a message role is not covered,
    /// and the errors of [`Self::encodespecial`] otherwise.
    pub fn applychattemplate(
        &self,
        messages: &[chatmessage],
    ) -> Result<Vec<tokenid>, tokenizererror> {
        let template = self
            .chattemplate
            .as_ref()
            .ok_or(tokenizererror::nochattemplate)?;
        let rendered = template.render(messages)?;
        let mut ids = Vec::new();
        if let Some(bos) = template.getbos() {
            ids.push(bos);
        }
        let mut body = self.encodespecial(&rendered)?;
        ids.append(&mut body);
        Ok(ids)
    }
}

/// Returns the `model` object after checking the model type is `"BPE"`.
fn parsemodel(root: &Map<String, Value>) -> Result<&Map<String, Value>, tokenizererror> {
    let model = root
        .get("model")
        .ok_or(tokenizererror::missingfield { field: "model" })?
        .as_object()
        .ok_or(tokenizererror::unexpectedtype {
            field: "model",
            expected: "object",
        })?;
    let modeltype = model
        .get("type")
        .ok_or(tokenizererror::missingfield {
            field: "model.type",
        })?
        .as_str()
        .ok_or(tokenizererror::unexpectedtype {
            field: "model.type",
            expected: "string",
        })?;
    if modeltype != "BPE" {
        return Err(tokenizererror::unsupportedmodel {
            modeltype: modeltype.to_string(),
        });
    }
    Ok(model)
}

/// Returns the `model.vocab` object.
fn parsevocabobject(model: &Map<String, Value>) -> Result<&Map<String, Value>, tokenizererror> {
    model
        .get("vocab")
        .ok_or(tokenizererror::missingfield {
            field: "model.vocab",
        })?
        .as_object()
        .ok_or(tokenizererror::unexpectedtype {
            field: "model.vocab",
            expected: "object",
        })
}

/// Returns the optional `model.byte_fallback` flag, defaulting to `false`.
fn parsebytefallback(model: &Map<String, Value>) -> Result<bool, tokenizererror> {
    match model.get("byte_fallback") {
        Some(value) => value.as_bool().ok_or(tokenizererror::unexpectedtype {
            field: "model.byte_fallback",
            expected: "boolean",
        }),
        None => Ok(false),
    }
}

/// Returns the optional `model.unk_token` string.
fn parseunkcontent(model: &Map<String, Value>) -> Result<Option<String>, tokenizererror> {
    match model.get("unk_token") {
        Some(value) => Ok(Some(
            value
                .as_str()
                .ok_or(tokenizererror::unexpectedtype {
                    field: "model.unk_token",
                    expected: "string",
                })?
                .to_string(),
        )),
        None => Ok(None),
    }
}

fn parsevocab(map: &Map<String, Value>) -> Result<Vec<(String, u32)>, tokenizererror> {
    let mut entries = Vec::with_capacity(map.len());
    for (content, value) in map {
        let raw = value.as_u64().ok_or(tokenizererror::unexpectedtype {
            field: "model.vocab",
            expected: "integer",
        })?;
        let id = u32::try_from(raw).map_err(|_| tokenizererror::idoutofrange { id: raw })?;
        entries.push((content.clone(), id));
    }
    Ok(entries)
}

fn parsemerges(value: &Value) -> Result<Vec<String>, tokenizererror> {
    let array = value.as_array().ok_or(tokenizererror::unexpectedtype {
        field: "model.merges",
        expected: "array",
    })?;
    let mut merges = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let merge = item
            .as_str()
            .ok_or(tokenizererror::malformedmerge { index })?;
        merges.push(merge.to_string());
    }
    Ok(merges)
}

fn parseaddedtokens(value: &Value) -> Result<Vec<(String, u32, bool)>, tokenizererror> {
    let array = value.as_array().ok_or(tokenizererror::unexpectedtype {
        field: "added_tokens",
        expected: "array",
    })?;
    let mut added = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let object = item
            .as_object()
            .ok_or(tokenizererror::malformedspecialtoken { index })?;
        let raw = object
            .get("id")
            .and_then(Value::as_u64)
            .ok_or(tokenizererror::malformedspecialtoken { index })?;
        let id = u32::try_from(raw).map_err(|_| tokenizererror::idoutofrange { id: raw })?;
        let content = object
            .get("content")
            .and_then(Value::as_str)
            .ok_or(tokenizererror::malformedspecialtoken { index })?
            .to_string();
        let special = match object.get("special") {
            Some(flag) => flag
                .as_bool()
                .ok_or(tokenizererror::malformedspecialtoken { index })?,
            None => false,
        };
        added.push((content, id, special));
    }
    Ok(added)
}
