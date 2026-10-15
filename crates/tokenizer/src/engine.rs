use lib::tokenid;
use serde_json::{Map, Value};

use crate::chat::{chatmessage, chattemplate};
use crate::decode::{decodebytelevel, decodetokens};
use crate::encode::{encodeplain, encodespecial};
use crate::error::tokenizererror;
use crate::jinja::renderjinja;
use crate::merges::bytepair;
use crate::normalize::{normalizer, parsenormalizer};
use crate::postprocess::{parsepostprocessor, postprocessor};
use crate::special::specialtokens;
use crate::vocab::vocabulary;

/// A byte-pair-encoding tokenizer loaded from a HuggingFace `tokenizer.json`
/// subset.
///
/// Supported model type is `"BPE"`. The loaded subset covers `model.vocab`,
/// `model.merges`, `model.byte_fallback`, `model.unk_token`, `added_tokens`,
/// an optional `normalizer`, a ByteLevel `pre_tokenizer`/`decoder`, a
/// `TemplateProcessing` `post_processor`, and a Jinja `chat_template`. Unknown
/// fields are ignored.
#[derive(Clone, Debug)]
pub struct tokenizer {
    vocabulary: vocabulary,
    bytepair: bytepair,
    specialtokens: specialtokens,
    chattemplate: Option<chattemplate>,
    chattemplatejinja: Option<String>,
    bostokencontent: Option<String>,
    eostokencontent: Option<String>,
    normalizer: Option<normalizer>,
    bytefallback: bool,
    bytelevelencode: bool,
    byteleveldecode: bool,
    postprocessor: Option<postprocessor>,
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
        let bytelevelencode = parsebytelevelencode(rootobject)?;
        let byteleveldecode = parsebyteleveldecode(rootobject)?;
        let postprocessor = parsepostprocessor(rootobject, &vocab)?;
        let chattemplatejinja =
            parseoptionalstring(rootobject, "chat_template")?.filter(|s| !s.is_empty());
        let bostokencontent = parseoptionalstring(rootobject, "bos_token")?;
        let eostokencontent = parseoptionalstring(rootobject, "eos_token")?;
        let normalizer = parsenormalizer(rootobject)?;
        Ok(Self {
            vocabulary: vocab,
            bytepair: bpe,
            specialtokens: specials,
            chattemplate: None,
            chattemplatejinja,
            bostokencontent,
            eostokencontent,
            normalizer,
            bytefallback,
            bytelevelencode,
            byteleveldecode,
            postprocessor,
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

    /// Returns the loaded post-processor, if any.
    #[must_use]
    pub fn getpostprocessor(&self) -> Option<&postprocessor> {
        self.postprocessor.as_ref()
    }

    /// Applies the loaded post-processor to `ids`, inserting the single
    /// template's special tokens. Without a post-processor this is the identity.
    #[must_use]
    pub fn postprocess(&self, ids: &[tokenid]) -> Vec<tokenid> {
        match &self.postprocessor {
            Some(processor) => processor.apply(ids),
            None => ids.to_vec(),
        }
    }

    /// Applies the loaded post-processor to a pair of sequences, inserting the
    /// pair template's special tokens. Without a post-processor this
    /// concatenates `ids` and `pair`.
    #[must_use]
    pub fn postprocesspair(&self, ids: &[tokenid], pair: &[tokenid]) -> Vec<tokenid> {
        match &self.postprocessor {
            Some(processor) => processor.applypair(ids, pair),
            None => {
                let mut out = ids.to_vec();
                out.extend_from_slice(pair);
                out
            }
        }
    }

    /// Encodes `text` as ordinary text, without interpreting special-token
    /// literals.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unknowntoken`] when a symbol has no vocabulary
    /// entry, no complete byte fallback, and no configured unknown token.
    pub fn encode(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        let normalized = self.normalized(text);
        encodeplain(
            &self.vocabulary,
            &self.bytepair,
            self.bytefallback,
            self.unk,
            self.bytelevelencode,
            normalized.as_ref(),
        )
    }

    /// Encodes `text`, recognizing special-token literals as structural markers.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::encode`] for the non-special segments.
    pub fn encodespecial(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        let normalized = self.normalized(text);
        encodespecial(
            &self.vocabulary,
            &self.bytepair,
            self.bytefallback,
            self.unk,
            self.bytelevelencode,
            &self.specialtokens,
            normalized.as_ref(),
        )
    }

    /// Applies the loaded normalizer, or borrows `text` unchanged.
    fn normalized<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.normalizer {
            Some(normalizer) => std::borrow::Cow::Owned(normalizer.apply(text)),
            None => std::borrow::Cow::Borrowed(text),
        }
    }

    /// Decodes token ids to a string.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unknowntokenid`] for an id absent from the
    /// vocabulary and [`tokenizererror::invalidutf8`] when byte-fallback tokens
    /// do not reconstruct valid UTF-8.
    pub fn decode(&self, ids: &[tokenid]) -> Result<String, tokenizererror> {
        if self.byteleveldecode {
            return decodebytelevel(&self.vocabulary, ids);
        }
        decodetokens(&self.vocabulary, ids)
    }

    /// Renders the installed chat template (Jinja or explicit) to a string.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::nochattemplate`] when neither a Jinja nor an
    /// explicit template is installed, [`tokenizererror::unsupportedrole`] for
    /// an uncovered role in the explicit template, and
    /// [`tokenizererror::malformedtemplate`] for a Jinja template outside the
    /// supported subset.
    pub fn renderchat(&self, messages: &[chatmessage]) -> Result<String, tokenizererror> {
        if let Some(template) = &self.chattemplatejinja {
            return renderjinja(
                template,
                messages,
                self.bostokencontent.as_deref().unwrap_or(""),
                self.eostokencontent.as_deref().unwrap_or(""),
            );
        }
        let template = self
            .chattemplate
            .as_ref()
            .ok_or(tokenizererror::nochattemplate)?;
        template.render(messages)
    }

    /// Renders and encodes a chat conversation.
    ///
    /// A Jinja template already emits its beginning-of-sequence token in the
    /// rendered string; an explicit template's BOS token is prepended before
    /// the rendered body.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::renderchat`] and [`Self::encodespecial`].
    pub fn applychattemplate(
        &self,
        messages: &[chatmessage],
    ) -> Result<Vec<tokenid>, tokenizererror> {
        let rendered = self.renderchat(messages)?;
        let mut ids = Vec::new();
        if self.chattemplatejinja.is_none()
            && let Some(bos) = self
                .chattemplate
                .as_ref()
                .and_then(|template| template.getbos())
        {
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

/// Returns whether the `pre_tokenizer` section requests ByteLevel encoding.
fn parsebytelevelencode(root: &Map<String, Value>) -> Result<bool, tokenizererror> {
    let Some(value) = root.get("pre_tokenizer") else {
        return Ok(false);
    };
    let object = value.as_object().ok_or(tokenizererror::unexpectedtype {
        field: "pre_tokenizer",
        expected: "object",
    })?;
    let pretype =
        object
            .get("type")
            .and_then(Value::as_str)
            .ok_or(tokenizererror::missingfield {
                field: "pre_tokenizer.type",
            })?;
    if pretype == "ByteLevel" {
        Ok(true)
    } else {
        Err(tokenizererror::unsupportedpretokenizer {
            pretype: pretype.to_string(),
        })
    }
}

/// Returns whether the `decoder` section requests ByteLevel decoding.
fn parsebyteleveldecode(root: &Map<String, Value>) -> Result<bool, tokenizererror> {
    let Some(value) = root.get("decoder") else {
        return Ok(false);
    };
    let object = value.as_object().ok_or(tokenizererror::unexpectedtype {
        field: "decoder",
        expected: "object",
    })?;
    let decodertype =
        object
            .get("type")
            .and_then(Value::as_str)
            .ok_or(tokenizererror::missingfield {
                field: "decoder.type",
            })?;
    if decodertype == "ByteLevel" {
        Ok(true)
    } else {
        Err(tokenizererror::unsupporteddecoder {
            decodertype: decodertype.to_string(),
        })
    }
}

/// Returns the optional top-level string field `field`.
fn parseoptionalstring(
    root: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<String>, tokenizererror> {
    match root.get(field) {
        Some(value) => {
            let content = value.as_str().ok_or(tokenizererror::unexpectedtype {
                field,
                expected: "string",
            })?;
            Ok(Some(content.to_string()))
        }
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
