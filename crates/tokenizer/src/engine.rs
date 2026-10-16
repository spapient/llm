use lib::tokenid;
use serde_json::{Map, Value};

use crate::chat::{chatmessage, chattemplate};
use crate::decode::{decoder, parsedecoder};
use crate::encode::{resolvenonvocab, resolvetoken};
use crate::error::tokenizererror;
use crate::jinja::renderjinja;
use crate::merges::bytepair;
use crate::normalize::{normalizer, parsenormalizer};
use crate::postprocess::{parsepostprocessor, postprocessor};
use crate::pretoken::{parsepretokenizer, pretokenizer};
use crate::scratch::{mergescratch, mergestep, resetlinks};
use crate::special::specialtokens;
use crate::unigram::{parseunigramvocab, unigrammodel};
use crate::vocab::vocabulary;

/// The pre-token byte source and its character start offsets, shared read-only
/// by the rank-merge candidate probes.
struct ranksource<'a> {
    pretoken: &'a str,
    charstart: &'a [usize],
}

impl ranksource<'_> {
    /// Returns the byte offset just past the symbol at `index`: where the next
    /// live symbol begins, or the end of the pre-token.
    fn end(&self, next: &[Option<usize>], index: usize) -> usize {
        next[index].map_or(self.pretoken.len(), |following| self.charstart[following])
    }
}

/// A tokenizer loaded from a HuggingFace `tokenizer.json` subset.
///
/// Supported model types are `"BPE"` and `"Unigram"`. The loaded subset covers
/// `model.vocab`/`model.merges`/`model.byte_fallback`/`model.unk_token`,
/// `model.vocab`/`model.unk_id` for Unigram, `added_tokens`, an optional
/// `normalizer`/`pre_tokenizer`, a `decoder`, a `post_processor`, and a Jinja
/// `chat_template`. Unknown fields are ignored.
#[derive(Clone, Debug)]
pub struct tokenizer {
    vocabulary: vocabulary,
    bytepair: bytepair,
    specialtokens: specialtokens,
    /// Every added token (special and non-special) sorted by content length in
    /// descending order, used for HF-compatible longest-prefix pre-splitting of
    /// pre-tokens. Non-special added tokens such as GPT-NeoX's space runs are
    /// only vocabulary entries otherwise and would never be produced by BPE.
    addedtokens: Vec<(String, tokenid)>,
    /// Unigram model, present only for `"model.type": "Unigram"` tokenizers.
    /// Its pieces also populate `vocabulary` so the shared decoder resolves
    /// contents; encoding routes through the Viterbi lattice instead of BPE.
    unigram: Option<unigrammodel>,
    chattemplate: Option<chattemplate>,
    chattemplatejinja: Option<String>,
    bostokencontent: Option<String>,
    eostokencontent: Option<String>,
    normalizer: Option<normalizer>,
    pretokenizer: Option<pretokenizer>,
    decoder: decoder,
    bytefallback: bool,
    ignoremerges: bool,
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

        let (modeltype, model) = parsemodel(rootobject)?;
        let added = match rootobject.get("added_tokens") {
            Some(value) => parseaddedtokens(value)?,
            None => Vec::new(),
        };
        let (vocab, bpe, bytefallback, ignoremerges, unk, unigram) = match modeltype.as_str() {
            "BPE" => {
                let vocabobject = parsevocabobject(model)?;
                let merges = match model.get("merges") {
                    Some(value) => parsemerges(value)?,
                    None => Vec::new(),
                };
                let bytefallback = parsebytefallback(model)?;
                let unkcontent = parseunkcontent(model)?;
                let ignoremerges = parseignoremerges(model)?;
                let mut entries = parsevocab(vocabobject)?;
                for (content, id, _) in &added {
                    entries.push((content.clone(), *id));
                }
                let vocab = vocabulary::fromentries(entries)?;
                let bpe = if ignoremerges {
                    bytepair::frommerges(&merges)?
                } else {
                    bytepair::frommergesandvocab(&merges, &vocab)?
                };
                let unk = match unkcontent {
                    Some(content) => Some(
                        vocab
                            .getid(&content)
                            .ok_or(tokenizererror::unknowntoken { token: content })?,
                    ),
                    None => None,
                };
                (vocab, bpe, bytefallback, ignoremerges, unk, None)
            }
            "Unigram" => {
                let pieces =
                    parseunigramvocab(model.get("vocab").ok_or(tokenizererror::missingfield {
                        field: "model.vocab",
                    })?)?;
                let unk_id = model.get("unk_id").and_then(Value::as_u64).ok_or(
                    tokenizererror::missingfield {
                        field: "model.unk_id",
                    },
                )? as u32;
                let mut entries: Vec<(String, u32)> = pieces
                    .iter()
                    .enumerate()
                    .map(|(index, (piece, _))| (piece.clone(), index as u32))
                    .collect();
                for (content, id, _) in &added {
                    entries.push((content.clone(), *id));
                }
                let vocab = vocabulary::fromentries(entries)?;
                let bpe = bytepair::frommerges(&[])?;
                let unk = Some(tokenid::new(unk_id));
                let unigram = Some(unigrammodel::fromvocab(&pieces, unk_id));
                (vocab, bpe, false, false, unk, unigram)
            }
            other => {
                return Err(tokenizererror::unsupportedmodel {
                    modeltype: other.to_string(),
                });
            }
        };
        let specials = specialtokens::new(
            added
                .iter()
                .filter(|(_, _, special)| *special)
                .map(|(content, id, _)| (content.clone(), *id))
                .collect(),
        );
        let mut addedtokens: Vec<(String, tokenid)> = added
            .iter()
            .map(|(content, id, _)| (content.clone(), tokenid::new(*id)))
            .collect();
        addedtokens.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0.len()));
        let pretokenizer = parsepretokenizer(rootobject)?;
        let decoder = parsedecoder(rootobject)?.unwrap_or(decoder::bytefallback {
            replace: None,
            stripleading: 0,
        });
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
            addedtokens,
            unigram,
            chattemplate: None,
            chattemplatejinja,
            bostokencontent,
            eostokencontent,
            normalizer,
            pretokenizer,
            decoder,
            bytefallback,
            ignoremerges,
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
        let mut scratch = mergescratch::default();
        let mut out = Vec::new();
        self.encode_into(text, &mut scratch, &mut out)?;
        Ok(out)
    }

    /// Encodes `text` as ordinary text, appending the token ids to `out` and
    /// reusing `scratch` for the merge-loop buffers instead of allocating them
    /// per call.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unknowntoken`] when a symbol has no vocabulary
    /// entry, no complete byte fallback, and no configured unknown token.
    pub fn encode_into(
        &self,
        text: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        scratch.wordcache.clear();
        self.encode_text(text, scratch, out)
    }

    /// Encodes a batch of texts, sharing one scratch and one word cache so a
    /// pre-token seen in any earlier text of the batch is reused in later
    /// texts. Appends one token-id sequence per input text to `out`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::encode_into`] for any text in the
    /// batch.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        scratch: &mut mergescratch,
        out: &mut Vec<Vec<tokenid>>,
    ) -> Result<(), tokenizererror> {
        scratch.wordcache.clear();
        for text in texts {
            let mut ids = Vec::new();
            self.encode_text(text, scratch, &mut ids)?;
            out.push(ids);
        }
        Ok(())
    }

    /// Encodes one non-empty text without clearing the word cache.
    fn encode_text(
        &self,
        text: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        if text.is_empty() {
            return Ok(());
        }
        let normalized = self.normalized(text);
        self.encodeplainsegment(normalized.as_ref(), scratch, out)
    }

    /// Encodes `text`, recognizing special-token literals as structural markers.
    ///
    /// Special tokens are split on the raw text before normalization, so a
    /// normalizer that prepends or rewrites never touches a special literal.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::encode`] for the non-special segments.
    pub fn encodespecial(&self, text: &str) -> Result<Vec<tokenid>, tokenizererror> {
        let mut scratch = mergescratch::default();
        let mut out = Vec::new();
        self.encodespecial_into(text, &mut scratch, &mut out)?;
        Ok(out)
    }

    /// Encodes `text`, recognizing special-token literals, appending to `out`
    /// and reusing `scratch` for the merge-loop buffers.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::encode`] for the non-special segments.
    pub fn encodespecial_into(
        &self,
        text: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        scratch.wordcache.clear();
        let mut segment = String::new();
        let mut index = 0;
        while index < text.len() {
            if let Some((literal, id)) = self
                .specialtokens
                .iterlongest()
                .find(|(literal, _)| text[index..].starts_with(literal))
            {
                self.encodenormalizedsegment(&segment, scratch, out)?;
                segment.clear();
                out.push(id);
                index += literal.len();
                continue;
            }
            let character = text[index..]
                .chars()
                .next()
                .ok_or(tokenizererror::invalidutf8)?;
            segment.push(character);
            index += character.len_utf8();
        }
        self.encodenormalizedsegment(&segment, scratch, out)?;
        Ok(())
    }

    /// Normalizes one non-special segment, then encodes it through the
    /// pre-tokenizer pipeline.
    ///
    /// HF-compatible added-token pre-splitting runs on the normalized text
    /// *before* pre-tokenization: at each position the longest added token
    /// starting there maps to its id directly, and the unmatched runs in
    /// between go through the normal pipeline. This matters for non-special
    /// added tokens such as GPT-NeoX's space runs (50254-50277), which HF
    /// matches against the raw text even though ByteLevel pre-tokenization
    /// would turn spaces into `Ġ` byte forms; the gpt2/llama/qwen families
    /// carry no such tokens, so their path is unchanged.
    fn encodenormalizedsegment(
        &self,
        segment: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        if segment.is_empty() {
            return Ok(());
        }
        let normalized = self.normalized(segment);
        let mut pos = 0usize;
        let mut runstart = 0usize;
        while pos < normalized.len() {
            if let Some((literal, id)) = self
                .addedtokens
                .iter()
                .find(|(content, _)| normalized[pos..].starts_with(content.as_str()))
            {
                if runstart < pos {
                    self.encodeplainsegment(&normalized[runstart..pos], scratch, out)?;
                }
                out.push(*id);
                pos += literal.len();
                runstart = pos;
            } else {
                match normalized[pos..].chars().next() {
                    Some(character) => pos += character.len_utf8(),
                    None => break,
                }
            }
        }
        if runstart < normalized.len() {
            self.encodeplainsegment(&normalized[runstart..], scratch, out)?;
        }
        Ok(())
    }

    /// Encodes one non-special text segment through the pre-tokenizer pipeline.
    ///
    /// When a pre-tokenizer is installed, a repeated pre-token within the same
    /// encode hits the per-encode word cache in `scratch`, so each distinct
    /// pre-token is merged only once.
    fn encodeplainsegment(
        &self,
        segment: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        let pretokens = match &self.pretokenizer {
            Some(pretokenizer) => pretokenizer.pretokenize(segment)?,
            None => return self.encodepretoken(segment, scratch, out),
        };
        for pretoken in pretokens {
            if let Some(ids) = scratch.wordcache.get(&pretoken) {
                out.extend_from_slice(ids);
                continue;
            }
            let start = out.len();
            self.encodepretoken(&pretoken, scratch, out)?;
            let ids: Vec<tokenid> = out[start..].to_vec();
            scratch.wordcache.insert(pretoken, ids);
        }
        Ok(())
    }

    /// Encodes one pre-token through the active merge path.
    fn encodepretoken(
        &self,
        pretoken: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        if let Some(unigram) = &self.unigram {
            out.extend(unigram.encode(pretoken));
            return Ok(());
        }
        if self.ignoremerges {
            self.rankmerge(pretoken, scratch, out)
        } else {
            self.encodebypairids(pretoken, scratch, out)
        }
    }

    /// Encodes a pre-token's characters through the integer-keyed merge table.
    ///
    /// A character that is a vocabulary token is collected into a run; a
    /// character absent from the vocabulary is a non-mergeable leaf, so the run
    /// is merged via [`applyids_into`](bytepair::applyids_into) and the leaf is
    /// resolved directly. The result equals the string-keyed `apply` followed by
    /// [`resolvetoken`](crate::encode::resolvetoken), without concatenating
    /// symbol strings per candidate.
    fn encodebypairids(
        &self,
        pretoken: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        let mergescratch {
            merge,
            run,
            bytetokens,
            ..
        } = scratch;
        run.clear();
        for character in pretoken.chars() {
            let mut buffer = [0u8; 4];
            let symbol = character.encode_utf8(&mut buffer);
            if let Some(id) = self.vocabulary.getid(symbol) {
                run.push(id);
            } else {
                if !run.is_empty() {
                    self.bytepair.applyids_into(run, merge, out);
                    run.clear();
                }
                resolvenonvocab(
                    &self.vocabulary,
                    self.bytefallback,
                    self.unk,
                    symbol,
                    bytetokens,
                    out,
                )?;
            }
        }
        if !run.is_empty() {
            self.bytepair.applyids_into(run, merge, out);
        }
        Ok(())
    }

    /// Rank-based greedy merge: repeatedly merge the adjacent symbol pair whose
    /// concatenation is in the vocabulary with the lowest id. This is the
    /// `ignore_merges` path (tiktoken-derived vocabularies) and is equivalent
    /// to applying the derived merge table, but probes the vocabulary directly.
    fn rankmerge(
        &self,
        pretoken: &str,
        scratch: &mut mergescratch,
        out: &mut Vec<tokenid>,
    ) -> Result<(), tokenizererror> {
        let count = pretoken.chars().count();
        if count == 0 {
            return Ok(());
        }
        if count == 1 {
            resolvetoken(
                &self.vocabulary,
                self.bytefallback,
                self.unk,
                pretoken,
                &mut scratch.bytetokens,
                out,
            )?;
            return Ok(());
        }

        // `charstart[i]` is the byte offset where character `i` begins, so a
        // live symbol at index `i` is the borrowed slice
        // `pretoken[charstart[i]..end]`; no per-symbol string is materialized.
        let mergescratch {
            merge,
            charstart,
            bytetokens,
            ..
        } = scratch;
        charstart.clear();
        charstart.extend(pretoken.char_indices().map(|(offset, _)| offset));
        let source = ranksource {
            pretoken,
            charstart: charstart.as_slice(),
        };
        resetlinks(merge, count);
        merge.heap.clear();
        for index in 0..count - 1 {
            self.queue_rank_pair(&source, &merge.next, &merge.alive, &mut merge.heap, index);
        }

        while let Some(step) = merge.heap.pop() {
            self.try_rank_merge(
                &source,
                &mut merge.prev,
                &mut merge.next,
                &mut merge.alive,
                &mut merge.heap,
                step,
            );
        }

        let mut cursor = 0usize;
        loop {
            let end = source.end(&merge.next, cursor);
            resolvetoken(
                &self.vocabulary,
                self.bytefallback,
                self.unk,
                &pretoken[charstart[cursor]..end],
                bytetokens,
                out,
            )?;
            match merge.next[cursor] {
                Some(following) => cursor = following,
                None => break,
            }
        }
        Ok(())
    }

    /// Applies one heap-selected rank merge when it is still live and its id is
    /// still current; otherwise it is a stale candidate and is skipped.
    fn try_rank_merge(
        &self,
        source: &ranksource,
        prev: &mut [Option<usize>],
        next: &mut [Option<usize>],
        alive: &mut [bool],
        heap: &mut std::collections::BinaryHeap<mergestep>,
        step: mergestep,
    ) {
        let Some(right) = next[step.left] else {
            return;
        };
        if !alive[step.left] || !alive[right] {
            return;
        }
        let end = source.end(next, right);
        if self
            .vocabulary
            .getid(&source.pretoken[source.charstart[step.left]..end])
            .map(tokenid::get)
            != Some(step.rank as u32)
        {
            return;
        }
        alive[right] = false;
        let following = next[right];
        next[step.left] = following;
        if let Some(after) = following {
            prev[after] = Some(step.left);
        }
        if let Some(before) = prev[step.left] {
            self.queue_rank_pair(source, next, alive, heap, before);
        }
        self.queue_rank_pair(source, next, alive, heap, step.left);
    }

    /// Queues the mergeable pair starting at `left`, if its concatenation is in
    /// the vocabulary and both symbols are live.
    fn queue_rank_pair(
        &self,
        source: &ranksource,
        next: &[Option<usize>],
        alive: &[bool],
        heap: &mut std::collections::BinaryHeap<mergestep>,
        left: usize,
    ) {
        if !alive[left] {
            return;
        }
        let Some(right) = next[left] else {
            return;
        };
        if !alive[right] {
            return;
        }
        let end = source.end(next, right);
        if let Some(id) = self
            .vocabulary
            .getid(&source.pretoken[source.charstart[left]..end])
        {
            heap.push(mergestep {
                rank: id.get() as usize,
                left,
            });
        }
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
        self.decoder.decode(&self.vocabulary, ids)
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
///
/// Some real `tokenizer.json` files (e.g. GPT-2) omit `model.type`; the only
/// supported model is BPE, so an absent type is treated as BPE.
fn parsemodel(root: &Map<String, Value>) -> Result<(String, &Map<String, Value>), tokenizererror> {
    let model = root
        .get("model")
        .ok_or(tokenizererror::missingfield { field: "model" })?
        .as_object()
        .ok_or(tokenizererror::unexpectedtype {
            field: "model",
            expected: "object",
        })?;
    let modeltype = match model.get("type") {
        Some(value) => value.as_str().ok_or(tokenizererror::unexpectedtype {
            field: "model.type",
            expected: "string",
        })?,
        None => "BPE",
    }
    .to_string();
    Ok((modeltype, model))
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

/// Returns the `model.ignore_merges` flag, defaulting to `false`.
fn parseignoremerges(model: &Map<String, Value>) -> Result<bool, tokenizererror> {
    match model.get("ignore_merges") {
        Some(value) => value.as_bool().ok_or(tokenizererror::unexpectedtype {
            field: "model.ignore_merges",
            expected: "boolean",
        }),
        None => Ok(false),
    }
}

/// Returns the optional `model.unk_token` string.
fn parseunkcontent(model: &Map<String, Value>) -> Result<Option<String>, tokenizererror> {
    match model.get("unk_token") {
        Some(value) if value.is_null() => Ok(None),
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
        // Two in-the-wild shapes: `"left right"` strings and `["left", "right"]`
        // pairs (qwen3 uses the latter).
        if let Some(merge) = item.as_str() {
            merges.push(merge.to_string());
            continue;
        }
        if let Some(pair) = item.as_array()
            && let (Some(left), Some(right)) = (
                pair.first().and_then(Value::as_str),
                pair.get(1).and_then(Value::as_str),
            )
        {
            merges.push(format!("{left} {right}"));
            continue;
        }
        return Err(tokenizererror::malformedmerge { index });
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
