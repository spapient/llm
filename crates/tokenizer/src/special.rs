use std::collections::BTreeMap;

use lib::tokenid;

/// An immutable set of special tokens keyed by both content and id.
///
/// Special tokens are the entries a tokenizer reserves for control markers such
/// as beginning-of-sequence, end-of-sequence, and role delimiters. They are
/// stored separately from the ordinary vocabulary so that encoding can treat
/// their literal strings as structural rather than as text to be byte-pair
/// encoded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct specialtokens {
    bycontent: BTreeMap<String, u32>,
    byid: BTreeMap<u32, String>,
}

impl specialtokens {
    /// Builds a special-token set from `(content, id)` entries.
    ///
    /// Later entries for the same content or id overwrite earlier ones. Callers
    /// that require conflict detection must validate through
    /// [`vocabulary`](crate::vocabulary) first, which rejects duplicate ids and
    /// duplicate content across the combined vocabulary.
    #[must_use]
    pub fn new(entries: Vec<(String, u32)>) -> Self {
        let mut bycontent = BTreeMap::new();
        let mut byid = BTreeMap::new();
        for (content, id) in entries {
            bycontent.insert(content.clone(), id);
            byid.insert(id, content);
        }
        Self { bycontent, byid }
    }

    /// Returns the id assigned to the special token `content`, if any.
    #[must_use]
    pub fn getid(&self, content: &str) -> Option<tokenid> {
        self.bycontent.get(content).copied().map(tokenid::new)
    }

    /// Returns the special token string assigned to `id`, if any.
    #[must_use]
    pub fn getcontent(&self, id: tokenid) -> Option<&str> {
        self.byid.get(&id.get()).map(String::as_str)
    }

    /// Reports whether `content` is a special token.
    #[must_use]
    pub fn contains(&self, content: &str) -> bool {
        self.bycontent.contains_key(content)
    }

    /// Returns the number of special tokens.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.byid.len()
    }

    /// Iterates over `(content, id)` pairs in content order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, tokenid)> + '_ {
        self.bycontent
            .iter()
            .map(|(content, id)| (content.as_str(), tokenid::new(*id)))
    }
}
