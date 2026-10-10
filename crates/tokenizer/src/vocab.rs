use std::collections::BTreeMap;

use lib::tokenid;

use crate::error::tokenizererror;

/// An immutable, bidirectional token vocabulary.
///
/// A `vocabulary` owns two maps: token string to id and id to token string.
/// Both directions are populated from the same validated entry list, so a
/// lookup in one direction always succeeds exactly when the other does. The
/// id to token direction is stored in a `BTreeMap` rather than a dense vector
/// so that an out-of-range id from untrusted input cannot cause a large
/// allocation (rule `IN4`); ids are validated against the inserted entries,
/// never against their numeric magnitude.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct vocabulary {
    bycontent: BTreeMap<String, u32>,
    byid: BTreeMap<u32, String>,
}

impl vocabulary {
    /// Builds a vocabulary from `(token, id)` entries.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::duplicateid`] when one id is assigned to two
    /// different token strings and [`tokenizererror::duplicatecontent`] when
    /// one token string is assigned to two different ids. Repeating the exact
    /// same `(token, id)` pair is accepted.
    pub fn fromentries(entries: Vec<(String, u32)>) -> Result<Self, tokenizererror> {
        let mut bycontent = BTreeMap::new();
        let mut byid = BTreeMap::new();
        for (content, raw) in entries {
            if let Some(existing) = byid.get(&raw)
                && existing != &content
            {
                return Err(tokenizererror::duplicateid { id: raw });
            }
            if let Some(existing) = bycontent.get(&content)
                && *existing != raw
            {
                return Err(tokenizererror::duplicatecontent { token: content });
            }
            byid.insert(raw, content.clone());
            bycontent.insert(content, raw);
        }
        Ok(Self { bycontent, byid })
    }

    /// Returns the id assigned to `content`, if any.
    #[must_use]
    pub fn getid(&self, content: &str) -> Option<tokenid> {
        self.bycontent.get(content).copied().map(tokenid::new)
    }

    /// Returns the token string assigned to `id`, if any.
    #[must_use]
    pub fn getcontent(&self, id: tokenid) -> Option<&str> {
        self.byid.get(&id.get()).map(String::as_str)
    }

    /// Returns the number of distinct tokens in the vocabulary.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.byid.len()
    }

    /// Returns the smallest token id present, if the vocabulary is not empty.
    #[must_use]
    pub fn getminid(&self) -> Option<tokenid> {
        self.byid.keys().next().copied().map(tokenid::new)
    }

    /// Returns the largest token id present, if the vocabulary is not empty.
    #[must_use]
    pub fn getmaxid(&self) -> Option<tokenid> {
        self.byid.keys().next_back().copied().map(tokenid::new)
    }
}
