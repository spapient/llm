use std::collections::HashMap;

use lib::tokenid;

use crate::error::tokenizererror;

/// An immutable, bidirectional token vocabulary.
///
/// A `vocabulary` owns two maps: token string to id and id to token string.
/// Both directions are populated from the same validated entry list, so a
/// lookup in one direction always succeeds exactly when the other does. Both
/// directions use `HashMap` for O(1) lookups rather than a dense vector, so an
/// out-of-range id from untrusted input cannot cause a large allocation (rule
/// `IN4`); ids are validated against the inserted entries, never against their
/// numeric magnitude. The smallest and largest ids are tracked during
/// construction for `getminid`/`getmaxid`. A 256-entry byte-fallback table maps
/// each byte to its `<0xNN>` token id when present, so byte-fallback encoding
/// does not allocate per byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct vocabulary {
    bycontent: HashMap<String, u32>,
    byid: HashMap<u32, String>,
    bytetokens: [Option<tokenid>; 256],
    minid: Option<u32>,
    maxid: Option<u32>,
}

impl Default for vocabulary {
    fn default() -> Self {
        Self {
            bycontent: HashMap::new(),
            byid: HashMap::new(),
            bytetokens: [None; 256],
            minid: None,
            maxid: None,
        }
    }
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
        let mut bycontent = HashMap::with_capacity(entries.len());
        let mut byid = HashMap::with_capacity(entries.len());
        let mut minid: Option<u32> = None;
        let mut maxid: Option<u32> = None;
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
            minid = Some(minid.map_or(raw, |current| current.min(raw)));
            maxid = Some(maxid.map_or(raw, |current| current.max(raw)));
            byid.insert(raw, content.clone());
            bycontent.insert(content, raw);
        }
        let mut bytetokens = [None; 256];
        for (byte, slot) in bytetokens.iter_mut().enumerate() {
            let token = format!("<0x{byte:02X}>");
            *slot = bycontent.get(&token).copied().map(tokenid::new);
        }
        Ok(Self {
            bycontent,
            byid,
            bytetokens,
            minid,
            maxid,
        })
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

    /// Returns the byte-fallback token id for `byte`, if the `<0xNN>` token is
    /// present in the vocabulary.
    pub(crate) fn getbytetokenid(&self, byte: u8) -> Option<tokenid> {
        self.bytetokens[byte as usize]
    }

    /// Returns the number of distinct tokens in the vocabulary.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.byid.len()
    }

    /// Returns the smallest token id present, if the vocabulary is not empty.
    #[must_use]
    pub fn getminid(&self) -> Option<tokenid> {
        self.minid.map(tokenid::new)
    }

    /// Returns the largest token id present, if the vocabulary is not empty.
    #[must_use]
    pub fn getmaxid(&self) -> Option<tokenid> {
        self.maxid.map(tokenid::new)
    }
}
