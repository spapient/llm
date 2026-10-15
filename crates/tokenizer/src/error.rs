use thiserror::Error;

/// Failures produced while constructing or using a [`tokenizer`](crate::tokenizer).
///
/// Every variant names a distinct cause a caller may act on: malformed JSON,
/// a missing or mistyped field, an unsupported model type, an inconsistent
/// vocabulary, a malformed merge or added-token record, an unknown token, an
/// out-of-vocabulary token id, invalid UTF-8 reconstruction, and an absent or
/// inapplicable chat template. None of these are collapsed into a single
/// opaque variant so that callers can distinguish them.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum tokenizererror {
    /// The input was not syntactically valid JSON.
    #[error("invalid JSON at line {line} column {column}")]
    invalidjson {
        /// One-based line number of the syntax error.
        line: usize,
        /// One-based column number of the syntax error.
        column: usize,
    },

    /// The JSON document root was not an object.
    #[error("tokenizer JSON root must be an object")]
    rootnotobject,

    /// A required field was absent.
    #[error("missing required field: {field}")]
    missingfield {
        /// The dotted path of the missing field.
        field: &'static str,
    },

    /// A present field had the wrong JSON type.
    #[error("field {field} must be a {expected}")]
    unexpectedtype {
        /// The dotted path of the offending field.
        field: &'static str,
        /// The JSON type the field was expected to have.
        expected: &'static str,
    },

    /// The declared tokenizer model type is not supported.
    #[error("unsupported tokenizer model type: {modeltype}")]
    unsupportedmodel {
        /// The unsupported model type string.
        modeltype: String,
    },

    /// The declared pre-tokenizer type is not supported.
    #[error("unsupported pre_tokenizer type: {pretype}")]
    unsupportedpretokenizer {
        /// The unsupported pre-tokenizer type string.
        pretype: String,
    },

    /// The declared decoder type is not supported.
    #[error("unsupported decoder type: {decodertype}")]
    unsupporteddecoder {
        /// The unsupported decoder type string.
        decodertype: String,
    },

    /// The declared post-processor type is not supported.
    #[error("unsupported post_processor type: {posttype}")]
    unsupportedpostprocessor {
        /// The unsupported post-processor type string.
        posttype: String,
    },

    /// A post-processor template entry was malformed.
    #[error("malformed post_processor entry at index {index}")]
    malformedpostprocessor {
        /// Position of the offending entry in the template array.
        index: usize,
    },

    /// A chat template could not be rendered by the Jinja subset.
    #[error("cannot render chat template: {detail}")]
    malformedtemplate {
        /// The rendering failure.
        detail: String,
    },

    /// The declared normalizer type is not supported.
    #[error("unsupported normalizer type: {normaltype}")]
    unsupportednormalizer {
        /// The unsupported normalizer type string.
        normaltype: String,
    },

    /// A pre-tokenizer Split pattern could not be compiled.
    #[error("cannot compile pre-tokenizer regex: {detail}")]
    invalidregex {
        /// The regex compilation failure.
        detail: String,
    },

    /// A token id did not fit in the supported `u32` range.
    #[error("token id {id} is out of range")]
    idoutofrange {
        /// The rejected raw id.
        id: u64,
    },

    /// Two distinct token strings claimed the same id.
    #[error("token id {id} is assigned to more than one token")]
    duplicateid {
        /// The conflicting id.
        id: u32,
    },

    /// The same token string was assigned two different ids.
    #[error("token {token:?} is assigned to more than one id")]
    duplicatecontent {
        /// The conflicting token string.
        token: String,
    },

    /// A merge entry was not a `"left right"` pair.
    #[error("malformed merge entry at index {index}")]
    malformedmerge {
        /// Position of the offending entry in the merges array.
        index: usize,
    },

    /// An added-token entry was missing a required member or typed incorrectly.
    #[error("malformed added token at index {index}")]
    malformedspecialtoken {
        /// Position of the offending entry in the added-tokens array.
        index: usize,
    },

    /// A symbol could not be represented by any vocabulary token.
    #[error("token {token:?} is not in the vocabulary and has no fallback")]
    unknowntoken {
        /// The unresolvable symbol.
        token: String,
    },

    /// A token id passed to decoding is not present in the vocabulary.
    #[error("token id {id} is not present in the vocabulary")]
    unknowntokenid {
        /// The rejected id.
        id: u32,
    },

    /// Byte tokens did not reconstruct valid UTF-8.
    #[error("token bytes do not form valid UTF-8")]
    invalidutf8,

    /// A byte-level token contained a character outside the byte-to-unicode
    /// alphabet.
    #[error("byte-level token {token:?} contains a character outside the byte-to-unicode alphabet")]
    invalidbytelevel {
        /// The offending token string.
        token: String,
    },

    /// A chat-template operation was requested but no template was installed.
    #[error("no chat template is installed")]
    nochattemplate,

    /// A message role is not covered by the installed chat template.
    #[error("chat template has no entry for role {role}")]
    unsupportedrole {
        /// The uncovered role name.
        role: &'static str,
    },
}
