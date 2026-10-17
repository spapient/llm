/// Failures produced while constructing or using a [`tokenizer`](crate::tokenizer).
///
/// Every variant names a distinct cause a caller may act on: malformed JSON,
/// a missing or mistyped field, an unsupported model type, an inconsistent
/// vocabulary, a malformed merge or added-token record, an unknown token, an
/// out-of-vocabulary token id, invalid UTF-8 reconstruction, and an absent or
/// inapplicable chat template. None of these are collapsed into a single
/// opaque variant so that callers can distinguish them.
#[derive(Debug, PartialEq, Eq)]
pub enum tokenizererror {
    /// The input was not syntactically valid JSON.
    invalidjson {
        /// One-based line number of the syntax error.
        line: usize,
        /// One-based column number of the syntax error.
        column: usize,
    },

    /// The JSON document root was not an object.
    rootnotobject,

    /// A required field was absent.
    missingfield {
        /// The dotted path of the missing field.
        field: &'static str,
    },

    /// A present field had the wrong JSON type.
    unexpectedtype {
        /// The dotted path of the offending field.
        field: &'static str,
        /// The JSON type the field was expected to have.
        expected: &'static str,
    },

    /// The declared tokenizer model type is not supported.
    unsupportedmodel {
        /// The unsupported model type string.
        modeltype: String,
    },

    /// The declared pre-tokenizer type is not supported.
    unsupportedpretokenizer {
        /// The unsupported pre-tokenizer type string.
        pretype: String,
    },

    /// The declared decoder type is not supported.
    unsupporteddecoder {
        /// The unsupported decoder type string.
        decodertype: String,
    },

    /// The declared post-processor type is not supported.
    unsupportedpostprocessor {
        /// The unsupported post-processor type string.
        posttype: String,
    },

    /// A post-processor template entry was malformed.
    malformedpostprocessor {
        /// Position of the offending entry in the template array.
        index: usize,
    },

    /// A chat template could not be rendered by the Jinja subset.
    malformedtemplate {
        /// The rendering failure.
        detail: String,
    },

    /// The declared normalizer type is not supported.
    unsupportednormalizer {
        /// The unsupported normalizer type string.
        normaltype: String,
    },

    /// A pre-tokenizer Split pattern could not be compiled.
    invalidregex {
        /// The regex compilation failure.
        detail: String,
    },

    /// A `.tiktoken` file or token could not be parsed.
    invalidtiktoken {
        /// The parse failure.
        detail: String,
    },

    /// A token id did not fit in the supported `u32` range.
    idoutofrange {
        /// The rejected raw id.
        id: u64,
    },

    /// Two distinct token strings claimed the same id.
    duplicateid {
        /// The conflicting id.
        id: u32,
    },

    /// The same token string was assigned two different ids.
    duplicatecontent {
        /// The conflicting token string.
        token: String,
    },

    /// A merge entry was not a `"left right"` pair.
    malformedmerge {
        /// Position of the offending entry in the merges array.
        index: usize,
    },

    /// A merge rule names a symbol absent from the vocabulary, so it cannot be
    /// expressed as an integer-keyed pair.
    mergenotinvocab {
        /// The merge-side or merge-result symbol missing from the vocabulary.
        symbol: String,
    },

    /// An added-token entry was missing a required member or typed incorrectly.
    malformedspecialtoken {
        /// Position of the offending entry in the added-tokens array.
        index: usize,
    },

    /// A symbol could not be represented by any vocabulary token.
    unknowntoken {
        /// The unresolvable symbol.
        token: String,
    },

    /// A token id passed to decoding is not present in the vocabulary.
    unknowntokenid {
        /// The rejected id.
        id: u32,
    },

    /// Byte tokens did not reconstruct valid UTF-8.
    invalidutf8,

    /// A byte-level token contained a character outside the byte-to-unicode
    /// alphabet.
    invalidbytelevel {
        /// The offending token string.
        token: String,
    },

    /// A chat-template operation was requested but no template was installed.
    nochattemplate,

    /// A message role is not covered by the installed chat template.
    unsupportedrole {
        /// The uncovered role name.
        role: &'static str,
    },
}

impl std::fmt::Display for tokenizererror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::invalidjson { line, column } => {
                write!(f, "invalid JSON at line {line} column {column}")
            }
            Self::rootnotobject => f.write_str("tokenizer JSON root must be an object"),
            Self::missingfield { field } => write!(f, "missing required field: {field}"),
            Self::unexpectedtype { field, expected } => {
                write!(f, "field {field} must be a {expected}")
            }
            Self::unsupportedmodel { modeltype } => {
                write!(f, "unsupported tokenizer model type: {modeltype}")
            }
            Self::unsupportedpretokenizer { pretype } => {
                write!(f, "unsupported pre_tokenizer type: {pretype}")
            }
            Self::unsupporteddecoder { decodertype } => {
                write!(f, "unsupported decoder type: {decodertype}")
            }
            Self::unsupportedpostprocessor { posttype } => {
                write!(f, "unsupported post_processor type: {posttype}")
            }
            Self::malformedpostprocessor { index } => {
                write!(f, "malformed post_processor entry at index {index}")
            }
            Self::malformedtemplate { detail } => {
                write!(f, "cannot render chat template: {detail}")
            }
            Self::unsupportednormalizer { normaltype } => {
                write!(f, "unsupported normalizer type: {normaltype}")
            }
            Self::invalidregex { detail } => {
                write!(f, "cannot compile pre-tokenizer regex: {detail}")
            }
            Self::invalidtiktoken { detail } => write!(f, "invalid tiktoken: {detail}"),
            Self::idoutofrange { id } => write!(f, "token id {id} is out of range"),
            Self::duplicateid { id } => {
                write!(f, "token id {id} is assigned to more than one token")
            }
            Self::duplicatecontent { token } => {
                write!(f, "token {token:?} is assigned to more than one id")
            }
            Self::malformedmerge { index } => write!(f, "malformed merge entry at index {index}"),
            Self::mergenotinvocab { symbol } => {
                write!(f, "merge symbol {symbol:?} is not in the vocabulary")
            }
            Self::malformedspecialtoken { index } => {
                write!(f, "malformed added token at index {index}")
            }
            Self::unknowntoken { token } => write!(
                f,
                "token {token:?} is not in the vocabulary and has no fallback"
            ),
            Self::unknowntokenid { id } => {
                write!(f, "token id {id} is not present in the vocabulary")
            }
            Self::invalidutf8 => f.write_str("token bytes do not form valid UTF-8"),
            Self::invalidbytelevel { token } => write!(
                f,
                "byte-level token {token:?} contains a character outside the byte-to-unicode alphabet"
            ),
            Self::nochattemplate => f.write_str("no chat template is installed"),
            Self::unsupportedrole { role } => {
                write!(f, "chat template has no entry for role {role}")
            }
        }
    }
}

impl std::error::Error for tokenizererror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
