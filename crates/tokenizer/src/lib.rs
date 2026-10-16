#![forbid(unsafe_code)]
//! Tokenization, vocabulary, byte-pair encoding, special tokens, and chat
//! templates for the LLM runtime.
//!
//! `tokenizer` owns text-to-token and token-to-text conversion. It loads a
//! HuggingFace `tokenizer.json` subset with `serde_json`, applies byte-pair
//! merges with optional byte fallback, resolves special tokens, and renders
//! explicit chat templates. It owns no tensor, model-weight, KV-cache, or
//! request-scheduling behavior.
//!
//! # Dependencies
//!
//! `tokenizer` depends on `lib` for [`tokenid`](lib::tokenid) and on
//! `serde_json` for parsing. It must not depend on any higher-level crate.
//!
//! # Errors
//!
//! Malformed input is rejected through [`tokenizererror`]. Token ids are
//! validated against the loaded vocabulary before use (rule `IN1`).
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export (`vocab` exports [`vocabulary`], `merges` exports [`bytepair`],
//! `special` exports [`specialtokens`], `engine` exports [`tokenizer`],
//! `chat` exports the chat types), because a module and a re-exported type
//! share one name namespace and cannot both use the same name.

mod chat;
mod decode;
mod encode;
mod engine;
mod error;
mod jinja;
mod merges;
mod normalize;
mod postprocess;
mod pretoken;
mod scratch;
mod special;
mod tiktoken;
mod unigram;
mod vocab;

pub use chat::{chatmessage, chatrole, chattemplate, chattemplateentry};
pub use decode::decoder;
pub use engine::tokenizer;
pub use error::tokenizererror;
pub use merges::bytepair;
pub use normalize::normalizer;
pub use postprocess::postprocessor;
pub use pretoken::{bytetounicode, gpt2split, o200ksplit, pretokenizer, unicodetobyte};
pub use scratch::mergescratch;
pub use special::specialtokens;
pub use tiktoken::tiktokenizer;
pub use vocab::vocabulary;
