#![forbid(unsafe_code)]
//! External model format parsing and structural validation for the LLM runtime.
//!
//! `format` owns parsing of external model artifacts: a Llama-family
//! `config.json` ([`modelconfig`]) and the safetensors container header
//! ([`safetensorfile`]). It builds validated [`tensormetadata`] and performs
//! structural validation of the artifact. It depends on `lib` for resource
//! limits and `tensor` for the storage dtype and shape taxonomy. It MUST NOT
//! execute models, manage request state, or depend on any application crate.
//!
//! # Trust Model
//!
//! All input bytes are untrusted. Every length, offset, dimension, and element
//! count is checked with checked arithmetic before use, and every artifact is
//! validated before any allocation or access derived from it. Truncated or
//! malformed input produces a typed [`formaterror`], never a panic. Dtypes
//! outside the supported storage taxonomy are rejected explicitly, so `BOOL`,
//! `I16`, and the wider integer tokens are not silently coerced.
//!
//! `f16` and `bf16` are accepted as *storage* dtypes so their metadata can be
//! described; this crate makes no claim that any backend can execute them.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types they
//! export because a module and a re-exported type share one name namespace at
//! the crate root.

mod configs;
mod error;
mod metadata;
mod safetensor;
mod validation;

pub use configs::modelconfig;
pub use error::formaterror;
pub use metadata::{dtypetoken, tensormetadata, tokendtype};
pub use safetensor::{HEADERLENGTHSIZE, safetensorfile};
pub use validation::checkoverlaps;
