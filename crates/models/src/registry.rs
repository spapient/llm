use crate::error::modelerror;
use crate::llama::{llamaconfig, llamamodel, llamaweights};

/// A recognized model architecture.
///
/// The registry maps an architecture name to the constructor that builds it.
/// Dispatch is a `match` over the enum, not a name-keyed table of trait objects,
/// so an unknown name is rejected once and every recognized architecture is
/// checked at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum architecture {
    /// The Llama family (RMSNorm, RoPE, grouped-query attention, SwiGLU).
    llama,
    /// The selective state-space family (the `arch: trap` base and Mamba-class
    /// models); built via `nn::ssm`, not [`Self::newmodel`].
    mamba,
}

impl architecture {
    /// Returns the canonical architecture name.
    #[must_use]
    pub const fn getname(self) -> &'static str {
        match self {
            Self::llama => "llama",
            Self::mamba => "mamba",
        }
    }

    /// Constructs a model of this architecture from a configuration and
    /// weights.
    ///
    /// The method is named `newmodel` rather than `new` because it returns a
    /// `llamamodel`, not `Self`; Clippy's `new_ret_no_self` rejects `new` for a
    /// factory of another type.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when the weights do not agree with the
    /// configuration.
    pub fn newmodel(
        self,
        config: llamaconfig,
        weights: llamaweights,
    ) -> Result<llamamodel, modelerror> {
        match self {
            Self::llama => llamamodel::new(config, weights),
            Self::mamba => Err(modelerror::config(
                "architecture",
                "mamba models are built via nn::ssm, not registry::newmodel",
            )),
        }
    }
}

/// Resolves an architecture name to its [`architecture`].
///
/// # Errors
///
/// Returns [`modelerror::unknownarchitecture`] when `name` is not recognized.
pub fn getarchitecture(name: &str) -> Result<architecture, modelerror> {
    match name {
        "llama" => Ok(architecture::llama),
        "mamba" => Ok(architecture::mamba),
        other => Err(modelerror::unknownarchitecture {
            name: other.to_string(),
        }),
    }
}
