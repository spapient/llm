//! Ed25519 signatures over a `.u` snapshot's root hash.
//!
//! Signing the 32-byte root hash authenticates the whole model: the root hash
//! commits to every block's content through the Merkle DAG, so a single
//! 64-byte signature covers the entire snapshot. Verification is offline and
//! needs only the public key.

use crate::error::checkpointerror;

/// Derives the 32-byte Ed25519 public key from a 32-byte seed.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidsignaturekey`] when the seed is all zero.
pub fn publickey(secret: &[u8; 32]) -> Result<[u8; 32], checkpointerror> {
    let kp = keypair(*secret)?;
    Ok(*kp.pk)
}

/// Signs `message` with the 32-byte Ed25519 seed, returning the 64-byte
/// signature.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidsignaturekey`] when the seed is all zero.
pub fn sign(secret: &[u8; 32], message: &[u8]) -> Result<[u8; 64], checkpointerror> {
    let kp = keypair(*secret)?;
    Ok(*kp.sk.sign(message, None))
}

/// Verifies `signature` over `message` against the 32-byte public key.
#[must_use]
pub fn verify(public: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    let pk = ed25519_compact::PublicKey::new(*public);
    let sig = ed25519_compact::Signature::new(*signature);
    pk.verify(message, &sig).is_ok()
}

/// Builds a key pair from a seed, mapping an all-zero seed to an error instead
/// of the panic that [`ed25519_compact::KeyPair::from_seed`] would raise.
fn keypair(secret: [u8; 32]) -> Result<ed25519_compact::KeyPair, checkpointerror> {
    ed25519_compact::KeyPair::try_from_seed(ed25519_compact::Seed::new(secret))
        .map_err(|_| checkpointerror::invalidsignaturekey)
}
