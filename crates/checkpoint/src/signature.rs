//! Ed25519 signatures over a `.u` snapshot's root hash.
//!
//! Signing the 32-byte root hash authenticates the whole model: the root hash
//! commits to every block's content through the Merkle DAG, so a single
//! 64-byte signature covers the entire snapshot. Verification is offline and
//! needs only the public key.

/// Derives the 32-byte Ed25519 public key from a 32-byte seed.
///
/// # Panics
///
/// Panics if the seed is all zero, matching [`ed25519_compact::KeyPair::from_seed`];
/// an all-zero seed is a programming error, not a runtime condition.
#[must_use]
pub fn publickey(secret: &[u8; 32]) -> [u8; 32] {
    let kp = ed25519_compact::KeyPair::from_seed(ed25519_compact::Seed::new(*secret));
    *kp.pk
}

/// Signs `message` with the 32-byte Ed25519 seed, returning the 64-byte
/// signature.
///
/// # Panics
///
/// Panics if the seed is all zero, matching [`ed25519_compact::KeyPair::from_seed`].
#[must_use]
pub fn sign(secret: &[u8; 32], message: &[u8]) -> [u8; 64] {
    let kp = ed25519_compact::KeyPair::from_seed(ed25519_compact::Seed::new(*secret));
    *kp.sk.sign(message, None)
}

/// Verifies `signature` over `message` against the 32-byte public key.
#[must_use]
pub fn verify(public: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    let pk = ed25519_compact::PublicKey::new(*public);
    let sig = ed25519_compact::Signature::new(*signature);
    pk.verify(message, &sig).is_ok()
}
