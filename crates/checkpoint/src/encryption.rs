//! AES-256-GCM block encryption for the `.u` snapshot format.
//!
//! Each block is encrypted independently with a deterministic nonce derived
//! from its 128-bit content identity, so identical plaintext still deduplicates
//! and a reader decrypts only the blocks a requested tensor spans.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};

use crate::error::checkpointerror;

/// The AES-256-GCM authentication tag length in bytes.
const TAGLEN: usize = 16;

/// Encrypts one block, returning the ciphertext (same length as the plaintext)
/// and the 16-byte authentication tag.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] when the key length is
/// unexpected (a programming invariant, not a runtime condition).
pub fn encryptblock(
    key: &[u8; 32],
    nonce: &[u8; 12],
    plaintext: &[u8],
) -> Result<(Vec<u8>, [u8; 16]), checkpointerror> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| checkpointerror::invalidmanifest {
        reason: "invalid aes key length".to_string(),
    })?;
    let nonce = Nonce::from_slice(nonce);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|_| checkpointerror::invalidmanifest {
            reason: "aes-gcm encryption failed".to_string(),
        })?;
    let split = ciphertext.len() - TAGLEN;
    let mut tag = [0u8; TAGLEN];
    tag.copy_from_slice(&ciphertext[split..]);
    Ok((ciphertext[..split].to_vec(), tag))
}

/// Decrypts one block, authenticating the ciphertext with its tag.
///
/// # Errors
///
/// Returns [`checkpointerror::decryptionfailed`] when the tag does not verify
/// (wrong key or tampered ciphertext).
pub fn decryptblock(
    key: &[u8; 32],
    nonce: &[u8; 12],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Result<Vec<u8>, checkpointerror> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| checkpointerror::invalidmanifest {
        reason: "invalid aes key length".to_string(),
    })?;
    let nonce = Nonce::from_slice(nonce);
    let mut combined = Vec::with_capacity(ciphertext.len() + TAGLEN);
    combined.extend_from_slice(ciphertext);
    combined.extend_from_slice(tag);
    cipher
        .decrypt(nonce, combined.as_ref())
        .map_err(|_| checkpointerror::decryptionfailed)
}

/// Derives the deterministic 96-bit nonce from a block's 128-bit content
/// identity.
#[must_use]
pub fn nonceof(blockhash: &[u8; 16]) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&blockhash[..12]);
    nonce
}
