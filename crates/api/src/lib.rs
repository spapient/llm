#![forbid(unsafe_code)]
//! Backend and execution contracts for the LLM runtime.
//!
//! `api` defines the narrow, static boundary between the runtime and a
//! concrete execution backend: device identity, declared capabilities, and the
//! generic kernel call contract. It owns no execution logic; concrete
//! implementations live in later layers (for example the `cpu` crate).
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated.
//!
//! # Dependencies
//!
//! `api` depends on `lib` for resource limits and on `tensor` for the
//! element, view, and owned tensor types used by the kernel contract. Neither
//! of those crates may depend on `api`.

// Module file names are chosen to differ from the public types they export: a
// module and a re-exported type share a single name namespace at the crate
// root, so `mod device` and `pub use device::device;` cannot coexist.
mod backends;
mod capability;
mod devices;
mod error;
mod kernels;

pub use backends::{backend, kernel};
pub use capability::capabilities;
pub use devices::{device, devicekind};
pub use error::backenderror;
pub use kernels::{binarykernel, compute, gatherkernel, weightedkernel};
