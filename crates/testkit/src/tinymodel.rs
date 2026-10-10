//! Reserved space for the tiny deterministic model fixture.
//!
//! A tiny model fixture needs an `nn` and a `models` layer to define layers,
//! parameters, and weight initialization. Neither crate exists yet, so this
//! module intentionally defines no items. It records where the fixture will
//! live once those layers land, keeping the reservation visible in the type
//! system rather than only in prose.
//!
//! See [design.md](../../../docs/design.md) for the model-support
//! contract and [engineering.md](../../../docs/engineering.md) for the role
//! of shared test utilities.
