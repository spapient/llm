//! Consolidated integration-test entry point.
//!
//! Every repository-root suite lives here as a module of one test target
//! instead of one crate per file. A single target compiles and links once,
//! which removes the per-file link and monomorphization cost that previously
//! scaled with the number of suite files.
//!
//! Shared fixtures live in [`common`]; each suite reaches them through
//! `crate::common` rather than declaring its own private copy.

mod apicontract;
mod apikernelcontract;
mod artifact;
mod autograd;
mod autogradunits;
mod benchunits;
mod cancellation;
mod checkpointunits;
mod cli;
mod cliunits;
mod common;
mod context;
mod contextunits;
mod cpucontract;
mod cpuunits;
mod crosscrate;
mod dataunits;
mod diagnostics;
mod formatunits;
mod generation;
mod inferunits;
mod isolation;
mod libcontract;
mod lossdecrease;
mod modelforward;
mod modelsunits;
mod nnlayers;
mod nnunits;
mod optim;
mod optimunits;
mod pipeline;
mod roundtrip;
mod server;
mod serveunits;
mod service;
mod tensorcontract;
mod tensorunits;
mod testkitunits;
mod tokenizerunits;
mod tokens;
mod trainingresume;
mod trainunits;
