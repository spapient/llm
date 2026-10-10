//! Thin command-line entry point for the `cli` crate.
//!
//! All argument parsing, dispatch, and output formatting live in the `cli`
//! library; this binary only maps the returned exit code to the process status.
#![forbid(unsafe_code)]

use std::io;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut out = io::stdout().lock();
    let mut err = io::stderr().lock();
    ExitCode::from(cli::runcli(std::env::args_os(), &mut out, &mut err))
}
