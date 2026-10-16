//! Unified GPU availability probing for contract suites.
//!
//! Every GPU backend contract suite answers the same question — *may I
//! dispatch a kernel in this environment?* — with the same protocol, so a
//! missing device is reported identically across backends and can never be
//! mistaken for a passing numerical check:
//!
//! 1. **Probe.** Run the platform's authoritative, minimal device probe:
//!    CUDA is `nvidia-smi -L` (driver + device in one call); MUSA is
//!    `musaInfo` (the MUSA toolkit device query); Metal has no CLI probe, so
//!    the `metalcontract` suite probes in-process by constructing the
//!    executor, which fails with `devicenotavailable` when no default device
//!    exists.
//! 2. **Gate.** When the probe fails, honor the per-backend
//!    `LLM_<BACKEND>_REQUIRE_DEVICE` hard gate: when set, the test *fails*
//!    (a required device is missing); when unset, the suite prints
//!    `E4 未执行` and returns `false` so the caller skips numerical
//!    assertions. The printed line is the record; the harness reports the
//!    test as `ok`, which is why the hard gate exists.
//!
//! The probes are exact-match command checks: the command must exist and exit
//! successfully. A probe that cannot run (command absent, driver absent, no
//! device) reports `false` through the same path.

/// The GPU platform to probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum gpuprobe {
    /// NVIDIA CUDA: `nvidia-smi -L`.
    cuda,
    /// Moore Threads MUSA: `musaInfo`.
    musa,
}

impl gpuprobe {
    /// The probe command for this platform.
    fn command(self) -> &'static str {
        match self {
            gpuprobe::cuda => "nvidia-smi",
            gpuprobe::musa => "musaInfo",
        }
    }

    /// The environment variable that turns a missing device into a hard
    /// failure for this platform.
    fn requirevar(self) -> &'static str {
        match self {
            gpuprobe::cuda => "LLM_CUDA_REQUIRE_DEVICE",
            gpuprobe::musa => "LLM_MUSA_REQUIRE_DEVICE",
        }
    }

    /// The human-readable platform name used in the `E4` record.
    fn label(self) -> &'static str {
        match self {
            gpuprobe::cuda => "CUDA",
            gpuprobe::musa => "MUSA",
        }
    }
}

/// Returns `true` when a device for `probe` is reachable.
///
/// The probe is exact-match: the command must exist and exit successfully.
/// CUDA is probed with `nvidia-smi -L` (driver + device in one call); MUSA
/// with a bare `musaInfo` (the toolkit device query reports devices and exits
/// successfully when one is present).
fn command_succeeds(probe: gpuprobe) -> bool {
    let mut cmd = std::process::Command::new(probe.command());
    if probe == gpuprobe::cuda {
        cmd.arg("-L");
    }
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Probes `probe` and applies the environment gate.
///
/// Returns `true` when a kernel may be dispatched. When no device is reachable
/// it honors `LLM_<BACKEND>_REQUIRE_DEVICE`: set → assert (hard failure,
/// because a required device is missing); unset → print
/// `E4 未执行: {test} needs a {platform} device; ...` and return `false` so
/// the caller must return early instead of asserting a numerical result.
pub fn requiredevice(test: &str, probe: gpuprobe) -> bool {
    if command_succeeds(probe) {
        return true;
    }
    assert!(
        std::env::var_os(probe.requirevar()).is_none(),
        "{test}: {} is set but no {} device was detected",
        probe.requirevar(),
        probe.label(),
    );
    eprintln!(
        "E4 未执行: {test} needs a {} device; `{}` was unavailable or \
         reported no device. Set {} to fail instead.",
        probe.label(),
        probe.command(),
        probe.requirevar(),
    );
    false
}
