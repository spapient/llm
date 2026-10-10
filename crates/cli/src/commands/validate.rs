use format::{modelconfig, safetensorfile};

use crate::DEFAULTLIMITS;
use crate::DEFAULTMAXARTIFACTBYTES;
use crate::args::validateargs;
use crate::clierror;
use crate::commandoutput;
use crate::commands::{artifactkind, classify, readartifact};

/// Structurally validates one artifact.
///
/// # Errors
///
/// Returns a [`clierror`] from path validation or the read, or the structured
/// `formaterror`/`checkpointerror` when the artifact is invalid. A returned
/// error means the artifact failed validation; success means it passed.
pub(crate) fn run(args: &validateargs) -> Result<commandoutput, clierror> {
    let bytes = readartifact(&args.path, DEFAULTMAXARTIFACTBYTES)?;
    let mut report = format!("path: {}\n", args.path.display());
    match classify(&bytes) {
        artifactkind::config => {
            let config = modelconfig::parse(&bytes)?;
            report.push_str("artifact: modelconfig\nstatus: valid\n");
            report.push_str("model_type: llama\n");
            report.push_str(&format!("vocab_size: {}\n", config.getvocabsize()));
            report.push_str(&format!("hidden_size: {}\n", config.gethiddensize()));
        }
        artifactkind::checkpoint => {
            let container = checkpoint::checkpoint::decode(&bytes)?;
            let manifest = container.getmanifest();
            report.push_str("artifact: checkpoint\nstatus: valid\n");
            report.push_str(&format!("version: {}\n", manifest.getversion()));
            report.push_str(&format!("tensors: {}\n", manifest.getlen()));
        }
        artifactkind::safetensors => {
            let file = safetensorfile::parse(&bytes, &DEFAULTLIMITS)?;
            report.push_str("artifact: safetensors\nstatus: valid\n");
            report.push_str(&format!("tensors: {}\n", file.gettensors().len()));
            report.push_str(&format!("databytes: {}\n", file.getdatalength()));
        }
    }
    Ok(commandoutput::new(report))
}
