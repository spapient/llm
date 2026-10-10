use format::{modelconfig, safetensorfile};

use crate::DEFAULTLIMITS;
use crate::DEFAULTMAXARTIFACTBYTES;
use crate::args::inspectargs;
use crate::clierror;
use crate::commandoutput;
use crate::commands::{artifactkind, classify, readartifact};

/// Inspects one artifact's metadata.
///
/// # Errors
///
/// Returns a [`clierror`] from path validation, the read, or the selected
/// artifact parser (`format` for a config or safetensors container,
/// `checkpoint` for a checkpoint).
pub(crate) fn run(args: &inspectargs) -> Result<commandoutput, clierror> {
    let bytes = readartifact(&args.path, DEFAULTMAXARTIFACTBYTES)?;
    let mut report = format!("path: {}\n", args.path.display());
    match classify(&bytes) {
        artifactkind::config => {
            let config = modelconfig::parse(&bytes)?;
            report.push_str("artifact: modelconfig\n");
            report.push_str(&formatconfig(&config));
        }
        artifactkind::checkpoint => {
            let container = checkpoint::checkpoint::decode(&bytes)?;
            report.push_str("artifact: checkpoint\n");
            report.push_str(&formatcheckpoint(&container));
        }
        artifactkind::safetensors => {
            let file = safetensorfile::parse(&bytes, &DEFAULTLIMITS)?;
            report.push_str("artifact: safetensors\n");
            report.push_str(&formatsafetensors(&file));
        }
    }
    Ok(commandoutput::new(report))
}

/// Renders a validated model configuration.
fn formatconfig(config: &modelconfig) -> String {
    let mut out = String::from("model_type: llama\n");
    out.push_str(&format!("architectures: {:?}\n", config.getarchitectures()));
    out.push_str(&format!("hidden_size: {}\n", config.gethiddensize()));
    out.push_str(&format!(
        "intermediate_size: {}\n",
        config.getintermediatesize()
    ));
    out.push_str(&format!(
        "num_attention_heads: {}\n",
        config.getattentionheads()
    ));
    out.push_str(&format!(
        "num_key_value_heads: {}\n",
        config.getkeyvalueheads()
    ));
    out.push_str(&format!(
        "num_hidden_layers: {}\n",
        config.gethiddenlayers()
    ));
    out.push_str(&format!(
        "max_position_embeddings: {}\n",
        config.getmaxpositions()
    ));
    out.push_str(&format!("rms_norm_eps: {}\n", config.getrmsnormeps()));
    out.push_str(&format!("vocab_size: {}\n", config.getvocabsize()));
    out.push_str(&format!("rope_theta: {}\n", config.getropetheta()));
    out.push_str(&format!("head_size: {}\n", config.getheadsize()));
    out.push_str(&format!(
        "key_value_groups: {}\n",
        config.getkeyvaluegroups()
    ));
    if let Some(dtype) = config.gettorchdtype() {
        out.push_str(&format!("torch_dtype: {}\n", format::dtypetoken(dtype)));
    }
    if let Some(tied) = config.gettiewordembeddings() {
        out.push_str(&format!("tie_word_embeddings: {tied}\n"));
    }
    out
}

/// Renders a decoded checkpoint's manifest.
fn formatcheckpoint(container: &checkpoint::checkpoint) -> String {
    let manifest = container.getmanifest();
    let mut out = format!("version: {}\n", manifest.getversion());
    if let Some(state) = manifest.gettraining() {
        out.push_str(&format!(
            "training: step={} epoch={}\n",
            state.getstep(),
            state.getepoch()
        ));
    }
    let metadata = manifest.getmetadata();
    if !metadata.is_empty() {
        out.push_str("metadata:\n");
        for (key, value) in metadata {
            out.push_str(&format!("  {key} = {value}\n"));
        }
    }
    out.push_str(&format!("tensors: {}\n", manifest.getlen()));
    for entry in manifest.gettensors() {
        out.push_str(&format!(
            "  {}  {}  {:?}  {} bytes  offset={}\n",
            entry.getname(),
            format::dtypetoken(entry.getdtype()),
            entry.getshape().getdims(),
            entry.getnbytes(),
            entry.getoffset()
        ));
    }
    out
}

/// Renders a parsed safetensors header.
fn formatsafetensors(file: &safetensorfile) -> String {
    let mut out = format!(
        "headerbytes: {}\ndatabytes: {}\n",
        file.getheaderlength(),
        file.getdatalength()
    );
    let metadata = file.getmetadata();
    if !metadata.is_empty() {
        out.push_str("metadata:\n");
        for (key, value) in metadata {
            out.push_str(&format!("  {key} = {value}\n"));
        }
    }
    out.push_str(&format!("tensors: {}\n", file.gettensors().len()));
    for entry in file.gettensors() {
        out.push_str(&format!(
            "  {}  {}  {:?}  {} bytes  offsets=[{}, {})\n",
            entry.getname(),
            format::dtypetoken(entry.getdtype()),
            entry.getshape().getdims(),
            entry.getnbytes(),
            entry.getbegin(),
            entry.getend()
        ));
    }
    out
}
