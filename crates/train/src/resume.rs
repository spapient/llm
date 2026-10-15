use std::collections::BTreeMap;
use std::path::Path;

use ::checkpoint::compatibility::{checkcompat, checkversion, tensorspec};
use ::checkpoint::{checkpoint, loadcheckpoint, savecheckpoint, trainingstate};
use ::tensor::{dtype, tensor};
use optim::{optimizer, optimizerstate, statetensor};

use crate::error::trainerror;
use crate::trainers::trainer;

/// The checkpoint container version `train` writes and accepts.
const CHECKPOINTVERSION: u32 = 1;

/// The tensor-name prefix for the parameter at a given index.
const PARAMPREFIX: &str = "p.";

/// Metadata key for the optimizer's stable name.
const OPTIMIZERNAMEKEY: &str = "optimizer.name";

/// Metadata key for the optimizer's completed-step count.
const OPTIMIZERSTEPKEY: &str = "optimizer.step";

/// Metadata key for the optimizer's learning rate.
const OPTIMIZERLRKEY: &str = "optimizer.lr";

/// Metadata key for the optimizer's tracked-parameter count.
const OPTIMIZERPARAMSKEY: &str = "optimizer.params";

/// Metadata key for the comma-separated optimizer tensor names.
const OPTIMIZERTENSORSKEY: &str = "optimizer.tensors";

/// The progress restored from a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct resumestate {
    step: u64,
    epoch: u64,
}

impl resumestate {
    /// Returns the restored completed-step count.
    #[must_use]
    pub const fn getstep(self) -> u64 {
        self.step
    }

    /// Returns the restored epoch count.
    #[must_use]
    pub const fn getepoch(self) -> u64 {
        self.epoch
    }
}

/// Writes a trainer's parameters, optimizer state, and progress to a checkpoint.
///
/// The persistence container is owned by `checkpoint`: `train` only chooses the
/// tensor names (`p.{i}` for parameters and each optimizer state tensor under
/// its own name) and records the optimizer scalars as manifest metadata. The
/// completed step and epoch are stored through
/// [`trainingstate`], which is the checkpoint crate's own type.
///
/// # Errors
///
/// Returns any `checkpointerror` raised while adding tensors or writing the
/// file.
pub fn saveresume<O: optimizer>(trainer: &trainer<O>, path: &Path) -> Result<(), trainerror> {
    let mut ckpt = checkpoint::new(CHECKPOINTVERSION);
    ckpt.settraining(trainingstate::new(trainer.getstep(), trainer.getepoch()));
    for (index, param) in trainer.getparams().iter().enumerate() {
        ckpt.addtensor(
            paramname(index),
            dtype::f32,
            param.getshape().clone(),
            &encodef32(param.getdata()),
        )?;
    }

    let optimizer = trainer.getoptimizer();
    let snapshot = optimizer.getstate();
    ckpt.addmetadata(
        OPTIMIZERNAMEKEY.to_string(),
        optimizer.getname().to_string(),
    );
    ckpt.addmetadata(OPTIMIZERSTEPKEY.to_string(), snapshot.getstep().to_string());
    ckpt.addmetadata(OPTIMIZERLRKEY.to_string(), snapshot.getlr().to_string());
    ckpt.addmetadata(
        OPTIMIZERPARAMSKEY.to_string(),
        snapshot.getparamcount().to_string(),
    );
    let names: Vec<String> = snapshot
        .gettensors()
        .iter()
        .map(|entry| entry.getname().to_string())
        .collect();
    ckpt.addmetadata(OPTIMIZERTENSORSKEY.to_string(), names.join(","));
    for entry in snapshot.gettensors() {
        ckpt.addtensor(
            entry.getname().to_string(),
            dtype::f32,
            entry.getshape().clone(),
            &entry.tobytes(),
        )?;
    }

    savecheckpoint(path, &ckpt)?;
    Ok(())
}

/// Restores a trainer's parameters, optimizer state, and progress from a checkpoint.
///
/// The load is atomic with respect to the trainer: the optimizer state is
/// validated and restored first, and only then are the parameters and counters
/// replaced. A rejected load therefore leaves the trainer exactly as it was
/// (`OW3`). The optimizer's restored step must agree with the persisted training
/// step and the optimizer name must match the trainer's optimizer, so a
/// checkpoint written for a different optimizer is rejected rather than applied
/// silently.
///
/// # Errors
///
/// Returns [`trainerror::state`] for missing or inconsistent metadata, a version
/// or tensor mismatch through [`checkcompat`]/[`checkversion`], the optimizer's
/// own error through [`optimizer::restore`], and any tensor decode error.
pub fn loadresume<O: optimizer>(
    trainer: &mut trainer<O>,
    path: &Path,
) -> Result<resumestate, trainerror> {
    let ckpt = loadcheckpoint(path)?;
    checkversion(&ckpt, CHECKPOINTVERSION)?;
    let training = ckpt
        .getmanifest()
        .gettraining()
        .ok_or_else(|| trainerror::state("checkpoint has no training state"))?;

    let expected: Vec<tensorspec> = trainer
        .getparams()
        .iter()
        .enumerate()
        .map(|(index, param)| {
            tensorspec::new(paramname(index), dtype::f32, param.getshape().clone())
        })
        .collect();
    checkcompat(&ckpt, &expected)?;

    let loaded = loadparameters(&ckpt, &expected)?;

    let metadata = ckpt.getmanifest().getmetadata();
    checkoptimizername(metadata, trainer.getoptimizer().getname())?;
    let optimstep = parseu64(metadata, OPTIMIZERSTEPKEY)?;
    let lr = parsef32(metadata, OPTIMIZERLRKEY)?;
    let paramcount = parseusize(metadata, OPTIMIZERPARAMSKEY)?;
    let tensorlist = metadata.get(OPTIMIZERTENSORSKEY).ok_or_else(|| {
        trainerror::state(format!(
            "checkpoint metadata {OPTIMIZERTENSORSKEY} is missing"
        ))
    })?;
    let tensors = loadoptimizertensors(&ckpt, tensorlist)?;

    if paramcount != loaded.len() {
        return Err(trainerror::state(format!(
            "checkpoint describes {paramcount} optimizer slots but {} parameters were loaded",
            loaded.len()
        )));
    }
    if optimstep != training.getstep() {
        return Err(trainerror::state(format!(
            "optimizer step {optimstep} disagrees with training step {}",
            training.getstep()
        )));
    }

    let snapshot = optimizerstate::new(optimstep, lr, paramcount, tensors);
    trainer.getoptimizermut().restore(&snapshot)?;
    trainer.replaceparams(loaded)?;
    trainer.setstep(training.getstep());
    trainer.setepoch(training.getepoch());
    Ok(resumestate {
        step: training.getstep(),
        epoch: training.getepoch(),
    })
}

/// Loads and validates the trainer parameters described by `expected`.
fn loadparameters(
    ckpt: &checkpoint,
    expected: &[tensorspec],
) -> Result<Vec<tensor<f32>>, trainerror> {
    let mut loaded = Vec::with_capacity(expected.len());
    for index in 0..expected.len() {
        let name = paramname(index);
        let entry = ckpt
            .gettensor(&name)
            .ok_or_else(|| trainerror::state(format!("checkpoint tensor {name:?} is missing")))?;
        if entry.getdtype() != dtype::f32 {
            return Err(trainerror::unsupportedtensordtype {
                name,
                dtype: entry.getdtype(),
            });
        }
        let shape = entry.getshape().clone();
        let bytes = ckpt.readtensor(&name).ok_or_else(|| {
            trainerror::state(format!("checkpoint tensor {name:?} cannot be read"))
        })?;
        loaded.push(tensor::new(shape, decodef32(&bytes)?)?);
    }
    Ok(loaded)
}

/// Verifies the checkpoint's optimizer name matches the trainer's optimizer.
fn checkoptimizername(
    metadata: &BTreeMap<String, String>,
    traineroptimizer: &str,
) -> Result<(), trainerror> {
    let name = metadata.get(OPTIMIZERNAMEKEY).ok_or_else(|| {
        trainerror::state(format!("checkpoint metadata {OPTIMIZERNAMEKEY} is missing"))
    })?;
    if name != traineroptimizer {
        return Err(trainerror::state(format!(
            "checkpoint was written for optimizer {name:?} but the trainer holds {traineroptimizer:?}"
        )));
    }
    Ok(())
}

/// Loads the optimizer state tensors named in the comma-separated `tensorlist`.
fn loadoptimizertensors(
    ckpt: &checkpoint,
    tensorlist: &str,
) -> Result<Vec<statetensor>, trainerror> {
    let names: Vec<&str> = if tensorlist.is_empty() {
        Vec::new()
    } else {
        tensorlist.split(',').collect()
    };
    let mut tensors = Vec::with_capacity(names.len());
    for tensorname in names {
        let entry = ckpt.gettensor(tensorname).ok_or_else(|| {
            trainerror::state(format!("optimizer tensor {tensorname:?} is missing"))
        })?;
        let shape = entry.getshape().clone();
        let bytes = ckpt.readtensor(tensorname).ok_or_else(|| {
            trainerror::state(format!("optimizer tensor {tensorname:?} cannot be read"))
        })?;
        tensors.push(statetensor::new(
            tensorname.to_string(),
            shape,
            decodef32(&bytes)?,
        )?);
    }
    Ok(tensors)
}

fn paramname(index: usize) -> String {
    format!("{PARAMPREFIX}{index}")
}

fn encodef32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(core::mem::size_of::<f32>()));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decodef32(bytes: &[u8]) -> Result<Vec<f32>, trainerror> {
    let width = core::mem::size_of::<f32>();
    if !bytes.len().is_multiple_of(width) {
        return Err(trainerror::state(format!(
            "tensor byte length {} is not a multiple of {width}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / width);
    for chunk in bytes.chunks_exact(width) {
        let raw: [u8; 4] = chunk
            .try_into()
            .map_err(|_| trainerror::state("f32 byte chunk is not four bytes"))?;
        out.push(f32::from_le_bytes(raw));
    }
    Ok(out)
}

fn parseu64(metadata: &BTreeMap<String, String>, key: &str) -> Result<u64, trainerror> {
    metadata
        .get(key)
        .ok_or_else(|| trainerror::state(format!("checkpoint metadata {key} is missing")))?
        .parse::<u64>()
        .map_err(|_| trainerror::state(format!("checkpoint metadata {key} is not an integer")))
}

fn parseusize(metadata: &BTreeMap<String, String>, key: &str) -> Result<usize, trainerror> {
    metadata
        .get(key)
        .ok_or_else(|| trainerror::state(format!("checkpoint metadata {key} is missing")))?
        .parse::<usize>()
        .map_err(|_| trainerror::state(format!("checkpoint metadata {key} is not an integer")))
}

fn parsef32(metadata: &BTreeMap<String, String>, key: &str) -> Result<f32, trainerror> {
    metadata
        .get(key)
        .ok_or_else(|| trainerror::state(format!("checkpoint metadata {key} is missing")))?
        .parse::<f32>()
        .map_err(|_| trainerror::state(format!("checkpoint metadata {key} is not a number")))
}
