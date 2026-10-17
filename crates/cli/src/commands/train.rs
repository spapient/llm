use lib::{jsonvalue, parse, seed};
use optim::{adamw, adamwconfig, sgd, sgdconfig};
use train::trainconfig;

use crate::ADAMWBETA1;
use crate::ADAMWBETA2;
use crate::ADAMWEPS;
use crate::DEFAULTMAXARTIFACTBYTES;
use crate::args::trainargs;
use crate::clierror;
use crate::commandoutput;
use crate::commands::readartifact;

/// The CLI-owned training-configuration document schema.
struct trainconfigfile {
    /// The number of micro-batches averaged into one optimizer step.
    accumulationsteps: usize,
    /// The maximum number of optimizer steps.
    maxsteps: u64,
    /// The reproducible run seed.
    seed: u64,
    /// The initial learning rate.
    learningrate: f32,
    /// The SGD momentum coefficient, ignored for AdamW.
    momentum: f32,
    /// The weight-decay coefficient.
    weightdecay: f32,
    /// The optimizer selection.
    optimizer: optimizerchoice,
}

impl trainconfigfile {
    /// Parses the configuration from its JSON document.
    ///
    /// # Errors
    ///
    /// Returns [`clierror::invalidconfigjson`] for malformed JSON and
    /// [`clierror::invalidvalue`] for a missing or mistyped field.
    fn parse(bytes: &[u8]) -> Result<Self, clierror> {
        let root = parse(bytes).map_err(|error| clierror::invalidconfigjson {
            line: error.getline(),
            column: error.getcolumn(),
            message: error.to_string(),
        })?;
        let object = root
            .getobject()
            .ok_or_else(|| configerror("train config", "document is not a JSON object"))?;
        let accumulationsteps = usizefield(object, "accumulationsteps")?;
        let maxsteps = u64field(object, "maxsteps")?;
        let seed = u64field(object, "seed")?;
        let learningrate = f32field(object, "learningrate")?;
        let momentum = object
            .get("momentum")
            .map(|_| f32field(object, "momentum"))
            .transpose()?
            .unwrap_or(0.0);
        let weightdecay = object
            .get("weightdecay")
            .map(|_| f32field(object, "weightdecay"))
            .transpose()?
            .unwrap_or(0.0);
        let optimizer = match object.get("optimizer") {
            Some(value) => match value.getstr() {
                Some("sgd") => optimizerchoice::sgd,
                Some("adamw") => optimizerchoice::adamw,
                _ => {
                    return Err(configerror("optimizer", "expected \"sgd\" or \"adamw\""));
                }
            },
            None => optimizerchoice::sgd,
        };
        Ok(Self {
            accumulationsteps,
            maxsteps,
            seed,
            learningrate,
            momentum,
            weightdecay,
            optimizer,
        })
    }
}

/// The supported optimizer selections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum optimizerchoice {
    /// Stochastic gradient descent with momentum.
    #[default]
    sgd,
    /// AdamW with decoupled weight decay.
    adamw,
}

/// Builds the JSON-level error for a configuration-field failure.
fn configerror(field: &str, reason: &str) -> clierror {
    clierror::invalidconfigjson {
        line: 1,
        column: 1,
        message: format!("field {field:?}: {reason}"),
    }
}

/// Extracts a required `usize` field.
fn usizefield(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &str,
) -> Result<usize, clierror> {
    object
        .get(field)
        .and_then(jsonvalue::getunsigned)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| configerror(field, "missing or not a non-negative integer"))
}

/// Extracts a required `u64` field.
fn u64field(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &str,
) -> Result<u64, clierror> {
    object
        .get(field)
        .and_then(jsonvalue::getunsigned)
        .ok_or_else(|| configerror(field, "missing or not a non-negative integer"))
}

/// Extracts a required `f32` field.
fn f32field(
    object: &std::collections::BTreeMap<String, jsonvalue>,
    field: &str,
) -> Result<f32, clierror> {
    object
        .get(field)
        .and_then(jsonvalue::getreal)
        .map(|value| value as f32)
        .ok_or_else(|| configerror(field, "missing or not a number"))
}

/// A validated optimizer selection.
struct optimizerplan {
    /// The stable optimizer name.
    name: &'static str,
    /// The effective learning rate.
    lr: f32,
}

/// Loads and validates a training configuration and delegates it to `train`.
///
/// Without `--check-only`, the command cannot complete a weight update because
/// `nn`/`models` expose no backward path, so it returns the declared
/// [`clierror::notsupported`] limitation. With `--check-only` it reports the
/// validated plan and succeeds.
///
/// # Errors
///
/// Returns a [`clierror`] for a path or read failure, a malformed
/// configuration document, an invalid `trainconfig` or optimizer
/// hyperparameter, or the declared unsupported limitation.
pub(crate) fn run(args: &trainargs) -> Result<commandoutput, clierror> {
    let bytes = readartifact(&args.config, DEFAULTMAXARTIFACTBYTES)?;
    let file = trainconfigfile::parse(&bytes)?;
    let config = trainconfig::new(
        file.accumulationsteps,
        file.maxsteps,
        None,
        seed::new(file.seed),
    )?;
    let plan = planfromfile(&file)?;
    if !args.check_only {
        return Err(clierror::notsupported {
            operation: "end-to-end llama training",
            detail: "nn/models are forward-only and autograd has no llama or cross-entropy backward path, so model-weight gradients cannot be produced",
        });
    }
    Ok(commandoutput::new(formatplan(args, &config, &plan)))
}

/// Builds and validates the selected optimizer from the configuration.
fn planfromfile(file: &trainconfigfile) -> Result<optimizerplan, clierror> {
    match file.optimizer {
        optimizerchoice::sgd => {
            let config = sgdconfig::new(file.learningrate, file.momentum, file.weightdecay)?;
            let value = sgd::new(config);
            Ok(optimizerplan {
                name: optim::optimizer::getname(&value),
                lr: optim::optimizer::getlr(&value),
            })
        }
        optimizerchoice::adamw => {
            let config = adamwconfig::new(
                file.learningrate,
                ADAMWBETA1,
                ADAMWBETA2,
                ADAMWEPS,
                file.weightdecay,
            )?;
            let value = adamw::new(config);
            Ok(optimizerplan {
                name: optim::optimizer::getname(&value),
                lr: optim::optimizer::getlr(&value),
            })
        }
    }
}

/// Renders the validated training plan.
fn formatplan(args: &trainargs, config: &trainconfig, plan: &optimizerplan) -> String {
    let mut out = format!("config: {}\n", args.config.display());
    out.push_str(&format!(
        "accumulationsteps: {}\n",
        config.getaccumulationsteps()
    ));
    out.push_str(&format!("maxsteps: {}\n", config.getmaxsteps()));
    out.push_str(&format!("seed: {}\n", config.getseed().get()));
    out.push_str(&format!("optimizer: {}\n", plan.name));
    out.push_str(&format!("learningrate: {}\n", plan.lr));
    out.push_str("schedule: none\n");
    out.push_str("mode: check-only\n");
    out.push_str("note: configuration validated and delegated to train; no weights were trained\n");
    out
}
