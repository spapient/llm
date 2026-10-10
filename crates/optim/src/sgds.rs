use ::tensor::tensor;

use crate::error::opterror;
use crate::gradientclip::{clipconfig, clipgradients};
use crate::optimizers::{
    checkcount, checklr, checknonnegative, checkshapes, checkslots, collectslots, optimizer,
    setslot, slotindex, zerolike,
};
use crate::state::{optimizerstate, statetensor};

/// Hyperparameters for [`sgd`].
///
/// The fields are fixed at construction, except the learning rate, which
/// [`sgd::setlr`] may change between steps so a schedule can drive it. The
/// default clipping configuration is [`clipconfig::none`]; use [`Self::withclip`]
/// to attach a validated one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct sgdconfig {
    lr: f32,
    momentum: f32,
    weightdecay: f32,
    clip: clipconfig,
}

impl sgdconfig {
    /// Creates a stochastic-gradient-descent configuration.
    ///
    /// `lr` and `weightdecay` must be finite and non-negative; `momentum` must
    /// be finite and in `[0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] for any value outside the
    /// domains above.
    pub fn new(lr: f32, momentum: f32, weightdecay: f32) -> Result<Self, opterror> {
        checklr(lr)?;
        if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
            return Err(opterror::hyperparameter(
                "momentum",
                "must be finite and in [0, 1)",
            ));
        }
        checknonnegative("weightdecay", weightdecay)?;
        Ok(Self {
            lr,
            momentum,
            weightdecay,
            clip: clipconfig::none(),
        })
    }

    /// Returns a copy of this configuration with the given clipping method.
    #[must_use]
    pub fn withclip(mut self, clip: clipconfig) -> Self {
        self.clip = clip;
        self
    }

    /// Returns the initial learning rate.
    #[must_use]
    pub const fn getlr(&self) -> f32 {
        self.lr
    }

    /// Returns the momentum coefficient.
    #[must_use]
    pub const fn getmomentum(&self) -> f32 {
        self.momentum
    }

    /// Returns the weight-decay coefficient.
    #[must_use]
    pub const fn getweightdecay(&self) -> f32 {
        self.weightdecay
    }

    /// Returns the gradient-clipping configuration.
    #[must_use]
    pub const fn getclip(&self) -> clipconfig {
        self.clip
    }
}

/// The stochastic-gradient-descent optimizer with momentum.
///
/// # Update rule
///
/// For a parameter `p` with gradient `g`:
///
/// ```text
/// g_tilde = g + weightdecay * p
/// b = momentum * b_prev + g_tilde
/// p = p - lr * b
/// ```
///
/// The velocity buffer `b` starts at zero, so the first update uses
/// `b = g_tilde`. Weight decay is coupled (added to the gradient before the
/// momentum accumulation), matching the classical L2 penalty. With
/// `momentum = 0` the rule reduces to plain SGD, `p = p - lr * (g + weightdecay * p)`.
///
/// Nesterov momentum and dampening are deliberately unsupported in this stage.
#[derive(Clone, Debug)]
pub struct sgd {
    config: sgdconfig,
    step: u64,
    velocity: Vec<tensor<f32>>,
}

impl sgd {
    /// Creates an optimizer from a validated configuration.
    #[must_use]
    pub fn new(config: sgdconfig) -> Self {
        Self {
            config,
            step: 0,
            velocity: Vec::new(),
        }
    }

    /// Returns the configuration in effect.
    #[must_use]
    pub const fn getconfig(&self) -> sgdconfig {
        self.config
    }

    /// Returns the number of parameter slots the optimizer currently tracks.
    ///
    /// This is zero until the first [`optimizer::update`] initializes the
    /// velocity buffers.
    #[must_use]
    pub fn getparamcount(&self) -> usize {
        self.velocity.len()
    }
}

impl optimizer for sgd {
    fn getname(&self) -> &'static str {
        "sgd"
    }

    fn getlr(&self) -> f32 {
        self.config.lr
    }

    fn setlr(&mut self, lr: f32) -> Result<(), opterror> {
        checklr(lr)?;
        self.config.lr = lr;
        Ok(())
    }

    fn getstep(&self) -> u64 {
        self.step
    }

    fn update(
        &mut self,
        params: &mut [tensor<f32>],
        grads: &mut [tensor<f32>],
    ) -> Result<(), opterror> {
        checkcount(params, grads)?;
        if params.is_empty() {
            return Ok(());
        }
        checkshapes(params, grads)?;
        if self.velocity.is_empty() {
            self.velocity = params.iter().map(zerolike).collect::<Result<_, _>>()?;
        } else {
            checkslots(&self.velocity, params, "velocity")?;
        }
        let clip = self.config.clip;
        if clip.isenabled() {
            clipgradients(grads, clip);
        }
        self.step += 1;
        let lr = self.config.lr;
        let momentum = self.config.momentum;
        let weightdecay = self.config.weightdecay;
        for ((param, grad), velocity) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.velocity.iter_mut())
        {
            for ((value, &g), b) in param
                .getdatamut()
                .iter_mut()
                .zip(grad.getdata().iter())
                .zip(velocity.getdatamut().iter_mut())
            {
                let decayed = g + weightdecay * *value;
                let updated = momentum * *b + decayed;
                *b = updated;
                *value -= lr * updated;
            }
        }
        Ok(())
    }

    fn getstate(&self) -> optimizerstate {
        let params = self.velocity.len();
        let mut tensors = Vec::with_capacity(params);
        for (index, buffer) in self.velocity.iter().enumerate() {
            tensors.push(statetensor::fromtensor(format!("momentum.{index}"), buffer));
        }
        optimizerstate::new(self.step, self.config.lr, params, tensors)
    }

    fn restore(&mut self, state: &optimizerstate) -> Result<(), opterror> {
        let params = state.getparamcount();
        let tensors = state.gettensors();
        if tensors.len() != params {
            return Err(opterror::state(format!(
                "expected {params} velocity tensors but found {}",
                tensors.len()
            )));
        }
        let mut velocity: Vec<Option<tensor<f32>>> = vec![None; params];
        for entry in tensors {
            let name = entry.getname();
            if let Some(index) = slotindex(name, "momentum.") {
                setslot(&mut velocity, index, entry, "velocity")?;
            } else {
                return Err(opterror::state(format!(
                    "unexpected state tensor name {name:?}"
                )));
            }
        }
        let velocity = collectslots(velocity, "velocity")?;
        checklr(state.getlr())?;
        self.step = state.getstep();
        self.config.lr = state.getlr();
        self.velocity = velocity;
        Ok(())
    }
}
