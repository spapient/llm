use ::tensor::tensor;

use crate::error::opterror;
use crate::gradientclip::{clipconfig, clipgradients};
use crate::optimizers::{
    checkcount, checklr, checknonnegative, checkshapes, checkslots, collectslots, optimizer,
    setslot, slotindex, zerolike,
};
use crate::state::{optimizerstate, statetensor};

/// Hyperparameters for [`adamw`].
///
/// The fields are fixed at construction, except the learning rate, which
/// [`adamw::setlr`] may change between steps so a schedule can drive it. The
/// default clipping configuration is [`clipconfig::none`]; use [`Self::withclip`]
/// to attach a validated one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct adamwconfig {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weightdecay: f32,
    clip: clipconfig,
}

impl adamwconfig {
    /// Creates an AdamW configuration.
    ///
    /// `lr` and `weightdecay` must be finite and non-negative; `beta1` and
    /// `beta2` must be finite and in `[0, 1)`; `eps` must be finite and
    /// positive.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] for any value outside the
    /// domains above.
    pub fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weightdecay: f32,
    ) -> Result<Self, opterror> {
        checklr(lr)?;
        checkunitinterval("beta1", beta1)?;
        checkunitinterval("beta2", beta2)?;
        if !eps.is_finite() || eps <= 0.0 {
            return Err(opterror::hyperparameter(
                "eps",
                "must be finite and positive",
            ));
        }
        checknonnegative("weightdecay", weightdecay)?;
        Ok(Self {
            lr,
            beta1,
            beta2,
            eps,
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

    /// Returns the first-moment decay.
    #[must_use]
    pub const fn getbeta1(&self) -> f32 {
        self.beta1
    }

    /// Returns the second-moment decay.
    #[must_use]
    pub const fn getbeta2(&self) -> f32 {
        self.beta2
    }

    /// Returns the denominator offset.
    #[must_use]
    pub const fn geteps(&self) -> f32 {
        self.eps
    }

    /// Returns the decoupled weight-decay coefficient.
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

/// The AdamW optimizer.
///
/// # Update rule
///
/// For a parameter `p` with gradient `g` at completed-step count `t` (starting
/// at 1 for the first update):
///
/// ```text
/// m = beta1 * m_prev + (1 - beta1) * g
/// v = beta2 * v_prev + (1 - beta2) * g * g
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// p = p - lr * ( m_hat / (sqrt(v_hat) + eps) + weightdecay * p )
/// ```
///
/// The first-moment buffer `m` and second-moment buffer `v` start at zero, so
/// the first update uses `m = (1 - beta1) * g` and `v = (1 - beta2) * g * g`.
/// Weight decay is decoupled (applied to the parameter, not the gradient), as
/// in Loshchilov & Hutter's AdamW. Bias-correction divides by a strictly
/// positive factor because `beta1, beta2 < 1` and `t >= 1`.
#[derive(Clone, Debug)]
pub struct adamw {
    config: adamwconfig,
    step: u64,
    first: Vec<tensor<f32>>,
    second: Vec<tensor<f32>>,
}

impl adamw {
    /// Creates an optimizer from a validated configuration.
    #[must_use]
    pub fn new(config: adamwconfig) -> Self {
        Self {
            config,
            step: 0,
            first: Vec::new(),
            second: Vec::new(),
        }
    }

    /// Returns the configuration in effect.
    #[must_use]
    pub const fn getconfig(&self) -> adamwconfig {
        self.config
    }

    /// Returns the number of parameter slots the optimizer currently tracks.
    ///
    /// This is zero until the first [`optimizer::update`] initializes the
    /// moment buffers.
    #[must_use]
    pub fn getparamcount(&self) -> usize {
        self.first.len()
    }
}

impl optimizer for adamw {
    fn getname(&self) -> &'static str {
        "adamw"
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
        if self.first.is_empty() {
            self.first = params.iter().map(zerolike).collect::<Result<_, _>>()?;
            self.second = params.iter().map(zerolike).collect::<Result<_, _>>()?;
        } else {
            checkslots(&self.first, params, "first-moment")?;
            checkslots(&self.second, params, "second-moment")?;
        }
        let clip = self.config.clip;
        if clip.isenabled() {
            clipgradients(grads, clip);
        }
        self.step += 1;
        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let weightdecay = self.config.weightdecay;
        let bias1 = 1.0 - f64::from(beta1).powf(self.step as f64);
        let bias2 = 1.0 - f64::from(beta2).powf(self.step as f64);
        for ((param, grad), (first, second)) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.first.iter_mut().zip(self.second.iter_mut()))
        {
            for (((value, &g), m), v) in param
                .getdatamut()
                .iter_mut()
                .zip(grad.getdata().iter())
                .zip(first.getdatamut().iter_mut())
                .zip(second.getdatamut().iter_mut())
            {
                let mnew = beta1 * *m + (1.0 - beta1) * g;
                let vnew = beta2 * *v + (1.0 - beta2) * g * g;
                let mhat = f64::from(mnew) / bias1;
                let vhat = f64::from(vnew) / bias2;
                let denominator = vhat.sqrt() + f64::from(eps);
                let update = mhat / denominator + f64::from(weightdecay) * f64::from(*value);
                *value -= lr * (update as f32);
                *m = mnew;
                *v = vnew;
            }
        }
        Ok(())
    }

    fn getstate(&self) -> optimizerstate {
        let params = self.first.len();
        let mut tensors = Vec::with_capacity(params * 2);
        for (index, (first, second)) in self.first.iter().zip(self.second.iter()).enumerate() {
            tensors.push(statetensor::fromtensor(format!("m.{index}"), first));
            tensors.push(statetensor::fromtensor(format!("v.{index}"), second));
        }
        optimizerstate::new(self.step, self.config.lr, params, tensors)
    }

    fn restore(&mut self, state: &optimizerstate) -> Result<(), opterror> {
        let params = state.getparamcount();
        let tensors = state.gettensors();
        if tensors.len() != params * 2 {
            return Err(opterror::state(format!(
                "expected {} first/second-moment tensors but found {}",
                params * 2,
                tensors.len()
            )));
        }
        let mut first: Vec<Option<tensor<f32>>> = vec![None; params];
        let mut second: Vec<Option<tensor<f32>>> = vec![None; params];
        for entry in tensors {
            let name = entry.getname();
            if let Some(index) = slotindex(name, "m.") {
                setslot(&mut first, index, entry, "first-moment")?;
            } else if let Some(index) = slotindex(name, "v.") {
                setslot(&mut second, index, entry, "second-moment")?;
            } else {
                return Err(opterror::state(format!(
                    "unexpected state tensor name {name:?}"
                )));
            }
        }
        let first = collectslots(first, "first-moment")?;
        let second = collectslots(second, "second-moment")?;
        checklr(state.getlr())?;
        self.step = state.getstep();
        self.config.lr = state.getlr();
        self.first = first;
        self.second = second;
        Ok(())
    }
}

/// Rejects a value that is not finite or not in `[0, 1)`.
fn checkunitinterval(field: &'static str, value: f32) -> Result<(), opterror> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(opterror::hyperparameter(
            field,
            "must be finite and in [0, 1)",
        ));
    }
    Ok(())
}
