use ::tensor::shape;
use lib::seed;
use nn::{
    embedding, fillone, initxavier, linear, mlp, module, nnerror, parameter, rmsnorm, statedict,
};

use crate::error::modelerror;
use crate::llamaconfig;

/// The immutable weights of one Llama transformer block.
///
/// The block owns its two normalization weights and its attention and
/// feed-forward projections as `nn` layers. Holding the layers (rather than raw
/// tensors) means the model reads the same values it validated, with no copy:
/// an `nn` layer owns its `parameter` and the block borrows it.
#[derive(Clone, Debug)]
pub struct llamalayerweights {
    attn_norm: rmsnorm,
    wq: linear,
    wk: linear,
    wv: linear,
    wo: linear,
    ffn_norm: rmsnorm,
    mlp: mlp,
}

impl llamalayerweights {
    /// Builds a block's weights and validates them against `config`.
    ///
    /// # Errors
    ///
    /// Returns [`modelerror::shapemismatch`] when a projection's feature counts
    /// disagree with the configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &llamaconfig,
        attn_norm: rmsnorm,
        wq: linear,
        wk: linear,
        wv: linear,
        wo: linear,
        ffn_norm: rmsnorm,
        mlp: mlp,
    ) -> Result<Self, modelerror> {
        checknorm("attn_norm", &attn_norm, config)?;
        checknorm("ffn_norm", &ffn_norm, config)?;
        checkprojection("wq", &wq, config.getdim(), config.getdim())?;
        checkprojection("wk", &wk, config.getdim(), config.getkvwidth())?;
        checkprojection("wv", &wv, config.getdim(), config.getkvwidth())?;
        checkprojection("wo", &wo, config.getdim(), config.getdim())?;
        if mlp.getfeatures() != config.getdim() || mlp.gethidden() != config.gethidden() {
            return Err(modelerror::shapemismatch {
                context: "mlp",
                expected: vec![config.getdim(), config.gethidden()],
                actual: vec![mlp.getfeatures(), mlp.gethidden()],
            });
        }
        Ok(Self {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            mlp,
        })
    }

    /// Returns the pre-attention normalization.
    #[must_use]
    pub fn getattnnorm(&self) -> &rmsnorm {
        &self.attn_norm
    }

    /// Returns the query projection.
    #[must_use]
    pub fn getwq(&self) -> &linear {
        &self.wq
    }

    /// Returns the key projection.
    #[must_use]
    pub fn getwk(&self) -> &linear {
        &self.wk
    }

    /// Returns the value projection.
    #[must_use]
    pub fn getwv(&self) -> &linear {
        &self.wv
    }

    /// Returns the attention output projection.
    #[must_use]
    pub fn getwo(&self) -> &linear {
        &self.wo
    }

    /// Returns the pre-feed-forward normalization.
    #[must_use]
    pub fn getffnnorm(&self) -> &rmsnorm {
        &self.ffn_norm
    }

    /// Returns the SwiGLU feed-forward network.
    #[must_use]
    pub fn getmlp(&self) -> &mlp {
        &self.mlp
    }
}

impl module for llamalayerweights {
    fn getname(&self) -> &'static str {
        "llamalayer"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        self.attn_norm
            .collect(&nn::joinname(prefix, "attn_norm"), dict)?;
        self.wq.collect(&nn::joinname(prefix, "wq"), dict)?;
        self.wk.collect(&nn::joinname(prefix, "wk"), dict)?;
        self.wv.collect(&nn::joinname(prefix, "wv"), dict)?;
        self.wo.collect(&nn::joinname(prefix, "wo"), dict)?;
        self.ffn_norm
            .collect(&nn::joinname(prefix, "ffn_norm"), dict)?;
        self.mlp.collect(&nn::joinname(prefix, "ffn"), dict)
    }
}

/// The immutable weights of a Llama causal language model.
///
/// The weights own the token embedding, one [`llamalayerweights`] per block, the
/// final normalization, and the output projection. The model borrows this value;
/// nothing here holds request-scoped state.
#[derive(Clone, Debug)]
pub struct llamaweights {
    embedding: embedding,
    layers: Vec<llamalayerweights>,
    norm: rmsnorm,
    output: linear,
}

impl llamaweights {
    /// Builds model weights and validates them against `config`.
    ///
    /// # Errors
    ///
    /// Returns [`modelerror::shapemismatch`] when the embedding, layer count,
    /// final normalization, or output projection disagrees with the
    /// configuration.
    pub fn new(
        config: &llamaconfig,
        embedding: embedding,
        layers: Vec<llamalayerweights>,
        norm: rmsnorm,
        output: linear,
    ) -> Result<Self, modelerror> {
        if embedding.getvocab() != config.getvocab() || embedding.getwidth() != config.getdim() {
            return Err(modelerror::shapemismatch {
                context: "embedding",
                expected: vec![config.getvocab(), config.getdim()],
                actual: vec![embedding.getvocab(), embedding.getwidth()],
            });
        }
        if layers.len() != config.getnlayers() {
            return Err(modelerror::config(
                "layers",
                "layer count must equal nlayers",
            ));
        }
        checknorm("norm", &norm, config)?;
        checkprojection("output", &output, config.getdim(), config.getvocab())?;
        Ok(Self {
            embedding,
            layers,
            norm,
            output,
        })
    }

    /// Returns the token embedding.
    #[must_use]
    pub fn getembedding(&self) -> &embedding {
        &self.embedding
    }

    /// Returns the transformer blocks in order.
    #[must_use]
    pub fn getlayers(&self) -> &[llamalayerweights] {
        &self.layers
    }

    /// Returns the final normalization.
    #[must_use]
    pub fn getnorm(&self) -> &rmsnorm {
        &self.norm
    }

    /// Returns the output projection.
    #[must_use]
    pub fn getoutput(&self) -> &linear {
        &self.output
    }

    /// Builds deterministic random weights for `config` from `s`.
    ///
    /// Every weight is drawn from a per-parameter substream derived from `s`, so
    /// the same seed and configuration produce identical weights. Normalization
    /// weights are initialized to one and projections are Xavier-uniform, with
    /// no bias.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when a shape is invalid or initialization fails.
    pub fn random(config: &llamaconfig, s: seed) -> Result<Self, modelerror> {
        let mut index = 0_u64;
        let mut next = move || {
            let derived = subseed(s, index);
            index = index.wrapping_add(1);
            derived
        };
        let dim = config.getdim();
        let embedding = embedding::new(parameter::new(
            "weight",
            initxavier(shape::new(vec![config.getvocab(), dim])?, next())?,
        ))?;
        let mut layers = Vec::with_capacity(config.getnlayers());
        for _ in 0..config.getnlayers() {
            let attn_norm = rmsnorm::new(
                parameter::new("weight", fillone(shape::new(vec![dim])?)?),
                config.getrmseps(),
            )?;
            let wq = randomprojection(dim, dim, next())?;
            let wv = randomprojection(dim, config.getkvwidth(), next())?;
            let wk = randomprojection(dim, config.getkvwidth(), next())?;
            let wo = randomprojection(dim, dim, next())?;
            let ffn_norm = rmsnorm::new(
                parameter::new("weight", fillone(shape::new(vec![dim])?)?),
                config.getrmseps(),
            )?;
            let hidden = config.gethidden();
            let gate = randomprojection(dim, hidden, next())?;
            let up = randomprojection(dim, hidden, next())?;
            let down = randomprojection(hidden, dim, next())?;
            layers.push(llamalayerweights::new(
                config,
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                mlp::new(gate, up, down)?,
            )?);
        }
        let norm = rmsnorm::new(
            parameter::new("weight", fillone(shape::new(vec![dim])?)?),
            config.getrmseps(),
        )?;
        let output = randomprojection(dim, config.getvocab(), next())?;
        Self::new(config, embedding, layers, norm, output)
    }
}

impl module for llamaweights {
    fn getname(&self) -> &'static str {
        "llamaweights"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        self.embedding
            .collect(&nn::joinname(prefix, "embedding"), dict)?;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.collect(&nn::joinname(prefix, &format!("layers.{i}")), dict)?;
        }
        self.norm.collect(&nn::joinname(prefix, "norm"), dict)?;
        self.output.collect(&nn::joinname(prefix, "output"), dict)
    }
}

/// Derives a per-parameter substream seed from a base seed and an index.
fn subseed(base: seed, index: u64) -> seed {
    seed::new(
        base.get()
            .wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
}

/// Builds a bias-free linear projection with Xavier-uniform weights.
fn randomprojection(
    in_features: usize,
    out_features: usize,
    s: seed,
) -> Result<linear, modelerror> {
    let weight = parameter::new(
        "weight",
        initxavier(shape::new(vec![in_features, out_features])?, s)?,
    );
    Ok(linear::new(weight, None)?)
}

/// Validates a normalization against the model dimension.
fn checknorm(
    context: &'static str,
    norm: &rmsnorm,
    config: &llamaconfig,
) -> Result<(), modelerror> {
    if norm.getfeatures() != config.getdim() {
        return Err(modelerror::shapemismatch {
            context,
            expected: vec![config.getdim()],
            actual: vec![norm.getfeatures()],
        });
    }
    Ok(())
}

/// Validates a projection against an expected input/output feature pair.
fn checkprojection(
    context: &'static str,
    projection: &linear,
    in_features: usize,
    out_features: usize,
) -> Result<(), modelerror> {
    if projection.getinfeatures() != in_features || projection.getoutfeatures() != out_features {
        return Err(modelerror::shapemismatch {
            context,
            expected: vec![in_features, out_features],
            actual: vec![projection.getinfeatures(), projection.getoutfeatures()],
        });
    }
    Ok(())
}
