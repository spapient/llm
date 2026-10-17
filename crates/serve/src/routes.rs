use core::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use api::compute;
use context::{cancellation, executionconfig, executioncontext};
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, samplingparams,
    stopcondition, stopreason,
};
use lib::{requestid, seed};
use models::causallm;

use crate::admit::admissions;
use crate::error::serveerror;
use crate::request::{httpmethod, httprequest, parseform, transportlimits};
use crate::response::httpresponse;

/// Immutable service configuration shared by every request.
///
/// The configuration carries the transport bounds, the per-request execution
/// controls, the admission controller, and an optional request deadline. It
/// holds no request data, so one value can be shared by concurrent requests.
pub struct serviceconfig {
    transport: transportlimits,
    execution: executionconfig,
    admission: admissions,
    deadline: Option<Duration>,
}

impl serviceconfig {
    /// Builds a service configuration.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::invalidconfig`] when `maxconcurrent` is zero.
    pub fn new(
        transport: transportlimits,
        execution: executionconfig,
        maxconcurrent: usize,
        deadline: Option<Duration>,
    ) -> Result<Self, serveerror> {
        Ok(Self {
            transport,
            execution,
            admission: admissions::new(maxconcurrent)?,
            deadline,
        })
    }

    /// Returns the transport bounds.
    #[must_use]
    pub fn gettransport(&self) -> &transportlimits {
        &self.transport
    }

    /// Returns the per-request execution controls.
    #[must_use]
    pub fn getexecution(&self) -> &executionconfig {
        &self.execution
    }

    /// Returns the admission controller.
    #[must_use]
    pub fn getadmission(&self) -> &admissions {
        &self.admission
    }

    /// Returns the request deadline the server enforces, if any.
    #[must_use]
    pub const fn getdeadline(&self) -> Option<Duration> {
        self.deadline
    }
}

/// The transport-neutral inference service.
///
/// The service borrows an immutable model and compute provider and owns only
/// immutable configuration plus an atomic request counter. Request state lives
/// in the per-request [`executioncontext`], [`generationrequest`], and sink, so
/// one service value serves concurrent requests without sharing mutable
/// generation state (OW2).
pub struct inferenceservice<'a, M: causallm, C: compute> {
    model: &'a M,
    provider: &'a C,
    config: serviceconfig,
    nextid: AtomicU64,
}

impl<'a, M: causallm, C: compute> inferenceservice<'a, M, C> {
    /// Builds a service over a borrowed model and compute provider.
    #[must_use]
    pub fn new(model: &'a M, provider: &'a C, config: serviceconfig) -> Self {
        Self {
            model,
            provider,
            config,
            nextid: AtomicU64::new(1),
        }
    }

    /// Returns the service configuration.
    #[must_use]
    pub fn getconfig(&self) -> &serviceconfig {
        &self.config
    }

    /// Handles one request without a transport cancellation hook.
    #[must_use]
    pub fn handle(&self, request: &httprequest) -> httpresponse {
        self.handlewith(request, |_| {})
    }

    /// Handles one request, exposing the request's cancellation handle.
    ///
    /// `onstart` is called after admission succeeds and immediately before the
    /// model runs, so a transport can map a deadline or disconnect to
    /// [`cancellation::cancel`]. If cancellation is requested before the first
    /// checkpoint, the engine returns an
    /// [`infererror::context`](infer::infererror::context) failure and no model
    /// work is observed.
    #[must_use]
    pub fn handlewith<F: FnOnce(&cancellation)>(
        &self,
        request: &httprequest,
        onstart: F,
    ) -> httpresponse {
        let outcome = match (request.getmethod(), request.getpath()) {
            (httpmethod::get, "/health") => Ok(httpresponse::ok(b"ok".to_vec())),
            (httpmethod::post, "/generate") => self.generate(request, onstart),
            (method, "/health" | "/generate") => Err(serveerror::unsupportedmethod {
                method: method.asstr().to_string(),
            }),
            _ => Err(serveerror::notfound {
                path: request.getpath().to_string(),
            }),
        };
        match outcome {
            Ok(response) => response,
            Err(error) => httpresponse::fromerror(&error),
        }
    }

    /// Validates, admits, and runs one generation request.
    fn generate<F: FnOnce(&cancellation)>(
        &self,
        request: &httprequest,
        onstart: F,
    ) -> Result<httpresponse, serveerror> {
        let limits = self.config.gettransport();
        let fields = extractfields(request, limits)?;
        let input = inputfromrequest(&fields, limits)?;
        // Admission is taken before any model work and held for its duration.
        let _permit = self.config.getadmission().tryadmit()?;
        let stop = stopcondition::new(input.eos, input.maxnewtokens)?;
        let params = generationparams::formodel(self.model.getconfig(), stop, input.sampling);
        let mut genreq = generationrequest::new(input.prompt, params)?;
        let id = requestid::new(self.nextid.fetch_add(1, Ordering::Relaxed));
        let exec = executioncontext::new(id, *self.config.getexecution());
        onstart(exec.getcancellation());

        let engine = generationengine::new(self.model, self.provider);
        let mut sink = collectingsink::new();
        let result = engine.generate(&exec, &mut genreq, &mut sink)?;

        Ok(formatresponse(&result))
    }
}

/// The validated, typed generation input derived from an HTTP request.
struct generationinput {
    prompt: Vec<usize>,
    maxnewtokens: usize,
    eos: Vec<usize>,
    sampling: samplingparams,
}

/// Extracts the raw form fields, validating media type and body size first.
fn extractfields(
    request: &httprequest,
    limits: &transportlimits,
) -> Result<Vec<(String, String)>, serveerror> {
    let media = request.getheader("content-type").unwrap_or("");
    if !request.getbody().is_empty() && !mediasupported(media) {
        return Err(serveerror::unsupportedmediatype {
            media: media.to_string(),
        });
    }
    let body = request.getbody();
    if body.len() > limits.getmaxbodybytes() {
        return Err(serveerror::bodytoolarge {
            len: body.len(),
            limit: limits.getmaxbodybytes(),
        });
    }
    parseform(body, limits)
}

/// Builds the typed generation input from bounded form fields.
fn inputfromrequest(
    fields: &[(String, String)],
    limits: &transportlimits,
) -> Result<generationinput, serveerror> {
    let prompt = getfield(fields, "prompt").ok_or(serveerror::missingfield { field: "prompt" })?;
    let rawmaxnew = getfield(fields, "maxnewtokens").ok_or(serveerror::missingfield {
        field: "maxnewtokens",
    })?;
    let prompt = parsetokens(prompt, "prompt", limits.getmaxprompttokens())?;
    let maxnewtokens = parsenumber(rawmaxnew, "maxnewtokens")?;
    if maxnewtokens == 0 {
        return Err(serveerror::invalidfield {
            field: "maxnewtokens",
            detail: "must be non-zero".to_string(),
        });
    }
    if maxnewtokens > limits.getmaxnewtokens() {
        return Err(serveerror::invalidfield {
            field: "maxnewtokens",
            detail: "exceeds the configured bound".to_string(),
        });
    }
    let eos = match getfield(fields, "eos") {
        Some(value) => parsetokens(value, "eos", limits.getmaxprompttokens())?,
        None => Vec::new(),
    };
    let sampling = parsesampling(fields)?;
    Ok(generationinput {
        prompt,
        maxnewtokens,
        eos,
        sampling,
    })
}

/// Parses an optional sampling parameter set, defaulting to greedy decoding.
fn parsesampling(fields: &[(String, String)]) -> Result<samplingparams, serveerror> {
    let temperature = match getfield(fields, "temperature") {
        Some(value) => Some(parsefloat(value, "temperature")?),
        None => None,
    };
    let topk = match getfield(fields, "topk") {
        Some(value) => Some(parsenumber(value, "topk")?),
        None => None,
    };
    let topp = match getfield(fields, "topp") {
        Some(value) => Some(parsefloat(value, "topp")?),
        None => None,
    };
    let randomseed = match getfield(fields, "seed") {
        Some(value) => Some(parseu64(value, "seed")?),
        None => None,
    };
    if temperature.is_none() && topk.is_none() && topp.is_none() && randomseed.is_none() {
        return Ok(samplingparams::greedy());
    }
    Ok(samplingparams::new(
        temperature.unwrap_or(0.0),
        topk,
        topp,
        seed::new(randomseed.unwrap_or(0)),
    )?)
}

/// Parses a comma-separated token list bounded by `limit`.
fn parsetokens(value: &str, field: &'static str, limit: usize) -> Result<Vec<usize>, serveerror> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    let count = value.split(',').count();
    if count > limit {
        return Err(serveerror::invalidfield {
            field,
            detail: "too many tokens".to_string(),
        });
    }
    let mut tokens = Vec::with_capacity(count);
    for item in value.split(',') {
        tokens.push(parsenumber(item, field)?);
    }
    Ok(tokens)
}

/// Parses an unsigned count, rejecting overflow and non-numeric input.
fn parsenumber(value: &str, field: &'static str) -> Result<usize, serveerror> {
    value
        .parse::<usize>()
        .map_err(|_| serveerror::invalidfield {
            field,
            detail: "must be a non-negative integer".to_string(),
        })
}

/// Parses an unsigned 64-bit seed.
fn parseu64(value: &str, field: &'static str) -> Result<u64, serveerror> {
    value.parse::<u64>().map_err(|_| serveerror::invalidfield {
        field,
        detail: "must be a non-negative integer".to_string(),
    })
}

/// Parses a finite-checkable floating-point parameter.
fn parsefloat(value: &str, field: &'static str) -> Result<f32, serveerror> {
    value.parse::<f32>().map_err(|_| serveerror::invalidfield {
        field,
        detail: "must be a number".to_string(),
    })
}

/// Reports whether `media` is the supported form media type, ignoring params.
fn mediasupported(media: &str) -> bool {
    media.split(';').next().is_some_and(|base| {
        base.trim()
            .eq_ignore_ascii_case("application/x-www-form-urlencoded")
    })
}

/// Returns the first field named `name`.
fn getfield<'a>(fields: &'a [(String, String)], name: &str) -> Option<&'a str> {
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

/// Formats a completed generation as a plain-text response body.
fn formatresponse(result: &infer::generationresult) -> httpresponse {
    let tokens = result
        .gettokens()
        .iter()
        .map(|token| token.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        "tokens={tokens}\nreason={}\nsteps={}\n",
        reasonstr(result.getreason()),
        result.getsteps()
    );
    httpresponse::ok(body.into_bytes())
}

/// Returns the wire spelling of a stop reason.
fn reasonstr(reason: stopreason) -> &'static str {
    match reason {
        stopreason::endoftoken => "endoftoken",
        stopreason::maxlength => "maxlength",
        stopreason::stopped => "stopped",
    }
}
