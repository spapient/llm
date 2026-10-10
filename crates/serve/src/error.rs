use infer::infererror;
use thiserror::Error;

/// Failures produced by the HTTP service.
///
/// The variants stay separated by responsibility:
///
/// * transport/protocol failures — a malformed or unsupported request that the
///   server rejects before any application work;
/// * routing failures — a target that does not match a route;
/// * application field validation — a form field that is missing or invalid;
/// * admission control — the concurrency ceiling is reached;
/// * model execution — carried transparently as [`infererror`] so no
///   distinction is lost (ER2) and no transport variant can absorb it.
///
/// Only the I/O error *kind* is retained, so an error response never exposes an
/// internal filesystem or socket detail (error-handling section 10.3).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum serveerror {
    /// The request line was missing, empty, or not exactly three parts.
    #[error("malformed request line")]
    malformedrequestline,

    /// A header line had no colon or an empty name.
    #[error("malformed header line")]
    malformedheader,

    /// The HTTP version was not `HTTP/1.0` or `HTTP/1.1`.
    #[error("unsupported HTTP version")]
    unsupportedversion,

    /// A request or header line exceeded the configured byte bound.
    #[error("line too long: {len} bytes exceeds {limit}")]
    linetoolong {
        /// The observed line length in bytes.
        len: usize,
        /// The configured bound.
        limit: usize,
    },

    /// The request carried more headers than the configured bound.
    #[error("too many headers: limit {limit}")]
    toomanyheaders {
        /// The configured header count bound.
        limit: usize,
    },

    /// The response ended before the blank line that terminates the headers.
    #[error("incomplete request headers")]
    incompleteheaders,

    /// The body was shorter than its declared `content-length`.
    #[error("incomplete request body: expected {expected} bytes, received {received}")]
    incompletebody {
        /// The declared content length.
        expected: usize,
        /// The number of body bytes actually read.
        received: usize,
    },

    /// A `content-length` header was present but not a valid count.
    #[error("invalid content-length header")]
    invalidcontentlength,

    /// The declared or observed body exceeded the configured bound.
    #[error("request body too large: {len} bytes exceeds {limit}")]
    bodytoolarge {
        /// The declared or observed body length in bytes.
        len: usize,
        /// The configured body bound.
        limit: usize,
    },

    /// The request content type was not supported for the route.
    #[error("unsupported media type: {media}")]
    unsupportedmediatype {
        /// The rejected media type, bounded by the header line bound.
        media: String,
    },

    /// The request method was not supported for the target.
    #[error("unsupported method: {method}")]
    unsupportedmethod {
        /// The rejected method.
        method: String,
    },

    /// The request target did not match a route.
    #[error("not found: {path}")]
    notfound {
        /// The request path that did not match.
        path: String,
    },

    /// A required form field was absent.
    #[error("missing required field: {field}")]
    missingfield {
        /// The name of the absent field.
        field: &'static str,
    },

    /// A form field was present but invalid.
    #[error("invalid field {field}: {detail}")]
    invalidfield {
        /// The name of the rejected field.
        field: &'static str,
        /// A description that never echoes unbounded client input.
        detail: String,
    },

    /// The form body carried more fields than the configured bound.
    #[error("too many form fields: limit {limit}")]
    toomanyfields {
        /// The configured field count bound.
        limit: usize,
    },

    /// Admission control rejected the request because the service is at its
    /// concurrency ceiling.
    #[error("too many concurrent requests: limit {limit}")]
    overloaded {
        /// The configured concurrency ceiling.
        limit: usize,
    },

    /// A configuration value is invalid.
    #[error("invalid service configuration: {detail}")]
    invalidconfig {
        /// A static explanation of the rejected configuration.
        detail: &'static str,
    },

    /// A dispatch worker ended without producing a response.
    #[error("request dispatch failed")]
    dispatchfailed,

    /// An I/O operation failed; only the error kind is retained.
    #[error("io error: {kind:?}")]
    io {
        /// The standard-library I/O error kind.
        kind: std::io::ErrorKind,
    },

    /// A model execution or generation failure, carried unchanged.
    #[error(transparent)]
    inference(#[from] infererror),
}

impl serveerror {
    /// Wraps a standard I/O error, retaining only its kind.
    #[must_use]
    pub(crate) fn fromio(error: &std::io::Error) -> Self {
        Self::io { kind: error.kind() }
    }
}
