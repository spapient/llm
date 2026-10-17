use infer::infererror;

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
#[derive(Debug, PartialEq, Eq)]
pub enum serveerror {
    /// The request line was missing, empty, or not exactly three parts.
    malformedrequestline,

    /// A header line had no colon or an empty name.
    malformedheader,

    /// The HTTP version was not `HTTP/1.0` or `HTTP/1.1`.
    unsupportedversion,

    /// A request or header line exceeded the configured byte bound.
    linetoolong {
        /// The observed line length in bytes.
        len: usize,
        /// The configured bound.
        limit: usize,
    },

    /// The request carried more headers than the configured bound.
    toomanyheaders {
        /// The configured header count bound.
        limit: usize,
    },

    /// The response ended before the blank line that terminates the headers.
    incompleteheaders,

    /// The body was shorter than its declared `content-length`.
    incompletebody {
        /// The declared content length.
        expected: usize,
        /// The number of body bytes actually read.
        received: usize,
    },

    /// A `content-length` header was present but not a valid count.
    invalidcontentlength,

    /// The declared or observed body exceeded the configured bound.
    bodytoolarge {
        /// The declared or observed body length in bytes.
        len: usize,
        /// The configured body bound.
        limit: usize,
    },

    /// The request content type was not supported for the route.
    unsupportedmediatype {
        /// The rejected media type, bounded by the header line bound.
        media: String,
    },

    /// The request method was not supported for the target.
    unsupportedmethod {
        /// The rejected method.
        method: String,
    },

    /// The request target did not match a route.
    notfound {
        /// The request path that did not match.
        path: String,
    },

    /// A required form field was absent.
    missingfield {
        /// The name of the absent field.
        field: &'static str,
    },

    /// A form field was present but invalid.
    invalidfield {
        /// The name of the rejected field.
        field: &'static str,
        /// A description that never echoes unbounded client input.
        detail: String,
    },

    /// The form body carried more fields than the configured bound.
    toomanyfields {
        /// The configured field count bound.
        limit: usize,
    },

    /// Admission control rejected the request because the service is at its
    /// concurrency ceiling.
    overloaded {
        /// The configured concurrency ceiling.
        limit: usize,
    },

    /// A configuration value is invalid.
    invalidconfig {
        /// A static explanation of the rejected configuration.
        detail: &'static str,
    },

    /// A dispatch worker ended without producing a response.
    dispatchfailed,

    /// An I/O operation failed; only the error kind is retained.
    io {
        /// The standard-library I/O error kind.
        kind: std::io::ErrorKind,
    },

    /// A model execution or generation failure, carried unchanged.
    inference(infererror),
}

impl serveerror {
    /// Wraps a standard I/O error, retaining only its kind.
    #[must_use]
    pub(crate) fn fromio(error: &std::io::Error) -> Self {
        Self::io { kind: error.kind() }
    }
}

impl std::fmt::Display for serveerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::malformedrequestline => f.write_str("malformed request line"),
            Self::malformedheader => f.write_str("malformed header line"),
            Self::unsupportedversion => f.write_str("unsupported HTTP version"),
            Self::linetoolong { len, limit } => {
                write!(f, "line too long: {len} bytes exceeds {limit}")
            }
            Self::toomanyheaders { limit } => write!(f, "too many headers: limit {limit}"),
            Self::incompleteheaders => f.write_str("incomplete request headers"),
            Self::incompletebody { expected, received } => write!(
                f,
                "incomplete request body: expected {expected} bytes, received {received}"
            ),
            Self::invalidcontentlength => f.write_str("invalid content-length header"),
            Self::bodytoolarge { len, limit } => {
                write!(f, "request body too large: {len} bytes exceeds {limit}")
            }
            Self::unsupportedmediatype { media } => write!(f, "unsupported media type: {media}"),
            Self::unsupportedmethod { method } => write!(f, "unsupported method: {method}"),
            Self::notfound { path } => write!(f, "not found: {path}"),
            Self::missingfield { field } => write!(f, "missing required field: {field}"),
            Self::invalidfield { field, detail } => write!(f, "invalid field {field}: {detail}"),
            Self::toomanyfields { limit } => write!(f, "too many form fields: limit {limit}"),
            Self::overloaded { limit } => write!(f, "too many concurrent requests: limit {limit}"),
            Self::invalidconfig { detail } => write!(f, "invalid service configuration: {detail}"),
            Self::dispatchfailed => f.write_str("request dispatch failed"),
            Self::io { kind } => write!(f, "io error: {kind:?}"),
            Self::inference(f0) => std::fmt::Display::fmt(f0, f),
        }
    }
}

impl std::error::Error for serveerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::inference(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<infererror> for serveerror {
    fn from(v: infererror) -> Self {
        Self::inference(v)
    }
}
