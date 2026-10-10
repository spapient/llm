use std::io::Write;

use context::contexterror;
use infer::infererror;

use crate::error::serveerror;

/// An HTTP response ready to be written to the wire.
///
/// The response is a self-contained value: a status, a static reason phrase, a
/// content type, and an owned body. `fromerror` is the single place
/// where a [`serveerror`] is translated into a status, keeping the protocol
/// mapping at the application boundary. Every response is written with
/// `connection: close`, so the server never depends on keep-alive state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct httpresponse {
    status: u16,
    reason: &'static str,
    contenttype: &'static str,
    body: Vec<u8>,
}

impl httpresponse {
    /// Builds a response from its parts.
    #[must_use]
    pub const fn new(
        status: u16,
        reason: &'static str,
        contenttype: &'static str,
        body: Vec<u8>,
    ) -> Self {
        Self {
            status,
            reason,
            contenttype,
            body,
        }
    }

    /// Builds a `200 OK` plain-text response.
    #[must_use]
    pub fn ok(body: Vec<u8>) -> Self {
        Self::new(200, "OK", "text/plain; charset=utf-8", body)
    }

    /// Returns the HTTP status code.
    #[must_use]
    pub const fn getstatus(&self) -> u16 {
        self.status
    }

    /// Returns the HTTP reason phrase.
    #[must_use]
    pub const fn getreason(&self) -> &'static str {
        self.reason
    }

    /// Returns the response content type.
    #[must_use]
    pub const fn getcontenttype(&self) -> &'static str {
        self.contenttype
    }

    /// Returns the raw response body.
    #[must_use]
    pub fn getbody(&self) -> &[u8] {
        &self.body
    }

    /// Returns the body as a string when it is valid UTF-8.
    #[must_use]
    pub fn getbodytext(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }

    /// Maps `error` to a response with a status chosen by failure category.
    #[must_use]
    pub(crate) fn fromerror(error: &serveerror) -> Self {
        let (status, reason) = statusfor(error);
        Self::new(
            status,
            reason,
            "text/plain; charset=utf-8",
            error.to_string().into_bytes(),
        )
    }

    /// Writes the response in HTTP/1.1 wire format.
    ///
    /// # Errors
    ///
    /// Returns any I/O error produced while writing.
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write!(writer, "HTTP/1.1 {} {}\r\n", self.status, self.reason)?;
        write!(writer, "content-type: {}\r\n", self.contenttype)?;
        write!(writer, "content-length: {}\r\n", self.body.len())?;
        write!(writer, "connection: close\r\n\r\n")?;
        writer.write_all(&self.body)
    }
}

/// Returns the status and reason for one error category.
fn statusfor(error: &serveerror) -> (u16, &'static str) {
    match error {
        serveerror::malformedrequestline
        | serveerror::malformedheader
        | serveerror::unsupportedversion
        | serveerror::linetoolong { .. }
        | serveerror::toomanyheaders { .. }
        | serveerror::incompleteheaders
        | serveerror::incompletebody { .. }
        | serveerror::invalidcontentlength
        | serveerror::missingfield { .. }
        | serveerror::invalidfield { .. }
        | serveerror::toomanyfields { .. } => (400, "Bad Request"),
        serveerror::unsupportedmethod { .. } => (405, "Method Not Allowed"),
        serveerror::unsupportedmediatype { .. } => (415, "Unsupported Media Type"),
        serveerror::bodytoolarge { .. } => (413, "Payload Too Large"),
        serveerror::notfound { .. } => (404, "Not Found"),
        serveerror::overloaded { .. } => (429, "Too Many Requests"),
        serveerror::invalidconfig { .. } | serveerror::dispatchfailed | serveerror::io { .. } => {
            (500, "Internal Server Error")
        }
        serveerror::inference(error) => inferstatus(error),
    }
}

/// Returns the status and reason for one model execution failure.
fn inferstatus(error: &infererror) -> (u16, &'static str) {
    match error {
        infererror::invalidrequest { .. }
        | infererror::invalidparams { .. }
        | infererror::tokenoutofrange { .. } => (400, "Bad Request"),
        infererror::context(contexterror::cancelled) => (408, "Request Timeout"),
        infererror::context(_) => (503, "Service Unavailable"),
        infererror::model(_) | infererror::emptylogits | infererror::degeneratelogits => {
            (500, "Internal Server Error")
        }
    }
}
