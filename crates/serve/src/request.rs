use std::io::{BufRead, Read};

use crate::error::serveerror;

/// The HTTP methods the service recognizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum httpmethod {
    /// The `GET` method.
    get,
    /// The `POST` method.
    post,
}

impl httpmethod {
    /// Returns the canonical uppercase spelling of the method.
    #[must_use]
    pub const fn asstr(self) -> &'static str {
        match self {
            Self::get => "GET",
            Self::post => "POST",
        }
    }
}

/// Explicit transport-level limits.
///
/// Every bound is an inclusive upper bound supplied by the caller; `serve`
/// invents no default. The values are consulted while parsing an untrusted
/// request, before the request reaches the model (IN1, IN4). The server also
/// caps the total bytes it reads at [`Self::getmaxrequestbytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct transportlimits {
    maxrequestbytes: usize,
    maxheaderbytes: usize,
    maxheaders: usize,
    maxbodybytes: usize,
    maxfields: usize,
    maxfieldbytes: usize,
    maxprompttokens: usize,
    maxnewtokens: usize,
}

impl transportlimits {
    /// Builds an explicit limit set.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        maxrequestbytes: usize,
        maxheaderbytes: usize,
        maxheaders: usize,
        maxbodybytes: usize,
        maxfields: usize,
        maxfieldbytes: usize,
        maxprompttokens: usize,
        maxnewtokens: usize,
    ) -> Self {
        Self {
            maxrequestbytes,
            maxheaderbytes,
            maxheaders,
            maxbodybytes,
            maxfields,
            maxfieldbytes,
            maxprompttokens,
            maxnewtokens,
        }
    }

    /// Returns the maximum bytes the server reads for one request.
    #[must_use]
    pub const fn getmaxrequestbytes(&self) -> usize {
        self.maxrequestbytes
    }

    /// Returns the maximum length of one request or header line.
    #[must_use]
    pub const fn getmaxheaderbytes(&self) -> usize {
        self.maxheaderbytes
    }

    /// Returns the maximum number of headers.
    #[must_use]
    pub const fn getmaxheaders(&self) -> usize {
        self.maxheaders
    }

    /// Returns the maximum request body size in bytes.
    #[must_use]
    pub const fn getmaxbodybytes(&self) -> usize {
        self.maxbodybytes
    }

    /// Returns the maximum number of form fields.
    #[must_use]
    pub const fn getmaxfields(&self) -> usize {
        self.maxfields
    }

    /// Returns the maximum length of one form field value.
    #[must_use]
    pub const fn getmaxfieldbytes(&self) -> usize {
        self.maxfieldbytes
    }

    /// Returns the maximum number of prompt tokens.
    #[must_use]
    pub const fn getmaxprompttokens(&self) -> usize {
        self.maxprompttokens
    }

    /// Returns the maximum number of new tokens a request may ask for.
    #[must_use]
    pub const fn getmaxnewtokens(&self) -> usize {
        self.maxnewtokens
    }
}

/// A parsed, transport-validated HTTP request.
///
/// Construction through [`Self::parse`] enforces the line, header-count,
/// content-length, and body bounds, so a value that exists holds only
/// bounded, UTF-8-validated metadata. The body is kept as raw bytes because a
/// caller may reject non-UTF-8 input with a field error rather than a panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct httprequest {
    method: httpmethod,
    target: String,
    path: String,
    query: Vec<(String, String)>,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl httprequest {
    /// Parses one HTTP/1.x request from `reader`.
    ///
    /// Header names are lowercased. Only the request line, headers, and the
    /// `content-length`-delimited body are read; the caller is responsible for
    /// capping the total bytes read (the server wraps the stream in a
    /// [`Read::take`] limit).
    ///
    /// # Errors
    ///
    /// Returns a [`serveerror`] transport variant when the request line is
    /// malformed, the HTTP version or method is unsupported, a line or the body
    /// exceeds its bound, the headers are incomplete, or the stream reports an
    /// I/O failure.
    pub fn parse<R: BufRead>(reader: &mut R, limits: &transportlimits) -> Result<Self, serveerror> {
        let requestline = readline(reader, limits.getmaxheaderbytes())?
            .ok_or(serveerror::malformedrequestline)?;
        let (method, target) = splitrequestline(&requestline)?;
        let (path, query) = splittarget(target, limits)?;
        let headers = readheaders(reader, limits)?;
        let body = readbody(reader, &headers, limits)?;
        Ok(Self {
            method,
            target: target.to_string(),
            path,
            query,
            headers,
            body,
        })
    }

    /// Returns the request method.
    #[must_use]
    pub const fn getmethod(&self) -> httpmethod {
        self.method
    }

    /// Returns the raw request target (path and query).
    #[must_use]
    pub fn gettarget(&self) -> &str {
        &self.target
    }

    /// Returns the request path, without the query string.
    #[must_use]
    pub fn getpath(&self) -> &str {
        &self.path
    }

    /// Returns the first query parameter named `name`.
    ///
    /// Query values are not percent-decoded; see the crate documentation.
    #[must_use]
    pub fn getquery(&self, name: &str) -> Option<&str> {
        getpair(&self.query, name)
    }

    /// Returns the first header named `name`, which must be lowercase.
    #[must_use]
    pub fn getheader(&self, name: &str) -> Option<&str> {
        getpair(&self.headers, name)
    }

    /// Returns the raw request body.
    #[must_use]
    pub fn getbody(&self) -> &[u8] {
        &self.body
    }

    /// Returns the body as a string.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::invalidfield`] when the body is not valid UTF-8.
    pub fn getbodytext(&self) -> Result<&str, serveerror> {
        std::str::from_utf8(&self.body).map_err(|_| serveerror::invalidfield {
            field: "body",
            detail: "body is not valid UTF-8".to_string(),
        })
    }
}

/// Parses a `name=value&name=value` form body within the configured bounds.
///
/// Values are not percent-decoded. The field count and each field value length
/// are validated before the pairs are returned (IN1, IN2, IN4).
///
/// # Errors
///
/// Returns [`serveerror::invalidfield`] when the body is not valid UTF-8 or a
/// value exceeds [`transportlimits::getmaxfieldbytes`], and
/// [`serveerror::toomanyfields`] when the field count exceeds
/// [`transportlimits::getmaxfields`].
pub(crate) fn parseform(
    body: &[u8],
    limits: &transportlimits,
) -> Result<Vec<(String, String)>, serveerror> {
    let text = std::str::from_utf8(body).map_err(|_| serveerror::invalidfield {
        field: "body",
        detail: "body is not valid UTF-8".to_string(),
    })?;
    let fields = parsepairs(text, '&', limits.getmaxfields())?;
    for (_, value) in &fields {
        if value.len() > limits.getmaxfieldbytes() {
            return Err(serveerror::invalidfield {
                field: "body",
                detail: "field value exceeds the configured bound".to_string(),
            });
        }
    }
    Ok(fields)
}

/// Reads one line, rejecting a line longer than `maxline`.
fn readline<R: BufRead>(reader: &mut R, maxline: usize) -> Result<Option<String>, serveerror> {
    let mut line = String::new();
    let read = reader
        .read_line(&mut line)
        .map_err(|error| serveerror::fromio(&error))?;
    if read == 0 {
        return Ok(None);
    }
    if line.len() > maxline {
        return Err(serveerror::linetoolong {
            len: line.len(),
            limit: maxline,
        });
    }
    Ok(Some(line))
}

/// Splits and validates the request line into a method and a target.
fn splitrequestline(line: &str) -> Result<(httpmethod, &str), serveerror> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    let (method, target, version) = match trimmed.split(' ').collect::<Vec<_>>().as_slice() {
        [method, target, version] => (*method, *target, *version),
        _ => return Err(serveerror::malformedrequestline),
    };
    if method.is_empty() || target.is_empty() {
        return Err(serveerror::malformedrequestline);
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(serveerror::unsupportedversion);
    }
    let method = match method {
        "GET" => httpmethod::get,
        "POST" => httpmethod::post,
        other => {
            return Err(serveerror::unsupportedmethod {
                method: other.to_string(),
            });
        }
    };
    Ok((method, target))
}

/// Splits a target into its path and bounded query pairs.
fn splittarget(
    target: &str,
    limits: &transportlimits,
) -> Result<(String, Vec<(String, String)>), serveerror> {
    match target.split_once('?') {
        Some((path, query)) => Ok((
            path.to_string(),
            parsepairs(query, '&', limits.getmaxfields())?,
        )),
        None => Ok((target.to_string(), Vec::new())),
    }
}

/// Reads headers until the blank terminator line.
fn readheaders<R: BufRead>(
    reader: &mut R,
    limits: &transportlimits,
) -> Result<Vec<(String, String)>, serveerror> {
    let mut headers = Vec::new();
    loop {
        let line =
            readline(reader, limits.getmaxheaderbytes())?.ok_or(serveerror::incompleteheaders)?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            return Ok(headers);
        }
        if headers.len() >= limits.getmaxheaders() {
            return Err(serveerror::toomanyheaders {
                limit: limits.getmaxheaders(),
            });
        }
        let (name, value) = trimmed.split_once(':').ok_or(serveerror::malformedheader)?;
        let name = name.trim();
        if name.is_empty() {
            return Err(serveerror::malformedheader);
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_string()));
    }
}

/// Reads the body declared by `content-length`, within the configured bound.
fn readbody<R: BufRead>(
    reader: &mut R,
    headers: &[(String, String)],
    limits: &transportlimits,
) -> Result<Vec<u8>, serveerror> {
    let declared = match getpair(headers, "content-length") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| serveerror::invalidcontentlength)?,
        None => 0,
    };
    if declared > limits.getmaxbodybytes() {
        return Err(serveerror::bodytoolarge {
            len: declared,
            limit: limits.getmaxbodybytes(),
        });
    }
    let mut body = Vec::new();
    if declared > 0 {
        reader
            .by_ref()
            .take(declared as u64)
            .read_to_end(&mut body)
            .map_err(|error| serveerror::fromio(&error))?;
    }
    if body.len() != declared {
        return Err(serveerror::incompletebody {
            expected: declared,
            received: body.len(),
        });
    }
    Ok(body)
}

/// Parses `text` into bounded `name=value` pairs.
fn parsepairs(
    text: &str,
    separator: char,
    maxpairs: usize,
) -> Result<Vec<(String, String)>, serveerror> {
    let mut pairs = Vec::new();
    if text.is_empty() {
        return Ok(pairs);
    }
    for pair in text.split(separator) {
        if pairs.len() >= maxpairs {
            return Err(serveerror::toomanyfields { limit: maxpairs });
        }
        let (name, value) = match pair.split_once('=') {
            Some((name, value)) => (name, value),
            None => (pair, ""),
        };
        pairs.push((name.to_string(), value.to_string()));
    }
    Ok(pairs)
}

/// Returns the first value named `name` in a lowercased pair list.
fn getpair<'a>(pairs: &'a [(String, String)], name: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}
