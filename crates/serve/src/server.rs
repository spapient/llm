use std::io::{BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use api::compute;
use models::causallm;

use crate::error::serveerror;
use crate::request::httprequest;
use crate::response::httpresponse;
use crate::routes::{inferenceservice, serviceconfig};

/// A blocking, standard-library-only HTTP/1.1 server.
///
/// The server accepts connections on a [`TcpListener`], parses one bounded
/// request per connection, dispatches it through the transport-neutral
/// [`inferenceservice`], and writes one response. It owns no request state: all
/// generation state lives in the per-request context created by the service.
///
/// Every response is written with `connection: close`, so the server keeps no
/// keep-alive or session state between requests.
///
/// # Cancellation
///
/// When the service configuration carries a deadline, dispatch runs on a scoped
/// worker and the connection thread waits with a timeout. If the deadline
/// elapses first, the connection thread cancels the request through the handle
/// the service exposed and returns HTTP 504. The worker observes the
/// cancellation at its next engine checkpoint, so it always terminates.
pub struct httpserver<'a, M: causallm + Sync, C: compute + Sync> {
    listener: TcpListener,
    service: inferenceservice<'a, M, C>,
    readtimeout: Option<Duration>,
}

impl<'a, M: causallm + Sync, C: compute + Sync> httpserver<'a, M, C> {
    /// Binds `addr` and builds a server with no read timeout.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::io`] when the address cannot be bound.
    pub fn bind(
        addr: &str,
        model: &'a M,
        provider: &'a C,
        config: serviceconfig,
    ) -> Result<Self, serveerror> {
        let listener = TcpListener::bind(addr).map_err(|error| serveerror::fromio(&error))?;
        Ok(Self {
            listener,
            service: inferenceservice::new(model, provider, config),
            readtimeout: None,
        })
    }

    /// Builds a server over an already-bound listener.
    #[must_use]
    pub fn fromlistener(
        listener: TcpListener,
        model: &'a M,
        provider: &'a C,
        config: serviceconfig,
        readtimeout: Option<Duration>,
    ) -> Self {
        Self {
            listener,
            service: inferenceservice::new(model, provider, config),
            readtimeout,
        }
    }

    /// Returns the address the server is bound to.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::io`] when the local address cannot be read.
    pub fn getlocaladdr(&self) -> Result<SocketAddr, serveerror> {
        self.listener
            .local_addr()
            .map_err(|error| serveerror::fromio(&error))
    }

    /// Returns the service the server dispatches to.
    #[must_use]
    pub fn getservice(&self) -> &inferenceservice<'a, M, C> {
        &self.service
    }

    /// Accepts connections and serves them until an accept error occurs.
    ///
    /// Each connection is handled on its own scoped thread; a per-connection
    /// failure does not stop the accept loop.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::io`] when accepting a connection fails.
    pub fn serve(&self) -> Result<(), serveerror> {
        initdiagnostics();
        std::thread::scope(|scope| -> Result<(), serveerror> {
            loop {
                let (stream, _peer) = self
                    .listener
                    .accept()
                    .map_err(|error| serveerror::fromio(&error))?;
                let service = &self.service;
                let readtimeout = self.readtimeout;
                scope.spawn(move || {
                    // A connection failure is reported by dropping the
                    // connection; it must not stop the accept loop.
                    let _ = handleconnection(service, stream, readtimeout);
                });
            }
        })
    }

    /// Accepts and serves exactly one connection.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::io`] when accepting the connection or writing the
    /// response fails.
    pub fn serveone(&self) -> Result<(), serveerror> {
        initdiagnostics();
        let (stream, _peer) = self
            .listener
            .accept()
            .map_err(|error| serveerror::fromio(&error))?;
        handleconnection(&self.service, stream, self.readtimeout)
    }
}

/// Installs the process-wide diagnostics subscriber if none is present.
///
/// The filter is read from the `RUST_LOG` environment variable and defaults to
/// `warn`, so the `debug`/`trace` events emitted on hot paths stay discarded
/// unless an operator opts in (`PF2`). A pre-existing global subscriber is kept
/// and the install is skipped explicitly rather than treated as an error.
fn initdiagnostics() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let subscriber = tracing_subscriber::fmt().with_env_filter(filter).finish();
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(()) => {}
        Err(_) => {
            // Diagnostics were already enabled by another component; keep them.
        }
    }
}

/// Parses, dispatches, and answers one connection.
fn handleconnection<M: causallm + Sync, C: compute + Sync>(
    service: &inferenceservice<'_, M, C>,
    stream: TcpStream,
    readtimeout: Option<Duration>,
) -> Result<(), serveerror> {
    if let Some(timeout) = readtimeout {
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|error| serveerror::fromio(&error))?;
    }
    let limits = service.getconfig().gettransport();
    let readerhalf = stream
        .try_clone()
        .map_err(|error| serveerror::fromio(&error))?;
    let mut reader = BufReader::new(readerhalf.take(limits.getmaxrequestbytes() as u64));
    let response = match httprequest::parse(&mut reader, limits) {
        Ok(request) => dispatch(service, &request),
        Err(error) => httpresponse::fromerror(&error),
    };
    let mut writer = stream;
    response
        .write(&mut writer)
        .map_err(|error| serveerror::fromio(&error))?;
    writer.flush().map_err(|error| serveerror::fromio(&error))
}

/// Dispatches one parsed request, applying the deadline when configured.
fn dispatch<M: causallm + Sync, C: compute + Sync>(
    service: &inferenceservice<'_, M, C>,
    request: &httprequest,
) -> httpresponse {
    match service.getconfig().getdeadline() {
        None => service.handle(request),
        Some(deadline) => dispatchwithdeadline(service, request, deadline),
    }
}

/// Runs the request on a scoped worker, mapping a deadline to cancellation.
fn dispatchwithdeadline<M: causallm + Sync, C: compute + Sync>(
    service: &inferenceservice<'_, M, C>,
    request: &httprequest,
    deadline: Duration,
) -> httpresponse {
    let (done_tx, done_rx) = mpsc::channel();
    let (cancel_tx, cancel_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let response = service.handlewith(request, move |cancel| {
                let _ = cancel_tx.send(cancel.clone());
            });
            let _ = done_tx.send(response);
        });
        match done_rx.recv_timeout(deadline) {
            Ok(response) => response,
            Err(RecvTimeoutError::Timeout) => {
                if let Ok(cancel) = cancel_rx.try_recv() {
                    cancel.cancel();
                }
                timeoutresponse()
            }
            Err(RecvTimeoutError::Disconnected) => {
                httpresponse::fromerror(&serveerror::dispatchfailed)
            }
        }
    })
}

/// Builds the response returned when the request deadline elapses.
fn timeoutresponse() -> httpresponse {
    httpresponse::new(
        504,
        "Gateway Timeout",
        "text/plain; charset=utf-8",
        b"request deadline exceeded".to_vec(),
    )
}
