use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

/// A narrow metrics sink.
///
/// The interface records counters and gauges by `&'static str` name. The
/// `'static` bound keeps metric cardinality bounded: a caller cannot label a
/// metric with an arbitrary runtime string, request id, or error message, which
/// matches the observability rule that metric labels must remain bounded. The
/// trait imposes no concrete backend, no global registry, and no dependency on
/// any other workspace crate.
pub trait metrics: Send + Sync {
    /// Adds `value` to the counter `name`.
    fn recordcounter(&self, name: &'static str, value: u64);

    /// Sets the gauge `name` to `value`.
    fn recordgauge(&self, name: &'static str, value: i64);
}

/// A metrics sink that stores the latest values in memory.
///
/// Intended for tests and simple diagnostics: it owns two mutex-protected maps
/// and exposes typed getters. It is not a production telemetry backend.
#[derive(Debug, Default)]
pub struct collectingmetrics {
    counters: Mutex<BTreeMap<&'static str, u64>>,
    gauges: Mutex<BTreeMap<&'static str, i64>>,
}

impl collectingmetrics {
    /// Creates an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current value of counter `name`, if it was recorded.
    #[must_use]
    pub fn getcounter(&self, name: &'static str) -> Option<u64> {
        self.counters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .copied()
    }

    /// Returns the current value of gauge `name`, if it was recorded.
    #[must_use]
    pub fn getgauge(&self, name: &'static str) -> Option<i64> {
        self.gauges
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .copied()
    }
}

impl metrics for collectingmetrics {
    fn recordcounter(&self, name: &'static str, value: u64) {
        let mut counters = self.counters.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = counters.entry(name).or_insert(0);
        *entry = entry.saturating_add(value);
    }

    fn recordgauge(&self, name: &'static str, value: i64) {
        self.gauges
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name, value);
    }
}

/// A metrics sink that discards every record.
///
/// Useful as the default sink where instrumentation is optional.
#[derive(Clone, Copy, Debug, Default)]
pub struct noopmetrics;

impl metrics for noopmetrics {
    fn recordcounter(&self, _name: &'static str, _value: u64) {}

    fn recordgauge(&self, _name: &'static str, _value: i64) {}
}
