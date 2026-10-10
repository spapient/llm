//! Black-box evidence for the production diagnostic seam.
//!
//! The seam is a production capability, not test code (see
//! `docs/decision-records/blackboxtesting.md`): workflow crates emit `tracing`
//! events at decision points, and the events are discarded unless an
//! application installs a subscriber with a matching filter. These tests drive
//! a public workflow under a thread-local subscriber and assert that the
//! expected diagnostic is emitted at one filter level and suppressed at
//! another, without recompiling the crate under test.
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;

use ::checkpoint::checkpoint;
use tensor::{dtype, shape};

/// A `Write` sink that appends formatted events to a shared buffer.
#[derive(Clone)]
struct capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "llm-diagnostics-{}-{tag}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn ckpt() -> checkpoint {
    let mut ckpt = checkpoint::new(1);
    let dims = shape::new(vec![2, 2]).unwrap();
    ckpt.addtensor("w".to_string(), dtype::f32, dims, &[0u8; 16])
        .unwrap();
    ckpt
}

/// Runs `work` with a thread-local subscriber writing at `level`, and returns
/// the captured diagnostic text.
fn capture_at(level: LevelFilter, work: impl FnOnce()) -> String {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&buffer);
    let subscriber = fmt::Subscriber::builder()
        .with_max_level(level)
        .with_writer(move || capture(Arc::clone(&writer)))
        .finish();
    tracing::subscriber::with_default(subscriber, work);
    let bytes = buffer.lock().unwrap().clone();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn a_debug_subscriber_captures_the_load_diagnostic() {
    let dir = scratch("debug");
    let path = dir.join("model.ckpt");
    ::checkpoint::savecheckpoint(&path, &ckpt()).unwrap();

    let text = capture_at(LevelFilter::DEBUG, || {
        ::checkpoint::loadcheckpoint(&path).unwrap();
    });

    assert!(text.contains("checkpoint loaded"), "captured: {text}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_warn_subscriber_suppresses_the_debug_diagnostic() {
    let dir = scratch("warn");
    let path = dir.join("model.ckpt");
    ::checkpoint::savecheckpoint(&path, &ckpt()).unwrap();

    let text = capture_at(LevelFilter::WARN, || {
        ::checkpoint::loadcheckpoint(&path).unwrap();
    });

    assert!(!text.contains("checkpoint loaded"), "captured: {text}");
    let _ = std::fs::remove_dir_all(&dir);
}
