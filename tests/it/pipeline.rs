//! End-to-end exercise of the `data` public API across its components.
//!
//! This is a cross-crate integration test: it links `data` as an external
//! crate and uses only the public interface, so it verifies the composition of
//! `textdataset`, `packing`, `batch`, `sampler`, and `loader` rather than their
//! internals.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::tokens;
use data::{batch, dataset, loader, loaderconfig, packing, record, sampler, textdataset};
use lib::limits;
use lib::seed;
use lib::tokenid;

fn budget() -> limits {
    limits::new(64, 4096, 16, 8)
}

#[test]
fn dataset_pack_batch_round_trip() {
    let samples =
        textdataset::fromtokens(vec![tokens(&[1, 2, 3]), tokens(&[4, 5, 6, 7])], &budget())
            .expect("valid dataset");
    assert_eq!(dataset::getlen(&samples), 2);
    assert_eq!(samples.gettotaltokens(), 7);

    // Packing concatenates records and yields a trailing partial sequence.
    let packed = packing::new(3)
        .expect("nonzero length")
        .pack(
            &[
                record::new(tokens(&[1, 2, 3])),
                record::new(tokens(&[4, 5, 6, 7])),
            ],
            &budget(),
        )
        .expect("within budget");
    assert_eq!(packed.len(), 3);
    assert_eq!(packed[0].gettokens(), tokens(&[1, 2, 3]).as_slice());
    assert_eq!(packed[1].gettokens(), tokens(&[4, 5, 6]).as_slice());
    assert_eq!(packed[2].gettokens(), tokens(&[7]).as_slice());

    // Batching pads the partial row into a rectangle.
    let batched = batch::fromrecords(
        &[record::new(tokens(&[1, 2, 3])), record::new(tokens(&[4]))],
        4,
        tokenid::new(0),
        &budget(),
    )
    .expect("within budget");
    assert_eq!(batched.getbatchsize(), 2);
    assert_eq!(
        batched.getrow(0).expect("row 0"),
        tokens(&[1, 2, 3, 0]).as_slice()
    );
    assert_eq!(
        batched.getrow(1).expect("row 1"),
        tokens(&[4, 0, 0, 0]).as_slice()
    );
}

#[test]
fn sampler_drives_a_reproducible_loader() {
    let samples = textdataset::fromtokens(
        vec![tokens(&[10]), tokens(&[20]), tokens(&[30]), tokens(&[40])],
        &budget(),
    )
    .expect("valid dataset");

    let config = loaderconfig::new(2, 2, tokenid::new(0), true, false);
    let mut first = loader::new(&samples, config, budget(), seed::new(7)).expect("loader");
    let mut second = loader::new(&samples, config, budget(), seed::new(7)).expect("loader");

    let mut firststream = Vec::new();
    while let Some(loaded) = first.getnext().expect("no failure") {
        assert_eq!(loaded.getbatchsize(), 2);
        assert_eq!(loaded.getseqlength(), 2);
        firststream.extend_from_slice(loaded.gettokens());
    }
    let mut secondstream = Vec::new();
    while let Some(loaded) = second.getnext().expect("no failure") {
        secondstream.extend_from_slice(loaded.gettokens());
    }

    // Two independent loaders with the same seed produce the same token stream.
    assert_eq!(firststream, secondstream);

    // The standalone sampler yields the same record order the loader applied.
    // With one token per record and sequence length two, consecutive records in
    // the shuffled order pack into one sequence with no padding.
    let order = sampler::new(seed::new(7))
        .shuffle(4, &budget())
        .expect("shuffle");
    let values = [10u32, 20, 30, 40];
    let expected: Vec<tokenid> = order.iter().map(|&i| tokenid::new(values[i])).collect();
    assert_eq!(firststream, expected);
}

#[test]
fn loader_terminates_and_resets() {
    let samples = textdataset::fromtokens(vec![tokens(&[1, 2, 3, 4])], &budget()).expect("dataset");
    let config = loaderconfig::new(1, 2, tokenid::new(0), false, false);
    let mut l = loader::new(&samples, config, budget(), seed::new(0)).expect("loader");

    assert_eq!(l.getepoch(), 1);
    assert_eq!(
        l.getnext().expect("batch").expect("some").gettokens(),
        tokens(&[1, 2]).as_slice()
    );
    assert_eq!(
        l.getnext().expect("batch").expect("some").gettokens(),
        tokens(&[3, 4]).as_slice()
    );
    assert!(l.getnext().expect("end").is_none());

    l.reset().expect("reset");
    assert_eq!(l.getepoch(), 2);
    assert!(l.getnext().expect("batch").is_some());
}
