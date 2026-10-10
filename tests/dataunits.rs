//! Relocated unit tests for the `data` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/data/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/data/src/batches.rs`.
mod batches {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::batch;
    use data::dataerror;
    use data::record;
    use lib::limits;
    use lib::tokenid;

    fn tokens(vals: &[u32]) -> Vec<tokenid> {
        vals.iter().copied().map(tokenid::new).collect()
    }

    fn lim() -> limits {
        limits::new(64, 1024, 8, 4)
    }

    #[test]
    fn new_accepts_a_rectangular_buffer() {
        let b = batch::new(tokens(&[1, 2, 3, 4]), 2, 2, &lim()).unwrap();
        assert_eq!(b.getbatchsize(), 2);
        assert_eq!(b.getseqlength(), 2);
        assert_eq!(b.gettokens(), tokens(&[1, 2, 3, 4]).as_slice());
        assert_eq!(b.getrow(0).unwrap(), tokens(&[1, 2]));
        assert_eq!(b.getrow(1).unwrap(), tokens(&[3, 4]));
    }

    #[test]
    fn new_rejects_length_mismatch() {
        assert_eq!(
            batch::new(tokens(&[1, 2, 3]), 2, 2, &lim()),
            Err(dataerror::lengthmismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn new_rejects_zero_dimensions() {
        assert_eq!(
            batch::new(Vec::new(), 0, 2, &lim()),
            Err(dataerror::zerobatchsize)
        );
        assert_eq!(
            batch::new(Vec::new(), 2, 0, &lim()),
            Err(dataerror::zeroseqlength)
        );
    }

    #[test]
    fn fromrecords_pads_short_records() {
        let records = vec![record::new(tokens(&[1, 2, 3])), record::new(tokens(&[4]))];
        let b = batch::fromrecords(&records, 4, tokenid::new(0), &lim()).unwrap();
        assert_eq!(b.getbatchsize(), 2);
        assert_eq!(b.getseqlength(), 4);
        assert_eq!(b.getrow(0).unwrap(), tokens(&[1, 2, 3, 0]));
        assert_eq!(b.getrow(1).unwrap(), tokens(&[4, 0, 0, 0]));
    }

    #[test]
    fn fromrecords_accepts_rows_equal_to_seqlength() {
        let records = vec![record::new(tokens(&[1, 2])), record::new(tokens(&[3, 4]))];
        let b = batch::fromrecords(&records, 2, tokenid::new(0), &lim()).unwrap();
        assert_eq!(b.getrow(0).unwrap(), tokens(&[1, 2]));
        assert_eq!(b.getrow(1).unwrap(), tokens(&[3, 4]));
    }

    #[test]
    fn fromrecords_rejects_overlong_record() {
        let records = vec![record::new(tokens(&[1, 2, 3]))];
        assert_eq!(
            batch::fromrecords(&records, 2, tokenid::new(0), &lim()),
            Err(dataerror::recordeceedseqlength {
                index: 0,
                len: 3,
                seqlength: 2,
            })
        );
    }

    #[test]
    fn fromrecords_rejects_empty_input() {
        assert_eq!(
            batch::fromrecords(&[], 2, tokenid::new(0), &lim()),
            Err(dataerror::emptybatch)
        );
    }

    #[test]
    fn getrow_rejects_out_of_range() {
        let b = batch::new(tokens(&[1, 2]), 1, 2, &lim()).unwrap();
        assert_eq!(
            b.getrow(1),
            Err(dataerror::indexoutofrange { index: 1, len: 1 })
        );
    }

    #[test]
    fn enforces_batch_and_sequence_budgets() {
        let l = limits::new(64, 1024, 4, 3);
        assert_eq!(
            batch::new(tokens(&[1, 2, 3, 4, 5, 6, 7, 8]), 4, 2, &l),
            Err(dataerror::batchlimitexceeded {
                requested: 4,
                limit: 3,
            })
        );
        assert_eq!(
            batch::new(tokens(&[1, 2, 3, 4, 5]), 1, 5, &l),
            Err(dataerror::sequencelenlimitexceeded {
                requested: 5,
                limit: 4,
            })
        );
    }

    #[test]
    fn enforces_element_budget() {
        let l = limits::new(4, 1024, 8, 4);
        assert_eq!(
            batch::new(tokens(&[1, 2, 3, 4, 5, 6]), 3, 2, &l),
            Err(dataerror::elementcountlimitexceeded {
                requested: 6,
                limit: 4,
            })
        );
    }
}

/// Tests relocated from `crates/data/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::dataerror;

    #[test]
    fn variants_carry_distinguishing_values() {
        assert_ne!(
            dataerror::indexoutofrange { index: 0, len: 1 },
            dataerror::indexoutofrange { index: 1, len: 1 }
        );
        assert_ne!(
            dataerror::elementcountlimitexceeded {
                requested: 9,
                limit: 8,
            },
            dataerror::batchlimitexceeded {
                requested: 9,
                limit: 8,
            }
        );
    }

    #[test]
    fn display_reports_carried_values() {
        let e = dataerror::indexoutofrange { index: 3, len: 2 };
        assert_eq!(
            e.to_string(),
            "record index 3 is out of range for 2 records"
        );
        let l = dataerror::sequencelenlimitexceeded {
            requested: 17,
            limit: 16,
        };
        assert_eq!(
            l.to_string(),
            "requested sequence length 17 exceeds limit 16"
        );
    }
}

/// Tests relocated from `crates/data/src/loaders.rs`.
mod loaders {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::dataerror;
    use data::dataset;
    use data::record;
    use data::textdataset;
    use data::{loader, loaderconfig};
    use lib::limits;
    use lib::seed;
    use lib::tokenid;

    fn tokens(vals: &[u32]) -> Vec<tokenid> {
        vals.iter().copied().map(tokenid::new).collect()
    }

    fn lim() -> limits {
        limits::new(64, 1024, 16, 8)
    }

    fn ds(docs: &[&[u32]]) -> textdataset {
        let records = docs.iter().map(|d| record::new(tokens(d))).collect();
        textdataset::new(records, &lim()).unwrap()
    }

    fn config(batchsize: usize, seqlength: usize, droplast: bool) -> loaderconfig {
        loaderconfig::new(batchsize, seqlength, tokenid::new(0), false, droplast)
    }

    #[test]
    fn one_epoch_packs_and_batches_with_padding() {
        // 10 tokens, seqlength 4 -> sequences [1,2,3,4],[5,6,7,8],[9,10,0,0].
        let data = ds(&[&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
        let mut l = loader::new(&data, config(2, 4, false), lim(), seed::new(1)).unwrap();
        let first = l.getnext().unwrap().unwrap();
        assert_eq!(first.getbatchsize(), 2);
        assert_eq!(first.getseqlength(), 4);
        assert_eq!(
            first.gettokens(),
            tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).as_slice()
        );
        // Second batch holds the padded tail sequence plus a fully padded row.
        let second = l.getnext().unwrap().unwrap();
        assert_eq!(
            second.gettokens(),
            tokens(&[9, 10, 0, 0, 0, 0, 0, 0]).as_slice()
        );
        assert!(l.getnext().unwrap().is_none());
    }

    #[test]
    fn droplast_discards_the_trailing_partial_batch() {
        let data = ds(&[&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
        let mut l = loader::new(&data, config(2, 4, true), lim(), seed::new(1)).unwrap();
        let first = l.getnext().unwrap().unwrap();
        assert_eq!(
            first.gettokens(),
            tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).as_slice()
        );
        assert!(l.getnext().unwrap().is_none());
    }

    #[test]
    fn spans_record_boundaries() {
        let data = ds(&[&[1, 2], &[3, 4], &[5]]);
        let mut l = loader::new(&data, config(1, 4, false), lim(), seed::new(1)).unwrap();
        let first = l.getnext().unwrap().unwrap();
        assert_eq!(first.getrow(0).unwrap(), tokens(&[1, 2, 3, 4]));
        let second = l.getnext().unwrap().unwrap();
        assert_eq!(second.getrow(0).unwrap(), tokens(&[5, 0, 0, 0]));
        assert!(l.getnext().unwrap().is_none());
    }

    #[test]
    fn empty_dataset_yields_no_batches() {
        let data = ds(&[]);
        let mut l = loader::new(&data, config(2, 4, false), lim(), seed::new(1)).unwrap();
        assert!(l.getnext().unwrap().is_none());
    }

    #[test]
    fn unshuffled_epoch_is_the_deterministic_identity_order() {
        let data = ds(&[&[1], &[2], &[3]]);
        let mut l = loader::new(&data, config(1, 2, false), lim(), seed::new(1)).unwrap();
        let a = l.getnext().unwrap().unwrap();
        let b = l.getnext().unwrap().unwrap();
        assert_eq!(a.getrow(0).unwrap(), tokens(&[1, 2]));
        assert_eq!(b.getrow(0).unwrap(), tokens(&[3, 0]));
    }

    #[test]
    fn shuffled_epoch_is_reproducible_from_the_seed() {
        let data = ds(&[&[1], &[2], &[3], &[4], &[5], &[6]]);
        let cfg = loaderconfig::new(2, 2, tokenid::new(0), true, false);
        let mut a = loader::new(&data, cfg, lim(), seed::new(11)).unwrap();
        let mut b = loader::new(&data, cfg, lim(), seed::new(11)).unwrap();
        let mut lefta = Vec::new();
        while let Some(batch) = a.getnext().unwrap() {
            lefta.extend_from_slice(batch.gettokens());
        }
        let mut leftb = Vec::new();
        while let Some(batch) = b.getnext().unwrap() {
            leftb.extend_from_slice(batch.gettokens());
        }
        assert_eq!(lefta, leftb);
    }

    #[test]
    fn reset_advances_epoch_and_repeats_a_reproducible_sequence() {
        let data = ds(&[&[1], &[2], &[3], &[4]]);
        let cfg = loaderconfig::new(1, 2, tokenid::new(0), true, false);
        let mut a = loader::new(&data, cfg, lim(), seed::new(5)).unwrap();
        let mut b = loader::new(&data, cfg, lim(), seed::new(5)).unwrap();
        for _ in 0..2 {
            a.getnext().unwrap();
            a.getnext().unwrap();
            assert!(a.getnext().unwrap().is_none());
            b.getnext().unwrap();
            b.getnext().unwrap();
            assert!(b.getnext().unwrap().is_none());
            a.reset().unwrap();
            b.reset().unwrap();
        }
        assert_eq!(a.getepoch(), 3);
        assert_eq!(b.getepoch(), 3);
    }

    #[test]
    fn successive_shuffled_epochs_differ() {
        let data = ds(&[&[1], &[2], &[3], &[4], &[5], &[6], &[7], &[8]]);
        let cfg = loaderconfig::new(8, 1, tokenid::new(0), true, false);
        let mut l = loader::new(&data, cfg, lim(), seed::new(2)).unwrap();
        let first: Vec<tokenid> = l.getnext().unwrap().unwrap().gettokens().to_vec();
        l.reset().unwrap();
        let second: Vec<tokenid> = l.getnext().unwrap().unwrap().gettokens().to_vec();
        assert_ne!(first, second);
    }

    #[test]
    fn terminal_loader_keeps_returning_none() {
        let data = ds(&[&[1, 2]]);
        let mut l = loader::new(&data, config(1, 2, false), lim(), seed::new(1)).unwrap();
        assert!(l.getnext().unwrap().is_some());
        assert!(l.getnext().unwrap().is_none());
        assert!(l.getnext().unwrap().is_none());
    }

    #[test]
    fn rejects_zero_dimensions_and_budgets() {
        let data = ds(&[&[1, 2, 3, 4]]);
        assert!(matches!(
            loader::new(&data, config(0, 4, false), lim(), seed::new(1)),
            Err(dataerror::zerobatchsize)
        ));
        assert!(matches!(
            loader::new(&data, config(1, 0, false), lim(), seed::new(1)),
            Err(dataerror::zeroseqlength)
        ));
        let small = limits::new(64, 1024, 4, 2);
        assert!(matches!(
            loader::new(&data, config(3, 4, false), small, seed::new(1)),
            Err(dataerror::batchlimitexceeded {
                requested: 3,
                limit: 2,
            })
        ));
    }

    #[test]
    fn dataset_trait_is_usable_through_the_loader_bound() {
        // Confirms the loader accepts any `dataset` implementation, not only
        // the concrete `textdataset`.
        fn drain<D: dataset>(data: &D) -> Vec<usize> {
            (0..data.getlen()).collect()
        }
        let data = ds(&[&[1], &[2]]);
        assert_eq!(drain(&data), vec![0, 1]);
    }
}

/// Tests relocated from `crates/data/src/packings.rs`.
mod packings {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::dataerror;
    use data::packing;
    use data::record;
    use lib::limits;
    use lib::tokenid;

    fn tokens(vals: &[u32]) -> Vec<tokenid> {
        vals.iter().copied().map(tokenid::new).collect()
    }

    fn lim() -> limits {
        limits::new(64, 1024, 8, 4)
    }

    #[test]
    fn zero_length_is_rejected() {
        assert_eq!(packing::new(0), Err(dataerror::zeroseqlength));
    }

    #[test]
    fn empty_input_yields_no_sequences() {
        let p = packing::new(4).unwrap();
        assert_eq!(p.pack(&[], &lim()), Ok(Vec::new()));
    }

    #[test]
    fn exact_multiple_has_no_partial_tail() {
        let p = packing::new(4).unwrap();
        let records = vec![record::new(tokens(&[1, 2, 3, 4, 5, 6, 7, 8]))];
        let packed = p.pack(&records, &lim()).unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].gettokens(), tokens(&[1, 2, 3, 4]));
        assert_eq!(packed[1].gettokens(), tokens(&[5, 6, 7, 8]));
    }

    #[test]
    fn partial_tail_is_kept_unpadded() {
        let p = packing::new(4).unwrap();
        let records = vec![record::new(tokens(&[1, 2, 3, 4, 5, 6]))];
        let packed = p.pack(&records, &lim()).unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].gettokens(), tokens(&[1, 2, 3, 4]));
        assert_eq!(packed[1].gettokens(), tokens(&[5, 6]));
    }

    #[test]
    fn concatenates_across_record_boundaries() {
        let p = packing::new(4).unwrap();
        let records = vec![
            record::new(tokens(&[1, 2, 3])),
            record::new(tokens(&[4, 5])),
            record::new(tokens(&[6])),
        ];
        let packed = p.pack(&records, &lim()).unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].gettokens(), tokens(&[1, 2, 3, 4]));
        assert_eq!(packed[1].gettokens(), tokens(&[5, 6]));
    }

    #[test]
    fn splits_records_longer_than_seqlength() {
        let p = packing::new(3).unwrap();
        let records = vec![record::new(tokens(&[1, 2, 3, 4, 5, 6, 7]))];
        let packed = p.pack(&records, &lim()).unwrap();
        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0].gettokens(), tokens(&[1, 2, 3]));
        assert_eq!(packed[1].gettokens(), tokens(&[4, 5, 6]));
        assert_eq!(packed[2].gettokens(), tokens(&[7]));
    }

    #[test]
    fn empty_records_are_skipped_without_emitting_sequences() {
        let p = packing::new(2).unwrap();
        let records = vec![
            record::new(Vec::new()),
            record::new(tokens(&[1])),
            record::new(Vec::new()),
        ];
        let packed = p.pack(&records, &lim()).unwrap();
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].gettokens(), tokens(&[1]));
    }

    #[test]
    fn rejects_total_tokens_above_the_budget() {
        let p = packing::new(2).unwrap();
        let l = limits::new(5, 1024, 8, 4);
        let records = vec![
            record::new(tokens(&[1, 2, 3])),
            record::new(tokens(&[4, 5, 6])),
        ];
        assert_eq!(
            p.pack(&records, &l),
            Err(dataerror::elementcountlimitexceeded {
                requested: 6,
                limit: 5,
            })
        );
    }
}

/// Tests relocated from `crates/data/src/records.rs`.
mod records {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::record;
    use lib::tokenid;

    fn tokens(vals: &[u32]) -> Vec<tokenid> {
        vals.iter().copied().map(tokenid::new).collect()
    }

    #[test]
    fn exposes_tokens_in_order() {
        let r = record::new(tokens(&[3, 1, 2]));
        assert_eq!(r.getlen(), 3);
        assert!(!r.isempty());
        assert_eq!(r.gettokens(), tokens(&[3, 1, 2]).as_slice());
    }

    #[test]
    fn empty_record_is_reported() {
        let r = record::new(Vec::new());
        assert_eq!(r.getlen(), 0);
        assert!(r.isempty());
        assert!(r.gettokens().is_empty());
    }
}

/// Tests relocated from `crates/data/src/samplers.rs`.
mod samplers {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::dataerror;
    use data::sampler;
    use lib::limits;
    use lib::seed;

    fn lim() -> limits {
        limits::new(32, 1024, 8, 4)
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut s = sampler::new(seed::new(7));
        let mut order = s.shuffle(10, &lim()).unwrap();
        order.sort_unstable();
        assert_eq!(order, (0..10).collect::<Vec<usize>>());
    }

    #[test]
    fn shuffle_is_reproducible_for_the_same_seed() {
        let mut a = sampler::new(seed::new(42));
        let mut b = sampler::new(seed::new(42));
        assert_eq!(
            a.shuffle(16, &lim()).unwrap(),
            b.shuffle(16, &lim()).unwrap()
        );
    }

    #[test]
    fn shuffle_advances_the_stream_between_calls() {
        let mut s = sampler::new(seed::new(42));
        let first = s.shuffle(16, &lim()).unwrap();
        let second = s.shuffle(16, &lim()).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn shuffle_of_short_inputs_is_the_identity() {
        let mut s = sampler::new(seed::new(1));
        assert_eq!(s.shuffle(0, &lim()).unwrap(), Vec::<usize>::new());
        assert_eq!(s.shuffle(1, &lim()).unwrap(), vec![0]);
    }

    #[test]
    fn shuffle_is_limited() {
        let l = limits::new(4, 1024, 8, 4);
        let mut s = sampler::new(seed::new(1));
        assert_eq!(
            s.shuffle(5, &l),
            Err(dataerror::elementcountlimitexceeded {
                requested: 5,
                limit: 4,
            })
        );
    }

    #[test]
    fn sample_stays_within_range_and_count() {
        let mut s = sampler::new(seed::new(3));
        let draws = s.sample(5, 20, &lim()).unwrap();
        assert_eq!(draws.len(), 20);
        assert!(draws.iter().all(|&i| i < 5));
    }

    #[test]
    fn sample_is_reproducible_for_the_same_seed() {
        let mut a = sampler::new(seed::new(9));
        let mut b = sampler::new(seed::new(9));
        assert_eq!(
            a.sample(7, 20, &lim()).unwrap(),
            b.sample(7, 20, &lim()).unwrap()
        );
    }

    #[test]
    fn sample_of_zero_count_is_empty() {
        let mut s = sampler::new(seed::new(0));
        assert_eq!(s.sample(0, 0, &lim()).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn sample_from_empty_dataset_is_rejected() {
        let mut s = sampler::new(seed::new(0));
        assert_eq!(s.sample(0, 3, &lim()), Err(dataerror::emptydataset));
    }

    #[test]
    fn sample_count_is_limited() {
        let l = limits::new(4, 1024, 8, 4);
        let mut s = sampler::new(seed::new(1));
        assert_eq!(
            s.sample(2, 5, &l),
            Err(dataerror::elementcountlimitexceeded {
                requested: 5,
                limit: 4,
            })
        );
    }
}

/// Tests relocated from `crates/data/src/textdatasets.rs`.
mod textdatasets {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use data::dataerror;
    use data::dataset;
    use data::record;
    use data::textdataset;
    use lib::limits;
    use lib::tokenid;

    fn tokens(vals: &[u32]) -> Vec<tokenid> {
        vals.iter().copied().map(tokenid::new).collect()
    }

    fn lim() -> limits {
        limits::new(16, 1024, 8, 4)
    }

    #[test]
    fn reports_length_and_total_tokens() {
        let ds = textdataset::fromtokens(vec![tokens(&[1, 2]), tokens(&[3])], &lim()).unwrap();
        assert_eq!(ds.getlen(), 2);
        assert!(!ds.isempty());
        assert_eq!(ds.gettotaltokens(), 3);
    }

    #[test]
    fn indexed_access_returns_records_in_order() {
        let ds = textdataset::fromtokens(vec![tokens(&[1, 2]), tokens(&[3])], &lim()).unwrap();
        assert_eq!(ds.getrecord(0).unwrap().gettokens(), tokens(&[1, 2]));
        assert_eq!(ds.getrecord(1).unwrap().gettokens(), tokens(&[3]));
        assert_eq!(
            ds.getrecord(2),
            Err(dataerror::indexoutofrange { index: 2, len: 2 })
        );
    }

    #[test]
    fn empty_dataset_is_reported() {
        let ds = textdataset::new(Vec::new(), &lim()).unwrap();
        assert_eq!(ds.getlen(), 0);
        assert!(ds.isempty());
        assert_eq!(ds.gettotaltokens(), 0);
    }

    #[test]
    fn rejects_too_many_records() {
        let l = limits::new(2, 1024, 8, 4);
        let records = vec![
            record::new(tokens(&[1])),
            record::new(tokens(&[2])),
            record::new(tokens(&[3])),
        ];
        assert_eq!(
            textdataset::new(records, &l).unwrap_err(),
            dataerror::elementcountlimitexceeded {
                requested: 3,
                limit: 2,
            }
        );
    }

    #[test]
    fn rejects_too_many_total_tokens() {
        let l = limits::new(5, 1024, 8, 4);
        let records = vec![
            record::new(tokens(&[1, 2, 3])),
            record::new(tokens(&[4, 5, 6])),
        ];
        assert_eq!(
            textdataset::new(records, &l).unwrap_err(),
            dataerror::elementcountlimitexceeded {
                requested: 6,
                limit: 5,
            }
        );
    }

    #[test]
    fn accepts_count_and_tokens_equal_to_the_bound() {
        let l = limits::new(3, 1024, 8, 4);
        let records = vec![record::new(tokens(&[1])), record::new(tokens(&[2, 3]))];
        let ds = textdataset::new(records, &l).unwrap();
        assert_eq!(ds.getlen(), 2);
        assert_eq!(ds.gettotaltokens(), 3);
    }
}
