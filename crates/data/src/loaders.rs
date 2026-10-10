use crate::batches::batch;
use crate::dataerror;
use crate::dataset;
use crate::samplers::sampler;
use lib::limits;
use lib::seed;
use lib::tokenid;

/// Configuration for a [`loader`].
///
/// The configuration is plain data; [`loader::new`] validates it against a
/// [`limits`] budget before any token buffer is allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct loaderconfig {
    batchsize: usize,
    seqlength: usize,
    pad: tokenid,
    shuffle: bool,
    droplast: bool,
}

impl loaderconfig {
    /// Creates a loader configuration.
    ///
    /// `seqlength` is the length of every packed sequence, `batchsize` is the
    /// number of sequences per batch, and `pad` fills the final partial
    /// sequence. `shuffle` selects a deterministic per-epoch permutation of the
    /// record order. `droplast` drops a trailing batch that is not full instead
    /// of padding it.
    #[must_use]
    pub const fn new(
        batchsize: usize,
        seqlength: usize,
        pad: tokenid,
        shuffle: bool,
        droplast: bool,
    ) -> Self {
        Self {
            batchsize,
            seqlength,
            pad,
            shuffle,
            droplast,
        }
    }

    /// Returns the number of sequences per batch.
    #[must_use]
    pub const fn getbatchsize(&self) -> usize {
        self.batchsize
    }

    /// Returns the length of every packed sequence.
    #[must_use]
    pub const fn getseqlength(&self) -> usize {
        self.seqlength
    }

    /// Returns the padding token used for the final partial sequence and any
    /// padded trailing batch.
    #[must_use]
    pub const fn getpad(&self) -> tokenid {
        self.pad
    }

    /// Returns whether each epoch shuffles the record order.
    #[must_use]
    pub const fn getshuffle(&self) -> bool {
        self.shuffle
    }

    /// Returns whether a trailing non-full batch is dropped.
    #[must_use]
    pub const fn getdroplast(&self) -> bool {
        self.droplast
    }
}

/// A lifecycle-explicit data loader over a [`dataset`].
///
/// The loader borrows the dataset immutably and owns all iteration state: the
/// current epoch's record order, the cursor into that order, the in-progress
/// packed sequence, and the in-progress batch. Because that state is private to
/// the loader, two loaders over the same dataset never share request-scoped
/// state.
///
/// # Epochs
///
/// Tokens are packed into `seqlength` sequences and grouped into batches of
/// `batchsize` sequences. Each epoch yields batches until the record order is
/// exhausted. [`Self::reset`] starts the next epoch: with `shuffle` set, the
/// order is a new deterministic permutation drawn from the sampler's stream, so
/// successive epochs differ while the whole sequence of epochs is reproducible
/// from the seed.
///
/// # Termination and failure
///
/// [`Self::getnext`] returns `Ok(None)` exactly when the epoch is exhausted and
/// `Ok(Some(batch))` for each batch before that. When `droplast` is set, a
/// trailing non-full batch is discarded; otherwise it is padded to a full batch
/// with the configured pad token before being returned. Once the loader reaches
/// that terminal state it keeps returning `Ok(None)` until [`Self::reset`] is
/// called. A failure returned from [`Self::getnext`] also marks the loader
/// terminal, so a failed loader is not silently reused; the caller must reset
/// or reconstruct it. All buffers are owned by the loader and released when it
/// is dropped.
pub struct loader<'a, D: dataset> {
    data: &'a D,
    config: loaderconfig,
    limits: limits,
    sampler: sampler,
    epoch: u64,
    order: Vec<usize>,
    position: usize,
    offset: usize,
    seqbuf: Vec<tokenid>,
    batchtokens: Vec<tokenid>,
    batchtotal: usize,
    terminated: bool,
}

impl<'a, D: dataset> loader<'a, D> {
    /// Creates a loader and starts its first epoch.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::zerobatchsize`] or [`dataerror::zeroseqlength`]
    /// when a dimension is zero, [`dataerror::elementcountoverflow`] when the
    /// elements per batch (`batchsize * seqlength`) do not fit in `usize`, the
    /// batch, sequence, or element limit variants when a dimension exceeds the
    /// corresponding bound in `lim`, and any error from the first epoch's order
    /// construction (see [`Self::reset`]).
    pub fn new(data: &'a D, config: loaderconfig, lim: limits, s: seed) -> Result<Self, dataerror> {
        if config.getbatchsize() == 0 {
            return Err(dataerror::zerobatchsize);
        }
        if config.getseqlength() == 0 {
            return Err(dataerror::zeroseqlength);
        }
        lim.checkbatch(config.getbatchsize())
            .map_err(|_| dataerror::batchlimitexceeded {
                requested: config.getbatchsize(),
                limit: lim.getmaxbatch(),
            })?;
        lim.checksequencelen(config.getseqlength()).map_err(|_| {
            dataerror::sequencelenlimitexceeded {
                requested: config.getseqlength(),
                limit: lim.getmaxsequencelen(),
            }
        })?;
        let batchtotal = config
            .getbatchsize()
            .checked_mul(config.getseqlength())
            .ok_or(dataerror::elementcountoverflow)?;
        lim.checkelements(batchtotal)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: batchtotal,
                limit: lim.getmaxelements(),
            })?;

        let mut loader = Self {
            data,
            config,
            limits: lim,
            sampler: sampler::new(s),
            epoch: 0,
            order: Vec::new(),
            position: 0,
            offset: 0,
            seqbuf: Vec::with_capacity(config.getseqlength()),
            batchtokens: Vec::with_capacity(batchtotal),
            batchtotal,
            terminated: false,
        };
        loader.reset()?;
        Ok(loader)
    }

    /// Returns the number of epochs started, counting from one after
    /// construction.
    #[must_use]
    pub const fn getepoch(&self) -> u64 {
        self.epoch
    }

    /// Starts the next epoch.
    ///
    /// The record order is rebuilt: a deterministic permutation when `shuffle`
    /// is set, otherwise the identity order. All cursors and buffers are reset
    /// and the loader becomes active again.
    ///
    /// # Errors
    ///
    /// Returns [`dataerror::elementcountlimitexceeded`] when the number of
    /// records exceeds [`limits::getmaxelements`].
    pub fn reset(&mut self) -> Result<(), dataerror> {
        let len = self.data.getlen();
        self.limits
            .checkelements(len)
            .map_err(|_| dataerror::elementcountlimitexceeded {
                requested: len,
                limit: self.limits.getmaxelements(),
            })?;
        self.order = if self.config.getshuffle() {
            self.sampler.shuffle(len, &self.limits)?
        } else {
            (0..len).collect()
        };
        self.position = 0;
        self.offset = 0;
        self.seqbuf.clear();
        self.batchtokens.clear();
        self.terminated = false;
        self.epoch += 1;
        Ok(())
    }

    /// Returns the next batch, or `Ok(None)` when the epoch is exhausted.
    ///
    /// # Errors
    ///
    /// Propagates record access and batch construction failures. A returned
    /// error leaves the loader terminal; see the type-level documentation.
    pub fn getnext(&mut self) -> Result<Option<batch>, dataerror> {
        if self.terminated {
            return Ok(None);
        }
        let seqlength = self.config.getseqlength();
        loop {
            if self.batchtokens.len() == self.batchtotal {
                let ready = batch::new(
                    std::mem::take(&mut self.batchtokens),
                    self.config.getbatchsize(),
                    seqlength,
                    &self.limits,
                )?;
                self.batchtokens = Vec::with_capacity(self.batchtotal);
                return Ok(Some(ready));
            }
            if !self.fillsequence(seqlength)? {
                return self.finish(seqlength);
            }
            while self.seqbuf.len() < seqlength {
                self.seqbuf.push(self.config.getpad());
            }
            self.batchtokens.extend_from_slice(&self.seqbuf);
        }
    }

    /// Fills `seqbuf` with up to `seqlength` tokens from the current record
    /// order, splitting records across sequences as needed.
    ///
    /// Returns `true` when at least one token was added, and `false` when the
    /// record order is exhausted.
    fn fillsequence(&mut self, seqlength: usize) -> Result<bool, dataerror> {
        self.seqbuf.clear();
        while self.seqbuf.len() < seqlength {
            if self.position >= self.order.len() {
                break;
            }
            let index = self.order[self.position];
            let rec = self.data.getrecord(index)?;
            let tokens = rec.gettokens();
            let remaining = tokens.get(self.offset..).ok_or(dataerror::lengthmismatch {
                expected: self.offset,
                actual: tokens.len(),
            })?;
            let take = (seqlength - self.seqbuf.len()).min(remaining.len());
            self.seqbuf.extend_from_slice(&remaining[..take]);
            self.offset += take;
            if self.offset >= tokens.len() {
                self.position += 1;
                self.offset = 0;
            }
        }
        Ok(!self.seqbuf.is_empty())
    }

    /// Handles the end of an epoch: drops or pads the trailing batch and marks
    /// the loader terminal.
    fn finish(&mut self, seqlength: usize) -> Result<Option<batch>, dataerror> {
        self.terminated = true;
        if self.batchtokens.is_empty() || self.config.getdroplast() {
            self.batchtokens.clear();
            return Ok(None);
        }
        while self.batchtokens.len() < self.batchtotal {
            for _ in 0..seqlength {
                self.batchtokens.push(self.config.getpad());
            }
        }
        batch::new(
            std::mem::take(&mut self.batchtokens),
            self.config.getbatchsize(),
            seqlength,
            &self.limits,
        )
        .map(Some)
    }
}
