//! 分段求解：段数由预算解出来，段界保留、段内重放。

use diagnostic::Result;

use crate::budget::{Budget, MemoryPolicy};
use crate::error::invalid;
use crate::facts::{Facts, Ledger, NONE, Scratch};
use crate::plan::{Plan, SITE};
use crate::schedule::Schedule;
use crate::tape::Tape;

impl Facts {
    /// 分段：段数由预算解出来，段界上的激活保留，段内的丢弃后重放。
    pub(crate) fn plan(
        &self,
        tape: &Tape,
        schedule: &Schedule,
        budget: Budget,
        policy: MemoryPolicy,
    ) -> Result<Plan> {
        if self.order.is_empty() {
            // total > budget 且 order 为空只可能是 total 与 budget 都为 0，已被全留路径拦下。
            return Err(invalid(SITE, "没有需要保留的激活，但预算为 0"));
        }
        let mut scratch = Scratch::new(self.bytes.len());
        let mut keep = vec![false; self.bytes.len()];
        let mut fitted: Option<(usize, Ledger)> = None;
        let mut minimum: Option<u64> = None;
        for segments in self.trials(budget) {
            self.segmented(segments, &mut keep);
            let ledger = self.measure(schedule, &keep, &mut scratch);
            minimum = Some(minimum.map_or(ledger.peak, |value| value.min(ledger.peak)));
            if ledger.peak <= budget.bytes() {
                // 试算按段数升序，后覆盖前：留下能装下的段数最多的一份。
                fitted = Some((segments, ledger));
            }
        }
        match fitted {
            Some((segments, mut ledger)) => {
                self.segmented(segments, &mut keep);
                ledger.release = self.release(tape, &keep);
                Ok(self.assemble(keep, ledger, budget, policy))
            }
            None => Err(invalid(
                SITE,
                format!(
                    "预算 {} 字节放不下：{} 个激活共 {} 字节，最小峰值约 {} 字节（下界 {} 字节）",
                    budget.bytes(),
                    self.order.len(),
                    self.total,
                    minimum.unwrap_or(self.total),
                    self.floor
                ),
            )),
        }
    }

    /// 候选段数：在 `√n` 与"保留量不超过预算一半"两个中心各扫 ±2 档，然后去重。
    fn trials(&self, budget: Budget) -> Vec<usize> {
        // 等字节切分下峰值 ≈ k·(total/n) + total/k，在 k = √n 处取到最小。
        let count = self.order.len();
        let root = (count as f64).sqrt() as usize;
        let quota = self.total.div_ceil((budget.bytes() / 2).max(1)) as usize;
        let mut trials: Vec<usize> = Vec::new();
        for centre in [root, quota] {
            for offset in [-2i64, -1, 0, 1, 2] {
                let trial = centre as i64 + offset;
                if trial >= 1 && trial as usize <= count {
                    trials.push(trial as usize);
                }
            }
        }
        trials.sort_unstable();
        trials.dedup();
        trials
    }

    /// 分段后的保留集：段界上是检查点；零字节的与 detach 出去的值无处重放，无条件留。
    fn segmented(&self, segments: usize, keep: &mut [bool]) {
        keep.fill(false);
        self.boundaries(segments, keep);
        self.candidates
            .iter()
            .filter(|id| self.bytes[**id as usize] == 0 || self.producer[**id as usize] == NONE)
            .for_each(|id| keep[*id as usize] = true);
    }

    /// 按字节数均匀地放 `segments` 个检查点：段界落在前缀和最接近等分点的那个激活上。
    fn boundaries(&self, segments: usize, keep: &mut [bool]) {
        let last = self.order.len() - 1;
        let total = self.total.max(1);
        for index in 1..=segments {
            let target = (u128::from(total) * index as u128 / segments as u128) as u64;
            let position = self.prefix.partition_point(|sum| *sum < target).min(last);
            keep[self.order[position] as usize] = true;
        }
        // 第一个激活没有更早的检查点可依，必须留；否则第一段无从重放。
        keep[self.order[0] as usize] = true;
    }
}
