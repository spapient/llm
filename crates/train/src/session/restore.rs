//! 职责：会话快照回载——矩状态、累积窗口与绑定表按槽位对齐校验后回填。

use diagnostic::{Code, Error, Result};
use shard::Collective;

use crate::Kernel;
use crate::optimizer::Optimizer;
use snapshot::{GradAccum, Moments, SlotMoments, Snapshot, Ties};

use super::{SITE, Session};

impl<K: Kernel, O: Optimizer, C: Collective> Session<K, O, C> {
    /// 把快照状态回载到这个**已装配**会话：矩状态走优化器 restore，累积窗口汇总、绑定表与
    /// 步数按槽位编号对齐校验。
    ///
    /// 本接缝不从快照重建 tape / 参数槽 / 内核 / 权重（真实权重通路属后续里程碑）；回载前
    /// 会话必须已按同一张图、同一槽位顺序装配。
    ///
    /// # Errors
    ///
    /// 矩表/累积表槽位数或槽内长度不符、ties 编号越界或两端长度不符，一律 [`Code::Invalid`]。
    pub fn restore_snapshot(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.restore_moments(snapshot.moments())?;
        self.restore_accum(snapshot.accum())?;
        self.restore_ties(snapshot.ties())?;
        self.step = snapshot.facts().step();
        Ok(())
    }

    /// 矩状态回载：空表对应无矩优化器；非空表槽位数与每槽长度必须对齐会话参数槽。
    fn restore_moments(&mut self, moments: &Moments) -> Result<()> {
        let slots = moments.slots();
        self.check_moment_slots(slots)?;
        let momentum = slots.iter().map(|s| s.m().to_vec()).collect();
        let variance = slots.iter().map(|s| s.v().to_vec()).collect();
        self.optimizer
            .restore(momentum, variance, moments.step())
            .map_err(|error| error.with(SITE))?;
        Ok(())
    }

    /// 矩表槽位数与每槽长度对账；空表（无矩优化器）跳过槽数校验。
    fn check_moment_slots(&self, slots: &[SlotMoments]) -> Result<()> {
        if !slots.is_empty() && slots.len() != self.slots.len() {
            let message = format!(
                "矩状态槽位数不符：快照 {} 槽，会话 {} 槽",
                slots.len(),
                self.slots.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        for (index, slot) in slots.iter().enumerate() {
            self.check_moment_slot_length(index, slot)?;
        }
        Ok(())
    }

    /// 单槽矩长度对账：一阶/二阶矩元素数必须等于会话槽参数元素数。
    fn check_moment_slot_length(&self, index: usize, slot: &SlotMoments) -> Result<()> {
        let want = self.slots[index].params().len();
        if slot.m().len() != want || slot.v().len() != want {
            let message = format!(
                "第 {index} 槽矩长度不符：快照 {} 元素，会话 {} 元素",
                slot.m().len(),
                want
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        Ok(())
    }

    /// 累积窗口回载：槽位数对齐；空梯度表表示该槽窗口内未装入，跳过不覆盖。
    fn restore_accum(&mut self, accum: &GradAccum) -> Result<()> {
        let slots = accum.slots();
        self.check_accum_slot_count(slots)?;
        for (index, grads) in slots.iter().enumerate() {
            self.load_accum_slot(index, grads)?;
        }
        self.accum = accum.count();
        Ok(())
    }

    /// 累积表槽位数对账。
    fn check_accum_slot_count(&self, slots: &[Vec<f32>]) -> Result<()> {
        if slots.len() != self.slots.len() {
            let message = format!(
                "累积梯度槽位数不符：快照 {} 槽，会话 {} 槽",
                slots.len(),
                self.slots.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        Ok(())
    }

    /// 单槽累积梯度回载：空表跳过；非空长度必须等于槽参数元素数，再装入缓冲。
    fn load_accum_slot(&mut self, index: usize, grads: &[f32]) -> Result<()> {
        if grads.is_empty() {
            return Ok(());
        }
        let want = self.slots[index].params().len();
        if grads.len() != want {
            let message = format!(
                "第 {index} 槽累积梯度长度不符：快照 {} 元素，会话 {} 元素",
                grads.len(),
                want
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        self.buffers
            .load_gradient(self.slots[index].var(), grads)
            .map_err(|error| error.with(SITE))?;
        Ok(())
    }

    /// 校验绑定两端参数槽长度一致；不符是 [`Code::Invalid`]。
    pub(super) fn check_tie_length(&self, source: usize, tied: usize) -> Result<()> {
        if self.slots[source].params().len() == self.slots[tied].params().len() {
            return Ok(());
        }
        let message = format!(
            "绑定两端长度不一致：源槽 {} 元素，回填槽 {} 元素",
            self.slots[source].params().len(),
            self.slots[tied].params().len()
        );
        Err(Error::new(Code::Invalid, message).with(SITE))
    }

    /// 绑定表回载：编号必须落在会话槽位内，两端长度一致；回载替换既有绑定表。
    fn restore_ties(&mut self, ties: &Ties) -> Result<()> {
        let slot_count = self.slots.len();
        for &(source, tied) in ties.pairs() {
            if source >= slot_count || tied >= slot_count {
                let message =
                    format!("绑定对 ({source}, {tied}) 编号越界：会话只有 {slot_count} 个参数槽");
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
            self.check_tie_length(source, tied)?;
        }
        self.ties = ties.pairs().to_vec();
        Ok(())
    }
}
