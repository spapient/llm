//! 职责：优化器视图装配——参数/梯度按槽整块取走、步进后整块归还与镜像。
//!
//! 梯度向量按槽整块取走（[`Buffers::take_gradient`]），步进结束后整块归还；
//! 参数槽与梯度表因此各只建立一条可变借用。

use diagnostic::{Code, Error, Result};
use shard::Collective;

use crate::optimizer::{Optimizer, OwnedSlot};
use crate::pool::Buffers;
use crate::{Data, Kernel};

use super::{ParamSlot, SITE, Session, SlotMeta};

impl<K: Kernel, O: Optimizer, C: Collective> Session<K, O, C> {
    /// 汇总各槽的参数/梯度可变视图；任一槽梯度未装入即报错。
    ///
    /// 梯度向量按槽整块取走（[`Buffers::take_gradient`]），步进结束后在
    /// [`Self::return_views`] 归还；参数槽与梯度表因此各只建立一条可变借用。
    pub(super) fn collect_views(
        slots: &mut [ParamSlot],
        buffers: &mut Buffers<K>,
    ) -> Result<Vec<OwnedSlot>> {
        let meta: Vec<SlotMeta> = slots
            .iter()
            .map(|slot| (slot.var(), slot.role(), slot.matrix()))
            .collect();
        let mut views = Vec::with_capacity(meta.len());
        for (index, (var, role, shape)) in meta.into_iter().enumerate() {
            let Some(grads) = buffers.take_gradient(var) else {
                // 中途归还已取走的梯度：失败路径不留悬空向量。
                Self::return_views_prefix(slots, buffers, views);
                return Err(Error::new(
                    Code::Missing,
                    format!("参数 {} 的梯度还没装入缓冲", var.id()),
                )
                .with(SITE));
            };
            let params = slots[index].params().to_vec();
            views.push(OwnedSlot {
                params,
                grads,
                role,
                shape,
            });
        }
        Ok(views)
    }

    /// 把 [`Self::collect_views`] 取走的视图按槽号归位：更新后的参数真值回填进槽，梯度向量
    /// 归还缓冲；是否清空梯度由步进收尾决定。
    pub(super) fn return_views(
        slots: &mut [ParamSlot],
        buffers: &mut Buffers<K>,
        views: Vec<OwnedSlot>,
    ) {
        for (slot, view) in slots.iter_mut().zip(views) {
            slot.params.copy_from_slice(&view.params);
            buffers.put_back_gradient(slot.var(), view.grads);
        }
    }

    /// 收集视图失败时的局部归还：已装配的前 `views.len()` 个槽对应 `slots` 前缀。
    fn return_views_prefix(
        slots: &mut [ParamSlot],
        buffers: &mut Buffers<K>,
        views: Vec<OwnedSlot>,
    ) {
        Self::return_views(&mut slots[..views.len()], buffers, views);
    }

    /// 把优化器写回后的参数值镜像回激活缓冲，供下一轮前向复用。
    pub(super) fn mirror_params(slots: &[ParamSlot], buffers: &mut Buffers<K>) -> Result<()> {
        for slot in slots {
            buffers
                .load(slot.var(), Data::F32(slot.params.clone()))
                .map_err(|error| error.with(SITE))?;
        }
        Ok(())
    }
}
