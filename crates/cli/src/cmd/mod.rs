//! 命令编排：把解析后的 [`crate::Invocation`] 接到各层库，装配、取值解析与人可读排版。
//!
//! 解析在本 crate 的文法层（[`crate::Command`] / [`crate::settings`]），业务实现与判定在各内核库；
//! 本模块只做"跑哪些层、用什么数据、结果怎么排版"，不定义新类型。

mod run;
mod status;
mod train;

pub(crate) use run::run;
pub(crate) use status::status;
pub(crate) use train::train_run;
