//! 回归套件的子模块装配：登记表 / 主机档案 / 分析件 / 记录 / 归档 / 用例。

pub(crate) use std::path::{Path, PathBuf};
pub(crate) use tool::bench::{
    BenchRecord, EntryKind, Geometry, Harness, Roofline as RecordRoofline,
};
pub(crate) use tool::regress::{Case, Direction, Judgement};

mod analysis;
mod archive;
mod cases;
mod host;
mod record;
mod table;

#[path = "../src/fixtures.rs"]
#[allow(dead_code)]
mod fixtures;
#[path = "../src/protocol.rs"]
mod protocol;

pub(crate) use analysis::*;
pub(crate) use archive::*;
pub(crate) use host::*;
pub(crate) use record::*;
pub(crate) use table::*;
