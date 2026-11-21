//! 见证的可执行判据：期望值上限、本机执行对拍、源码自足。
//!
//! 与 `device` 分开，是因为两层的判据不同：`device` 判"写得对不对"（签名、目录、供给），
//! 这里判"算得对不对"（编出来、跑起来、与 `expect` 对上）。`impl` 落在另一个文件里仍然是
//! 同一个类型，于是新增设备只碰数据，两边的判据都不动。

use crate::device::Contract;
use crate::witness::{Call, MAX_OUT, run};

/// 见证对拍的容差，与 `cutile::witness` 同一个口径。
pub const TOLERANCE: f64 = 1e-5;

impl<'a> Contract<'a> {
    /// 见证源码必须自足。夹具把它复制到临时目录里编译，`#include "..."` 在那份副本旁边解析
    /// 不到——引号包含会在设备目录里静默成功、在宿主执行时静默失败；尖括号里的标准头不受影响。
    pub fn assert_self_contained(&self) {
        for (index, line) in self.source.lines().enumerate() {
            let Some((_, rest)) = line.split_once("#include") else {
                continue;
            };
            let target = rest.trim();
            assert!(
                !target.starts_with('"'),
                "`{}` 的见证第 {} 行用引号包含了 `{target}`：副本编译时相对路径不存在",
                self.id,
                index + 1
            );
        }
    }

    /// 期望值的长度就是见证函数的返回值，执行器按 `MAX_OUT` 定缓冲：超了就是读数溢出。
    pub fn assert_expectations(&self) {
        for provide in self.provides {
            let Some(witness) = provide.witness else {
                continue;
            };
            assert!(
                (1..=MAX_OUT).contains(&witness.expect.len()),
                "`{}` 的 `{}/{}` 期望值有 {} 个：见证必须写 1..={MAX_OUT} 个元素",
                self.id,
                provide.prim,
                provide.impl_name,
                witness.expect.len()
            );
        }
    }

    /// 每条供给的见证都要在本机算出档案写的值。算不出来就是"这条供给的算法是编的"。
    ///
    /// 本机没有编译器时登记跳过而不是通过：跳过会进进程内计数，通过会让红变绿。
    pub fn assert_witnesses(&self) {
        let calls: Vec<Call<'_>> = self
            .provides
            .iter()
            .filter_map(|provide| {
                provide.witness.map(|witness| Call {
                    func: witness.func,
                    expect: witness.expect,
                })
            })
            .collect();
        let readings = match run(self.source, self.ext, &calls) {
            Ok(readings) => readings,
            Err(error) if error.contains("不可执行") => {
                assert!(crate::gating::skip(format!("本机没有编译器：{error}")));
                return;
            }
            Err(error) => panic!("`{}` 的见证源码过不了本机编译器：{error}", self.id),
        };
        assert_eq!(
            readings.len(),
            calls.len(),
            "`{}` 回读到的读数比供给少",
            self.id
        );
        for reading in &readings {
            assert!(
                reading.max_rel_err <= TOLERANCE,
                "`{}` 的见证 `{}` 不成立：得到 {:?}，期望 {:?}，max_rel_err {:.3e} > {TOLERANCE:.0e}",
                self.id,
                reading.func,
                reading.values,
                calls
                    .iter()
                    .find(|call| call.func == reading.func)
                    .map(|call| call.expect)
                    .unwrap_or_default(),
                reading.max_rel_err
            );
        }
    }
}
