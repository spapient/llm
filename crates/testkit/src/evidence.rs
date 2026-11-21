//! 见证的可执行判据：源码自足、期望值上限、本机执行对拍。
//!
//! 与 `device` 分开，是因为两层的判据不同：`device` 判"写得对不对"（签名、目录、供给），
//! 这里判"算得对不对"（编出来、跑起来、与 `expect` 对上）。`impl` 落在另一个文件里仍然是
//! 同一个类型，于是新增设备只碰数据，两边的判据都不动。
//!
//! 与 `device` 一样只产出缺陷、不自己失败；唯一的例外是"本机没有编译器"，那是**跳过**
//! 而不是通过——跳过进进程内计数，通过会让红变绿。

use crate::device::Contract;
use crate::witness::{Call, MAX_OUT, Reading, run};

/// 见证对拍的容差，与 `cutile::witness` 同一个口径。
pub const TOLERANCE: f64 = 1e-5;

impl<'a> Contract<'a> {
    /// 见证源码必须自足。夹具把它复制到临时目录里编译，`#include "..."` 在那份副本旁边解析
    /// 不到——引号包含会在设备目录里静默成功、在宿主执行时静默失败；尖括号里的标准头不受影响。
    ///
    /// # Errors
    ///
    /// 第一条引号包含返回原因（带行号）。
    pub fn check_self_contained(&self) -> Result<(), String> {
        for (index, line) in self.source.lines().enumerate() {
            let Some((_, rest)) = line.split_once("#include") else {
                continue;
            };
            let target = rest.trim();
            if target.starts_with('"') {
                return Err(format!(
                    "`{}` 的见证第 {} 行用引号包含了 `{target}`：副本编译时相对路径不存在",
                    self.id,
                    index + 1
                ));
            }
        }
        Ok(())
    }

    /// 期望值的长度就是见证函数的返回值，执行器按 `MAX_OUT` 定缓冲：超了就是读数溢出。
    ///
    /// # Errors
    ///
    /// 第一条期望值个数不在 `1..=MAX_OUT` 的供给返回原因。
    pub fn check_expectations(&self) -> Result<(), String> {
        for provide in self.provides {
            let Some(witness) = provide.witness else {
                continue;
            };
            let len = witness.expect.len();
            if !(1..=MAX_OUT).contains(&len) {
                return Err(format!(
                    "`{}` 的 `{}/{}` 期望值有 {len} 个：见证必须写 1..={MAX_OUT} 个元素",
                    self.id, provide.prim, provide.impl_name
                ));
            }
        }
        Ok(())
    }

    /// 每条供给的见证都要在本机算出档案写的值。算不出来就是"这条供给的算法是编的"。
    ///
    /// 本机没有编译器时登记跳过并返回 `Ok`（跳过会进进程内计数，通过会让红变绿）。
    ///
    /// # Errors
    ///
    /// 见证源码编不过、程序非零退出、读数比供给少、或某条读数超出 [`TOLERANCE`] 时返回原因。
    pub fn check_witnesses(&self) -> Result<(), String> {
        let calls = self.calls();
        let Some(readings) = execute(self, &calls)? else {
            return Ok(());
        };
        judge(self.id, &calls, &readings)
    }

    /// 档案里带见证的供给，摊成执行器要的 `Call`。
    fn calls(&self) -> Vec<Call<'a>> {
        let provides: &'a [crate::device::Provision<'a>] = self.provides;
        provides
            .iter()
            .filter_map(|provide| {
                provide.witness.map(|witness| Call {
                    func: witness.func,
                    expect: witness.expect,
                })
            })
            .collect()
    }
}

/// 编译并执行一遍见证；`Ok(None)` 表示本机没有编译器（已登记跳过）。
fn execute(contract: &Contract<'_>, calls: &[Call<'_>]) -> Result<Option<Vec<Reading>>, String> {
    match run(contract.source, contract.ext, calls) {
        Ok(readings) => Ok(Some(readings)),
        Err(error) if error.contains("不可执行") => {
            Ok(crate::gating::skip_none(format!("本机没有编译器：{error}")))
        }
        Err(error) => Err(format!(
            "`{}` 的见证源码过不了本机编译器：{error}",
            contract.id
        )),
    }
}

/// 读数与期望的对拍：条数先对齐，再逐条比相对误差。
fn judge(id: &str, calls: &[Call<'_>], readings: &[Reading]) -> Result<(), String> {
    if readings.len() != calls.len() {
        return Err(format!(
            "`{id}` 回读到的读数比供给少：{} < {}",
            readings.len(),
            calls.len()
        ));
    }
    for reading in readings {
        if reading.max_rel_err <= TOLERANCE {
            continue;
        }
        return Err(format!(
            "`{id}` 的见证 `{}` 不成立：得到 {:?}，期望 {:?}，max_rel_err {:.3e} > {TOLERANCE:.0e}",
            reading.func,
            reading.values,
            expected(calls, &reading.func),
            reading.max_rel_err
        ));
    }
    Ok(())
}

/// 某条见证在档案里登记的期望值；找不到时是空切片（对拍已经证明有问题，这里只求可读）。
fn expected<'a>(calls: &'a [Call<'a>], func: &str) -> &'a [f32] {
    calls
        .iter()
        .find(|call| call.func == func)
        .map_or(&[], |call| call.expect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Contract, Provision, Witness};

    const SIGNED: &str = "/* cutile-sign: load/scalar_copy */\n$0 = $1;";
    const SOURCE: &str = "int witness_load(float *out) { out[0] = 2.0f; return 1; }\n";

    fn witness<'a>(expect: &'a [f32]) -> Provision<'a> {
        Provision {
            prim: "load",
            impl_name: "scalar_copy",
            witness: Some(Witness {
                func: "witness_load",
                expect,
            }),
        }
    }

    /// 一份最小探针：一条 `load/scalar_copy` 供给，签名、入口、见证都齐。
    fn probe<'a>(provides: &'a [Provision<'a>], source: &'a str) -> Contract<'a> {
        Contract {
            id: "probe",
            ext: "c",
            dir: ".",
            p0: &["load"],
            fragments: &[("load", SIGNED)],
            prims: &[("load", "witness_load")],
            provides,
            source,
        }
    }

    #[test]
    fn a_self_contained_witness_is_accepted_but_a_quoted_include_is_not() {
        let good = [witness(&[2.0])];
        assert_eq!(probe(&good, SOURCE).check_self_contained(), Ok(()));
        assert_eq!(
            probe(&good, "#include <math.h>\n").check_self_contained(),
            Ok(()),
            "尖括号里的标准头不破坏自足"
        );
        let defect = probe(&good, "#include \"preamble.h\"\n").check_self_contained();
        assert!(
            defect
                .as_ref()
                .is_err_and(|reason| reason.contains("引号包含")),
            "引号包含在副本旁边解析不到，必须被发现：{defect:?}"
        );
    }

    #[test]
    fn expectations_must_fit_the_reading_buffer() {
        let good = [witness(&[2.0])];
        assert_eq!(probe(&good, SOURCE).check_expectations(), Ok(()));
        let overflow = [2.0; MAX_OUT + 1];
        let loud = [witness(&overflow)];
        let defect = probe(&loud, SOURCE).check_expectations();
        assert!(
            defect
                .as_ref()
                .is_err_and(|reason| reason.contains("期望值")),
            "超过读数缓冲的期望值必须被发现：{defect:?}"
        );
    }

    #[test]
    fn a_witness_that_computes_something_else_is_a_defect() {
        let good = [witness(&[2.0])];
        assert_eq!(probe(&good, SOURCE).check_witnesses(), Ok(()));
        let wrong = [witness(&[3.0])];
        let defect = probe(&wrong, SOURCE).check_witnesses();
        assert!(
            defect
                .as_ref()
                .is_err_and(|reason| reason.contains("不成立")),
            "算错值的见证必须被发现：{defect:?}"
        );
    }
}
