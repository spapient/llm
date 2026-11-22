//! 见证的可执行判据：源码自足、期望值上限、本机执行对拍。

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
        let prims = self.provides.iter().filter_map(|provide| {
            provide
                .witness
                .map(|witness| (provide.prim, provide.impl_name, witness.expect))
        });
        let epis = self
            .epilogue
            .iter()
            .map(|entry| (entry.form, entry.impl_name, entry.witness.expect));
        prims
            .chain(epis)
            .try_for_each(|(kind, name, expect)| self.check_length(kind, name, expect))
    }

    /// 期望值的个数就是见证函数的返回值，执行器按 `MAX_OUT` 定缓冲。
    fn check_length(&self, kind: &str, name: &str, expect: &[f32]) -> Result<(), String> {
        let len = expect.len();
        if (1..=MAX_OUT).contains(&len) {
            return Ok(());
        }
        Err(format!(
            "`{}` 的 `{kind}/{name}` 期望值有 {len} 个：见证必须写 1..={MAX_OUT} 个元素",
            self.id
        ))
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

    /// 档案里带见证的供给与写回形式，摊成执行器要的 `Call`。写回形式同样要算出档案写的值：
    /// 六种形式只是"没有可协商的签名"，不是"可以不验"。
    fn calls(&self) -> Vec<Call<'a>> {
        let prims = self.provides.iter().filter_map(|provide| provide.witness);
        let epis = self.epilogue.iter().map(|entry| entry.witness);
        prims.chain(epis).map(call).collect()
    }
}

/// 一条见证摊成执行器要的 `Call`。
fn call(witness: crate::device::Witness<'_>) -> Call<'_> {
    Call {
        func: witness.func,
        expect: witness.expect,
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

/// 见证文件里的 `int witness_*(float *out)` 定义名。注释里那种 `int witness_<name>(...)` 因为
/// 名字里带着 `<`，在这里被过滤掉。
pub(crate) fn definitions(source: &str) -> Vec<&str> {
    source
        .match_indices("int ")
        .filter_map(|(index, _)| defined_name(&source[index + 4..]))
        .collect()
}

/// 从 `int ` 之后的那段正文里读出函数名：只认 `witness_*` 且参数表里有 `float *out` 的定义。
fn defined_name(rest: &str) -> Option<&str> {
    let (name, args) = rest.split_once('(')?;
    let args = args.split_once(')')?.0;
    (is_witness_name(name) && args.contains("float *out")).then_some(name)
}

/// 见证入口的命名形状：`witness_` 前缀，其余全是标识符字符（于是 `witness_<name>` 不算）。
fn is_witness_name(name: &str) -> bool {
    name.strip_prefix("witness_").is_some_and(|rest| {
        !rest.is_empty() && rest.chars().all(|c| c.is_alphanumeric() || c == '_')
    })
}

#[cfg(test)]
#[path = "../tests/evidence_tests.rs"]
mod evidence_tests;
