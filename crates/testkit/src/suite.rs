//! 片段层的用例套：九条判据 × 每台设备一遍。
//!
//! `rocm` 与 `opencl` 的用例正文逐字相同，只有 `DEVICE` / `FRAGMENTS` / `PRIMS` / `ROOT`
//! 与 P0 闭集不同——逐字复制两份正是 N7 要消掉的同构。于是套子只在这里写一次，设备 crate
//! 用一个宏把它摊成 `#[test]` 函数：
//!
//! ```ignore
//! use super::*;
//! testkit::device_suite!(P0);
//! ```
//!
//! 断言用 `assert!` 而不是 `expect`：判据本身只**产出缺陷**（`Result<(), String>`），失败
//! 发生在测试里——`src/` 上的静态门禁不给 `panic`/`expect` 留预算，宏体也是 `src/`。
//!
//! 语言门不在这里：`cc` 编得过一份源码只说明它是合法 C，不说明它是该设备语言。各自的
//! 设备前端（`ccec` / `clang -cl-std`）由 `cutile::gate` 与设备 crate 自己的用例管。

/// 把一台设备的片段层判据摊成 `#[test]` 函数。
#[macro_export]
macro_rules! device_suite {
    ($p0:expr) => {
        use prof::prof::{Profile, builtin_profiles};

        /// 档案里与 `DEVICE.vendor` 同名的那台设备。
        fn profile() -> Result<Profile, String> {
            let wanted = DEVICE.vendor;
            builtin_profiles()
                .iter()
                .find(|profile| profile.meta.vendor == wanted)
                .cloned()
                .ok_or_else(|| format!("档案里没有厂商 `{wanted}`"))
        }

        /// 在契约上跑一条判据。`ext` 取 `DEVICE.ext` 而不是档案的 `emit`——两处不相等由
        /// 设备行那条用例单独指出，其余用例只在一个口径上判定。
        fn with_contract<T>(
            body: impl FnOnce(&$crate::device::Contract<'_>) -> T,
        ) -> Result<T, String> {
            let profile = profile()?;
            let fragments: Vec<(&str, &str)> = FRAGMENTS
                .iter()
                .map(|(prim, text)| (prim.as_str(), *text))
                .collect();
            let prims: Vec<(&str, &str)> = PRIMS
                .iter()
                .map(|(prim, func)| (prim.as_str(), *func))
                .collect();
            let provides: Vec<$crate::device::Provision<'_>> = profile
                .provides
                .iter()
                .map(|provide| $crate::device::Provision {
                    prim: provide.prim.as_str(),
                    impl_name: provide.impl_name.as_str(),
                    witness: provide
                        .witness
                        .as_ref()
                        .map(|witness| $crate::device::Witness {
                            func: witness.func.as_str(),
                            expect: &witness.expect,
                        }),
                })
                .collect();
            Ok(body(&$crate::device::Contract {
                id: &profile.meta.id,
                ext: DEVICE.ext,
                dir: ROOT,
                p0: &$p0,
                fragments: &fragments,
                prims: &prims,
                provides: &provides,
                epilogue_stem: prof::sign::EPILOGUE_STEM,
                source: DEVICE.source,
            }))
        }

        /// 断言一条判据：失败信息带上缺陷本身，而不是"断言失败"。
        fn judged(
            check: impl FnOnce(&$crate::device::Contract<'_>) -> Result<(), String>,
            claim: &str,
        ) {
            let verdict = with_contract(check);
            assert!(
                verdict.as_ref().is_ok_and(|inner| inner.is_ok()),
                "{claim}：{verdict:?}"
            );
        }

        /// 设备行与档案行必须逐项相等：五项元数据加一份入口表。
        fn row(profile: &Profile) -> Result<(), String> {
            let pairs = [
                ("厂商键", DEVICE.vendor, profile.meta.vendor.as_str()),
                ("输出语言名", DEVICE.emit, profile.meta.emit.as_str()),
                ("编译工具", DEVICE.tool, profile.meta.tool.as_str()),
                ("工具版本", DEVICE.tool_ver, profile.meta.tool_ver.as_str()),
                ("片段扩展名", DEVICE.ext, profile.meta.emit.ext()),
            ];
            for (claim, got, want) in pairs {
                if got != want {
                    return Err(format!("{claim}：设备行写 `{got}`，档案写 `{want}`"));
                }
            }
            if DEVICE.prims == &PRIMS[..] {
                return Ok(());
            }
            Err(String::from("设备行里的原语入口必须就是生成表那份"))
        }

        #[test]
        fn the_fragment_layer_is_exactly_the_p0_set() {
            judged(
                |contract| contract.check_p0_set(),
                "片段层必须恰好是 P0 闭集",
            );
        }

        #[test]
        fn every_signature_is_a_profile_provide_that_has_a_witness() {
            judged(
                |contract| contract.check_signatures(),
                "每条签名必须是一条有见证的供给",
            );
        }

        #[test]
        fn every_witness_function_is_claimed_by_exactly_one_entry() {
            judged(
                |contract| contract.check_witness_entries(),
                "每个见证函数必须恰好被一条入口认领",
            );
        }

        #[test]
        fn every_provision_rests_on_a_fragment_or_a_witness() {
            judged(
                |contract| contract.check_evidence(),
                "每条供给必须有片段正文或见证",
            );
        }

        #[test]
        fn the_fragment_directory_is_a_closed_set() {
            judged(
                |contract| contract.check_closed_directory(),
                "片段目录必须是闭集",
            );
        }

        #[test]
        fn every_expectation_fits_the_readback_buffer() {
            judged(
                |contract| contract.check_expectations(),
                "期望值必须放得进回读缓冲",
            );
        }

        #[test]
        fn the_witness_compiles_wherever_it_is_copied_to() {
            judged(|contract| contract.check_self_contained(), "见证必须自足");
        }

        #[test]
        fn every_witness_computes_what_the_profile_expects() {
            judged(
                |contract| contract.check_witnesses(),
                "每条见证必须算出档案写的值",
            );
        }

        #[test]
        fn the_device_row_is_the_profile_it_was_generated_from() {
            let verdict = profile().and_then(|profile| row(&profile));
            assert!(
                verdict.is_ok(),
                "生成表的设备行必须就是档案那一行：{verdict:?}"
            );
        }
    };
}
