//! 片段层的声明式用例套：九条判据 × 每台设备一遍，装配结果只构建一次。
//!
//! 判据只产出 `Result`，失败统一发生在生成的 `#[test]` 中；设备模块只提供生成表，
//! 不复制判据，也不能替换 P0 闭集。

/// 把一台设备的片段层判据摊成独立 `#[test]` 函数。
#[macro_export]
macro_rules! device_suite {
    () => {
        use ::llm_prof::prof::{Profile, builtin_profiles};

        /// 本设备全部测试共享的只读装配；九条判据不再重复解析档案和分配三张 Vec。
        struct Fixture {
            fragments: Vec<(&'static str, &'static str)>,
            prims: Vec<(&'static str, &'static str)>,
            provides: Vec<$crate::device::Provision<'static>>,
        }

        /// 档案只解析一次，其静态引用再供装配表借用，避免自引用结构。
        fn profile() -> Result<&'static Profile, String> {
            static PROFILE: ::std::sync::OnceLock<Result<Profile, String>> =
                ::std::sync::OnceLock::new();
            PROFILE.get_or_init(|| {
                let wanted = DEVICE.vendor;
                builtin_profiles()
                    .map_err(|error| error.to_string())?
                    .iter()
                    .find(|profile| profile.meta.vendor == wanted)
                    .cloned()
                    .ok_or_else(|| format!("档案里没有厂商 `{wanted}`"))
            }).as_ref().map_err(Clone::clone)
        }

        /// 从静态档案与生成表构建一次夹具。
        fn build_fixture() -> Result<Fixture, String> {
            let profile = profile()?;
            let fragments = FRAGMENTS.iter().map(|(prim, text)| (prim.as_str(), *text)).collect();
            let prims = PRIMS.iter().map(|(prim, func)| (prim.as_str(), *func)).collect();
            let provides = profile
                .provides
                .iter()
                .map(|provide| $crate::device::Provision {
                    prim: provide.prim.as_str(),
                    impl_name: provide.impl_name.as_str(),
                    witness: provide.witness.as_ref().map(|witness| $crate::device::Witness {
                        func: witness.func.as_str(),
                        expect: &witness.expect,
                    }),
                })
                .collect();
            Ok(Fixture { fragments, prims, provides })
        }

        /// 每台设备一个缓存；测试并行时 `OnceLock` 保证装配恰好一次。
        fn fixture() -> Result<&'static Fixture, String> {
            static FIXTURE: ::std::sync::OnceLock<Result<Fixture, String>> =
                ::std::sync::OnceLock::new();
            FIXTURE.get_or_init(build_fixture).as_ref().map_err(Clone::clone)
        }

        /// 在共享夹具上运行一条判据。
        fn with_contract<T>(
            body: impl FnOnce(&$crate::device::Contract<'_>) -> T,
        ) -> Result<T, String> {
            let fixture = fixture()?;
            let profile = profile()?;
            Ok(body(&$crate::device::Contract {
                id: &profile.meta.id,
                ext: DEVICE.ext,
                dir: ROOT,
                p0: &$crate::device::P0,
                fragments: &fixture.fragments,
                prims: &fixture.prims,
                provides: &fixture.provides,
                epilogue_stem: ::llm_prof::stems::EPILOGUE_STEM,
                source: DEVICE.source,
            }))
        }

        /// 断言一条判据，并保留缺陷文本。
        fn judged(
            check: impl FnOnce(&$crate::device::Contract<'_>) -> Result<(), String>,
            claim: &str,
        ) {
            let verdict = with_contract(check);
            assert!(verdict.as_ref().is_ok_and(|inner| inner.is_ok()), "{claim}：{verdict:?}");
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

        $crate::test_cases!(contract_checks {
            the_fragment_layer_is_exactly_the_p0_set =>
                ($crate::device::Contract::check_p0_set, "片段层必须恰好是 P0 闭集"),
            every_signature_is_a_profile_provide_that_has_a_witness =>
                ($crate::device::Contract::check_signatures, "每条签名必须是一条有见证的供给"),
            every_witness_function_is_claimed_by_exactly_one_entry =>
                ($crate::device::Contract::check_witness_entries, "每个见证函数必须恰好被一条入口认领"),
            every_provision_rests_on_a_fragment_or_a_witness =>
                ($crate::device::Contract::check_evidence, "每条供给必须有片段正文或见证"),
            the_fragment_directory_is_a_closed_set =>
                ($crate::device::Contract::check_closed_directory, "片段目录必须是闭集"),
            every_expectation_fits_the_readback_buf =>
                ($crate::device::Contract::check_expectations, "期望值必须放得进回读缓冲"),
            the_witness_compiles_wherever_it_is_copied_to =>
                ($crate::device::Contract::check_self_contained, "见证必须自足"),
            every_witness_computes_what_the_profile_expects =>
                ($crate::device::Contract::check_witnesses, "每条见证必须算出档案写的值"),
        }, |(check, claim)| judged(check, claim));

        #[test]
        fn the_device_row_is_the_profile_it_was_generated_from() {
            let verdict = profile().and_then(row);
            assert!(verdict.is_ok(), "生成表的设备行必须就是档案那一行：{verdict:?}");
        }
    };
}
