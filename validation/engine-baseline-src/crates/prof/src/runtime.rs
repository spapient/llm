//! 设备运行时后端岛：提交契约、FFI 句柄、命令队列、缓冲、管线与执行器实现收在这里，不进上层编排。

#[path = "exec.rs"]
pub mod exec;
#[cfg(target_os = "macos")]
#[path = "metal/runtime.rs"]
pub mod metal;
#[cfg(windows)]
#[path = "opencl.rs"]
pub mod opencl;

/// `gemm_micro.k` 的安全静态 ABI 桥；所有裸指针都受切片长度、坐标、溢出与特性门约束。
#[allow(unsafe_code)]
pub mod gemm_micro {
    use super::exec::{add, area, c_int};
    use diagnostic::{Code, Error, Result};

    include!(concat!(env!("OUT_DIR"), "/gemm_micro_contract.rs"));

    type Sizes = (usize, usize, usize);
    type FmaFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32, i32, i32, i32);

    fn invalid<T>(message: impl Into<String>) -> Result<T> {
        Err(Error::new(Code::Invalid, message.into()).with("prof::runtime::gemm_micro"))
    }

    /// 三份缓冲的长度校验只有一个实现；元组顺序固定为 A／B／输出。
    fn ensure(need: Sizes, got: Sizes, what: &str) -> Result<()> {
        if got.0 < need.0 || got.1 < need.1 || got.2 < need.2 {
            return invalid(format!("{what} 面板或输出缓冲不足"));
        }
        Ok(())
    }

    fn position(row: usize, col: usize, mr: usize, nr: usize) -> Result<(usize, usize)> {
        Ok((add(row, mr, "行坐标")?, add(col, nr, "列坐标")?))
    }

    /// A、B、输出的元素数；调用方分别给出两种面板的行数。
    fn sizes(a_rows: usize, b_rows: usize, n: usize, k: usize) -> Result<Sizes> {
        Ok((
            area(a_rows, k, "A")?,
            area(b_rows, k, "B")?,
            area(a_rows, n, "输出")?,
        ))
    }

    fn c_args<const N: usize>(values: [usize; N]) -> Result<[i32; N]> {
        let mut out = [0; N];
        for (dst, value) in out.iter_mut().zip(values) {
            *dst = c_int(value, "ABI 参数")?;
        }
        Ok(out)
    }

    fn fma_needs(tile: (usize, usize, usize, usize)) -> Result<(Sizes, usize)> {
        let (row, col, n, kb) = tile;
        let (rows, cols) = position(row, col, FMA_MR, FMA_NR)?;
        let needs = (
            area(kb, FMA_MR, "FMA A")?,
            area(kb, FMA_NR, "FMA B")?,
            area(rows, n, "FMA 输出")?,
        );
        Ok((needs, cols))
    }

    /// FMA 单块：元组依次是行、列、输出列宽与本次 K 长度。
    pub fn fma_micro(
        panel: &[f32],
        b: &[f32],
        out: &mut [f32],
        tile: (usize, usize, usize, usize),
        first: bool,
    ) -> Result<()> {
        let (need, cols) = fma_needs(tile)?;
        if cols > tile.2 {
            return invalid("FMA 列坐标越界");
        }
        ensure(need, (panel.len(), b.len(), out.len()), "FMA")?;
        dispatch_fma(
            panel,
            b,
            out,
            c_args([tile.0, tile.1, tile.2, tile.3])?,
            first,
        );
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    fn fma_isa() -> FmaFn {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return cutile_fma_micro_scalar;
        }
        #[cfg(target_env = "msvc")]
        {
            cutile_fma_micro_avx2
        }
        #[cfg(not(target_env = "msvc"))]
        {
            cutile_fma_micro
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn fma_isa() -> FmaFn {
        cutile_fma_micro
    }

    fn dispatch_fma(a: &[f32], b: &[f32], out: &mut [f32], args: [i32; 4], first: bool) {
        unsafe {
            fma_isa()(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                args[0],
                args[1],
                args[2],
                args[3],
                i32::from(first),
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn sdot_available() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    fn sdot_available() -> bool {
        false
    }

    fn sdot_ready(g: (usize, usize, usize, usize, usize)) -> bool {
        sdot_available() && g.4 >= 16 && g.4 <= g.3 && g.4.is_multiple_of(16)
    }

    fn checked_sdot(
        a: &[i8],
        b: &[i8],
        out: &[i32],
        g: (usize, usize, usize, usize, usize),
    ) -> Result<[i32; 5]> {
        let (rows, cols) = position(g.0, g.1, SDOT_MR, SDOT_NR)?;
        ensure(
            sizes(rows, cols, g.2, g.3)?,
            (a.len(), b.len(), out.len()),
            "sdot",
        )?;
        c_args([g.0, g.1, g.2, g.3, g.4])
    }

    /// sdot 单块：元组依次是行、列块、输出列宽、K 与向量化 K 长度。
    pub fn sdot_micro(
        a: &[i8],
        b: &[i8],
        out: &mut [i32],
        g: (usize, usize, usize, usize, usize),
    ) -> Result<bool> {
        if !sdot_ready(g) {
            return Ok(false);
        }
        let args = checked_sdot(a, b, out, g)?;
        unsafe {
            cutile_sdot_micro(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
            );
        }
        Ok(true)
    }

    #[cfg(all(nightly, target_arch = "aarch64"))]
    fn sme_available() -> bool {
        std::arch::is_aarch64_feature_detected!("sme")
    }

    /// Apple stable 尚无 SME 探测宏；只读内核发布的 FEAT_SME 能力位。
    #[cfg(all(not(nightly), target_os = "macos", target_arch = "aarch64"))]
    fn sme_available() -> bool {
        use core::ffi::{c_int, c_void};
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const core::ffi::c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *mut c_void,
                newlen: usize,
            ) -> c_int;
        }
        let mut supported = 0u32;
        let mut len = size_of::<u32>();
        unsafe {
            sysctlbyname(
                c"hw.optional.arm.FEAT_SME".as_ptr(),
                (&mut supported as *mut u32).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            ) == 0
                && len == size_of::<u32>()
                && supported != 0
        }
    }

    #[cfg(any(
        not(target_arch = "aarch64"),
        all(not(nightly), not(target_os = "macos"))
    ))]
    fn sme_available() -> bool {
        false
    }

    /// SME 单瓦片对拍；无设备时明确返回未执行。
    pub fn sme_micro(a: &[f32], b: &[f32], out: &mut [f32], n: usize, k: usize) -> Result<bool> {
        if k == 0 || n < SME_NR || !sme_available() {
            return Ok(false);
        }
        checked_sme_micro(a, b, out, n, k)
    }

    fn checked_sme_micro(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        n: usize,
        k: usize,
    ) -> Result<bool> {
        ensure(
            sizes(SME_MR, n, n, k)?,
            (a.len(), b.len(), out.len()),
            "SME",
        )?;
        unsafe {
            cutile_sme_micro(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, k);
        }
        Ok(true)
    }

    fn sme_tiled(m: usize, n: usize, k: usize) -> bool {
        m > 0 && n > 0 && k > 0 && m.is_multiple_of(SME_MR) && n.is_multiple_of(SME_NR)
    }

    fn sme_needs(m: usize, n: usize, k: usize) -> Result<(Sizes, bool)> {
        let need = sizes(m, n, n, k)?;
        Ok((need, area(need.2, k, "SME 规模")? >= SME_MIN_ELEMS))
    }

    /// SME 整铺生产入口；面板由公开调用面的线程私有缓存注入。
    pub fn sme_gemm(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        panel: &mut [f32],
    ) -> Result<bool> {
        if !sme_tiled(m, n, k) || !sme_available() {
            return Ok(false);
        }
        let (need, large_enough) = sme_needs(m, n, k)?;
        if !large_enough {
            return Ok(false);
        }
        checked_sme_gemm(a, b, (m, n, k), out, panel, need)
    }

    fn checked_sme_gemm(
        a: &[f32],
        b: &[f32],
        shape: (usize, usize, usize),
        out: &mut [f32],
        panel: &mut [f32],
        need: Sizes,
    ) -> Result<bool> {
        ensure(need, (a.len(), b.len(), out.len()), "SME")?;
        if panel.len() < need.0 {
            return invalid("SME 打包面板不足");
        }
        let (m, n, k) = shape;
        sme_packed(a, b, m, n, k, out, panel);
        Ok(true)
    }

    /// 线程配置由 `LLM_THREADS` 覆盖，否则使用可用并行度。
    pub fn threads_override() -> usize {
        std::env::var("LLM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
            })
    }

    pub(crate) fn sme_threads(m: usize, n: usize) -> usize {
        threads_override().min(m / SME_MR).min(n / SME_NR).max(1)
    }

    fn sme_packed(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        panel: &mut [f32],
    ) {
        let mblk = m / SME_MR;
        let threads = sme_threads(m, n);
        if threads == 1 {
            sme_chunk(a, b, panel, out, (n, k, 0, mblk), None);
            return;
        }
        let chunk = mblk.div_ceil(threads);
        let barrier = std::sync::Barrier::new(mblk.div_ceil(chunk));
        std::thread::scope(|scope| {
            let jobs = panel
                .chunks_mut(chunk * k * SME_MR)
                .zip(out.chunks_mut(chunk * SME_MR * n));
            for (slot, (panel, out)) in jobs.enumerate() {
                let blocks = panel.len() / (k * SME_MR);
                let barrier = &barrier;
                scope.spawn(move || {
                    sme_chunk(
                        a,
                        b,
                        panel,
                        out,
                        (n, k, slot * chunk, blocks),
                        Some(barrier),
                    )
                });
            }
        });
    }

    /// 每个工作槽先打包，再在屏障后复用生产 `sme_stage`；单线程不引入屏障。
    fn sme_chunk(
        a: &[f32],
        b: &[f32],
        panel: &mut [f32],
        out: &mut [f32],
        geo: (usize, usize, usize, usize),
        barrier: Option<&std::sync::Barrier>,
    ) {
        let (n, k, first, blocks) = geo;
        unsafe {
            cutile_sme_pack(a.as_ptr(), panel.as_mut_ptr(), k, first, blocks);
        }
        if let Some(barrier) = barrier {
            barrier.wait();
        }
        unsafe {
            cutile_sme_stage(panel.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, k, blocks);
        }
    }
}
