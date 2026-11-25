//! 设备运行时后端岛：提交契约、FFI 句柄、命令队列、缓冲、管线与执行器实现收在这里，不进上层编排。

#[path = "exec.rs"]
pub mod exec;
#[path = "gemm.rs"]
mod kernel;
#[cfg(target_os = "macos")]
#[path = "metal/runtime.rs"]
pub mod metal;
#[cfg(windows)]
#[path = "opencl.rs"]
pub mod opencl;

/// `gemm_micro.k` 的安全静态 ABI 桥；所有裸指针都受切片长度、坐标、溢出与特性门约束。
#[allow(unsafe_code)]
pub mod gemm_micro {
    use super::exec::{add, area};
    use diagnostic::{Code, Error, Result};

    include!(concat!(env!("OUT_DIR"), "/gemm_micro_contract.rs"));

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    unsafe extern "C" {
        fn cutile_sme_pure(rounds: usize);
        fn cutile_sme_mix(rounds: usize, a: *const f32, b: *const f32);
    }

    type Sizes = (usize, usize, usize);

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
        dispatch_fma(panel, b, out, tile, first);
        Ok(())
    }

    fn dispatch_fma(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        tile: (usize, usize, usize, usize),
        first: bool,
    ) {
        let (row, col, n, kb) = tile;
        #[cfg(target_arch = "x86_64")]
        {
            if avx2_fma_detected() {
                unsafe { super::kernel::fma_avx2(a, b, out, row, col, n, kb, first) };
            } else {
                super::kernel::fma_scalar(a, b, out, row, col, n, kb, first);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { super::kernel::fma_neon(a, b, out, row, col, n, kb, first) };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            super::kernel::fma_scalar(a, b, out, row, col, n, kb, first);
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn avx2_fma_detected() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[cfg(target_arch = "aarch64")]
    fn sdot_available() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    fn sdot_available() -> bool {
        false
    }

    /// 向量化段门槛：dotprod 可用，且 kvec 非零、不超 K、为 16 的倍数；其余平台交上层标量兜底。
    fn sdot_ready(g: (usize, usize, usize, usize, usize)) -> bool {
        sdot_available() && g.4 >= 16 && g.4 <= g.3 && g.4.is_multiple_of(16)
    }

    fn checked_sdot(
        a: &[i8],
        b: &[i8],
        out: &[i32],
        g: (usize, usize, usize, usize, usize),
    ) -> Result<()> {
        let (rows, cols) = position(g.0, g.1, SDOT_MR, SDOT_NR)?;
        ensure(
            sizes(rows, cols, g.2, g.3)?,
            (a.len(), b.len(), out.len()),
            "sdot",
        )?;
        Ok(())
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
        checked_sdot(a, b, out, g)?;
        #[cfg(target_arch = "aarch64")]
        {
            let (row, blk, n, k, kvec) = g;
            unsafe { super::kernel::sdot_dotprod(a, b, out, row, blk, n, k, kvec) };
            Ok(true)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Ok(false)
        }
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

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[doc = "FMA 纯端口探针：环内无内存访问。"]
    pub fn port_fma(rounds: usize) -> bool {
        if rounds == 0 {
            return false;
        }
        unsafe {
            super::kernel::fma_pure_probe(rounds);
        }
        true
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[doc = "sdot 纯端口探针：无 dotprod 时拒绝执行。"]
    pub fn port_sdot(rounds: usize) -> bool {
        if rounds == 0 || !sdot_available() {
            return false;
        }
        unsafe {
            super::kernel::sdot_pure_probe(rounds);
        }
        true
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    /// 取数探针公共门槛：`k` 非零且两面切片长度足够。
    fn probe_ready<T>(a: &[T], b: &[T], need: usize, k: usize) -> bool {
        k != 0 && a.len() >= need && b.len() >= need
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    /// sdot 取数探针额外门槛：向量化段长合规且 dotprod 可用。
    fn sdot_probe_ready(k: usize) -> bool {
        k.is_multiple_of(16) && sdot_available()
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[doc = "FMA 取数探针：调用与生产相同的 NEON 瓦片内核。"]
    pub fn port_fma_block(rounds: usize, a: &[f32], b: &[f32], k: usize) -> bool {
        let Ok(need) = area(FMA_NR, k, "FMA probe") else {
            return false;
        };
        if rounds == 0 || !probe_ready(a, b, need, k) {
            return false;
        }
        unsafe {
            super::kernel::fma_block_probe(rounds, a.as_ptr(), b.as_ptr(), k);
        }
        true
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[doc = "sdot 取数探针：调用与生产相同的 dotprod 瓦片内核。"]
    pub fn port_sdot_block(rounds: usize, a: &[i8], b: &[i8], k: usize) -> bool {
        let Ok(need) = area(SDOT_MR, k, "sdot probe") else {
            return false;
        };
        if rounds == 0 || !probe_ready(a, b, need, k) || !sdot_probe_ready(k) {
            return false;
        }
        unsafe {
            super::kernel::sdot_block_probe(rounds, a.as_ptr(), b.as_ptr(), k);
        }
        true
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[doc = "SME 纯外积探针：与生产共用 fmopa 核心，环内无内存访问。"]
    pub fn port_run(rounds: usize) -> bool {
        if rounds == 0 || !sme_available() {
            return false;
        }
        unsafe {
            cutile_sme_pure(rounds);
        }
        true
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[doc = "SME 固定缓存行取数探针：与生产共用 fmopa 核心。"]
    pub fn port_mix(rounds: usize, a: &[f32; SME_MR], b: &[f32; SME_NR]) -> bool {
        if rounds == 0 || !sme_available() {
            return false;
        }
        unsafe {
            cutile_sme_mix(rounds, a.as_ptr(), b.as_ptr());
        }
        true
    }
}
