//! 设备运行时后端岛：提交契约、FFI 句柄、命令队列、缓冲、管线与执行器实现收在这里，不进上层编排。

#[path = "exec.rs"]
pub mod exec;

#[cfg(target_os = "macos")]
#[path = "metal/runtime.rs"]
pub mod metal;

#[cfg(windows)]
#[path = "opencl.rs"]
pub mod opencl;

/// `gemm_micro.k` 的安全静态 ABI 桥；FMA、sdot 与 SME 生产均由此进入同一正文。
///
/// 越界仅限这个已登记模块：对外只收安全切片与强类型几何；每个入口先检查面板长度、坐标与整数
/// 转换，输入只读、输出可写且 Rust 借用保证不重叠。裸指针只在本次调用存活，不跨线程、不逃逸。
#[allow(unsafe_code)]
pub mod gemm_micro {
    use super::exec::{add, area, c_int};
    pub use crate::native::{gemm8x8::Tile, i8::Gram};
    use diagnostic::{Code, Error, Result};

    include!(concat!(env!("OUT_DIR"), "/gemm_micro_contract.rs"));

    struct FmaArgs {
        row: i32,
        col: i32,
        n: i32,
        kb: i32,
        first: bool,
    }

    fn invalid<T>(message: impl Into<String>) -> Result<T> {
        Err(Error::new(Code::Invalid, message.into()).with("prof::runtime::gemm_micro"))
    }

    /// C／Rust 对拍用 FMA 入口；几何、长度与整数 ABI 全部验证后才进入 ISA。
    pub fn fma_micro(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) -> Result<()> {
        let args = checked_fma(panel, b, out, &tile)?;
        dispatch_fma(panel, b, out, args);
        Ok(())
    }

    fn checked_fma(panel: &[f32], b: &[f32], out: &[f32], tile: &Tile) -> Result<FmaArgs> {
        let (a_need, b_need) = fma_panel_sizes(tile.kb)?;
        let (out_need, cols) = fma_output_size(tile)?;
        ensure_fma(panel, b, out, tile, (a_need, b_need, out_need, cols))?;
        fma_arguments(tile)
    }

    fn fma_panel_sizes(kb: usize) -> Result<(usize, usize)> {
        let a = area(kb, FMA_MR, "FMA A 面板")?;
        let b = area(kb, FMA_NR, "FMA B 面板")?;
        Ok((a, b))
    }

    fn fma_output_size(tile: &Tile) -> Result<(usize, usize)> {
        let rows = add(tile.row0, FMA_MR, "FMA 行坐标")?;
        let cols = add(tile.j, FMA_NR, "FMA 列坐标")?;
        let out = area(rows, tile.n, "FMA 输出")?;
        Ok((out, cols))
    }

    fn ensure_fma(
        panel: &[f32],
        b: &[f32],
        out: &[f32],
        tile: &Tile,
        needs: (usize, usize, usize, usize),
    ) -> Result<()> {
        let (a_need, b_need, out_need, cols) = needs;
        if panel.len() < a_need || b.len() < b_need || cols > tile.n || out.len() < out_need {
            return invalid("FMA 面板或输出缓冲不足");
        }
        Ok(())
    }

    fn fma_arguments(tile: &Tile) -> Result<FmaArgs> {
        Ok(FmaArgs {
            row: c_int(tile.row0, "row")?,
            col: c_int(tile.j, "col")?,
            n: c_int(tile.n, "n")?,
            kb: c_int(tile.kb, "kb")?,
            first: tile.first,
        })
    }

    fn dispatch_fma(panel: &[f32], b: &[f32], out: &mut [f32], args: FmaArgs) {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if !(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
            {
                cutile_fma_micro_scalar(
                    panel.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    args.row,
                    args.col,
                    args.n,
                    args.kb,
                    i32::from(args.first),
                );
                return;
            }
            macro_rules! launch {
                ($kernel:ident) => {
                    $kernel(
                        panel.as_ptr(),
                        b.as_ptr(),
                        out.as_mut_ptr(),
                        args.row,
                        args.col,
                        args.n,
                        args.kb,
                        i32::from(args.first),
                    );
                };
            }
            #[cfg(all(target_arch = "x86_64", target_env = "msvc"))]
            launch!(cutile_fma_micro_avx2);
            #[cfg(all(target_arch = "x86_64", not(target_env = "msvc")))]
            launch!(cutile_fma_micro);
            #[cfg(not(target_arch = "x86_64"))]
            launch!(cutile_fma_micro);
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

    fn sdot_ready(gram: &Gram) -> bool {
        sdot_available() && gram.kvec >= 16 && gram.kvec <= gram.k && gram.kvec.is_multiple_of(16)
    }

    /// C／Rust 对拍用 sdot 入口；无 dotprod 时不执行向量指令。
    pub fn sdot_micro(a: &[i8], bpack: &[i8], out: &mut [i32], gram: Gram) -> Result<bool> {
        if !sdot_ready(&gram) {
            return Ok(false);
        }
        let (row, blk, n, k, kvec) = checked_sdot(a, bpack, out, &gram)?;
        unsafe {
            cutile_sdot_micro(
                a.as_ptr(),
                bpack.as_ptr(),
                out.as_mut_ptr(),
                row,
                blk,
                n,
                k,
                kvec,
            );
        }
        Ok(true)
    }

    fn checked_sdot(
        a: &[i8],
        b: &[i8],
        out: &[i32],
        gram: &Gram,
    ) -> Result<(i32, i32, i32, i32, i32)> {
        let (rows, cols) = sdot_position(gram)?;
        let (a_need, b_need, out_need) = sdot_sizes(gram, rows, cols)?;
        ensure_sdot(a, b, out, a_need, b_need, out_need)?;
        let (row, blk, n, k, kvec) = sdot_arguments(gram)?;
        Ok((row, blk, n, k, kvec))
    }

    fn sdot_position(gram: &Gram) -> Result<(usize, usize)> {
        let rows = add(gram.r, SDOT_MR, "sdot 行坐标")?;
        let cols = add(gram.blk, SDOT_NR, "sdot 列坐标")?;
        Ok((rows, cols))
    }

    fn sdot_sizes(gram: &Gram, rows: usize, cols: usize) -> Result<(usize, usize, usize)> {
        let a = area(rows, gram.k, "sdot A")?;
        let b = area(cols, gram.k, "sdot B")?;
        let out = area(rows, gram.n, "sdot 输出")?;
        Ok((a, b, out))
    }

    fn ensure_sdot(
        a: &[i8],
        b: &[i8],
        out: &[i32],
        a_need: usize,
        b_need: usize,
        out_need: usize,
    ) -> Result<()> {
        if a.len() < a_need || b.len() < b_need || out.len() < out_need {
            return invalid("sdot 面板或输出缓冲不足");
        }
        Ok(())
    }

    fn sdot_arguments(gram: &Gram) -> Result<(i32, i32, i32, i32, i32)> {
        let (row, blk) = sdot_position_arguments(gram)?;
        let (n, k, kvec) = sdot_shape_arguments(gram)?;
        Ok((row, blk, n, k, kvec))
    }

    fn sdot_position_arguments(gram: &Gram) -> Result<(i32, i32)> {
        let row = c_int(gram.r, "row")?;
        let blk = c_int(gram.blk, "blk")?;
        Ok((row, blk))
    }

    fn sdot_shape_arguments(gram: &Gram) -> Result<(i32, i32, i32)> {
        let n = c_int(gram.n, "n")?;
        let k = c_int(gram.k, "k")?;
        let kvec = c_int(gram.kvec, "kvec")?;
        Ok((n, k, kvec))
    }

    #[cfg(all(nightly, target_arch = "aarch64"))]
    fn sme_available() -> bool {
        std::arch::is_aarch64_feature_detected!("sme")
    }

    /// Apple stable 工具链尚无 SME 探测宏；只读内核已发布的 FEAT_SME 能力位。
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

    /// C／Rust 对拍用单个 16×64 SME 瓦片。
    pub fn sme_micro(a: &[f32], b: &[f32], out: &mut [f32], n: usize, k: usize) -> Result<bool> {
        if k == 0 || n < SME_NR || !sme_available() {
            return Ok(false);
        }
        checked_sme_micro(a, b, out, n, k)?;
        Ok(true)
    }

    fn checked_sme_micro(a: &[f32], b: &[f32], out: &mut [f32], n: usize, k: usize) -> Result<()> {
        let needs = sme_sizes(SME_MR, n, k)?;
        ensure_sme(a, b, out, needs)?;
        unsafe {
            cutile_sme_micro(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, k);
        }
        Ok(())
    }

    /// SME 整铺生产入口；无设备、非整铺或低于规模门槛时明确回退。
    pub fn sme_gemm(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<bool> {
        if !sme_tiled(m, n, k) || !sme_available() {
            return Ok(false);
        }
        checked_sme_gemm(a, b, m, n, k, out)
    }

    fn sme_tiled(m: usize, n: usize, k: usize) -> bool {
        m > 0 && n > 0 && k > 0 && m.is_multiple_of(SME_MR) && n.is_multiple_of(SME_NR)
    }

    fn checked_sme_gemm(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<bool> {
        let (needs, volume) = sme_work_sizes(m, n, k)?;
        if volume < SME_MIN_ELEMS {
            return Ok(false);
        }
        ensure_sme(a, b, out, needs)?;
        sme_packed(a, b, m, n, k, out, needs.0);
        Ok(true)
    }

    fn sme_work_sizes(m: usize, n: usize, k: usize) -> Result<((usize, usize, usize), usize)> {
        let needs = sme_sizes(m, n, k)?;
        let volume = area(needs.2, k, "SME 规模")?;
        Ok((needs, volume))
    }

    /// 行块与列组共同限制线程数，不产生空工作槽。
    pub(crate) fn sme_threads(m: usize, n: usize) -> usize {
        crate::native::threads_override()
            .min(m / SME_MR)
            .min(n / SME_NR)
            .max(1)
    }

    fn ensure_sme(a: &[f32], b: &[f32], out: &[f32], needs: (usize, usize, usize)) -> Result<()> {
        if a.len() < needs.0 || b.len() < needs.1 || out.len() < needs.2 {
            return invalid("SME 面板或输出缓冲不足");
        }
        Ok(())
    }

    fn sme_sizes(m: usize, n: usize, k: usize) -> Result<(usize, usize, usize)> {
        Ok((
            area(m, k, "SME A")?,
            area(k, n, "SME B")?,
            area(m, n, "SME 输出")?,
        ))
    }

    /// Rust 只编排线程与借用；转置打包、流式模式配对和 ZA 外积全在同一 C 正文。
    fn sme_packed(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        panel_len: usize,
    ) {
        let mblk = m / SME_MR;
        let threads = sme_threads(m, n);
        crate::native::with_panels(panel_len, |panel| {
            if threads == 1 {
                unsafe {
                    cutile_sme_pack(a.as_ptr(), panel.as_mut_ptr(), k, 0, mblk);
                    cutile_sme_stage(panel.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, k, mblk);
                }
                return;
            }
            let barrier = std::sync::Barrier::new(threads);
            std::thread::scope(|scope| {
                let barrier = &barrier;
                let mut panel_rest = &mut *panel;
                let mut out_rest = &mut *out;
                for slot in 0..threads {
                    let (r0, r1) = (slot * mblk / threads, (slot + 1) * mblk / threads);
                    let blocks = r1 - r0;
                    let (panel_chunk, panel_tail) = panel_rest.split_at_mut(blocks * k * SME_MR);
                    let (out_chunk, out_tail) = out_rest.split_at_mut(blocks * SME_MR * n);
                    panel_rest = panel_tail;
                    out_rest = out_tail;
                    scope.spawn(move || {
                        unsafe {
                            cutile_sme_pack(a.as_ptr(), panel_chunk.as_mut_ptr(), k, r0, blocks);
                        }
                        barrier.wait();
                        unsafe {
                            cutile_sme_stage(
                                panel_chunk.as_ptr(),
                                b.as_ptr(),
                                out_chunk.as_mut_ptr(),
                                n,
                                k,
                                blocks,
                            );
                        }
                    });
                }
            });
        });
    }
}
