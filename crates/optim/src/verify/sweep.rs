//! 真机校验的驱动：一台设备、一组形状，逐格算并记账。

use super::*;

/// 一台设备的 L2 证据：逐形状编译、执行、与参考对拍。回答什么：产物在形状矩阵每一格上算出来的数
/// 是不是标量参考的数。不回答什么：见证是否成立、协商该不该命中首档。
///
/// # Errors
///
/// 设备不可编译、缺编译器、编译/执行失败、输出个数不符或对拍超出容差时返回说明。
pub fn verify_one(
    op: &OpDecl,
    profile: &Profile,
    dirs: &Dirs,
    proof: &Proof,
) -> diagnostic::Result<Verified> {
    let run = Run::new(op, profile, proof.clone())?;
    let (lane, tool) = Lane::ready(profile)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    check_rejected_shapes(u64::from(run.model.group.max(1)))?;
    let mut verified = blank(&run, tool.as_ref());
    verified.variant = path.variant.clone();
    let sweep = Sweep {
        run: &run,
        dirs,
        lane,
        tool,
        path,
        tolerance: cutile::reference::tolerance(run.model.dtype),
        verified,
    };
    sweep.measure()
}

/// 一个形状“怎么编译执行”：设备差异只在这里。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// 本机 C 编译器 + 产物缓存（命中就不调编译器）。
    Host,
    /// nvcc 临时编译、GPU 执行、回读；不进产物缓存，产物因此为空。
    Cuda,
    /// xcrun metal + metallib 临时编译、Apple GPU 执行、回读；不进产物缓存。
    Metal,
    /// ccec 临时编译、NPU 执行、AscendCL 回读；不进产物缓存。
    Ascend,
}

impl Lane {
    /// 档案声明的 `emit` 对应哪条通道，以及这条通道此刻能不能跑（`None` 表示没有编译器，只能复用产物）。本机没有执行路径时在这里 fail-fast。
    pub(crate) fn ready(
        profile: &Profile,
    ) -> diagnostic::Result<(Self, Option<toolchain::Toolchain>)> {
        match profile.meta.emit {
            Emit::C => Ok((Self::Host, toolchain::detect())),
            Emit::Cuda => Self::ready_device(profile, Self::Cuda),
            Emit::Metal => Self::ready_device(profile, Self::Metal),
            Emit::Ascendc => Self::ready_device(profile, Self::Ascend),
            _ => Err(Error::new(Code::Unsupported, unsupported_emit(profile))),
        }
    }

    /// 加速器通道共用的「设备在不在」判断：在则经 [`crate::backend::Driver::toolchain`] 取实测
    /// 工具链返回，不在则 fail-fast（`NO_DEVICE`）。工具链身份只经 `Backend` 缝取，机制层不直接
    /// 调 `detect_nvcc` / `detect_metal`——后端实现搬去 `engine` 后这里一行不改。
    fn ready_device(
        profile: &Profile,
        lane: Self,
    ) -> diagnostic::Result<(Self, Option<toolchain::Toolchain>)> {
        match crate::device::spec(profile.meta.emit) {
            Some(backend) if backend.available(profile) => Ok((lane, backend.toolchain())),
            _ => Err(Error::new(Code::Unsupported, NO_DEVICE.to_string())),
        }
    }
}

/// 一次遍历：流水线、执行通道、选中的路径、对拍阈值，以及逐形状填的报告。
struct Sweep<'a> {
    run: &'a Run<'a>,
    dirs: &'a Dirs,
    lane: Lane,
    tool: Option<toolchain::Toolchain>,
    path: Path,
    tolerance: f64,
    verified: Verified,
}

impl Sweep<'_> {
    /// 逐形状跑一遍：一格失败即整台失败，证据链不接受“其余形状还行”；失败只给错误，不给半份报告。
    fn measure(mut self) -> diagnostic::Result<Verified> {
        for shape in verify_shapes(&self.run.model) {
            self.cell(shape)?;
        }
        Ok(self.verified)
    }

    /// 一格：样本、计划、参考与判据两条通道完全相同，只有“怎么编译执行”按通道分。
    /// 回答什么：这一格算不算数；不回答什么：这一格为什么不算（错误原样上报，不在这里解释）。
    fn cell(&mut self, shape: (u64, u64)) -> diagnostic::Result<()> {
        let (sample, prepared) = self.prepare(shape)?;
        let (bits, artifact) = self.kernel(&sample, &prepared)?;
        let want = (self.run.model.reference)(&sample)?;
        let expect = self.expected(&sample, &want);
        let error = parity(&bits, &want, expect, shape, self.tolerance)?;
        let out = &mut self.verified;
        out.max_rel_err = out.max_rel_err.max(error);
        out.shapes.push(shape);
        out.artifacts.extend(artifact);
        out.fragment = Fragment::new(prepared.source.lines());
        out.key = prepared.key.hex();
        out.tile = prepared.plan.tile.to_string();
        Ok(())
    }

    /// 形状 → 样本 + 产物：两条通道共用的前半段。
    fn prepare(&self, (m, n): (u64, u64)) -> diagnostic::Result<(Sample, Prepared)> {
        let (m, n) = (m as usize, n as usize);
        let sample = (self.run.model.sample)(m, n);
        let prepared = self.run.prepare(&self.path, self.dirs, m, n)?;
        Ok((sample, prepared))
    }

    /// 输出个数该是多少：C 通道看样本声明的 `y`，设备通道（CUDA / Metal / 昇腾）看参考输出的长度。
    fn expected(&self, sample: &Sample, want: &[f32]) -> usize {
        match self.lane {
            Lane::Host => sample.buf("y").map_or(sample.n, |y| y.values.len()),
            Lane::Cuda | Lane::Metal | Lane::Ascend => want.len(),
        }
    }

    /// 一格怎么跑：返回回读的输出与产物记录（CUDA / Metal / 昇腾 通道每次临时编译，没有产物）。
    fn kernel(
        &mut self,
        sample: &Sample,
        prepared: &Prepared,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        match self.lane {
            Lane::Host => {
                // C 参考通道本身就是串行语义基准：不参与「形态是否并行」的判定，也不该被读成缺口。
                self.verified.channel = Channel::Reference;
                self.host(sample, prepared)
            }
            Lane::Cuda => self.device(sample, prepared, Emit::Cuda),
            Lane::Metal => self.device(sample, prepared, Emit::Metal),
            // 昇腾没有并行内核（P0 片段只到标量 C）：`parallel` 如实 `Unsupported`，回落后由
            // `AscendBackend::harness` 给那份单源 AscendC 程序，报告里的通道写“语义门”。
            Lane::Ascend => self.device(sample, prepared, Emit::Ascendc),
        }
    }

    /// C 通道：缓存编译（命中就不调编译器），产物因此进报告。
    fn host(
        &self,
        sample: &Sample,
        prepared: &Prepared,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        let program = prepared.harness(self.run.op, sample, 1)?;
        let (outcome, artifact) = build::execute(
            self.run.profile,
            self.tool.as_ref(),
            &prepared.key,
            &program,
            self.dirs,
        )?;
        Ok((outcome.bits, Some(artifact)))
    }

    /// 设备通道：nvcc / xcrun metal / ccec 临时编译、设备执行、回读；签名与样本先对齐（与 C 通道同一条判据）。
    fn device(
        &mut self,
        sample: &Sample,
        prepared: &Prepared,
        emit: Emit,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        let (backend, launch) = self.channel(emit, sample)?;
        let (program, channel) = self.program(backend, sample, prepared, &launch)?;
        let (stdout, _stderr) =
            backend.compile_run(self.run.profile, self.dirs, &program, "verify")?;
        // 只有真跑通了才写「并行」：回落的单线程执行不得借此升级。
        self.verified.channel = channel;
        Ok((runner::parse_bits(&stdout), None))
    }

    /// 通道前置：签名与样本先对齐（与 C 通道同一条判据），再取后端与派发几何。
    fn channel(
        &self,
        emit: Emit,
        sample: &Sample,
    ) -> diagnostic::Result<(&'static dyn crate::backend::Backend, crate::backend::Launch)> {
        self.run.op.signature.check(sample)?;
        let backend = backend(emit)?;
        let launch = crate::execution::launch(self.run.op, sample)?;
        Ok((backend, launch))
    }

    /// 要编译的那份程序：并行网格（生产态形态）优先，该形态没有实现时显式回落单线程语义门。
    fn program(
        &self,
        backend: &dyn crate::backend::Backend,
        sample: &Sample,
        prepared: &Prepared,
        launch: &crate::backend::Launch,
    ) -> diagnostic::Result<(crate::backend::Source, Channel)> {
        let signature = &self.run.op.signature;
        let source = &prepared.source.text;
        match backend.parallel(source, signature, sample, launch) {
            Ok(program) => Ok((program, Channel::Parallel)),
            Err(error) if error.code() == Code::Unsupported => Ok((
                backend.harness(source, signature, sample)?,
                Channel::Harness,
            )),
            Err(error) => Err(error),
        }
    }
}

/// 空报告：设备与工具链身份先写进去，形状层面的字段随每一格填。
fn blank(run: &Run<'_>, tool: Option<&toolchain::Toolchain>) -> Verified {
    Verified {
        device: String::new(),
        // 缺省是语义门：`Parallel` 必须由确实跑通的并行程序写进去，不能靠缺省拿到。
        channel: Channel::Harness,
        vendor: run.profile.meta.vendor.to_string(),
        id: run.profile.meta.id.to_string(),
        variant: String::new(),
        tile: String::new(),
        shapes: Vec::new(),
        witnesses: 0,
        self_checked: 0,
        max_rel_err: 0.0,
        fragment: Fragment::new(0),
        tool_ver: run.profile.meta.tool_ver.clone(),
        toolchain: tool.map(toolchain::Toolchain::id),
        toolchain_agrees: tool.is_some_and(|tool| tool.agrees_with(&run.profile.meta.tool_ver)),
        flags: run.profile.flags.join(" "),
        key: String::new(),
        artifacts: Vec::new(),
    }
}

/// 对拍：个数与误差两条判据，返回实测最大相对误差。个数先于值：个数不符时连参考都不必逐点比。
/// 它不判断证据等级（等级由见证决定），也不解释误差从哪来。
fn parity(
    bits: &[f32],
    want: &[f32],
    expect: usize,
    (m, n): (u64, u64),
    tolerance: f64,
) -> diagnostic::Result<f64> {
    if bits.len() != expect {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "形状 {m}x{n}：内核只输出 {} 个数，期望 {expect}",
                bits.len()
            ),
        ));
    }
    let error = cutile::reference::max_rel_err(bits, want);
    if error > tolerance {
        return Err(Error::new(
            Code::Invalid,
            format!("形状 {m}x{n} 对拍失败：max_rel_err {error:.3e} > {tolerance:.0e}"),
        ));
    }
    Ok(error)
}

/// 非 C 前端的说明：设备在本机与不在本机是两句不同的话。
fn unsupported_emit(profile: &Profile) -> String {
    let declared: Vec<String> = profile
        .provides
        .iter()
        .map(|have| format!("{}={}", have.prim, have.grade))
        .collect();
    let why = if witness::device_available(profile) {
        format!(
            "设备在本机（`emit = {}`），内核执行路径尚未接通（`c` / `cuda` / `metal` / `ascendc` 已接，\
             其余仍只有见证）：见证已经在这台设备上跑过，内核文本还没有对应的编译/回读路径",
            profile.meta.emit
        )
    } else {
        NO_DEVICE.to_string()
    };
    format!("{why}；档案声明的等级：{}", declared.join(" "))
}

/// 负例也是判据：组大小约束被放宽时，这里必须先在 `verify` 里失败，而不是等到错数据。
fn check_rejected_shapes(group: u64) -> diagnostic::Result<()> {
    for (m, n) in rejected_shapes(group) {
        let shape = format!("{m}x{n}");
        if parse_shape(&shape, group).is_ok() {
            return Err(Error::new(
                Code::Internal,
                format!("形状 `{shape}` 必须被拒绝：M 必须是组大小 {group} 的整数倍"),
            ));
        }
    }
    Ok(())
}
