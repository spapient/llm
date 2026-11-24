//! L2 证据的测量面：一台设备逐形状编译、执行、与标量参考对拍。
//!
//! 与 [`crate::verify`] 分住两个文件是因为它们是两件事：`verify` 回答"协商、通道、报告"
//! 该长什么样（`Negotiation` / `VerifyEntry` / `VerifyReport` / `Verified`），本模块回答
//! "这一格算出来的数对不对"（[`verify_one`] / `Lane` / `Sweep`）。报告类型不依赖执行设备，
//! 测量面反过来要碰编译器与 GPU；分开后两份文件都在静态门的 600 行以内，
//! `verify::verify_one` 的历史路径由 `verify` 的重导出保持。

use crate::build;
use crate::pipeline::{Prepared, Run, parse_shape, rejected_shapes, verify_shapes};
use crate::toolchain;
use crate::verify::{Channel, Verified};
use cutile::cache::Artifact;
use cutile::cap::{Path, Proof};
use cutile::dirs::Dirs;
use cutile::emit::Fragment;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error};
use prof::decl::OpDecl;
use prof::model::Sample;

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
    let (_lane, tool) = Lane::ready(profile)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    check_rejected_shapes(u64::from(run.model.group.max(1)))?;
    let mut verified = blank(&run, tool.as_ref());
    verified.variant = path.variant.clone();
    let sweep = Sweep {
        run: &run,
        dirs,
        tool,
        path,
        tolerance: cutile::reference::tolerance(run.model.dtype),
        verified,
    };
    sweep.measure()
}

/// 一个形状“怎么编译执行”：当前只保留本机 C 参考通道；设备生产态执行走 `prof::runtime` FFI 后端岛。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// 本机 C 编译器 + 产物缓存（命中就不调编译器）。
    Host,
}

impl Lane {
    /// 档案声明的 `emit` 对应哪条通道，以及这条通道此刻能不能跑（`None` 表示没有编译器，只能复用产物）。
    pub(crate) fn ready(
        profile: &Profile,
    ) -> diagnostic::Result<(Self, Option<toolchain::Toolchain>)> {
        match profile.meta.emit {
            Emit::C => Ok((Self::Host, toolchain::detect())),
            _ => Err(Error::new(
                Code::Unsupported,
                format!(
                    "设备 `{}` 的生产态 verify 走 prof::runtime FFI 后端岛，shard 只保留 C 参考通道",
                    profile.meta.id
                ),
            )),
        }
    }
}

/// 一次遍历：流水线、执行通道、选中的路径、对拍阈值，以及逐形状填的报告。
struct Sweep<'a> {
    run: &'a Run<'a>,
    dirs: &'a Dirs,
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
        let sample = self.run.sample(m, n);
        let prepared = self.run.prepare(&self.path, self.dirs, m, n)?;
        Ok((sample, prepared))
    }

    /// 输出个数该是多少：C 通道看样本声明的 `y`。
    fn expected(&self, sample: &Sample, _want: &[f32]) -> usize {
        sample.buf("y").map_or(sample.n, |y| y.values.len())
    }

    /// 一格怎么跑：返回回读的输出与产物记录。
    fn kernel(
        &mut self,
        sample: &Sample,
        prepared: &Prepared,
    ) -> diagnostic::Result<(Vec<f32>, Option<Artifact>)> {
        // C 参考通道本身就是串行语义基准：不参与「形态是否并行」的判定，也不该被读成缺口。
        self.verified.channel = Channel::Reference;
        self.host(sample, prepared)
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
}

/// 空报告：设备与工具链身份先写进去，形状层面的字段随每一格填。
fn blank(run: &Run<'_>, tool: Option<&toolchain::Toolchain>) -> Verified {
    Verified {
        device: String::new(),
        // 缺省是语义门：`Parallel` 必须由确实跑通的并行程序写进去，不能靠缺省拿到。
        channel: Channel::Harness,
        vendor: run.profile.meta.vendor.clone(),
        id: run.profile.meta.id.clone(),
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
