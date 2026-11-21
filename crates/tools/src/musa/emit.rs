//! 设备源生成：算子名 + 形状 → 单文件程序 + 逐元素参考值。
//!
//! 分发是一张表（[`LANES`]），不是一条 `match`：前缀决定走哪条车道，车道只做
//! 「设备源 + host 驱动拼成一文件」这一件事，具体 lowering 全在 `cutile` / `engine` 里。

use cutile::builtin::Catalogue;
use cutile::dirs::Dirs;
use diagnostic::{Code, Error, Result};
use op::decl::OpDecl;
use op::model::Sample;

/// 逐点车道的块内线程数：与 `engine::cuda::cuda_point_driver` 的末块守卫配套，改这里要一起改。
const POINT_THREADS: usize = 256;

/// 一次测量作业：算子名 + 形状 + 是否走张量核特化。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    /// 算子声明名（`op` 全量算子表里的一格）。
    pub op: String,
    /// 规约维度。
    pub m: usize,
    /// 输出维度。
    pub n: usize,
    /// `gemm_f32` 走 WMMA 张量核特化；其余算子忽略此开关。
    pub wmma: bool,
}

impl Job {
    /// 缺省作业：`64 × 1`，与算子语义表的 `sample` 口径一致。
    pub fn of(op: &str) -> Self {
        Self {
            op: op.to_string(),
            m: 64,
            n: 1,
            wmma: false,
        }
    }

    /// 形状：规约维度与输出维度。
    pub fn shape(mut self, m: usize, n: usize) -> Self {
        self.m = m;
        self.n = n;
        self
    }

    /// 走张量核特化。只对 `gemm_f32` 有意义：`m`、`n` 不是 64 的整数倍时走守卫路径，精度是
    /// tf32 级（norm ~3e-4），不满足 `gemm_f32` 的 f32 契约。
    pub fn tensor(mut self) -> Self {
        self.wmma = true;
        self
    }
}

/// 生成结果：设备程序 + 逐元素参考值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// `mcc -x musa` 直接编译的单文件程序。
    pub device: String,
    /// 标量参考，每行一个 `f32`（九位有效数字，够判到 `1e-5`）。
    pub reference: String,
}

/// 生成：`job` → 设备程序与参考值。
///
/// 设备后端是进程级注册表，`cutile` 的入口按 `Emit::Cuda` 查表；没装配时这里拿到的是
/// `Unsupported`（而不是一个空源），所以装配就放在这里而不是让调用方记得调。
pub fn program(job: &Job) -> Result<Program> {
    engine::backends::install();
    let catalogue = Catalogue::try_builtin()?;
    let dirs = Dirs::under(std::env::temp_dir().join("llm-musa"));
    let decl = declare(&catalogue, &job.op)?;
    let model = op::model(&job.op).ok_or_else(|| missing(&job.op, "语义表"))?;
    let sample = (model.sample)(job.m, job.n);
    let device = source(&catalogue, &dirs, job, decl, &sample)?;
    let want = (model.reference)(&sample)?;
    Ok(Program {
        device,
        reference: lines(&want),
    })
}

/// 车道：算子名的一个前缀 + 一个「设备源怎么拼」的函数。
type Lane = (&'static str, fn(&Ctx<'_>) -> Result<String>);

/// 车道表。**顺序即优先级**：`gemm_f32` 的张量核形态要与循环注入形态区分开，因此先于 `gemm_`。
const LANES: [Lane; 7] = [
    ("rmsnorm_f32", rmsnorm),
    ("softmax_f32", softmax),
    ("reduce_", reduce),
    ("scan_f32", scan),
    ("gemm_f32", gemm),
    ("gemv_", gemm),
    ("gemm_", gemm),
];

/// 一条车道的输入。
struct Ctx<'a> {
    catalogue: &'a Catalogue,
    dirs: &'a Dirs,
    job: &'a Job,
    decl: &'a OpDecl,
    sample: &'a Sample,
}

impl Ctx<'_> {
    /// 本算子在声明里的分块尺寸。
    fn chunk(&self) -> Result<usize> {
        self.decl
            .chunk
            .as_ref()
            .map(|chunk| chunk.size)
            .ok_or_else(|| missing(&self.job.op, "[chunk] 声明"))
    }

    /// 第一个非缓冲形参的取值（归约长度、并行归约的组数）。
    fn scalar(&self) -> Result<usize> {
        self.decl
            .signature
            .params()
            .iter()
            .find(|param| !param.ctype.is_buffer())
            .and_then(|param| self.sample.scalar(&param.name))
            .map(|value| value.max(0) as usize)
            .ok_or_else(|| missing(&self.job.op, "整数形参"))
    }
}

/// 按车道表拼设备源；没有车道认领前缀时报错，不退回逐点车道（那会把形状错误算成精度错误）。
fn source(
    catalogue: &Catalogue,
    dirs: &Dirs,
    job: &Job,
    decl: &OpDecl,
    sample: &Sample,
) -> Result<String> {
    let ctx = Ctx {
        catalogue,
        dirs,
        job,
        decl,
        sample,
    };
    let lane = LANES
        .iter()
        .find(|(prefix, _)| job.op.starts_with(prefix))
        .ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                format!("`{}` 没有对应的设备源车道（见 musa::emit::LANES）", job.op),
            )
        })?;
    (lane.1)(&ctx)
}

/// 分块归一化：两段（部分和/最大值回读折叠后再归一化），host 驱动把两段拼成一文件。
fn rmsnorm(ctx: &Ctx<'_>) -> Result<String> {
    let job = ctx.job;
    let (partial, norm) =
        cutile::execution::cuda_rmsnorm_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?;
    engine::norm::cuda_rmsnorm_driver(&[partial, norm], &ctx.decl.signature, ctx.sample, ctx.chunk()?)
}

/// 分块归一化：三段（最大值、指数和、归一化），折叠与 rmsnorm 同形。
fn softmax(ctx: &Ctx<'_>) -> Result<String> {
    let job = ctx.job;
    let (max, expsum, norm) =
        cutile::execution::cuda_softmax_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?;
    engine::norm::cuda_softmax_driver(
        &[max, expsum, norm],
        &ctx.decl.signature,
        ctx.sample,
        ctx.chunk()?,
    )
}

/// 分块归约：网格按长度切块，块内串行、块间由 `cuda_grid_driver` 派发。
fn reduce(ctx: &Ctx<'_>) -> Result<String> {
    let job = ctx.job;
    let device =
        cutile::execution::cuda_reduce_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?;
    let grid = ctx.scalar()?.div_ceil(ctx.chunk()?);
    engine::cuda::cuda_grid_driver(&device, &ctx.decl.signature, ctx.sample, grid, grid)
}

/// 分块扫描：局部前缀 + 段间折叠 + 校正，`decay` 由声明给出。
fn scan(ctx: &Ctx<'_>) -> Result<String> {
    let job = ctx.job;
    let (local, correct, decay) =
        cutile::execution::cuda_scan_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?;
    engine::scan::cuda_scan_driver(
        &[local, correct],
        &ctx.decl.signature,
        ctx.sample,
        ctx.chunk()?,
        decay,
    )
}

/// 矩阵乘：`gemm_f32` 且要张量核时走 WMMA 特化，其余走循环注入（`[grid] dims` 由声明给）。
fn gemm(ctx: &Ctx<'_>) -> Result<String> {
    let job = ctx.job;
    let wmma = job.wmma && job.op == "gemm_f32";
    let device = if wmma {
        cutile::execution::cuda_wmma_gemm_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?
    } else {
        let (device, dims) =
            cutile::execution::cuda_gemm_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?;
        return engine::cuda::cuda_loop_driver(&device, &ctx.decl.signature, ctx.sample, &dims);
    };
    engine::wmma::gemm_driver(&device, &ctx.decl.signature, ctx.sample)
}

/// 逐点：输出元素数 → 块数，块内 `POINT_THREADS` 线程。
fn point(ctx: &Ctx<'_>) -> Result<String> {
    let job = ctx.job;
    let device =
        cutile::execution::cuda_grid_program(ctx.catalogue, ctx.dirs, &job.op, job.m, job.n)?;
    let blocks = ctx.out_len()?.div_ceil(POINT_THREADS);
    engine::cuda::cuda_point_driver(&device, &ctx.decl.signature, ctx.sample, blocks, POINT_THREADS)
}

impl Ctx<'_> {
    /// 可写输出的元素个数。
    fn out_len(&self) -> Result<usize> {
        let output = self
            .decl
            .signature
            .output()
            .ok_or_else(|| missing(&self.job.op, "可写输出"))?;
        self.sample
            .f32(output)
            .map(<[f32]>::len)
            .ok_or_else(|| missing(&self.job.op, "f32 输出缓冲"))
    }
}

/// 声明里查算子；缺声明报 `Missing`，不拿空 `OpDecl` 顶替。
fn declare<'a>(catalogue: &'a Catalogue, op: &str) -> Result<&'a OpDecl> {
    catalogue
        .op(op)
        .ok_or_else(|| missing(op, "算子声明"))
}

/// 参考值排版：每行一个九位有效数字的 `f32`，与程序打印的位模式同量级比较。
fn lines(values: &[f32]) -> String {
    values.iter().map(|value| format!("{value:.9e}\n")).collect()
}

/// 缺件：名字 + 缺的是什么。
fn missing(op: &str, what: &str) -> Error {
    Error::new(Code::Missing, format!("`{op}` 缺{what}"))
}
