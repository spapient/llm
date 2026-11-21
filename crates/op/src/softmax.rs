//! f32 小算子四合一：softmax / rmsnorm / embed / rope 的语义、样本与参考。
//!
//! 四者都是对单行激活的逐元素/归约变换，形状与校验逻辑同构，所以放在一个文件里而不是
//! 四个目录：每个模块只管自己的算术，公开路径仍由 `lib.rs` 的 re-export 保持为
//! `op::softmax_f32`、`op::rmsnorm_f32`、`op::embed_f32`、`op::rope_f32`。

pub mod embed {
    //! Embedding 行查表：`y[n, :] = table[idx[n], :]`（行向量 gather）。
    //!
    //! 索引以 f32 承载（当前 `Values` 只有 f32 变体），内核取下标时转成整数；权重表 `[V, D]` 按行
    //! 展平，输出 `[n, D]` 同样按行展平。行数 `V` 由表长除以行宽推出（样本档取 [`VOCAB`]），行宽 `D`
    //! 来自第二轴。本模块自带形状、代价、样本、参考与遍历。
    use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
    use diagnostic::{Code, Error};

    /// 权重表行数（闭集内的固定档）。
    pub const VOCAB: usize = 64;

    /// 两条基本轴：token 轴 `n`（输入）、行宽轴 `d`（输出）。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'n',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'd',
        },
    ];

    /// `(m, n)` 即 `(token 数 n, 行宽 d)`。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(tokens: usize, dim: usize) -> Result<ShapeContext, Error> {
        ShapeContext::new([('n', tokens), ('d', dim)])
    }

    /// 标量代价：每个 token 搬一行 `D`（读表一行 + 写输出一行），下标转换按 1 次/token 估算。
    ///
    /// 常驻表按被访问的行数计价，不把整张词表读一遍（表流量口径属于代价模型，层不重复推）。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        let n = shape
            .get('n')
            .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `n`"))?;
        let d = shape
            .get('d')
            .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `d`"))?;
        let elements = (n as u64) * (d as u64);
        Ok(Cost {
            flops: n as f64,
            read: elements * 4,
            write: elements * 4,
        })
    }

    /// 确定性样本：`idx`（f32 承载的整数下标，长度 n）、`table`（长度 `V·D`）、`y`（长度 `n·D`）。
    pub fn sample(tokens: usize, dim: usize) -> Sample {
        let table = (0..VOCAB * dim)
            .map(|index| (index % VOCAB) as f32 * 0.5)
            .collect();
        Sample {
            m: tokens,
            n: dim,
            scalars: vec![("N", tokens as i64), ("V", VOCAB as i64), ("D", dim as i64)],
            buffers: vec![
                Buffer {
                    name: "idx",
                    values: Values::F32((0..tokens).map(|i| (i % VOCAB) as f32).collect()),
                },
                Buffer {
                    name: "table",
                    values: Values::F32(table),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; tokens * dim]),
                },
            ],
        }
    }

    /// 标量参考：与内核同一行 gather 公式。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let idx = sample.f32("idx")?;
        let table = sample.f32("table")?;
        gather(idx, table, sample.n)
    }

    /// 共享行查表：参考与内核同一套；`dim` 为行宽，表长必须是 `dim` 的整数倍，行数由此推出。
    fn gather(idx: &[f32], table: &[f32], dim: usize) -> diagnostic::Result<Vec<f32>> {
        if dim == 0 {
            return Err(Error::new(Code::Invalid, "embed 行宽不能为零"));
        }
        if !table.len().is_multiple_of(dim) {
            return Err(Error::new(
                Code::Invalid,
                format!("embedding 表长 {} 不是行宽 {dim} 的整数倍", table.len()),
            ));
        }
        let rows = table.len() / dim;
        let mut out = Vec::with_capacity(idx.len() * dim);
        for value in idx {
            let k = *value as usize;
            if k >= rows {
                return Err(Error::new(
                    Code::Invalid,
                    format!("embedding 下标 {k} 越界（词表 {rows}）"),
                ));
            }
            let row = k * dim;
            out.extend_from_slice(&table[row..row + dim]);
        }
        Ok(out)
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// 行宽 `D` 由输出长度与 token 数推出：`out.len() == idx.len()·D`；表长必须是 `D` 的整数倍。
    ///
    /// # Errors
    ///
    /// 缓冲长度不自洽、表长不符或下标越界时返回说明。
    pub fn forward(idx: &[f32], table: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        let tokens = idx.len();
        if tokens == 0 {
            return Err(Error::new(Code::Invalid, "embed 索引为空"));
        }
        if !out.len().is_multiple_of(tokens) {
            return Err(Error::new(
                Code::Invalid,
                format!("输出长度 {} 不能按 token 数 {tokens} 整除出行宽", out.len()),
            ));
        }
        let dim = out.len() / tokens;
        let gathered = gather(idx, table, dim)?;
        out.copy_from_slice(&gathered);
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Embed;

    /// Embedding 的具体语义别名。
    #[annotation::model(name = "embed_f32", shape = [8, 64], dtype = "f32", skeleton = "embed")]
    pub type EmbedF32 = Embed;

    crate::operator_here!(Embed, "embed_f32", 2, (8, 64), true, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/embed.rs");
    }
}

pub mod rmsnorm {
    //! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
    //!
    //! 两趟遍历同一行：先累加平方和求均方根，再归一化并乘权重。常量 `eps = 1e-6`。本模块自带
    //! 形状、代价、样本、标量参考与遍历，`#[annotation::model]` 把这几个 fn 装配成 `MODEL`。
    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
    use diagnostic::{Code, Error};

    /// 数值稳定项：均方根里加的小常数。
    pub const EPS: f32 = 1e-6;

    // 单轴定义与形状投影复用 unary 骨架：本文件只声明 rmsnorm 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：两趟读 `x`、一趟读 `gamma`、一趟写 `y`，运算按每元素约 4 次（平方、加法、
    /// 除法、乘法）估算。这里给出可审计的显式口径，不伪装成一条 einsum。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        cost_n(shape, "rmsnorm", 4.0, 3)
    }

    /// 确定性样本：`x`、`gamma`、`y` 三个等长缓冲。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            buffers: vec![
                x_buffer(len),
                Buffer {
                    name: "gamma",
                    values: Values::F32((0..len).map(|i| 0.5 + (i as f32) * 0.01).collect()),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 参考两参前置：长度一致且非空（空行没有可统计的均方根）。
    fn check_ref(x: &[f32], gamma: &[f32]) -> diagnostic::Result<()> {
        if x.len() != gamma.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x 与 gamma 长度不一致：{} 与 {}", x.len(), gamma.len()),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "rmsnorm 输入不能为空：均方根无定义",
            ));
        }
        Ok(())
    }

    /// 内核三参前置：等长且非空。
    fn check_kernel(x: &[f32], gamma: &[f32], out: &[f32]) -> diagnostic::Result<()> {
        if x.len() != gamma.len() || x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲长度不一致：x {}、gamma {}、out {}",
                    x.len(),
                    gamma.len(),
                    out.len()
                ),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "rmsnorm 输入不能为空：均方根无定义",
            ));
        }
        Ok(())
    }

    /// 归一化第一趟：平方和。累加是串行依赖，保留逐元素循环；参考与内核共用同一份口径。
    fn sum_sq(x: &[f32]) -> f32 {
        let mut sum = 0.0;
        for value in x {
            sum += value * value;
        }
        sum
    }

    /// 标量参考：两趟，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 输入为空时返回说明：空行没有可统计的均方根。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        check_ref(x, gamma)?;
        let rms = (sum_sq(x) / (x.len() as f32) + EPS).sqrt();
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(value, weight)| weight * value / rms)
            .collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 三个缓冲等长，或输入为空时返回说明：空行没有可统计的均方根。
    pub fn forward(x: &[f32], gamma: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        check_kernel(x, gamma, out)?;
        let rms = (sum_sq(x) / (x.len() as f32) + EPS).sqrt();
        for ((out, x), gamma) in out.iter_mut().zip(x).zip(gamma) {
            *out = gamma * x / rms;
        }
        Ok(())
    }

    /// 批段前置：行宽非空、gamma 与行宽等长、x/out 等长且为行宽整数倍；返回批内段数。
    fn rmsnorm_batch_segments(
        x: &[f32],
        gamma: &[f32],
        out: &[f32],
        row_dim: usize,
    ) -> Result<usize, Error> {
        if row_dim == 0 {
            return Err(Error::new(Code::Invalid, "rmsnorm 行宽不能为 0"));
        }
        if gamma.len() != row_dim {
            return Err(Error::new(
                Code::Invalid,
                format!("gamma 长度 {} 不等于行宽 {row_dim}", gamma.len()),
            ));
        }
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
            ));
        }
        check_batch_rows(x, row_dim)?;
        Ok(x.len() / row_dim)
    }

    /// 批行校验：缓冲非空且为行宽整数倍；空行没有可统计的均方根。
    fn check_batch_rows(x: &[f32], row_dim: usize) -> Result<(), Error> {
        if x.is_empty() || !x.len().is_multiple_of(row_dim) {
            return Err(Error::new(Code::Invalid, "rmsnorm 批缓冲不是行宽整数倍"));
        }
        Ok(())
    }

    /// 批量入口：`[batch, row_dim]` 段连续，每段独立 RMS 归一化，gamma 按行宽复用。
    ///
    /// # Errors
    ///
    /// 行宽为 0、gamma 与行宽不等长、x/out 不等长、缓冲非行宽倍数或为空时返回说明。
    pub fn forward_batch(
        x: &[f32],
        gamma: &[f32],
        out: &mut [f32],
        row_dim: usize,
    ) -> diagnostic::Result<()> {
        let batch = rmsnorm_batch_segments(x, gamma, out, row_dim)?;
        for b in 0..batch {
            let start = b * row_dim;
            forward(
                &x[start..start + row_dim],
                gamma,
                &mut out[start..start + row_dim],
            )?;
        }
        Ok(())
    }

    /// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
    #[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
    pub type RmsNormF32 = RmsNorm;

    crate::operator_here!(RmsNorm, "rmsnorm_f32", 2, (4096, 4096), true, reference);

    /// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
    pub struct RmsNorm;

    #[cfg(test)]
    mod tests {
        include!("../tests/rmsnorm.rs");
    }
}

pub mod rope {
    //! RoPE（旋转位置编码）：按位置 `p` 对一对对分量做二维旋转。
    //!
    //! 对偶数下标 `2i` 与奇数下标 `2i+1`，频率 `freq_i = base^(-2i/d)`，角度 `angle = p*freq_i`：
    //!
    //! ```text
    //! y[2i]   = x[2i] cos(angle) - x[2i+1] sin(angle)
    //! y[2i+1] = x[2i] sin(angle) + x[2i+1] cos(angle)
    //! ```
    //!
    //! 常量 `base = 10000`。本模块自带形状、代价、样本、标量参考与遍历，`#[annotation::model]` 装配。
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
    use diagnostic::{Code, Error};

    /// 频率底数。
    pub const BASE: f32 = 10000.0;

    thread_local! {
        /// 段长 → 频率表。`freq_i = base^(-2i/d)` 只依赖 `d`，与位置、数据都无关。
        ///
        /// 单槽就够：一份模型的所有层、所有头、每一步用的都是同一个头宽，跨调用几乎必然命中；
        /// 两个头宽交替时退化成"每次重建"，与没有这张表时同价——缓存只可能等价或更快。
        /// 存的是 `&'static` 之外的纯函数值，不参与任何判定，所以确定性不变。
        static FREQUENCIES: RefCell<Option<(usize, Rc<[f32]>)>> = const { RefCell::new(None) };
    }

    /// 段长 `d` 的频率表：`d/2` 条，`freq[i] = base^(-2i/d)`。
    ///
    /// 缓存的是**同一个表达式**的结果，而不是 `freq *= ratio` 那样的递推近似：递推会随 `i`
    /// 累积约 `d·eps` 的漂移（`d = 1024` 实测 7.7e-7），而这里与逐项 `powf` 逐位相同，
    /// `rope_matches_the_scalar_reference` 的逐位断言因此不受影响。
    ///
    /// 每对一次 `powf` 是这条路径上最贵的一条指令：预计算把 `len = 1024` 的单头旋转从
    /// 2313 ns 降到 881 ns（2.6×），多头与批形态更是把它从 `heads` 次调用压到 1 次。
    pub fn frequencies(d: usize) -> Rc<[f32]> {
        FREQUENCIES.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some((cached, table)) = slot.as_ref()
                && *cached == d
            {
                return Rc::clone(table);
            }
            let table: Rc<[f32]> = (0..d / 2)
                .map(|i| BASE.powf(-((2 * i) as f32) / d as f32))
                .collect();
            *slot = Some((d, Rc::clone(&table)));
            table
        })
    }

    // 单轴定义与形状投影复用 unary 骨架。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：每对分量约 6 次运算（两对乘加 + 一次 sin/cos 视为共享），按元素约 4 次估算。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        cost_n(shape, "rope", 4.0, 1)
    }

    /// 确定性样本：`x`、`y` 两个等长（偶数）缓冲，位置固定为 7。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64), ("pos", 7)],
            buffers: vec![
                x_buffer(len),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 标量参考：与内核同一公式，位置取样本标量 `P`。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let pos = sample
            .scalar("pos")
            .ok_or_else(|| Error::new(Code::Invalid, "rope 样本缺标量 `pos`"))?
            as f32;
        rotate(x, pos)
    }

    /// 共享旋转：参考与内核的同一套数学。末尾若有无法配对的元素，原样透传。
    fn rotate(x: &[f32], pos: f32) -> diagnostic::Result<Vec<f32>> {
        let mut out = vec![0.0; x.len()];
        rotate_into(x, pos, &mut out)?;
        Ok(out)
    }

    /// 旋转写进给定缓冲：参考与内核共用的算术本体。内核路径由此直写调用方缓冲，
    /// 不再先落一份临时 `Vec` 再整块拷贝；末尾不成对的元素原样透传。
    fn rotate_into(x: &[f32], pos: f32, out: &mut [f32]) -> diagnostic::Result<()> {
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        rotate_rows(x, pos, frequencies(x.len()).as_ref(), out);
        Ok(())
    }

    /// 按已备好的频率表旋转：`freq[i]` 就是第 `i` 对的频率，长度校验已在调用方做完。
    ///
    /// 表由 [`frequencies`] 一次给出、多个头 / 多个序列复用，内层于是只剩 `mul`、`sin_cos`
    /// 与四条乘加——没有 `powf`，也没有长度判断。
    fn rotate_rows(x: &[f32], pos: f32, freq: &[f32], out: &mut [f32]) {
        let d = x.len();
        for (i, &f) in freq.iter().enumerate() {
            let (sin, cos) = (pos * f).sin_cos();
            let (e, o) = (x[2 * i], x[2 * i + 1]);
            out[2 * i] = e * cos - o * sin;
            out[2 * i + 1] = e * sin + o * cos;
        }
        if d % 2 == 1 {
            out[d - 1] = x[d - 1];
        }
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出长度不一致时返回说明；末尾无法配对的奇数元素原样透传（不报错）。
    pub fn forward(pos: usize, x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        rotate_into(x, pos as f32, out)
    }

    /// 校验多头旋转长度并派生头数；非法头宽、不等长或非头宽倍数在此一次性拦下。
    fn rope_heads(x: &[f32], out: &[f32], head_dim: usize) -> diagnostic::Result<usize> {
        if head_dim == 0 || x.len() != out.len() || !x.len().is_multiple_of(head_dim) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "RoPE 头宽 {head_dim} 非法，或长度 {} 与其/输出长度不符",
                    x.len()
                ),
            ));
        }
        Ok(x.len() / head_dim)
    }

    /// 多头旋转：头优先块布局 `[heads, head_dim]`，每段独立按头宽重计频率（段内下标从 0 起），
    /// 不把整段长度当一个头宽；段内旋转算术复用 `rotate_into`，这里只做头分段编排。
    ///
    /// # Errors
    ///
    /// 头宽为 0、缓冲不是头宽的整数倍，或输入输出长度不一致时返回说明。
    pub fn forward_heads(
        pos: usize,
        x: &[f32],
        out: &mut [f32],
        head_dim: usize,
    ) -> diagnostic::Result<()> {
        rope_heads(x, out, head_dim)?;
        rotate_heads(pos, x, out, head_dim, frequencies(head_dim).as_ref());
        Ok(())
    }

    /// 按已备好的频率表旋转整段头优先块：`heads` 个头，每头 `head_dim`，段内下标从 0 重计。
    ///
    /// 头与头之间只有偏移不同，频率表共用一份——`powf` 因此每段只算 `head_dim/2` 次，而不是每头一次。
    fn rotate_heads(pos: usize, x: &[f32], out: &mut [f32], head_dim: usize, freq: &[f32]) {
        for start in (0..x.len()).step_by(head_dim) {
            let span = start..start + head_dim;
            rotate_rows(&x[span.clone()], pos as f32, freq, &mut out[span]);
        }
    }

    /// 批量校验：返回单序列段长 `heads·head_dim`；批数取 `positions.len()`，段内头宽/倍数守卫
    /// 交给 [`forward_heads`]。
    fn rope_batch_segments(
        positions: &[usize],
        x: &[f32],
        out: &[f32],
        head_dim: usize,
    ) -> diagnostic::Result<usize> {
        let batch = positions.len();
        if batch == 0 || head_dim == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("批量序列数 {batch} 与头宽 {head_dim} 都必须非零"),
            ));
        }
        if x.len() != out.len() || !x.len().is_multiple_of(batch * head_dim) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "x 长度 {} 不是 batch·head_dim（{}）的整数倍，或与输出不等长",
                    x.len(),
                    batch * head_dim
                ),
            ));
        }
        Ok(x.len() / batch)
    }

    /// 批量旋转：`[batch, heads, head_dim]` 头优先块，第 `b` 段连续 `heads·head_dim`，每序列
    /// 独立位置 `positions[b]`；段内复用 [`forward_heads`] 头分段编排，段间互不串段。
    ///
    /// # Errors
    ///
    /// `positions` 为空、头宽为 0、输入输出不等长或长度不是批段整数倍时返回说明。
    pub fn forward_batch(
        positions: &[usize],
        x: &[f32],
        out: &mut [f32],
        head_dim: usize,
    ) -> diagnostic::Result<()> {
        let seg = rope_batch_segments(positions, x, out, head_dim)?;
        // 先把每个序列的段形状校验完，再统一写：任一序列非法时不会留下写了一半的输出。
        for start in (0..x.len()).step_by(seg) {
            let span = start..start + seg;
            rope_heads(&x[span.clone()], &out[span], head_dim)?;
        }
        let freq = frequencies(head_dim);
        for (seq, &pos) in positions.iter().enumerate() {
            let start = seq * seg;
            let span = start..start + seg;
            rotate_heads(
                pos,
                &x[span.clone()],
                &mut out[span],
                head_dim,
                freq.as_ref(),
            );
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Rope;

    /// RoPE 的具体语义别名。
    #[annotation::model(name = "rope_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rope")]
    pub type RopeF32 = Rope;

    crate::operator_here!(Rope, "rope_f32", 1, (4096, 4096), true, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/rope.rs");
    }
}

pub mod softmax {
    //! Softmax：`y[n] = exp(x[n] - max(x)) / sum_k exp(x[k] - max(x))`。
    //!
    //! 三趟遍历同一行：先求最大值（数值稳定，减去它再取指数不会溢出），再求指数和，最后归一化。
    //! 本模块自带形状、代价、样本、标量参考与遍历，`#[annotation::model]` 装配成 `MODEL`。
    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
    use diagnostic::{Code, Error};

    // 单轴定义与形状投影复用 unary 骨架：本文件只声明 softmax 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：三趟读 `x`、一趟写 `y`，每元素约 3 次运算（max、exp、除法）。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
        cost_n(shape, "softmax", 3.0, 3)
    }

    /// 确定性样本：`x`、`y` 两个等长缓冲。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            buffers: vec![
                x_buffer(len),
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 标量参考：三趟，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 输入为空时返回说明：空行没有可归一化的分母。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "softmax 输入不能为空：归一化分母无定义",
            ));
        }
        let max = x.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        let exps: Vec<f32> = x
            .iter()
            .map(|value| {
                let e = (value - max).exp();
                sum += e;
                e
            })
            .collect();
        Ok(exps.iter().map(|e| e / sum).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出等长，或输入为空时返回说明：空行没有可归一化的分母。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "softmax 输入不能为空：归一化分母无定义",
            ));
        }
        let max = x.iter().copied().fold(f32::MIN, f32::max);
        // 一趟 `exp`：先用写入的指数值累加分母，再原地除掉。旧实现算了两遍 `exp`
        // （一遍求 `sum`、一遍求结果），而 `exp` 正是这里最贵的指令；`(x-max).exp()`
        // 的取值与从左到右的累加顺序都和旧实现一致，所以输出逐位不变。
        let mut sum = 0.0;
        for (out, x) in out.iter_mut().zip(x) {
            let e = (x - max).exp();
            sum += e;
            *out = e;
        }
        for out in out.iter_mut() {
            *out /= sum;
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Softmax;

    /// Softmax 的具体语义别名。
    #[annotation::model(name = "softmax_f32", shape = [4096, 4096], dtype = "f32", skeleton = "softmax")]
    pub type SoftmaxF32 = Softmax;

    crate::operator_here!(Softmax, "softmax_f32", 1, (4096, 4096), true, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/softmax.rs");
    }
}
