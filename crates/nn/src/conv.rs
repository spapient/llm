//! 神经算子：卷积 / 反卷积（`geom` / `pool` / `interp` / `deconv`）、注意力（`attention`）、
//! 池化与范数（`classic`）、扫描骨架（`scan`）。

pub mod attention {
    //! 注意力：`O = softmax(QKᵀ/√d)·V`。

    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx};
    use diagnostic::{Code, Error};

    /// 两条基本轴：序列轴 `s`、头宽轴 `d`，都既是输入也是输出。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 's',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'd',
        },
    ];

    /// `(m, n)` 即 `(序列长 s, 头宽 d)`。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(s: usize, d: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('s', s), ('d', d)])
    }

    /// 取形状轴；缺失时按模块前缀与轴名报错，避免各调用点各写一遍分支。
    pub(crate) fn axis(shape: &ShapeCtx, what: &str, label: char) -> Result<usize, Error> {
        shape
            .get(label)
            .ok_or_else(|| Error::new(Code::Invalid, format!("{what} 形状缺轴 `{label}`")))
    }

    /// 标量代价：分数矩阵 `s²` 个点（每个 d 维内积）+ softmax + 对 `V` 的加权。
    ///
    /// # Errors
    ///
    /// 缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        let s = axis(shape, "attention", 's')?;
        let d = axis(shape, "attention", 'd')?;
        Ok(Cost {
            flops: (s as f64) * (s as f64) * ((d as f64) * 2.0 + 3.0),
            read: (s as u64) * (d as u64) * 4 * 3,
            write: (s as u64) * (d as u64) * 4,
        })
    }

    /// 确定性 q/k/v/y 缓冲：`q` 与 `y` 取 `q_len`，`k`/`v` 取 `kv_len`。
    pub(crate) fn sample_bufs(q_len: usize, kv_len: usize) -> Vec<Buf> {
        let buf = Buf::f32;
        vec![
            buf("q", crate::model::linear(q_len, 0.13, -0.5)),
            buf("k", crate::model::linear(kv_len, -0.09, 0.4)),
            buf("v", crate::model::linear(kv_len, 0.11, -0.2)),
            buf("y", vec![0.0; q_len]),
        ]
    }

    /// 确定性样本：`q`/`k`/`v` 三个输入与输出 `o`，均为 `s*d`。
    pub fn sample(s: usize, d: usize) -> Sample {
        let len = s * d;
        Sample {
            m: s,
            n: d,
            scalars: vec![("S", s as i64), ("D", d as i64)],
            bufs: sample_bufs(len, len),
        }
    }

    /// 三个输入缓冲：朴素与 flash 路径读同一份 `q/k/v`，缺失时的文案因此只写一次。
    pub(crate) fn qkv(sample: &Sample) -> diagnostic::Result<(&[f32], &[f32], &[f32])> {
        Ok((sample.f32("q")?, sample.f32("k")?, sample.f32("v")?))
    }

    /// 行 `row` 与行 `col` 的点积：四个朴素/flash 实现共用，免得每处重写 `d` 长的累加循环。
    fn dot(x: &[f32], y: &[f32], row: usize, col: usize, d: usize) -> f32 {
        crate::matmul::dot_lanes(&x[row * d..][..d], &y[col * d..][..d])
    }

    /// 头宽不能为 0：`√d` 作分数除数，`d=0` 会让每条分数除零成 NaN。
    pub(crate) fn check_width(d: usize) -> diagnostic::Result<()> {
        if d == 0 {
            return Err(Error::new(Code::Invalid, "注意力头宽不能为 0"));
        }
        Ok(())
    }

    /// 注意力的一行机器：三个操作数与 K/V 列几何（`cols`、读宽 `d`、写宽 `write`、`√d`）——
    /// 稠密注意力与潜空间注意力共用同一份算式，差只差在写宽可以比读宽窄。
    pub(crate) struct Rows<'a> {
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        cols: usize,
        d: usize,
        write: usize,
        scale: f32,
    }

    impl<'a> Rows<'a> {
        /// 以操作数与 K/V 列几何建行机器：`scale = √d`。
        pub(crate) fn new(q: &'a [f32], k: &'a [f32], v: &'a [f32], cols: usize, d: usize) -> Self {
            Self {
                q,
                k,
                v,
                cols,
                d,
                write: d,
                scale: (d as f32).sqrt(),
            }
        }

        /// 窄写版本：读宽 `d`、写宽 `write`（`v` 的列宽与输出列宽都是 `write`）。
        fn narrow(
            q: &'a [f32],
            k: &'a [f32],
            v: &'a [f32],
            cols: usize,
            d: usize,
            write: usize,
        ) -> Self {
            Self {
                write,
                ..Self::new(q, k, v, cols, d)
            }
        }

        /// 从样本取三个操作数。
        fn of(sample: &'a Sample) -> diagnostic::Result<Self> {
            let (q, k, v) = qkv(sample)?;
            check_width(sample.n)?;
            Ok(Self::new(q, k, v, sample.m, sample.n))
        }

        /// 朴素路径的一行：保留原迭代器顺序，缓冲容量由整次前向复用。
        fn naive(&self, row: usize, scores: &mut Vec<f32>, exps: &mut Vec<f32>, out: &mut [f32]) {
            scores.clear();
            scores.extend(
                (0..self.cols).map(|col| dot(self.q, self.k, row, col, self.d) / self.scale),
            );
            let max = scores.iter().copied().fold(f32::MIN, f32::max);
            exps.clear();
            exps.extend(scores.iter().map(|score| (score - max).exp()));
            let sum: f32 = exps.iter().sum();
            // `V` 按 `col` 分块，每块是 `write` 个输出列；按块求和让行列关系写在类型系统里，
            // 而不是靠 `j * write + h` 手算偏移。
            let targets = &mut out[row * self.write..][..self.write];
            for (h, slot) in targets.iter_mut().enumerate() {
                *slot = exps
                    .iter()
                    .zip(self.v.chunks_exact(self.write))
                    .map(|(prob, vrow)| (prob / sum) * vrow[h])
                    .sum();
            }
        }

        /// online（flash）路径的一行：跑一遍 `col` 维护最大值、归一化分母与累加向量。
        fn online(&self, row: usize, acc: &mut [f32]) -> f32 {
            let mut max = f32::MIN;
            let mut sum = 0.0f32;
            acc.fill(0.0);
            for col in 0..self.cols {
                let score = dot(self.q, self.k, row, col, self.d) / self.scale;
                let new_max = max.max(score);
                let reweight = (max - new_max).exp();
                let prob = (score - new_max).exp();
                sum = sum * reweight + prob;
                let vrow = &self.v[col * self.write..][..self.write];
                for (slot, value) in acc.iter_mut().zip(vrow) {
                    *slot = *slot * reweight + prob * value;
                }
                max = new_max;
            }
            sum
        }
    }

    /// 一行机器两种实现共用的执行开关：朴素直接写 `out`，online 先累加再除归一化和。
    #[derive(Clone, Copy)]
    pub(crate) enum Mode {
        /// 三阶段朴素：落地全部分数再 softmax。
        Naive,
        /// online softmax：单遍 K/V，不落地分数矩阵。
        Online,
    }

    /// 按执行模式跑 `rows_count` 行，结果写进 `out`。
    pub(crate) fn run_rows(rows: &Rows<'_>, rows_count: usize, out: &mut [f32], mode: Mode) {
        match mode {
            Mode::Naive => {
                let (mut scores, mut exps) =
                    (Vec::with_capacity(rows.cols), Vec::with_capacity(rows.cols));
                for row in 0..rows_count {
                    rows.naive(row, &mut scores, &mut exps, out);
                }
            }
            Mode::Online => {
                // 预分配一条累加缓冲，所有行复用；online 每行 fill 清零后写回。
                // 消除热路径上「每行一次堆分配」，prefill（rows_count = s）收益随序列长增长。
                let mut acc = vec![0.0f32; rows.write];
                for row in 0..rows_count {
                    let sum = rows.online(row, &mut acc);
                    for h in 0..rows.write {
                        out[row * rows.write + h] = acc[h] / sum;
                    }
                }
            }
        }
    }

    /// 朴素参考：落地完整分数矩阵，逐行 softmax，再乘 `V`。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let rows = Rows::of(sample)?;
        let mut out = vec![0.0f32; rows.cols * rows.d];
        run_rows(&rows, rows.cols, &mut out, Mode::Naive);
        Ok(out)
    }

    /// Flash 参考：online softmax 分块（这里块取整条序列，语义与分块等价），单遍 K/V。
    pub fn flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let rows = Rows::of(sample)?;
        let mut out = vec![0.0f32; rows.cols * rows.d];
        run_rows(&rows, rows.cols, &mut out, Mode::Online);
        Ok(out)
    }

    /// 朴素内核本体：prefill 自注意力是 `decode` 的 `q_rows = hist = s` 特例，结果写进 `out`。
    ///
    /// `q`/`k`/`v` 都是 `s × d` 连续二维视图，序列长与头宽从视图形状取；长度不符仍由内部守卫拦下。
    ///
    /// # Errors
    ///
    /// 秩不是二维、长度不符、头宽为 0，或序列长为 0 时返回说明。
    pub fn forward(
        q: tensor::View<'_, f32>,
        k: tensor::View<'_, f32>,
        v: tensor::View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let (s, d) = crate::matmul::dim2(q.dims(), "attention_f32")?;
        crate::matmul::dim2(k.dims(), "attention_f32")?;
        crate::matmul::dim2(v.dims(), "attention_f32")?;
        decode(
            q.as_slice(),
            k.as_slice(),
            v.as_slice(),
            out,
            s,
            s,
            d,
            Mode::Naive,
        )
    }

    /// Flash 内核本体：online softmax，单遍 K/V，写进 `out`。prefill 口径同 [`forward`]。
    ///
    /// # Errors
    ///
    /// 秩不是二维、长度不符、头宽为 0，或序列长为 0 时返回说明。
    pub fn flash_forward(
        q: tensor::View<'_, f32>,
        k: tensor::View<'_, f32>,
        v: tensor::View<'_, f32>,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let (s, d) = crate::matmul::dim2(q.dims(), "flash_attention_f32")?;
        crate::matmul::dim2(k.dims(), "flash_attention_f32")?;
        crate::matmul::dim2(v.dims(), "flash_attention_f32")?;
        decode(
            q.as_slice(),
            k.as_slice(),
            v.as_slice(),
            out,
            s,
            s,
            d,
            Mode::Online,
        )
    }

    use crate::conv::decode;

    pub mod latent {
        //! 潜空间注意力（MLA 吸收式单步）：`O[r,c] = Σ_j softmax_j(q[r]·k[j]/√W) · v[j,c]`。
        //!
        //! 行 = `(查询行 × 头)` 压平，这与 [`attention`](super) 的几何完全同构；差别只有一处：**输出宽度
        //! `dc` 可以小于读入宽度 `W`**。压缩存的是 `W`（潜向量 ‖ 解耦 RoPE 段），头内乘积写出的是 `dc`，
        //! 上投影 `W^{UV}` 就折在这个口径差里。实测几何（`d_c = 512`、`d_h^R = 64`）是 `W = 576 > dc`，
        //! 所以不能反过来把 `dc` 当声明轴——那样真实几何根本喂不进来。
        //!
        //! RoPE 段必须对同一查询行的各头相同：这是 MLA 的解耦键只有一个、被所有头共享的直接后果。构造
        //! `q` 时把这段复制到每个头，本算子照常按整行做内积即可，不需要知道"哪几列是 RoPE"。

        use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx};
        use diagnostic::{Code, Error};

        use super::axis;

        /// 两个轴：压平后的查询行 `r`、读入行宽 `w`。
        pub const AXES: &[AxisSpec] = &[
            AxisSpec {
                role: AxisRole::Input,
                label: 'r',
            },
            AxisSpec {
                role: AxisRole::Output,
                label: 'w',
            },
        ];

        /// 输出缓冲名：只用来在样本里量出 `dc`。
        const OUT: &str = "y";

        /// `(m, n)` 即 `(压平后的查询行数 r, 读入行宽 W)`。
        ///
        /// # Errors
        ///
        /// 当前实现不会失败。
        pub fn project_shape(r: usize, w: usize) -> Result<ShapeCtx, Error> {
            ShapeCtx::new([('r', r), ('w', w)])
        }

        /// 标量代价：每个查询行对 `t` 条键各做一次宽 `W` 的内积，再按 `dc` 列加权。
        ///
        /// 计价口径取 `dc = W / 2`：声明只有两个轴，`dc` 不进去，只能按同口径的半宽估；真实 `dc` 更小
        /// 时实际代价更低，所以这是上界而不是虚报。
        ///
        /// # Errors
        ///
        /// 缺轴时返回说明。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            let r = axis(shape, "latent_attention", 'r')?;
            let w = axis(shape, "latent_attention", 'w')?;
            let (dot, weighted) = pricing(r, w);
            Ok(Cost {
                flops: dot + weighted,
                read: (rows_of(r, w) * 4) as u64,
                write: (r * w.div_ceil(2) * 4) as u64,
            })
        }

        /// 分数段与加权段的浮点计数。
        fn pricing(r: usize, w: usize) -> (f64, f64) {
            let dc = w.div_ceil(2);
            let rows = r as f64;
            let t = r as f64;
            (rows * t * w as f64 * 2.0, rows * t * dc as f64 * 2.0)
        }

        /// 读侧元素数：`q`/`k` 是宽 `W` 的行，`v` 是宽 `W/2` 的行。
        fn rows_of(r: usize, w: usize) -> usize {
            let dc = w.div_ceil(2);
            r * w + r * w + r * dc
        }

        /// 确定性样本：`q`/`k` 是 `r × W`，`v` 是 `r × W/2`（前视同构，键行数取 `r`），`y` 是 `r × W/2`。
        ///
        /// 四个标量是 `.k` 的签名实参（`R` 行数、`T` 键行数、`W` 读宽、`DC` 写宽）——它们是同一份几何的
        /// 另一处投影，不是额外的自由度：调用方按长度算出后传进去，与算子内部的折法一致。
        pub fn sample(m: usize, n: usize) -> Sample {
            let (r, w) = (m.max(1), n.max(1));
            let dc = w.div_ceil(2);
            let buf = Buf::f32;
            Sample {
                m: r,
                n: w,
                scalars: vec![
                    ("R", r as i64),
                    ("T", r as i64),
                    ("W", w as i64),
                    ("DC", dc as i64),
                ],
                bufs: vec![
                    buf("q", crate::model::linear(r * w, 0.13, -0.5)),
                    buf("k", crate::model::linear(r * w, -0.05, 0.17)),
                    buf("v", crate::model::linear(r * dc, 0.09, -0.23)),
                    buf(OUT, vec![0.0; r * dc]),
                ],
            }
        }

        /// 一次调用的几何：压平行数、键行数、读宽、写宽——四个数都由长度直接折出，没有标量可依赖。
        type Geometry = (usize, usize, usize, usize);

        /// 由样本声明与输出长度定几何（前视同构：键行数取查询行数）。
        fn sample_geometry(sample: &Sample) -> diagnostic::Result<Geometry> {
            let (r, w) = (sample.m, sample.n);
            Ok((r, r, w, sample.f32(OUT)?.len() / r.max(1)))
        }

        /// 由运行期张量定几何。
        ///
        /// # Errors
        ///
        /// 长度不是读宽的整数倍时返回说明。
        fn tensor_geometry(
            q: usize,
            k: usize,
            out: usize,
            w: usize,
        ) -> diagnostic::Result<Geometry> {
            let site = "latent_attention_f32";
            let rows = fold(site, q, w, "q")?;
            Ok((rows, fold(site, k, w, "k")?, w, out / rows.max(1)))
        }

        /// 按读宽把长度折成行数。
        fn fold(site: &str, len: usize, w: usize, what: &str) -> diagnostic::Result<usize> {
            if len.is_multiple_of(w) {
                return Ok(len / w);
            }
            Err(Error::new(
                Code::Invalid,
                format!("{site} 的 `{what}` 长度 {len} 不是行宽 {w} 的整数倍"),
            ))
        }

        /// 参考与内核共用父模块的行机器，只是写宽比读宽窄；两条入口分别落在两种算法上，
        /// 于是对拍仍然是在比算法，而不是在比同一段代码。
        fn machine<'a>(q: &'a [f32], k: &'a [f32], v: &'a [f32], g: Geometry) -> super::Rows<'a> {
            super::Rows::narrow(q, k, v, g.1, g.2, g.3)
        }

        /// 参考：`online softmax` 单遍 K/V，逐行归一化后写进压缩宽度。
        ///
        /// # Errors
        ///
        /// 操作数缺失或长度不符时返回说明。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let g = sample_geometry(sample)?;
            let (q, k, v) = super::qkv(sample)?;
            let mut out = vec![0.0f32; g.0 * g.3];
            super::run_rows(&machine(q, k, v, g), g.0, &mut out, super::Mode::Online);
            Ok(out)
        }

        /// 可执行内核：`naive` 三阶段（落地分数再归一化），几何只从张量长度取，不借道样本结构。
        ///
        /// `q`/`k` 是 `r × W` / `t × W` 连续二维视图，读宽从视图形状取；`v` 与输出是压缩宽度 `dc`，
        /// 仍保留裸切片（写宽与读宽不同，不进同一张量视图）。
        ///
        /// # Errors
        ///
        /// 秩不是二维、宽度不匹配，或长度不整除时返回说明。
        pub fn forward(
            q: tensor::View<'_, f32>,
            k: tensor::View<'_, f32>,
            v: &[f32],
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            let (_, w) = crate::matmul::dim2(q.dims(), "latent_attention_f32")?;
            let (_, kw) = crate::matmul::dim2(k.dims(), "latent_attention_f32")?;
            if kw != w {
                return Err(Error::new(
                    Code::Invalid,
                    format!("latent_attention_f32 读宽不符：q·w={w}，k·w={kw}"),
                ));
            }
            let g = tensor_geometry(q.as_slice().len(), k.as_slice().len(), out.len(), w)?;
            super::run_rows(
                &machine(q.as_slice(), k.as_slice(), v, g),
                g.0,
                out,
                super::Mode::Naive,
            );
            Ok(())
        }

        /// 标记类型：承载 `#[model]` 语义。
        pub struct LatentAttention;

        /// 潜空间注意力的语义别名；codegen 与 `nn` 都按 `latent_attention_f32` 这个名字取它。
        #[annotation::model(name = "latent_attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
        pub type LatentAttentionF32 = LatentAttention;

        // 基座契约（ADR 0050）：三个操作数（q、k、v），输出 y；写宽比读宽窄，反向按两口径散射
        // （dV 走压缩宽度、dQ/dK 走读入宽度），公式在 `crates/train` 的公式目录里。
        crate::operator_here!(
            LatentAttention,
            "latent_attention_f32",
            3,
            (8, 8),
            true,
            reference
        );
    }
    pub use decode::{
        decode_flash_forward, decode_flash_forward_batch, decode_flash_forward_batch_var,
        decode_flash_forward_heads, decode_forward, decode_forward_batch, decode_forward_batch_var,
        decode_forward_heads,
    };
    pub use latent::{LatentAttention, LatentAttentionF32, forward as latent_forward};

    use decode::decode;

    // 语义表对外的名字保持 `attention::decode_attention`（`nn::conv::decode_attention` 由 lib.rs 再导出）。
    // 写 `super::decode` 而非 `decode`：`decode` 实体是 `conv::decode`（本模块的 super 层），
    // 镜像门禁据此把别名解析到真实实体文件，而不是误拼成 `conv::attention::decode`。
    pub use super::decode::semantics as decode_attention;

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Attention;

    /// 朴素注意力语义别名。
    #[annotation::model(name = "attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
    pub type AttentionF32 = Attention;

    /// Flash 注意力标记类型：形状、样本、代价与朴素 [`Attention`] 完全共用，只把标量参考换成
    /// online softmax 版本（[`flash_reference`]）。`const = "FLASH_MODEL"` 让两个算子同模块各持一份语义表。
    pub struct FlashAttention;

    /// Flash 注意力语义别名；codegen 按注册名选 [`flash_reference`]。
    #[annotation::model(name = "flash_attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention", const = "FLASH_MODEL")]
    pub type FlashAttentionF32 = FlashAttention;

    // 基座契约（ADR 0050）：两条算子共用形状/代价/样本，只有参考入口分叉。
    crate::operator_here!(Attention, "attention_f32", 3, (8, 8), true, reference);
    crate::operator_here!(
        FlashAttention,
        "flash_attention_f32",
        3,
        (8, 8),
        true,
        flash_reference
    );

    #[cfg(test)]
    mod attention_suite {
        include!("../tests/conv_attention_suite.rs");
    }
}
pub mod classic {
    //! 池化 / 插值 / 归一化的算子声明：几何是声明档固定常量，算术与标量参考直接复用
    //! [`crate::conv`] 与 [`crate::norm`] 的真实现。

    use crate::Scalar;
    use crate::conv::ConvGeom;
    use crate::model::{Buf, Cost, Sample, ShapeCtx, x_buf};
    use crate::unary::{AXES, project_shape};
    use diagnostic::Error;

    /// 池化声明共用的 same 3×3 几何：单位步长、单侧补零 1。
    const POOL_SAME_3X3: ConvGeom = ConvGeom {
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
    };

    /// 一次算子装配：标记类型、语义别名与 [`crate::operator_here`] 的基座实现，外加 `#[annotation::model]`
    /// 自检常量。`$m`/`$n` 同时喂给 `operator_here!`（元组形状）与注解（数组形状），两处从同一对字面量派生。
    macro_rules! classic_op {
    ($ty:ident, $alias:ident, $op:tt, $arity:literal, $m:tt, $n:tt) => {
        #[doc = concat!("`", $op, "` 的标记类型。")]
        #[derive(Debug, Clone, Copy)]
        pub struct $ty;

        crate::operator_here!($ty, $op, $arity, ($m, $n), true, reference);

        #[annotation::model(name = $op, shape = [$m, $n], dtype = "f32", skeleton = "elementwise")]
        /// 语义别名（`#[annotation::model]` 在此生成 `MODEL` 自检常量）。
        pub type $alias = $ty;
    };
}

    /// 窗口族公开面：形状、样本、参考、前向与代价在一处平铺声明；算术仍由模块的 `run` 提供。
    macro_rules! window_op {
        ($ty:ident, $alias:ident, $op:tt, $m:tt, $n:tt, $in_len:expr, $out_len:expr,
     $flops:expr, $reads:expr) => {
            /// 构造这一窗口声明档的确定性输入与输出缓冲。
            pub fn sample(_: usize, _: usize) -> Sample {
                Sample::f32_io(
                    $out_len,
                    $out_len,
                    $in_len,
                    $out_len,
                    vec![("N", $out_len as i64)],
                )
            }

            /// 用模块的真实窗口实现计算参考输出。
            pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
                let x = sample.f32("x")?;
                let mut out = vec![0.0f32; $out_len];
                run::<f32>(x, &mut out)?;
                Ok(out)
            }

            /// 声明档前向：与参考共用同一条 `run`，几何与样本一致。
            pub fn forward(x: tensor::View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
                run::<f32>(x.as_slice(), out)
            }

            /// 标量代价：由声明的浮点运算数与读操作数推出。
            pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
                crate::model::cost_n(shape, $op, $flops, $reads)
            }

            classic_op!($ty, $alias, $op, 1, $m, $n);
        };
    }

    /// 归一化族公开面：共享 64×64 注解档；批／组／实例可省略共同的 `N*S*C` 与 `C` 几何。
    macro_rules! norm_op {
        ($ty:ident, $alias:ident, $op:tt, $total:expr, $channels:expr, $scalars:expr,
     $gamma:expr, $beta:expr, $flops:expr, $reads:expr) => {
            /// 声明档样本：`x` 与 `gamma`/`beta` 的长度由模块几何常量给出。
            pub fn sample(_: usize, _: usize) -> Sample {
                norm_sample($total, $channels, $scalars, $gamma, $beta)
            }

            /// 标量参考：读 `x`/`gamma`/`beta`、跑模块的 `run::<f32>`、返回 `y`（真统计）。
            pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
                norm_reference(sample, $total, run::<f32>)
            }

            /// 声明档前向：与参考共用同一条 `run`，几何与样本一致。
            pub fn forward(
                x: tensor::View<'_, f32>,
                gamma: tensor::View<'_, f32>,
                beta: tensor::View<'_, f32>,
                out: &mut [f32],
            ) -> diagnostic::Result<()> {
                run::<f32>(x.as_slice(), gamma.as_slice(), beta.as_slice(), out)
            }

            /// 标量代价：显式口径，浮点运算与字节读写见本模块装配行的注释。
            pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
                crate::model::cost_n(shape, $op, $flops, $reads)
            }

            classic_op!($ty, $alias, $op, 3, 64, 64);
        };
        ($ty:ident, $alias:ident, $op:tt, $scalars:expr, $gamma:expr, $beta:expr,
     $flops:expr, $reads:expr) => {
            norm_op!(
                $ty,
                $alias,
                $op,
                N * S * C,
                C,
                $scalars,
                $gamma,
                $beta,
                $flops,
                $reads
            );
        };
    }

    /// 归一化骨架的确定性样本：`x`（总长 `total`）、`gamma`/`beta`（各 `channels` 个）、`y`。
    fn norm_sample(
        total: usize,
        channels: usize,
        scalars: Vec<(&'static str, i64)>,
        gamma: f32,
        beta: f32,
    ) -> Sample {
        Sample {
            m: total,
            n: total,
            scalars,
            bufs: vec![
                x_buf(total),
                Buf::f32("gamma", vec![gamma; channels]),
                Buf::f32("beta", vec![beta; channels]),
                Buf::f32("y", vec![0.0; total]),
            ],
        }
    }

    /// 归一化骨架的标量参考：读 `x`/`gamma`/`beta`、跑 `run`、返回 `y`。
    fn norm_reference(
        sample: &Sample,
        total: usize,
        run: impl Fn(&[f32], &[f32], &[f32], &mut [f32]) -> diagnostic::Result<()>,
    ) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        let beta = sample.f32("beta")?;
        let mut out = vec![0.0f32; total];
        run(x, gamma, beta, &mut out)?;
        Ok(out)
    }

    /// 二维卷积（声明档 `4×4, cin 2, cout 3, k=3, s=1, p=1`，same 输出）。
    pub mod conv2d {
        use super::*;

        /// 输入高度。
        pub const H: usize = 4;
        /// 输入宽度或归一化宽度。
        pub const W: usize = 4;
        /// 输入通道数。
        pub const CIN: usize = 2;
        /// 输出通道数。
        pub const COUT: usize = 3;
        /// 卷积核边长。
        pub const K: usize = 3;
        /// 步长或空间尺寸。
        pub const S: usize = 1;
        /// 单侧补零宽度。
        pub const P: usize = 1;
        /// 输出高度。
        pub const OH: usize = 4;
        /// 输出宽度。
        pub const OW: usize = 4;
        /// 单个输出元素对应的卷积输入元素数。
        pub const PATCH: usize = K * K * CIN;

        fn conv() -> crate::conv::Conv2d {
            crate::conv::Conv2d::new(
                CIN,
                COUT,
                ConvGeom {
                    kh: K,
                    kw: K,
                    sh: S,
                    sw: S,
                    ph: P,
                    pw: P,
                },
            )
        }

        /// 构造确定性的卷积输入、权重、偏置与输出样本。
        pub fn sample(_: usize, _: usize) -> Sample {
            Sample {
                m: OH * OW,
                n: COUT,
                scalars: vec![("N", 1)],
                bufs: vec![
                    x_buf(H * W * CIN),
                    Buf::f32(
                        "weight",
                        (0..PATCH * COUT)
                            .map(|i| 0.05 + (i % 7) as f32 * 0.03)
                            .collect(),
                    ),
                    Buf::f32("bias", vec![0.4; COUT]),
                    Buf::f32("y", vec![0.0; OH * OW * COUT]),
                ],
            }
        }

        /// 标量参考：读样本输入 / 权重 / 偏置，跑声明档二维卷积，返回输出向量。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let x = sample.f32("x")?;
            let weight = sample.f32("weight")?;
            let bias = sample.f32("bias")?;
            let mut out = vec![0.0f32; OH * OW * COUT];
            conv().forward_f32(x, weight, bias, 1, (H, W), &mut out)?;
            Ok(out)
        }

        /// 执行声明档二维卷积前向计算。
        pub fn forward(
            x: tensor::View<'_, f32>,
            weight: tensor::View<'_, f32>,
            bias: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            conv().forward_f32(
                x.as_slice(),
                weight.as_slice(),
                bias.as_slice(),
                1,
                (H, W),
                out,
            )
        }

        /// 按声明档形状推导出算子代价（FLOPs 与权重条数）。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            crate::model::cost_n(shape, "conv2d_f32", 2.0 * PATCH as f64, 3)
        }

        classic_op!(Conv2dModel, Conv2d, "conv2d_f32", 3, 16, 3);
    }

    /// 窗口族共用 `window_op!`；子模块只保留各自几何与真执行入口。
    pub mod pool {
        /// 自适应平均池化（声明档 4×4 → 2×2）。
        pub mod adaptive {
            use super::super::*;

            /// 输入高度。
            pub const H: usize = 4;
            /// 输入宽度或归一化宽度。
            pub const W: usize = 4;
            /// 输出高度。
            pub const OH: usize = 2;
            /// 输出宽度。
            pub const OW: usize = 2;

            pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
                crate::conv::AdaptiveAvgPool2d { oh: OH, ow: OW }.forward(x, 1, H, W, 1, out)
            }

            window_op!(
                AdaptivePoolModel,
                AdaptivePool,
                "adaptive_pool_f32",
                64,
                64,
                H * W,
                OH * OW,
                5.0,
                2
            );
        }

        /// 二维平均池化（声明档几何 `3×3, s=1, p=1`，补零不计入分母）。
        pub mod avg2d {
            use super::super::*;

            /// 输入高度。
            pub const H: usize = 4;
            /// 输入宽度或归一化宽度。
            pub const W: usize = 4;

            fn pool() -> crate::conv::AvgPool2d {
                crate::conv::AvgPool2d {
                    geom: POOL_SAME_3X3,
                    count_include_pad: false,
                }
            }

            pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
                pool().forward(x, 1, H, W, 1, out)
            }

            window_op!(
                AvgPool2dModel,
                AvgPool2d,
                "avg_pool2d_f32",
                64,
                64,
                H * W,
                H * W,
                10.0,
                4
            );
        }

        /// 一维最大池化（声明档几何 `k=2, s=2, p=0`，单通道）。
        pub mod max1d {
            use super::super::*;

            /// 输出序列长度。
            pub const OL: usize = 8;
            /// 输入序列长度。
            pub const L: usize = 16;

            pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
                crate::conv::MaxPool1d { k: 2, s: 2, p: 0 }.forward(x, 1, L, 1, out)
            }

            window_op!(
                MaxPool1dModel,
                MaxPool1d,
                "max_pool1d_f32",
                64,
                64,
                L,
                OL,
                2.0,
                2
            );
        }

        /// 二维最大池化（声明档几何 `3×3, s=1, p=1`，same 输出）。
        pub mod max2d {
            use super::super::*;

            /// 输入高度。
            pub const H: usize = 4;
            /// 输入宽度或归一化宽度。
            pub const W: usize = 4;

            pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
                crate::conv::MaxPool2d {
                    geom: POOL_SAME_3X3,
                }
                .forward(x, 1, H, W, 1, out)
            }

            window_op!(
                MaxPool2dModel,
                MaxPool2d,
                "max_pool2d_f32",
                64,
                64,
                H * W,
                H * W,
                8.0,
                4
            );
        }
    }

    /// 双线性插值（声明档 2×2 → 4×4，`align_corners = false`）。
    pub mod interpolate {
        use super::*;

        /// 输入高度。
        pub const H: usize = 2;
        /// 输入宽度或归一化宽度。
        pub const W: usize = 2;
        /// 输出高度。
        pub const OH: usize = 4;
        /// 输出宽度。
        pub const OW: usize = 4;

        pub(crate) fn run<S: Scalar>(x: &[S], out: &mut [S]) -> diagnostic::Result<()> {
            crate::conv::Interpolate {
                sampler: crate::conv::Sampler::Linear,
                oh: OH,
                ow: OW,
            }
            .forward(x, 1, H, W, 1, out)
        }

        window_op!(
            InterpolateModel,
            Interpolate,
            "interpolate_f32",
            64,
            64,
            H * W,
            OH * OW,
            8.0,
            4
        );
    }

    /// 四种归一化只保留各自统计域与声明档参数。
    pub mod norm {
        /// 批／组／实例归一化共享的通道数。
        pub const C: usize = 4;
        /// 共享的空间尺寸。
        pub const S: usize = 2;
        /// 共享的批大小。
        pub const N: usize = 2;
        /// 四种归一化共享的数值稳定性 epsilon。
        pub const EPS: f32 = 1e-5;

        /// 批归一化（声明档 `C=4, spatial=2, n=2`）。
        pub mod batch {
            use super::super::*;
            pub use super::{C, EPS, N, S};

            pub(crate) fn run<S: Scalar>(
                x: &[S],
                gamma: &[S],
                beta: &[S],
                out: &mut [S],
            ) -> diagnostic::Result<()> {
                let mut mean = vec![S::ZERO; C];
                let mut var = vec![S::ONE; C];
                crate::norm::BatchNorm2d {
                    channels: C,
                    eps: S::from_f64(f64::from(EPS)),
                    momentum: S::from_f64(0.1),
                }
                .forward_train(x, gamma, beta, N, 1, S, &mut mean, &mut var, out)
            }

            norm_op!(
                BatchNormModel,
                BatchNorm,
                "batch_norm_f32",
                vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
                1.5,
                0.25,
                4.0,
                3
            );
        }

        /// 组归一化（声明档 `G=2, C=4, spatial=2, n=2`）。
        pub mod group {
            use super::super::*;
            pub use super::{C, EPS, N, S};

            /// 归一化分组数。
            pub const G: usize = 2;

            pub(crate) fn run<S: Scalar>(
                x: &[S],
                gamma: &[S],
                beta: &[S],
                out: &mut [S],
            ) -> diagnostic::Result<()> {
                crate::norm::GroupNorm {
                    groups: G,
                    channels: C,
                    eps: S::from_f64(f64::from(EPS)),
                }
                .forward(x, gamma, beta, N, S, out)
            }

            norm_op!(
                GroupNormModel,
                GroupNorm,
                "group_norm_f32",
                vec![
                    ("N", (N * S * C) as i64),
                    ("G", G as i64),
                    ("C", C as i64),
                    ("S", S as i64),
                ],
                1.25,
                0.5,
                4.0,
                3
            );
        }

        /// 实例归一化（声明档 `C=4, spatial=2, n=2`）。
        pub mod instance {
            use super::super::*;
            pub use super::{C, EPS, N, S};

            pub(crate) fn run<S: Scalar>(
                x: &[S],
                gamma: &[S],
                beta: &[S],
                out: &mut [S],
            ) -> diagnostic::Result<()> {
                crate::norm::InstanceNorm {
                    channels: C,
                    eps: S::from_f64(f64::from(EPS)),
                }
                .forward(x, gamma, beta, N, S, out)
            }

            norm_op!(
                InstanceNormModel,
                InstanceNorm,
                "instance_norm_f32",
                vec![("N", (N * S * C) as i64), ("C", C as i64), ("S", S as i64)],
                0.75,
                -0.25,
                4.0,
                3
            );
        }

        /// 层归一化（声明档 `3 行 × 4 宽`）。
        pub mod layer {
            use super::super::*;
            pub use super::EPS;

            /// 层归一化行数。
            pub const R: usize = 3;
            /// 输入宽度或归一化宽度。
            pub const W: usize = 4;

            pub(crate) fn run<S: Scalar>(
                x: &[S],
                gamma: &[S],
                beta: &[S],
                out: &mut [S],
            ) -> diagnostic::Result<()> {
                crate::norm::LayerNorm {
                    width: W,
                    eps: S::from_f64(f64::from(EPS)),
                }
                .forward(x, gamma, beta, out)
            }

            norm_op!(
                LayerNormModel,
                LayerNorm,
                "layer_norm_f32",
                R * W,
                W,
                vec![("N", (R * W) as i64), ("R", R as i64), ("W", W as i64)],
                1.0,
                0.0,
                4.0,
                3
            );
        }
    }

    // 公开路径保持为 `classic::{max_pool1d, …, layer_norm}`，`lib.rs` 与测试据此取名。
    pub use norm::{
        batch as batch_norm, group as group_norm, instance as instance_norm, layer as layer_norm,
    };
    pub use pool::{
        adaptive as adaptive_pool, avg2d as avg_pool2d, max1d as max_pool1d, max2d as max_pool2d,
    };

    #[cfg(test)]
    mod tests {
        include!("../tests/conv_classic_tests.rs");
    }
}
pub(crate) mod decode {
    //! decode 注意力的两个面：内核算术（[`kernel`]）与语义表（[`semantics`]）。

    mod kernel {
        //! decode 内核族：query 行数（`q_rows`）与 K/V 历史行数（`hist`）解耦的通用路径。
        use diagnostic::{Code, Error};

        use crate::conv::attention::{Mode, Rows, check_width, run_rows};

        /// decode 查询与历史行数都必须非零。
        fn check_rows(q_rows: usize, hist: usize) -> diagnostic::Result<()> {
            if q_rows == 0 || hist == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("decode 查询行数 {q_rows} 与历史行数 {hist} 都必须非零"),
                ));
            }
            Ok(())
        }

        /// 一对切片都必须等于预期长度。
        fn check_pair(
            first: &[f32],
            second: &[f32],
            expect: usize,
            what: &str,
        ) -> diagnostic::Result<()> {
            if first.len() != expect || second.len() != expect {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{what} 长度与预期 {expect} 不一致"),
                ));
            }
            Ok(())
        }

        pub(crate) fn check_decode_len(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &[f32],
            q_rows: usize,
            hist: usize,
            d: usize,
        ) -> diagnostic::Result<()> {
            check_width(d)?;
            check_rows(q_rows, hist)?;
            let q_len = q_rows * d;
            let kv_len = hist * d;
            check_pair(q, out, q_len, "q/out")?;
            check_pair(k, v, kv_len, "k/v")?;
            Ok(())
        }

        /// decode 内核共享：`decode_forward` 与 `decode_flash_forward` 只差 softmax 策略（[`Mode`]）。
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn decode(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
            mode: Mode,
        ) -> diagnostic::Result<()> {
            check_decode_len(q, k, v, out, q_rows, hist, d)?;
            let rows = Rows::new(q, k, v, hist, d);
            run_rows(&rows, q_rows, out, mode);
            Ok(())
        }

        /// 单头朴素内核（裸切片骨架）：`q`（`q_rows·d`）对历史 K/V（`hist·d`）做注意力，结果写进 `out`（`q_rows·d`）。
        /// 供多头 / 批量编排按段复用；公共 [`decode_forward`](super::decode_forward) 出口在其外接二维视图。
        pub(crate) fn single_naive(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
        ) -> diagnostic::Result<()> {
            decode(q, k, v, out, q_rows, hist, d, Mode::Naive)
        }

        /// 单头 flash 内核（裸切片骨架）：online softmax 单遍扫历史 K/V，写进 `out`。口径同 `single_naive`。
        pub(crate) fn single_flash(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
        ) -> diagnostic::Result<()> {
            decode(q, k, v, out, q_rows, hist, d, Mode::Online)
        }

        /// 二维视图公共出口：朴素 decode，几何从视图形状取，解包后走 `single_naive`。
        pub fn decode_forward(
            q: tensor::View<'_, f32>,
            k: tensor::View<'_, f32>,
            v: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            let (q_rows, d) = crate::matmul::dim2(q.dims(), "decode_attention_f32")?;
            let (hist, kd) = crate::matmul::dim2(k.dims(), "decode_attention_f32")?;
            if kd != d {
                return Err(Error::new(
                    Code::Invalid,
                    format!("decode_attention_f32 头宽不符：q·d={d}，k·d={kd}"),
                ));
            }
            crate::matmul::dim2(v.dims(), "decode_attention_f32")?;
            single_naive(
                q.as_slice(),
                k.as_slice(),
                v.as_slice(),
                out,
                q_rows,
                hist,
                d,
            )
        }

        /// 二维视图公共出口：flash decode，口径同 [`decode_forward`]，走 `single_flash`。
        pub fn decode_flash_forward(
            q: tensor::View<'_, f32>,
            k: tensor::View<'_, f32>,
            v: tensor::View<'_, f32>,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            let (q_rows, d) = crate::matmul::dim2(q.dims(), "decode_flash_attention_f32")?;
            let (hist, kd) = crate::matmul::dim2(k.dims(), "decode_flash_attention_f32")?;
            if kd != d {
                return Err(Error::new(
                    Code::Invalid,
                    format!("decode_flash_attention_f32 头宽不符：q·d={d}，k·d={kd}"),
                ));
            }
            crate::matmul::dim2(v.dims(), "decode_flash_attention_f32")?;
            single_flash(
                q.as_slice(),
                k.as_slice(),
                v.as_slice(),
                out,
                q_rows,
                hist,
                d,
            )
        }

        /// 单头 decode 内核的函数指针：朴素与 flash 包装共用同一段头分段编排。
        type SingleHead =
            fn(&[f32], &[f32], &[f32], &mut [f32], usize, usize, usize) -> diagnostic::Result<()>;

        /// 多头 decode：头优先块布局——`q`/`out` 为 `heads · q_rows · d`，`k`/`v` 为 `heads · hist · d`，
        /// 第 `h` 个头是一段连续的 `q_rows · d` / `hist · d`。头数由 `q` 长度按 `q_rows · d` 推出，
        /// 头间互不耦合、每个头独立 softmax；头内算术复用传入的单头内核，这里只做头分段编排。
        ///
        /// # Errors
        ///
        /// `q` 不是 `q_rows · d` 的整数倍、缓冲长度不成头块倍数，或任一头段长度/宽度/行数不符时返回说明。
        /// 校验多头块长度并派生头数：`q`/`out` 是 `heads·q_rows·d`，`k`/`v` 是 `heads·hist·d`；
        /// 头内宽度与行数守卫交给逐段调用的单头内核。
        fn headed_heads(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &[f32],
            q_rows: usize,
            hist: usize,
            d: usize,
        ) -> diagnostic::Result<usize> {
            let q_stride = q_rows * d;
            let kv_stride = hist * d;
            if q_stride == 0 || !q.len().is_multiple_of(q_stride) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("q 长度 {} 不是 q_rows·d（{q_stride}）的整数倍", q.len()),
                ));
            }
            let heads = q.len() / q_stride;
            check_pair(q, out, heads * q_stride, "q/out")?;
            check_pair(k, v, heads * kv_stride, "k/v")?;
            Ok(heads)
        }

        #[allow(clippy::too_many_arguments)]
        fn decode_heads(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
            single: SingleHead,
        ) -> diagnostic::Result<()> {
            let heads = headed_heads(q, k, v, out, q_rows, hist, d)?;
            let q_stride = q_rows * d;
            let kv_stride = hist * d;
            for head in 0..heads {
                let q_start = head * q_stride;
                let kv_start = head * kv_stride;
                let qh = &q[q_start..q_start + q_stride];
                let kh = &k[kv_start..kv_start + kv_stride];
                let vh = &v[kv_start..kv_start + kv_stride];
                let oh = &mut out[q_start..q_start + q_stride];
                single(qh, kh, vh, oh, q_rows, hist, d)?;
            }
            Ok(())
        }

        /// 生成一组 `*_forward_*` / `*_flash_forward_*` 公开包装对：朴素与 flash 只差单头内核选择，
        /// 签名与布局随形状函数保持一致。
        macro_rules! decode_shape_pair {
    (
        $naive:ident, $flash:ident, $target:ident,
        $naive_doc:literal, $flash_doc:literal,
        $( $arg:ident : $ty:ty ),* $(,)?
    ) => {
        #[doc = $naive_doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $naive($( $arg: $ty ),*) -> diagnostic::Result<()> {
            $target($( $arg ),*, single_naive)
        }

        #[doc = $flash_doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $flash($( $arg: $ty ),*) -> diagnostic::Result<()> {
            $target($( $arg ),*, single_flash)
        }
    };
}

        decode_shape_pair!(
            decode_forward_heads,
            decode_flash_forward_heads,
            decode_heads,
            "多头朴素 decode 内核：头数由 `q` 长度派生，布局同 `decode_heads`，头内走 [`decode_forward`]。\n\n# Errors\n\n同 `decode_heads`。",
            "多头 flash decode 内核：头数由 `q` 长度派生，布局同 `decode_heads`，头内走 `decode_flash_forward`。\n\n# Errors\n\n同 `decode_heads`。",
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize
        );

        /// 批量校验：批内各序列等长历史（统一 `hist`）。`heads` 是显式几何参数，不按总长反推，
        /// 避免批数不足时把头数压成 1 而放过错长缓冲；段内宽度/行数守卫交给 `decode_heads`。
        #[allow(clippy::too_many_arguments)]
        fn headed_batch(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &[f32],
            q_rows: usize,
            hist: usize,
            d: usize,
            batch: usize,
            heads: usize,
        ) -> diagnostic::Result<()> {
            if batch == 0 || heads == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("批量序列数 {batch} 与头数 {heads} 都必须非零"),
                ));
            }
            let q_seq = heads * q_rows * d;
            let kv_seq = heads * hist * d;
            check_pair(q, out, batch * q_seq, "q/out")?;
            check_pair(k, v, batch * kv_seq, "k/v")?;
            Ok(())
        }

        /// 批量 decode：`q`/`out` 按批分段，第 `b` 段是 `heads·q_rows·d`，`k`/`v` 第 `b` 段是
        /// `heads·hist·d`；批内段间独立 softmax、互不串段。段内复用 `decode_heads` 头分段编排，
        /// 这里只做批切片。变长历史（各段独立 `hist_b`）不在本入口口径内。
        #[allow(clippy::too_many_arguments)]
        fn decode_batch(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
            heads: usize,
            batch: usize,
            single: SingleHead,
        ) -> diagnostic::Result<()> {
            headed_batch(q, k, v, out, q_rows, hist, d, batch, heads)?;
            let q_seq = heads * q_rows * d;
            let kv_seq = heads * hist * d;
            for seq in 0..batch {
                let q_start = seq * q_seq;
                let kv_start = seq * kv_seq;
                decode_heads(
                    &q[q_start..q_start + q_seq],
                    &k[kv_start..kv_start + kv_seq],
                    &v[kv_start..kv_start + kv_seq],
                    &mut out[q_start..q_start + q_seq],
                    q_rows,
                    hist,
                    d,
                    single,
                )?;
            }
            Ok(())
        }

        decode_shape_pair!(
            decode_forward_batch,
            decode_flash_forward_batch,
            decode_batch,
            "批量朴素 decode 内核：段内走 [`decode_forward`]，布局同 `decode_batch`。\n\n# Errors\n\n同 `decode_batch`。",
            "批量 flash decode 内核：段内走 `decode_flash_forward`，布局同 `decode_batch`。\n\n# Errors\n\n同 `decode_batch`。",
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
            heads: usize,
            batch: usize
        );

        /// 非零几何守卫：批数 / 头数 / 查询行数 / 头宽任一为零即拒绝。
        fn batch_geometry_nonzero(
            batch: usize,
            heads: usize,
            q_rows: usize,
            d: usize,
        ) -> diagnostic::Result<()> {
            if batch == 0 || heads == 0 || q_rows == 0 || d == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "批量序列数 {batch}、头数 {heads}、查询行数 {q_rows}、头宽 {d} 都必须非零"
                    ),
                ));
            }
            Ok(())
        }

        /// 变长批量校验：`hist` 逐序列历史行数；`q`/`out` 等长段 `heads·q_rows·d`，`k`/`v` 总长
        /// 是各段 `heads·hist_b·d` 之和。头数 / 批数显式传入，不从总长反推，避免批数不足时把头数压成 1。
        #[allow(clippy::too_many_arguments)]
        fn headed_batch_var(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &[f32],
            q_rows: usize,
            hist: &[usize],
            d: usize,
            heads: usize,
        ) -> diagnostic::Result<()> {
            let batch = hist.len();
            batch_geometry_nonzero(batch, heads, q_rows, d)?;
            let q_seq = heads * q_rows * d;
            check_pair(q, out, batch * q_seq, "q/out")?;
            let kv_total: usize = hist.iter().map(|h| heads * h * d).sum();
            check_pair(k, v, kv_total, "k/v")?;
            Ok(())
        }

        /// 变长批量 decode：`q`/`out` 等长段 `heads·q_rows·d`；`k`/`v` 第 `b` 段是
        /// `heads·hist_b·d`，段间紧排、长度随 `hist[b]` 不同。段内复用 `decode_heads` 头分段编排，
        /// 这里只按累计历史游标切批段；各段独立 softmax、互不串段。段内行数 / 宽度守卫交给 `decode_heads`。
        #[allow(clippy::too_many_arguments)]
        fn decode_batch_var(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: &[usize],
            d: usize,
            heads: usize,
            single: SingleHead,
        ) -> diagnostic::Result<()> {
            headed_batch_var(q, k, v, out, q_rows, hist, d, heads)?;
            let q_seq = heads * q_rows * d;
            let mut kv_cursor = 0;
            for (seq, &hist_b) in hist.iter().enumerate() {
                let kv_seq = heads * hist_b * d;
                let q_start = seq * q_seq;
                decode_heads(
                    &q[q_start..q_start + q_seq],
                    &k[kv_cursor..kv_cursor + kv_seq],
                    &v[kv_cursor..kv_cursor + kv_seq],
                    &mut out[q_start..q_start + q_seq],
                    q_rows,
                    hist_b,
                    d,
                    single,
                )?;
                kv_cursor += kv_seq;
            }
            Ok(())
        }

        decode_shape_pair!(
            decode_forward_batch_var,
            decode_flash_forward_batch_var,
            decode_batch_var,
            "变长批量朴素 decode：段内走 [`decode_forward`]，布局同 `decode_batch_var`。\n\n# Errors\n\n同 `decode_batch_var`。",
            "变长批量 flash decode：段内走 `decode_flash_forward`，布局同 `decode_batch_var`。\n\n# Errors\n\n同 `decode_batch_var`。",
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            q_rows: usize,
            hist: &[usize],
            d: usize,
            heads: usize
        );
    }

    pub mod semantics {
        //! decode 注意力语义：query 行数与 K/V 历史行数解耦。
        use crate::model::{AxisRole, AxisSpec, Cost, Sample, ShapeCtx};
        use diagnostic::{Code, Error};

        /// 样本固定历史长度：真实 `hist` 由执行侧按已缓存页数推出，语义表样本用一份确定长度。
        const HIST: usize = 4;

        /// 两条基本轴：query 行轴 `q`、头宽轴 `d`；历史行轴 `h` 是样本派生轴。
        pub const AXES: &[AxisSpec] = &[
            AxisSpec {
                role: AxisRole::Input,
                label: 'q',
            },
            AxisSpec {
                role: AxisRole::Output,
                label: 'd',
            },
        ];

        /// `(m, n)` 即 `(q_rows, d)`；历史长度固定进形状上下文作为派生轴。
        ///
        /// # Errors
        ///
        /// 轴长为 0 或字母重复时由 [`ShapeCtx::new`] 返回说明。
        pub fn project_shape(q_rows: usize, d: usize) -> Result<ShapeCtx, Error> {
            ShapeCtx::new([('q', q_rows), ('d', d), ('h', HIST)])
        }

        /// 标量代价：每行 query 扫 `hist` 条历史（每个 d 维内积）+ softmax + 对 `V` 加权。
        ///
        /// # Errors
        ///
        /// 缺轴时返回说明。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            let q = crate::conv::attention::axis(shape, "decode attention", 'q')?;
            let d = crate::conv::attention::axis(shape, "decode attention", 'd')?;
            let h = crate::conv::attention::axis(shape, "decode attention", 'h')?;
            Ok(Cost {
                flops: (q as f64) * (h as f64) * ((d as f64) * 2.0 + 3.0),
                read: (q as u64) * (d as u64) * 4 + (h as u64) * (d as u64) * 4 * 2,
                write: (q as u64) * (d as u64) * 4,
            })
        }

        /// 确定性样本：`q` 与输出 `y` 为 `q_rows·d`，`k`/`v` 为 `hist·d`。
        pub fn sample(q_rows: usize, d: usize) -> Sample {
            let hist = HIST;
            Sample {
                m: q_rows,
                n: d,
                scalars: vec![
                    ("Q_ROWS", q_rows as i64),
                    ("HIST", hist as i64),
                    ("D", d as i64),
                ],
                bufs: crate::conv::attention::sample_bufs(q_rows * d, hist * d),
            }
        }

        /// 取样本标量并转行数；缺标量时按名字报错。
        fn scalar_or(sample: &Sample, name: &str) -> Result<usize, Error> {
            sample
                .scalar(name)
                .ok_or_else(|| Error::new(Code::Invalid, format!("decode 样本缺标量 `{name}`")))
                .map(|value| value as usize)
        }

        /// 从样本读出 decode 三元几何 `(q_rows, hist, d)`。
        fn geometry(sample: &Sample) -> diagnostic::Result<(usize, usize, usize)> {
            let q_rows = scalar_or(sample, "Q_ROWS")?;
            let hist = scalar_or(sample, "HIST")?;
            let d = scalar_or(sample, "D")?;
            Ok((q_rows, hist, d))
        }

        /// 把样本三条切片包成 `q_rows×d` / `hist×d` 二维视图。
        fn views<'a>(
            q: &'a [f32],
            k: &'a [f32],
            v: &'a [f32],
            q_rows: usize,
            hist: usize,
            d: usize,
        ) -> diagnostic::Result<(
            tensor::View<'a, f32>,
            tensor::View<'a, f32>,
            tensor::View<'a, f32>,
        )> {
            Ok((
                tensor::View::matrix(q_rows, d, q)?,
                tensor::View::matrix(hist, d, k)?,
                tensor::View::matrix(hist, d, v)?,
            ))
        }

        /// 朴素参考：query 对历史 K/V 做一遍数值稳定 softmax，再对 `V` 加权。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let (q_rows, hist, d) = geometry(sample)?;
            let (q, k, v) = crate::conv::attention::qkv(sample)?;
            let (qv, kv, vv) = views(q, k, v, q_rows, hist, d)?;
            let mut out = vec![0.0; q_rows * d];
            crate::conv::attention::decode_forward(qv, kv, vv, &mut out)?;
            Ok(out)
        }

        /// Flash 参考：flash 只是同一数学的**优化**（online softmax 单遍扫 K/V），语义锚仍取朴素定义。
        /// 指回 `decode_flash_forward` 会让"内核对不对"与"参考对不对"落在同一行代码上（F6 自我对拍）。
        pub fn decode_flash_reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            reference(sample)
        }

        /// 标记类型：承载 `#[model]` 语义。
        pub struct DecodeAttention;

        /// 朴素 decode 注意力语义别名。
        #[annotation::model(name = "decode_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention")]
        pub type DecodeAttentionF32 = DecodeAttention;

        /// Flash decode 注意力标记类型：形状、样本、代价与朴素 [`DecodeAttention`] 共用，语义锚也共用朴素
        /// 定义（[`decode_flash_reference`] 转调 `reference`）；两者差在**内核**，不在数学。
        /// `const = "DECODE_FLASH_MODEL"` 让两个算子同模块各持一份语义表。
        pub struct DecodeFlashAttention;

        /// Flash decode 注意力语义别名；注册名 `decode_flash_attention_f32` 指向同一套数学。
        #[annotation::model(name = "decode_flash_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention", const = "DECODE_FLASH_MODEL")]
        pub type DecodeFlashAttentionF32 = DecodeFlashAttention;

        // 基座契约（ADR 0050）：同上一对，只把参考入口换成 flash 版本。
        crate::operator_here!(
            DecodeAttention,
            "decode_attention_f32",
            3,
            (1, 8),
            true,
            reference
        );
        crate::operator_here!(
            DecodeFlashAttention,
            "decode_flash_attention_f32",
            3,
            (1, 8),
            true,
            decode_flash_reference
        );
    }
    pub use kernel::{
        decode_flash_forward, decode_flash_forward_batch, decode_flash_forward_batch_var,
        decode_flash_forward_heads, decode_forward, decode_forward_batch, decode_forward_batch_var,
        decode_forward_heads,
    };

    pub(super) use kernel::decode;
}
pub mod deconv {
    //! 转置卷积 `ConvTranspose2d`（解卷积/上采样），NHWC，权重布局对齐 PyTorch
    //! `[Cin, Cout, kh, kw]`。

    use std::ops::AddAssign;

    use crate::guard::check_all;
    use diagnostic::{Code, Error, Result};

    use crate::conv::ConvGeom;
    use crate::fast;
    use tensor::NumericElement;

    /// 前向标量缝：f32 与对称 i8（i32 累加）共用同一套前向/散射算法，
    /// 只有 GEMM 后端与零值随类型切换。换后端只需新增一个 impl，调用方不变。
    ///
    /// 元素收紧到 [`NumericElement`](tensor::NumericElement)：乘加族只接浮点 / 整数封闭分组。
    trait DeconvScalar: Copy + Default + NumericElement {
        /// 累加/输出类型：f32 → f32，i8 → i32。
        type Acc: Copy + Default + AddAssign;

        /// 列矩阵 GEMM：`col = x · wt`，收缩维为 `inner`。
        fn gemm(
            x: &[Self],
            wt: &[Self],
            rows: usize,
            cols: usize,
            inner: usize,
            col: &mut [Self::Acc],
        );
    }

    impl DeconvScalar for f32 {
        type Acc = f32;

        fn gemm(x: &[f32], wt: &[f32], rows: usize, cols: usize, inner: usize, col: &mut [f32]) {
            fast::gemm_f32(x, wt, rows, cols, inner, col);
        }
    }

    impl DeconvScalar for i8 {
        type Acc = i32;

        fn gemm(x: &[i8], wt: &[i8], rows: usize, cols: usize, inner: usize, col: &mut [i32]) {
            fast::gemm_i8(x, wt, rows, cols, inner, col);
        }
    }

    /// 二维转置卷积：输入 `[n,h,w,cin]`，权重 `[cin,cout,kh,kw]`，输出 `[n,oh,ow,cout]`。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConvTranspose2d {
        /// 输入通道。
        pub cin: usize,
        /// 输出通道。
        pub cout: usize,
        /// 核、步长、补零（与正向卷积同一套语义）。
        pub geom: ConvGeom,
    }

    impl ConvTranspose2d {
        /// 由通道与几何构造。
        pub fn new(cin: usize, cout: usize, geom: ConvGeom) -> Self {
            Self { cin, cout, geom }
        }

        /// 输出高：`(h−1)·sh − 2·ph + kh`。
        ///
        /// # Errors
        ///
        /// 步长为 0 或结果下溢时返回说明。
        pub fn oh(&self, h: usize) -> Result<usize> {
            self.out_len(h, self.geom.sh, self.geom.kh, self.geom.ph, "h")
        }

        /// 输出宽；口径同 [`ConvTranspose2d::oh`]。
        ///
        /// # Errors
        ///
        /// 步长为 0 或结果下溢时返回说明。
        pub fn ow(&self, w: usize) -> Result<usize> {
            self.out_len(w, self.geom.sw, self.geom.kw, self.geom.pw, "w")
        }

        fn out_len(&self, inp: usize, s: usize, k: usize, p: usize, axis: &str) -> Result<usize> {
            if s == 0 {
                return Err(Error::new(Code::Invalid, format!("{axis}: 步长必须非零")));
            }
            let padded = (inp.saturating_sub(1)) * s;
            if padded + k < 2 * p {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{axis}: 输出尺寸下溢（{inp}×{s}+{k} 小于 2·{p}）"),
                ));
            }
            Ok(padded + k - 2 * p)
        }

        /// f32 前向：GEMM 得列矩阵，再散射累加到输出。
        ///
        /// # Errors
        ///
        /// 缓冲长度与形状不符时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn forward_f32(
            &self,
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            n: usize,
            h: usize,
            w: usize,
            out: &mut [f32],
        ) -> Result<()> {
            forward(self, x, weight, bias, n, h, w, out)
        }

        /// i8 前向（对称量化）：GEMM 走多槽池，散射累加在 i32 上。
        ///
        /// # Errors
        ///
        /// 缓冲长度与形状不符时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn forward_i8(
            &self,
            x: &[i8],
            weight: &[i8],
            bias: &[i32],
            n: usize,
            h: usize,
            w: usize,
            out: &mut [i32],
        ) -> Result<()> {
            forward(self, x, weight, bias, n, h, w, out)
        }
    }

    /// 二维前向核心：标量类型经 [`DeconvScalar`] 参数化，f32/i8 共用同一份算法。
    #[allow(clippy::too_many_arguments)]
    fn forward<S: DeconvScalar>(
        conv: &ConvTranspose2d,
        x: &[S],
        weight: &[S],
        bias: &[S::Acc],
        n: usize,
        h: usize,
        w: usize,
        out: &mut [S::Acc],
    ) -> Result<()> {
        let oh = conv.oh(h)?;
        let ow = conv.ow(w)?;
        let (cin, cout) = (conv.cin, conv.cout);
        let karea = conv.geom.kh * conv.geom.kw;
        check_all(&[
            (x.len(), n * h * w * cin, "x"),
            (weight.len(), cin * cout * karea, "weight"),
            (bias.len(), cout, "bias"),
            (out.len(), n * oh * ow * cout, "out"),
        ])?;

        let rows = n * h * w;
        let cols = cout * karea;
        let mut wt = vec![S::default(); cin * cols];
        transpose_weight_2d(weight, &mut wt, cin, cout, conv.geom.kh, conv.geom.kw);
        let mut col = vec![S::Acc::default(); rows * cols];
        S::gemm(x, &wt, rows, cols, cin, &mut col);

        out[..n * oh * ow * cout].fill(S::Acc::default());
        scatter(&col, out, n, (h, w), (oh, ow), conv, cout);
        add_bias_2d(out, bias, n, oh, ow, cout);
        Ok(())
    }

    /// 散射：把列矩阵 `[n·h·w, cout·kh·kw]` 的第 `(ki,kj,co)` 项累加到 `(hi·sh−ph+ki, wi·sw−pw+kj)`。
    fn scatter<T: Copy + AddAssign>(
        col: &[T],
        out: &mut [T],
        n: usize,
        (h, w): (usize, usize),
        (oh, ow): (usize, usize),
        conv: &ConvTranspose2d,
        cout: usize,
    ) {
        let g = conv.geom;
        let karea = cout * g.kh * g.kw;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let row = &col[((ni * h + hi) * w + wi) * karea..][..karea];
                    scatter_row_2d(row, out, g, oh, ow, cout, ni, hi, wi);
                }
            }
        }
    }

    /// 一维反卷积：NHWC，权重布局对齐 PyTorch `[Cin, Cout, k]`。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConvTranspose1d {
        /// 输入通道。
        pub cin: usize,
        /// 输出通道。
        pub cout: usize,
        /// 窗口长。
        pub k: usize,
        /// 步长。
        pub s: usize,
        /// 两端补零（等价于输出两侧裁掉的量）。
        pub p: usize,
    }

    impl ConvTranspose1d {
        /// 构造：通道、窗口与步长都必须非零。
        #[must_use]
        pub fn new(cin: usize, cout: usize, k: usize, s: usize, p: usize) -> Self {
            Self { cin, cout, k, s, p }
        }

        /// 输出长度 `(l − 1)·s − 2p + k`。
        ///
        /// # Errors
        ///
        /// 步长/窗口为 0，或 `(l − 1)·s < 2p − k + 1`（输出长度非正）时返回说明。
        pub fn len(&self, l: usize) -> Result<usize> {
            if self.s == 0 || self.k == 0 {
                return Err(Error::new(Code::Invalid, "反卷积窗口与步长必须大于 0"));
            }
            let base = l
                .checked_sub(1)
                .and_then(|v| v.checked_mul(self.s))
                .ok_or(Error::new(Code::Invalid, "反卷积输出长度溢出"))?;
            let span = base + self.k;
            let pad = 2 * self.p;
            // 输出必须严格为正：`span − pad ≥ 1`。
            if span < pad + 1 {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "反卷积输出长度非正（l={l}, s={}, p={}, k={}）",
                        self.s, self.p, self.k
                    ),
                ));
            }
            Ok(span - pad)
        }

        /// f32 前向：GEMM 得列矩阵，再散射累加。
        ///
        /// # Errors
        ///
        /// 缓冲长度与几何不符时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn forward_f32(
            &self,
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            n: usize,
            l: usize,
            out: &mut [f32],
        ) -> Result<()> {
            forward_1d(self, x, weight, bias, n, l, out)
        }

        /// i8 前向：GEMM 走多槽池，散射累加在 i32 上。
        ///
        /// # Errors
        ///
        /// 缓冲长度与几何不符时返回说明。
        pub fn forward_i8(
            &self,
            x: &[i8],
            weight: &[i8],
            bias: &[i32],
            n: usize,
            l: usize,
            out: &mut [i32],
        ) -> Result<()> {
            forward_1d(self, x, weight, bias, n, l, out)
        }
    }

    /// 一维前向核心：标量类型经 [`DeconvScalar`] 参数化，f32/i8 共用同一份算法。
    fn forward_1d<S: DeconvScalar>(
        conv: &ConvTranspose1d,
        x: &[S],
        weight: &[S],
        bias: &[S::Acc],
        n: usize,
        l: usize,
        out: &mut [S::Acc],
    ) -> Result<()> {
        let ol = conv.len(l)?;
        let (cin, cout, k) = (conv.cin, conv.cout, conv.k);
        let cols = cout * k;
        check_all(&[
            (x.len(), n * l * cin, "x"),
            (weight.len(), cin * cols, "weight"),
            (bias.len(), cout, "bias"),
            (out.len(), n * ol * cout, "out"),
        ])?;

        let mut wt = vec![S::default(); cin * cols];
        transpose_weight_1d(weight, &mut wt, cin, cout, k);
        let mut col = vec![S::Acc::default(); n * l * cols];
        S::gemm(x, &wt, n * l, cols, cin, &mut col);

        out[..n * ol * cout].fill(S::Acc::default());
        scatter1(&col, out, n, l, ol, conv, cout);
        add_bias_1d(out, bias, n, ol, cout);
        Ok(())
    }

    /// 列矩阵 `[n·l, cout·k]` 散射累加到 `[n, ol, cout]`：越界位置裁掉。
    fn scatter1<T: Copy + AddAssign>(
        col: &[T],
        out: &mut [T],
        n: usize,
        l: usize,
        ol: usize,
        conv: &ConvTranspose1d,
        cout: usize,
    ) {
        for ni in 0..n {
            for i in 0..l {
                let src = (ni * l + i) * cout * conv.k;
                scatter1_at(col, out, ol, conv, cout, ni, i, src);
            }
        }
    }

    /// `0 <= idx < n`：散射越界裁剪共用的边界谓词。
    fn in_range(idx: isize, n: usize) -> bool {
        idx >= 0 && idx < n as isize
    }

    /// 二维权重转置 `[cin,cout,kh,kw]` → `[cin, cout·kh·kw]`。
    fn transpose_weight_2d<T: Copy>(
        weight: &[T],
        wt: &mut [T],
        cin: usize,
        cout: usize,
        kh: usize,
        kw: usize,
    ) {
        let cols = cout * kh * kw;
        for ci in 0..cin {
            for co in 0..cout {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let src = ((ci * cout + co) * kh + ki) * kw + kj;
                        wt[ci * cols + (ki * kw + kj) * cout + co] = weight[src];
                    }
                }
            }
        }
    }

    /// 二维偏置累加：`out[n,oh,ow,cout] += bias[cout]`。
    fn add_bias_2d<T: Copy + AddAssign>(
        out: &mut [T],
        bias: &[T],
        n: usize,
        oh: usize,
        ow: usize,
        cout: usize,
    ) {
        for p in 0..n * oh * ow {
            for co in 0..cout {
                out[p * cout + co] += bias[co];
            }
        }
    }

    /// 单个输入行的散射：沿核高 `kh` 散射，越界行裁掉。
    #[allow(clippy::too_many_arguments)]
    fn scatter_row_2d<T: Copy + AddAssign>(
        row: &[T],
        out: &mut [T],
        g: ConvGeom,
        oh: usize,
        ow: usize,
        cout: usize,
        ni: usize,
        hi: usize,
        wi: usize,
    ) {
        for ki in 0..g.kh {
            let ii = (hi * g.sh + ki) as isize - g.ph as isize;
            if !in_range(ii, oh) {
                continue;
            }
            scatter_kernel_2d(row, out, g, oh, ow, cout, ni, ii as usize, wi, ki);
        }
    }

    /// 单个核位置的散射：沿核宽 `kw` 散射，越界列裁掉，`cout` 通道逐个累加。
    #[allow(clippy::too_many_arguments)]
    fn scatter_kernel_2d<T: Copy + AddAssign>(
        row: &[T],
        out: &mut [T],
        g: ConvGeom,
        oh: usize,
        ow: usize,
        cout: usize,
        ni: usize,
        ii: usize,
        wi: usize,
        ki: usize,
    ) {
        for kj in 0..g.kw {
            let jj = (wi * g.sw + kj) as isize - g.pw as isize;
            if !in_range(jj, ow) {
                continue;
            }
            let dst = ((ni * oh + ii) * ow + jj as usize) * cout;
            let src = (ki * g.kw + kj) * cout;
            for co in 0..cout {
                out[dst + co] += row[src + co];
            }
        }
    }

    /// 一维权重转置 `[cin,cout,k]` → `[cin, cout·k]`。
    fn transpose_weight_1d<T: Copy>(weight: &[T], wt: &mut [T], cin: usize, cout: usize, k: usize) {
        let cols = cout * k;
        for ci in 0..cin {
            for co in 0..cout {
                for ki in 0..k {
                    wt[ci * cols + ki * cout + co] = weight[(ci * cout + co) * k + ki];
                }
            }
        }
    }

    /// 一维偏置累加：`out[n,ol,cout] += bias[cout]`。
    fn add_bias_1d<T: Copy + AddAssign>(
        out: &mut [T],
        bias: &[T],
        n: usize,
        ol: usize,
        cout: usize,
    ) {
        for pos in 0..n * ol {
            for co in 0..cout {
                out[pos * cout + co] += bias[co];
            }
        }
    }

    /// 一维单个输入位置的散射：沿窗口 `k` 散射，越界位置裁掉。
    #[allow(clippy::too_many_arguments)]
    fn scatter1_at<T: Copy + AddAssign>(
        col: &[T],
        out: &mut [T],
        ol: usize,
        conv: &ConvTranspose1d,
        cout: usize,
        ni: usize,
        i: usize,
        src: usize,
    ) {
        for ki in 0..conv.k {
            let dst = (i * conv.s + ki) as isize - conv.p as isize;
            if !in_range(dst, ol) {
                continue;
            }
            let d = (ni * ol + dst as usize) * cout;
            let src_k = src + ki * cout;
            for co in 0..cout {
                out[d + co] += col[src_k + co];
            }
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/conv_deconv_tests.rs");
    }
}
mod geom {
    //! 职责：卷积几何：输入输出体量与切片步幅计算。
    use diagnostic::{Code, Error, Result};

    use crate::fast;

    use crate::conv::support::*;

    /// 卷积几何参数：核、步长、补零。1d 层把 `kh`/`sh` 固定为 1 以复用 2d 的展开。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConvGeom {
        /// 卷积核高（1d 恒为 1）。
        pub kh: usize,
        /// 卷积核宽。
        pub kw: usize,
        /// 步长高（1d 恒为 1）。
        pub sh: usize,
        /// 步长宽。
        pub sw: usize,
        /// 补零高。
        pub ph: usize,
        /// 补零宽。
        pub pw: usize,
    }

    impl ConvGeom {
        /// 输出高：`floor((h + 2·ph - kh) / sh) + 1`，下溢即形状不合法。
        ///
        /// # Errors
        ///
        /// `h + 2·ph < kh` 或 `sh == 0` 时返回错误。
        pub fn oh(&self, h: usize) -> Result<usize> {
            self.out_len(h, self.ph, self.kh, self.sh, "h")
        }

        /// 输出宽；口径同 [`ConvGeom::oh`]。
        pub fn ow(&self, w: usize) -> Result<usize> {
            self.out_len(w, self.pw, self.kw, self.sw, "w")
        }

        fn out_len(
            &self,
            input: usize,
            pad: usize,
            k: usize,
            s: usize,
            axis: &str,
        ) -> Result<usize> {
            if s == 0 || k == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{axis}: 核与步长必须非零"),
                ));
            }
            let padded = padded_len(input, pad, axis)?;
            if padded < k {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{axis}: 补零后 {padded} 小于核 {k}，没有合法输出"),
                ));
            }
            Ok((padded - k) / s + 1)
        }

        /// 一行 im2col 的饱和估算长度；执行入口使用 [`Self::checked_patch`] 拒绝溢出。
        pub fn patch(&self, cin: usize) -> usize {
            self.kh.saturating_mul(self.kw).saturating_mul(cin)
        }

        /// 一行 im2col 的受检长度，任一乘法溢出时返回诊断。
        pub fn checked_patch(&self, cin: usize) -> Result<usize> {
            self.kh
                .checked_mul(self.kw)
                .and_then(|kernel| kernel.checked_mul(cin))
                .ok_or_else(|| Error::new(Code::Invalid, "卷积核展开长度溢出"))
        }
    }

    /// 补零后的轴长 `input + 2·pad`：受检乘法与加法，任何溢出都在输出长度计算前被拒绝。
    fn padded_len(input: usize, pad: usize, axis: &str) -> Result<usize> {
        pad.checked_mul(2)
            .and_then(|padding| input.checked_add(padding))
            .ok_or_else(|| Error::new(Code::Invalid, format!("{axis}: 补零尺寸溢出")))
    }

    /// 一维卷积：输入 `[n, l, cin]`、权重 `[kw, cin, cout]`、输出 `[n, ol, cout]`。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Conv1d {
        /// 输入/输出通道与核宽、步长、补零。
        pub cin: usize,
        /// 输出通道。
        pub cout: usize,
        /// 核、步长、补零（`kh`/`sh`/`ph` 恒为 1）。
        pub geom: ConvGeom,
    }

    impl Conv1d {
        /// 按核宽与步长构造；补零默认 0。
        pub fn new(cin: usize, cout: usize, kw: usize, stride: usize, pad: usize) -> Self {
            Self {
                cin,
                cout,
                geom: ConvGeom {
                    kh: 1,
                    kw,
                    sh: 1,
                    sw: stride,
                    ph: 0,
                    pw: pad,
                },
            }
        }

        /// 输出长度。
        ///
        /// # Errors
        ///
        /// 形状不合法时返回说明。
        pub fn len(&self, l: usize) -> Result<usize> {
            self.geom.ow(l)
        }

        /// 输出长度是否为 0（clippy 要求与 `len` 成对出现）。
        pub fn is_empty(&self, l: usize) -> Result<bool> {
            Ok(self.len(l)? == 0)
        }

        /// f32 前向：im2col → GEMM，再按需加偏置。
        ///
        /// # Errors
        ///
        /// 缓冲长度与形状不符时返回说明。
        pub fn forward_f32(
            &self,
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            n: usize,
            l: usize,
            out: &mut [f32],
        ) -> Result<()> {
            let ol = self.len(l)?;
            let cin = self.cin;
            let cout = self.cout;
            let patch = self.geom.checked_patch(cin)?;
            check_shapes(&[
                (x.len(), n * l * cin, "输入"),
                (weight.len(), patch * cout, "权重"),
                (bias.len(), cout, "偏置"),
                (out.len(), n * ol * cout, "输出"),
            ])?;

            let mut col = vec![0.0f32; n * ol * patch];
            im2col1d(x, &mut col, n, l, ol, self.geom, cin);
            gemm(&col, weight, n * ol, cout, patch, out);
            add_bias(out, bias, cout);
            Ok(())
        }
    }

    /// 二维卷积：输入 `[n, h, w, cin]`、权重 `[kh, kw, cin, cout]`、输出 `[n, oh, ow, cout]`。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Conv2d {
        /// 输入通道。
        pub cin: usize,
        /// 输出通道。
        pub cout: usize,
        /// 核、步长、补零。
        pub geom: ConvGeom,
    }

    impl Conv2d {
        /// 按核、步长、补零构造。
        pub fn new(cin: usize, cout: usize, geom: ConvGeom) -> Self {
            Self { cin, cout, geom }
        }

        /// 输出形状 `(oh, ow)`。
        ///
        /// # Errors
        ///
        /// 形状不合法时返回说明。
        pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
            Ok((self.geom.oh(h)?, self.geom.ow(w)?))
        }

        /// f32 前向：im2col → GEMM → 偏置。
        ///
        /// # Errors
        ///
        /// 缓冲长度与形状不符时返回说明。
        pub fn forward_f32(
            &self,
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            n: usize,
            (h, w): (usize, usize),
            out: &mut [f32],
        ) -> Result<()> {
            let (oh, ow) = self.shape(h, w)?;
            let cin = self.cin;
            let cout = self.cout;
            let patch = self.geom.checked_patch(cin)?;
            check_shapes(&[
                (x.len(), n * h * w * cin, "输入"),
                (weight.len(), patch * cout, "权重"),
                (bias.len(), cout, "偏置"),
                (out.len(), n * oh * ow * cout, "输出"),
            ])?;

            let rows = n * oh * ow;
            let mut col = vec![0.0f32; rows * patch];
            im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
            gemm(&col, weight, rows, cout, patch, out);
            add_bias(out, bias, cout);
            Ok(())
        }

        /// f32 反向：上游梯度 `dy` → `dx / dweight / dbias`。
        ///
        /// 与前向共用同一份 `im2col` 与卷积几何，三个输出缓冲**先清零再累加**（多上游可叠加）。
        ///
        /// # Errors
        ///
        /// 缓冲长度与形状不符时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn backward_f32(
            &self,
            x: &[f32],
            weight: &[f32],
            dy: &[f32],
            n: usize,
            h: usize,
            w: usize,
            dx: &mut [f32],
            dweight: &mut [f32],
            dbias: &mut [f32],
        ) -> Result<()> {
            let (oh, ow) = self.shape(h, w)?;
            let (cin, cout) = (self.cin, self.cout);
            let patch = self.geom.checked_patch(cin)?;
            let rows = n * oh * ow;
            check_shapes(&[
                (x.len(), n * h * w * cin, "输入"),
                (weight.len(), patch * cout, "权重"),
                (dy.len(), rows * cout, "上游梯度"),
                (dx.len(), n * h * w * cin, "输入梯度"),
                (dweight.len(), patch * cout, "权重梯度"),
                (dbias.len(), cout, "偏置梯度"),
            ])?;

            let mut col = vec![0.0f32; rows * patch];
            im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
            dx.fill(0.0);
            dweight.fill(0.0);
            dbias.fill(0.0);
            accumulate_weight_grad(dy, &col, dbias, dweight, rows, cout, patch);
            let mut dcol = vec![0.0f32; rows * patch];
            compute_dcol(dy, weight, &mut dcol, rows, cout, patch);
            col2im2d(&dcol, dx, (h, w), self, (oh, ow));
            Ok(())
        }

        /// i8 前向（对称量化）：输入与权重已量化，输出 i32 累加，`bias` 为同尺度整数偏置。
        ///
        /// 量化卷积的全部算力都落在 `fast::gemm_i8` 的多槽池上；`bias` 传全 0 可省。
        ///
        /// # Errors
        ///
        /// 缓冲长度与形状不符时返回说明。
        /// 复用外部 `col` 缓冲的 i8 卷积：大形状下省掉每轮 37.7MB 的零页分配与缺页。
        ///
        /// `col` 只需足够长（`rows * patch`）；不足时按需扩容，内容由 im2col 全量覆盖。
        #[allow(clippy::too_many_arguments)]
        pub fn forward_i8_with(
            &self,
            x: &[i8],
            weight: &[i8],
            bias: &[i32],
            n: usize,
            (h, w): (usize, usize),
            out: &mut [i32],
            col: &mut Vec<i8>,
        ) -> Result<()> {
            let (oh, ow) = self.shape(h, w)?;
            let cout = self.cout;
            let patch = self.geom.checked_patch(self.cin)?;
            check_shapes(&[
                (x.len(), n * h * w * self.cin, "输入"),
                (weight.len(), patch * cout, "权重"),
                (bias.len(), cout, "偏置"),
                (out.len(), n * oh * ow * cout, "输出"),
            ])?;

            let rows = n * oh * ow;
            col.resize(rows * patch, 0);
            im2col2d(x, col, n, (h, w), self, (oh, ow));
            fast::gemm_i8(col, weight, rows, cout, patch, out);
            add_bias(out, bias, cout);
            Ok(())
        }

        /// i8 卷积：`col` 内部现分配，单次调用；批量场景用 [`Self::forward_i8_with`] 复用缓冲。
        pub fn forward_i8(
            &self,
            x: &[i8],
            weight: &[i8],
            bias: &[i32],
            n: usize,
            (h, w): (usize, usize),
            out: &mut [i32],
        ) -> Result<()> {
            with_scratch_col(|col| self.forward_i8_with(x, weight, bias, n, (h, w), out, col))
        }
    }

    pub mod interp {
        //! 双线性插值采样与坐标重映射。
        use diagnostic::{Code, Error, Result};

        use crate::Scalar;
        use crate::conv::support::*;

        /// 插值取样方式：最近邻或双线性。
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Sampler {
            /// 取左上角样本，源坐标 `floor(i·h/oh)`。
            Nearest,
            /// 双线性，源坐标 `(i+0.5)·h/oh − 0.5` 后夹到边界。
            Linear,
        }

        /// 二维上/下采样，NHWC。
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct Interpolate {
            /// 取样方式。
            pub sampler: Sampler,
            /// 目标高。
            pub oh: usize,
            /// 目标宽。
            pub ow: usize,
        }

        impl Interpolate {
            /// 前向。
            ///
            /// # Errors
            ///
            /// 目标尺寸为 0 或缓冲长度不符时返回说明。
            pub fn forward<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                h: usize,
                w: usize,
                c: usize,
                out: &mut [S],
            ) -> Result<()> {
                self.ensure_nonzero(h, w)?;
                check_shapes(&[
                    (x.len(), n * h * w * c, "输入"),
                    (out.len(), n * self.oh * self.ow * c, "输出"),
                ])?;
                match self.sampler {
                    Sampler::Nearest => self.forward_nearest(x, n, h, w, c, out),
                    Sampler::Linear => self.forward_linear(x, n, h, w, c, out),
                }
                Ok(())
            }

            fn ensure_nonzero(&self, h: usize, w: usize) -> Result<()> {
                if self.oh == 0 || self.ow == 0 {
                    return Err(Error::new(Code::Invalid, "插值目标尺寸必须大于 0"));
                }
                if h == 0 || w == 0 {
                    return Err(Error::new(Code::Invalid, "插值输入尺寸必须大于 0"));
                }
                Ok(())
            }

            fn forward_nearest<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                h: usize,
                w: usize,
                c: usize,
                out: &mut [S],
            ) {
                for ni in 0..n {
                    for oi in 0..self.oh {
                        let ii = (oi * h / self.oh).min(h - 1);
                        for oj in 0..self.ow {
                            let jj = (oj * w / self.ow).min(w - 1);
                            let dst = ((ni * self.oh + oi) * self.ow + oj) * c;
                            let src = ((ni * h + ii) * w + jj) * c;
                            out[dst..dst + c].copy_from_slice(&x[src..src + c]);
                        }
                    }
                }
            }

            fn forward_linear<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                h: usize,
                w: usize,
                c: usize,
                out: &mut [S],
            ) {
                for ni in 0..n {
                    for oi in 0..self.oh {
                        let (y0, y1, ly) = axis_samples::<S>(oi, h, self.oh);
                        for oj in 0..self.ow {
                            let (x0, x1, lx) = axis_samples::<S>(oj, w, self.ow);
                            let dst = ((ni * self.oh + oi) * self.ow + oj) * c;
                            for ci in 0..c {
                                let a = x[((ni * h + y0) * w + x0) * c + ci];
                                let b = x[((ni * h + y0) * w + x1) * c + ci];
                                let d = x[((ni * h + y1) * w + x0) * c + ci];
                                let e = x[((ni * h + y1) * w + x1) * c + ci];
                                let top = a + (b - a) * lx;
                                let bot = d + (e - d) * lx;
                                out[dst + ci] = top + (bot - top) * ly;
                            }
                        }
                    }
                }
            }
        }

        fn axis_samples<S: Scalar>(i: usize, len: usize, out_len: usize) -> (usize, usize, S) {
            let f = ((S::from_usize(i) + S::from_f64(0.5)) * S::from_usize(len)
                / S::from_usize(out_len)
                - S::from_f64(0.5))
            .max(S::ZERO);
            let lo = f.floor().min(S::from_usize(len - 1)).to_usize();
            let hi = (lo + 1).min(len - 1);
            let lam = if lo == hi {
                S::ZERO
            } else {
                f - S::from_usize(lo)
            };
            (lo, hi, lam)
        }
    }

    pub mod pool {
        //! 职责：池化：max/avg 窗口归约与输出体量。
        use diagnostic::{Code, Error, Result};

        use crate::Scalar;

        use crate::conv::support::*;

        use super::ConvGeom;

        /// 二维最大池化：补零位置不参与比较（与框架语义一致）。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct MaxPool2d {
            /// 池化核、步长、补零（补零位置不参与比较）。
            pub geom: ConvGeom,
        }

        /// 二维平均池化：口径同 [`MaxPool2d`]，`count_include_pad` 决定补零是否计入分母。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct AvgPool2d {
            /// 池化核、步长、补零。
            pub geom: ConvGeom,
            /// 是否把补零位置计入平均的分母。
            pub count_include_pad: bool,
        }

        impl MaxPool2d {
            /// 输出形状。
            ///
            /// # Errors
            ///
            /// 形状不合法时返回说明。
            pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
                out_shape(&self.geom, h, w)
            }

            /// 前向：逐窗口取最大，补零位不参与。
            ///
            /// # Errors
            ///
            /// 缓冲长度不符时返回说明。
            pub fn forward<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                h: usize,
                w: usize,
                c: usize,
                out: &mut [S],
            ) -> Result<()> {
                run(&self.geom, x, n, (h, w), c, out, || {
                    MaxFold(S::NEG_INFINITY)
                })
            }
        }

        impl AvgPool2d {
            /// 输出形状。
            ///
            /// # Errors
            ///
            /// 形状不合法时返回说明。
            pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
                out_shape(&self.geom, h, w)
            }

            /// 前向：窗口求和后按口径取分母；`count_include_pad=false` 时只除有效位置。
            ///
            /// # Errors
            ///
            /// 缓冲长度不符时返回说明。
            pub fn forward<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                h: usize,
                w: usize,
                c: usize,
                out: &mut [S],
            ) -> Result<()> {
                let fold = || AvgFold {
                    sum: S::ZERO,
                    count: 0,
                    area: self.geom.kh * self.geom.kw,
                    include_pad: self.count_include_pad,
                };
                run(&self.geom, x, n, (h, w), c, out, fold)
            }
        }

        /// 池化窗口的归约口径：最大 / 平均。窗口遍历（补零裁剪、下标推导）全仓只有一份，
        /// 两条口径只在累加器与收尾不同；泛型化后各自单态展开，内层循环里没有分支。

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct MaxPool1d {
            /// 窗口长。
            pub k: usize,
            /// 步长。
            pub s: usize,
            /// 两端补零。
            pub p: usize,
        }

        impl MaxPool1d {
            /// 输出长度 `(l + 2p − k) / s + 1`。
            ///
            /// # Errors
            ///
            /// 窗口/步长为 0，或窗口比补零后的输入还长时返回说明。
            pub fn len(&self, l: usize) -> Result<usize> {
                if self.k == 0 || self.s == 0 {
                    return Err(Error::new(Code::Invalid, "池化窗口与步长必须大于 0"));
                }
                if l + 2 * self.p < self.k {
                    return Err(Error::new(Code::Invalid, "池化窗口大于补零后的输入"));
                }
                Ok((l + 2 * self.p - self.k) / self.s + 1)
            }

            /// 前向：逐窗口取最大。
            ///
            /// # Errors
            ///
            /// 缓冲长度不符时返回说明。
            pub fn forward<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                l: usize,
                c: usize,
                out: &mut [S],
            ) -> Result<()> {
                let ol = self.len(l)?;
                check_shapes(&[
                    (x.len(), n * l * c, "输入"),
                    (out.len(), n * ol * c, "输出"),
                ])?;
                self.pool_max(x, out, n, l, c, ol);
                Ok(())
            }

            fn pool_max<S: Scalar>(
                &self,
                x: &[S],
                out: &mut [S],
                n: usize,
                l: usize,
                c: usize,
                ol: usize,
            ) {
                for ni in 0..n {
                    for oi in 0..ol {
                        for ci in 0..c {
                            out[(ni * ol + oi) * c + ci] = self.window_max(x, ni, oi, ci, l, c);
                        }
                    }
                }
            }

            fn window_max<S: Scalar>(
                &self,
                x: &[S],
                ni: usize,
                oi: usize,
                ci: usize,
                l: usize,
                c: usize,
            ) -> S {
                let base = oi * self.s;
                let mut best = S::NEG_INFINITY;
                for ki in 0..self.k {
                    let src = base + ki;
                    if src < self.p || src - self.p >= l {
                        continue;
                    }
                    best = best.max(x[(ni * l + src - self.p) * c + ci]);
                }
                // 整窗都落在补零上（只会在极端几何下出现）按 0 处理。
                if best == S::NEG_INFINITY {
                    S::ZERO
                } else {
                    best
                }
            }
        }

        /// 二维自适应平均池化：任意输入尺寸都压到 `(oh, ow)`。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct AdaptiveAvgPool2d {
            /// 输出高。
            pub oh: usize,
            /// 输出宽。
            pub ow: usize,
        }

        impl AdaptiveAvgPool2d {
            /// 前向：每个输出格取覆盖窗口的平均。
            ///
            /// # Errors
            ///
            /// 输出尺寸为 0 或缓冲长度不符时返回说明。
            pub fn forward<S: Scalar>(
                &self,
                x: &[S],
                n: usize,
                h: usize,
                w: usize,
                c: usize,
                out: &mut [S],
            ) -> Result<()> {
                if self.oh == 0 || self.ow == 0 {
                    return Err(Error::new(Code::Invalid, "自适应池化输出尺寸必须大于 0"));
                }
                check_shapes(&[
                    (x.len(), n * h * w * c, "输入"),
                    (out.len(), n * self.oh * self.ow * c, "输出"),
                ])?;
                self.pool_avg(x, out, n, (h, w), c);
                Ok(())
            }

            fn pool_avg<S: Scalar>(
                &self,
                x: &[S],
                out: &mut [S],
                n: usize,
                (h, w): (usize, usize),
                c: usize,
            ) {
                for ni in 0..n {
                    for oi in 0..self.oh {
                        let (h0, h1) = adaptive_span(oi, h, self.oh);
                        for oj in 0..self.ow {
                            let (w0, w1) = adaptive_span(oj, w, self.ow);
                            let area = S::from_usize((h1 - h0) * (w1 - w0));
                            for ci in 0..c {
                                out[((ni * self.oh + oi) * self.ow + oj) * c + ci] =
                                    cell_sum(x, (ni, ci), (h0, h1), (w0, w1), (h, w), c) / area;
                            }
                        }
                    }
                }
            }
        }

        /// 自适应池化的连续输出区间 `[lo, hi)`。
        fn adaptive_span(i: usize, len: usize, out_len: usize) -> (usize, usize) {
            let lo = i * len / out_len;
            let hi = ((i + 1) * len).div_ceil(out_len).min(len);
            (lo, hi)
        }

        /// 求和 `x` 上矩形区域 `[h0,h1)×[w0,w1)` 的通道 `ci`。
        fn cell_sum<S: Scalar>(
            x: &[S],
            (ni, ci): (usize, usize),
            (h0, h1): (usize, usize),
            (w0, w1): (usize, usize),
            (h, w): (usize, usize),
            c: usize,
        ) -> S {
            let mut sum = S::ZERO;
            for ii in h0..h1 {
                for jj in w0..w1 {
                    sum = sum + x[((ni * h + ii) * w + jj) * c + ci];
                }
            }
            sum
        }
    }
}
pub mod scan {
    //! β 指数梯形扫描语义（§5.6，数学验证 5）：幺半群前缀扫描 + 对角 SSM 半群离散化。

    pub use crate::Scalar;

    /// 幺半群：`combine` 满足结合律，[`Monoid::IDENT`] 是幺元。
    pub trait Monoid: Clone {
        /// 幺元：`IDENT.combine(x)` 与 `x.combine(IDENT)` 都等于 `x`。
        const IDENT: Self;
        /// 结合合成（⊕）。
        fn combine(self, next: Self) -> Self;
    }

    /// 串行前缀（金标，§5.10 逐位一致）：`out[i]` = 前 `i` 个元素的合成，`out[0] = IDENT`。
    pub fn scan_serial<M: Monoid>(items: &[M]) -> Vec<M> {
        let mut out = Vec::with_capacity(items.len());
        let mut acc = M::IDENT;
        for item in items {
            out.push(acc.clone());
            acc = acc.combine(item.clone());
        }
        out
    }

    /// Blelloch 非交换独占扫描（数学验证 5b：≡ [`scan_serial`]）。
    pub fn scan_blelloch<M: Monoid>(items: &[M]) -> Vec<M> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let m = n.next_power_of_two();
        let mut buf = vec![M::IDENT; m];
        buf[..n].clone_from_slice(items);
        let d = up_sweep(&mut buf, m);
        down_sweep(&mut buf, m, d);
        buf.truncate(n);
        buf
    }

    /// up-sweep：归并成二叉树，`buf[right] = buf[left] ⊕ buf[right]`（左在前，保持时序），返回层数。
    fn up_sweep<M: Monoid>(buf: &mut [M], m: usize) -> usize {
        let mut d = 0usize;
        while (1usize << d) < m {
            let stride = 1usize << (d + 1);
            let mut k = 0usize;
            while k < m {
                let left = k + (1usize << d) - 1;
                let right = k + stride - 1;
                buf[right] = buf[left].clone().combine(buf[right].clone());
                k += stride;
            }
            d += 1;
        }
        d
    }

    /// down-sweep：把「本段之前的合成」从根向下传播。
    fn down_sweep<M: Monoid>(buf: &mut [M], m: usize, mut d: usize) {
        buf[m - 1] = M::IDENT;
        while d > 0 {
            d -= 1;
            let stride = 1usize << (d + 1);
            let mut k = 0usize;
            while k < m {
                let left = k + (1usize << d) - 1;
                let right = k + stride - 1;
                let before = buf[right].clone();
                let left_val = buf[left].clone();
                buf[left] = before.clone();
                buf[right] = before.combine(left_val);
                k += stride;
            }
        }
    }

    /// 对角 SSM 半群步：状态推进 `s' = a·s + b`（逐元素），是 [`Monoid`] 的实例。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct SsmStep<S> {
        /// 对角衰减系数（逐维）。
        pub a: S,
        /// 加性偏置（逐维）。
        pub b: S,
    }

    impl<S: Scalar> SsmStep<S> {
        /// 幺元步（状态不变）。
        pub const fn identity() -> Self {
            Self {
                a: S::ONE,
                b: S::ZERO,
            }
        }

        /// 施加一步：`s' = a·s + b`。
        pub fn apply(self, state: S) -> S {
            self.a * state + self.b
        }
    }

    impl<S: Scalar> Monoid for SsmStep<S> {
        const IDENT: Self = SsmStep {
            a: S::ONE,
            b: S::ZERO,
        };

        fn combine(self, next: Self) -> Self {
            SsmStep {
                a: next.a * self.a,
                b: next.a * self.b + next.b,
            }
        }
    }

    /// 零阶保持离散化（对角 `A`，§5.6）：连续 `h' = λ h + b x`，步长 `dt`，步内输入取常数。
    pub fn exp_zero_order<S: Scalar>(lam: S, b: S, dt: S) -> SsmStep<S> {
        let a = (lam * dt).exp();
        let b = if lam == S::ZERO {
            dt * b
        } else {
            (lam * dt).exp_m1() / lam * b
        };
        SsmStep { a, b }
    }

    /// 复值状态的 2×2 实值块：`s' = ā·s + b̄·(B·x)`，`s` 是相邻两个实值分量（实部、虚部）。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PlaneStep<S> {
        /// 复乘子 `ā`：`[实部, 虚部]`。
        pub a: [S; 2],
        /// 复输入系数 `b̄`：`[实部, 虚部]`。
        pub b: [S; 2],
    }

    impl<S: Scalar> PlaneStep<S> {
        /// 幺元步：`ā = 1`、`b̄ = 0`（状态不变）。
        pub const fn identity() -> Self {
            Self {
                a: [S::ONE, S::ZERO],
                b: [S::ZERO, S::ZERO],
            }
        }

        /// 施加一步：`(re', im') = ā·(re, im) + b̄·select`。
        pub fn apply(&self, state: [S; 2], select: S) -> [S; 2] {
            let [a_re, a_im] = self.a;
            let [b_re, b_im] = self.b;
            let [re, im] = state;
            [
                a_re * re - a_im * im + b_re * select,
                a_im * re + a_re * im + b_im * select,
            ]
        }
    }

    /// 双抽头指数梯形离散化（§5.6，Mamba-3）：`λ = λ_re + i·ω`，输入取 `x_{t−1}` 与 `x_t` 的平均。
    pub fn exp_trapezoid<S: Scalar>(lam_re: S, omega: S, dt: S, prev: S, cur: S) -> PlaneStep<S> {
        let decay = lam_re * dt;
        let phase = omega * dt;
        let magnitude = decay.exp();
        let [sin, cos] = [phase.sin(), phase.cos()];
        let [a_re, a_im] = [magnitude * cos, magnitude * sin];
        // (dt/2)·(ā·x_{t−1} + x_t)：`ā·x_{t−1}` 是复值，`x_t` 只进实部。
        let half = dt / (S::ONE + S::ONE);
        let b = [half * (a_re * prev + cur), half * (a_im * prev)];
        PlaneStep { a: [a_re, a_im], b }
    }

    /// 沿半群步序列推进状态，返回各步状态（`out[t]` = 施加前 `t` 步后的状态，`out[0] = init`）。
    pub fn apply_scan<S: Scalar>(steps: &[SsmStep<S>], init: S) -> Vec<S> {
        let mut out = Vec::with_capacity(steps.len() + 1);
        let mut state = init;
        out.push(state);
        for step in steps {
            state = step.apply(state);
            out.push(state);
        }
        out
    }

    #[cfg(test)]
    mod scan_suite {
        include!("../tests/conv_scan_suite.rs");
    }

    pub mod f32 {
        //! `scan_f32`：β 指数梯形扫描（§5.6）的离散递推 `s_t = a·s_{t-1} + b·x_t`。
        use crate::model::{Cost, Sample, ShapeCtx, cost_n};
        use diagnostic::{Code, Error};
        use tensor::View;

        // 单轴定义与形状投影复用 unary 骨架：时间轴 `n` 既是输入轴也是输出轴。
        pub use crate::unary::{AXES, project_shape};

        /// 对角衰减系数 `ā = exp(λ·dt)`（模板常量 `{{a}}`）。
        const DECAY: f32 = 0.5;
        /// 输入系数 `b̄ = expm1(λ·dt)/λ · b`（模板常量 `{{b}}`）。
        const INPUT: f32 = 1.0;

        /// 标量代价：每步一次乘加（2 flop）、读一次 `x`、写一次 `y`。
        ///
        /// # Errors
        ///
        /// 形状缺轴 `n` 时返回说明。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            cost_n(shape, "scan", 2.0, 1)
        }

        /// 确定性样本：`x`、`y` 两个等长缓冲，标量 `N` 是时间步数。
        pub fn sample(len: usize, _: usize) -> Sample {
            Sample::f32_io(len, len, len, len, vec![("N", len as i64)])
        }

        /// 标量参考：`s = ā·s + b̄·x[t]`，初态 `s = 0`，与内核同一公式。
        ///
        /// # Errors
        ///
        /// 样本缺 `x` 缓冲时返回说明。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let x = sample.f32("x")?;
            let mut state = 0.0f32;
            Ok(x.iter()
                .map(|value| {
                    state = DECAY * state + INPUT * value;
                    state
                })
                .collect())
        }

        /// 内核本体：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 输入与输出不等长时返回说明。
        pub fn forward(x: View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
            let x = x.as_slice();
            if x.len() != out.len() {
                return Err(Error::new(
                    Code::Invalid,
                    format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
                ));
            }
            let mut state = 0.0f32;
            for (out, value) in out.iter_mut().zip(x) {
                state = DECAY * state + INPUT * value;
                *out = state;
            }
            Ok(())
        }

        /// 标记类型：承载 `#[model]` 语义。
        pub struct Scan;

        /// β 指数梯形扫描的具体语义别名。
        #[annotation::model(name = "scan_f32", shape = [4096, 4096], dtype = "f32", skeleton = "scan")]
        pub type ScanF32 = Scan;

        crate::operator_here!(Scan, "scan_f32", 1, (4096, 4096), false, reference);
    }
}
mod support {
    //! 职责：卷积共享支撑：打包、切片与线程分派。
    use std::thread;

    use diagnostic::{Code, Error, Result};

    use crate::Scalar;
    use crate::fast;

    use super::geom::{Conv2d, ConvGeom};

    pub(crate) trait Fold<S: Scalar> {
        /// 吸收一个窗口内的有效元素（越界补零位不调用）。
        fn feed(&mut self, value: S);

        /// 收尾成该窗口的输出值。
        fn finish(self) -> S;
    }

    /// 取最大：起点 `-∞`，空窗口也返回 `-∞`。
    pub(crate) struct MaxFold<S>(pub(crate) S);

    impl<S: Scalar> Fold<S> for MaxFold<S> {
        fn feed(&mut self, value: S) {
            self.0 = self.0.max(value);
        }

        fn finish(self) -> S {
            self.0
        }
    }

    /// 取平均：分母按 `include_pad` 取窗口面积或有效格数（下限 1，空窗口不除零）。
    pub(crate) struct AvgFold<S> {
        pub(crate) sum: S,
        pub(crate) count: usize,
        pub(crate) area: usize,
        pub(crate) include_pad: bool,
    }

    impl<S: Scalar> Fold<S> for AvgFold<S> {
        fn feed(&mut self, value: S) {
            self.sum = self.sum + value;
            self.count += 1;
        }

        fn finish(self) -> S {
            let denom = if self.include_pad {
                self.area
            } else {
                self.count.max(1)
            };
            self.sum / S::from_usize(denom)
        }
    }

    /// 池化输出形状：两个池化类型的 `shape` 共用。
    pub(crate) fn out_shape(geom: &ConvGeom, h: usize, w: usize) -> Result<(usize, usize)> {
        Ok((geom.oh(h)?, geom.ow(w)?))
    }

    /// 池化前向骨架：形状推导、长度校验、遍历顺序共用，窗口归约由 `fold` 注入。
    pub(crate) fn run<S: Scalar, F: Fold<S>>(
        geom: &ConvGeom,
        x: &[S],
        n: usize,
        (h, w): (usize, usize),
        c: usize,
        out: &mut [S],
        fold: impl FnMut() -> F,
    ) -> Result<()> {
        let (oh, ow) = out_shape(geom, h, w)?;
        check_shapes(&[
            (x.len(), n * h * w * c, "输入"),
            (out.len(), n * oh * ow * c, "输出"),
        ])?;
        scatter(geom, x, n, (h, w), c, out, (oh, ow), fold);
        Ok(())
    }

    /// 按 `(n, oh, ow, c)` 四重遍历写满输出；窗口归约仍由 `fold` 注入。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scatter<S: Scalar, F: Fold<S>>(
        geom: &ConvGeom,
        x: &[S],
        n: usize,
        (h, w): (usize, usize),
        c: usize,
        out: &mut [S],
        (oh, ow): (usize, usize),
        mut fold: impl FnMut() -> F,
    ) {
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for ci in 0..c {
                        out[((ni * oh + oi) * ow + oj) * c + ci] =
                            window(geom, x, (ni, oi, oj, ci), (h, w), c, fold());
                    }
                }
            }
        }
    }

    /// 单个输出格的窗口归约：按 `(kh, kw)` 扫描，越界（补零）位置直接跳过。
    pub(crate) fn window<S: Scalar, F: Fold<S>>(
        geom: &ConvGeom,
        x: &[S],
        (ni, oi, oj, ci): (usize, usize, usize, usize),
        (h, w): (usize, usize),
        c: usize,
        mut fold: F,
    ) -> S {
        for ki in 0..geom.kh {
            let ii = (oi * geom.sh + ki) as isize - geom.ph as isize;
            if !in_bounds_1d(ii, h) {
                continue;
            }
            for kj in 0..geom.kw {
                let jj = (oj * geom.sw + kj) as isize - geom.pw as isize;
                if !in_bounds_1d(jj, w) {
                    continue;
                }
                fold.feed(x[((ni * h + ii as usize) * w + jj as usize) * c + ci]);
            }
        }
        fold.finish()
    }

    /// 一维边界判定。
    pub(crate) fn in_bounds_1d(idx: isize, len: usize) -> bool {
        idx >= 0 && (idx as usize) < len
    }

    /// 二维边界判定。
    pub(crate) fn in_bounds(ii: isize, jj: isize, h: usize, w: usize) -> bool {
        in_bounds_1d(ii, h) && in_bounds_1d(jj, w)
    }

    /// f32/i8 共用的 im2col：行 = `(n, oh, ow)`，列 = `(kh, kw, cin)`。
    pub(crate) fn im2col2d<T: Copy + Send + Sync + Default>(
        x: &[T],
        col: &mut [T],
        n: usize,
        (h, w): (usize, usize),
        conv: &Conv2d,
        (oh, ow): (usize, usize),
    ) {
        let patch = conv.geom.patch(conv.cin);
        let rows = n * oh * ow;
        parallel_rows(x, col, rows, patch, |src, row, r| {
            fill_row(src, row, r, (h, w), conv, (oh, ow));
        });
    }

    /// 按可用并行度决定单线程还是线程池分片；小张量退化为顺序，避免并行负加速。
    pub(crate) fn thread_count(rows: usize, patch: usize) -> usize {
        if rows * patch < (1 << 16) {
            1
        } else {
            thread::available_parallelism()
                .map_or(1, std::num::NonZero::get)
                .min(rows)
        }
    }

    /// 把 `rows` 行按线程分片后逐行调用 `fill`。
    pub(crate) fn parallel_rows<T: Copy + Send + Sync>(
        x: &[T],
        col: &mut [T],
        rows: usize,
        patch: usize,
        fill: impl Fn(&[T], &mut [T], usize) + Sync + Send + Copy,
    ) {
        let threads = thread_count(rows, patch);
        if threads <= 1 {
            for r in 0..rows {
                fill(x, &mut col[r * patch..(r + 1) * patch], r);
            }
            return;
        }
        let per = rows.div_ceil(threads);
        thread::scope(|s| {
            for (bi, chunk) in col.chunks_mut(per * patch).enumerate() {
                let r0 = bi * per;
                s.spawn(move || fill_rows(x, chunk, patch, r0, fill));
            }
        });
    }

    /// 一个分片上把行逐一交给 `fill`：`chunk` 覆盖 `r0` 起的连续行。
    pub(crate) fn fill_rows<T: Copy>(
        x: &[T],
        chunk: &mut [T],
        patch: usize,
        r0: usize,
        fill: impl Fn(&[T], &mut [T], usize) + Sync + Copy,
    ) {
        for base in 0..chunk.len() / patch {
            fill(x, &mut chunk[base * patch..(base + 1) * patch], r0 + base);
        }
    }

    /// 展开一行：`row_idx = (ni, oi, oj)` 对应的一整块 `(kh, kw, cin)` 拷贝，越界补零。
    pub(crate) fn fill_row<T: Copy + Default>(
        x: &[T],
        row: &mut [T],
        row_idx: usize,
        (h, w): (usize, usize),
        conv: &Conv2d,
        (oh, ow): (usize, usize),
    ) {
        let cin = conv.cin;
        let g = conv.geom;
        let ni = row_idx / (oh * ow);
        let rem = row_idx % (oh * ow);
        let oi = rem / ow;
        let oj = rem % ow;
        for ki in 0..g.kh {
            let ii = (oi * g.sh + ki) as isize - g.ph as isize;
            for kj in 0..g.kw {
                let jj = (oj * g.sw + kj) as isize - g.pw as isize;
                let tail = (ki * g.kw + kj) * cin;
                if in_bounds(ii, jj, h, w) {
                    let src = ((ni * h + ii as usize) * w + jj as usize) * cin;
                    row[tail..tail + cin].copy_from_slice(&x[src..src + cin]);
                } else {
                    row[tail..tail + cin].fill(T::default());
                }
            }
        }
    }

    /// `im2col` 的逆向：把每行的收缩梯度按同一窗口位置累加回输入（`stride < k` 的重叠位也要累加）。
    pub(crate) fn col2im2d(
        col: &[f32],
        dx: &mut [f32],
        (h, w): (usize, usize),
        conv: &Conv2d,
        (oh, ow): (usize, usize),
    ) {
        let patch = conv.geom.patch(conv.cin);
        let rows = col.len() / patch;
        for row in 0..rows {
            accumulate_col_row(col, dx, row, (h, w), conv, (oh, ow));
        }
    }

    /// 散射一行（`row` 对应 `(ni, oi, oj)`）的列梯度。
    pub(crate) fn accumulate_col_row(
        col: &[f32],
        dx: &mut [f32],
        row: usize,
        (h, w): (usize, usize),
        conv: &Conv2d,
        (oh, ow): (usize, usize),
    ) {
        let patch = conv.geom.patch(conv.cin);
        let cin = conv.cin;
        let g = conv.geom;
        let ni = row / (oh * ow);
        let rem = row % (oh * ow);
        let oi = rem / ow;
        let oj = rem % ow;
        for ki in 0..g.kh {
            let ii = (oi * g.sh + ki) as isize - g.ph as isize;
            if !in_bounds_1d(ii, h) {
                continue;
            }
            for kj in 0..g.kw {
                let jj = (oj * g.sw + kj) as isize - g.pw as isize;
                if !in_bounds_1d(jj, w) {
                    continue;
                }
                let tail = (ki * g.kw + kj) * cin;
                let dst = ((ni * h + ii as usize) * w + jj as usize) * cin;
                accumulate_cell(dx, col, dst, row * patch + tail, cin);
            }
        }
    }

    /// 把一个列片段的 `cin` 个值累加进 `dx`。
    pub(crate) fn accumulate_cell(dx: &mut [f32], col: &[f32], dst: usize, src: usize, cin: usize) {
        for ci in 0..cin {
            dx[dst + ci] += col[src + ci];
        }
    }

    /// 一维 im2col：行 = `(n, ol)`，列 = `(kw, cin)`。
    pub(crate) fn im2col1d(
        x: &[f32],
        col: &mut [f32],
        n: usize,
        l: usize,
        ol: usize,
        g: ConvGeom,
        cin: usize,
    ) {
        let patch = g.patch(cin);
        for ni in 0..n {
            for oi in 0..ol {
                fill_row_1d(
                    x,
                    &mut col[(ni * ol + oi) * patch..(ni * ol + oi + 1) * patch],
                    ni,
                    oi,
                    l,
                    g,
                    cin,
                );
            }
        }
    }

    /// 展开一维卷积的一行感受野。
    pub(crate) fn fill_row_1d(
        x: &[f32],
        row: &mut [f32],
        ni: usize,
        oi: usize,
        l: usize,
        g: ConvGeom,
        cin: usize,
    ) {
        for ki in 0..g.kw {
            let src = (oi * g.sw + ki) as isize - g.pw as isize;
            for ci in 0..cin {
                row[ki * cin + ci] = if in_bounds_1d(src, l) {
                    x[(ni * l + src as usize) * cin + ci]
                } else {
                    0.0
                };
            }
        }
    }

    /// f32 GEMM 转调：`out[m,n] = a[m,k] · b[k,n]`，走 `fast` 的多线程内核。
    pub(crate) fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
        fast::gemm_f32(a, b, m, n, k, out);
    }

    /// 每行加同一个偏置向量，f32/i32 共用。
    pub(crate) fn add_bias<T: Copy + core::ops::AddAssign>(out: &mut [T], bias: &[T], cout: usize) {
        for row in out.chunks_mut(cout) {
            for (slot, b) in row.iter_mut().zip(bias.iter()) {
                *slot += *b;
            }
        }
    }

    /// 批量缓冲长度校验：任一缓冲小于所需即报错，错误信息里带缓冲名。
    pub(crate) fn check_shapes(items: &[(usize, usize, &str)]) -> Result<()> {
        for &(actual, required, what) in items {
            if actual < required {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{what}: 缓冲 {actual}，需要 {required}"),
                ));
            }
        }
        Ok(())
    }

    /// 线程本地复用的 i8 im2col 缓冲：大形状下省掉每次的零页分配与缺页。
    pub(crate) fn with_scratch_col(f: impl FnOnce(&mut Vec<i8>) -> Result<()>) -> Result<()> {
        thread_local! {
            /// 大形状下 im2col 缓冲达几十 MB，跨调用复用可以省掉每次的零页分配与缺页。
            static COL: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        COL.with(|cell| {
            let mut loan = cell.try_borrow_mut();
            match loan.as_mut() {
                Ok(col) => f(col),
                // 同线程嵌套调用（借出中）时才退回一次性缓冲。
                Err(_) => f(&mut Vec::new()),
            }
        })
    }

    /// 权重梯度累计：`dweight += col^T · dy`，`dbias += Σ dy`。
    pub(crate) fn accumulate_weight_grad(
        dy: &[f32],
        col: &[f32],
        dbias: &mut [f32],
        dweight: &mut [f32],
        rows: usize,
        cout: usize,
        patch: usize,
    ) {
        for r in 0..rows {
            add_row_bias(dbias, &dy[r * cout..(r + 1) * cout]);
            accumulate_dweight_row(
                &col[r * patch..(r + 1) * patch],
                &dy[r * cout..(r + 1) * cout],
                dweight,
                cout,
            );
        }
    }

    pub(crate) fn add_row_bias(dbias: &mut [f32], dy_row: &[f32]) {
        for (slot, value) in dbias.iter_mut().zip(dy_row) {
            *slot += *value;
        }
    }

    pub(crate) fn accumulate_dweight_row(
        col_row: &[f32],
        dy_row: &[f32],
        dweight: &mut [f32],
        cout: usize,
    ) {
        for (k, &value) in col_row.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            for (slot, upstream) in dweight[k * cout..(k + 1) * cout].iter_mut().zip(dy_row) {
                *slot += value * *upstream;
            }
        }
    }

    /// `dcol = dy · weight^T`，供 col2im 散射回 `dx`。
    pub(crate) fn compute_dcol(
        dy: &[f32],
        weight: &[f32],
        dcol: &mut [f32],
        rows: usize,
        cout: usize,
        patch: usize,
    ) {
        for r in 0..rows {
            let dy_row = &dy[r * cout..(r + 1) * cout];
            let dcol_row = &mut dcol[r * patch..(r + 1) * patch];
            for (k, slot) in dcol_row.iter_mut().enumerate() {
                *slot = dot(dy_row, &weight[k * cout..(k + 1) * cout]);
            }
        }
    }

    pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| *x * *y).sum()
    }
}

pub use attention::decode_attention;
pub use classic::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d,
};
pub use geom::interp;
pub use geom::pool;
pub use geom::{Conv1d, Conv2d, ConvGeom};
pub use interp::{Interpolate, Sampler};
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use scan::f32 as scan_f32;

#[cfg(test)]
#[path = "../tests/conv_suite.rs"]
mod conv_suite;
