//! 装配：把 MLA 注意力、MoE 前馈与 mHC 多流残差串成**一层**可跑的前向。
//!
//! 前面各模块都给的是「一件机制的数学 + 它的可测不变量」——潜在 KV 只存 `(c^KV, k^R)`、
//! 稀疏专家只激活一部分 θ、多流残差只做双随机混合。各自单测都在，但**没有任何一次前向同时用到
//! 它们**，于是「机制齐了」与「能跑一层」之间还差一格。本模块就是那一格：一层 = 混流 → 注意力
//! 残差 → 前馈残差，三件都调各自模块的唯一入口，本层不重写任何一条算式。
//!
//! # 次序与每处取舍的出处
//!
//! 1. **层首混流**：`B_l` 作用在进入本层的 `M` 条流上。这与「层末混流」（`x_{l+1} = B_l x_l`）
//!    是同一件事——边界只是画在哪一步。放在层首有两个具体好处：[`Mhc::mix_streams`] 自带
//!    「条数 = `M`、各条等宽」的入口校验，于是几何错在**任何写入之前**就报掉；而且它返回新向量、
//!    不改输入，天然就是「算完再提交」的那一半。
//! 2. **注意力只读缓存**：本 token 的 `(c^KV, k^R)` 由调用方在 [`Hybrid::step`] 之前 `push`。
//!    缓存的行数归页账（`engine::kv`）所有，本层既不持有也不推进，所以 `step` 失败不会让序列
//!    多一行或少一行——「失败不写半截」在这里是结构性的，不是靠回滚。
//! 3. **先注意力、后前馈，两者都是加**：残差用 [`crate::csa::accumulate`] 加权累加、权重恒
//!    `S::ONE`，不另写一遍逐元素加法。前馈吃的是**加过注意力的主路**，与
//!    「`AttentionBlock` → FFN」的串行是同一个次序。
//! 4. **只有主路进注意力与前馈**：KV 只有一份。多流是**残差的副本**，不是 `M` 份 KV——否则 mHC
//!    会把 MLA 的行宽收益按 `M` 乘回去，而 V3 的 `576` 字节/token 是「每层」的账，不是「每流」的。
//! 5. **算出的一份增量只加回主路**：其余流带走混流后的值。同一份增量加到每条流上会让它们从此
//!    完全相同，多流当场退化成单流；只加一条、层间靠 `B_l` 再把它们摊开，流之间才始终留着可被
//!    读取的差异（[`crate::attnres`] 读的正是这份差异）。
//!
//! # 本模块不拥有什么
//!
//! 投影矩阵（`q`/`k`/`v` 的 `W` 与查询的两段都从外面递进来）、归一化与残差缩放（`nn` 里
//! `RmsNorm`/`Scale` 是**声明**而非算术，执行侧才有实现）、通道记忆与跨层读取（[`crate::kda`]、
//! [`crate::attnres`] 各自是自己的入口，本层不转包同名方法——多一层转发只会让「这一层拥有什么」
//! 变模糊），以及训练状态。
//!
//! 也不声称这是 K3 的完整逐层配方：3:1 的 KDA:MLA 配比、归一化的位置与投影形状都属装配侧。
//! 本层承诺的只有一件事——**这五件里被选中的三件，按上面这个次序，能跑出一层前向**，且每一处
//! 连接点的理由都写在上面的清单里。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::accumulate;
use crate::mhc::Mhc;
use crate::mla::{Latent, Mla, Path};
use crate::moe::Moe;

/// 报错出处：几何与位置判定都收在装配的入口，不留到矩阵乘的中途。
const SITE: &str = "nn::hybrid";

/// 本 token 的查询两段。分开传是因为它们**不是**同一张投影的切片：`q^{no-nope}` 进吸收式路径，
/// `q^R` 直接与解耦 RoPE 键点积（见 [`Mla::attend`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Query<'a, S> {
    /// `q^{no-nope}`，宽 `n_h · d_h`。
    pub nope: &'a [S],
    /// `q^R`，宽 `n_h · d_h^R`。
    pub rope: &'a [S],
}

/// 一层：MLA 注意力 + MoE 前馈 + mHC 多流残差。
///
/// 三件都是构造期定格的（不持可变参数），一层里唯一的运行期状态是调用方手里的 `streams` 与
/// 页账上的 `cache`——于是「这一步算了什么」完全由入参决定，本层不做隐藏的累积。
#[derive(Debug, Clone, PartialEq)]
pub struct Hybrid<S: Scalar = f32> {
    attention: Mla<S>,
    ffn: Moe<S>,
    residual: Mhc<S>,
}

impl<S: Scalar> Hybrid<S> {
    /// 装配一层。
    ///
    /// # Errors
    ///
    /// 注意力输出宽 `n_h · d_h` 必须等于前馈宽——不等的话每一步都会把「另一层的数」喂进专家，
    /// 内积照算、结果却是另一个数，所以在这里拒绝；`n_h · d_h` 溢出同样返回 `Code::Invalid`。
    pub fn new(attention: Mla<S>, ffn: Moe<S>, residual: Mhc<S>) -> Result<Self> {
        let width = attention
            .heads()
            .checked_mul(attention.head_dim())
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "注意力输出宽 n_h × d_h 溢出（{} × {}）",
                        attention.heads(),
                        attention.head_dim()
                    ),
                )
                .with(SITE)
            })?;
        if width != ffn.width() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "注意力输出宽 n_h × d_h = {width}，前馈宽 {}：两者必须相等，否则残差相加会把错位的数据拼在一起",
                    ffn.width()
                ),
            )
            .with(SITE));
        }
        Ok(Self {
            attention,
            ffn,
            residual,
        })
    }

    /// 本层宽度（= `n_h · d_h` = 前馈宽）：三条流与两段查询都由它定标。
    pub fn width(&self) -> usize {
        self.ffn.width()
    }

    /// 残差流条数 `M`。
    pub fn streams(&self) -> usize {
        self.residual.streams()
    }

    /// 注意力侧（只读视图；本层不替调用方缓存，也不转包它的方法）。
    pub fn attention(&self) -> &Mla<S> {
        &self.attention
    }

    /// 前馈侧（只读视图）。
    pub fn ffn(&self) -> &Moe<S> {
        &self.ffn
    }

    /// 多流残差侧（只读视图）。
    pub fn residual(&self) -> &Mhc<S> {
        &self.residual
    }

    /// 把一条隐藏向量铺成 `M` 条流：多流的初值就是同一个值（混合之后才分开）。
    ///
    /// # Errors
    ///
    /// `hidden` 的宽度不是本层宽度时返回 `Code::Invalid`——长度不对的初值会被后面的混流当成
    /// 「另一层的数」混进来，而混流本身看不出这一点。
    pub fn seed(&self, hidden: &[S]) -> Result<Vec<Vec<S>>> {
        if hidden.len() != self.width() {
            return Err(Error::new(
                Code::Invalid,
                format!("初值宽度是 {}，本层宽度是 {}", hidden.len(), self.width()),
            )
            .with(SITE));
        }
        Ok(vec![hidden.to_vec(); self.streams()])
    }

    /// 第 `t` 个位置的一次前向：混流 → 注意力残差 → 前馈残差，末尾一次性写回 `streams`。
    ///
    /// 缓存的最后一行就是本 token 的键：位置取 `cache.seq() - 1`，于是「先 `push` 再 `step`」
    /// 由调用方的次序保证，本层无法读到还没入缓存的位置。
    ///
    /// 算出的增量只加回**主路**（第 0 条），其余流带走混流后的值：同一份增量加给每条流会让它们
    /// 从此完全相同，多流就退化成单流。
    ///
    /// # Errors
    ///
    /// 流条数/宽度不符、查询几何不符、缓存里一个位置都没有——都返回 `Code::Invalid`，且
    /// `streams` 与 `cache` **都没有被改动**（几何与位置判定全部先于第一次写入）。
    pub fn step(
        &self,
        streams: &mut [Vec<S>],
        cache: &Latent<S>,
        query: &Query<'_, S>,
        path: Path,
    ) -> Result<()> {
        let mut mixed = self.residual.mix_streams(streams)?;
        self.check_width(&mixed)?;
        let t = cursor(cache)?;
        self.attach(&mut mixed[0], cache, query, t, path)?;
        commit(streams, mixed);
        Ok(())
    }

    /// 主路的一层增量：先注意力、后前馈，两者都是加——权重恒 `S::ONE`，与「`AttentionBlock` →
    /// FFN」是同一个次序；前馈吃的是**加过注意力**的那一份。
    fn attach(
        &self,
        main: &mut [S],
        cache: &Latent<S>,
        query: &Query<'_, S>,
        t: usize,
        path: Path,
    ) -> Result<()> {
        let attended = self
            .attention
            .attend(cache, query.nope, query.rope, t, path)?;
        accumulate(main, &attended, S::ONE);
        let fed = self.ffn.forward(main)?;
        accumulate(main, &fed, S::ONE);
        Ok(())
    }

    /// 混流之后的宽度必须正好是本层宽度：宽度多一格时 [`accumulate`] 会静默只加重叠部分。
    fn check_width(&self, mixed: &[Vec<S>]) -> Result<()> {
        let width = self.width();
        match mixed.iter().position(|row| row.len() != width) {
            Some(bad) => Err(Error::new(
                Code::Invalid,
                format!(
                    "第 {bad} 条流宽度是 {}，本层宽度是 {width}（= n_h · d_h = 前馈宽）",
                    mixed[bad].len()
                ),
            )
            .with(SITE)),
            None => Ok(()),
        }
    }
}

/// 本 token 的位置：缓存的最后一行。
fn cursor<S: Scalar>(cache: &Latent<S>) -> Result<usize> {
    cache.seq().checked_sub(1).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            "缓存里没有任何位置：本 token 的 (c^KV, k^R) 必须先 push 再 step，否则注意力会对着空历史算",
        )
        .with(SITE)
    })
}

/// 算完之后一次性写回调用方的流：中途失败过的调用不会留下「一半新一半旧」。
fn commit<S: Scalar>(streams: &mut [Vec<S>], mixed: Vec<Vec<S>>) {
    for (slot, row) in streams.iter_mut().zip(mixed) {
        *slot = row;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 小几何：2 头 × 头宽 2 = 4（= 前馈宽），潜在 2 + RoPE 1 = 每 token 3 个元素。
    const WIDTH: usize = 4;

    /// 投影 `[n_h · d_h, d_c]` = `[4, 2]`：`seed` 只用来让两个投影不一样。
    fn projection(seed: f64) -> Vec<f64> {
        (0..8).map(|i| seed + 0.125 * i as f64).collect()
    }

    /// 注意力层。
    fn attention() -> Mla<f64> {
        Mla::new(2, 2, 2, 1, &projection(-0.5), &projection(0.25)).expect("几何合法")
    }

    /// 专家：每 5 个位置一个 `scale`，其余为 0，于是「哪个专家参与」能直接从输出上读出来。
    fn expert(scale: f64) -> Vec<f64> {
        (0..16)
            .map(|i| if i % 5 == 0 { scale } else { 0.0 })
            .collect()
    }

    /// 路由矩阵 `[专家数, width]`：全部质量给第 0 个专家，于是选中集可以预期。
    fn router() -> Vec<f64> {
        let mut router = vec![0.0; 2 * WIDTH];
        router[0] = 10.0;
        router
    }

    /// 两个专家、只激活 1 个；路由矩阵让第 0 个专家拿到全部质量。
    fn ffn_with_second(scale: f64) -> Moe<f64> {
        Moe::new(WIDTH, 1, router(), vec![expert(1.0), expert(scale)], None).expect("几何合法")
    }

    /// 默认前馈：第 1 个专家按 2 倍缩放，但路由只选第 0 个。
    fn ffn() -> Moe<f64> {
        ffn_with_second(2.0)
    }

    /// 前馈恒零：用来把「注意力残差」单独读出来。
    fn silent_ffn() -> Moe<f64> {
        Moe::new(WIDTH, 1, router(), vec![vec![0.0; 16], vec![0.0; 16]], None).expect("几何合法")
    }

    /// 两个位置（第二个就是本 token），潜在与 RoPE 两段宽度合法。
    fn cache() -> Latent<f64> {
        let mut cache = attention().empty_cache();
        cache.push(&[0.5, -0.25], &[0.75]).expect("宽度合法");
        cache.push(&[-1.0, 0.125], &[0.5]).expect("宽度合法");
        cache
    }

    /// 本 token 的查询两段。
    fn query() -> Query<'static, f64> {
        Query {
            nope: &[1.0, -0.5, 0.25, 2.0],
            rope: &[0.5, -0.25],
        }
    }

    /// 一层：注意力 + 前馈 + `M` 条单位混合流。
    fn layer(ffn: Moe<f64>, streams: usize) -> Hybrid<f64> {
        let residual = Mhc::identity(streams).expect("流数非零");
        Hybrid::new(attention(), ffn, residual).expect("宽度一致")
    }

    /// 单流 + 单位混合时，这一层必须逐位等于「主路 + 注意力 + 前馈」：装配不许自己长出新算术。
    #[test]
    fn one_stream_identity_mixing_is_plain_residual_arithmetic() {
        let layer = layer(ffn(), 1);
        let cache = cache();
        let query = query();
        let mut streams = vec![vec![1.0, 2.0, 3.0, 4.0]];
        layer
            .step(&mut streams, &cache, &query, Path::Absorb)
            .expect("前向");

        let mut expected = vec![1.0, 2.0, 3.0, 4.0];
        let attended = attention()
            .attend(&cache, query.nope, query.rope, 1, Path::Absorb)
            .expect("位置在缓存内");
        accumulate(&mut expected, &attended, 1.0);
        let fed = ffn().forward(&expected).expect("宽度一致");
        accumulate(&mut expected, &fed, 1.0);
        assert_eq!(streams[0], expected, "装配层与三件单独调用必须给出同一份数");
    }

    /// 两条路径在装配层里也给出同一个数（`f64` 按 `1e-12`）：等价性不是模块内的巧合。
    #[test]
    fn the_two_attention_paths_agree_inside_the_layer() {
        let layer = layer(ffn(), 1);
        let cache = cache();
        let query = query();
        let mut absorbed = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut expanded = absorbed.clone();
        layer
            .step(&mut absorbed, &cache, &query, Path::Absorb)
            .expect("吸收式");
        layer
            .step(&mut expanded, &cache, &query, Path::Upref)
            .expect("升维参考");
        for (a, b) in absorbed[0].iter().zip(&expanded[0]) {
            assert!((a - b).abs() < 1e-12, "两条路径差 {}", (a - b).abs());
        }
    }

    /// 前馈恒零时只剩注意力残差：钉住「注意力在前、前馈在后」这一处次序。
    #[test]
    fn a_silent_ffn_leaves_only_the_attention_residual() {
        let layer = layer(silent_ffn(), 1);
        let cache = cache();
        let query = query();
        let hidden = vec![0.5, -1.0, 2.0, 0.25];
        let mut streams = vec![hidden.clone()];
        layer
            .step(&mut streams, &cache, &query, Path::Absorb)
            .expect("前向");

        let mut expected = hidden;
        let attended = attention()
            .attend(&cache, query.nope, query.rope, 1, Path::Absorb)
            .expect("位置在缓存内");
        accumulate(&mut expected, &attended, 1.0);
        assert_eq!(streams[0], expected);
    }

    /// 未选中的专家改了也影响不到输出：装配层里稀疏性仍然成立。
    #[test]
    fn an_inactive_expert_cannot_change_the_output() {
        let hidden = vec![0.5, -1.0, 2.0, 0.25];
        let chosen: Vec<usize> = ffn()
            .route(&hidden)
            .expect("宽度一致")
            .into_iter()
            .map(|(expert, _)| expert)
            .collect();
        assert_eq!(chosen, vec![0], "测试前提：主路只选第 0 个专家");

        let cache = cache();
        let query = query();
        let mut calm = vec![hidden.clone()];
        let mut loud = vec![hidden];
        layer(ffn(), 1)
            .step(&mut calm, &cache, &query, Path::Absorb)
            .expect("前向");
        layer(ffn_with_second(500.0), 1)
            .step(&mut loud, &cache, &query, Path::Absorb)
            .expect("前向");
        assert_eq!(calm[0], loud[0], "第 1 个专家没被选中，改它不该改结果");
    }

    /// 缓存只读：本层不推进 `seq`，`M` 条流也不会变成 `M` 份 KV。
    #[test]
    fn the_layer_never_touches_the_cache() {
        let layer = layer(ffn(), 4);
        let cache = cache();
        let before = (cache.seq(), cache.cache_elements());
        let (latent, rope) = cache.rows();
        let rows = (latent.to_vec(), rope.to_vec());

        let mut streams = layer.seed(&[1.0, 2.0, 3.0, 4.0]).expect("初值宽度合法");
        layer
            .step(&mut streams, &cache, &query(), Path::Absorb)
            .expect("前向");

        assert_eq!((cache.seq(), cache.cache_elements()), before);
        let (latent, rope) = cache.rows();
        assert_eq!((latent.to_vec(), rope.to_vec()), rows, "缓存内容逐位未变");
    }

    /// 层首混流把各条流摊成同一份；增量只加回主路，另一条流带走混流后的值。
    #[test]
    fn mixing_spreads_the_streams_and_only_the_main_stream_takes_the_delta() {
        let residual = Mhc::new(&[1.0, 1.0, 1.0, 1.0], 2).expect("两流");
        let layer = Hybrid::new(attention(), ffn(), residual).expect("宽度一致");
        let cache = cache();
        let query = query();
        let mut streams = vec![vec![0.0; WIDTH], vec![8.0; WIDTH]];
        layer
            .step(&mut streams, &cache, &query, Path::Absorb)
            .expect("前向");

        let mixed = vec![4.0; WIDTH];
        assert_eq!(
            streams[1], mixed,
            "非主路流只带走混流后的值（全 1 阵归一后取均值）"
        );

        let mut expected = mixed;
        let attended = attention()
            .attend(&cache, query.nope, query.rope, 1, Path::Absorb)
            .expect("位置在缓存内");
        accumulate(&mut expected, &attended, 1.0);
        let fed = ffn().forward(&expected).expect("宽度一致");
        accumulate(&mut expected, &fed, 1.0);
        assert_eq!(streams[0], expected, "主路必须比另一条流多出这一层的增量");
        assert_ne!(
            streams[0], streams[1],
            "增量只加一条流：多流不会被拉平成单流"
        );
    }

    /// 空缓存必须在写到任何地方之前被拒：位置读不到「还没生成到那一步」。
    #[test]
    fn an_empty_cache_is_refused_before_anything_is_touched() {
        let layer = layer(ffn(), 1);
        let empty = attention().empty_cache();
        let original = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut streams = original.clone();
        let err = layer
            .step(&mut streams, &empty, &query(), Path::Absorb)
            .expect_err("空缓存不该通过");
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(streams, original, "失败之后流必须原样退回");
    }

    /// 流几何错在入口被拒，且流不被改动：条数错、宽度不齐、宽度不等于本层宽度。
    #[test]
    fn stream_geometry_is_checked_at_entry() {
        let layer = layer(ffn(), 2);
        let cache = cache();
        let cases: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.0; WIDTH]],
            vec![vec![1.0; WIDTH], vec![2.0; WIDTH + 1]],
            vec![vec![1.0; WIDTH], vec![2.0; WIDTH - 1]],
        ];
        for case in cases {
            let mut streams = case.clone();
            let err = layer
                .step(&mut streams, &cache, &query(), Path::Absorb)
                .expect_err("几何不对不该通过");
            assert_eq!(err.code(), Code::Invalid);
            assert_eq!(streams, case, "失败之后流必须原样退回");
        }
    }

    /// 装配期就拒绝宽度不一致：注意力输出宽与前馈宽必须相等。
    #[test]
    fn assembly_rejects_a_width_mismatch() {
        let narrow = Moe::new(
            WIDTH - 1,
            1,
            vec![1.0; 9],
            vec![vec![1.0; 9], vec![1.0; 9], vec![1.0; 9]],
            None,
        )
        .expect("宽度 3 的两专家");
        let err = Hybrid::new(attention(), narrow, Mhc::identity(1).expect("单流"))
            .expect_err("宽度不一致不该通过");
        assert_eq!(err.code(), Code::Invalid);
    }

    /// 铺初值：每条流都是同一个值；宽度不对则拒绝。
    #[test]
    fn seeding_copies_the_hidden_state_into_every_stream() {
        let layer = layer(ffn(), 3);
        let seeded = layer.seed(&[1.0, 2.0, 3.0, 4.0]).expect("宽度合法");
        assert_eq!(seeded.len(), 3);
        assert!(seeded.iter().all(|row| row == &vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(layer.width(), WIDTH);
        assert_eq!(layer.streams(), 3);
        assert_eq!(
            layer.seed(&[1.0, 2.0, 3.0]).expect_err("宽度不对").code(),
            Code::Invalid
        );
    }
}
