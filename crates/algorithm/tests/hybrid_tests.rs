//! `hybrid` 的装配测试：一层 7 条、一栈 12 条。
//!
//! 这里是唯一放装配算术对拍的地方——生产代码只保留装配与提交，"算得对不对"全部由本文件的
//! 用例回答，于是 `hybrid.rs` 只装结构、不装断言。

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

/// 主路的参考值：注意力与前馈各跑一遍、增量按 `1.0` 累加，次序与 `Hybrid::step` 的装配一致。
fn attending_then_feeding(main: &[f64], cache: &Latent<f64>, query: &Query<'_, f64>) -> Vec<f64> {
    let mut expected = main.to_vec();
    let attended = attention()
        .attend(cache, query.nope, query.rope, 1, Path::Absorb)
        .expect("位置在缓存内");
    accumulate(&mut expected, &attended, 1.0);
    let fed = ffn().forward(&expected).expect("宽度一致");
    accumulate(&mut expected, &fed, 1.0);
    expected
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

    let expected = attending_then_feeding(&[1.0, 2.0, 3.0, 4.0], &cache, &query);
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

    let expected = attending_then_feeding(&mixed, &cache, &query);
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

/// 记忆层的几何：键维 2（门与查询宽）、值维 = 本栈宽度。
fn kda() -> crate::kda::Kda {
    crate::kda::Kda::new(2, WIDTH).expect("键维 2、值维 4")
}

/// 一栈：`groups` 组「三记忆 + 一混合」，流条数由 `streams` 定。
fn tower(groups: usize, streams: usize) -> Tower<f64> {
    let attention = (0..groups).map(|_| layer(ffn(), streams)).collect();
    let memory = (0..groups * MEMORY_PER_ATTENTION).map(|_| kda()).collect();
    Tower::new(attention, memory).expect("配比与宽度都合法")
}

/// 一份「门全开、写强度 1」的记忆输入：键与查询都取 `key`，值取 `value`。
fn full_write<'a>(key: &'a [f64], value: &'a [f64]) -> MemoryStep<'a, f64> {
    MemoryStep {
        gate: &[1.0; 2],
        write: 1.0,
        key,
        value,
        query: key,
    }
}

/// 写强度 0：只遗忘、不写。零记忆上于是什么都不发生。
fn quiet<'a>(key: &'a [f64], value: &'a [f64]) -> MemoryStep<'a, f64> {
    MemoryStep {
        write: 0.0,
        ..full_write(key, value)
    }
}

/// 层序是「三记忆 + 一混合」重复：配比是构造条件，不是注释。
#[test]
fn the_order_is_three_memory_layers_then_one_mixed_layer() {
    let tower = tower(2, 1);
    assert_eq!(tower.depth(), 2 * (MEMORY_PER_ATTENTION + 1));
    assert_eq!(
        tower.order().collect::<Vec<_>>(),
        vec![
            Kind::Memory,
            Kind::Memory,
            Kind::Memory,
            Kind::Attend,
            Kind::Memory,
            Kind::Memory,
            Kind::Memory,
            Kind::Attend,
        ]
    );
    assert_eq!(tower.state().layers(), 2 * MEMORY_PER_ATTENTION);
    assert!(tower.state().is_pristine());
}

/// 零记忆 + 写强度 0 时，整栈与单独跑混合层逐位相同：记忆层的存在不改主路。
#[test]
fn a_pristine_memory_with_no_writes_leaves_the_mixed_layer_untouched() {
    let tower = tower(1, 1);
    let cache = cache();
    let query = query();
    let key = [0.0, 1.0];
    let value = [0.0; WIDTH];
    let steps: Vec<MemoryStep<'_, f64>> = (0..MEMORY_PER_ATTENTION)
        .map(|_| quiet(&key, &value))
        .collect();

    let mut state = tower.state();
    let mut streams = vec![vec![1.0, 2.0, 3.0, 4.0]];
    tower
        .step(
            &mut streams,
            &mut state,
            &cache,
            &query,
            &steps,
            Path::Absorb,
        )
        .expect("推进");

    let mut expected = vec![vec![1.0, 2.0, 3.0, 4.0]];
    layer(ffn(), 1)
        .step(&mut expected, &cache, &query, Path::Absorb)
        .expect("推进");
    assert_eq!(streams, expected, "记忆层没写就不该留下任何痕迹");
    assert!(state.is_pristine(), "写强度 0 只遗忘零记忆，仍然是零");
}

/// 写进去立刻读得出来，且记忆跨格保留：写的一栈从第一格起就与只遗忘的一栈分叉。
#[test]
fn a_memory_written_once_is_read_back_on_the_next_step() {
    let tower = tower(1, 1);
    let cache = cache();
    let query = query();
    let key = [0.0, 1.0];
    let value = [0.25, -0.5, 1.0, 0.75];
    let writes: Vec<MemoryStep<'_, f64>> = (0..MEMORY_PER_ATTENTION)
        .map(|_| full_write(&key, &value))
        .collect();
    let quiet: Vec<MemoryStep<'_, f64>> = (0..MEMORY_PER_ATTENTION)
        .map(|_| quiet(&key, &value))
        .collect();

    let mut written = tower.state();
    let mut first = vec![vec![0.0; WIDTH]];
    tower
        .step(
            &mut first,
            &mut written,
            &cache,
            &query,
            &writes,
            Path::Absorb,
        )
        .expect("推进");
    assert!(!written.is_pristine(), "写强度 1 必须留下记忆");

    let mut pristine = tower.state();
    let mut blank = vec![vec![0.0; WIDTH]];
    tower
        .step(
            &mut blank,
            &mut pristine,
            &cache,
            &query,
            &quiet,
            Path::Absorb,
        )
        .expect("推进");
    assert!(pristine.is_pristine(), "门全开、写强度 0 不改零记忆");
    assert_ne!(first, blank, "同一格写进去、立刻读得出来，记忆当步就进主路");

    // 第二格不再写：记忆仍在，读数继续把两栈拉开。
    let mut continued = first.clone();
    tower
        .step(
            &mut continued,
            &mut written,
            &cache,
            &query,
            &quiet,
            Path::Absorb,
        )
        .expect("推进");
    let mut never = blank.clone();
    tower
        .step(
            &mut never,
            &mut pristine,
            &cache,
            &query,
            &quiet,
            Path::Absorb,
        )
        .expect("推进");
    assert_ne!(continued, never, "记忆跨格保留，第二格仍读得出来");
}

/// 每个记忆层各持一份状态：三层写不同的值维，各自只留下自己那一行。
#[test]
fn each_memory_layer_owns_its_state() {
    let tower = tower(1, 1);
    let cache = cache();
    let key = [0.0, 1.0];
    let values = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
    ];
    let steps: Vec<MemoryStep<'_, f64>> =
        values.iter().map(|value| full_write(&key, value)).collect();

    let mut state = tower.state();
    let mut streams = vec![vec![0.0; WIDTH]];
    tower
        .step(
            &mut streams,
            &mut state,
            &cache,
            &query(),
            &steps,
            Path::Absorb,
        )
        .expect("推进");

    // 记忆是 `[值维, 键维]`：写 `v·e_i` 且 `β = 1`、键是单位向量时，第 `i` 行就是 `v_i · k̂`。
    for (layer, value) in values.iter().enumerate() {
        let memory = state.memory(layer).expect("记忆层存在");
        let written = value.iter().position(|&v| v != 0.0).expect("只有一行非零");
        assert_eq!(
            memory.row(written).expect("行在值维内"),
            &[0.0, value[written]],
            "第 {layer} 层只该记住自己写入的那一行"
        );
        for other in (0..WIDTH).filter(|&row| row != written) {
            assert_eq!(memory.row(other).expect("行在值维内"), &[0.0, 0.0]);
        }
    }
}

/// 层栈只读缓存：`seq` 与缓存元素都不动。
#[test]
fn the_tower_never_touches_the_cache() {
    let tower = tower(2, 2);
    let cache = cache();
    let before = (cache.seq(), cache.cache_elements());
    let key = [0.0, 1.0];
    let value = [1.0; WIDTH];
    let steps: Vec<MemoryStep<'_, f64>> = (0..2 * MEMORY_PER_ATTENTION)
        .map(|_| full_write(&key, &value))
        .collect();

    let mut state = tower.state();
    let mut streams = tower.seed(&[1.0, 2.0, 3.0, 4.0]).expect("宽度合法");
    tower
        .step(
            &mut streams,
            &mut state,
            &cache,
            &query(),
            &steps,
            Path::Absorb,
        )
        .expect("推进");
    assert_eq!((cache.seq(), cache.cache_elements()), before);
}

/// 跑一次必定失败的一步：流与全部记忆都必须原样退回（后写的层不能留、前几层写过的也不能留）。
fn failed_step_rolls_back(
    tower: &Tower<f64>,
    cache: &Latent<f64>,
    steps: &[MemoryStep<'_, f64>],
    why: &str,
) {
    let mut state = tower.state();
    let mut streams = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let original = streams.clone();
    let err = tower
        .step(
            &mut streams,
            &mut state,
            cache,
            &query(),
            steps,
            Path::Absorb,
        )
        .expect_err(why);
    assert_eq!(err.code(), Code::Invalid);
    assert_eq!(streams, original, "失败之后流必须原样退回");
    assert!(state.is_pristine(), "前几层写过的记忆也不能留下");
}

/// 某一格失败时，流与全部记忆都原样退回：前几层写过的也不留。
#[test]
fn a_bad_memory_input_rolls_the_whole_layer_stack_back() {
    let tower = tower(1, 1);
    let cache = cache();
    let key = [0.0, 1.0];
    let value = [1.0; WIDTH];
    let mut steps: Vec<MemoryStep<'_, f64>> = (0..MEMORY_PER_ATTENTION)
        .map(|_| full_write(&key, &value))
        .collect();
    steps[2] = MemoryStep {
        gate: &[1.0, 1.0, 1.0],
        ..steps[2]
    };

    failed_step_rolls_back(&tower, &cache, &steps, "门宽 3 不等于键维 2");
}

/// 记忆输入的份数必须等于记忆层个数：少一份是错位，多一份是无主数据。
#[test]
fn the_number_of_memory_inputs_must_match_the_memory_layers() {
    let tower = tower(1, 1);
    let cache = cache();
    let key = [0.0, 1.0];
    let value = [1.0; WIDTH];
    for count in [MEMORY_PER_ATTENTION - 1, MEMORY_PER_ATTENTION + 1] {
        let steps: Vec<MemoryStep<'_, f64>> =
            (0..count).map(|_| full_write(&key, &value)).collect();
        failed_step_rolls_back(&tower, &cache, &steps, "份数不对不该通过");
    }
}

/// 流几何在入口判定：条数与宽度都必须正好是本栈的。
#[test]
fn stream_geometry_is_checked_before_the_first_memory_write() {
    let tower = tower(1, 2);
    let cache = cache();
    let key = [0.0, 1.0];
    let value = [1.0; WIDTH];
    let steps: Vec<MemoryStep<'_, f64>> = (0..MEMORY_PER_ATTENTION)
        .map(|_| full_write(&key, &value))
        .collect();
    for case in [
        vec![vec![1.0; WIDTH]],
        vec![vec![1.0; WIDTH], vec![2.0; WIDTH - 1]],
    ] {
        let mut state = tower.state();
        let mut streams = case.clone();
        let err = tower
            .step(
                &mut streams,
                &mut state,
                &cache,
                &query(),
                &steps,
                Path::Absorb,
            )
            .expect_err("几何不对不该通过");
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(streams, case);
        assert!(state.is_pristine());
    }
}

/// 配比与几何在装配期拒绝：不是 3:1、没有混合层、读出宽不齐、混合层之间几何不同源。
#[test]
fn assembly_enforces_the_recipe_and_the_geometry() {
    let short = Tower::new(vec![layer(ffn(), 1)], vec![kda(), kda()]);
    assert_eq!(short.expect_err("少一个记忆层").code(), Code::Invalid);
    assert_eq!(
        Tower::<f64>::new(Vec::new(), Vec::new())
            .expect_err("没有混合层")
            .code(),
        Code::Invalid
    );
    let narrow = crate::kda::Kda::new(2, WIDTH - 1).expect("键维 2、值维 3");
    assert_eq!(
        Tower::new(vec![layer(ffn(), 1)], vec![kda(), kda(), narrow])
            .expect_err("读出宽不齐")
            .code(),
        Code::Invalid
    );
    let memory: Vec<_> = (0..2 * MEMORY_PER_ATTENTION).map(|_| kda()).collect();
    assert_eq!(
        Tower::new(vec![layer(ffn(), 1), layer(ffn(), 2)], memory)
            .expect_err("流条数不同源")
            .code(),
        Code::Invalid
    );
}
