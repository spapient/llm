use super::*;

/// 写一份两层稠密权重（`[out, in]` 行主序），取值落在 2 的负幂网格上：乘积与累加在 f32
/// 里都是精确的，因此手算的结果可以被逐位钉住。
fn write_dense_model(name: &str, layers: &[(&str, usize, usize)]) -> std::path::PathBuf {
    let mut entries: Vec<(String, Vec<f32>)> = Vec::new();
    for (entry, out, input) in layers {
        let values = (0..out * input)
            .map(|idx| ((idx * 5) % 17) as f32 * 0.25 - 1.0)
            .collect();
        entries.push((format!("{entry}{}", nn::WEIGHT_SUFFIX), values));
    }
    let mut json = String::from("{");
    let mut offset = 0usize;
    for (idx, (name, values)) in entries.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        let end = offset + values.len() * 4;
        let (_, out, input) = layers[idx];
        json.push_str(&format!(
                "\"{name}\": {{\"dtype\": \"F32\", \"shape\": [{out}, {input}], \"data_offsets\": [{offset}, {end}]}}"
            ));
        offset = end;
    }
    json.push('}');
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    for (_, values) in &entries {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let path = std::env::temp_dir().join(format!("{}-{name}", std::process::id()));
    std::fs::write(&path, &bytes).expect("临时文件可写");
    path
}

#[test]
fn a_real_weight_file_runs_through_generated_kernels() {
    if shard::toolchain::detect().is_none() {
        // 没有 C 编译器就没有执行证据；这条用例在这里不成立，不是通过。
        return;
    }
    let path = write_dense_model(
        "llm-run-test.safetensors",
        &[("blk.0.attn_q", 4, 8), ("blk.0.attn_o", 4, 4)],
    );
    let input = "-0.5,-0.375,-0.25,-0.125,0,0.125,0.25,0.375";
    run(&path.display().to_string(), Some(input), Some("cpu")).expect("两层链应当跑通");
}

#[test]
fn the_first_layer_computes_the_hand_arithmetic_on_its_own_bytes() {
    if shard::toolchain::detect().is_none() {
        return;
    }
    let path = write_dense_model("llm-run-arithmetic.safetensors", &[("blk.0.attn_q", 4, 8)]);
    let file = weights::safetensors::File::open(&path).expect("文件合法");
    let entry = file
        .header()
        .get(&format!("blk.0.attn_q{}", nn::WEIGHT_SUFFIX))
        .expect("条目存在");
    let w = weights::load_f32(entry, &file).expect("f32 条目");
    let x: Vec<f32> = (0..8).map(|idx| (idx % 8) as f32 * 0.125 - 0.5).collect();
    let model = nn::registry::model(nn::Linear::DENSE).expect("稠密语义");
    let sample = nn::model::bind(&model, 8, 4, &x, &w).expect("形状自洽");

    let settings = crate::settings::Settings::load()
        .map_err(|error| error.with("配置"))
        .expect("默认设置");
    let dirs = settings.dirs();
    let catalogue = cutile::builtin::Catalogue::load(&dirs).expect("内置数据");
    let executed = shard::execution::execute(&catalogue, &dirs, nn::Linear::DENSE, "cpu", &sample)
        .expect("执行");
    let want: Vec<f32> = (0..4)
        .map(|row| (0..8).map(|column| w[row * 8 + column] * x[column]).sum())
        .collect();
    assert_eq!(executed.output, want, "真实权重算出来的数必须等于手算");
    assert_eq!(executed.max_rel_err, 0.0, "同一份舍入顺序应当逐位相等");
}
