llm/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── rustfmt.toml
├── clippy.toml
├── .gitignore
├── LICENSE
├── README.md
├── AGENTS.md
│
├── crates/
│   ├── lib/                         # 全 workspace 基础类型
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs                 # 公共配置与验证
│   │       ├── error.rs                  # 基础错误类型
│   │       ├── ids.rs                    # TokenId、RequestId 等
│   │       ├── limits.rs                 # 长度、容量和资源限制
│   │       ├── precision.rs               # 精度类型
│   │       └── seed.rs                   # 可复现随机种子
│   │
│   ├── tensor/                       # 张量元数据和存储抽象
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dtype.rs
│   │       ├── layout.rs
│   │       ├── shape.rs
│   │       ├── storage.rs
│   │       ├── tensor.rs
│   │       ├── view.rs
│   │       └── error.rs
│   │
│   ├── api/                  # 计算设备和算子契约
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── backend.rs
│   │       ├── device.rs
│   │       ├── kernel.rs
│   │       └── capabilities.rs
│   │
│   ├── cpu/                  # CPU 算子实现
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── backend.rs
│   │       ├── executor.rs
│   │       ├── matmul.rs
│   │       ├── attention.rs
│   │       ├── softmax.rs
│   │       ├── normalization.rs
│   │       ├── activation.rs
│   │       ├── embedding.rs
│   │       ├── quantization.rs
│   │       └── simd.rs                   # 仅在经过验证后启用
│   │
│   ├── autograd/                     # 自动微分
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── graph.rs
│   │       ├── node.rs
│   │       ├── variable.rs
│   │       ├── backward.rs
│   │       ├── gradient.rs
│   │       ├── no_grad.rs
│   │       └── ops/
│   │           ├── mod.rs
│   │           ├── matmul.rs
│   │           ├── softmax.rs
│   │           ├── normalization.rs
│   │           └── activation.rs
│   │
│   ├── nn/                           # 神经网络基础层和参数
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── module.rs
│   │       ├── parameter.rs
│   │       ├── statedict.rs
│   │       ├── init.rs
│   │       └── layers/
│   │           ├── mod.rs
│   │           ├── linear.rs
│   │           ├── embedding.rs
│   │           ├── rms_norm.rs
│   │           ├── rotary.rs
│   │           ├── attention.rs
│   │           └── mlp.rs
│   │
│   ├── models/                       # 具体模型结构
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── causallm.rs              # 自回归模型统一接口
│   │       ├── registry.rs               # 支持的模型架构注册
│   │       └── llama/
│   │           ├── mod.rs
│   │           ├── config.rs
│   │           ├── model.rs
│   │           ├── block.rs
│   │           └── weights.rs
│   │
│   ├── tokenizer/                    # Tokenizer 和对话模板
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tokenizer.rs
│   │       ├── vocabulary.rs
│   │       ├── encode.rs
│   │       ├── decode.rs
│   │       ├── bytepair.rs
│   │       ├── specialtokens.rs
│   │       └── chattemplate.rs
│   │
│   ├── data/                         # 训练数据输入与批处理
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dataset.rs
│   │       ├── record.rs
│   │       ├── textdataset.rs
│   │       ├── packing.rs
│   │       ├── batch.rs
│   │       ├── sampler.rs
│   │       └── loader.rs
│   │
│   ├── format/                       # 外部文件格式
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── modelconfig.rs
│   │       ├── safetensors.rs
│   │       ├── tensormetadata.rs
│   │       └── validation.rs
│   │
│   ├── checkpoint/                   # 权重和训练状态持久化
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── load.rs
│   │       ├── save.rs
│   │       ├── manifest.rs
│   │       ├── trainingstate.rs
│   │       ├── atomicwrite.rs
│   │       └── compatibility.rs
│   │
│   ├── optim/                        # 优化器和学习率策略
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── optimizer.rs
│   │       ├── adamw.rs
│   │       ├── sgd.rs
│   │       ├── gradientclip.rs
│   │       ├── scheduler.rs
│   │       └── state.rs
│   │
│   ├── context/                      # 执行上下文与资源控制
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── execution.rs
│   │       ├── cancellation.rs
│   │       ├── memorybudget.rs
│   │       ├── worker.rs
│   │       └── metrics.rs
│   │
│   ├── train/                        # 训练循环与评估
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── trainer.rs
│   │       ├── trainconfig.rs
│   │       ├── loss.rs
│   │       ├── step.rs
│   │       ├── gradientaccumulation.rs
│   │       ├── evaluation.rs
│   │       └── resume.rs
│   │
│   ├── infer/                        # 自回归生成和 KV Cache
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs
│   │       ├── request.rs
│   │       ├── generation.rs
│   │       ├── prefill.rs
│   │       ├── decode.rs
│   │       ├── kvcache.rs
│   │       ├── sampling.rs
│   │       ├── stopcondition.rs
│   │       └── stream.rs
│   │
│   ├── cli/                          # 用户命令行入口
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── args.rs
│   │       ├── error.rs
│   │       └── command/
│   │           ├── mod.rs
│   │           ├── train.rs
│   │           ├── generate.rs
│   │           ├── inspect.rs
│   │           └── validate.rs
│   │
│   ├── serve/                        # 可选：本地 HTTP 服务
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── server.rs
│   │       ├── routes.rs
│   │       ├── request.rs
│   │       └── response.rs
│   │
│   ├── testkit/                      # 集成测试辅助，不进入生产依赖
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── reference.rs
│   │       ├── tinymodel.rs
│   │       ├── fixtures.rs
│   │       └── assertions.rs
│   │
│   └── bench/                        # 基准测试程序
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── tensor.rs
│           ├── model.rs
│           ├── training.rs
│           └── inference.rs
│
├── configs/
│   ├── train-cpu.toml
│   ├── infer-cpu.toml
│   └── serve-local.toml
│
├── examples/
│   ├── train_small.rs
│   ├── generate.rs
│   └── load_checkpoint.rs
│
├── tests/
│   ├── checkpoint_roundtrip.rs
│   ├── model_forward.rs
│   ├── training_resume.rs
│   ├── generation_greedy.rs
│   └── request_isolation.rs
│
├── testdata/
│   ├── tiny-model/
│   ├── tokenizer/
│   └── malformed-files/
│
├── benches/
│   ├── cpu_matmul.rs
│   ├── model_prefill.rs
│   └── model_decode.rs
│
├── scripts/
│   ├── check.sh
│   ├── test.sh
│   └── bench.sh
│
└── docs/
    ├── architecture.md
    ├── crate-boundaries.md
    ├── tensor-layout.md
    ├── model-support.md
    ├── checkpoint-format.md
    ├── training.md
    ├── inference.md
    └── performance.md