# fuzz 语料目录

- `<target>/seed.bin`：由 `tests/gen_corpus.rs` 从库编码生成的合法范例，不手抄、不入库；
  CI fuzz job 先跑生成器，再把语料目录挂给 libFuzzer 做覆盖引导。
- `<target>/edge-*.bin`：与解析器无关的边界输入（空串、零字节、上界字节），同样由生成器
  产出，让覆盖引导从最短长度与极值字节起步。
- `dict/<target>.dict`：libFuzzer 关键字字典，入库维护；词条与对应解析器的文法保持一致。
  字典只认双引号 C 字面量、不支持注释行；新增子命令/选项或字段时同步这里。

本地重新生成种子：

```sh
cargo test -p llm-fuzz --test gen_corpus -- crates/fuzz/corpus
```
