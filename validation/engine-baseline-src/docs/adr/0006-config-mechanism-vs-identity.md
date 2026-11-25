# 0006：配置的机制与身份分离

- 状态：部分被 [ADR 0020](0020-settings-layer-absorption.md) 取代（机制仍成立，身份只剩一个实现，类型参数已删除）
- 日期：2026-11-12

## 背景

`llm-config` 曾经把两件事写在一起：

```
机制：默认 < 用户级文件 < 环境变量 < --set，每个值带来源，路径是绝对路径
身份：FILE_NAME = "cutile.toml"、环境变量前缀 CUTILE_、用户级目录名 cutile
```

身份被写死的后果是可预期的：`llm` / `train` / `serve` 想读自己的配置时只有两条路——
寄生在 `CUTILE_*` 下（"为什么 llm 的设置叫 cutile"），或者把分层逻辑复制一遍
（"同一个值在不同模块取到不同结果"）。

## 决策

把身份提成类型参数：

```rust
pub trait App {
    const DIR: &'static str;         // 用户级目录：<cache home>/<DIR>/{cache,work}
    const FILE: &'static str;        // 配置文件：<config home>/<DIR>/<FILE>
    const ENV_PREFIX: &'static str;  // 环境变量：<PREFIX>_<后缀>
}

pub struct Settings<A: App = Cutile> { /* roots + sources + PhantomData */ }
```

- `Key` 只登记**后缀**（`CACHE_DIR` / `CACHE` / `WORK_DIR` / `DATA_DIR`），
  完整变量名由 `Key::env_vars::<A>()` 拼出；
- `Roots` 的 `default_cache_dir(app)` / `default_work_dir(app)` / `user_config_file(app, file)`
  只拼目录，不认识任何产品名；
- `Cutile` 是默认类型参数，`Settings::<Cutile>` 与旧行为逐字一致。

## 已知代价：调用点必须显式

关联函数路径**不套用**类型参数默认值，所以 `Settings::load()` 在泛型化之后不再成立，
必须写 `Settings::<Cutile>::load()`（或 `Settings::<Llm>::load()`）。这是刻意接受的代价：
显式身份比隐式默认更诚实，而且一旦某个调用点选错身份，编译器会立刻要求它写清楚。

## 不做的事

- **不做运行期 `App` 值**：那会把"身份"变成可以传错参数的东西，且每个方法都要多一个参数；
- **不复制键闭集**：`Key` 仍是唯一闭集，多个应用共享同一组键；给某个应用增加专属键
  （如 `model.dir`）必须同时给出消费方与测试，否则就是"转了不生效的旋钮"；
- **不在 `llm-config` 里出现领域语义**：解析器、分层、路径是机制；`ops/`、`devices/`、`prim/`
  是数据，归 `op` / `prof` / `prim`。

## 后果

- 一个测试里定义第二个应用（`demo.toml` + `DEMO_*`），证明它读不到 `CUTILE_*`，
  且默认目录互不干扰；
- 新增应用的成本是 3 个常量 + 一个标记类型，不是复制一层配置逻辑；
- `FILE_NAME` 常量保留但指向 `Cutile::FILE`，新增代码不应再引用它。
