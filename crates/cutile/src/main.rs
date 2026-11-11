//! `cutile` 命令行入口：`doctor` / `verify` / `bench`（DESIGN §16）。

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(cutile::cli::main(&args));
}
