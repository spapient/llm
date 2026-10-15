// The Metal backend links CoreGraphics explicitly: `MTLCreateSystemDefaultDevice`
// requires it (objc2-metal's documented requirement), but the `objc2-core-graphics`
// dependency contributes no symbol this crate uses, so its `#[link]` is dead code
// and is dropped from the final binary. Without this, the device probe silently
// returns `None` on a machine that has a GPU.
#![allow(missing_docs)]

fn main() {
    // The crate only compiles its real body on macOS; on other targets it is an
    // empty library and a framework link would be rejected by the linker.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }
}
