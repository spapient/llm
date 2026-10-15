// The Metal backend links CoreGraphics explicitly: `MTLCreateSystemDefaultDevice`
// requires it (objc2-metal's documented requirement), but the `objc2-core-graphics`
// dependency contributes no symbol this crate uses, so its `#[link]` is dead code
// and is dropped from the final binary. Without this, the device probe silently
// returns `None` on a machine that has a GPU.
#![allow(missing_docs)]

fn main() {
    // The crate only compiles its real body on macOS; on other targets it is an
    // empty library, where an extra framework link is harmless. Emitting the link
    // unconditionally keeps every target (tests, examples, benches) consistent.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
}
