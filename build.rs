/// build.rs — AOT Metal kernel compilation.
///
/// Compiles `src/kernels.metal` → `kernels.air` → `kernels.metallib` using
/// xcrun on macOS. The resulting `.metallib` path is embedded into the binary
/// via the `VELO_METALLIB_PATH` env var so the runtime can load it without
/// JIT compilation on first inference.
///
/// Falls back gracefully (emits a warning, skips compilation) when:
///   - The build host is not macOS.
///   - `xcrun` is not found (e.g., CI without Xcode CLI tools).
///   - The Metal source file is not present.

use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    // Always re-run if the shader source changes.
    println!("cargo:rerun-if-changed=src/kernels.metal");
    println!("cargo:rerun-if-env-changed=VELO_SKIP_METALLIB");

    // Allow CI or cross-compile environments to opt out.
    if env::var("VELO_SKIP_METALLIB").is_ok() {
        return;
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("cargo:warning=Skipping AOT Metal compilation: not macOS.");
        return;
    }

    #[cfg(target_os = "macos")]
    compile_metallib();
}

#[cfg(target_os = "macos")]
fn compile_metallib() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = PathBuf::from("src/kernels.metal");

    if !src.exists() {
        println!("cargo:warning=src/kernels.metal not found; skipping AOT Metal compilation.");
        return;
    }

    // Check xcrun is available without hard-failing.
    if !xcrun_available() {
        println!("cargo:warning=xcrun not found; falling back to runtime shader compilation.");
        return;
    }

    let air_path = out_dir.join("kernels.air");
    let metallib_path = out_dir.join("kernels.metallib");

    // Step 1: .metal → .air
    let status = Command::new("xcrun")
        .args([
            "-sdk", "macosx",
            "metal",
            "-c", src.to_str().unwrap(),
            "-o", air_path.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            println!("cargo:warning=Metal compiler (xcrun metal) failed; falling back to runtime JIT.");
            return;
        }
        Err(e) => {
            println!("cargo:warning=Failed to invoke xcrun metal: {}; falling back to runtime JIT.", e);
            return;
        }
    }

    // Step 2: .air → .metallib
    let status = Command::new("xcrun")
        .args([
            "-sdk", "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o", metallib_path.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            // Expose the path so the Rust runtime can load it via newLibraryWithURL.
            println!("cargo:rustc-env=VELO_METALLIB_PATH={}", metallib_path.display());
        }
        Ok(_) => {
            println!("cargo:warning=xcrun metallib packaging failed; falling back to runtime JIT.");
        }
        Err(e) => {
            println!("cargo:warning=Failed to invoke xcrun metallib: {}; falling back to runtime JIT.", e);
        }
    }
}

#[cfg(target_os = "macos")]
fn xcrun_available() -> bool {
    Command::new("xcrun")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
