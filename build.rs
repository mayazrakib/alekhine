use std::env;
use std::process::Command;

fn main() {
    let compiler = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let compiler_output = Command::new(compiler).arg("--version").output().expect("Failed to identify the Rust compiler.");
    assert!(compiler_output.status.success());
    let compiler_identity = String::from_utf8(compiler_output.stdout).expect("Failed to decode the Rust compiler identity.");
    let revision = env::var("ALEKHINE_REVISION").unwrap_or_else(|_| "uncommitted".to_owned());
    println!(
        "cargo:rustc-env=ALEKHINE_COMPILER={}",
        compiler_identity.trim(),
    );
    println!("cargo:rustc-env=ALEKHINE_REVISION={revision}");
    println!("cargo:rerun-if-env-changed=ALEKHINE_REVISION");
    println!("cargo:rerun-if-changed=build.rs");
}
