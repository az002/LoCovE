//! Build provenance for the ISOLATED submission crate.
//!
//! Same construction as the shared `rustcore/build.rs` -- FNV-1a over every `src/*.rs`, the
//! lock and the crate version, stamped into the binary -- but under its OWN environment
//! variable, so `subcore.build_id()` can never be confused with the installed historical
//! `rustcore.build_id()`. GenomeCoords is deliberately not a git repository, so without this
//! nothing outside the source tree can say which code produced a recorded number.
//!
//! No hashing dependency on purpose: a provenance stamp that can fail to build offline is
//! worse than a simple one.
use std::fs;

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(PRIME);
    }
}

fn main() {
    let mut h = OFFSET;
    fnv(&mut h, b"method1_rust_cli");
    fnv(&mut h, env!("CARGO_PKG_VERSION").as_bytes());
    // Historical ancestor of the frozen release source, not the ID of this build.
    // Keep the research output field while allowing a standalone Cargo build.
    let lineage = std::env::var("METHOD1_RUSTCORE_LINEAGE")
        .unwrap_or_else(|_| "617c2b841785fda2".to_string());
    assert!(lineage.len() == 16 && lineage.chars().all(|c| c.is_ascii_hexdigit()), "{}", lineage);
    fnv(&mut h, lineage.as_bytes());
    // Feature choices alter the public interface and must distinguish build IDs.
    fnv(&mut h, if std::env::var_os("CARGO_FEATURE_EXPERIMENTAL_EM").is_some() {
        b"experimental-em" as &[u8]
    } else { b"coverage" as &[u8] });
    println!("cargo:rerun-if-env-changed=METHOD1_RUSTCORE_LINEAGE");
    println!("cargo:rustc-env=METHOD1_RUSTCORE_LINEAGE={}", lineage);
    let mut files: Vec<_> = fs::read_dir("src")
        .expect("src/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x == "rs"))
        .collect();
    files.sort();
    files.push("Cargo.lock".into());
    files.push("Cargo.toml".into());
    // 2026-10-02: the build script itself is part of the provenance (it defines the stamp)
    files.push("build.rs".into());
    for p in &files {
        println!("cargo:rerun-if-changed={}", p.display());
        fnv(&mut h, p.file_name().unwrap().to_string_lossy().as_bytes());
        let b = fs::read(p).expect("build provenance input must be readable");
        fnv(&mut h, &b);
    }
    println!("cargo:rustc-env=METHOD1_CLI_BUILD_ID={:016x}", h);
}
