//! method1-coverage: standalone Rust coverage CLI candidate (2026-10-01 handoff).
//!
//!   method1-coverage run --index P --record-map M --out DIR [--threads 7] [--block B]
//!                        [--loader fast|reference] READS...
//!   method1-coverage profile-load --index P [--loader fast|reference]
//!   method1-coverage build --out INDEX [--sampler fracminhash|syncmer] [--k 21 --g 300 --c 20]
//!                          [--meta-json FILE] [--record-map-out FILE] FASTA...
//!   method1-coverage describe --index P [--loader fast|reference]
//!   method1-coverage exposures --index P --out EXPOSURES FASTA...
//!   run ... [--exposures EXPOSURES --em-attribution [--em-tolerance 1e-5|1e-9]]
//!   build ... [--exposures-out EXPOSURES]
//!
//! 2026-10-05: OPT-IN EM attribution (`em.rs`, `exposure.rs`): local depth attributed to each
//! segment's own genome by the EXISTING scaled-Poisson abundance EM, from a reference-only
//! exposure file bound to one index (and so to its sampler). Default outputs are unchanged and
//! nothing EM-related is loaded or computed unless `--em-attribution` is given.
//!
//! 2026-10-02: the coverage-anchor SAMPLER is an opt-in INDEX-BUILD choice (`sampler.rs`).
//! `build` writes the index (default fracminhash = the unchanged original format; syncmer = the
//! approved production nominal open-syncmer selection). `run`/`profile-load`/`describe` honour
//! the sampler stored in the index; `--sampler` there is an assertion that must match it.
//!
//! `run` = the complete workflow of the Python-driven pipeline (`minimal_timing.ours` with the
//! method1_keyfirst_zlibrs_core module): load the product index, stream and count the reads
//! (key-first, zlib-rs, `threads` counting workers + one pipelined reader; serial when 1),
//! reduce segments, all five depth variants and breadth, write segments.npz / record-map.json
//! / report.json, fsync them, then phases.json. No Python and no Python extension is loaded.
mod kernel;
mod em;
mod exposure;
mod load;
mod out;
mod product;
mod query;
mod reduce;
mod refindex;
mod sampler;
mod selection;
mod ser;

use reduce::{obj, J};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BUILD_ID: &str = env!("METHOD1_CLI_BUILD_ID");
const RUSTCORE_LINEAGE: &str = env!("METHOD1_RUSTCORE_LINEAGE");

struct Args {
    cmd: String,
    index: Option<String>,
    record_map: Option<String>,
    out: Option<String>,
    threads: usize,
    block: usize,
    loader: String,
    reads: Vec<String>,
    sampler: Option<String>,
    k: Option<usize>,
    g: Option<usize>,
    c: Option<usize>,
    meta_json: Option<String>,
    record_map_out: Option<String>,
    exposures: Option<String>,
    exposures_out: Option<String>,
    em_attribution: bool,
    em_tolerance: Option<f64>,
    given: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or("usage: method1-coverage run|profile-load|build|describe ...")?;
    let mut a = Args { cmd, index: None, record_map: None, out: None, threads: 7, block: 1 << 24,
                       loader: "fast".into(), reads: Vec::new(), sampler: None, k: None,
                       g: None, c: None, meta_json: None, record_map_out: None,
                       exposures: None, exposures_out: None, em_attribution: false,
                       em_tolerance: None, given: Vec::new() };
    let num = |x: &str, v: String| -> Result<usize, String> {
        v.parse().map_err(|e| format!("{}: {}", x, e))
    };
    while let Some(x) = it.next() {
        if x.starts_with("--") {
            if a.given.contains(&x) {
                return Err(format!("{} given twice", x));
            }
            a.given.push(x.clone());
        }
        let mut val = || it.next().ok_or(format!("{} needs a value", x));
        match x.as_str() {
            "--sampler" => a.sampler = Some(val()?),
            "--k" => a.k = Some(num("--k", val()?)?),
            "--g" => a.g = Some(num("--g", val()?)?),
            "--c" => a.c = Some(num("--c", val()?)?),
            "--meta-json" => a.meta_json = Some(val()?),
            "--record-map-out" => a.record_map_out = Some(val()?),
            "--exposures" => a.exposures = Some(val()?),
            "--exposures-out" => a.exposures_out = Some(val()?),
            "--em-attribution" => a.em_attribution = true,
            "--em-tolerance" => a.em_tolerance = Some(val()?.parse::<f64>()
                .map_err(|e| format!("--em-tolerance: {}", e))?),
            "--index" => a.index = Some(val()?),
            "--record-map" => a.record_map = Some(val()?),
            "--out" => a.out = Some(val()?),
            "--threads" => a.threads = val()?.parse().map_err(|e| format!("--threads: {}", e))?,
            "--block" => a.block = val()?.parse().map_err(|e| format!("--block: {}", e))?,
            "--loader" => a.loader = val()?,
            s if s.starts_with("--") => return Err(format!("unknown option {}", s)),
            _ => a.reads.push(x),
        }
    }
    if a.threads == 0 || a.threads > 1024 {
        return Err(format!("threads={} is outside 1..=1024", a.threads));
    }
    // Release packaging only: keep legacy attribution separate from the coverage CLI.
    // This is not the optimized abundance workflow used in the latest manuscript timings.
    if !cfg!(feature = "experimental-em") && (a.cmd == "exposures"
        || a.exposures.is_some() || a.exposures_out.is_some()
        || a.em_attribution || a.em_tolerance.is_some()) {
        return Err("legacy EM commands require the experimental-em Cargo feature".into());
    }
    // each command accepts only its own options: an option silently ignored by a command
    // (e.g. --c on a query, which cannot change an index) is refused
    let allowed: &[&str] = match a.cmd.as_str() {
        "run" => &["--index", "--record-map", "--out", "--threads", "--block", "--loader",
                   "--sampler", "--exposures", "--em-attribution", "--em-tolerance"],
        "profile-load" | "describe" => &["--index", "--loader", "--sampler"],
        "build" => &["--out", "--sampler", "--k", "--g", "--c", "--meta-json",
                     "--record-map-out", "--exposures-out"],
        "exposures" => &["--index", "--out", "--loader"],
        _ => &[],
    };
    for x in &a.given {
        if !allowed.contains(&x.as_str()) {
            return Err(format!("{} is not an option of `{}`", x, a.cmd));
        }
    }
    if matches!(a.cmd.as_str(), "profile-load" | "describe") && !a.reads.is_empty() {
        return Err(format!("`{}` takes no positional arguments", a.cmd));
    }
    if let Some(s) = &a.sampler {
        sampler::mode_of(s)?;
    }
    // EM options travel together: an exposure file nobody uses, or a tolerance without the EM,
    // is refused rather than ignored
    if a.em_attribution && a.exposures.is_none() {
        return Err("--em-attribution needs --exposures (the index's reference-only exposure \
file; build it with `method1-coverage exposures` or `build --exposures-out`)".into());
    }
    if !a.em_attribution && (a.exposures.is_some() || a.em_tolerance.is_some()) {
        return Err("--exposures/--em-tolerance are only used with --em-attribution".into());
    }
    if let Some(t) = a.em_tolerance {
        if !em::TOLERANCES.contains(&t) {
            return Err(format!("--em-tolerance {} is not an approved value (1e-5, or the 1e-9 \
development comparison)", t));
        }
    }
    Ok(a)
}

fn abspath(p: &str) -> Result<String, String> {
    std::path::absolute(p).map(|x| x.to_string_lossy().into_owned())
        .map_err(|e| format!("{}: {}", p, e))
}

fn load(a: &Args) -> Result<(product::ProductIndex, product::LoadReport, load::LoadPhases), String> {
    let p = abspath(a.index.as_deref().ok_or("--index is required")?)?;
    let r = match a.loader.as_str() {
        "fast" => load::load_fast(&p, 8),
        "reference" => load::load_reference(&p),
        x => Err(format!("unknown loader {}", x)),
    }?;
    sampler::check_expected(&r.0, a.sampler.as_deref())?;
    Ok(r)
}

fn phases_json(ph: &load::LoadPhases) -> J {
    obj(vec![("loader", J::Str(ph.loader.clone())), ("read", J::Float(ph.read)),
             ("checksum_thread", J::Float(ph.checksum_thread)),
             ("parse_and_decode_concurrent", J::Float(ph.parse_and_decode)),
             ("checksum_wait_after_decode", J::Float(ph.checksum_wait)),
             ("validate", J::Float(ph.validate)), ("lookup_init", J::Float(ph.lookup_init)),
             ("total", J::Float(ph.total)), ("bytes", J::Int(ph.bytes as i64))])
}

fn status_field(name: &str) -> i64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines()
        .find(|l| l.starts_with(name))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))).unwrap_or(-1)
}
fn vmhwm_kib() -> i64 {
    status_field("VmHWM:")
}
fn os_threads() -> i64 {
    status_field("Threads:")
}

fn run(a: &Args, entry: Instant, pool: &rayon::ThreadPool) -> Result<(), String> {
    let out = PathBuf::from(a.out.as_deref().ok_or("--out is required")?);
    let rmap = a.record_map.as_deref().ok_or("--record-map is required")?;
    if !out.is_dir() {
        return Err(format!("{}: output directory does not exist", out.display()));
    }
    let reads: Vec<String> = a.reads.iter().map(|r| abspath(r)).collect::<Result<_, _>>()?;
    let mut sec: Vec<(&str, J)> = Vec::new();
    let mut thr: Vec<(&str, J)> = vec![("at_start", J::Int(os_threads()))];
    let t = Instant::now();
    let (idx, rep, lph) = pool.install(|| load(a))?;
    sec.push(("index_load", J::Float(t.elapsed().as_secs_f64())));
    let exposures = match &a.exposures {
        Some(xp) => {
            let t = Instant::now();
            let xp = abspath(xp)?;
            let x = exposure::Exposures::load(&xp, &idx, rep.checksum)?;
            sec.push(("em_exposure_load", J::Float(t.elapsed().as_secs_f64())));
            Some((xp, x))
        }
        None => None,
    };
    thr.push(("after_load", J::Int(os_threads())));
    let t = Instant::now();
    let acc = if a.threads <= 1 {
        query::query_serial(&idx, &reads)?
    } else {
        query::query_parallel_in(pool, &idx, &reads, a.threads, a.block)?
    };
    thr.push(("after_query", J::Int(os_threads())));
    let qsecs = t.elapsed().as_secs_f64();
    sec.push(("query_rust_counting", J::Float(qsecs)));
    let t = Instant::now();
    let sm = reduce::sample(&acc, idx.k)?;
    let red = pool.install(|| reduce::reduce(&idx.anchor_seg, idx.st.len(), &acc, &sm))?;
    sec.push(("reduction", J::Float(t.elapsed().as_secs_f64())));
    let attribution = match &exposures {
        Some((_, x)) => {
            let t = Instant::now();
            let att = em::attribute(&idx, x, &acc.raw, &sm,
                                    a.em_tolerance.unwrap_or(em::TOLERANCES[0]))?;
            sec.push(("em_fit_and_attribution", J::Float(t.elapsed().as_secs_f64())));
            Some(att)
        }
        None => None,
    };
    drop(acc);
    let t = Instant::now();
    let mut members: Vec<(&str, out::Arr)> = vec![
        ("seg_start", out::Arr::I64(&idx.st)), ("seg_end", out::Arr::I64(&idx.en)),
        ("seg_record", out::Arr::I64(&idx.seg_rec)), ("n_anchors", out::Arr::I64(&red.n_anchors)),
        ("breadth_anchor", out::Arr::F64(&red.breadth))];
    let names: Vec<String> = red.depth.iter().map(|(n, _)| format!("depth_{}", n)).collect();
    for (i, (_, d)) in red.depth.iter().enumerate() {
        members.push((names[i].as_str(), out::Arr::F64(d)));
    }
    out::write_npz(&out.join("segments.npz"), &members)?;
    std::fs::copy(rmap, out.join("record-map.json")).map_err(|e| format!("{}: {}", rmap, e))?;
    let engine = format!("method1-coverage standalone Rust CLI {} (method1_keyfirst_zlibrs_core \
counting: {} counting threads + {} pipelined reader thread; Rust reduction and output; \
coverage-anchor sampler {} from the index)",
                         BUILD_ID, a.threads, if a.threads <= 1 { 0 } else { 1 },
                         sampler::cli_name(idx.sel.mode));
    let mut report_items: Vec<(&str, J)> = vec![
        ("rustcore_build_id", J::Null),
        ("rustcore_build_id_lineage", obj(vec![
            ("build_id", J::Str(RUSTCORE_LINEAGE.into())),
            ("kind", J::Str("build-time lineage, embedded; NOT a runtime import (no Python or \
rustcore is loaded by this binary) and NOT this binary's identity (see method1_cli_build_id)"
                .into())),
            ("verified_by", J::Str("build_spanfill.py: rustcore.build_id() of the build \
environment, checked against the archived subcore build record's installed_rustcore_build_id \
and installed-module hashes".into()))])),
        ("method1_cli_build_id", J::Str(BUILD_ID.into())),
        ("parameters", obj(vec![("k", J::Int(idx.k as i64)), ("g", J::Int(idx.g as i64)),
                                ("c", J::Int(idx.c as i64))])),
        ("coordinates", J::Str("zero-based half-open, per record".into())),
        ("breadth", J::Str("observed anchor fraction; not converted base breadth".into())),
        ("undefined", J::Str("NaN where no eligible anchors; zero remains zero".into())),
        ("depth_info", red.depth_info.clone()),
        ("sufficient_statistics", sm.sufficient.clone()),
        ("query_seconds_inside_module", J::Float(qsecs)),
        ("coverage_selection", sampler::describe(&idx)),
        ("engine", J::Str(engine)),
        ("output", J::Str("uncompressed NPZ, no per-anchor intermediates; all five depth \
variants".into())),
    ];
    let mut em_files: Vec<&str> = Vec::new();
    if let (Some(att), Some((xp, x))) = (&attribution, &exposures) {
        em_files = write_attribution(&out, att, x, xp, &idx, &red, &sm)?;
        report_items.push(("em_attribution", obj(vec![
            ("files", J::List(em_files.iter().map(|f| J::Str(f.to_string())).collect())),
            ("kind", J::Str("OPT-IN source-attributed local depth from the existing \
scaled-Poisson abundance EM; model-dependent; the pooled outputs above are unchanged".into())),
            ("converged", J::Bool(att.fit.converged)),
            ("iterations", J::Int(att.fit.iterations as i64))])));
    }
    let report = obj(report_items);
    out::write_json(&out.join("report.json"), &report)?;
    for f in ["segments.npz", "record-map.json", "report.json"].iter().chain(em_files.iter()) {
        out::fsync(&out.join(f))?;
    }
    sec.push(("output_write_and_fsync", J::Float(t.elapsed().as_secs_f64())));
    sec.push(("total_in_process", J::Float(entry.elapsed().as_secs_f64())));
    let ph = obj(vec![
        ("seconds", obj(sec)), ("index_load_detail", phases_json(&lph)),
        ("index_bytes", J::Int(rep.bytes as i64)), ("index_checksum", J::Str(format!("{:016x}", rep.checksum))),
        ("map_slots", J::Int(rep.map_slots as i64)),
        ("reduction_parallel_segments", J::Bool(red.parallel_segments)),
        ("peak_rss_kib_vmhwm", J::Int(vmhwm_kib())), ("build_id", J::Str(BUILD_ID.into())),
        ("threads", J::Int(a.threads as i64)), ("block", J::Int(a.block as i64)),
        ("os_threads", obj(thr)),
        ("thread_policy", J::Str(format!("one rayon pool of {} workers (load decode/read tasks, \
counting, reduction) + 1 pipelined FASTQ reader thread during counting + 1 checksum thread \
during load only + the coordinating main thread", a.threads))),
        ("rustcore_build_id_lineage", J::Str(RUSTCORE_LINEAGE.into())),
        ("coverage_sampler", J::Str(selection::mode_name(idx.sel.mode).into()))]);
    out::write_json(&out.join("phases.json"), &ph)?;
    Ok(())
}

/// The opt-in attribution files (written only with --em-attribution).
fn write_attribution(out: &Path, att: &em::Attribution, x: &exposure::Exposures, xp: &str,
                     idx: &product::ProductIndex, red: &reduce::Reduced, sm: &reduce::Sample)
    -> Result<Vec<&'static str>, String> {
    let mut m: Vec<(&str, out::Arr)> = vec![
        ("own_genome", out::Arr::I64(&att.own_genome)), ("own_group", out::Arr::I64(&att.own_group)),
        ("own_group_size", out::Arr::I64(&att.own_group_size)),
        ("own_group_identifiable", out::Arr::I64(&att.own_group_identifiable)),
        ("own_exposure", out::Arr::I64(&att.own_exposure)),
        ("observed_support", out::Arr::I64(&att.observed_support)),
        ("modeled_support", out::Arr::F64(&att.modeled_support)),
        ("own_responsibility", out::Arr::F64(&att.own_responsibility)),
        ("own_attributed_support", out::Arr::F64(&att.own_attributed_support)),
        ("nonidentifiable_support", out::Arr::F64(&att.nonidentifiable_support))];
    for (n, d) in &att.attributed {
        m.push((n.as_str(), out::Arr::F64(d)));
    }
    out::write_npz(&out.join("em-segments.npz"), &m)?;
    out::write_npz(&out.join("em-attribution-sparse.npz"), &[
        ("row_offsets", out::Arr::I64(&att.sp_off)), ("group", out::Arr::I64(&att.sp_group)),
        ("exposure", out::Arr::I64(&att.sp_exposure)),
        ("expected", out::Arr::F64(&att.sp_expected)),
        ("responsibility", out::Arr::F64(&att.sp_resp)),
        ("support", out::Arr::F64(&att.sp_support))])?;
    let fj = |v: f64| if v.is_finite() { J::Float(v) } else { J::Null };
    let fl = |v: &[f64]| J::List(v.iter().map(|&x| fj(x)).collect());
    let xbytes = std::fs::read(xp).map_err(|e| format!("{}: {}", xp, e))?;
    let xck = u64::from_le_bytes(xbytes[xbytes.len() - 8..].try_into().unwrap());
    let pooled: Vec<(String, Vec<f64>)> = red.depth.iter()
        .map(|(n, d)| (format!("depth_{}", n), d.clone())).collect();
    let fit = obj(vec![
        ("model", J::Str("scaled_poisson_em: native port of benchmarks/typing/models.py (fit, \
arm scaled_poisson_em); private-key initialization floored at 1e-6; absolute max-delta \
stopping; identical rows compressed for the fit, attribution on the original segment rows".into())),
        ("tolerance", J::Float(att.tolerance)), ("maximum_iterations", J::Int(em::MAX_ITER as i64)),
        ("diagnostics", obj(vec![
            ("iterations", J::Int(att.fit.iterations as i64)),
            ("converged", J::Bool(att.fit.converged)),
            ("final_max_delta", fj(att.fit.final_max_delta)),
            ("tolerance", J::Float(att.tolerance)),
            ("maximum_iterations", J::Int(em::MAX_ITER as i64)),
            ("compressed_rows", J::Int(att.fit.compressed_rows as i64)),
            ("original_rows", J::Int(att.fit.original_rows as i64))])),
        ("groups", J::List(att.st.groups.iter().map(|g| J::List(g.iter()
            .map(|&s| J::Int(s as i64)).collect())).collect())),
        ("group_identifiable", J::List(att.st.identifiable.iter().map(|&b| J::Bool(b)).collect())),
        ("rank", J::Int(att.st.rank as i64)),
        ("rank_method", J::Str("exact rational rank and identifiability, PROVEN: lower bound \
from a nonzero Gram minor mod p, upper bound from integer null vectors verified A w = 0 in exact \
arithmetic on every original row (identifiable = touched by no null vector); refused without a \
complete certificate. numpy's SVD-tolerance rank can differ on near-singular matrices".into())),
        ("rank_certificate", obj(vec![
            ("primes_used", J::List(att.st.certificate.primes_used.iter()
                .map(|&p| J::Str(p.to_string())).collect())),
            ("verified_null_vectors", J::List(att.st.certificate.null_vectors.iter()
                .map(|w| J::List(w.iter().map(|&x| J::Str(x.to_string())).collect())).collect())),
            ("nullity", J::Int(att.st.certificate.null_vectors.len() as i64))])),
        ("group_theta_raw", fl(&att.group_theta)),
        ("genome_theta_raw", fl(&att.genome_theta)),
        ("group_theta_raw_unmasked", fl(&att.fit.theta)),
        ("private_initial_raw", fl(&att.initial)),
        ("units", J::Str("theta and responsibilities from RAW anchor-occurrence counts; theta \
is expected occurrences per reference key copy (raw-mean units); variant units divide by the \
same sample scalars as the pooled variants".into())),
        ("attributed_variants", att.attributed_info.clone()),
        ("genomes", em::genome_summaries(idx, att, &pooled, sm, &idx.genome_names)),
        ("exposures", obj(vec![("path", J::Str(xp.to_string())),
                               ("checksum", J::Str(format!("{:016x}", xck))),
                               ("index_checksum", J::Str(format!("{:016x}", x.index_checksum))),
                               ("sampler", J::Str(selection::mode_name(idx.sel.mode).into()))])),
        ("semantics", obj(vec![
            ("responsibility", J::Str("r_jG = A_jG theta_G / lambda_j on each ORIGINAL segment \
row (a fraction); NaN for non-identifiable groups and where lambda_j = 0".into())),
            ("support", J::Str("Z_jG = Y_j r_jG (raw counts); exactly 0 where Y_j = 0".into())),
            ("expected", J::Str("A_jG theta_G, the fitted expectation (not observed support)".into())),
            ("groups", J::Str("identical genome columns form one group and are never split; a \
multi-member group's attribution and theta belong to the group, members' genome theta is NaN".into())),
            ("own_attribution", J::Str("own_attributed_support / own_exposure (= n_anchors) is \
the source-attributed local raw depth; attributed_depth_* apply the pooled variant's scalars \
once. Source-attributed and model-dependent, not improved pooled truth; k-mers missing from the \
reference are not recovered".into())),
            ("summaries", J::Str("per genome over its OWN segments only, weights scored ACGT \
bases, mask n_anchors > 0 and value defined; not equal to fitted theta by construction".into())),
            ("not_implemented", J::Str("no uncertainty, entropy thresholds, resampling, model-fit \
flags or donor-coordinate projection".into()))])),
        ("build_id", J::Str(BUILD_ID.into())),
        ("rustcore_build_id_lineage", J::Str(RUSTCORE_LINEAGE.into()))]);
    out::write_json(&out.join("em-fit.json"), &fit)?;
    Ok(vec!["em-segments.npz", "em-attribution-sparse.npz", "em-fit.json"])
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().expect("rayon pool")
}

fn profile_load(a: &Args) -> Result<(), String> {
    // fast loader decodes with the CURRENT rayon pool: bound it to 8 like the timed workflow
    let (idx, rep, ph) = pool(8).install(|| load(a))?;
    let mut s = String::new();
    out::json(&obj(vec![("load", phases_json(&ph)), ("checksum", J::Str(format!("{:016x}", rep.checksum))),
                         ("coverage_sampler", J::Str(selection::mode_name(idx.sel.mode).into())),
                         ("peak_rss_kib_vmhwm", J::Int(vmhwm_kib())),
                         ("build_id", J::Str(BUILD_ID.into()))]), 0, &mut s)?;
    println!("{}", s);
    Ok(())
}

fn main() {
    if matches!(std::env::args().nth(1).as_deref(), Some("--help" | "-h" | "help")) {
        println!("LoCovE: alignment-free local coverage estimation\n\
Usage:\n\
  locove build --out INDEX --record-map-out MAP [--k 21 --c 20 --g 300] FASTA...\n\
  locove run --index INDEX --record-map MAP --out EXISTING_DIR [--threads 7] FASTQ...\n\
  locove describe --index INDEX\n\
  locove build-id\n\
One FASTA per genome. FASTQ and FASTA inputs may be gzip-compressed.\n\
Create the output directory before running; existing output files are not overwritten.\n\
--threads counts query workers, plus one reader when threads > 1.\n\
See README.md and docs/output.md for output definitions.");
        return;
    }
    let entry = Instant::now();
    let res = parse_args().and_then(|a| match a.cmd.as_str() {
        "run" => {
            // ONE pool, owned by main; main coordinates from outside it (no install around run)
            let p = pool(a.threads);
            run(&a, entry, &p)
        }
        "profile-load" => profile_load(&a),
        "build" => build(&a, entry),
        "exposures" => exposures_cmd(&a, entry),
        "describe" => describe(&a),
        "build-id" => {
            println!("{}", BUILD_ID);
            Ok(())
        }
        "lineage" => {
            println!("{}", RUSTCORE_LINEAGE);
            Ok(())
        }
        x => Err(format!("unknown command {}", x)),
    });
    if let Err(e) = res {
        eprintln!("method1-coverage: error: {}", e);
        std::process::exit(1);
    }
}

/// `describe`: the stored sampler and index provenance, after a full validated load.
fn describe(a: &Args) -> Result<(), String> {
    let (idx, rep, ph) = pool(8).install(|| load(a))?;
    let mut s = String::new();
    out::json(&obj(vec![
        ("index", J::Str(abspath(a.index.as_deref().unwrap_or(""))?)),
        ("loader", J::Str(ph.loader.clone())),
        ("checksum", J::Str(format!("{:016x}", rep.checksum))), ("bytes", J::Int(rep.bytes as i64)),
        ("parameters", obj(vec![("k", J::Int(idx.k as i64)), ("g", J::Int(idx.g as i64)),
                                ("c", J::Int(idx.c as i64))])),
        ("genomes", J::Int(idx.n_genomes as i64)), ("records", J::Int(idx.rec_len.len() as i64)),
        ("segments", J::Int(idx.st.len() as i64)), ("anchors", J::Int(idx.anchor_seg.len() as i64)),
        ("lookup_keys", J::Int(idx.sel_keys.len() as i64)),
        ("coverage_selection", sampler::describe(&idx)),
        ("build_id", J::Str(BUILD_ID.into()))]), 0, &mut s)?;
    println!("{}", s);
    Ok(())
}

/// `build`: the reference index from FASTAs (one path = one genome; records stay separate), with
/// the chosen coverage-anchor sampler. Single-threaded, as the incumbent builder. The index is
/// written to a temporary name beside OUT and renamed only after it reloads through the
/// reference loader and the sampler admission; an existing OUT is never overwritten.
fn build(a: &Args, entry: Instant) -> Result<(), String> {
    let outp = abspath(a.out.as_deref().ok_or("--out (the index path) is required")?)?;
    if Path::new(&outp).exists() {
        return Err(format!("{}: exists; build never overwrites an index", outp));
    }
    if let Some(m) = &a.record_map_out {
        if Path::new(m).exists() {
            return Err(format!("{}: exists; build never overwrites a record map", m));
        }
    }
    if let Some(m) = &a.exposures_out {
        if Path::new(m).exists() {
            return Err(format!("{}: exists; build never overwrites an exposure file", m));
        }
    }
    if a.reads.is_empty() {
        return Err("build needs at least one FASTA (one path per genome)".into());
    }
    let fastas: Vec<String> = a.reads.iter().map(|r| abspath(r)).collect::<Result<_, _>>()?;
    let (k, g, c) = (a.k.unwrap_or(21), a.g.unwrap_or(300), a.c.unwrap_or(20));
    if k == 0 || k > 32 {
        return Err(format!("k={} is outside 1..=32", k));
    }
    if g == 0 || c == 0 {
        return Err(format!("g={} c={}: both must be positive", g, c));
    }
    let mode = sampler::mode_of(a.sampler.as_deref().unwrap_or("fracminhash"))?;
    let sel = sampler::for_build(mode, k, c)?;
    let meta_in = match &a.meta_json {
        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| format!("{}: {}", p, e))?),
        None => None,
    };
    let t = Instant::now();
    let r = if mode == selection::MODE_SHIPPED {
        refindex::Reference::from_fastas(k, g, c, &fastas)?
    } else {
        refindex::Reference::from_fastas_sel(k, g, c, &fastas, sel)?
    };
    let build_s = t.elapsed().as_secs_f64();
    let parts = r.to_product_parts();
    let (valid, elig, cand) = r.elig_counts;
    let meta = match meta_in {
        Some(m) => m,
        None => {
            let mut inputs = Vec::new();
            for p in &fastas {
                let b = std::fs::read(p).map_err(|e| format!("{}: {}", p, e))?;
                inputs.push(obj(vec![("path", J::Str(p.clone())), ("bytes", J::Int(b.len() as i64)),
                                     ("fnv1a64", J::Str(format!("{:016x}", ser::fnv1a(&b))))]));
            }
            let mut m = String::new();
            out::json(&obj(vec![
                ("k", J::Int(k as i64)), ("g", J::Int(g as i64)), ("c", J::Int(c as i64)),
                ("genomes", J::List(parts.genome_names.iter().map(|x| J::Str(x.clone())).collect())),
                ("records", J::List(parts.rec_names.iter().map(|x| J::Str(x.clone())).collect())),
                ("n_segments", J::Int(parts.st.len() as i64)),
                ("n_anchors", J::Int(parts.anchor_seg.len() as i64)),
                ("coverage_sampler", J::Str(selection::mode_name(r.sel.mode).into())),
                ("inputs", J::List(inputs)),
                ("built_by", J::Str(format!("method1-coverage build {}", BUILD_ID))),
                ("rustcore_build_id_lineage", J::Str(RUSTCORE_LINEAGE.into()))]), 0, &mut m)?;
            m
        }
    };
    let tmp = format!("{}.partial-{}", outp, std::process::id());
    let t = Instant::now();
    let nbytes = product::ProductIndex::save_parts(&parts, &tmp, &meta)
        .map_err(|e| format!("{}: {}", tmp, e))?;
    out::fsync(Path::new(&tmp))?;
    drop(parts);
    let shipped_count = r.shipped_anchor_count;
    let (genomes, records, rec_genome) =
        (r.genome_names.clone(), r.rec_names.clone(), r.rec_genome.clone());
    let keep_r = if a.exposures_out.is_some() { Some(r) } else { drop(r); None };
    // self-check: the written file must load through the reference loader and be admitted
    let (idx, rep, _) = match load::load_reference(&tmp) {
        Ok(x) => x,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("the written index failed its own reload: {}", e));
        }
    };
    if Path::new(&outp).exists() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: appeared during the build; not overwritten", outp));
    }
    std::fs::rename(&tmp, &outp).map_err(|e| format!("{} -> {}: {}", tmp, outp, e))?;
    if let (Some(xo), Some(r)) = (&a.exposures_out, &keep_r) {
        write_exposures(r, &idx, rep.checksum, &abspath(xo)?, &outp)?;
    }
    drop(keep_r);
    if let Some(m) = &a.record_map_out {
        out::write_json(Path::new(m), &obj(vec![
            ("genomes", J::List(genomes.into_iter().map(J::Str).collect())),
            ("records", J::List(records.into_iter().map(J::Str).collect())),
            ("rec_genome", J::List(rec_genome.into_iter().map(J::Int).collect()))]))?;
        out::fsync(Path::new(m))?;
    }
    let write_s = t.elapsed().as_secs_f64();
    let mut s = String::new();
    out::json(&obj(vec![
        ("index", J::Str(outp.clone())), ("bytes", J::Int(nbytes as i64)),
        ("checksum", J::Str(format!("{:016x}", rep.checksum))),
        ("parameters", obj(vec![("k", J::Int(k as i64)), ("g", J::Int(g as i64)),
                                ("c", J::Int(c as i64))])),
        ("segments", J::Int(idx.st.len() as i64)), ("anchors", J::Int(idx.anchor_seg.len() as i64)),
        ("shipped_rule_anchor_count", J::Int(shipped_count as i64)),
        ("valid_starts", J::Int(valid as i64)), ("eligible_starts", J::Int(elig as i64)),
        ("candidates_before_thinning", J::Int(cand as i64)),
        ("eligible_anchor_density", if elig > 0 {
            J::Float(idx.anchor_seg.len() as f64 / elig as f64) } else { J::Null }),
        ("coverage_selection", sampler::describe(&idx)),
        ("meta", J::Str(if a.meta_json.is_some() { "caller-supplied, stored verbatim".into() }
                        else { "generated by this build".into() })),
        ("seconds", obj(vec![("build", J::Float(build_s)), ("write_and_reload", J::Float(write_s)),
                             ("total_in_process", J::Float(entry.elapsed().as_secs_f64()))])),
        ("peak_rss_kib_vmhwm", J::Int(vmhwm_kib())),
        ("build_id", J::Str(BUILD_ID.into())),
        ("rustcore_build_id_lineage", J::Str(RUSTCORE_LINEAGE.into()))]), 0, &mut s)?;
    println!("{}", s);
    Ok(())
}

fn write_exposures(r: &refindex::Reference, idx: &product::ProductIndex, checksum: u64,
                   xo: &str, index_path: &str) -> Result<exposure::Exposures, String> {
    if Path::new(xo).exists() {
        return Err(format!("{}: exists; exposure files are never overwritten", xo));
    }
    let x = exposure::Exposures::compute(r, idx, checksum)?;
    let mut m = String::new();
    out::json(&obj(vec![
        ("kind", J::Str("reference-only km per distinct anchor key (EM exposures)".into())),
        ("index", J::Str(index_path.to_string())),
        ("index_checksum", J::Str(format!("{:016x}", checksum))),
        ("sampler", J::Str(selection::mode_name(idx.sel.mode).into())),
        ("built_by", J::Str(format!("method1-coverage {}", BUILD_ID))),
        ("rustcore_build_id_lineage", J::Str(RUSTCORE_LINEAGE.into()))]), 0, &mut m)?;
    let tmp = format!("{}.partial-{}", xo, std::process::id());
    x.save(&tmp, &m)?;
    out::fsync(Path::new(&tmp))?;
    exposure::Exposures::load(&tmp, idx, checksum).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("the written exposure file failed its own reload: {}", e)
    })?;
    std::fs::rename(&tmp, xo).map_err(|e| format!("{} -> {}: {}", tmp, xo, e))?;
    Ok(x)
}

/// `exposures`: for an EXISTING index, rebuild its reference from the FASTAs with the index's
/// own k/g/c and stored sampler, require the rebuilt product to equal the index exactly, then
/// write the exposure file bound to it.
fn exposures_cmd(a: &Args, entry: Instant) -> Result<(), String> {
    let xo = abspath(a.out.as_deref().ok_or("--out (the exposure file) is required")?)?;
    if Path::new(&xo).exists() {
        return Err(format!("{}: exists; exposure files are never overwritten", xo));
    }
    if a.reads.is_empty() {
        return Err("exposures needs the index's FASTAs (one path per genome, same order)".into());
    }
    let (idx, rep, _) = pool(8).install(|| load(a))?;
    let fastas: Vec<String> = a.reads.iter().map(|r| abspath(r)).collect::<Result<_, _>>()?;
    let t = Instant::now();
    let r = if idx.sel.is_shipped() {
        refindex::Reference::from_fastas(idx.k, idx.g, idx.c, &fastas)?
    } else {
        let sel = sampler::for_build(idx.sel.mode, idx.k, idx.c)?;
        refindex::Reference::from_fastas_sel(idx.k, idx.g, idx.c, &fastas, sel)?
    };
    let parts = r.to_product_parts();
    let same = parts.sel_keys == idx.sel_keys && parts.sel_csr == idx.sel_csr
        && parts.aoff == idx.aoff && parts.aidx == idx.aidx && parts.anchor_seg == idx.anchor_seg
        && parts.st == idx.st && parts.en == idx.en && parts.seg_rec == idx.seg_rec
        && parts.dom_physical == idx.dom_physical && parts.dom_bases == idx.dom_bases
        && parts.dom_starts == idx.dom_starts && parts.dom_anchors == idx.dom_anchors
        && parts.rec_len == idx.rec_len && parts.rec_genome == idx.rec_genome
        && parts.genome_names == idx.genome_names && parts.rec_names == idx.rec_names
        && parts.thresh == idx.thresh && parts.n_genomes == idx.n_genomes
        && parts.selection.clone().unwrap_or_else(|| idx.sel.to_array()) == idx.sel.to_array();
    if !same {
        return Err("the FASTAs do not reproduce this index (segments, anchors, names or \
selection differ); exposures would not belong to it".into());
    }
    drop(parts);
    let build_s = t.elapsed().as_secs_f64();
    let x = write_exposures(&r, &idx, rep.checksum, &xo,
                            &abspath(a.index.as_deref().unwrap_or(""))?)?;
    let mut s = String::new();
    out::json(&obj(vec![
        ("exposures", J::Str(xo.clone())), ("index_checksum", J::Str(format!("{:016x}", rep.checksum))),
        ("sampler", J::Str(selection::mode_name(idx.sel.mode).into())),
        ("slots", J::Int(x.n_slots as i64)), ("entries", J::Int(x.key_genome.len() as i64)),
        ("seconds", obj(vec![("rebuild_and_check", J::Float(build_s)),
                             ("total_in_process", J::Float(entry.elapsed().as_secs_f64()))])),
        ("peak_rss_kib_vmhwm", J::Int(vmhwm_kib())), ("build_id", J::Str(BUILD_ID.into())),
        ("rustcore_build_id_lineage", J::Str(RUSTCORE_LINEAGE.into()))]), 0, &mut s)?;
    println!("{}", s);
    Ok(())
}
