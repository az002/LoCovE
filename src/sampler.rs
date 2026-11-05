//! Coverage-anchor SAMPLER policy of the standalone CLI (2026-10-02 handoff: open syncmers as an
//! opt-in index-build option).
//!
//! The selection semantics live in `selection.rs` and `refindex.rs`, copied UNCHANGED from the
//! frozen syncmer_core (build 0fdc8d3fdd5a7ee7). This module only decides which of them the
//! CLI offers, and pins the stored metadata to the approved settings at load:
//!
//!   fracminhash  (default) shipped FracMinHash: anchor iff splitmix64(canonical) < 2^64/c.
//!                Original-format index (kind 'P'), byte-identical to the incumbent builder.
//!   syncmer      production_syncmer_s9t6_nominal: open syncmer (k=21, s=9, central
//!                zero-based t=6 of the 13 s-mer offsets, centre AMONG tied minima, s-mer
//!                order splitmix64(canonical s-mer)) AND fmix64(canonical) < q * 2^64 with
//!                q = 13/c fixed for every catalog (0.65 at c=20). q > 1 (c < 13) is refused.
//!                Candidate index (kind 'Y') carrying the selection array.
//!
//! Selection is an INDEX property: queries honour the stored sampler, and `--sampler` on a
//! query is an assertion that must match it, never a switch. Boundaries, per-genome repeat
//! eligibility, exact-key counting, the novelty sampler/membership/counters, the correction
//! variants and undefined masks are not touched by either sampler.
//!
//! The per-catalog count-matched EXPERIMENT selections that syncmer_core can also write are
//! refused here: they are not production settings (HANDOFF production policy), and running
//! them from the production CLI would let a catalog-calibrated threshold pass as one.

use crate::product::ProductIndex;
use crate::reduce::{obj, J};
use crate::selection::{mode_name, Selection, MODE_SHIPPED, MODE_SYNCMER_NOMINAL};

/// The approved scheme. Not generalized: other k/s/t need a separate approval.
pub const APPROVED_K: usize = 21;
pub const APPROVED_S: usize = 9;
pub const APPROVED_T: usize = 6;

/// CLI sampler name -> stored selection mode. The stored (artifact) names are accepted too.
pub fn mode_of(name: &str) -> Result<u64, String> {
    match name {
        "fracminhash" | "shipped_fracminhash" => Ok(MODE_SHIPPED),
        "syncmer" | "production_syncmer_s9t6_nominal" => Ok(MODE_SYNCMER_NOMINAL),
        x => Err(format!("unknown sampler {:?}: use fracminhash (default) or syncmer", x)),
    }
}

pub fn cli_name(mode: u64) -> &'static str {
    match mode {
        MODE_SHIPPED => "fracminhash",
        MODE_SYNCMER_NOMINAL => "syncmer",
        _ => "unsupported",
    }
}

/// The selection a `build` constructs. Thinning is set later by the builder (`set_nominal`).
pub fn for_build(mode: u64, k: usize, c: usize) -> Result<Selection, String> {
    if mode == MODE_SHIPPED {
        return Ok(Selection::shipped(k));
    }
    if k != APPROVED_K {
        return Err(format!("the syncmer sampler is approved only for k={} (s={}, t={}); k={} \
needs a separately approved scheme", APPROVED_K, APPROVED_S, APPROVED_T, k));
    }
    let mut sel = Selection::new(mode, k, APPROVED_S, APPROVED_T)?;
    // refuse infeasible q > 1 BEFORE reading any FASTA (the builder would refuse it too)
    sel.set_nominal(c as u64)?;
    Ok(Selection::new(mode, k, APPROVED_S, APPROVED_T)?)
}

/// Load-time admission, applied by BOTH loaders after the structural validation: the stored
/// selection must be one this CLI runs, with exactly the approved settings and the nominal
/// thinning re-derived from the stored c. A checksum-valid file whose selection metadata was
/// altered is refused here rather than run under a label it does not satisfy.
pub fn admit(idx: &ProductIndex, path: &str) -> Result<(), String> {
    let s = &idx.sel;
    if s.mode == MODE_SHIPPED {
        return Ok(());
    }
    if s.mode != MODE_SYNCMER_NOMINAL {
        return Err(format!("{}: index selection {} is an EXPERIMENT selection; this CLI runs \
only shipped_fracminhash and production_syncmer_s9t6_nominal indexes", path, mode_name(s.mode)));
    }
    if s.k != APPROVED_K || s.s != APPROVED_S || s.t != APPROVED_T || idx.k != APPROVED_K {
        return Err(format!("{}: syncmer index declares k={} s={} t={}; only the approved k={} \
s={} t={} is accepted", path, s.k, s.s, s.t, APPROVED_K, APPROVED_S, APPROVED_T));
    }
    let mut want = Selection::new(s.mode, s.k, s.s, s.t)?;
    want.set_nominal(idx.c as u64).map_err(|e| format!("{}: {}", path, e))?;
    if s.q_num != want.q_num || s.q_den != want.q_den || s.thin != want.thin {
        return Err(format!("{}: stored nominal thinning (q={}/{}, threshold {:016x}) does not \
match q={}/{} (threshold {:016x}) for c={}", path, s.q_num, s.q_den, s.thin, want.q_num,
                           want.q_den, want.thin, idx.c));
    }
    if s.next_group_anchors != 0 {
        return Err(format!("{}: a nominal selection records a count-calibration residual", path));
    }
    Ok(())
}

/// `--sampler` on a query or profile: an assertion against the stored sampler.
pub fn check_expected(idx: &ProductIndex, expected: Option<&str>) -> Result<(), String> {
    if let Some(e) = expected {
        let m = mode_of(e)?;
        if m != idx.sel.mode {
            return Err(format!("--sampler {} was requested but the index was built with {} ({}); \
selection is an index-build choice, rebuild the index with `method1-coverage build --sampler {}`",
                               e, cli_name(idx.sel.mode), mode_name(idx.sel.mode), e));
        }
    }
    Ok(())
}

fn hex(x: u64) -> J {
    J::Str(format!("{:016x}", x))
}

/// Report/provenance description of the stored coverage-anchor sampler.
pub fn describe(idx: &ProductIndex) -> J {
    let s = &idx.sel;
    let novelty = J::Str(format!("unchanged shipped sampler: read windows with \
splitmix64(canonical) < 2^64/{} judged against full-reference membership", idx.c));
    if s.mode == MODE_SHIPPED {
        return obj(vec![
            ("sampler", J::Str("fracminhash".into())),
            ("name", J::Str(mode_name(s.mode).into())),
            ("index_kind", J::Str("P (original format)".into())),
            ("anchor_rule", J::Str(format!("splitmix64(canonical k-mer) < 2^64/{}", idx.c))),
            ("anchor_threshold_exclusive", hex(idx.thresh)),
            ("anchors", J::Int(idx.anchor_seg.len() as i64)),
            ("novelty", novelty)]);
    }
    let thin_frac = (s.thin as f64 + 1.0) / 18446744073709551616.0;
    obj(vec![
        ("sampler", J::Str(cli_name(s.mode).into())),
        ("name", J::Str(mode_name(s.mode).into())),
        ("index_kind", J::Str("Y (candidate selection array)".into())),
        ("anchor_rule", J::Str(format!("open syncmer (k={}, s={}, central t={} of {} offsets, \
centre among tied minima, splitmix64 canonical s-mer order) AND fmix64(canonical k-mer) <= \
thin_inclusive; nominal q = {}/{}", s.k, s.s, s.t, s.n_off, s.q_num, s.q_den))),
        ("k", J::Int(s.k as i64)), ("s", J::Int(s.s as i64)), ("t", J::Int(s.t as i64)),
        ("n_offsets", J::Int(s.n_off as i64)),
        ("nominal_q", J::List(vec![J::Int(s.q_num as i64), J::Int(s.q_den as i64)])),
        ("thin_inclusive", hex(s.thin)),
        ("thin_fraction_of_2_64", J::Float(thin_frac)),
        ("anchors", J::Int(s.realized as i64)),
        ("shipped_rule_anchor_count_at_build", J::Int(s.target as i64)),
        ("novelty", novelty),
        ("selection_array_decimal", J::List(s.to_array().into_iter().map(|x| J::Str(format!("{}", x)))
                                    .collect()))])
}
