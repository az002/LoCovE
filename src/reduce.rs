//! Segment reduction, the five approved depth variants and anchor breadth -- a port of the
//! archived Python reducer (`sub_compiled.reduce_all` -> `sub_count.segment_depth` /
//! `segment_breadth_anchors`, and `CompiledCounts`' sample-level quantities) that reproduces its
//! values BIT FOR BIT:
//!   * per-segment totals are `np.bincount(anchor_seg, weights)`: zero-initialised f64 sums in
//!     ANCHOR ORDER (raw counts cast to f64 first; breadth adds 1.0 per anchor with raw > 0);
//!     segments may be summed in parallel only because each segment's anchors are summed in
//!     the same order (contiguous runs when anchor_seg is non-decreasing; otherwise serial);
//!   * depth = where(n > 0, total / f64(max(n, 1)) / edge, NaN); novelty variants then divide by
//!     (1.0 - novelty) or become all-NaN with the R04 cause; geometry variants are all-NaN with
//!     the I3 cause when the sample has no k-mer window (sum_full_windows == 0);
//!   * edge_global = windows/bases, novelty = novel/total, novelty_weighted = nw/tw: integer
//!     ratios are exact here because both operands are below 2^53 (checked);
//!   * NaN is the constant 0x7ff8000000000000 (np.nan), undefined stays distinct from zero.

use crate::query::{Acc, NO_LEN};

pub const VARIANTS: [(&str, &str, bool); 5] = [
    ("raw_mean", "raw", false), ("geom_global", "global", false),
    ("geom_per_read", "per_read", false), ("geom_global_novelty", "global", true),
    ("geom_per_read_novelty", "per_read", true)];

/// A JSON-able value (the report is written by `out.rs`).
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<J>),
    Obj(Vec<(String, J)>),
}

pub fn obj(v: Vec<(&str, J)>) -> J {
    J::Obj(v.into_iter().map(|(k, x)| (k.to_string(), x)).collect())
}
fn opt_f(x: Option<f64>) -> J {
    x.map(J::Float).unwrap_or(J::Null)
}

pub struct Sample {
    pub edge_global: Option<f64>,
    pub novelty: Option<f64>,
    pub novelty_weighted: Option<f64>,
    pub geometry_defined: bool,
    pub sufficient: J,
    pub geo_cause: String,
    pub geo_detail: String,
}

fn exact_ratio(n: u64, d: u64, what: &str) -> Result<f64, String> {
    if n >= 1 << 53 || d >= 1 << 53 {
        return Err(format!("{}: operands {} / {} exceed 2^53; the Python reference divides \
exactly and f64 operands would not", what, n, d));
    }
    Ok(n as f64 / d as f64)
}

pub fn sample(acc: &Acc, k: usize) -> Result<Sample, String> {
    let (s, w) = (acc.sum_len, acc.sum_windows);
    let edge_global = if s != 0 { Some(exact_ratio(w, s, "edge_global")?) } else { None };
    let (t, n) = (acc.total, acc.novel);
    let novelty = if t != 0 { Some(exact_ratio(n, t, "novelty")?) } else { None };
    let (tw, nw) = (acc.total_w, acc.novel_w);
    let novelty_weighted = if tw != 0.0 { Some(nw / tw) } else { None };
    let sbf = if s != 0 { Some(exact_ratio(acc.sum_short_len, s, "short_read_base_fraction")?) }
              else { None };
    let min_len = if acc.min_len == NO_LEN { J::Null } else { J::Int(acc.min_len as i64) };
    let max_len = if acc.n_reads == 0 { J::Null } else { J::Int(acc.max_len as i64) };
    let range = if acc.n_reads != 0 { J::List(vec![min_len.clone(), max_len.clone()]) }
                else { J::Null };
    let (mn_j, mpw) = if acc.min_len_ge_k == NO_LEN { (J::Null, J::Null) } else {
        let mn = acc.min_len_ge_k;
        (J::Int(mn as i64), J::Float(exact_ratio(mn, mn - k as u64 + 1, "max_per_read_weight")?))
    };
    let sufficient = obj(vec![
        ("total_occurrences", J::Int(t as i64)), ("novel_occurrences", J::Int(n as i64)),
        ("total_weighted", J::Float(tw)), ("novel_weighted", J::Float(nw)),
        ("n_reads", J::Int(acc.n_reads as i64)),
        ("n_reads_shorter_than_k", J::Int(acc.n_short as i64)),
        ("sum_read_bases", J::Int(s as i64)), ("sum_full_windows", J::Int(w as i64)),
        ("edge_global", opt_f(edge_global)), ("edge_fixed", J::Null),
        ("novelty", opt_f(novelty)), ("novelty_weighted", opt_f(novelty_weighted)),
        ("i3", obj(vec![
            ("n_reads_shorter_than_k", J::Int(acc.n_short as i64)),
            ("sum_short_read_bases", J::Int(acc.sum_short_len as i64)),
            ("short_read_base_fraction", opt_f(sbf)),
            ("read_length_range", range),
            ("min_read_length_at_least_k", mn_j),
            ("max_per_read_weight", mpw),
            ("no_opportunity", J::Bool(w == 0)),
            ("undefined_note", J::Str("a None here is an explicit undefined -- an empty \
sample has no length range, and a sample with no read reaching k has no maximum per-read \
weight. No filter and no weight clipping is applied.".into())),
        ])),
    ]);
    let (cause, detail) = if w != 0 { (String::new(), String::new()) } else if s == 0 {
        ("no_queried_read_bases".to_string(), "the sample contributed zero read bases".to_string())
    } else {
        ("no_potential_kmer_window".to_string(),
         format!("the sample contributed {} read bases but no read reached k, so no k-mer \
window ever existed", s))
    };
    Ok(Sample { edge_global, novelty, novelty_weighted, geometry_defined: w != 0, sufficient,
                geo_cause: cause, geo_detail: detail })
}

pub struct Reduced {
    pub n_anchors: Vec<i64>,
    pub breadth: Vec<f64>,
    pub depth: Vec<(String, Vec<f64>)>,
    pub depth_info: J,
    pub parallel_segments: bool,
}

/// Per-segment (n, sum raw as f64, sum weighted, sum hit) in anchor order.
fn segment_totals(anchor_seg: &[i64], raw: &[i64], wtd: &[f64], ns: usize)
    -> Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, bool), String> {
    use rayon::prelude::*;
    let mut n = vec![0i64; ns];
    for &s in anchor_seg {
        if s < 0 || s as usize >= ns {
            return Err(format!("anchor segment {} outside 0..{}", s, ns));
        }
        n[s as usize] += 1;
    }
    let (mut tr, mut tw, mut th) = (vec![0f64; ns], vec![0f64; ns], vec![0f64; ns]);
    let monotone = anchor_seg.windows(2).all(|p| p[0] <= p[1]);
    if monotone {
        let mut off = vec![0usize; ns + 1];
        for s in 0..ns {
            off[s + 1] = off[s] + n[s] as usize;
        }
        tr.par_iter_mut().zip(tw.par_iter_mut()).zip(th.par_iter_mut()).enumerate()
            .with_min_len(4096)
            .for_each(|(s, ((r, w), h))| {
                for a in off[s]..off[s + 1] {
                    *r += raw[a] as f64;
                    *w += wtd[a];
                    *h += if raw[a] > 0 { 1.0 } else { 0.0 };
                }
            });
    } else {
        for (a, &s) in anchor_seg.iter().enumerate() {
            let s = s as usize;
            tr[s] += raw[a] as f64;
            tw[s] += wtd[a];
            th[s] += if raw[a] > 0 { 1.0 } else { 0.0 };
        }
    }
    Ok((n, tr, tw, th, monotone))
}

pub fn reduce(anchor_seg: &[i64], ns: usize, acc: &Acc, sm: &Sample) -> Result<Reduced, String> {
    let (n, tr, tw, th, mono) = segment_totals(anchor_seg, &acc.raw, &acc.wtd, ns)?;
    let na_count = n.iter().filter(|&&x| x == 0).count() as i64;
    let mean = |tot: &[f64], edge: f64| -> Vec<f64> {
        tot.iter().zip(&n).map(|(&t, &c)| if c > 0 { t / (c.max(1) as f64) / edge }
                                        else { f64::NAN }).collect()
    };
    let mut depth = Vec::new();
    let mut infos = Vec::new();
    for (name, arm, nov) in VARIANTS {
        if (arm == "global" || arm == "per_read") && !sm.geometry_defined {
            depth.push((name.to_string(), vec![f64::NAN; ns]));
            infos.push((name.to_string(), obj(vec![
                ("arm", J::Str(arm.into())), ("edge", J::Null), ("novelty_used", J::Null),
                ("segments_na_no_eligible_anchor", J::Int(na_count)),
                ("novelty_correction", J::Str(if nov { "requested" } else { "off" }.into())),
                ("arm_undefined", J::Bool(true)),
                ("arm_undefined_cause", J::Str(sm.geo_cause.clone())),
                ("arm_undefined_detail", J::Str(sm.geo_detail.clone())),
                ("note", J::Str("I3: no potential k-mer window in this sample, so every \
GEOMETRY-ADJUSTED variant is NA with this reason. The raw arm and breadth are retained and still \
reported.".into())),
            ])));
            continue;
        }
        let (tot, edge, novv): (&[f64], f64, Option<f64>) = match arm {
            "raw" => (&tr, 1.0, None),
            "global" => {
                let e = sm.edge_global.filter(|&e| e != 0.0)
                    .ok_or("global geometry undefined: no queried read bases")?;
                (&tr, e, sm.novelty)
            }
            _ => (&tw, 1.0, sm.novelty_weighted),
        };
        let mut d = mean(tot, edge);
        let mut info = vec![("arm", J::Str(arm.into())), ("edge", J::Float(edge)),
                            ("novelty_used", opt_f(novv)),
                            ("segments_na_no_eligible_anchor", J::Int(na_count))];
        let mut corr = "off".to_string();
        let mut cause = None;
        if nov {
            match novv {
                Some(v) if 1.0 - v != 0.0 => {
                    let f = 1.0 - v;
                    for x in d.iter_mut() {
                        *x /= f;
                    }
                    corr = "applied".into();
                }
                other => {
                    corr = "undefined".into();
                    cause = Some(if other.is_none() { "zero selected occurrences" }
                                 else { "novelty equals 1" });
                    d = vec![f64::NAN; ns];
                }
            }
        }
        info.push(("novelty_correction", J::Str(corr)));
        if let Some(c) = cause {
            info.push(("novelty_correction_cause", J::Str(c.into())));
        }
        info.push(("arm_undefined", J::Bool(false)));
        depth.push((name.to_string(), d));
        infos.push((name.to_string(), obj(info)));
    }
    let breadth: Vec<f64> = th.iter().zip(&n).map(|(&h, &c)| if c > 0 { h / (c.max(1) as f64) }
                                                  else { f64::NAN }).collect();
    Ok(Reduced { n_anchors: n, breadth, depth, depth_info: J::Obj(infos), parallel_segments: mono })
}
