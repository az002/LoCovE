//! Streaming FASTQ input and the method-1 accumulators, against the PRODUCT index.
//!
//! Compiled counterpart of `src/sub_count.py::query`. Three things are deliberately absent:
//!
//!   * no read is assigned to a source, and no method-2 / read-assignment statistic is
//!     computed. The approved product is occurrence counts over eligible anchors; a second arm
//!     here would be a new product, not a compiled version of this one.
//!   * no whole-sample materialisation. The file is streamed record by record, or in
//!     record-aligned blocks for the parallel path, so a 10 Gbp sample is never a buffer.
//!   * no dense reference array. The query reads the product index only: selected membership,
//!     anchor postings and segment metadata. `fwd`, `abase`, `gmult` and the per-position
//!     hash track are truth-side and are not opened here.
//!
//! ONE PROBE PER OCCURRENCE. Membership and postings share a single open-addressed lookup:
//! `None` is novel, `MEMBER_ONLY` is present-but-not-an-anchor, anything else is a CSR slot.
//! The oracle does two binary searches per occurrence because numpy has no cheap hash map;
//! that is the correctness prototype's shape, not the product's, and copying it would have
//! made the compiled path carry the prototype's cost structure.
//!
//! METHOD-1 PIPELINE CHANGE (depth_adjustments/method1_core, 2026-09-29). Only the PARALLEL
//! path changed, and only in two places; what is counted and in what order is not:
//!   * PIPELINED READING: one reader thread decompresses/parses the NEXT record-aligned block
//!     (the unchanged `for_each_block`) while the counting threads process the current one.
//!     Blocks, their byte split across workers, and each worker's read order are unchanged.
//!     The reader is one extra OS thread beside the `threads` counting workers.
//!   * PARALLEL MERGE: worker arrays are merged in parallel over anchor chunks, in place into
//!     worker 0; every anchor still adds workers 1..T in fixed order after worker 0, which is
//!     bit-identical to the archived `0 + w0 + w1 + ...` (0.0 + x == x: weights are sums of
//!     positive terms, never -0.0). Scalar counters merge exactly as before.
//! The serial path, the per-read kernel, the index and every statistic are untouched.
//!
//! KEY-FIRST CANDIDATE (depth_adjustments/method1_keyfirst_core, 2026-09-30). On top of the
//! pipelined build (2a7c05c6be14a465), per-worker counting accumulates per POSTING SLOT `j`
//! (one CSR slot per distinct anchor key) instead of per anchor: each non-novel, non-member-only
//! selected occurrence does `cnt[j] += 1; wsum[j] += w` once, instead of one scattered update per
//! posted anchor. After the slot-level merge the slot totals are EXPANDED once to the anchors of
//! each slot. Exact, not approximate, because `ProductIndex::validate` (run by every load, the
//! only way a queryable index is made) requires `aidx` to be a permutation of all anchors: every
//! anchor is in exactly one slot's postings, exactly once, so its archived per-worker sequence of
//! additions `0 + w_1 + w_2 + ...` is precisely that slot's sequence, in the same read and
//! occurrence order. Integer counts are equal; each anchor's f64 weight is bit-identical per
//! worker, and the slot merge adds workers in the archived fixed order. Novelty, member-only,
//! repeat policy, weights and every scalar are computed exactly as before.
//!
//! ZLIB-RS CANDIDATE (depth_adjustments/method1_keyfirst_zlibrs_core, 2026-10-01): identical to
//! method1_keyfirst_core except that flate2 decodes gzip with zlib-rs instead of miniz_oxide
//! (Cargo.toml only). Decoded bytes are defined by the gzip format and CRC-checked by flate2
//! with either backend, so reads, blocks and their split are unchanged.
//!
//! EXACTNESS. The serial path adds to each per-anchor accumulator in exactly the order the
//! oracle does -- reads in file order, occurrences in ascending start order, anchors in the
//! stored CSR order -- so the f64 weighted channel is bit-identical to the oracle's, not merely
//! close. The parallel path sums the same values in a different association order; its integer
//! channels stay bit-identical and its weighted channel agrees to f64 rounding, which is the
//! only tolerance anywhere in this comparison and is justified by arithmetic alone.

use crate::kernel::scan_select;
use crate::product::{ProductIndex, MEMBER_ONLY};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Acc {
    pub raw: Vec<i64>,
    pub wtd: Vec<f64>,
    pub total: u64,
    pub novel: u64,
    pub total_w: f64,
    pub novel_w: f64,
    pub n_reads: u64,
    pub n_short: u64,
    pub sum_len: u64,
    pub sum_windows: u64,
    /// I3 reporting counters, accumulated in the same single pass so nothing needs a second
    /// read of the FASTQ after the query:
    ///   `sum_short_len`  bases in reads with L < k -- the input-base fraction I3 asks for
    ///   `min_len`/`max_len`  the observed length range
    ///   `min_len_ge_k`   the shortest read that reached k, which is where the per-read weight
    ///                    L/(L-k+1) is at its MAXIMUM. Kept as a length rather than a float so
    ///                    the maximum is derived exactly and merges by `min`.
    pub sum_short_len: u64,
    pub min_len: u64,
    pub max_len: u64,
    pub min_len_ge_k: u64,
    keys: Vec<u64>,
    /// syncmer_core: per-read s-mer hash track (candidate selections only)
    hs: Vec<u64>,
}

/// Sentinel for "no read in this class was seen". Reported to Python as an explicit undefined
/// rather than as a number that would read as a measured length.
pub const NO_LEN: u64 = u64::MAX;

impl Acc {
    pub fn new(n_anchors: usize) -> Acc {
        Acc {
            raw: vec![0i64; n_anchors],
            wtd: vec![0.0f64; n_anchors],
            total: 0, novel: 0, total_w: 0.0, novel_w: 0.0,
            n_reads: 0, n_short: 0, sum_len: 0, sum_windows: 0,
            sum_short_len: 0, min_len: NO_LEN, max_len: 0, min_len_ge_k: NO_LEN,
            keys: Vec::new(), hs: Vec::new(),
        }
    }

    /// Fold one read in. `read` is the sequence line, already newline-stripped.
    #[inline]
    pub fn push_read(&mut self, r: &ProductIndex, read: &[u8]) {
        let l = read.len();
        self.n_reads += 1;
        self.sum_len += l as u64;
        let lu = l as u64;
        if lu < self.min_len {
            self.min_len = lu;
        }
        if lu > self.max_len {
            self.max_len = lu;
        }
        let k = r.k;
        if l >= k && lu < self.min_len_ge_k {
            self.min_len_ge_k = lu;
        }
        if l < k {
            self.sum_short_len += lu;
            // No window exists, so this read contributes bases and nothing else. Its bases
            // still sit in the R03a denominator (`sum_len`), which is what makes the global
            // edge a ratio of sums rather than an average of per-read factors.
            self.n_short += 1;
            return;
        }
        self.sum_windows += (l - k + 1) as u64;
        let w = l as f64 / (l - k + 1) as f64;
        if !r.sel.is_shipped() {
            self.push_windows_selected(r, read, w);
            return;
        }
        let mut keys = std::mem::take(&mut self.keys);
        keys.clear();
        scan_select(read, k, r.thresh, |_p, h| keys.push(h));
        if keys.is_empty() {
            self.keys = keys;
            return;
        }
        let nk = keys.len() as u64;
        self.total += nk;
        self.total_w += w * nk as f64;
        let mut nv = 0u64;
        for &q in keys.iter() {
            match r.map.get(q) {
                None => nv += 1,                      // S3: absent from every valid start
                Some(MEMBER_ONLY) => {}               // present, but no eligible anchor
                Some(j) => {
                    // key-first: `raw`/`wtd` are indexed by posting slot while counting
                    let j = j as usize;
                    self.raw[j] += 1;
                    self.wtd[j] += w;
                }
            }
        }
        self.novel += nv;
        self.novel_w += w * nv as f64;
        self.keys = keys;
    }

    /// syncmer_core candidate selection, one read (length >= k). Every valid window is visited
    /// once: NOVELTY uses the unchanged shipped sampler (splitmix64 < thresh) against the lookup
    /// set, which for such windows is exactly full-reference membership; ANCHOR credit uses the
    /// configured selection. Repeated occurrences in a read each count, as in the shipped path;
    /// the f64 additions follow window order, as in the shipped path.
    fn push_windows_selected(&mut self, r: &ProductIndex, read: &[u8], w: f64) {
        use crate::kernel::{kmask, splitmix, LUT};
        let k = r.k;
        let sel = &r.sel;
        let n_off = sel.n_off;
        let need = sel.needs_syncmer();
        let mut hs = std::mem::take(&mut self.hs);
        if need {
            crate::selection::smer_track(read, sel.s, &mut hs);
        }
        let n = read.len();
        let mask = kmask(k);
        let sh = 2 * (k - 1);
        let (mut f, mut rv, mut run) = (0u64, 0u64, 0usize);
        let (mut nk, mut nv) = (0u64, 0u64);
        for i in 0..n {
            let c = LUT[read[i] as usize];
            let ok = c >= 0;
            let cs = if ok { c as u64 } else { 0 };
            f = ((f << 2) | cs) & mask;
            rv = (rv >> 2) | ((3 - cs) << sh);
            run = if ok { run + 1 } else { 0 };
            if i + 1 >= k && run >= k {
                let p = i + 1 - k;
                let canon = if f <= rv { f } else { rv };
                let h = splitmix(canon);
                let old = h < r.thresh;
                let new = if need { sel.anchor_ok(canon, &hs[p..p + n_off]) }
                          else { sel.anchor_ok(canon, &[]) };
                if old {
                    nk += 1;
                }
                if old || new {
                    match r.map.get(h) {
                        None => {
                            if old {
                                nv += 1;
                            }
                        }
                        Some(MEMBER_ONLY) => {}
                        Some(j) => {
                            if new {
                                let j = j as usize;
                                self.raw[j] += 1;
                                self.wtd[j] += w;
                            }
                        }
                    }
                }
            }
        }
        self.hs = hs;
        if nk > 0 {
            self.total += nk;
            self.total_w += w * nk as f64;
            self.novel += nv;
            self.novel_w += w * nv as f64;
        }
    }

    /// Fold another accumulator in. Integer channels are exact; the f64 channel re-associates.
    pub fn merge(&mut self, o: &Acc) {
        for (a, b) in self.raw.iter_mut().zip(&o.raw) {
            *a += b;
        }
        for (a, b) in self.wtd.iter_mut().zip(&o.wtd) {
            *a += b;
        }
        self.merge_scalars(o);
    }

    /// The scalar part of `merge` (the parallel path merges the per-anchor arrays separately).
    pub fn merge_scalars(&mut self, o: &Acc) {
        self.total += o.total;
        self.novel += o.novel;
        self.total_w += o.total_w;
        self.novel_w += o.novel_w;
        self.n_reads += o.n_reads;
        self.n_short += o.n_short;
        self.sum_len += o.sum_len;
        self.sum_windows += o.sum_windows;
        self.sum_short_len += o.sum_short_len;
        self.min_len = self.min_len.min(o.min_len);
        self.max_len = self.max_len.max(o.max_len);
        self.min_len_ge_k = self.min_len_ge_k.min(o.min_len_ge_k);
    }

    /// Per-segment reduction of the two count channels plus the raw breadth numerator.
    ///
    /// Returned alongside the per-anchor vectors rather than instead of them: the approved
    /// reducer needs the anchor-level counts to be reusable for rescoring, and a workflow that
    /// wrote only a segment mean would have to re-query to change a correction.
    pub fn segment_sums(&self, r: &ProductIndex) -> (Vec<i64>, Vec<f64>, Vec<i64>) {
        let ns = r.n_segments();
        let (mut sr, mut sw, mut sh) = (vec![0i64; ns], vec![0.0f64; ns], vec![0i64; ns]);
        for (a, &sg) in r.anchor_seg.iter().enumerate() {
            let s = sg as usize;
            sr[s] += self.raw[a];
            sw[s] += self.wtd[a];
            if self.raw[a] > 0 {
                sh[s] += 1;
            }
        }
        (sr, sw, sh)
    }
}

fn open(path: &str) -> Result<Box<dyn BufRead>, String> {
    let f = File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    Ok(if path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(f)))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, f))
    })
}

fn strip(v: &mut Vec<u8>) {
    while matches!(v.last(), Some(b'\n') | Some(b'\r')) {
        v.pop();
    }
}

/// Stream a FASTQ, handing each validated sequence to `f`.
///
/// The record structure is VALIDATED, never assumed, with the same rules and the same messages
/// as the oracle: a header that does not start with `@`, a separator that is not `+`, a
/// truncated record, or a quality string whose length differs from the sequence. Accepting any
/// of those silently makes a corrupt or truncated input look like a smaller successful run.
pub fn for_each_read<F: FnMut(&[u8])>(path: &str, mut f: F) -> Result<u64, String> {
    let mut rd = open(path)?;
    let (mut h, mut s, mut p, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n = 0u64;
    loop {
        h.clear();
        if rd.read_until(b'\n', &mut h).map_err(|e| e.to_string())? == 0 {
            return Ok(n);
        }
        n += 1;
        if h.first() != Some(&b'@') {
            let show = String::from_utf8_lossy(&h[..h.len().min(60)]).to_string();
            return Err(format!("{} record {}: header does not start with \"@\": {:?}",
                               path, n, show));
        }
        s.clear();
        rd.read_until(b'\n', &mut s).map_err(|e| e.to_string())?;
        strip(&mut s);
        p.clear();
        let np = rd.read_until(b'\n', &mut p).map_err(|e| e.to_string())?;
        q.clear();
        rd.read_until(b'\n', &mut q).map_err(|e| e.to_string())?;
        strip(&mut q);
        if np == 0 || q.is_empty() {
            return Err(format!("{} record {}: truncated record", path, n));
        }
        if p.first() != Some(&b'+') {
            let show = String::from_utf8_lossy(&p[..p.len().min(20)]).to_string();
            return Err(format!("{} record {}: separator is {:?}, not \"+\"", path, n, show));
        }
        if q.len() != s.len() {
            return Err(format!("{} record {}: quality length {} != sequence length {}",
                               path, n, q.len(), s.len()));
        }
        f(&s);
    }
}

/// Stream a FASTQ in record-aligned blocks of roughly `target` bases.
///
/// Blocks end on record boundaries, so a read is never split and the parallel and serial paths
/// see exactly the same reads. Validation is identical to `for_each_read`, which is where any
/// error surfaces, so the block assembler needs no error channel of its own.
pub fn for_each_block<F: FnMut(&[u8], &[usize])>(path: &str, target: usize, mut f: F)
    -> Result<(), String> {
    let mut flat: Vec<u8> = Vec::with_capacity(target + (1 << 16));
    let mut offs: Vec<usize> = vec![0];
    for_each_read(path, |s| {
        flat.extend_from_slice(s);
        offs.push(flat.len());
        if flat.len() >= target {
            f(&flat, &offs);
            flat.clear();
            offs.clear();
            offs.push(0);
        }
    })?;
    if offs.len() > 1 {
        f(&flat, &offs);
    }
    Ok(())
}

/// `for_each_block` with the decompression/parsing of the NEXT block on one reader thread, so it
/// overlaps the processing of the current one. Same blocks, same order, same error: the reader
/// runs the unchanged `for_each_block`, and its `Result` is returned after every block it
/// produced has been handed to `f`, exactly as the in-line version does.
fn for_each_block_pipelined<F: FnMut(&[u8], &[usize])>(path: &str, target: usize, mut f: F)
    -> Result<(), String> {
    use std::sync::mpsc::{channel, sync_channel};
    let (tx, rx) = sync_channel::<(Vec<u8>, Vec<usize>)>(1);
    let (back_tx, back_rx) = channel::<(Vec<u8>, Vec<usize>)>();
    std::thread::scope(|s| {
        let reader = s.spawn(move || {
            for_each_block(path, target, |flat, offs| {
                let (mut fb, mut ob) = back_rx.try_recv()
                    .unwrap_or_else(|_| (Vec::with_capacity(target + (1 << 16)), Vec::new()));
                fb.clear();
                fb.extend_from_slice(flat);
                ob.clear();
                ob.extend_from_slice(offs);
                // a send can only fail if the consumer has gone, which only a panic does
                let _ = tx.send((fb, ob));
            })
        });
        for (fb, ob) in rx.iter() {
            f(&fb, &ob);
            let _ = back_tx.send((fb, ob));
        }
        reader.join().map_err(|_| format!("{}: reader thread panicked", path))?
    })
}

/// Number of posting slots (distinct anchor keys) -- the length of the counting arrays.
fn n_slots(r: &ProductIndex) -> usize {
    r.aoff.len().saturating_sub(1)
}

/// Slot totals -> per-anchor channels: every anchor of slot `j` receives slot `j`'s count and
/// weight, once. Anchors are a permutation over slots (load-validated), so each anchor is
/// written exactly once and nothing is added: the value IS the archived per-anchor sum.
fn expand_slots(r: &ProductIndex, acc: &mut Acc, parallel: bool) {
    let na = r.n_anchors();
    let (cnt, wsum) = (std::mem::take(&mut acc.raw), std::mem::take(&mut acc.wtd));
    let mut raw = vec![0i64; na];
    let mut wtd = vec![0f64; na];
    let fill = |raw: &mut [i64], wtd: &mut [f64], j0: usize, j1: usize| {
        for j in j0..j1 {
            let (c, w) = (cnt[j], wsum[j]);
            if c == 0 {
                continue; // anchors keep the zeroed 0 and +0.0 the archived arrays start with
            }
            for &a in &r.aidx[r.aoff[j] as usize..r.aoff[j + 1] as usize] {
                raw[a as usize] = c;
                wtd[a as usize] = w;
            }
        }
    };
    if parallel {
        // Disjoint writes: slot ranges are split across tasks, and because `aidx` is a
        // validated permutation no anchor belongs to two slots. The shared pointers are only
        // ever written at distinct indices.
        struct P(*mut i64, *mut f64, usize);
        unsafe impl Sync for P {}
        let p = P(raw.as_mut_ptr(), wtd.as_mut_ptr(), na);
        let ns = cnt.len();
        let parts = rayon::current_num_threads().max(1) * 4;
        use rayon::prelude::*;
        (0..parts).into_par_iter().for_each(|t| {
            let (j0, j1) = (ns * t / parts, ns * (t + 1) / parts);
            let pr = &p;
            let raw = unsafe { std::slice::from_raw_parts_mut(pr.0, pr.2) };
            let wtd = unsafe { std::slice::from_raw_parts_mut(pr.1, pr.2) };
            fill(raw, wtd, j0, j1);
        });
    } else {
        fill(&mut raw, &mut wtd, 0, cnt.len());
    }
    acc.raw = raw;
    acc.wtd = wtd;
}

/// Serial query over one or more FASTQ paths. This is the oracle-exact path.
pub fn query_serial(r: &ProductIndex, paths: &[String]) -> Result<Acc, String> {
    let mut acc = Acc::new(n_slots(r));
    for p in paths {
        for_each_read(p, |s| acc.push_read(r, s))?;
    }
    expand_slots(r, &mut acc, false);
    Ok(acc)
}

/// Bounded-parallel query. `threads` is an EXPLICIT bound, never "all cores".
///
/// Reads are the unit of parallelism because they are genuinely independent: a read touches its
/// own anchors and nothing else. Each worker owns a private accumulator allocated once and
/// reused across blocks, and the partials are merged in FIXED thread order, so a repeated run
/// is bit-identical to itself -- a nondeterministic reduction would make the serial/parallel
/// comparison untestable rather than merely approximate.
pub fn query_parallel(r: &ProductIndex, paths: &[String], threads: usize, block: usize)
    -> Result<Acc, String> {
    if threads <= 1 {
        return query_serial(r, paths);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| format!("rayon pool: {}", e))?;
    query_parallel_in(&pool, r, paths, threads, block)
}

/// method1_rust_cli (2026-10-01): the same parallel query on a CALLER-OWNED pool of exactly
/// `threads` workers, so the CLI keeps ONE pool for loading, counting and reduction instead of
/// a second idle one. Called from outside the pool (the coordinating main thread), exactly as
/// the Python path calls `query_parallel` from the interpreter thread. Body unchanged.
pub fn query_parallel_in(pool: &rayon::ThreadPool, r: &ProductIndex, paths: &[String],
                         threads: usize, block: usize) -> Result<Acc, String> {
    if threads <= 1 {
        return query_serial(r, paths);
    }
    if pool.current_num_threads() != threads {
        return Err(format!("pool has {} workers, query asked for {}",
                           pool.current_num_threads(), threads));
    }
    let ns = n_slots(r);
    let mut accs: Vec<Acc> = (0..threads).map(|_| Acc::new(ns)).collect();
    for p in paths {
        for_each_block_pipelined(p, block, |flat, offs| {
            let nr = offs.len() - 1;
            if nr == 0 {
                return;
            }
            let nbytes = offs[nr];
            // Split by BYTES, not by read count: a byte split keeps ragged reads balanced.
            let mut cut: Vec<usize> = Vec::with_capacity(threads + 1);
            cut.push(0);
            for i in 1..threads {
                let want = nbytes * i / threads;
                cut.push(offs.partition_point(|&o| o < want).min(nr));
            }
            cut.push(nr);
            pool.install(|| {
                use rayon::prelude::*;
                accs.par_iter_mut().enumerate().for_each(|(i, a)| {
                    for j in cut[i]..cut[i + 1] {
                        a.push_read(r, &flat[offs[j]..offs[j + 1]]);
                    }
                });
            });
        })?;
    }
    // scalars: exactly the archived order (a zeroed total, then workers 0..T)
    let mut total = Acc::new(0);
    for a in accs.iter() {
        total.merge_scalars(a);
    }
    // per-SLOT arrays: in place into worker 0, each slot adding workers 1..T in order (each
    // anchor's archived merge is exactly its slot's merge)
    let (first, rest) = accs.split_at_mut(1);
    let base = &mut first[0];
    pool.install(|| {
        use rayon::prelude::*;
        const CH: usize = 1 << 16;
        let sr: Vec<&[i64]> = rest.iter().map(|a| &a.raw[..]).collect();
        base.raw.par_chunks_mut(CH).enumerate().for_each(|(ci, ch)| {
            let o = ci * CH;
            let n = ch.len();
            for s in sr.iter() {
                for (x, y) in ch.iter_mut().zip(&s[o..o + n]) {
                    *x += y;
                }
            }
        });
        let sw: Vec<&[f64]> = rest.iter().map(|a| &a.wtd[..]).collect();
        base.wtd.par_chunks_mut(CH).enumerate().for_each(|(ci, ch)| {
            let o = ci * CH;
            let n = ch.len();
            for s in sw.iter() {
                for (x, y) in ch.iter_mut().zip(&s[o..o + n]) {
                    *x += y;
                }
            }
        });
    });
    total.raw = std::mem::take(&mut base.raw);
    total.wtd = std::mem::take(&mut base.wtd);
    drop(accs);
    pool.install(|| expand_slots(r, &mut total, true));
    Ok(total)
}
