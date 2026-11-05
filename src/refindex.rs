//! The reference index: records, terminal-inclusive segmentation, genome-wide multiplicity,
//! approved anchors, and a versioned persisted form that a FRESH PROCESS can load without
//! touching a FASTA.
//!
//! This is the compiled counterpart of `src/sub_ref.py`, rule for rule. The shipped
//! `rustcore/src/catalog.rs` cannot be reused for it, and the four reasons are worth stating
//! because each is a silent wrong answer rather than an error:
//!
//!   R07c  the shipped FASTA reader CONCATENATES every record into one buffer, so a k-mer
//!         window spans the junction between two records that were never adjacent.
//!   R07b  the shipped segmentation emits only the pairs BETWEEN consecutive landmarks, so
//!         the head and tail of every record are silently outside every segment.
//!   I6    the shipped landmark selector runs over the whole hash track, where MASK64
//!         sentinels tie as minima and manufacture cuts inside ambiguous sequence.
//!   R08a  the shipped multiplicity is over the whole reference COLLECTION; the approved rule
//!         is within a genome, across all of that genome's records, with cross-genome sharing
//!         retained rather than collapsed.

use crate::kernel::{forward_codes, hash_track, run_local_minima, threshold, valid_starts, LUT,
                    MASK64};
use crate::product::{ProductParts, MEMBER_ONLY};
use crate::ser::{Reader, Writer, KIND_DENSE};

/// One FASTA record, kept separate. Records are never joined (R07c).
///
/// The record NAME is not carried here: names, genome names and input provenance live in the
/// metadata string the caller supplies and this module stores verbatim. Rust needs only the
/// genome each record belongs to, because that is what multiplicity groups by.
pub struct Record {
    pub genome: usize,
    pub name: String,
    pub seq: Vec<u8>,
}

/// Read one FASTA (optionally gzipped) as SEPARATE records (R07c).
///
/// The shipped `rustcore::io::read_fasta` concatenates every record into one buffer, which
/// silently creates k-mer windows spanning a junction between records that were never
/// adjacent. Nothing downstream can detect that, so the reader is replaced rather than reused.
/// Case is preserved: validity is case-insensitive but the stored sequence is not altered.
///
/// 2026-10-07 (Marine repair): ASCII formatting whitespace (space, tab, CR, VT, FF) inside
/// sequence lines is dropped, as `isspace` in the source-truth engine and htslib faidx do.
/// Keeping it made 617 Marine records one byte longer than their publisher FAI (a trailing
/// space). Every other byte, including N and IUPAC codes, is kept unchanged.
pub fn read_fasta_records(path: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut rd: Box<dyn BufRead> = if path.ends_with(".gz") {
        Box::new(std::io::BufReader::with_capacity(
            1 << 20, flate2::read::MultiGzDecoder::new(f)))
    } else {
        Box::new(std::io::BufReader::with_capacity(1 << 20, f))
    };
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    let mut name: Option<String> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut line: Vec<u8> = Vec::new();
    loop {
        line.clear();
        if rd.read_until(b'\n', &mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        while matches!(line.last(), Some(b'\n') | Some(b'\r')) {
            line.pop();
        }
        if line.first() == Some(&b'>') {
            if let Some(nm) = name.take() {
                out.push((nm, std::mem::take(&mut buf)));
            }
            let hdr = String::from_utf8_lossy(&line[1..]).to_string();
            name = Some(hdr.split_whitespace().next().unwrap_or("").to_string());
        } else if line.iter().any(|&c| is_fasta_space(c)) {
            buf.extend(line.iter().copied().filter(|&c| !is_fasta_space(c)));
        } else {
            buf.extend_from_slice(&line);
        }
    }
    if let Some(nm) = name.take() {
        out.push((nm, buf));
    }
    Ok(out)
}

/// Formatting whitespace in a FASTA sequence line: C `isspace` in the "C" locale.
fn is_fasta_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// `aoff_keys` and the CSR offsets into a key-sorted anchor permutation.
///
/// `aoff[j] .. aoff[j+1]` indexes the anchors carrying distinct key `aoff_keys[j]`; `aoff[0]`
/// is 0 and the final entry is the anchor count. Identical in shape to the oracle's
/// `np.unique(..., return_index=True)` plus a sentinel.
fn csr_from_sorted(anchor_key: &[u64], ord: &[u32]) -> (Vec<u64>, Vec<i64>) {
    let mut keys: Vec<u64> = Vec::new();
    let mut off: Vec<i64> = Vec::new();
    for (pos, &i) in ord.iter().enumerate() {
        let x = anchor_key[i as usize];
        if keys.last() != Some(&x) {
            keys.push(x);
            off.push(pos as i64);
        }
    }
    off.push(ord.len() as i64);
    (keys, off)
}

pub struct Reference {
    pub k: usize,
    pub g: usize,
    pub c: usize,
    pub thresh: u64,
    pub n_genomes: usize,
    /// Names, carried for reporting and export. They are persisted as their own string blob
    /// rather than parsed out of the metadata, so the loader never has to read JSON.
    pub genome_names: Vec<String>,
    pub rec_names: Vec<String>,
    pub rec_len: Vec<i64>,
    pub rec_genome: Vec<i64>,
    /// offsets into the flat per-START arrays / per-BASE arrays / per-SEGMENT arrays
    pub roff: Vec<i64>,
    pub boff: Vec<i64>,
    pub soff: Vec<i64>,
    pub key: Vec<u64>,
    pub fwd: Vec<u64>,
    pub vstart: Vec<u8>,
    pub abase: Vec<u8>,
    pub st: Vec<i64>,
    pub en: Vec<i64>,
    pub seg_rec: Vec<i64>,
    pub gmult: Vec<i32>,
    pub anchor_pos: Vec<i64>,
    pub anchor_seg: Vec<i64>,
    pub anchor_key: Vec<u64>,
    /// Query lookup structures, PERSISTED so a fresh load never re-scans the reference.
    /// `akey_sorted[i]` is `anchor_key[aidx[i]]`; `aoff_keys` are its distinct values and
    /// `aoff` the CSR offsets into `aidx`.
    pub aidx: Vec<u32>,
    pub aoff_keys: Vec<u64>,
    pub aoff: Vec<i64>,
    /// S3 membership: every canonical key at a valid reference start, deduplicated and sorted.
    /// Repeat and non-interior occurrences are included -- absence from the eligible anchors
    /// alone is NOT novelty.
    pub member: Vec<u64>,
    pub meta_json: String,
    /// syncmer_core: the coverage-anchor selection (SHIPPED unless built with another mode)
    pub sel: crate::selection::Selection,
    /// syncmer_core: shipped-rule eligible-anchor count of this catalog (always computed)
    pub shipped_anchor_count: u64,
    /// syncmer_core: (valid starts, eligible starts, candidates before thinning) at build
    pub elig_counts: (u64, u64, u64),
}

fn segment_record(n_bases: i64, cuts: &[i64]) -> (Vec<i64>, Vec<i64>) {
    // Tile the WHOLE record; terminals retained (R07b). A record with no landmark becomes a
    // single segment spanning it.
    let mut edges: Vec<i64> = Vec::with_capacity(cuts.len() + 2);
    edges.push(0);
    for &b in cuts {
        if b > 0 && b < n_bases {
            edges.push(b);
        }
    }
    edges.push(n_bases);
    edges.sort_unstable();
    edges.dedup();
    if edges.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    (edges[..edges.len() - 1].to_vec(), edges[1..].to_vec())
}

/// Boundary BASE coordinates under I6.
pub fn landmark_cuts(h: &[u64], k: usize, g: usize, valid: &[u8]) -> Vec<i64> {
    let nk = h.len();
    let r = std::cmp::max(g / 2, 1);
    let mut out: Vec<i64> = Vec::new();
    if nk == 0 {
        return out;
    }
    let mut i = 0usize;
    while i < nk {
        if valid[i] == 0 {
            i += 1;
            continue;
        }
        let s = i;
        while i < nk && valid[i] != 0 {
            i += 1;
        }
        let e = i;
        if e - s <= 2 * r {
            continue; // too short for the full context: contributes none
        }
        for m in run_local_minima(&h[s..e], r) {
            out.push((s + m + k / 2) as i64);
        }
    }
    out
}

impl Reference {
    /// Build from FASTA paths: one path is one genome, its records kept separate.
    pub fn from_fastas(k: usize, g: usize, c: usize, paths: &[String])
        -> Result<Reference, String> {
        let mut records: Vec<Record> = Vec::new();
        let mut genome_names: Vec<String> = Vec::new();
        for (gi, p) in paths.iter().enumerate() {
            let base = std::path::Path::new(p).file_name()
                .map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|| p.clone());
            let mut gname = base;
            for suf in [".gz", ".fna", ".fa", ".fasta"] {
                if gname.ends_with(suf) {
                    gname.truncate(gname.len() - suf.len());
                }
            }
            genome_names.push(gname);
            for (nm, sq) in read_fasta_records(p)? {
                records.push(Record { genome: gi, name: nm, seq: sq });
            }
        }
        Ok(Reference::build(k, g, c, &records, &genome_names))
    }

    /// syncmer_core: build with a coverage-anchor selection (reference-only calibration).
    pub fn from_fastas_sel(k: usize, g: usize, c: usize, paths: &[String],
                           sel: crate::selection::Selection) -> Result<Reference, String> {
        let mut records: Vec<Record> = Vec::new();
        let mut genome_names: Vec<String> = Vec::new();
        for (gi, p) in paths.iter().enumerate() {
            let base = std::path::Path::new(p).file_name()
                .map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|| p.clone());
            let mut gname = base;
            for suf in [".gz", ".fna", ".fa", ".fasta"] {
                if gname.ends_with(suf) {
                    gname.truncate(gname.len() - suf.len());
                }
            }
            genome_names.push(gname);
            for (nm, sq) in read_fasta_records(p)? {
                records.push(Record { genome: gi, name: nm, seq: sq });
            }
        }
        Reference::build_sel(k, g, c, &records, &genome_names, sel)
    }

    pub fn build(k: usize, g: usize, c: usize, records: &[Record],
                 genome_names: &[String]) -> Reference {
        Reference::build_sel(k, g, c, records, genome_names,
                             crate::selection::Selection::shipped(k))
            .expect("the shipped selection cannot fail")
    }

    pub fn build_sel(k: usize, g: usize, c: usize, records: &[Record],
                     genome_names: &[String], sel: crate::selection::Selection)
        -> Result<Reference, String> {
        let thresh = threshold(c as u64);
        let (mut key, mut fwd, mut vstart, mut abase) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut st, mut en, mut seg_rec) = (Vec::new(), Vec::new(), Vec::new());
        let (mut rec_len, mut rec_genome) = (Vec::new(), Vec::new());
        let (mut roff, mut boff, mut soff) = (vec![0i64], vec![0i64], vec![0i64]);

        for (ri, rec) in records.iter().enumerate() {
            let seq = &rec.seq;
            let n = seq.len() as i64;
            let h = hash_track(seq, k);
            let v: Vec<u8> = valid_starts(seq, k).into_iter().map(|b| b as u8).collect();
            assert_eq!(h.len(), v.len(), "hash track and valid-start lengths disagree");
            // C1: the MASK64 sentinel is CHECKED against the independent ACGT-run mask rather
            // than trusted. They are two computations of the same fact and a disagreement
            // means one of them is wrong.
            for j in 0..h.len() {
                assert_eq!((h[j] != MASK64) as u8, v[j],
                           "MASK64 sentinel disagrees with ACGT-run validity at start {}", j);
            }
            let cuts = landmark_cuts(&h, k, g, &v);
            let (s, e) = segment_record(n, &cuts);
            let f = forward_codes(seq, k);
            let base_ok: Vec<u8> = seq.iter().map(|&b| (LUT[b as usize] >= 0) as u8).collect();

            roff.push(roff[roff.len() - 1] + h.len() as i64);
            boff.push(boff[boff.len() - 1] + n);
            soff.push(soff[soff.len() - 1] + s.len() as i64);
            key.extend_from_slice(&h);
            fwd.extend_from_slice(&f);
            vstart.extend_from_slice(&v);
            abase.extend_from_slice(&base_ok);
            seg_rec.extend(std::iter::repeat(ri as i64).take(s.len()));
            st.extend_from_slice(&s);
            en.extend_from_slice(&e);
            rec_len.push(n);
            rec_genome.push(rec.genome as i64);
        }

        let mut r = Reference {
            k, g, c, thresh, n_genomes: genome_names.len(),
            genome_names: genome_names.to_vec(),
            rec_names: records.iter().map(|x| x.name.clone()).collect(),
            rec_len, rec_genome,
            roff, boff, soff, key, fwd, vstart, abase, st, en, seg_rec,
            gmult: Vec::new(), anchor_pos: Vec::new(), anchor_seg: Vec::new(),
            anchor_key: Vec::new(), aidx: Vec::new(), aoff_keys: Vec::new(),
            aoff: vec![0], member: Vec::new(), meta_json: String::new(),
            sel, shipped_anchor_count: 0, elig_counts: (0, 0, 0),
        };
        r.multiplicity();
        r.anchors();
        r.shipped_anchor_count = r.anchor_key.len() as u64;
        if !r.sel.is_shipped() {
            r.anchors_selected(records)?;
        }
        r.elig_counts = r.eligibility_counts(records);
        r.build_lookup_arrays();
        Ok(r)
    }

    /// Eligible starts in position order (valid, genome multiplicity 1, window inside its own
    /// segment: the UNCHANGED shipped eligibility), restricted to the syncmer filter when the
    /// mode has one, as (flat start j, segment, canonical code).
    fn for_each_candidate<F: FnMut(usize, i64, u64)>(&self, records: &[Record], mut f: F) {
        let k = self.k;
        let (s, n_off, need) = (self.sel.s, self.sel.n_off, self.sel.needs_syncmer());
        let mut hs: Vec<u64> = Vec::new();
        for r in 0..self.rec_len.len() {
            let (a, b) = (self.roff[r] as usize, self.roff[r + 1] as usize);
            if b <= a {
                continue;
            }
            if need {
                crate::selection::smer_track(&records[r].seq, s, &mut hs);
                assert_eq!(hs.len(), b - a + n_off - 1, "s-mer track length");
            }
            for p in 0..(b - a) {
                let j = a + p;
                if self.vstart[j] == 0 || self.gmult[j] != 1 {
                    continue;
                }
                let sg = self.seg_of_start(r, p as i64);
                if sg < self.soff[r] || (p as i64) + k as i64 > self.en[sg as usize] {
                    continue;
                }
                if need && !crate::selection::centre_is_min(&hs[p..p + n_off], self.sel.t) {
                    continue;
                }
                let fw = self.fwd[j];
                let rc = crate::selection::revcomp(fw, k);
                f(j, sg, if fw <= rc { fw } else { rc });
            }
        }
    }

    /// syncmer_core diagnostics (reference-only): valid starts, eligible starts (shipped
    /// eligibility), and candidates before thinning (eligible AND syncmer when the mode has a
    /// syncmer filter; equal to eligible otherwise).
    pub fn eligibility_counts(&self, records: &[Record]) -> (u64, u64, u64) {
        let valid = self.vstart.iter().filter(|&&v| v != 0).count() as u64;
        let mut elig = 0u64;
        for r in 0..self.rec_len.len() {
            let (a, b) = (self.roff[r] as usize, self.roff[r + 1] as usize);
            for p in 0..b.saturating_sub(a) {
                let j = a + p;
                if self.vstart[j] == 0 || self.gmult[j] != 1 {
                    continue;
                }
                let sg = self.seg_of_start(r, p as i64);
                if sg >= self.soff[r] && (p as i64) + self.k as i64 <= self.en[sg as usize] {
                    elig += 1;
                }
            }
        }
        let mut cand = 0u64;
        if !self.sel.is_shipped() {
            self.for_each_candidate(records, |_, _, _| cand += 1);
        } else {
            cand = elig;
        }
        (valid, elig, cand)
    }

    /// syncmer_core: replace the shipped anchors by the configured selection. Count modes pick
    /// the per-catalog INCLUSIVE fmix64 threshold giving the largest final eligible-anchor count
    /// not exceeding the shipped count; keys enter whole (equal fmix64 <=> equal canonical
    /// k-mer). Two-pass bucketed search, so nothing proportional to the catalog is sorted.
    fn anchors_selected(&mut self, records: &[Record]) -> Result<(), String> {
        use crate::selection::{fmix64, MODE_INDHASH_COUNT, MODE_SYNCMER_COUNT, MODE_SYNCMER_NOMINAL};
        let target = self.shipped_anchor_count;
        let mode = self.sel.mode;
        if mode == MODE_SYNCMER_NOMINAL {
            self.sel.set_nominal(self.c as u64)?;
            self.sel.target = target;
        } else if mode == MODE_SYNCMER_COUNT || mode == MODE_INDHASH_COUNT {
            const B: u32 = 20;
            let mut hist = vec![0u64; 1 << B];
            let mut total = 0u64;
            self.for_each_candidate(records, |_, _, canon| {
                hist[(fmix64(canon) >> (64 - B)) as usize] += 1;
                total += 1;
            });
            if total <= target {
                self.sel.thin = u64::MAX;
                self.sel.next_group_anchors = 0;
            } else {
                let mut cum = 0u64;
                let mut bstar = 0usize;
                while cum + hist[bstar] <= target {
                    cum += hist[bstar];
                    bstar += 1;
                }
                let mut vals: Vec<u64> = Vec::new();
                let mut maxprev: Option<u64> = None;
                self.for_each_candidate(records, |_, _, canon| {
                    let h = fmix64(canon);
                    let bk = (h >> (64 - B)) as usize;
                    if bk == bstar {
                        vals.push(h);
                    } else if bk < bstar {
                        maxprev = Some(maxprev.map_or(h, |m: u64| m.max(h)));
                    }
                });
                vals.sort_unstable();
                // all of bucket < bstar is in; the frozen threshold is TIGHT: the largest
                // included fmix64 value (identical selection to any looser threshold below the
                // next candidate, but one canonical number per catalog)
                let mut thin = maxprev;
                let mut i = 0usize;
                let mut next = 0u64;
                while i < vals.len() {
                    let mut e = i + 1;
                    while e < vals.len() && vals[e] == vals[i] {
                        e += 1;
                    }
                    let gsz = (e - i) as u64;
                    if cum + gsz > target {
                        next = gsz;
                        break;
                    }
                    cum += gsz;
                    thin = Some(vals[i]);
                    i = e;
                }
                self.sel.thin = match thin {
                    Some(t) => t,
                    None => return Err("count calibration: the first key group alone exceeds the target; no threshold can realize a nonzero count".to_string()),
                };
                self.sel.next_group_anchors = next;
            }
            self.sel.target = target;
        } else {
            return Err(format!("selection mode {} has no anchor rule", mode));
        }
        let (mut ap, mut asg, mut ak) = (Vec::new(), Vec::new(), Vec::new());
        let thin = self.sel.thin;
        self.for_each_candidate(records, |j, sg, canon| {
            if fmix64(canon) <= thin {
                ap.push(j as i64);
                asg.push(sg);
                ak.push(crate::kernel::splitmix(canon));
            }
        });
        for (&j, &q) in ap.iter().zip(&ak) {
            assert_eq!(self.key[j as usize], q, "anchor key disagrees with the hash track");
        }
        self.anchor_pos = ap;
        self.anchor_seg = asg;
        self.anchor_key = ak;
        self.sel.realized = self.anchor_key.len() as u64;
        if (mode == MODE_SYNCMER_COUNT || mode == MODE_INDHASH_COUNT) && self.sel.realized > target {
            return Err("count calibration exceeded its target".to_string());
        }
        Ok(())
    }

    /// Genome-wide multiplicity over ALL valid starts of ALL records of the genome (R08a).
    ///
    /// Within a genome, never per record: a k-mer repeated across two contigs of the same
    /// assembly is a repeat. Across genomes it is SHARING and is retained, which is why the
    /// pool is rebuilt per genome instead of once over everything.
    fn multiplicity(&mut self) {
        self.gmult = vec![0i32; self.key.len()];
        for gi in 0..self.n_genomes {
            let recs: Vec<usize> = (0..self.rec_len.len())
                .filter(|&r| self.rec_genome[r] == gi as i64)
                .collect();
            let mut pool: Vec<u64> = Vec::new();
            for &r in &recs {
                let (a, b) = (self.roff[r] as usize, self.roff[r + 1] as usize);
                for j in a..b {
                    if self.vstart[j] != 0 {
                        pool.push(self.key[j]);
                    }
                }
            }
            pool.sort_unstable();
            for &r in &recs {
                let (a, b) = (self.roff[r] as usize, self.roff[r + 1] as usize);
                for j in a..b {
                    if self.vstart[j] == 0 {
                        continue;
                    }
                    let q = self.key[j];
                    let lo = pool.partition_point(|&x| x < q);
                    let hi = pool.partition_point(|&x| x <= q);
                    self.gmult[j] = (hi - lo) as i32;
                }
            }
        }
    }

    /// Segment owning a start, by START position (R10b). `pos` is record-local.
    pub fn seg_of_start(&self, rec: usize, pos: i64) -> i64 {
        let (s0, s1) = (self.soff[rec] as usize, self.soff[rec + 1] as usize);
        let j = self.st[s0..s1].partition_point(|&x| x <= pos);
        s0 as i64 + j as i64 - 1
    }

    /// Estimator anchors: hash-selected, valid, genome-unique (R08a), fully internal (R08b).
    fn anchors(&mut self) {
        let (mut ap, mut asg, mut ak) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..self.rec_len.len() {
            let (a, b) = (self.roff[r] as usize, self.roff[r + 1] as usize);
            if b <= a {
                continue;
            }
            for p in 0..(b - a) {
                let j = a + p;
                if self.vstart[j] == 0 || self.key[j] >= self.thresh || self.gmult[j] != 1 {
                    continue;
                }
                let sg = self.seg_of_start(r, p as i64);
                if sg < self.soff[r] {
                    continue;
                }
                if (p as i64) + self.k as i64 > self.en[sg as usize] {
                    continue; // R08b: the anchor must lie wholly inside its own segment
                }
                ap.push(j as i64);
                asg.push(sg);
                ak.push(self.key[j]);
            }
        }
        self.anchor_pos = ap;
        self.anchor_seg = asg;
        self.anchor_key = ak;
    }

    /// The query-side lookup arrays, built once here and PERSISTED.
    ///
    /// A fresh process must not have to re-scan the reference to query it, and it must not
    /// silently rebuild what the index was supposed to contain. These are the two structures
    /// the query needs: the CSR from a distinct anchor key to the anchors carrying it, and the
    /// sorted S3 membership set.
    fn build_lookup_arrays(&mut self) {
        let mut ord: Vec<u32> = (0..self.anchor_key.len() as u32).collect();
        // STABLE, matching the oracle's `argsort(kind='stable')`, so `aidx` is identical and
        // the per-anchor accumulation order is identical -- which is what makes the weighted
        // f64 channel bit-comparable to the oracle, not merely close.
        ord.sort_by_key(|&i| self.anchor_key[i as usize]);
        let (keys, off) = csr_from_sorted(&self.anchor_key, &ord);
        self.aidx = ord;
        self.aoff_keys = keys;
        self.aoff = off;

        let mut m: Vec<u64> = Vec::new();
        for j in 0..self.key.len() {
            if self.vstart[j] != 0 {
                m.push(self.key[j]);
            }
        }
        m.sort_unstable();
        m.dedup();
        self.member = m;
    }

    pub fn n_segments(&self) -> usize {
        self.st.len()
    }
}

// ------------------------------------------------------------------ domains and export

/// The three approved per-segment weights plus the eligible-anchor count.
///
/// They are kept SEPARATE on purpose (S2/R10e): base breadth and depth weight by scored ACGT
/// bases, full-k-mer breadth by dense valid-start count, and physical length is neither. The
/// product index stores all four precomputed so the timed workflow can write complete segment
/// output without opening a dense array.
pub struct Domains {
    pub physical: Vec<i64>,
    pub bases: Vec<i64>,
    pub starts: Vec<i64>,
    pub anchors: Vec<i64>,
}

impl Reference {
    pub fn domains(&self) -> Domains {
        let ns = self.st.len();
        let mut bases = vec![0i64; ns];
        let mut starts = vec![0i64; ns];
        for r in 0..self.rec_len.len() {
            let (s0, s1) = (self.soff[r] as usize, self.soff[r + 1] as usize);
            if s1 <= s0 {
                continue;
            }
            let (ab, bb) = (self.boff[r] as usize, self.boff[r + 1] as usize);
            let nb = bb - ab;
            let mut cs = vec![0i64; nb + 1];
            for j in 0..nb {
                cs[j + 1] = cs[j] + self.abase[ab + j] as i64;
            }
            let (a, b) = (self.roff[r] as usize, self.roff[r + 1] as usize);
            let nv = b - a;
            let mut cv = vec![0i64; nv + 1];
            for j in 0..nv {
                cv[j + 1] = cv[j] + self.vstart[a + j] as i64;
            }
            for s in s0..s1 {
                let (lo, hi) = (self.st[s] as usize, self.en[s] as usize);
                bases[s] = cs[hi.min(nb)] - cs[lo.min(nb)];
                starts[s] = cv[hi.min(nv)] - cv[lo.min(nv)];
            }
        }
        let mut anchors = vec![0i64; ns];
        for &sg in &self.anchor_seg {
            anchors[sg as usize] += 1;
        }
        let physical: Vec<i64> = (0..ns).map(|s| self.en[s] - self.st[s]).collect();
        Domains { physical, bases, starts, anchors }
    }

    /// Selected valid keys, sorted and deduplicated -- the product's S3 membership set.
    ///
    /// EXACT for every key a query can present: an observed key is hash-selected, so it lies in
    /// the reference's full valid-key set if and only if it lies in this selected subset.
    /// Repeats and non-interior occurrences are included; what is dropped is the unselected
    /// keys, which no query can look up.
    pub fn selected_keys(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.key.iter().enumerate()
            .filter(|(j, &q)| self.vstart[*j] != 0 && q < self.thresh)
            .map(|(_, &q)| q)
            .collect();
        // syncmer_core: the lookup set is the NOVELTY membership set (shipped-selected valid
        // keys, repeats included) UNION the anchor keys of the configured selection. For a
        // shipped-selected read window, presence in the union equals presence in membership
        // (every union key is a valid reference key); anchor keys add the new postings.
        if !self.sel.is_shipped() {
            v.extend_from_slice(&self.anchor_key);
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The full valid-key membership set the oracle uses. Truth/debug only; never persisted in
    /// the product artifact. Kept so a fixture can assert the two agree on selected keys.
    pub fn member_all(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.key.iter().enumerate()
            .filter(|(j, _)| self.vstart[*j] != 0)
            .map(|(_, &q)| q)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Everything the timed query needs, and nothing that only truth needs.
    pub fn to_product_parts(&self) -> ProductParts {
        let dom = self.domains();
        let sel_keys = self.selected_keys();
        // Anchor postings in key-sorted order, exactly the oracle's `argsort(kind='stable')`
        // permutation, so the per-anchor accumulation order -- and therefore the f64 weighted
        // channel -- is identical rather than merely close.
        let mut ord: Vec<u32> = (0..self.anchor_key.len() as u32).collect();
        ord.sort_by_key(|&i| self.anchor_key[i as usize]);
        let (akeys, aoff) = csr_from_sorted(&self.anchor_key, &ord);
        let mut sel_csr = vec![MEMBER_ONLY; sel_keys.len()];
        for (j, q) in akeys.iter().enumerate() {
            let i = sel_keys.binary_search(q)
                .expect("an eligible anchor key is always a selected valid key");
            sel_csr[i] = j as u32;
        }
        ProductParts {
            k: self.k, g: self.g, c: self.c, thresh: self.thresh,
            n_genomes: self.n_genomes,
            sel_keys, sel_csr, aoff, aidx: ord,
            anchor_seg: self.anchor_seg.clone(),
            st: self.st.clone(), en: self.en.clone(), seg_rec: self.seg_rec.clone(),
            dom_physical: dom.physical, dom_bases: dom.bases, dom_starts: dom.starts,
            dom_anchors: dom.anchors,
            rec_len: self.rec_len.clone(), rec_genome: self.rec_genome.clone(),
            genome_names: self.genome_names.clone(), rec_names: self.rec_names.clone(),
            selection: if self.sel.is_shipped() { None } else { Some(self.sel.to_array()) },
        }
    }
}

// ------------------------------------------------------------------ dense persistence
//
// The DENSE artifact: truth construction, oracle parity and debugging. It is deliberately a
// different `kind` byte from the product index, so asking for one and being handed the other
// is an error rather than a misread.

pub const DENSE_ARRAYS: [&str; 16] = [
    "key", "fwd", "vstart", "abase", "st", "en", "seg_rec", "rec_len", "rec_genome",
    "roff", "boff", "soff", "gmult", "anchor_pos", "anchor_seg", "anchor_key",
];

impl Reference {
    pub fn save_dense(&self, path: &str, meta_json: &str) -> std::io::Result<u64> {
        let mut w = Writer::new();
        w.scalar(self.k as u64).scalar(self.g as u64).scalar(self.c as u64)
            .scalar(self.thresh).scalar(self.n_genomes as u64)
            .scalar(self.genome_names.len() as u64).scalar(self.rec_names.len() as u64);
        w.meta(meta_json);
        let mut names = self.genome_names.clone();
        names.extend(self.rec_names.iter().cloned());
        w.strings(&names);
        w.u64s("key", &self.key);
        w.u64s("fwd", &self.fwd);
        w.u8s("vstart", &self.vstart);
        w.u8s("abase", &self.abase);
        w.i64s("st", &self.st);
        w.i64s("en", &self.en);
        w.i64s("seg_rec", &self.seg_rec);
        w.i64s("rec_len", &self.rec_len);
        w.i64s("rec_genome", &self.rec_genome);
        w.i64s("roff", &self.roff);
        w.i64s("boff", &self.boff);
        w.i64s("soff", &self.soff);
        w.i32s("gmult", &self.gmult);
        w.i64s("anchor_pos", &self.anchor_pos);
        w.i64s("anchor_seg", &self.anchor_seg);
        w.u64s("anchor_key", &self.anchor_key);
        w.write(path, KIND_DENSE)
    }

    pub fn load_dense(path: &str) -> Result<Reference, String> {
        let mut r = Reader::open(path, KIND_DENSE, &DENSE_ARRAYS)?;
        if r.scalars.len() != 7 {
            return Err(format!("{}: {} scalars, expected 7", path, r.scalars.len()));
        }
        let (k, g, c, thresh, n_genomes, ngn, nrn) =
            (r.scalars[0] as usize, r.scalars[1] as usize, r.scalars[2] as usize,
             r.scalars[3], r.scalars[4] as usize, r.scalars[5] as usize,
             r.scalars[6] as usize);
        if thresh != crate::kernel::threshold(c as u64) {
            return Err(format!("{}: stored threshold {} does not match c={}", path, thresh, c));
        }
        if r.strings.len() != ngn + nrn || ngn != n_genomes {
            return Err(format!("{}: {} names for {} genomes and {} records",
                               path, r.strings.len(), ngn, nrn));
        }
        let genome_names = r.strings[..ngn].to_vec();
        let rec_names = r.strings[ngn..].to_vec();
        let meta_json = r.meta.clone();
        let mut o = Reference {
            k, g, c, thresh, n_genomes, genome_names, rec_names,
            key: r.u64s("key")?,
            fwd: r.u64s("fwd")?,
            vstart: r.u8s("vstart")?,
            abase: r.u8s("abase")?,
            st: r.i64s("st")?,
            en: r.i64s("en")?,
            seg_rec: r.i64s("seg_rec")?,
            rec_len: r.i64s("rec_len")?,
            rec_genome: r.i64s("rec_genome")?,
            roff: r.i64s("roff")?,
            boff: r.i64s("boff")?,
            soff: r.i64s("soff")?,
            gmult: r.i32s("gmult")?,
            anchor_pos: r.i64s("anchor_pos")?,
            anchor_seg: r.i64s("anchor_seg")?,
            anchor_key: r.u64s("anchor_key")?,
            aidx: Vec::new(), aoff_keys: Vec::new(), aoff: vec![0],
            member: Vec::new(), meta_json,
            sel: crate::selection::Selection::shipped(k), shipped_anchor_count: 0,
            elig_counts: (0, 0, 0),
        };
        o.validate(path)?;
        o.build_lookup_arrays();
        Ok(o)
    }

    /// Structural validation of the dense form. Each check corresponds to a way a wrong index
    /// would otherwise produce plausible coordinates instead of an error.
    pub fn validate(&self, path: &str) -> Result<(), String> {
        let nrec = self.rec_len.len();
        if self.rec_genome.len() != nrec || self.rec_names.len() != nrec {
            return Err(format!("{}: record arrays disagree in length", path));
        }
        for &gi in &self.rec_genome {
            if gi < 0 || gi as usize >= self.n_genomes {
                return Err(format!("{}: a record names genome {} of {}",
                                   path, gi, self.n_genomes));
            }
        }
        for (name, off, total) in [("roff", &self.roff, self.key.len()),
                                   ("boff", &self.boff, self.abase.len()),
                                   ("soff", &self.soff, self.st.len())] {
            if off.len() != nrec + 1 {
                return Err(format!("{}: {} has {} entries for {} records",
                                   path, name, off.len(), nrec));
            }
            if off[0] != 0 {
                return Err(format!("{}: {} does not start at 0", path, name));
            }
            for w in off.windows(2) {
                if w[1] < w[0] {
                    return Err(format!("{}: {} is not monotone", path, name));
                }
            }
            if off[nrec] != total as i64 {
                return Err(format!("{}: {} ends at {} but the array holds {} elements",
                                   path, name, off[nrec], total));
            }
        }
        if self.fwd.len() != self.key.len() || self.vstart.len() != self.key.len()
            || self.gmult.len() != self.key.len() {
            return Err(format!("{}: per-start arrays disagree in length", path));
        }
        if self.en.len() != self.st.len() || self.seg_rec.len() != self.st.len() {
            return Err(format!("{}: per-segment arrays disagree in length", path));
        }
        for r in 0..nrec {
            if self.boff[r + 1] - self.boff[r] != self.rec_len[r] {
                return Err(format!("{}: record {} declares length {} but spans {} bases",
                                   path, r, self.rec_len[r], self.boff[r + 1] - self.boff[r]));
            }
            let want = if self.rec_len[r] >= self.k as i64 {
                self.rec_len[r] - self.k as i64 + 1
            } else {
                0
            };
            if self.roff[r + 1] - self.roff[r] != want {
                return Err(format!("{}: record {} has {} starts, expected {} at k={}",
                                   path, r, self.roff[r + 1] - self.roff[r], want, self.k));
            }
            let (s0, s1) = (self.soff[r] as usize, self.soff[r + 1] as usize);
            if s1 > s0 {
                if self.st[s0] != 0 || self.en[s1 - 1] != self.rec_len[r] {
                    return Err(format!("{}: record {} segments do not tile [0, {}) -- a \
terminal segment is missing (R07b)", path, r, self.rec_len[r]));
                }
                for s in s0..s1 {
                    if self.en[s] <= self.st[s] {
                        return Err(format!("{}: segment {} is empty or inverted", path, s));
                    }
                    if s + 1 < s1 && self.en[s] != self.st[s + 1] {
                        return Err(format!("{}: segments {} and {} leave a gap or overlap",
                                           path, s, s + 1));
                    }
                    if self.seg_rec[s] != r as i64 {
                        return Err(format!("{}: segment {} claims record {} but lies in {}",
                                           path, s, self.seg_rec[s], r));
                    }
                }
            } else if self.rec_len[r] > 0 {
                return Err(format!("{}: record {} has {} bases and no segment",
                                   path, r, self.rec_len[r]));
            }
        }
        let na = self.anchor_key.len();
        if self.anchor_pos.len() != na || self.anchor_seg.len() != na {
            return Err(format!("{}: anchor arrays disagree in length", path));
        }
        for i in 0..na {
            let p = self.anchor_pos[i];
            if p < 0 || p >= self.key.len() as i64 {
                return Err(format!("{}: anchor {} points outside the start arrays", path, i));
            }
            let p = p as usize;
            if self.anchor_key[i] != self.key[p] {
                return Err(format!("{}: anchor {} key does not match the position it names",
                                   path, i));
            }
            if self.anchor_key[i] >= self.thresh {
                return Err(format!("{}: anchor {} is not hash-selected", path, i));
            }
            if self.vstart[p] == 0 {
                return Err(format!("{}: anchor {} sits on an invalid start", path, i));
            }
            if self.gmult[p] != 1 {
                return Err(format!("{}: anchor {} is not genome-unique (R08a)", path, i));
            }
            let sg = self.anchor_seg[i];
            if sg < 0 || sg as usize >= self.st.len() {
                return Err(format!("{}: anchor {} names segment {}", path, i, sg));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod fasta_tests {
    use super::read_fasta_records;

    fn parse(name: &str, bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
        let p = std::env::temp_dir().join(format!("m1c-fasta-{}-{}.fa", name, std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        let r = read_fasta_records(p.to_str().unwrap()).unwrap();
        std::fs::remove_file(&p).unwrap();
        r
    }

    fn rec(n: &str, s: &str) -> (String, Vec<u8>) {
        (n.to_string(), s.as_bytes().to_vec())
    }

    #[test]
    fn trailing_space_like_marine() {
        assert_eq!(parse("ts", b">CP011806.1 desc\nACGTN \n"), vec![rec("CP011806.1", "ACGTN")]);
    }

    #[test]
    fn spaces_and_tabs_anywhere_in_sequence_lines() {
        assert_eq!(parse("st", b">a\n AC GT\t\nNN\t\x0b\x0cRY \n"), vec![rec("a", "ACGTNNRY")]);
    }

    #[test]
    fn lf_and_crlf_are_identical() {
        let lf = parse("lf", b">a x\nACGT\nacgn\n>b\nTTTT\n");
        let crlf = parse("crlf", b">a x\r\nACGT\r\nacgn\r\n>b\r\nTTTT\r\n");
        assert_eq!(lf, vec![rec("a", "ACGTacgn"), rec("b", "TTTT")]);
        assert_eq!(lf, crlf);
    }

    #[test]
    fn multi_record_boundaries_and_blank_lines() {
        assert_eq!(parse("mr", b">r1\nAC\n\n  \nGT\n>r2 two\n\n>r3\nG"),
                   vec![rec("r1", "ACGT"), rec("r2", ""), rec("r3", "G")]);
    }

    #[test]
    fn ambiguous_bases_and_case_are_kept() {
        let s = "NNNNacgtRYKMSWBDHVN-*";
        assert_eq!(parse("amb", format!(">x\n{}\n", s).as_bytes()), vec![rec("x", s)]);
    }
}
