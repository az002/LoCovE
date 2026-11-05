//! The PRODUCT index: exactly what a timed query has to load, and nothing else.
//!
//! WHY THIS IS SEPARATE FROM THE DENSE REFERENCE. An earlier draft persisted one artifact
//! holding the whole dense reference -- `key`, `fwd`, `vstart`, `abase`, `gmult` at every
//! position, plus a membership set over EVERY valid k-mer. Compiling that and loading it in
//! the timer would have made the product's I/O the oracle's data structure: `fwd` exists only
//! for T2 truth orientation, `abase` only for the R10e base domain, and the unselected 19/20
//! of the membership set can never be looked up because a query key is selected by
//! construction. None of it is wrong; all of it is truth-side, and carrying it would have made
//! the compiled path look expensive for reasons the product does not have.
//!
//! So there are two artifacts:
//!
//!   `.subidx`    THIS one. Selected membership, anchor postings, per-segment domains and the
//!                output coordinates. Loaded by the timed workflow.
//!   `.subdense`  the dense reference (`refindex.rs`). Truth construction, oracle parity and
//!                debugging. NEVER opened by the query path, which the fixtures assert by
//!                watching the files the process actually opens.
//!
//! MEMBERSHIP IS EXACT, not an approximation of the oracle's. S3 asks whether an observed key
//! occurs at any valid reference start. Every observed key is hash-selected, and a selected key
//! occurring at a valid start is by definition in the selected-key set, so for the only keys
//! that can ever be looked up the two sets give the same answer. Repeated and non-interior
//! occurrences are still included -- what is dropped is the UNSELECTED keys, not any occurrence
//! class. `fixtures/test_compiled_oracle.py` asserts the two sets agree on every selected key
//! rather than leaving that as an argument.

use crate::kernel::splitmix;
use crate::ser::{Reader, Writer, KIND_PRODUCT};

/// Open-addressed `u64 -> u32`. One probe per observed occurrence.
///
/// Adapted from the shipped `catalog.rs::U64Map`, for the reason that crate gives: anchor
/// lookup is the query's inner loop and a binary search over a large sorted key array costs a
/// chain of dependent cache misses. The draft this replaces did TWO binary searches per
/// occurrence -- one for membership and one for the postings -- which is the correctness
/// prototype's shape, not the product's.
pub struct KeyIndex {
    mask: usize,
    keys: Vec<u64>,
    vals: Vec<u32>,
}

pub const EMPTY: u32 = u32::MAX;
/// Value stored for a key that IS in the reference but carries no eligible anchor.
pub const MEMBER_ONLY: u32 = u32::MAX - 1;

impl KeyIndex {
    pub fn build(keys: &[u64], vals: &[u32]) -> KeyIndex {
        let mut cap = 16usize;
        while cap < keys.len() * 2 {
            cap <<= 1;
        }
        let mut m = KeyIndex { mask: cap - 1, keys: vec![0; cap], vals: vec![EMPTY; cap] };
        for (i, &k) in keys.iter().enumerate() {
            m.insert(k, vals[i]);
        }
        m
    }
    #[inline(always)]
    fn insert(&mut self, key: u64, val: u32) {
        let mut i = (splitmix(key) as usize) & self.mask;
        loop {
            if self.vals[i] == EMPTY {
                self.keys[i] = key;
                self.vals[i] = val;
                return;
            }
            if self.keys[i] == key {
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
    /// `None` = not in the reference at all (S3 novel). `Some(MEMBER_ONLY)` = present but not
    /// an eligible anchor. `Some(j)` = present, with postings at CSR slot `j`.
    #[inline(always)]
    pub fn get(&self, key: u64) -> Option<u32> {
        let mut i = (splitmix(key) as usize) & self.mask;
        loop {
            let v = self.vals[i];
            if v == EMPTY {
                return None;
            }
            if self.keys[i] == key {
                return Some(v);
            }
            i = (i + 1) & self.mask;
        }
    }
    pub fn slots(&self) -> usize {
        self.keys.len()
    }
}

pub struct ProductIndex {
    pub k: usize,
    pub g: usize,
    pub c: usize,
    pub thresh: u64,
    pub n_genomes: usize,
    /// Selected valid reference keys, sorted and deduplicated (S3 membership).
    pub sel_keys: Vec<u64>,
    /// Per selected key: the CSR slot of its eligible-anchor postings, or `MEMBER_ONLY`.
    pub sel_csr: Vec<u32>,
    /// CSR over `aidx`: postings of distinct anchor key `j` are `aidx[aoff[j]..aoff[j+1]]`.
    pub aoff: Vec<i64>,
    pub aidx: Vec<u32>,
    /// Per anchor (in anchor id order): the segment it reports on.
    pub anchor_seg: Vec<i64>,
    /// Output coordinates and the three approved per-segment weights, PRECOMPUTED so the
    /// timed workflow writes complete segment output without touching a dense array.
    pub st: Vec<i64>,
    pub en: Vec<i64>,
    pub seg_rec: Vec<i64>,
    pub dom_physical: Vec<i64>,
    pub dom_bases: Vec<i64>,
    pub dom_starts: Vec<i64>,
    pub dom_anchors: Vec<i64>,
    pub rec_len: Vec<i64>,
    pub rec_genome: Vec<i64>,
    pub genome_names: Vec<String>,
    pub rec_names: Vec<String>,
    pub meta_json: String,
    pub map: KeyIndex,
    /// syncmer_core: coverage-anchor selection (SHIPPED for original-format products)
    pub sel: crate::selection::Selection,
}

/// syncmer_core candidate products use their own kind byte, so a loader that does not know the
/// selection array refuses them instead of treating their anchors as shipped FracMinHash.
pub const KIND_SYNCMER: u8 = b'Y';
pub const SYNCMER_PRODUCT_ARRAYS: [&str; 15] = [
    "sel_keys", "sel_csr", "aoff", "aidx", "anchor_seg", "st", "en", "seg_rec",
    "dom_physical", "dom_bases", "dom_starts", "dom_anchors", "rec_len", "rec_genome",
    "selection",
];

pub const PRODUCT_ARRAYS: [&str; 14] = [
    "sel_keys", "sel_csr", "aoff", "aidx", "anchor_seg", "st", "en", "seg_rec",
    "dom_physical", "dom_bases", "dom_starts", "dom_anchors", "rec_len", "rec_genome",
];

pub struct LoadReport {
    pub bytes: u64,
    pub read_seconds: f64,
    pub lookup_init_seconds: f64,
    pub checksum: u64,
    pub map_slots: usize,
}

#[allow(clippy::too_many_arguments)]
pub struct ProductParts {
    pub k: usize,
    pub g: usize,
    pub c: usize,
    pub thresh: u64,
    pub n_genomes: usize,
    pub sel_keys: Vec<u64>,
    pub sel_csr: Vec<u32>,
    pub aoff: Vec<i64>,
    pub aidx: Vec<u32>,
    pub anchor_seg: Vec<i64>,
    pub st: Vec<i64>,
    pub en: Vec<i64>,
    pub seg_rec: Vec<i64>,
    pub dom_physical: Vec<i64>,
    pub dom_bases: Vec<i64>,
    pub dom_starts: Vec<i64>,
    pub dom_anchors: Vec<i64>,
    pub rec_len: Vec<i64>,
    pub rec_genome: Vec<i64>,
    pub genome_names: Vec<String>,
    pub rec_names: Vec<String>,
    pub selection: Option<Vec<u64>>,
}

impl ProductIndex {
    pub fn n_segments(&self) -> usize {
        self.st.len()
    }
    pub fn n_anchors(&self) -> usize {
        self.anchor_seg.len()
    }

    pub fn save_parts(p: &ProductParts, path: &str, meta_json: &str) -> std::io::Result<u64> {
        let mut w = Writer::new();
        w.scalar(p.k as u64).scalar(p.g as u64).scalar(p.c as u64).scalar(p.thresh)
            .scalar(p.n_genomes as u64)
            .scalar(p.genome_names.len() as u64)
            .scalar(p.rec_names.len() as u64);
        w.meta(meta_json);
        let mut names = p.genome_names.clone();
        names.extend(p.rec_names.iter().cloned());
        w.strings(&names);
        w.u64s("sel_keys", &p.sel_keys);
        w.u32s("sel_csr", &p.sel_csr);
        w.i64s("aoff", &p.aoff);
        w.u32s("aidx", &p.aidx);
        w.i64s("anchor_seg", &p.anchor_seg);
        w.i64s("st", &p.st);
        w.i64s("en", &p.en);
        w.i64s("seg_rec", &p.seg_rec);
        w.i64s("dom_physical", &p.dom_physical);
        w.i64s("dom_bases", &p.dom_bases);
        w.i64s("dom_starts", &p.dom_starts);
        w.i64s("dom_anchors", &p.dom_anchors);
        w.i64s("rec_len", &p.rec_len);
        w.i64s("rec_genome", &p.rec_genome);
        match &p.selection {
            None => w.write(path, KIND_PRODUCT),
            Some(sel) => {
                w.u64s("selection", sel);
                w.write(path, KIND_SYNCMER)
            }
        }
    }

    pub fn load(path: &str) -> Result<(ProductIndex, LoadReport), String> {
        let t0 = std::time::Instant::now();
        let kind = {
            use std::io::Read;
            let mut h = [0u8; 13];
            std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut h))
                .map_err(|e| format!("{}: {}", path, e))?;
            h[12]
        };
        let mut r = if kind == KIND_SYNCMER {
            Reader::open(path, KIND_SYNCMER, &SYNCMER_PRODUCT_ARRAYS)?
        } else {
            Reader::open(path, KIND_PRODUCT, &PRODUCT_ARRAYS)?
        };
        if r.scalars.len() != 7 {
            return Err(format!("{}: {} scalars, expected 7", path, r.scalars.len()));
        }
        let (k, g, c, thresh, n_genomes, ngn, nrn) =
            (r.scalars[0] as usize, r.scalars[1] as usize, r.scalars[2] as usize,
             r.scalars[3], r.scalars[4] as usize, r.scalars[5] as usize,
             r.scalars[6] as usize);
        // The BUILDER rejects k outside 1..=32, but a checksum-valid file that was never
        // produced by this builder must be rejected too -- a validator that only re-states
        // what its own writer guarantees checks nothing.
        if k == 0 || k > 32 || g == 0 {
            return Err(format!("{}: k={} g={} is not a usable configuration (k must be 1..=32, g positive)", path, k, g));
        }
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
        let idx = ProductIndex {
            k, g, c, thresh, n_genomes,
            sel_keys: r.u64s("sel_keys")?,
            sel_csr: r.u32s("sel_csr")?,
            aoff: r.i64s("aoff")?,
            aidx: r.u32s("aidx")?,
            anchor_seg: r.i64s("anchor_seg")?,
            st: r.i64s("st")?,
            en: r.i64s("en")?,
            seg_rec: r.i64s("seg_rec")?,
            dom_physical: r.i64s("dom_physical")?,
            dom_bases: r.i64s("dom_bases")?,
            dom_starts: r.i64s("dom_starts")?,
            dom_anchors: r.i64s("dom_anchors")?,
            rec_len: r.i64s("rec_len")?,
            rec_genome: r.i64s("rec_genome")?,
            genome_names, rec_names, meta_json,
            map: KeyIndex { mask: 0, keys: Vec::new(), vals: Vec::new() },
            sel: if kind == KIND_SYNCMER {
                let v = r.u64s("selection")?;
                let sel = crate::selection::Selection::from_array(&v)
                    .map_err(|e| format!("{}: {}", path, e))?;
                if sel.is_shipped() || sel.k != k {
                    return Err(format!("{}: candidate product carries an inconsistent selection",
                                       path));
                }
                sel
            } else {
                crate::selection::Selection::shipped(k)
            },
        };
        idx.validate(path)?;
        let read_t = t0.elapsed().as_secs_f64();
        // The one structure BUILT at load rather than stored: the open-addressed key index.
        // It is derived from the persisted sorted arrays in a single linear pass, touches no
        // reference sequence and rebuilds nothing, and its cost is MEASURED and reported as
        // `lookup_init_seconds` rather than folded into the read -- the timing contract needs
        // load accounted for, not merely fast.
        let t1 = std::time::Instant::now();
        let map = KeyIndex::build(&idx.sel_keys, &idx.sel_csr);
        let init_t = t1.elapsed().as_secs_f64();
        let mut idx = idx;
        let slots = map.slots();
        idx.map = map;
        Ok((idx, LoadReport { bytes: r.bytes, read_seconds: read_t,
                              lookup_init_seconds: init_t, checksum: r.checksum,
                              map_slots: slots }))
    }

    /// Structural validation. Each check corresponds to a way a wrong index would otherwise
    /// produce plausible coordinates instead of an error.
    pub(crate) fn validate(&self, path: &str) -> Result<(), String> {   // method1_rust_cli: pub(crate) for load.rs
        let ns = self.st.len();
        for (n, v) in [("en", &self.en), ("seg_rec", &self.seg_rec),
                       ("dom_physical", &self.dom_physical), ("dom_bases", &self.dom_bases),
                       ("dom_starts", &self.dom_starts), ("dom_anchors", &self.dom_anchors)] {
            if v.len() != ns {
                return Err(format!("{}: {} has {} entries for {} segments",
                                   path, n, v.len(), ns));
            }
        }
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
        // Segments must TILE each record with no gap and no overlap, terminals included
        // (R07b). This is the check that would have caught the shipped catalog's dropped head
        // and tail, so it is done on the artifact rather than trusted from the builder.
        let mut seen_bases: Vec<i64> = vec![0; nrec];
        let mut s = 0usize;
        while s < ns {
            let r = self.seg_rec[s];
            if r < 0 || r as usize >= nrec {
                return Err(format!("{}: segment {} names record {}", path, s, r));
            }
            let r = r as usize;
            let mut e = s;
            while e < ns && self.seg_rec[e] == r as i64 {
                e += 1;
            }
            if self.st[s] != 0 || self.en[e - 1] != self.rec_len[r] {
                return Err(format!("{}: record {} segments do not tile [0, {}) -- a terminal \
segment is missing (R07b)", path, r, self.rec_len[r]));
            }
            for j in s..e {
                if self.en[j] <= self.st[j] {
                    return Err(format!("{}: segment {} is empty or inverted", path, j));
                }
                if j + 1 < e && self.en[j] != self.st[j + 1] {
                    return Err(format!("{}: segments {} and {} leave a gap or overlap",
                                       path, j, j + 1));
                }
                if self.dom_physical[j] != self.en[j] - self.st[j] {
                    return Err(format!("{}: segment {} physical length disagrees with its \
coordinates", path, j));
                }
                if self.dom_bases[j] < 0 || self.dom_bases[j] > self.dom_physical[j] {
                    return Err(format!("{}: segment {} scored bases {} exceed its length",
                                       path, j, self.dom_bases[j]));
                }
                // The three approved weights are nested by construction: an eligible anchor
                // sits at a valid start, a valid start begins with an ACGT base, and a scored
                // base lies inside the segment. A stored value that breaks the chain is a
                // fabricated denominator -- and the reducer DIVIDES by two of these -- so it
                // is rejected rather than used.
                if !(0 <= self.dom_anchors[j] && self.dom_anchors[j] <= self.dom_starts[j]
                     && self.dom_starts[j] <= self.dom_bases[j]
                     && self.dom_bases[j] <= self.dom_physical[j]) {
                    return Err(format!("{}: segment {} violates the domain chain 0 <= eligible anchors ({}) <= valid starts ({}) <= scored ACGT bases ({}) <= physical length ({})",
                                       path, j, self.dom_anchors[j], self.dom_starts[j],
                                       self.dom_bases[j], self.dom_physical[j]));
                }
            }
            seen_bases[r] = 1;
            s = e;
        }
        for r in 0..nrec {
            if seen_bases[r] == 0 && self.rec_len[r] > 0 {
                return Err(format!("{}: record {} has {} bases and no segment",
                                   path, r, self.rec_len[r]));
            }
        }
        // Membership and postings
        for w in self.sel_keys.windows(2) {
            if w[0] >= w[1] {
                return Err(format!("{}: the selected-key set is not strictly sorted", path));
            }
        }
        if self.sel_csr.len() != self.sel_keys.len() {
            return Err(format!("{}: sel_csr has {} entries for {} selected keys",
                               path, self.sel_csr.len(), self.sel_keys.len()));
        }
        if self.sel.is_shipped() {
            for &q in &self.sel_keys {
                if q >= self.thresh {
                    return Err(format!("{}: the selected-key set holds an unselected key", path));
                }
            }
        } else {
            // every lookup key must be a novelty-membership key (shipped-selected) or an anchor
            // key that satisfies the stored selection, re-derived from the canonical code
            for (&q, &v) in self.sel_keys.iter().zip(&self.sel_csr) {
                let member = q < self.thresh;
                let anchor_ok = v != MEMBER_ONLY
                    && self.sel.anchor_ok_code(crate::selection::splitmix_inv(q));
                if v != MEMBER_ONLY && !anchor_ok {
                    return Err(format!("{}: anchor key {:016x} does not satisfy the stored selection", path, q));
                }
                if v == MEMBER_ONLY && !member {
                    return Err(format!("{}: a member-only key is not shipped-selected", path));
                }
            }
            if self.sel.realized != self.anchor_seg.len() as u64 {
                return Err(format!("{}: selection records {} anchors, product has {}", path,
                                   self.sel.realized, self.anchor_seg.len()));
            }
        }
        if self.sel_csr.len() != self.sel_keys.len() {
            return Err(format!("{}: sel_csr has {} entries for {} selected keys",
                               path, self.sel_csr.len(), self.sel_keys.len()));
        }
        let na = self.anchor_seg.len();
        if self.aidx.len() != na {
            return Err(format!("{}: aidx has {} entries for {} anchors",
                               path, self.aidx.len(), na));
        }
        let ndist = self.aoff.len().saturating_sub(1);
        if self.aoff.first() != Some(&0) || self.aoff.last() != Some(&(na as i64)) {
            return Err(format!("{}: the anchor CSR does not span [0, {}]", path, na));
        }
        for w in self.aoff.windows(2) {
            if w[1] < w[0] {
                return Err(format!("{}: the anchor CSR is not monotone", path));
            }
        }
        let mut seen = vec![false; na];
        for &i in &self.aidx {
            if i as usize >= na {
                return Err(format!("{}: aidx entry {} is out of range", path, i));
            }
            if seen[i as usize] {
                return Err(format!("{}: aidx is not a permutation (duplicate {})", path, i));
            }
            seen[i as usize] = true;
        }
        for &sg in &self.anchor_seg {
            if sg < 0 || sg as usize >= ns {
                return Err(format!("{}: an anchor names segment {} of {}", path, sg, ns));
            }
        }
        let mut used = vec![false; ndist.max(1)];
        let mut n_anchor_keys = 0usize;
        for &v in &self.sel_csr {
            if v == MEMBER_ONLY {
                continue;
            }
            if v as usize >= ndist {
                return Err(format!("{}: sel_csr slot {} exceeds the {} distinct anchor keys",
                                   path, v, ndist));
            }
            if used[v as usize] {
                return Err(format!("{}: two selected keys claim CSR slot {}", path, v));
            }
            used[v as usize] = true;
            n_anchor_keys += 1;
        }
        if n_anchor_keys != ndist {
            return Err(format!("{}: {} selected keys carry postings but the CSR has {} slots",
                               path, n_anchor_keys, ndist));
        }
        // Per-segment eligible-anchor counts must agree with the postings themselves: the
        // reducer divides by `dom_anchors`, so a stale count would silently rescale depth.
        let mut cnt = vec![0i64; ns];
        for &sg in &self.anchor_seg {
            cnt[sg as usize] += 1;
        }
        for j in 0..ns {
            if cnt[j] != self.dom_anchors[j] {
                return Err(format!("{}: segment {} declares {} eligible anchors but {} point \
at it", path, j, self.dom_anchors[j], cnt[j]));
            }
        }
        Ok(())
    }
}
