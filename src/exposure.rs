//! Reference-only EXPOSURE artifact for the opt-in EM attribution (2026-10-05 handoff).
//!
//! The existing scaled-Poisson abundance EM (benchmarks/typing/models.py, used unchanged by the
//! F4 typing and real-mock paths) models a segment's raw anchor-count total as
//!     Y_j ~ Poisson( sum_s A_js theta_s ),   A_js = sum over segment j's anchors of km[key, s],
//! where km[key, s] is the number of VALID reference positions of genome s (all its records,
//! repeats and non-anchor positions included) carrying that canonical key. This file stores
//! exactly km, per distinct anchor key ("slot", the index's CSR order), sparse over genomes,
//! plus per-genome total bases. A_js is derived from it and the index postings at query time,
//! so nothing is duplicated.
//!
//! BINDING: km depends on WHICH keys are anchors, so it is valid for one index only. The file
//! records the index's container checksum, k/g/c, segment/anchor/slot/genome counts and its
//! full selection array (sampler); the loader refuses any other index -- a FracMinHash
//! exposure file can never be used with a syncmer index or vice versa.
//! OWN-GENOME RULE (checked on load for EVERY anchor): an anchor is eligible only if its key
//! occurs exactly once in its own genome (genome-wide repeat exclusion, R08a, `gmult == 1`), so
//! km[key, own genome] == 1 and the own-genome exposure of a segment equals its eligible-anchor
//! count n_j -- the denominator of the pooled depth.

use crate::product::{ProductIndex, MEMBER_ONLY};
use crate::refindex::Reference;
use crate::ser::{Reader, Writer};

pub const KIND_EXPOSURE: u8 = b'X';
pub const EXPOSURE_VERSION: u64 = 1;
pub const EXPOSURE_ARRAYS: [&str; 5] = ["key_off", "key_genome", "key_mult", "genome_bases",
                                        "selection"];

pub struct Exposures {
    pub k: usize,
    pub g: usize,
    pub c: usize,
    pub n_genomes: usize,
    pub n_segments: usize,
    pub n_anchors: usize,
    pub n_slots: usize,
    pub index_checksum: u64,
    pub selection: Vec<u64>,
    pub key_off: Vec<i64>,
    pub key_genome: Vec<u32>,
    pub key_mult: Vec<u32>,
    pub genome_bases: Vec<i64>,
    pub meta_json: String,
}

/// Distinct anchor key of every slot (slot order = ascending key, as the builder writes it).
pub fn slot_keys(idx: &ProductIndex) -> Result<Vec<u64>, String> {
    let ns = idx.aoff.len().saturating_sub(1);
    let mut v = vec![0u64; ns];
    let mut seen = vec![false; ns];
    for (&q, &c) in idx.sel_keys.iter().zip(&idx.sel_csr) {
        if c != MEMBER_ONLY {
            v[c as usize] = q;
            seen[c as usize] = true;
        }
    }
    if seen.iter().any(|&s| !s) || v.windows(2).any(|w| w[0] >= w[1]) {
        return Err("index slots are not the ascending distinct anchor keys".into());
    }
    Ok(v)
}

fn genome_bases(rec_len: &[i64], rec_genome: &[i64], n_genomes: usize) -> Vec<i64> {
    let mut b = vec![0i64; n_genomes];
    for (&l, &g) in rec_len.iter().zip(rec_genome) {
        b[g as usize] += l;
    }
    b
}

impl Exposures {
    /// km from the reference the index was built from (`r` must be that reference: the caller
    /// either just built both, or has checked the rebuilt product against the index).
    pub fn compute(r: &Reference, idx: &ProductIndex, index_checksum: u64)
        -> Result<Exposures, String> {
        let keys = slot_keys(idx)?;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for rec in 0..r.rec_len.len() {
            let gen = r.rec_genome[rec] as u32;
            let (a, b) = (r.roff[rec] as usize, r.roff[rec + 1] as usize);
            for j in a..b {
                if r.vstart[j] == 0 {
                    continue;
                }
                if let Ok(s) = keys.binary_search(&r.key[j]) {
                    pairs.push((s as u32, gen));
                }
            }
        }
        pairs.sort_unstable();
        let mut key_off = vec![0i64; keys.len() + 1];
        let (mut key_genome, mut key_mult) = (Vec::new(), Vec::new());
        let mut i = 0usize;
        while i < pairs.len() {
            let mut e = i + 1;
            while e < pairs.len() && pairs[e] == pairs[i] {
                e += 1;
            }
            key_off[pairs[i].0 as usize + 1] += 1;
            key_genome.push(pairs[i].1);
            key_mult.push((e - i) as u32);
            i = e;
        }
        for s in 0..keys.len() {
            key_off[s + 1] += key_off[s];
        }
        let x = Exposures {
            k: idx.k, g: idx.g, c: idx.c, n_genomes: idx.n_genomes, n_segments: idx.st.len(),
            n_anchors: idx.anchor_seg.len(), n_slots: keys.len(), index_checksum,
            selection: idx.sel.to_array(), key_off, key_genome, key_mult,
            genome_bases: genome_bases(&idx.rec_len, &idx.rec_genome, idx.n_genomes),
            meta_json: String::new(),
        };
        x.validate(idx, index_checksum, "computed exposures")?;
        Ok(x)
    }

    pub fn save(&self, path: &str, meta_json: &str) -> Result<u64, String> {
        let mut w = Writer::new();
        for v in [EXPOSURE_VERSION, self.k as u64, self.g as u64, self.c as u64,
                  self.n_genomes as u64, self.n_segments as u64, self.n_anchors as u64,
                  self.n_slots as u64, self.index_checksum] {
            w.scalar(v);
        }
        w.meta(meta_json);
        w.strings(&[]);
        w.i64s("key_off", &self.key_off);
        w.u32s("key_genome", &self.key_genome);
        w.u32s("key_mult", &self.key_mult);
        w.i64s("genome_bases", &self.genome_bases);
        w.u64s("selection", &self.selection);
        w.write(path, KIND_EXPOSURE).map_err(|e| format!("{}: {}", path, e))
    }

    /// Load and validate against the index it must belong to (`index_checksum` = the loaded
    /// index's container checksum).
    pub fn load(path: &str, idx: &ProductIndex, index_checksum: u64) -> Result<Exposures, String> {
        let mut r = Reader::open(path, KIND_EXPOSURE, &EXPOSURE_ARRAYS)?;
        let s = &r.scalars;
        if s.len() != 9 || s[0] != EXPOSURE_VERSION {
            return Err(format!("{}: not a version-{} exposure artifact", path, EXPOSURE_VERSION));
        }
        let mut x = Exposures {
            k: s[1] as usize, g: s[2] as usize, c: s[3] as usize, n_genomes: s[4] as usize,
            n_segments: s[5] as usize, n_anchors: s[6] as usize, n_slots: s[7] as usize,
            index_checksum: s[8], selection: Vec::new(), key_off: Vec::new(),
            key_genome: Vec::new(), key_mult: Vec::new(), genome_bases: Vec::new(),
            meta_json: r.meta.clone(),
        };
        x.key_off = r.i64s("key_off").map_err(|e| format!("{}: {}", path, e))?;
        x.key_genome = r.u32s("key_genome").map_err(|e| format!("{}: {}", path, e))?;
        x.key_mult = r.u32s("key_mult").map_err(|e| format!("{}: {}", path, e))?;
        x.genome_bases = r.i64s("genome_bases").map_err(|e| format!("{}: {}", path, e))?;
        x.selection = r.u64s("selection").map_err(|e| format!("{}: {}", path, e))?;
        x.validate(idx, index_checksum, path)?;
        Ok(x)
    }

    pub fn validate(&self, idx: &ProductIndex, index_checksum: u64, path: &str)
        -> Result<(), String> {
        let want_sel = idx.sel.to_array();
        if self.selection != want_sel {
            return Err(format!("{}: exposures were built for sampler {} but the index uses {} \
(each sampler needs its own exposures)", path,
                crate::selection::mode_name(*self.selection.get(1).unwrap_or(&u64::MAX)),
                crate::selection::mode_name(idx.sel.mode)));
        }
        if self.index_checksum != index_checksum {
            return Err(format!("{}: exposures belong to index checksum {:016x}, not to this \
index ({:016x})", path, self.index_checksum, index_checksum));
        }
        let ns = idx.aoff.len().saturating_sub(1);
        if (self.k, self.g, self.c, self.n_genomes, self.n_segments, self.n_anchors, self.n_slots)
            != (idx.k, idx.g, idx.c, idx.n_genomes, idx.st.len(), idx.anchor_seg.len(), ns) {
            return Err(format!("{}: exposure dimensions do not match the index", path));
        }
        let o = &self.key_off;
        if o.len() != ns + 1 || o[0] != 0 || *o.last().unwrap() as usize != self.key_genome.len()
            || self.key_genome.len() != self.key_mult.len() || o.windows(2).any(|w| w[1] <= w[0]) {
            return Err(format!("{}: malformed key-multiplicity CSR (every slot needs >= 1 \
genome)", path));
        }
        for s in 0..ns {
            let (a, b) = (o[s] as usize, o[s + 1] as usize);
            for i in a..b {
                if self.key_genome[i] as usize >= self.n_genomes || self.key_mult[i] == 0
                    || (i > a && self.key_genome[i] <= self.key_genome[i - 1]) {
                    return Err(format!("{}: slot {} has an invalid genome multiplicity entry",
                                       path, s));
                }
            }
        }
        if self.genome_bases != genome_bases(&idx.rec_len, &idx.rec_genome, idx.n_genomes) {
            return Err(format!("{}: genome base totals do not match the index records", path));
        }
        // own-genome rule on EVERY anchor
        for s in 0..ns {
            let (a, b) = (o[s] as usize, o[s + 1] as usize);
            let gens = &self.key_genome[a..b];
            for i in idx.aoff[s] as usize..idx.aoff[s + 1] as usize {
                let an = idx.aidx[i] as usize;
                let seg = idx.anchor_seg[an] as usize;
                let own = idx.rec_genome[idx.seg_rec[seg] as usize] as u32;
                match gens.binary_search(&own) {
                    Ok(p) if self.key_mult[a + p] == 1 => {}
                    _ => return Err(format!("{}: anchor {} (slot {}) does not have multiplicity \
exactly 1 in its own genome {}; the own-genome exposure would not equal the anchor denominator \
(genome-wide repeat exclusion)", path, an, s, own)),
                }
            }
        }
        Ok(())
    }
}
