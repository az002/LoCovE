//! COVERAGE-ANCHOR SELECTION (HANDOFF 2026-10-02, open-syncmer + FracMinHash experiment).
//!
//! Only WHICH reference positions become coverage anchors changes. Segment boundaries (the
//! shipped splitmix64 hash track), repeat eligibility (genome multiplicity 1, window inside its
//! own segment), exact canonical key matching, the correction variants and the NOVELTY sampler
//! (read windows with splitmix64(canonical) < 2^64/c, judged against full-reference membership)
//! are untouched. Every predicate below is a function of the CANONICAL k-mer alone, so a key is
//! selected identically in every record, genome, reference and read.
//!
//!   SHIPPED            anchor iff splitmix64(canonical) < 2^64/c (the incumbent; the code path
//!                      is unchanged and a product of this mode is the original format).
//!   SYNCMER_COUNT      open syncmer (below) AND fmix64(canonical) <= T, with T the per-catalog
//!                      reference-only threshold giving the LARGEST final eligible-anchor count
//!                      not exceeding the catalog's SHIPPED eligible-anchor count (EXPERIMENT).
//!   INDHASH_COUNT      fmix64(canonical) <= T, same count rule, no syncmer filter: FracMinHash
//!                      on an independent hash, the matched-count control (EXPERIMENT).
//!   SYNCMER_NOMINAL    open syncmer AND fmix64(canonical) < q * 2^64, q = n_off / c fixed for
//!                      every catalog (the approved PRODUCTION policy; infeasible q > 1 refused).
//!
//! OPEN SYNCMER (approved: s = 9, central zero-based offset t = 6 of the n_off = k - s + 1 = 13
//! s-mer offsets of the canonical k = 21-mer): the s-mer order is splitmix64(canonical s-mer
//! code); the k-mer is selected iff the s-mer at offset t attains the MINIMUM over all n_off
//! offsets (ties: selected when the centre is AMONG the minima). Reverse complement maps offset
//! j to n_off-1-j and fixes the centre, and the canonical s-mer hash is strand-invariant, so the
//! predicate is the same on either strand. FMIX64 is the MurmurHash3 64-bit finalizer of the
//! canonical k-mer code: a different function from the shipped splitmix64 key.

use crate::kernel::{splitmix, LUT, MASK64};

pub const MODE_SHIPPED: u64 = 0;
pub const MODE_SYNCMER_COUNT: u64 = 1;
pub const MODE_INDHASH_COUNT: u64 = 2;
pub const MODE_SYNCMER_NOMINAL: u64 = 3;
pub const SELECTION_VERSION: u64 = 1;

pub fn mode_name(m: u64) -> &'static str {
    match m {
        MODE_SHIPPED => "shipped_fracminhash",
        MODE_SYNCMER_COUNT => "experiment_syncmer_s9t6_countmatched",
        MODE_INDHASH_COUNT => "experiment_indhash_fracminhash_countmatched",
        MODE_SYNCMER_NOMINAL => "production_syncmer_s9t6_nominal",
        _ => "unknown",
    }
}

pub fn mode_from_name(s: &str) -> Option<u64> {
    for m in [MODE_SHIPPED, MODE_SYNCMER_COUNT, MODE_INDHASH_COUNT, MODE_SYNCMER_NOMINAL] {
        if mode_name(m) == s {
            return Some(m);
        }
    }
    None
}

/// MurmurHash3 fmix64.
#[inline(always)]
pub fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// Inverse of `kernel::splitmix`: recovers the canonical k-mer code from a stored key, so a
/// product's anchor keys can be re-checked against the selection predicate at load.
pub fn splitmix_inv(mut z: u64) -> u64 {
    fn unxorshift(y: u64, s: u32) -> u64 {
        // y = x ^ (x >> s)  =>  x is the fixed point of x = y ^ (x >> s)
        let mut x = y;
        for _ in 0..(64 / s + 1) {
            x = y ^ (x >> s);
        }
        x
    }
    fn inv_mul(a: u64) -> u64 {
        // Newton iteration for the multiplicative inverse modulo 2^64 (a odd)
        let mut x = a;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        }
        x
    }
    z = unxorshift(z, 31);
    z = z.wrapping_mul(inv_mul(0x94D049BB133111EB));
    z = unxorshift(z, 27);
    z = z.wrapping_mul(inv_mul(0xBF58476D1CE4E5B9));
    z = unxorshift(z, 30);
    z.wrapping_sub(0x9E3779B97F4A7C15)
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub mode: u64,
    pub k: usize,
    pub s: usize,
    pub t: usize,
    pub n_off: usize,
    /// INCLUSIVE second-hash threshold: anchor requires fmix64(canonical) <= thin
    pub thin: u64,
    /// count rule: the shipped eligible-anchor count of the same catalog (0 otherwise)
    pub target: u64,
    /// realized final eligible anchors
    pub realized: u64,
    /// distinct keys / anchors in the boundary group that did NOT fit (residual diagnostics)
    pub next_group_anchors: u64,
    /// nominal q as a rational n_off / c (production mode); 0 otherwise
    pub q_num: u64,
    pub q_den: u64,
}

impl Selection {
    pub fn shipped(k: usize) -> Selection {
        Selection { mode: MODE_SHIPPED, k, s: 0, t: 0, n_off: 0, thin: 0, target: 0, realized: 0,
                    next_group_anchors: 0, q_num: 0, q_den: 0 }
    }
    pub fn new(mode: u64, k: usize, s: usize, t: usize) -> Result<Selection, String> {
        if mode == MODE_SHIPPED {
            return Ok(Selection::shipped(k));
        }
        if mode > MODE_SYNCMER_NOMINAL {
            return Err(format!("unknown selection mode {}", mode));
        }
        let syn = mode == MODE_SYNCMER_COUNT || mode == MODE_SYNCMER_NOMINAL;
        if syn {
            if s == 0 || s >= k || s > 32 || s % 2 == 0 {
                return Err(format!("syncmer s={} must be odd and in 1..k (k={})", s, k));
            }
            let n_off = k - s + 1;
            if t >= n_off {
                return Err(format!("syncmer offset t={} outside 0..{}", t, n_off));
            }
            if 2 * t + 1 != n_off {
                return Err(format!("t={} is not the central offset of n_off={} (strand symmetry \
requires the centre)", t, n_off));
            }
            Ok(Selection { mode, k, s, t, n_off, thin: 0, target: 0, realized: 0,
                           next_group_anchors: 0, q_num: 0, q_den: 0 })
        } else {
            Ok(Selection { mode, k, s: 0, t: 0, n_off: 0, thin: 0, target: 0, realized: 0,
                           next_group_anchors: 0, q_num: 0, q_den: 0 })
        }
    }
    #[inline(always)]
    pub fn is_shipped(&self) -> bool {
        self.mode == MODE_SHIPPED
    }
    #[inline(always)]
    pub fn needs_syncmer(&self) -> bool {
        self.mode == MODE_SYNCMER_COUNT || self.mode == MODE_SYNCMER_NOMINAL
    }
    /// Production rule: q = n_off / c; the inclusive threshold is the largest x with
    /// x < q * 2^64. Refuses q > 1 (would claim a density the scheme cannot reach).
    pub fn set_nominal(&mut self, c: u64) -> Result<(), String> {
        if c == 0 || (self.n_off as u64) > c {
            return Err(format!("nominal q = {}/{} exceeds 1: syncmer density 1/{} is below the \
requested 1/{}; an approved denser scheme is required", self.n_off, c, self.n_off, c));
        }
        self.q_num = self.n_off as u64;
        self.q_den = c;
        let lim = ((self.n_off as u128) << 64) / c as u128;      // floor(q * 2^64)
        self.thin = if lim == 0 { 0 } else { (lim - 1).min(u64::MAX as u128) as u64 };
        if (self.n_off as u64) == c {
            self.thin = u64::MAX;
        }
        Ok(())
    }

    /// Serialized as the product's `selection` array.
    pub fn to_array(&self) -> Vec<u64> {
        vec![SELECTION_VERSION, self.mode, self.k as u64, self.s as u64, self.t as u64,
             self.n_off as u64, self.thin, self.target, self.realized, self.next_group_anchors,
             self.q_num, self.q_den]
    }
    pub fn from_array(v: &[u64]) -> Result<Selection, String> {
        if v.len() != 12 || v[0] != SELECTION_VERSION {
            return Err(format!("selection array {:?} is not version {}", v, SELECTION_VERSION));
        }
        let mut s = Selection::new(v[1], v[2] as usize, v[3] as usize, v[4] as usize)?;
        if s.n_off as u64 != v[5] {
            return Err("selection n_off disagrees with k and s".to_string());
        }
        s.thin = v[6];
        s.target = v[7];
        s.realized = v[8];
        s.next_group_anchors = v[9];
        s.q_num = v[10];
        s.q_den = v[11];
        Ok(s)
    }

    /// The anchor predicate on one canonical k-mer, given that k-mer's n_off s-mer hashes
    /// (`smer`, in FORWARD read/record coordinates; strand does not matter, see header).
    #[inline(always)]
    pub fn anchor_ok(&self, canon: u64, smer: &[u64]) -> bool {
        if fmix64(canon) > self.thin {
            return false;
        }
        !self.needs_syncmer() || centre_is_min(smer, self.t)
    }

    /// The same predicate from the canonical code alone (load-time validation, tests).
    pub fn anchor_ok_code(&self, canon: u64) -> bool {
        if fmix64(canon) > self.thin {
            return false;
        }
        if !self.needs_syncmer() {
            return true;
        }
        let h = smer_hashes_of_code(canon, self.k, self.s);
        centre_is_min(&h, self.t)
    }
}

/// Centre-among-minima.
#[inline(always)]
pub fn centre_is_min(h: &[u64], t: usize) -> bool {
    let c = h[t];
    if c == MASK64 {
        return false;
    }
    h.iter().all(|&x| x >= c)
}

/// splitmix64 of the canonical s-mer code.
#[inline(always)]
pub fn smer_hash(fwd: u64, s: usize) -> u64 {
    let rc = revcomp(fwd, s);
    splitmix(if fwd <= rc { fwd } else { rc })
}

#[inline(always)]
pub fn revcomp(x: u64, len: usize) -> u64 {
    let mut c = !x;
    let mut y = 0u64;
    for _ in 0..len {
        y = (y << 2) | (c & 3);
        c >>= 2;
    }
    y
}

/// The n_off s-mer hashes of one k-mer given its 2-bit code (first base most significant).
pub fn smer_hashes_of_code(code: u64, k: usize, s: usize) -> Vec<u64> {
    let n_off = k - s + 1;
    let m = if s == 32 { u64::MAX } else { (1u64 << (2 * s)) - 1 };
    (0..n_off).map(|j| smer_hash((code >> (2 * (n_off - 1 - j))) & m, s)).collect()
}

/// Per sequence position i: splitmix64 of the canonical s-mer starting at i, MASK64 when the
/// s-mer is not all-ACGT; length max(0, len - s + 1). Reuses `out`.
pub fn smer_track(seq: &[u8], s: usize, out: &mut Vec<u64>) {
    out.clear();
    let n = seq.len();
    if s == 0 || n < s {
        return;
    }
    let mask = if s == 32 { u64::MAX } else { (1u64 << (2 * s)) - 1 };
    let sh = 2 * (s - 1);
    let (mut f, mut r, mut run) = (0u64, 0u64, 0usize);
    for i in 0..n {
        let c = LUT[seq[i] as usize];
        let ok = c >= 0;
        let cs = if ok { c as u64 } else { 0 };
        f = ((f << 2) | cs) & mask;
        r = (r >> 2) | ((3 - cs) << sh);
        run = if ok { run + 1 } else { 0 };
        if i + 1 >= s {
            out.push(if run >= s { splitmix(if f <= r { f } else { r }) } else { MASK64 });
        }
    }
}
