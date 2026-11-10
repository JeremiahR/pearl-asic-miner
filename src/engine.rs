//! Functional + activity model of one "Pearl core": a weight-stationary array
//! of DP32 atoms with the lottery epilogue (fold + keyed BLAKE3 ticket).
//!
//! Dataflow (one core, `cols` output columns, depth `k`):
//!
//! ```text
//!            B' (cols x k) is STATIONARY: atom PE (j, c) holds B'[j, 32c..32c+32]
//!
//!  A' row i ─┬─► PE(0,0) ─► PE(1,0) ─► ... ─► PE(cols-1,0)     atoms c = 0..3   ┐ window 0
//!  (k bytes) ├─► PE(0,1) ─► ...                                 chain the carry ┘
//!  skewed    ├─► ...                                            vertically
//!  by chunk  └─► PE(0,k/32-1) ...
//!                     │ window results
//!                     ▼
//!            FP32 promotion chain per column (ascending windows, pipelined)
//!                     ▼
//!            C[i, 0..cols]  ──►  tile buffer (h rows) ──► fold (16 lanes) ──► BLAKE3 ──► compare
//! ```
//!
//! One A' row enters per cycle, so a core retires `cols * k` MACs per cycle.
//! The functional model below executes exactly that order of operations; the
//! cycle timing is analytic (see `timing`).

use crate::atom::{self, ATOM_K, Term, WINDOW_K};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub const JACKPOT_ENTRIES: usize = 16;
pub const LABEL_JACKPOT: &[u8] = b"pearl/v4/FP8/jackpot";

/// Bit-toggle / truncation counters gathered while streaming, used to
/// calibrate the energy model's activity factors.
#[derive(Default, Clone, Debug)]
pub struct Activity {
    pub macs: u64,
    pub live_products: u64,
    pub dropped_products: u64,
    /// A' operand bus bits that toggled between consecutive rows (per hop).
    pub operand_toggles: u64,
    pub operand_bits: u64,
    /// Adder-tree input bits (16b aligned two's-complement terms) that toggled
    /// between consecutive rows at the same PE.
    pub tree_toggles: u64,
    pub tree_bits: u64,
}

impl Activity {
    pub fn operand_alpha(&self) -> f64 {
        self.operand_toggles as f64 / self.operand_bits.max(1) as f64
    }
    pub fn tree_alpha(&self) -> f64 {
        self.tree_toggles as f64 / self.tree_bits.max(1) as f64
    }
    pub fn dropped_fraction(&self) -> f64 {
        self.dropped_products as f64 / self.live_products.max(1) as f64
    }
}

pub struct Core {
    pub cols: usize,
    pub k: usize,
    /// Stationary B' (cols x k, row-major: row j = output column j).
    b: Vec<u8>,
    prev_row: Option<Vec<u8>>,
    prev_tree: Vec<u16>,
    pub activity: Activity,
}

fn aligned_word(t: &Term, anchor: i32) -> u16 {
    if !t.live {
        return 0;
    }
    let sh = (anchor - t.exp) as u32;
    let v = if sh >= atom::MAX_SHIFT { 0 } else { t.sig >> sh } as i32;
    (if t.neg { -v } else { v }) as u16
}

impl Core {
    pub fn new(cols: usize, k: usize, b: Vec<u8>) -> Self {
        assert!(k.is_multiple_of(ATOM_K));
        assert_eq!(b.len(), cols * k);
        let atoms = cols * k / ATOM_K;
        Core { cols, k, b, prev_row: None, prev_tree: vec![0; atoms * (ATOM_K + 1)], activity: Activity::default() }
    }

    /// Stream one A' row through the array, returning `cols` FP32 outputs.
    /// If `track` is set, also records toggle activity (slower).
    pub fn stream_row(&mut self, a_row: &[u8], track: bool) -> Vec<f32> {
        assert_eq!(a_row.len(), self.k);
        let mut out = Vec::with_capacity(self.cols);
        let b = std::mem::take(&mut self.b);
        for j in 0..self.cols {
            let b_row = &b[j * self.k..(j + 1) * self.k];
            let mut total = 0.0f32;
            for (w, (wa, wb)) in a_row.chunks(WINDOW_K).zip(b_row.chunks(WINDOW_K)).enumerate() {
                let mut local = 0.0f32;
                for (c, (ca, cb)) in wa.chunks(ATOM_K).zip(wb.chunks(ATOM_K)).enumerate() {
                    if track {
                        self.track_tree(j, w * 4 + c, local, ca, cb);
                    }
                    let r = atom::atom(local, ca, cb).expect("fp32 overflow on mining data");
                    local = r.value;
                    self.activity.dropped_products += r.dropped as u64;
                    self.activity.live_products += ca.iter().zip(cb).filter(|(x, y)| atom::product(**x, **y).live).count() as u64;
                }
                total = atom::promote(total, local).expect("fp32 overflow on mining data");
            }
            out.push(total);
        }
        self.b = b;
        self.activity.macs += (self.cols * self.k) as u64;
        if track {
            if let Some(prev) = &self.prev_row {
                // Each A' byte hops through `cols` PE registers.
                let t: u64 = prev.iter().zip(a_row).map(|(p, q)| (p ^ q).count_ones() as u64).sum();
                self.activity.operand_toggles += t * self.cols as u64;
                self.activity.operand_bits += (self.k * 8 * self.cols) as u64;
            }
            self.prev_row = Some(a_row.to_vec());
        }
        out
    }

    fn track_tree(&mut self, j: usize, c: usize, carry: f32, a: &[u8], b: &[u8]) {
        let mut terms = [Term { neg: false, exp: 0, sig: 0, live: false }; ATOM_K + 1];
        terms[0] = atom::carry_term(carry);
        for i in 0..a.len() {
            terms[i + 1] = atom::product(a[i], b[i]);
        }
        let Some(anchor) = terms.iter().filter(|t| t.live).map(|t| t.exp).max() else { return };
        let base = (j * (self.k / ATOM_K) + c) * (ATOM_K + 1);
        let have_prev = self.prev_row.is_some();
        for (s, t) in terms.iter().enumerate() {
            let w = aligned_word(t, anchor);
            if have_prev {
                self.activity.tree_toggles += (w ^ self.prev_tree[base + s]).count_ones() as u64;
                self.activity.tree_bits += 16;
            }
            self.prev_tree[base + s] = w;
        }
    }
}

/// Lottery epilogue: `lane = rotl32(lane * 0x9E3779B1 + bits, 13)` over each
/// lane's committed subtile, serialised LE into one 64-byte BLAKE3 block.
pub fn xor_fold(c_tile: &[f32], lanes: &[Vec<usize>]) -> [u8; 64] {
    assert_eq!(lanes.len(), JACKPOT_ENTRIES);
    let mut acc = [0u32; JACKPOT_ENTRIES];
    for (lane, idxs) in acc.iter_mut().zip(lanes) {
        for &i in idxs {
            *lane = lane.wrapping_mul(0x9E37_79B1).wrapping_add(c_tile[i].to_bits()).rotate_left(13);
        }
    }
    let mut out = [0u8; 64];
    for (i, l) in acc.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&l.to_le_bytes());
    }
    out
}

/// `J = BLAKE3_keyed(msg, key = BLAKE3_keyed("pearl/v4/FP8/jackpot", seedA))`.
/// The sub-key is computed once per commitment by the controller, so the
/// per-ticket cost is exactly one BLAKE3 compression.
pub fn ticket(seed_a: &[u8; 32], msg: &[u8; 64]) -> [u8; 32] {
    let subkey = *blake3::keyed_hash(seed_a, LABEL_JACKPOT).as_bytes();
    *blake3::keyed_hash(&subkey, msg).as_bytes()
}

/// `hash <= bound` with both as little-endian 256-bit integers.
pub fn meets(hash: &[u8; 32], bound_le: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if hash[i] != bound_le[i] {
            return hash[i] < bound_le[i];
        }
    }
    true
}

/// Simple contiguous lane layout for an `h x w` tile split into a 4x4 grid of
/// subtiles (the shape a rows=[fold h/4, blake 4] / cols=[fold w/4, blake 4]
/// pattern produces). Real deployments take this from `lane_assignment`.
pub fn grid_lanes(h: usize, w: usize) -> Vec<Vec<usize>> {
    assert!(h.is_multiple_of(4) && w.is_multiple_of(4));
    let (fh, fw) = (h / 4, w / 4);
    let mut lanes = Vec::new();
    for br in 0..4 {
        for bc in 0..4 {
            let mut l = Vec::new();
            for r in 0..fh {
                for c in 0..fw {
                    l.push((br * fh + r) * w + bc * fw + c);
                }
            }
            lanes.push(l);
        }
    }
    lanes
}

/// Synthetic post-noisy-quantisation operands. On-chain A' = fp8(A/alpha +
/// E@F): the injected rank-32 noise makes every code close to i.i.d. and dense,
/// which is what the jackpot policy enforces (liveness, noise floor, <5%
/// skippable summands). We draw fp8(N(0, sigma)) with sigma chosen so the
/// largest codes stay below the 448 saturation point.
pub struct OperandGen {
    rng: StdRng,
    pub sigma: f32,
}

impl OperandGen {
    pub fn new(seed: u64, sigma: f32) -> Self {
        OperandGen { rng: StdRng::seed_from_u64(seed), sigma }
    }
    fn gauss(&mut self) -> f32 {
        let u1: f32 = self.rng.random::<f32>().max(1e-12);
        let u2: f32 = self.rng.random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    pub fn row(&mut self, k: usize) -> Vec<u8> {
        (0..k).map(|_| crate::fp8::from_f32((self.gauss() * self.sigma).clamp(-448.0, 448.0))).collect()
    }
    pub fn random_codes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| loop {
                let c: u8 = self.rng.random();
                if !crate::fp8::is_nan(c) {
                    break c;
                }
            })
            .collect()
    }
}

/// Analytic timing of a core.
pub mod timing {
    use super::*;
    /// Pipeline fill latency (cycles) for one row through a core: the A' skew
    /// across k/32 atom slots (1 cycle each, the carry chain is 4 deep per
    /// window and windows run in parallel), the promotion chain (one FP32 add
    /// stage per window), and the column hops.
    pub fn fill_latency(cols: usize, k: usize) -> usize {
        let windows = k / WINDOW_K;
        atom::WINDOW_ATOMS + windows + cols
    }
    /// Cycles between tickets for an `h`-row tile (one row per cycle).
    pub fn cycles_per_ticket(h: usize) -> usize {
        h
    }
}
