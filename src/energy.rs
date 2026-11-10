//! First-principles energy / area model for the DP32-atom mining ASIC.
//!
//! Method: count gate-equivalents (GE, one NAND2) per atom from the bit widths
//! the bit-exact model in `atom.rs` proves sufficient, then
//!   E/op = GE_logic * alpha * glitch * e_ge(V)          (datapath)
//!        + flops * (e_clk + alpha_d * e_data)            (registers)
//!        + operand-hop wire energy                       (systolic A' flow)
//!        + leakage * area / f(V)                         (static)
//! and add amortised per-chip overheads (A' generation, Merkle hashing, noise,
//! noisy-quant, epilogue, control/IO) and system losses (VRM/PSU, cooling).
//! Activity factors `alpha` come from the simulator (`engine::Activity`).
//!
//! Calibration anchor for `e_ge`: Horowitz (ISSCC 2014, 45 nm, 0.9 V): an
//! 8-bit add is 0.03 pJ. That is ~8 full adders = ~48 GE at alpha ~0.5, i.e.
//! ~1.25 fJ per GE toggle. 45 nm -> N3 cuts switched capacitance by ~7x and
//! (0.75/0.9)^2 = 0.69 for voltage, giving ~0.12 fJ per GE toggle at 0.75 V.
//! This is the single most important (and most uncertain, +-2x) constant.
//! Sanity check: it implies ~35 fJ per FP8 MAC at nominal voltage, while 3 nm
//! SRAM digital CIM macros report 125 TOPS/W INT8 (= 16 fJ/MAC, macro only,
//! low voltage, favourable data). So the model is not optimistic at nominal
//! voltage.

use crate::atom::{ATOM_K, WINDOW_ATOMS};

#[derive(Clone, Debug)]
pub struct Node {
    pub name: &'static str,
    /// Energy per GE output toggle at `v_ref` (fJ), including local wiring.
    pub e_ge_fj: f64,
    pub v_ref: f64,
    /// Effective threshold voltage for the alpha-power delay law.
    pub vt: f64,
    pub alpha_pow: f64,
    /// Atom clock at `v_ref` (GHz), two pipeline stages per atom.
    pub f_ref_ghz: f64,
    /// Placed area per GE (um^2) at ~70% utilisation.
    pub um2_per_ge: f64,
    /// Leakage per GE at `v_ref` (nW), mixed SVT/LVT library.
    pub leak_nw_per_ge: f64,
    /// Wire capacitance (fF/um), local/intermediate metal.
    pub c_wire_ff_um: f64,
    /// Wafer price (USD) for the capex model.
    pub wafer_usd: f64,
}

pub fn n3() -> Node {
    Node {
        name: "TSMC N3E-class",
        e_ge_fj: 0.12,
        v_ref: 0.75,
        vt: 0.30,
        alpha_pow: 1.4,
        f_ref_ghz: 1.2,
        um2_per_ge: 0.040,
        leak_nw_per_ge: 0.5,
        c_wire_ff_um: 0.20,
        wafer_usd: 20_000.0,
    }
}

pub fn n2() -> Node {
    Node {
        name: "TSMC N2-class",
        // ~20% lower switching energy than N3 at iso-voltage (GAA, lower C).
        e_ge_fj: 0.096,
        v_ref: 0.75,
        vt: 0.28,
        alpha_pow: 1.35,
        f_ref_ghz: 1.3,
        um2_per_ge: 0.034,
        leak_nw_per_ge: 0.4,
        c_wire_ff_um: 0.19,
        wafer_usd: 28_000.0,
    }
}

/// Gate inventory of one DP32 atom (H100 semantics), from `atom.rs` widths.
#[derive(Clone, Debug)]
pub struct AtomInventory {
    pub products: f64,
    pub max_exp_tree: f64,
    pub shift_amount: f64,
    pub align_shifters: f64,
    pub cond_negate: f64,
    pub csa_tree: f64,
    pub cpa_norm_pack: f64,
    pub promote_share: f64,
    /// Registers that clock every cycle (A' operand, carry, mid-pipe).
    pub switching_flops: f64,
    /// Clock-gated B' storage latches (leak only).
    pub stationary_latches: f64,
}

const GE_FA: f64 = 6.0; // mirror full adder
const GE_AND: f64 = 1.33;
const GE_XOR: f64 = 2.0;
const GE_MUX: f64 = 2.33;
const GE_FLOP: f64 = 6.0;
const GE_LATCH: f64 = 4.0;

impl AtomInventory {
    /// `window_bits`: fractional bits kept below the anchor (H100 13, B200 25).
    pub fn new(window_bits: u32) -> Self {
        let lanes = ATOM_K as f64;
        let terms = lanes + 1.0;
        let w = window_bits as f64 + 3.0; // aligned term width incl. sign/integer bits (16 for H100)
        let tree_w = w + 6.0; // CSA width incl. growth for 33 inputs (~22)
        let shift_stages = (w).log2().ceil(); // 4 for H100, 5 for B200
        AtomInventory {
            // 4x4 significand multiplier (16 AND + ~12 FA) + 4b exponent add + sign + zero detect
            products: lanes * (16.0 * GE_AND + 12.0 * GE_FA + 4.0 * GE_FA + GE_XOR + 2.0),
            // 32 compare-select nodes on 8-bit exponents
            max_exp_tree: (terms - 1.0) * (8.0 * 3.0 + 8.0 * GE_MUX),
            // anchor - exp, saturate to shifter range
            shift_amount: terms * (8.0 * GE_FA + 6.0),
            align_shifters: terms * w * shift_stages * GE_MUX,
            cond_negate: terms * w * GE_XOR,
            // 3:2 compressor tree, 33 -> 2 rows
            csa_tree: (terms - 2.0) * tree_w * GE_FA,
            // prefix CPA + LZC + normalising shifter + exponent adjust + pack
            cpa_norm_pack: tree_w * 10.0 + tree_w * 4.0 + tree_w * 5.0 * GE_MUX + 9.0 * GE_FA + 120.0,
            // one FP32 RNE adder (~2000 GE) shared by the 4 atoms of a window
            promote_share: 2000.0 / WINDOW_ATOMS as f64,
            switching_flops: lanes * 8.0 + 32.0 + 2.0 * tree_w,
            stationary_latches: lanes * 8.0,
        }
    }
    pub fn logic_ge(&self) -> f64 {
        self.products
            + self.max_exp_tree
            + self.shift_amount
            + self.align_shifters
            + self.cond_negate
            + self.csa_tree
            + self.cpa_norm_pack
            + self.promote_share
    }
    pub fn area_ge(&self) -> f64 {
        self.logic_ge() + self.switching_flops * GE_FLOP + self.stationary_latches * GE_LATCH
    }
}

/// Measured switching activity (from `engine::Activity`).
#[derive(Clone, Copy, Debug)]
pub struct ActivityFactors {
    /// Operand-bus bit toggle probability per hop.
    pub operand: f64,
    /// Adder-tree input bit toggle probability; used for all datapath logic.
    pub logic: f64,
}

#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: &'static str,
    pub node: Node,
    pub vdd: f64,
    /// Output columns sharing one A' operand register (multi-column PE).
    pub cols_per_pe: f64,
    pub glitch: f64,
    /// Global clock distribution on top of local flop clock energy.
    pub clock_tree_overhead: f64,
    /// Output columns per chip that share one A' generator.
    pub n_chip: f64,
    pub k: f64,
    /// Power-delivery efficiency (VRM + PSU). Voltage-stacked boards ~0.93.
    pub power_delivery_eff: f64,
    /// Cooling (fans / pumps) as a fraction of IT power.
    pub cooling_frac: f64,
    /// Controller, PLLs, IO, misc as a fraction of chip core power.
    pub misc_frac: f64,
    /// Non-array die area (control, A' generator, SRAM, IO) as a fraction.
    pub die_overhead_frac: f64,
}

#[derive(Clone, Debug)]
pub struct Breakdown {
    pub scenario: Scenario,
    pub f_ghz: f64,
    pub fj_logic: f64,
    pub fj_flops: f64,
    pub fj_wire: f64,
    pub fj_leak: f64,
    pub fj_overhead: f64,
    pub fj_chip: f64,
    pub fj_wall: f64,
    pub tflops_per_w_chip: f64,
    pub tflops_per_w_wall: f64,
    pub die_mm2: f64,
    pub pflops_per_chip: f64,
    pub watts_per_chip: f64,
    pub w_per_mm2: f64,
    pub ge_per_mac: f64,
}

impl Scenario {
    pub fn freq_ghz(&self) -> f64 {
        let n = &self.node;
        let d = |v: f64| (v - n.vt).max(1e-3).powf(n.alpha_pow) / v;
        n.f_ref_ghz * d(self.vdd) / d(n.v_ref)
    }

    pub fn evaluate(&self, inv: &AtomInventory, act: ActivityFactors) -> Breakdown {
        let n = &self.node;
        let v2 = (self.vdd / n.v_ref).powi(2);
        let e_ge = n.e_ge_fj * v2;
        let macs = ATOM_K as f64;
        let f = self.freq_ghz();

        // Datapath logic.
        let fj_logic = inv.logic_ge() * act.logic * self.glitch * e_ge / macs;

        // Registers: clock every cycle + data toggles. A' operand register is
        // shared by `cols_per_pe` columns.
        let e_clk = 4.0 * e_ge;
        let e_data = 4.0 * e_ge;
        let operand_flops = macs * 8.0 / self.cols_per_pe;
        let other_flops = inv.switching_flops - macs * 8.0;
        let fj_flops = ((operand_flops + other_flops) * e_clk * (1.0 + self.clock_tree_overhead)
            + operand_flops * act.operand * e_data
            + other_flops * act.logic * e_data)
            / macs;

        // Systolic A' hop: 256 bits move one PE pitch per (cols_per_pe) columns.
        let pe_area_um2 = inv.area_ge() * n.um2_per_ge * self.cols_per_pe;
        let pitch_um = pe_area_um2.sqrt();
        let e_bit_hop = 0.5 * n.c_wire_ff_um * pitch_um * self.vdd * self.vdd; // fJ per toggle
        let carry_bits = 32.0;
        let fj_wire = (macs * 8.0 / self.cols_per_pe * act.operand + carry_bits * act.logic) * e_bit_hop / macs;

        // Leakage (scales ~V^3 with DIBL), charged per op at frequency f.
        let leak_nw = inv.area_ge() * n.leak_nw_per_ge * (self.vdd / n.v_ref).powi(3);
        let fj_leak = leak_nw * 1e-9 / (f * 1e9) * 1e15 / macs;

        let core = fj_logic + fj_flops + fj_wire + fj_leak;

        // Per-chip A' generation, amortised over n_chip columns:
        //  * noise E_A @ F_A: one 32-product atom per A' element  -> 32 MAC-eq
        //  * noisy quant, prequant, PRNG: ~400 GE-toggles per element
        //  * Merkle BLAKE3 over 1.25 B/element (+ per-row E line):
        //    ~0.2 pJ/B at 0.75 V N3 (SHA-256d miners reach ~0.1 pJ/B at 3 nm)
        let per_elem = 32.0 * core + 400.0 * e_ge + 1.25 * 200.0 * v2 * (n.e_ge_fj / 0.12);
        let fj_gen = per_elem / self.n_chip;
        // Epilogue: fold (32b mul+add+rot per output) + one BLAKE3 compression
        // per tile; per output element ~ (600 GE + 3000/256 GE) / k MACs.
        let fj_epi = (600.0 + 3000.0 / 256.0) * act.logic * e_ge / self.k;
        let fj_overhead = fj_gen + fj_epi + core * self.misc_frac;

        let fj_chip = core + fj_overhead;
        let fj_wall = fj_chip / self.power_delivery_eff * (1.0 + self.cooling_frac);
        let tf = |fj_per_mac: f64| 2.0 / (fj_per_mac * 1e-15) / 1e12; // TFLOPS/W (2 FLOP / MAC)

        // Die sizing: n_chip columns x k depth.
        let atoms = self.n_chip * self.k / macs;
        let array_mm2 = atoms * inv.area_ge() * n.um2_per_ge / 1e6;
        let die_mm2 = array_mm2 / (1.0 - self.die_overhead_frac);
        let macs_per_s = atoms * macs * f * 1e9;
        let pflops = 2.0 * macs_per_s / 1e15;
        let watts = macs_per_s * fj_chip * 1e-15;

        Breakdown {
            scenario: self.clone(),
            f_ghz: f,
            fj_logic,
            fj_flops,
            fj_wire,
            fj_leak,
            fj_overhead,
            fj_chip,
            fj_wall,
            tflops_per_w_chip: tf(fj_chip),
            tflops_per_w_wall: tf(fj_wall),
            die_mm2,
            pflops_per_chip: pflops,
            watts_per_chip: watts,
            w_per_mm2: watts / die_mm2,
            ge_per_mac: inv.area_ge() / macs,
        }
    }
}

/// The versioned design corners (see `designs.rs`).
pub fn scenarios() -> Vec<Scenario> {
    crate::designs::all().into_iter().map(|d| d.scenario).collect()
}

/// GPU reference points (FP8 dense, no 2:4 sparsity: the jackpot policy's
/// liveness rule rejects structured-sparse operands).
#[derive(Clone, Debug)]
pub struct Gpu {
    pub name: &'static str,
    pub dense_fp8_tflops: f64,
    pub board_w: f64,
    /// Wall power per GPU in its reference 8-GPU server.
    pub wall_w: f64,
    pub capex_usd: f64,
}

pub fn gpus() -> Vec<Gpu> {
    vec![
        // 1,979 TF dense FP8 @ 700 W TDP; DGX H100 ~10.2 kW / 8 GPUs.
        Gpu { name: "H100 SXM", dense_fp8_tflops: 1979.0, board_w: 700.0, wall_w: 10_200.0 / 8.0, capex_usd: 30_000.0 },
        // 4,500 TF dense FP8 @ 1,000 W; DGX B200 ~14.3 kW / 8 GPUs.
        Gpu { name: "B200", dense_fp8_tflops: 4500.0, board_w: 1000.0, wall_w: 14_300.0 / 8.0, capex_usd: 45_000.0 },
    ]
}

impl Gpu {
    pub fn tflops_per_w_board(&self) -> f64 {
        self.dense_fp8_tflops / self.board_w
    }
    pub fn tflops_per_w_wall(&self) -> f64 {
        self.dense_fp8_tflops / self.wall_w
    }
}
