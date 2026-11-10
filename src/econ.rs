//! Mining economics: cost per unit of Pearl work for an ASIC, a pure-mining
//! GPU, and a GPU that mines as a ~1% by-product of paid FP8 inference.
//!
//! Pearl difficulty is normalised to MACs (`hash <= target * h * w * k`), so
//! "work" is simply FP8 MACs (2 FLOP each). We price 1 EFLOP (1e18 FLOP).

use crate::energy::{Breakdown, Gpu};

#[derive(Clone, Debug)]
pub struct Market {
    pub usd_per_kwh: f64,
    pub lifetime_years: f64,
    pub uptime: f64,
    /// Marginal mining overhead of an inference GPU (user-supplied: ~1%).
    pub inference_overhead: f64,
    /// Package + test + substrate per ASIC die (USD).
    pub pkg_usd: f64,
    /// Boards, PSUs, cooling, chassis per watt of ASIC capacity (USD/W).
    /// Bitcoin hydro/air units land ~$0.5-1/W excluding chips.
    pub system_usd_per_w: f64,
    pub die_yield: f64,
}

impl Default for Market {
    fn default() -> Self {
        Market {
            usd_per_kwh: 0.05,
            lifetime_years: 3.0,
            uptime: 0.95,
            inference_overhead: 0.01,
            pkg_usd: 10.0,
            system_usd_per_w: 0.8,
            die_yield: 0.80,
        }
    }
}

const EFLOP: f64 = 1e18;
const WAFER_MM2: f64 = 70_686.0; // 300 mm wafer
const EDGE_LOSS: f64 = 0.88;

#[derive(Clone, Debug)]
pub struct Cost {
    pub energy_usd: f64,
    pub capex_usd: f64,
}
impl Cost {
    pub fn total(&self) -> f64 {
        self.energy_usd + self.capex_usd
    }
}

impl Market {
    fn lifetime_s(&self) -> f64 {
        self.lifetime_years * 365.25 * 86_400.0 * self.uptime
    }
    fn usd_per_j(&self) -> f64 {
        self.usd_per_kwh / 3.6e6
    }

    pub fn asic_chip_usd(&self, b: &Breakdown) -> f64 {
        let dies = WAFER_MM2 * EDGE_LOSS / b.die_mm2 * self.die_yield;
        b.scenario.node.wafer_usd / dies + self.pkg_usd
    }

    pub fn asic(&self, b: &Breakdown) -> Cost {
        let j_per_flop = b.fj_wall * 1e-15 / 2.0;
        let energy = j_per_flop * EFLOP * self.usd_per_j();
        let wall_w_per_chip = b.watts_per_chip / b.scenario.power_delivery_eff * (1.0 + b.scenario.cooling_frac);
        let capex_per_chip = self.asic_chip_usd(b) + self.system_usd_per_w * wall_w_per_chip;
        let flops_lifetime = b.pflops_per_chip * 1e15 * self.lifetime_s();
        Cost { energy_usd: energy, capex_usd: capex_per_chip / flops_lifetime * EFLOP }
    }

    pub fn gpu_pure(&self, g: &Gpu) -> Cost {
        let j_per_flop = g.wall_w / (g.dense_fp8_tflops * 1e12);
        let flops_lifetime = g.dense_fp8_tflops * 1e12 * self.lifetime_s();
        Cost { energy_usd: j_per_flop * EFLOP * self.usd_per_j(), capex_usd: g.capex_usd / flops_lifetime * EFLOP }
    }

    /// Marginal cost of Pearl work for a GPU already doing paid FP8 inference:
    /// only the overhead fraction of its energy; capex is paid by inference.
    pub fn gpu_inference_marginal(&self, g: &Gpu) -> Cost {
        let pure = self.gpu_pure(g);
        Cost { energy_usd: pure.energy_usd * self.inference_overhead, capex_usd: 0.0 }
    }
}
