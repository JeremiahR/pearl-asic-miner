//! Versioned ASIC designs. Each version is a full design corner for the
//! energy/area/cost model, plus the rationale for what changed. Spec sheets in
//! `docs/designs.md` and the figures in `docs/figures/` are generated from this
//! table (`cargo run --release -- figures`), so numbers never drift from code.

use crate::energy::{Scenario, n2, n3};

#[derive(Clone, Debug)]
pub struct Design {
    /// Version id, e.g. "PA-2" (Pearl ASIC, generation 2).
    pub version: &'static str,
    pub codename: &'static str,
    pub status: &'static str,
    pub summary: &'static str,
    /// What changed relative to the previous version, and why.
    pub changes: &'static [&'static str],
    pub scenario: Scenario,
}

fn corner(node: crate::energy::Node, vdd: f64, cols_per_pe: f64, pde: f64, cooling: f64) -> Scenario {
    Scenario {
        name: "",
        node,
        vdd,
        cols_per_pe,
        glitch: 1.3,
        clock_tree_overhead: 0.25,
        n_chip: 2048.0,
        k: 2048.0,
        power_delivery_eff: pde,
        cooling_frac: cooling,
        misc_frac: 0.03,
        die_overhead_frac: 0.15,
    }
}

pub fn all() -> Vec<Design> {
    let mut v = vec![
        Design {
            version: "PA-1",
            codename: "Baseline",
            status: "reference only — thermally impractical",
            summary: "N3 at nominal 0.75 V, 1-column PEs. The straightforward port of the verified datapath.",
            changes: &[
                "DP32 atom with H100 semantics (13-bit window): 667 GE/MAC vs 958 for B200 semantics",
                "Weight-stationary B' in clock-gated latches; A' generated on chip (no DRAM)",
                "2048 columns x k=2048 per die; 16x16 lottery tile per core",
            ],
            scenario: corner(n3(), 0.75, 1.0, 0.90, 0.06),
        },
        Design {
            version: "PA-2",
            codename: "Near-threshold",
            status: "RECOMMENDED — cost-optimal on N3",
            summary: "N3 at 0.46 V (cost-optimal point of the voltage sweep), 2-column PEs sharing the A' register.",
            changes: &[
                "Vdd 0.75 V -> 0.46 V: ~2.7x less switching energy; power density 2.1 -> 0.27 W/mm^2 (air-coolable)",
                "2-column PEs halve A' operand-register and wire energy per MAC",
                "0.46 V chosen by minimum USD/EFLOP (energy + capex), not maximum efficiency",
            ],
            scenario: corner(n3(), 0.46, 2.0, 0.92, 0.05),
        },
        Design {
            version: "PA-3",
            codename: "Stacked N2",
            status: "stretch — maximum efficiency",
            summary: "N2 at 0.38 V, 4-column PEs, boards voltage-stacked Bitcoin-miner style.",
            changes: &[
                "N3 -> N2 (GAA): ~20% lower switched capacitance, denser cells",
                "Vdd 0.38 V with series-stacked chips (~25 per string): power delivery 0.92 -> 0.94",
                "4-column PEs: A' movement amortised over 4 outputs",
                "Trades ~25% higher cost per EFLOP than the N2 cost optimum for maximum TFLOPS/W",
            ],
            scenario: corner(n2(), 0.38, 4.0, 0.94, 0.04),
        },
    ];
    for d in &mut v {
        d.scenario.name = d.version;
    }
    v
}

pub fn recommended() -> Design {
    all().into_iter().find(|d| d.version == "PA-2").unwrap()
}
