//! `pearl-asic-sim` — run the activity simulation and print the design report
//! tables (energy, area, GPU comparison, economics), or run a mining demo.
//!
//!   cargo run --release              # full report
//!   cargo run --release -- sweep     # voltage sweep for the cost-optimal operating point
//!   cargo run --release -- figures   # regenerate docs/figures/*.svg|csv and docs/designs.md
//!   cargo run --release -- demo      # stream tiles, hash tickets, count hits

use pearl_asic_sim::econ::Market;
use pearl_asic_sim::energy::{self, ActivityFactors, AtomInventory};
use pearl_asic_sim::engine::{self, Core, OperandGen};

fn measure_activity(cols: usize, k: usize, rows: usize) -> engine::Activity {
    let mut g = OperandGen::new(42, 48.0);
    let b: Vec<u8> = (0..cols).flat_map(|_| g.row(k)).collect();
    let mut core = Core::new(cols, k, b);
    for _ in 0..rows {
        let r = g.row(k);
        core.stream_row(&r, true);
    }
    core.activity
}

fn report() {
    println!("== Pearl FP8 (cert v4) mining ASIC — model report ==\n");

    let act = measure_activity(16, 2048, 48);
    println!("-- Simulated switching activity (16 cols x k=2048, 48 rows of fp8(N(0,48)) operands)");
    println!("   MACs simulated            {:>14}", act.macs);
    println!("   A' operand toggle rate    {:>14.3}", act.operand_alpha());
    println!("   adder-tree input toggle   {:>14.3}", act.tree_alpha());
    println!("   products truncated to 0   {:>13.3}%  (policy allows <= 5% skippable)\n", 100.0 * act.dropped_fraction());
    let af = ActivityFactors { operand: act.operand_alpha(), logic: act.tree_alpha() };

    let h = AtomInventory::new(13);
    let b2 = AtomInventory::new(25);
    println!("-- DP32 atom gate inventory (GE = NAND2 equivalents)");
    println!("   {:<28}{:>12}{:>12}", "block", "H100-sem", "B200-sem");
    let rows: [(&str, f64, f64); 10] = [
        ("32x product units", h.products, b2.products),
        ("max-exponent tree", h.max_exp_tree, b2.max_exp_tree),
        ("shift-amount subtract", h.shift_amount, b2.shift_amount),
        ("align shifters", h.align_shifters, b2.align_shifters),
        ("conditional negate", h.cond_negate, b2.cond_negate),
        ("CSA tree 33->2", h.csa_tree, b2.csa_tree),
        ("CPA/LZC/normalise/pack", h.cpa_norm_pack, b2.cpa_norm_pack),
        ("FP32 promote (1/4 share)", h.promote_share, b2.promote_share),
        ("flops (switching) x6", h.switching_flops * 6.0, b2.switching_flops * 6.0),
        ("B' latches (gated) x4", h.stationary_latches * 4.0, b2.stationary_latches * 4.0),
    ];
    for (n, a, b) in rows {
        println!("   {n:<28}{a:>12.0}{b:>12.0}");
    }
    println!("   {:<28}{:>12.0}{:>12.0}", "TOTAL per atom", h.area_ge(), b2.area_ge());
    println!("   {:<28}{:>12.0}{:>12.0}\n", "GE per MAC", h.area_ge() / 32.0, b2.area_ge() / 32.0);

    let gpus = energy::gpus();
    println!("-- GPU reference (dense FP8)");
    for g in &gpus {
        println!(
            "   {:<10} {:>6.0} TF  board {:>5.2} TF/W ({:>4.0} fJ/MAC)   wall {:>5.2} TF/W ({:>4.0} fJ/MAC)",
            g.name,
            g.dense_fp8_tflops,
            g.tflops_per_w_board(),
            2000.0 / g.tflops_per_w_board(),
            g.tflops_per_w_wall(),
            2000.0 / g.tflops_per_w_wall()
        );
    }
    println!();

    let market = Market::default();
    let h100 = &gpus[0];
    let b200 = &gpus[1];
    let mut results = Vec::new();
    for d in pearl_asic_sim::designs::all() {
        let s = d.scenario.clone();
        let r = s.evaluate(&h, af);
        println!("-- {} \"{}\" ({}): {}", d.version, d.codename, d.status, d.summary);
        println!("   clock {:.2} GHz | die {:.0} mm^2 | {:.1} PFLOPS/chip | {:.0} W/chip | {:.2} W/mm^2 | {:.0} GE/MAC",
            r.f_ghz, r.die_mm2, r.pflops_per_chip, r.watts_per_chip, r.w_per_mm2, r.ge_per_mac);
        println!("   fJ/MAC: logic {:.2}  flops {:.2}  wire {:.2}  leak {:.2}  overhead {:.2}  => chip {:.2}  wall {:.2}",
            r.fj_logic, r.fj_flops, r.fj_wire, r.fj_leak, r.fj_overhead, r.fj_chip, r.fj_wall);
        println!("   efficiency: chip {:.0} TFLOPS/W, wall {:.0} TFLOPS/W", r.tflops_per_w_chip, r.tflops_per_w_wall);
        println!("   vs H100: {:>5.1}x board-to-chip, {:>5.1}x wall-to-wall | vs B200: {:>5.1}x board-to-chip, {:>5.1}x wall-to-wall",
            r.tflops_per_w_chip / h100.tflops_per_w_board(), r.tflops_per_w_wall / h100.tflops_per_w_wall(),
            r.tflops_per_w_chip / b200.tflops_per_w_board(), r.tflops_per_w_wall / b200.tflops_per_w_wall());
        println!("   est. chip cost ${:.0} (die + package)\n", market.asic_chip_usd(&r));
        results.push(r);
    }

    println!("-- Sensitivity: wall-to-wall multiple vs H100 if e_ge (switching energy) is off by 0.5x / 2x");
    for s in energy::scenarios() {
        let mut lo = s.clone();
        lo.node.e_ge_fj *= 0.5;
        let mut hi = s.clone();
        hi.node.e_ge_fj *= 2.0;
        let x = |sc: &energy::Scenario| sc.evaluate(&h, af).tflops_per_w_wall / h100.tflops_per_w_wall();
        println!("   {:<45} {:>5.0}x .. {:>5.0}x .. {:>5.0}x", s.name, x(&hi), x(&s), x(&lo));
    }
    println!();

    println!("-- Economics: USD per EFLOP of Pearl work (${:.3}/kWh, {:.0}-yr life, {:.0}% inference overhead)",
        market.usd_per_kwh, market.lifetime_years, market.inference_overhead * 100.0);
    println!("   {:<46}{:>10}{:>10}{:>10}", "miner", "energy", "capex", "total");
    for g in &gpus {
        let c = market.gpu_pure(g);
        println!("   {:<46}{:>10.4}{:>10.4}{:>10.4}", format!("{} pure mining", g.name), c.energy_usd, c.capex_usd, c.total());
        let c = market.gpu_inference_marginal(g);
        println!("   {:<46}{:>10.4}{:>10.4}{:>10.4}", format!("{} inference by-product (marginal)", g.name), c.energy_usd, c.capex_usd, c.total());
    }
    for r in &results {
        let c = market.asic(r);
        println!("   {:<46}{:>10.4}{:>10.4}{:>10.4}", format!("ASIC {}", r.scenario.name), c.energy_usd, c.capex_usd, c.total());
    }
    println!();
    for r in &results {
        let a = market.asic(r).total();
        println!("   ASIC {} cost / H100-inference marginal = {:.2}x, / B200-inference marginal = {:.2}x  (<1 means ASIC is cheaper)",
            r.scenario.name, a / market.gpu_inference_marginal(h100).total(), a / market.gpu_inference_marginal(b200).total());
    }
}

fn sweep() {
    let act = measure_activity(16, 2048, 24);
    let af = ActivityFactors { operand: act.operand_alpha(), logic: act.tree_alpha() };
    let inv = AtomInventory::new(13);
    let market = Market::default();
    let gpus = energy::gpus();
    let (h100, b200) = (&gpus[0], &gpus[1]);
    let floor = market.gpu_inference_marginal(h100).total();
    println!("-- Voltage sweep: total USD/EFLOP (energy + 3-yr capex). '*' = cost-optimal, '!' = >1 W/mm^2 (thermally impractical)");
    for (node, cpp) in [(energy::n3(), 2.0), (energy::n2(), 4.0)] {
        let mut best: Option<(f64, f64)> = None;
        let mut rows = Vec::new();
        let mut v = 0.34;
        while v <= 0.801 {
            let mut s = pearl_asic_sim::designs::recommended().scenario;
            s.node = node.clone();
            s.vdd = v;
            s.cols_per_pe = cpp;
            let r = s.evaluate(&inv, af);
            let c = market.asic(&r);
            if r.w_per_mm2 <= 1.0 && best.is_none_or(|(_, bc)| c.total() < bc) {
                best = Some((v, c.total()));
            }
            rows.push((v, r, c));
            v += 0.04;
        }
        println!("   {} ({}-col PE)", node.name, cpp);
        println!("   {:>6}{:>8}{:>9}{:>10}{:>10}{:>12}{:>12}{:>12}", "Vdd", "GHz", "W/mm2", "TF/W wall", "xH100 wall", "energy $", "capex $", "total $");
        for (v, r, c) in rows {
            let mark = if best.is_some_and(|(bv, _)| (bv - v).abs() < 1e-9) { "*" } else if r.w_per_mm2 > 1.0 { "!" } else { " " };
            println!("  {mark}{:>6.2}{:>8.2}{:>9.2}{:>10.0}{:>10.0}{:>12.5}{:>12.5}{:>12.5}",
                v, r.f_ghz, r.w_per_mm2, r.tflops_per_w_wall, r.tflops_per_w_wall / h100.tflops_per_w_wall(), c.energy_usd, c.capex_usd, c.total());
        }
        if let Some((bv, bc)) = best {
            // Inference-fleet size at which ASICs are priced out, per $100k/day of block rewards.
            let h100_eflop_day = h100.dense_fp8_tflops * 1e12 * 86_400.0 / 1e18;
            let b200_eflop_day = b200.dense_fp8_tflops * 1e12 * 86_400.0 / 1e18;
            println!("   optimum {:.2} V: ${:.5}/EFLOP = {:.1}x the H100 inference-marginal cost (${:.6})", bv, bc, bc / floor, floor);
            println!("   ASICs stay profitable while the inference fleet mining Pearl at 100% GEMM duty is below");
            println!("   {:.0} H100-equivalents (or {:.0} B200s) per $100k/day of block rewards\n",
                100_000.0 / (bc * h100_eflop_day), 100_000.0 / (bc * b200_eflop_day));
        }
    }
}

fn demo() {
    // Easy-target demo: stream a few 16x16 tiles, hash tickets, count hits.
    let (h, w, k) = (16, 16, 1024);
    let mut g = OperandGen::new(5, 48.0);
    let b: Vec<u8> = (0..w).flat_map(|_| g.row(k)).collect();
    let mut core = Core::new(w, k, b);
    let lanes = engine::grid_lanes(h, w);
    let seed_a = [7u8; 32];
    let mut bound = [0xFFu8; 32];
    bound[31] = 0x1F; // 1/8 of tickets win
    let tiles = 256;
    let mut hits = 0;
    for t in 0..tiles {
        let tile: Vec<f32> = (0..h).flat_map(|_| { let r = g.row(k); core.stream_row(&r, false) }).collect();
        let msg = engine::xor_fold(&tile, &lanes);
        let j = engine::ticket(&seed_a, &msg);
        if engine::meets(&j, &bound) {
            hits += 1;
            if hits <= 5 {
                println!("tile {t:>3}: HIT  J=0x{}..", hex(&j));
            }
        }
    }
    println!("{hits}/{tiles} tickets met a 1/8 target (expect ~32); {} MACs; {} cycles/ticket at 1 row/cycle; fill latency {} cycles",
        core.activity.macs, engine::timing::cycles_per_ticket(h), engine::timing::fill_latency(w, k));
}

fn hex(b: &[u8]) -> String {
    b.iter().rev().take(8).map(|x| format!("{x:02x}")).collect()
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("demo") => demo(),
        Some("sweep") => sweep(),
        Some("figures") => {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
            pearl_asic_sim::figures::write_all(&dir).expect("write figures");
            println!("wrote {}/figures/*.svg, *.csv and {}/designs.md", dir.display(), dir.display());
        }
        _ => report(),
    }
}
