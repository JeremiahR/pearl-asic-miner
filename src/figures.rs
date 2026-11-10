//! Generates `docs/figures/*.svg` (+ CSV table views) and `docs/designs.md`
//! from the model, so documentation numbers always match the code.

use crate::designs::{self, Design};
use crate::econ::{Cost, Market};
use crate::energy::{self, ActivityFactors, AtomInventory, Breakdown, Gpu};
use crate::engine::{Activity, Core, OperandGen};
use crate::plot::{LinAxis, LogAxis, Svg, fmt_num, fmt_tick};
use std::fmt::Write as _;
use std::path::Path;

pub struct DesignResult {
    pub design: Design,
    pub r: Breakdown,
    pub cost: Cost,
    /// Wall TFLOPS/W with switching energy x2 (pessimistic) and x0.5 (optimistic).
    pub tfw_lo: f64,
    pub tfw_hi: f64,
}

pub struct SweepPoint {
    pub vdd: f64,
    pub r: Breakdown,
    pub cost: Cost,
}

pub struct Results {
    pub activity: Activity,
    pub inv: AtomInventory,
    pub gpus: Vec<Gpu>,
    pub market: Market,
    pub designs: Vec<DesignResult>,
    pub sweeps: Vec<(&'static str, Vec<SweepPoint>)>,
}

pub fn measure_activity(cols: usize, k: usize, rows: usize) -> Activity {
    let mut g = OperandGen::new(42, 48.0);
    let b: Vec<u8> = (0..cols).flat_map(|_| g.row(k)).collect();
    let mut core = Core::new(cols, k, b);
    for _ in 0..rows {
        let r = g.row(k);
        core.stream_row(&r, true);
    }
    core.activity
}

pub fn compute() -> Results {
    let activity = measure_activity(16, 2048, 48);
    let af = ActivityFactors { operand: activity.operand_alpha(), logic: activity.tree_alpha() };
    let inv = AtomInventory::new(13);
    let market = Market::default();
    let designs = designs::all()
        .into_iter()
        .map(|d| {
            let r = d.scenario.evaluate(&inv, af);
            let cost = market.asic(&r);
            let mut lo = d.scenario.clone();
            lo.node.e_ge_fj *= 2.0;
            let mut hi = d.scenario.clone();
            hi.node.e_ge_fj *= 0.5;
            let tfw_lo = lo.evaluate(&inv, af).tflops_per_w_wall;
            let tfw_hi = hi.evaluate(&inv, af).tflops_per_w_wall;
            DesignResult { design: d, r, cost, tfw_lo, tfw_hi }
        })
        .collect();
    let base = designs::recommended().scenario;
    let mut sweeps = Vec::new();
    for (label, node, cpp) in [("N3 (2-col PE)", energy::n3(), 2.0), ("N2 (4-col PE)", energy::n2(), 4.0)] {
        let mut pts = Vec::new();
        let mut v = 0.34;
        while v <= 0.781 {
            let mut s = base.clone();
            s.node = node.clone();
            s.vdd = v;
            s.cols_per_pe = cpp;
            let r = s.evaluate(&inv, af);
            let cost = market.asic(&r);
            pts.push(SweepPoint { vdd: v, r, cost });
            v += 0.02;
        }
        sweeps.push((label, pts));
    }
    Results { activity, inv, gpus: energy::gpus(), market, designs, sweeps }
}

// ---------------------------------------------------------------------------

fn log_ticks(svg: &mut Svg, ax: &LogAxis, top: f64, bottom: f64, ticks: &[f64], fmt: &dyn Fn(f64) -> String) {
    for &t in ticks {
        let x = ax.map(t);
        svg.line(x, top, x, bottom, "grid");
        svg.text(x, bottom + 16.0, &fmt(t), "tick", "middle");
    }
}

/// Figure 1: net (wall-plug) efficiency of every design version and GPU.
fn fig_efficiency(res: &Results) -> (String, String) {
    let (w, row_h, top) = (780.0, 46.0, 92.0);
    let rows = res.designs.len() + res.gpus.len();
    let h = top + row_h * rows as f64 + 54.0;
    let mut svg = Svg::new(w, h, "Net efficiency: ASIC design versions vs NVIDIA GPUs",
        "Wall-plug FP8 dense TFLOPS per watt, log scale. Whiskers: switching energy x2 (left) to x0.5 (right).");
    svg.legend(20.0, 72.0, &[("f1", "ASIC design version"), ("f2", "GPU (nameplate, DGX wall power)")]);
    let ax = LogAxis { lo: 1.0, hi: 600.0, a: 190.0, b: w - 150.0 };
    let bottom = top + row_h * rows as f64;
    log_ticks(&mut svg, &ax, top - 6.0, bottom, &[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], &|t| format!("{t:.0}"));
    svg.line(ax.a, bottom, ax.b, bottom, "base");
    let h100 = res.gpus[0].tflops_per_w_wall();
    let mut csv = String::from("name,kind,wall_tflops_per_w,low,high,x_h100\n");
    let mut y = top + row_h / 2.0;
    for d in res.designs.iter().rev() {
        let (v, lo, hi) = (d.r.tflops_per_w_wall, d.tfw_lo, d.tfw_hi);
        svg.text(20.0, y - 2.0, &format!("{} {}", d.design.version, d.design.codename), "lab", "start");
        svg.text(20.0, y + 13.0, &format!("{} · {:.2} V", d.design.scenario.node.name.replace("TSMC ", "").replace("-class", ""), d.design.scenario.vdd), "lab2", "start");
        svg.line(ax.map(lo), y, ax.map(hi), y, "w1");
        svg.line(ax.map(lo), y - 5.0, ax.map(lo), y + 5.0, "w1");
        svg.line(ax.map(hi), y - 5.0, ax.map(hi), y + 5.0, "w1");
        svg.dot(ax.map(v), y, 6.0, "f1");
        svg.text(ax.map(hi) + 10.0, y + 4.0, &format!("{} TF/W · {:.0}× H100", fmt_num(v), v / h100), "val", "start");
        let _ = writeln!(csv, "{} {},asic,{v:.2},{lo:.2},{hi:.2},{:.1}", d.design.version, d.design.codename, v / h100);
        y += row_h;
    }
    for g in res.gpus.iter().rev() {
        let v = g.tflops_per_w_wall();
        svg.text(20.0, y - 2.0, g.name, "lab", "start");
        svg.text(20.0, y + 13.0, &format!("GPU · {:.0} W board", g.board_w), "lab2", "start");
        svg.dot(ax.map(v), y, 6.0, "f2");
        svg.text(ax.map(v) + 12.0, y + 4.0, &format!("{:.2} TF/W · {:.1}× H100", v, v / h100), "val", "start");
        let _ = writeln!(csv, "{},gpu,{v:.2},{v:.2},{v:.2},{:.2}", g.name, v / h100);
        y += row_h;
    }
    svg.text(ax.b, bottom + 34.0, "TFLOPS/W at the wall (log)", "note", "end");
    (svg.finish("Net efficiency of ASIC design versions vs H100 and B200"), csv)
}

/// Figure 2: efficiency multiple vs H100 and vs B200 per design version.
fn fig_multiples(res: &Results) -> (String, String) {
    let (w, top, group_h) = (780.0, 96.0, 74.0);
    let h = top + group_h * res.designs.len() as f64 + 50.0;
    let mut svg = Svg::new(w, h, "How many times more efficient than a GPU?",
        "Wall-to-wall FP8 efficiency multiple, log scale, with the x0.5 / x2 switching-energy band.");
    svg.legend(20.0, 74.0, &[("f1", "vs H100 SXM"), ("f2", "vs B200")]);
    let ax = LogAxis { lo: 3.0, hi: 400.0, a: 190.0, b: w - 110.0 };
    let bottom = top + group_h * res.designs.len() as f64;
    log_ticks(&mut svg, &ax, top - 8.0, bottom, &[3.0, 10.0, 30.0, 100.0, 300.0], &|t| format!("{t:.0}×"));
    svg.line(ax.a, bottom, ax.b, bottom, "base");
    let x100 = ax.map(100.0);
    svg.line(x100, top - 8.0, x100, bottom, "ref");
    svg.text(x100 + 4.0, top - 12.0, "100× goal", "lab2", "start");
    let mut csv = String::from("version,vs,multiple,low,high\n");
    let mut y = top + 22.0;
    for d in &res.designs {
        svg.text(20.0, y + 6.0, &format!("{} {}", d.design.version, d.design.codename), "lab", "start");
        svg.text(20.0, y + 21.0, d.design.status.split(" — ").next().unwrap_or(""), "lab2", "start");
        for (i, g) in res.gpus.iter().enumerate() {
            let base = g.tflops_per_w_wall();
            let (m, lo, hi) = (d.r.tflops_per_w_wall / base, d.tfw_lo / base, d.tfw_hi / base);
            let yy = y + i as f64 * 22.0;
            let (wc, fc) = if i == 0 { ("w1", "f1") } else { ("w2", "f2") };
            svg.line(ax.map(lo), yy, ax.map(hi), yy, wc);
            svg.dot(ax.map(m), yy, 5.0, fc);
            svg.text(ax.map(hi) + 8.0, yy + 4.0, &format!("{:.0}× {}", m, g.name.split(' ').next().unwrap()), "val", "start");
            let _ = writeln!(csv, "{},{},{m:.1},{lo:.1},{hi:.1}", d.design.version, g.name);
        }
        y += group_h;
    }
    (svg.finish("Efficiency multiple of each ASIC design version vs H100 and B200"), csv)
}

/// Figure 3: total cost per EFLOP across the voltage sweep, vs the
/// inference-GPU marginal floor.
fn fig_cost_sweep(res: &Results) -> (String, String) {
    let (w, h) = (780.0, 430.0);
    let mut svg = Svg::new(w, h, "Cheapest operating point: total cost of Pearl work vs core voltage",
        "USD per EFLOP (energy + 3-year capex), log scale. Lower is better.");
    svg.legend(20.0, 72.0, &[("f1", "N3 ASIC"), ("f2", "N2 ASIC")]);
    let (l, r, t, b) = (80.0, w - 120.0, 92.0, h - 56.0);
    let x = LinAxis { lo: 0.34, hi: 0.78, a: l, b: r };
    let y = LogAxis { lo: 4e-5, hi: 1.5e-3, a: b, b: t };
    // Thermal-limit region (>1 W/mm^2 for either node).
    let hot_v = res.sweeps.iter().flat_map(|(_, p)| p.iter().filter(|p| p.r.w_per_mm2 > 1.0).map(|p| p.vdd)).fold(f64::MAX, f64::min) - 0.01;
    svg.rect(x.map(hot_v), t, r - x.map(hot_v), b - t, 0.0, "wash");
    svg.text(x.map(hot_v) + 6.0, t + 14.0, "> 1 W/mm²: too hot to cool", "note", "start");
    for v in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3] {
        let yy = y.map(v);
        svg.line(l, yy, r, yy, "grid");
        svg.text(l - 8.0, yy + 4.0, &format!("${}", fmt_tick(v)), "tick", "end");
    }
    for v in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75] {
        svg.text(x.map(v), b + 16.0, &format!("{v:.2}"), "tick", "middle");
    }
    svg.line(l, b, r, b, "base");
    svg.text(r, b + 34.0, "core voltage (V)", "note", "end");
    // Inference-marginal floors.
    for g in &res.gpus {
        let c = res.market.gpu_inference_marginal(g).total();
        let yy = y.map(c);
        svg.line(l, yy, r, yy, "ref");
        svg.text(r + 6.0, yy + 4.0, &format!("{} inference", g.name.split(' ').next().unwrap()), "lab2", "start");
        svg.text(r + 6.0, yy + 17.0, &format!("by-product ${}", fmt_num(c)), "lab2", "start");
    }
    let mut csv = String::from("node,vdd,ghz,w_per_mm2,wall_tflops_per_w,energy_usd_per_eflop,capex_usd_per_eflop,total_usd_per_eflop\n");
    for (i, (label, pts)) in res.sweeps.iter().enumerate() {
        let (lc, fc) = if i == 0 { ("l1", "f1") } else { ("l2", "f2") };
        let line: Vec<(f64, f64)> = pts.iter().map(|p| (x.map(p.vdd), y.map(p.cost.total()))).collect();
        svg.polyline(&line, lc);
        let best = pts.iter().filter(|p| p.r.w_per_mm2 <= 1.0).min_by(|a, b| a.cost.total().total_cmp(&b.cost.total())).unwrap();
        let (bx, by) = (x.map(best.vdd), y.map(best.cost.total()));
        svg.dot(bx, by, 6.0, fc);
        let dy = if i == 0 { -14.0 } else { 22.0 };
        svg.text(bx, by + dy, &format!("{} optimum {:.2} V: ${}", label.split(' ').next().unwrap(), best.vdd, fmt_num(best.cost.total())), "val", "middle");
        let last = pts.last().unwrap();
        svg.text(x.map(last.vdd) + 6.0, y.map(last.cost.total()) + 4.0, label, "lab", "start");
        for p in pts {
            let _ = writeln!(csv, "{label},{:.2},{:.3},{:.3},{:.1},{:.6},{:.6},{:.6}", p.vdd, p.r.f_ghz, p.r.w_per_mm2, p.r.tflops_per_w_wall, p.cost.energy_usd, p.cost.capex_usd, p.cost.total());
        }
    }
    (svg.finish("Total cost per EFLOP vs core voltage for N3 and N2 ASICs, with GPU inference floors"), csv)
}

/// Figure 4: efficiency multiple vs voltage (energy-optimal vs cost-optimal).
fn fig_eff_sweep(res: &Results) -> String {
    let (w, h) = (780.0, 400.0);
    let mut svg = Svg::new(w, h, "Lower voltage buys efficiency, until silicon cost takes over",
        "Wall-to-wall efficiency multiple vs H100 as core voltage falls. Dots mark each node's cost-optimal point.");
    svg.legend(20.0, 72.0, &[("f1", "N3 ASIC"), ("f2", "N2 ASIC")]);
    let (l, r, t, b) = (70.0, w - 110.0, 92.0, h - 56.0);
    let x = LinAxis { lo: 0.34, hi: 0.78, a: l, b: r };
    let y = LinAxis { lo: 0.0, hi: 160.0, a: b, b: t };
    for v in [0.0, 40.0, 80.0, 120.0, 160.0] {
        svg.line(l, y.map(v), r, y.map(v), if v == 0.0 { "base" } else { "grid" });
        svg.text(l - 8.0, y.map(v) + 4.0, &format!("{v:.0}×"), "tick", "end");
    }
    for v in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75] {
        svg.text(x.map(v), b + 16.0, &format!("{v:.2}"), "tick", "middle");
    }
    svg.text(r, b + 34.0, "core voltage (V)", "note", "end");
    svg.line(l, y.map(100.0), r, y.map(100.0), "ref");
    svg.text(r + 6.0, y.map(100.0) + 4.0, "100× goal", "lab2", "start");
    let h100 = res.gpus[0].tflops_per_w_wall();
    for (i, (label, pts)) in res.sweeps.iter().enumerate() {
        let (lc, fc) = if i == 0 { ("l1", "f1") } else { ("l2", "f2") };
        let line: Vec<(f64, f64)> = pts.iter().map(|p| (x.map(p.vdd), y.map(p.r.tflops_per_w_wall / h100))).collect();
        svg.polyline(&line, lc);
        let best = pts.iter().filter(|p| p.r.w_per_mm2 <= 1.0).min_by(|a, b| a.cost.total().total_cmp(&b.cost.total())).unwrap();
        let m = best.r.tflops_per_w_wall / h100;
        svg.dot(x.map(best.vdd), y.map(m), 6.0, fc);
        let (dx, dy, anchor) = if i == 0 { (-10.0, 24.0, "end") } else { (10.0, -10.0, "start") };
        svg.text(x.map(best.vdd) + dx, y.map(m) + dy, &format!("{:.0}× at {:.2} V (cost-optimal)", m, best.vdd), "val", anchor);
        let last = pts.last().unwrap();
        let ly = y.map(last.r.tflops_per_w_wall / h100) + if i == 0 { 14.0 } else { -6.0 };
        svg.text(x.map(last.vdd) + 6.0, ly, label, "lab", "start");
    }
    svg.finish("Efficiency multiple vs H100 across core voltage for N3 and N2")
}

/// Figure 5: where the energy goes, per design version.
fn fig_breakdown(res: &Results) -> (String, String) {
    let (w, top, row_h) = (780.0, 96.0, 50.0);
    let h = top + row_h * res.designs.len() as f64 + 64.0;
    let mut svg = Svg::new(w, h, "Where the energy goes",
        "Chip energy per FP8 MAC (fJ), by block. For scale, an H100 spends 707 fJ/MAC at the board.");
    svg.legend(20.0, 74.0, &[("f1", "datapath logic"), ("f2", "registers + clock"), ("f3", "operand wires"), ("f4", "leakage + A′ gen + IO")]);
    let ax = LinAxis { lo: 0.0, hi: 60.0, a: 170.0, b: w - 120.0 };
    let bottom = top + row_h * res.designs.len() as f64;
    for v in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0] {
        svg.line(ax.map(v), top - 6.0, ax.map(v), bottom, if v == 0.0 { "base" } else { "grid" });
        svg.text(ax.map(v), bottom + 16.0, &format!("{v:.0}"), "tick", "middle");
    }
    svg.text(ax.b, bottom + 34.0, "fJ per MAC (chip)", "note", "end");
    let mut csv = String::from("version,logic_fj,registers_clock_fj,wires_fj,leak_overhead_fj,chip_fj,wall_fj\n");
    let mut y = top + 10.0;
    for d in &res.designs {
        let r = &d.r;
        let parts = [(r.fj_logic, "f1"), (r.fj_flops, "f2"), (r.fj_wire, "f3"), (r.fj_leak + r.fj_overhead, "f4")];
        svg.text(20.0, y + 13.0, &format!("{} {}", d.design.version, d.design.codename), "lab", "start");
        let mut x = ax.map(0.0);
        let bh = 22.0;
        for (i, (v, c)) in parts.iter().enumerate() {
            let wpx = ax.map(*v) - ax.map(0.0);
            let seg = (wpx - 2.0).max(1.0); // 2px surface gap between segments
            if i == parts.len() - 1 {
                svg.bar_end_rounded(x, y, seg, bh, 4.0, c);
            } else {
                svg.rect(x, y, seg, bh, 0.0, c);
            }
            x += wpx;
        }
        svg.text(x + 8.0, y + 15.0, &format!("{:.1} fJ chip · {:.1} wall", r.fj_chip, r.fj_wall), "val", "start");
        let _ = writeln!(csv, "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}", d.design.version, r.fj_logic, r.fj_flops, r.fj_wire, r.fj_leak + r.fj_overhead, r.fj_chip, r.fj_wall);
        y += row_h;
    }
    (svg.finish("Energy breakdown per FP8 MAC for each ASIC design version"), csv)
}

/// Figure 6: cost per EFLOP for every kind of miner.
fn fig_cost(res: &Results) -> (String, String) {
    let mut rows: Vec<(String, String, &str, Cost)> = Vec::new();
    for g in &res.gpus {
        rows.push((format!("{} bought to mine", g.name), "energy + GPU capex".into(), "f2", res.market.gpu_pure(g)));
    }
    for d in &res.designs {
        rows.push((format!("ASIC {} {}", d.design.version, d.design.codename), "energy + chip/board capex".into(), "f1", d.cost.clone()));
    }
    for g in &res.gpus {
        rows.push((format!("{} doing inference", g.name), "1% marginal energy only".into(), "f2", res.market.gpu_inference_marginal(g)));
    }
    let (w, top, row_h) = (780.0, 96.0, 40.0);
    let h = top + row_h * rows.len() as f64 + 56.0;
    let mut svg = Svg::new(w, h, "What a unit of Pearl work costs each kind of miner",
        "USD per EFLOP (10¹⁸ FP8 FLOP), log scale. $0.05/kWh, 3-year life. Lower is better.");
    svg.legend(20.0, 74.0, &[("f1", "ASIC"), ("f2", "GPU")]);
    let ax = LogAxis { lo: 2e-5, hi: 0.5, a: 230.0, b: w - 130.0 };
    let bottom = top + row_h * rows.len() as f64;
    log_ticks(&mut svg, &ax, top - 6.0, bottom, &[1e-4, 1e-3, 1e-2, 1e-1], &|t| format!("${}", fmt_tick(t)));
    svg.line(ax.a, bottom, ax.b, bottom, "base");
    let mut csv = String::from("miner,basis,energy_usd,capex_usd,total_usd\n");
    let mut y = top + row_h / 2.0;
    for (name, basis, class, c) in &rows {
        svg.text(20.0, y - 1.0, name, "lab", "start");
        svg.text(20.0, y + 13.0, basis, "lab2", "start");
        svg.line(ax.a, y, ax.map(c.total()), y, "grid");
        svg.dot(ax.map(c.total()), y, 6.0, class);
        svg.text(ax.map(c.total()) + 12.0, y + 4.0, &format!("${}", fmt_num(c.total())), "val", "start");
        let _ = writeln!(csv, "{name},{basis},{:.6},{:.6},{:.6}", c.energy_usd, c.capex_usd, c.total());
        y += row_h;
    }
    (svg.finish("Cost per EFLOP of Pearl work for GPUs and ASIC design versions"), csv)
}

/// Overhead levels for the inference-mining sensitivity analysis.
pub const OVERHEADS: [f64; 5] = [0.01, 0.02, 0.03, 0.04, 0.05];

/// Overhead at which an ASIC's total cost equals a GPU's inference-marginal cost.
pub fn breakeven_overhead(res: &Results, d: &DesignResult, g: &Gpu) -> f64 {
    d.cost.total() / res.market.gpu_pure(g).energy_usd
}

/// Figure 7: inference-GPU marginal cost at 1-5% mining overhead vs each
/// ASIC version's total cost.
fn fig_overhead(res: &Results) -> (String, String) {
    let (w, h) = (780.0, 440.0);
    let mut svg = Svg::new(w, h, "How the answer depends on Pearl's mining overhead",
        "Marginal cost of Pearl work for GPUs already doing FP8 inference, at 1–5% overhead, vs each ASIC version's total cost.");
    svg.legend(20.0, 72.0, &[("f1", "H100 doing inference"), ("f2", "B200 doing inference")]);
    let (l, r, t, b) = (90.0, w - 190.0, 96.0, h - 56.0);
    let x = LinAxis { lo: 0.01, hi: 0.05, a: l, b: r };
    let y = LogAxis { lo: 4e-5, hi: 1.2e-3, a: b, b: t };
    for v in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3] {
        svg.line(l, y.map(v), r, y.map(v), "grid");
        svg.text(l - 8.0, y.map(v) + 4.0, &format!("${}", fmt_tick(v)), "tick", "end");
    }
    for o in OVERHEADS {
        svg.text(x.map(o), b + 16.0, &format!("{:.0}%", o * 100.0), "tick", "middle");
    }
    svg.line(l, b, r, b, "base");
    svg.text(r, b + 34.0, "Pearl mining overhead on an inference GPU", "note", "end");
    // ASIC versions as labelled reference lines.
    // Versions whose costs are within 10% share one labelled line.
    let mut groups: Vec<(f64, Vec<String>)> = Vec::new();
    for d in &res.designs {
        let c = d.cost.total();
        let label = format!("{} ${}", d.design.version, fmt_num(c));
        match groups.iter_mut().find(|(gc, _)| (c / *gc - 1.0).abs() < 0.10) {
            Some((_, v)) => v.push(label),
            None => groups.push((c, vec![label])),
        }
    }
    for (c, labels) in &groups {
        svg.line(l, y.map(*c), r, y.map(*c), "ref");
        for (i, lab) in labels.iter().enumerate() {
            svg.text(r + 6.0, y.map(*c) + 4.0 + 14.0 * i as f64, &format!("ASIC {lab}"), "lab2", "start");
        }
    }
    let mut csv = String::from("overhead_pct,gpu,marginal_usd_per_eflop");
    for d in &res.designs {
        let _ = write!(csv, ",{}_cost_ratio", d.design.version);
    }
    csv.push('\n');
    for (i, g) in res.gpus.iter().enumerate() {
        let (lc, fc) = if i == 0 { ("l1", "f1") } else { ("l2", "f2") };
        let energy = res.market.gpu_pure(g).energy_usd;
        let pts: Vec<(f64, f64)> = OVERHEADS.iter().map(|&o| (x.map(o), y.map(energy * o))).collect();
        svg.polyline(&pts, lc);
        for (&o, &(px, py)) in OVERHEADS.iter().zip(&pts) {
            svg.dot(px, py, 4.5, fc);
            let _ = write!(csv, "{:.0},{},{:.6}", o * 100.0, g.name, energy * o);
            for d in &res.designs {
                let _ = write!(csv, ",{:.2}", d.cost.total() / (energy * o));
            }
            csv.push('\n');
        }
        let (lx, ly) = *pts.last().unwrap();
        svg.text(lx - 8.0, ly - 12.0, &format!("{} ${}", g.name.split(' ').next().unwrap(), fmt_num(energy * 0.05)), "val", "end");
        // Break-even marker against the recommended PA-2.
        let pa2 = res.designs.iter().find(|d| d.design.version == "PA-2").unwrap();
        let be = breakeven_overhead(res, pa2, g);
        if be <= 0.05 {
            let bx = x.map(be);
            svg.line(bx, t, bx, y.map(pa2.cost.total()), "ref");
            svg.dot(bx, y.map(pa2.cost.total()), 4.5, fc);
            svg.text(bx - 6.0, t + 12.0, &format!("PA-2 breaks even with {} at {:.1}%", g.name.split(' ').next().unwrap(), be * 100.0), "lab2", "end");
        }
    }
    svg.text(l + 6.0, b - 8.0, "Where a GPU line sits below an ASIC line, inference-mined work is cheaper than that ASIC.", "note", "start");
    (svg.finish("Inference-GPU marginal cost at 1-5% mining overhead vs ASIC design versions"), csv)
}

/// Headline figure: effective Pearl efficiency of frontier NVIDIA GPUs mining
/// as a by-product of inference (wall TFLOPS/W / overhead) vs our ASICs.
fn fig_frontier(res: &Results) -> (String, String) {
    let fams = energy::frontier_gpus();
    let pa2 = res.designs.iter().find(|d| d.design.version == "PA-2").unwrap();
    let pa3 = res.designs.iter().find(|d| d.design.version == "PA-3").unwrap();
    let e_pa2 = pa2.r.tflops_per_w_wall;
    let e_pa3 = pa3.r.tflops_per_w_wall;
    // All-in: total USD/EFLOP converted to the energy-equivalent TFLOPS/W.
    let j_per_eflop = pa2.cost.total() / (res.market.usd_per_kwh / 3.6e6);
    let e_pa2_allin = 1e6 / j_per_eflop;

    let (w, h) = (820.0, 600.0);
    let mut svg = Svg::new(w, h, "The bar our best ASIC must clear moves with Pearl's overhead, and with every NVIDIA generation",
        "Effective Pearl efficiency of a GPU already doing FP8 inference = wall TFLOPS/W ÷ mining overhead. Labels: 1% → 5%. Log scale.");
    svg.legend(20.0, 72.0, &[("f1", "Hopper H100/H200"), ("f2", "Blackwell B200/B300/GB200"), ("f3", "Rubin VR200 (announced)")]);
    let (l, r, t, b) = (70.0, w - 230.0, 100.0, h - 110.0);
    let x = LinAxis { lo: 0.01, hi: 0.05, a: l, b: r };
    let y = LogAxis { lo: 20.0, hi: 1000.0, a: b, b: t };
    // Regions relative to the PA-2 line.
    let ya = y.map(e_pa2);
    svg.rect(l, ya, r - l, b - ya, 0.0, "wash");
    // Region labels: who wins on either side of our best ASIC's line.
    svg.text(r - 12.0, t + 30.0, "GPUs WIN", "region", "end");
    svg.text(r - 12.0, t + 48.0, "inference GPUs mine Pearl more efficiently than our ASIC", "regionsub", "end");
    svg.text(l + 12.0, b - 30.0, "ASIC WINS", "region", "start");
    svg.text(l + 12.0, b - 12.0, "our ASIC mines Pearl more efficiently (shaded)", "regionsub", "start");
    for v in [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0] {
        svg.line(l, y.map(v), r, y.map(v), "grid");
        svg.text(l - 8.0, y.map(v) + 4.0, &format!("{v:.0}"), "tick", "end");
    }
    for o in OVERHEADS {
        svg.text(x.map(o), b + 16.0, &format!("{:.0}%", o * 100.0), "tick", "middle");
    }
    svg.line(l, b, r, b, "base");
    svg.text(r, b + 34.0, "Pearl mining overhead on an inference GPU", "note", "end");
    svg.text(l - 8.0, t - 12.0, "effective TFLOPS/W", "note", "start");

    // ASIC reference lines.
    let refs = [
        (e_pa3, format!("PA-3 stretch: {:.0} TF/W", e_pa3), "ref", "lab2"),
        (e_pa2, format!("Our best ASIC (PA-2): {:.0} TF/W", e_pa2), "refbold", "strong"),
        (e_pa2_allin, format!("PA-2 incl. chip capex: ~{:.0} TF/W-eq.", e_pa2_allin), "ref", "lab2"),
    ];
    for (v, label, lc, tc) in &refs {
        svg.line(l, y.map(*v), r, y.map(*v), lc);
        svg.text(r + 8.0, y.map(*v) + 4.0, label, tc, "start");
    }
    svg.text(r + 8.0, ya - 18.0, "▲ above: GPUs win", "lab", "start");
    svg.text(r + 8.0, ya + 24.0, "▼ below: ASIC wins", "lab", "start");

    let mut csv = String::from("family,members,overhead_pct,effective_tflops_per_w_low,effective_tflops_per_w_high,vs_pa2_energy_low,vs_pa2_energy_high\n");
    let mut notes = Vec::new();
    for (i, f) in fams.iter().enumerate() {
        let (lc, bc, fc) = [("l1", "b1", "f1"), ("l2", "b2", "f2"), ("l3", "b3", "f3")][i];
        let lo: Vec<(f64, f64)> = OVERHEADS.iter().map(|&o| (x.map(o), y.map(f.wall_tfw_lo / o))).collect();
        let hi: Vec<(f64, f64)> = OVERHEADS.iter().map(|&o| (x.map(o), y.map(f.wall_tfw_hi / o))).collect();
        // Smooth curve: sample finely.
        let n = 40;
        let curve = |e: f64| -> Vec<(f64, f64)> {
            (0..=n).map(|j| { let o = 0.01 + 0.04 * j as f64 / n as f64; (x.map(o), y.map(e / o)) }).collect()
        };
        if f.wall_tfw_hi > f.wall_tfw_lo * 1.001 {
            let mut band = curve(f.wall_tfw_hi);
            band.extend(curve(f.wall_tfw_lo).into_iter().rev());
            svg.polygon(&band, bc);
        }
        let mid = (f.wall_tfw_lo * f.wall_tfw_hi).sqrt();
        svg.polyline(&curve(mid), lc);
        for (&o, (pl, ph)) in OVERHEADS.iter().zip(lo.iter().zip(&hi)) {
            let _ = writeln!(csv, "{},{},{:.0},{:.1},{:.1},{:.2},{:.2}", f.name, f.members, o * 100.0, f.wall_tfw_lo / o, f.wall_tfw_hi / o, f.wall_tfw_lo / o / e_pa2, f.wall_tfw_hi / o / e_pa2);
            let _ = (pl, ph);
        }
        // Curve labels placed where no line crosses the text.
        let label = format!("{}: {:.0} → {:.0} TF/W", f.name, mid / 0.01, mid / 0.05);
        match i {
            0 => svg.text(x.map(0.029), y.map(f.wall_tfw_lo / 0.029) + 18.0, &label, "val", "end"),
            1 => svg.text(x.map(0.049), y.map(f.wall_tfw_lo / 0.049) + 18.0, &label, "val", "end"),
            _ => svg.text(x.map(0.013), y.map(f.wall_tfw_hi / 0.013) - 10.0, &format!("{label} (power unconfirmed)"), "val", "start"),
        }
        // Crossover with the PA-2 line (overhead where GPU effective = PA-2).
        let (c_lo, c_hi) = (f.wall_tfw_lo / e_pa2, f.wall_tfw_hi / e_pa2);
        if c_lo <= 0.05 {
            let cx = x.map((c_lo * c_hi).sqrt());
            svg.dot(cx, ya, 5.5, fc);
            let txt = if (c_hi - c_lo) / c_lo < 0.02 { format!("{:.1}%", c_lo * 100.0) } else { format!("{:.1}–{:.1}%", c_lo * 100.0, c_hi * 100.0) };
            svg.text(cx, ya + 18.0, &txt, "val", "middle");
        }
        notes.push(format!("{}: {:.1}–{:.1} wall TF/W ({})", f.members, f.wall_tfw_lo, f.wall_tfw_hi, f.basis));
    }
    let rubin = &fams[2];
    svg.text(20.0, h - 40.0,
        &format!("Rubin stays above our best ASIC at every overhead up to 5% (it would cross at {:.1}–{:.1}%). Dots mark where each GPU family falls below PA-2.",
            rubin.wall_tfw_lo / e_pa2 * 100.0, rubin.wall_tfw_hi / e_pa2 * 100.0), "lab", "start");
    svg.text(20.0, h - 20.0, "Energy only, GPUs at nameplate dense FP8. Counting ASIC chip capex (PA-2 all-in line), even an H100 wins up to ~4.7% overhead.", "note", "start");
    let _ = h;
    for n in notes {
        let _ = writeln!(csv, "# {n}");
    }
    (svg.finish("Effective Pearl efficiency of frontier NVIDIA GPUs at 1-5% overhead vs our best ASIC"), csv)
}

// ---------------------------------------------------------------------------

pub fn designs_markdown(res: &Results) -> String {
    let h100 = &res.gpus[0];
    let b200 = &res.gpus[1];
    let mut s = String::new();
    s.push_str("# ASIC design versions\n\n");
    s.push_str("*Generated by `cargo run --release -- figures` from `src/designs.rs`. Do not edit by hand.*\n\n");
    s.push_str("All versions share the verified DP32 datapath (H100 semantics, `src/atom.rs`), the weight-stationary core (`src/engine.rs`), 2048 output columns × k = 2048 per die, and on-chip A′ generation. Efficiency is net, i.e. wall-plug including power delivery and cooling. GPUs are at nameplate dense FP8 with DGX wall power.\n\n");
    s.push_str("## Summary\n\n| version | codename | status | node · Vdd | PE | TFLOPS/W (wall) | × H100 | × B200 | USD/EFLOP |\n|---|---|---|---|---|---|---|---|---|\n");
    for d in &res.designs {
        let sc = &d.design.scenario;
        let _ = writeln!(s, "| **{}** | {} | {} | {} · {:.2} V | {}-col | {:.0} ({:.0}–{:.0}) | {:.0}× | {:.0}× | ${:.5} |",
            d.design.version, d.design.codename, d.design.status, sc.node.name, sc.vdd, sc.cols_per_pe,
            d.r.tflops_per_w_wall, d.tfw_lo, d.tfw_hi, d.r.tflops_per_w_wall / h100.tflops_per_w_wall(),
            d.r.tflops_per_w_wall / b200.tflops_per_w_wall(), d.cost.total());
    }
    let _ = writeln!(s, "| H100 SXM | GPU | — | TSMC 4N | — | {:.2} | 1× | {:.2}× | ${:.4} pure / ${:.6} inference |",
        h100.tflops_per_w_wall(), h100.tflops_per_w_wall() / b200.tflops_per_w_wall(), res.market.gpu_pure(h100).total(), res.market.gpu_inference_marginal(h100).total());
    let _ = writeln!(s, "| B200 | GPU | — | TSMC 4NP | — | {:.2} | {:.2}× | 1× | ${:.4} pure / ${:.6} inference |\n",
        b200.tflops_per_w_wall(), b200.tflops_per_w_wall() / h100.tflops_per_w_wall(), res.market.gpu_pure(b200).total(), res.market.gpu_inference_marginal(b200).total());
    s.push_str("The range in parentheses is the switching-energy band (×2 pessimistic to ×0.5 optimistic).\n\n");
    s.push_str("![Net efficiency vs GPUs](figures/efficiency.svg)\n\n");
    s.push_str("## Sensitivity to Pearl's mining overhead\n\nRatio of each ASIC's total cost per EFLOP to the marginal cost for a GPU that is already doing FP8 inference, at different mining overheads. Below 1.0 the ASIC is cheaper. Break-even is the overhead at which the two are equal.\n\n| version | vs | 1% | 2% | 3% | 4% | 5% | break-even overhead |\n|---|---|---|---|---|---|---|---|\n");
    for d in &res.designs {
        for g in &res.gpus {
            let e = res.market.gpu_pure(g).energy_usd;
            let _ = write!(s, "| {} | {} inference |", d.design.version, g.name);
            for o in OVERHEADS {
                let _ = write!(s, " {:.2}× |", d.cost.total() / (e * o));
            }
            let _ = writeln!(s, " **{:.1}%** |", breakeven_overhead(res, d, g) * 100.0);
        }
    }
    s.push_str("\n![Overhead sensitivity](figures/overhead-sensitivity.svg)\n\n");
    for d in &res.designs {
        let sc = &d.design.scenario;
        let r = &d.r;
        let _ = writeln!(s, "## {} — {}\n\n**Status:** {}  \n{}\n\n**Changes:**\n", d.design.version, d.design.codename, d.design.status, d.design.summary);
        for c in d.design.changes {
            let _ = writeln!(s, "- {c}");
        }
        let _ = writeln!(s, "\n| spec | value |\n|---|---|");
        let rows = [
            ("process", sc.node.name.to_string()),
            ("core voltage", format!("{:.2} V", sc.vdd)),
            ("clock", format!("{:.2} GHz", r.f_ghz)),
            ("die area", format!("{:.0} mm²", r.die_mm2)),
            ("FP8 MACs / cycle", format!("{:.2} M", sc.n_chip * sc.k / 1e6)),
            ("throughput", format!("{:.1} PFLOPS FP8 dense", r.pflops_per_chip)),
            ("chip power", format!("{:.0} W", r.watts_per_chip)),
            ("power density", format!("{:.2} W/mm²", r.w_per_mm2)),
            ("logic density", format!("{:.0} GE/MAC", r.ge_per_mac)),
            ("energy (chip / wall)", format!("{:.1} / {:.1} fJ per MAC", r.fj_chip, r.fj_wall)),
            ("efficiency (chip / wall)", format!("{:.0} / {:.0} TFLOPS/W", r.tflops_per_w_chip, r.tflops_per_w_wall)),
            ("vs H100 (wall-to-wall)", format!("{:.0}× ({:.0}–{:.0}×)", r.tflops_per_w_wall / h100.tflops_per_w_wall(), d.tfw_lo / h100.tflops_per_w_wall(), d.tfw_hi / h100.tflops_per_w_wall())),
            ("vs B200 (wall-to-wall)", format!("{:.0}× ({:.0}–{:.0}×)", r.tflops_per_w_wall / b200.tflops_per_w_wall(), d.tfw_lo / b200.tflops_per_w_wall(), d.tfw_hi / b200.tflops_per_w_wall())),
            ("die + package cost", format!("${:.0}", res.market.asic_chip_usd(r))),
            ("USD per EFLOP", format!("${:.5} (energy {:.5} + capex {:.5})", d.cost.total(), d.cost.energy_usd, d.cost.capex_usd)),
            ("vs H100 inference by-product", format!("{:.1}× its marginal cost", d.cost.total() / res.market.gpu_inference_marginal(h100).total())),
        ];
        for (k, v) in rows {
            let _ = writeln!(s, "| {k} | {v} |");
        }
        s.push('\n');
    }
    s
}

pub fn write_all(dir: &Path) -> std::io::Result<Results> {
    let res = compute();
    let figs = dir.join("figures");
    std::fs::create_dir_all(&figs)?;
    let put = |name: &str, body: &str| std::fs::write(figs.join(name), body);
    let (svg, csv) = fig_efficiency(&res);
    put("efficiency.svg", &svg)?;
    put("efficiency.csv", &csv)?;
    let (svg, csv) = fig_multiples(&res);
    put("vs-gpus.svg", &svg)?;
    put("vs-gpus.csv", &csv)?;
    let (svg, csv) = fig_cost_sweep(&res);
    put("cost-sweep.svg", &svg)?;
    put("voltage-sweep.csv", &csv)?;
    put("efficiency-sweep.svg", &fig_eff_sweep(&res))?;
    let (svg, csv) = fig_breakdown(&res);
    put("energy-breakdown.svg", &svg)?;
    put("energy-breakdown.csv", &csv)?;
    let (svg, csv) = fig_frontier(&res);
    put("frontier-gpus-vs-asic.svg", &svg)?;
    put("frontier-gpus-vs-asic.csv", &csv)?;
    let (svg, csv) = fig_overhead(&res);
    put("overhead-sensitivity.svg", &svg)?;
    put("overhead-sensitivity.csv", &csv)?;
    let (svg, csv) = fig_cost(&res);
    put("cost-per-eflop.svg", &svg)?;
    put("cost-per-eflop.csv", &csv)?;
    std::fs::write(dir.join("designs.md"), designs_markdown(&res))?;
    Ok(res)
}
