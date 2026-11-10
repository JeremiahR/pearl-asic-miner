# Energy, area and economics model: method and assumptions

Code: `src/energy.rs` and `src/econ.rs`. The CLI `cargo run --release` prints every intermediate number. Change a constant, rerun, and every table updates.

## 1. Activity (measured)

`engine::Core::stream_row(.., track = true)` records:

- **A′ operand-bus toggles**, the Hamming distance between consecutive rows, counted per column hop: α ≈ **0.44**;
- **adder-tree input toggles**, the 16-bit aligned two's-complement terms at each PE between consecutive rows: α ≈ **0.41**;
- **truncated products**, live products whose aligned value is 0: ≈ **0.25%**. The jackpot policy allows 5%.

Operands are `fp8(N(0, 48))`. That approximates noisy-quant output, which the rank-32 noise makes dense and close to i.i.d.

## 2. Gate inventory (`AtomInventory::new(window_bits)`)

Counts are in NAND2 gate-equivalents (GE): full adder = 6, AND = 1.33, XOR = 2, MUX2 = 2.33, flop = 6, latch = 4. Totals:

- **H100 semantics**: 21.4k GE per atom = **667 GE/MAC**
- **B200 semantics**: 30.7k GE per atom = 958 GE/MAC

The run prints a per-block breakdown.

## 3. Energy per MAC

```
logic   = GE_logic · α_logic · glitch(1.3) · e_ge(V) / 32
flops   = [(operand_flops + other_flops) · e_clk · (1 + clock_tree 0.25)
           + operand_flops · α_operand · e_data + other_flops · α_logic · e_data] / 32
          e_clk = e_data = 4 · e_ge; operand flops are divided by cols_per_pe
wire    = (256/cols_per_pe · α_operand + 32 · α_logic) · ½ · C_wire · pitch · V² / 32
leak    = area_GE · leak/GE · (V/Vref)³ / f(V) / 32
overhead = [32·core + 400·e_ge + 1.25 B · 0.2 pJ/B (BLAKE3)] / n_chip      (A′ generation)
         + epilogue + 3% controller/IO
wall    = chip / power_delivery_eff · (1 + cooling)
f(V)    = f_ref · [(V − Vt)^a / V] / [(Vref − Vt)^a / Vref]                (alpha-power law)
```

## 4. Key constants and where they come from

| constant | N3 | N2 | basis |
|---|---|---|---|
| `e_ge` (fJ/toggle at 0.75 V) | 0.12 | 0.096 | Horowitz 45 nm (8-bit add 0.03 pJ ≈ 1.25 fJ/GE-toggle) × ~1/7 capacitance × 0.69 V². **±2x uncertainty**; the CLI prints the sensitivity |
| Vt, alpha-power exponent | 0.30, 1.4 | 0.28, 1.35 | typical SVT/LVT FinFET/GAA fits |
| f at 0.75 V | 1.2 GHz | 1.3 GHz | 2-stage atom, deep CSA tree |
| area per GE | 0.040 µm² | 0.034 µm² | HD cells at ~70% utilisation |
| leakage per GE | 0.5 nW | 0.4 nW | mixed-Vt library |
| wire C | 0.20 fF/µm | 0.19 fF/µm | intermediate metal |
| wafer | $20k | $28k | industry estimates |

Cross-check: 3 nm SRAM digital-CIM macros report 125 TOPS/W INT8, about 16 fJ/MAC at macro level. The model's whole-chip FP8 figure for PA-2 (0.46 V) is 18.6 fJ/MAC.

## 5. Design versions (`designs::all()`, spec sheets in `designs.md`)

| | node | V | cols/PE | power delivery | cooling |
|---|---|---|---|---|---|
| PA-1 | N3 | 0.75 | 1 | 0.90 | 6% |
| PA-2 | N3 | 0.46 | 2 | 0.92 | 5% |
| PA-3 | N2 | 0.38 | 4 | 0.94 (voltage-stacked) | 4% |

All versions use `n_chip = 2048` columns, k = 2048, and 15% non-array die area.

## 6. GPU baselines (`energy::gpus()`)

| GPU | dense FP8 | board | wall per GPU | capex |
|---|---|---|---|---|
| H100 SXM | 1,979 TF | 700 W | 1,275 W (DGX H100 10.2 kW / 8) | $30k |
| B200 | 4,500 TF | 1,000 W | 1,788 W (DGX B200 14.3 kW / 8) | $45k |

These use nameplate throughput, which flatters the GPU. The 2:4-sparse peak is excluded because the liveness rule rejects structured sparsity.

### Frontier families (`energy::frontier_gpus()`, headline figure)

| family | members | wall TFLOPS/W | basis |
|---|---|---|---|
| Hopper | H100 / H200 | 1.55 | 1,979 TF; DGX 10.2 kW / 8 |
| Blackwell | B200 / B300 / GB200 | 2.52–2.67 | 4.5 / 4.5 / 5.0 PF; DGX 14.3 kW/8, DGX 14 kW/8, 1.87 kW all-in |
| Rubin (announced) | VR200 | 4.9–6.2 | 17.5 PF; 1.8–2.3 kW board (unconfirmed) × GB200 wall/board ratio 1.56 |

Effective Pearl efficiency for an inference GPU = wall TFLOPS/W ÷ overhead. The ASIC's "all-in" line converts its total USD/EFLOP (energy + capex) into the energy-equivalent TFLOPS/W at the same electricity price.

## 7. Economics (`econ::Market`)

Defaults: $0.05/kWh, 3 years, 95% uptime, 1% inference overhead, $10 package per die, $0.80/W for system hardware, 80% die yield, 300 mm wafers with 12% edge loss.

- **ASIC** cost per EFLOP = wall energy + (die + package + system $/W × wall W) ÷ lifetime FLOPs.
- **Pure-GPU** cost per EFLOP = wall energy + GPU capex ÷ lifetime FLOPs.
- **Inference-GPU marginal** cost = 1% × pure-GPU energy, with zero capex (inference pays for the hardware).
- **Viability threshold**: ASICs are priced out when the inference fleet's Pearl work rate W_inf reaches R / c_ASIC, where R is block rewards in $/day. The sweep prints this in H100- and B200-equivalents per $100k/day.

## 8. Known limitations

- No synthesis yet. `e_ge`, the flop costs and the wire model are estimates; replace them with post-layout numbers from a real PDK.
- Glitch, clock-tree and IR-drop factors are generic.
- Voltage stacking is modelled only as a better power-delivery efficiency.
- Leakage at near-threshold is sensitive to the library's Vt mix.
- NRE (masks and design, tens of millions of USD) is **not** in the per-EFLOP capex. It is a go/no-go hurdle; see the risks in `REPORT.md`.
- The operand generator approximates noisy-quant output. Policy compliance of the real generator should be checked end-to-end with `verify_plain_proof_for_cert_version`.
