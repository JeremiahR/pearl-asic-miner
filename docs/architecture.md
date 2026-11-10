# Chip microarchitecture

The golden reference for everything here is `src/atom.rs` (datapath) and `src/engine.rs` (dataflow). Both are bit-exact against Pearl's verifier; see `tests/oracle.rs`.

## Hierarchy

| level | contents | numbers (PA-2: N3, 0.46 V) |
|---|---|---|
| board | voltage-stacked string of ~25 chips, controller, Ethernet | ~0.9 kW, ~97 PFLOPS |
| chip | A′ generator, 128 cores, epilogues, controller | ~130 mm², 4.2M MAC/cycle, ~0.46 GHz, ~3.9 PFLOPS, ~36 W |
| core | 16 output columns × 64 atom slots (k = 2048) | 65,536 MAC/cycle |
| PE | 2 columns sharing one A′ operand register (4 on N2) | |
| DP32 atom | 32 FP8 products + carry → one 14-bit-significand result | 32 MAC/cycle, ~21k GE |

## The DP32 atom (H100 semantics)

```
 a[0..32], b[0..32] (E4M3)                      carry (FP32, window-local)
   ▼ ×32 PRODUCT: 4b×4b significand (hidden bit incl.) → 8b; exp 4b+4b; sign xor; live = (p ≠ 0)
   ▼                                           ▼ CARRY UNPACK: top 14 of 24 significand bits; live = full-width ≠ 0
 MAX-EXP TREE   max of 33 live exponents (8-bit compare-select ×32)
 SHIFT AMOUNT   anchor − exp, saturate at 15
 ALIGN          15-bit right shifter (4 stages), truncation toward zero
 COND. NEGATE   XOR + carry-in into the CSA
 CSA 33→2       16-bit terms; CPA 22-bit signed
 NORMALISE      LZC; shift to 14 bits (truncate); exponent = anchor + width − 14; clamp at 2⁻¹²⁶ (subnormal)
 PACK           FP32 bits = sign ‖ field ‖ (sig14 << 10)
```

Bit widths (proven sufficient by the model and its oracle tests):

- aligned term < 2¹⁵, 33 terms → |sum| < 2²¹ → 22-bit signed accumulator;
- a shift ≥ 15 always yields 0, so the shifter needs 4 control bits plus a saturate flag;
- **`live` must be the non-zero detect of the full-width operand**, not of the 14-bit projection. A tiny subnormal carry still participates in the max-exponent tree. Getting this wrong breaks bit-exactness on rare inputs.

**Window and promotion.** Four atoms chain (carry from atom c to c+1) starting from +0. The window result goes through an FP32 RNE add into the column's running total, and −0 is canonicalised to +0 after each add. Windows are independent until promotion, so all k/128 windows of a cell run in parallel. The promotion is a pipelined chain of FP32 adders, one per window.

## Core dataflow (weight-stationary systolic)

```
            B′ stationary: PE(j, c) holds B′[j, 32c .. 32c+32] in clock-gated latches
 A′ row i ─► col 0 ─► col 1 ─► … ─► col 15 ─► next core …   (A′ snakes through all 2048 chip columns)
             │ atoms c=0..3 chain the carry ↓ (window 0) … atoms 4..7 (window 1) …
             ▼ window results → FP32 promotion chain (ascending) → C[i, j]
```

- **One A′ row per cycle** per core. Rows are skewed by atom slot so that each window's chain and the promotion chain are fully pipelined.
- **Fill latency** is about `4 + k/128 + cols` cycles: 28 for k = 1024 and 16 columns.
- **Operand energy**: each A′ byte moves one PE pitch per use (per `cols_per_pe` columns). B′ never moves, the 32-bit carry hops once per atom, and outputs leave once per k MACs.

## A′ generator (one per chip)

Pass 1 (commit):
- PRNG(seed) → BF16 A rows → prequant (int8 codes + BF16 scale per 8 codes) → BLAKE3 Merkle tree → `H_A`;
- controller → `seedA`, jackpot sub-key.

Pass 2 (compute):
- regenerate the same rows;
- generate the E_A line per row (BLAKE3 XOF, 32 values, normalised);
- compute E_A·F_A on a small bank of DP32 atoms (F_A, k × 32 FP8 = 64 KB SRAM), one atom per A′ element;
- noisy-quantise → FP8 A′ row → array.

Cost is amortised over 2048 columns. The noise costs 32 MAC-equivalents per element, i.e. 32/2048 = 1.6% of core energy before the other terms. The whole generator plus epilogue and IO is about 0.9 fJ/MAC of 18.6 for PA-2.

The B′ side runs once per B commitment and ancestor window: generate B, commit (`H_B`), noise, quantise, and load the latches over a slow scan chain.

## Epilogue (per core)

- **Tile buffer**: 16 rows × 16 columns × FP32 = 1 KB.
- **Fold unit**: 16 lanes, each doing a 32-bit multiply, add and rotate.
- **Ticket**: one BLAKE3 compression per tile, keyed by the jackpot sub-key, then a U256 comparison against `target·h·w·k`.
- **Throughput**: 1 ticket per 16 cycles per core, about 3.7 billion tickets per second per chip (PA-2).
- **On a hit**: push (job id, tile base, row and column ids) to the controller FIFO.

## Host integration

1. The host (or pool) issues a job: block template, ancestor header, A and B seeds, dimensions, tile pattern and target.
2. On a hit, the host regenerates the winning rows from seeds and builds the Merkle openings and `PlainProof` with `pearl-mining`. It submits through `pearl-gateway`, which produces the plonky2 ZK proof and the block.

The ASIC never needs the ZK prover.

## Recommended protocol parameters

| parameter | value | reason |
|---|---|---|
| device | H100 | 30% fewer GE/MAC; same difficulty |
| k | 2048 | keeps B′ latch storage at 32 B per atom; no effect on work per win |
| tile | 16 × 16 = 256 | one tile per 16 cycles; minimum epilogue buffering |
| m per commitment | ~2²⁰ rows | amortises pass-1 hashing; job length of seconds |
| n per B commitment | ~16k–64k columns | spans a board; B′ reused for the ancestor window |
