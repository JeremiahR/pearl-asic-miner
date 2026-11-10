# Pearl's FP8 proof of work: what a miner must compute

This document reconstructs the cert-v4 FP8 scheme from the Pearl source, branch `fp8-scheme` @ `630b731641ef31d706634b3baf3838369cb2d204`. File references are relative to the Pearl repository.

## Pipeline

```
 clean A (m×k, BF16)            clean B (n×k, BF16)
      │ pre_quant                    │ pre_quant           int8 codes + BF16 scale per 8 codes (1.25 B/elem)
      ▼                              ▼
 Merkle(BLAKE3, keyA) = H_A     Merkle(BLAKE3, keyB) = H_B
      │                              │
      │      seedB = H("seed-B", H_B ‖ keyB ‖ pB)          keyB = H("key-B", ancestor header)
      └──► seedA = H("seed-A", H_A ‖ seedB ‖ keyA ‖ pA)    keyA = H("key-A", proposed header)
      │                              │
 E_A (rows, seedA), F_A (k×32, seedB)   E_B, F_B (seedB)    rank-32 noise lines from keyed BLAKE3
      ▼                              ▼
 A′ = fp8(A/α + E_A·F_A)        B′ = fp8(B/α + E_B·F_B)    FP8 E4M3, per-row scale α
      └──────────── C = A′·B′ᵀ  (bit-exact device semantics) ────────────┘
                                │
                   tile T (h×w cells) at a valid base
                                ▼
         fold: lane = rotl32(lane·0x9E3779B1 + bits(C_ij), 13) over 16 lanes → 64 B
                                ▼
         J = BLAKE3(fold, key = BLAKE3("pearl/v4/FP8/jackpot", key = seedA))
                                ▼
                win iff  J ≤ target · h · w · k   (little-endian U256)
```

Sources: `zk-pow/src/api/fp8/transcript.rs` (key and seed chain, ticket), `noise.rs`, `quantization.rs`, `prequant.rs`, `zk-pow/src/api/proof_utils.rs::check_jackpot_difficulty`, `zk-pow/src/api/layout.rs` (tile and lane layout).

## Device semantics (`zk-pow/src/api/fp8/utils.rs`)

The proof commits to a `Device` byte. The verifier replays the winning tile on that device's exact FP8 MMA datapath.

| | H100 (`Device = 0`) | B200 (`Device = 1`) |
|---|---|---|
| group | carry + 32 exact products | carry + 32 exact products |
| alignment | to the group's max stored exponent; each term truncated toward zero | same |
| kept bits below anchor | **13** (14-bit significand) | 25 (26-bit) |
| chaining | 4 atoms form a 128-product window starting from +0; the window is promoted into an FP32 total with one RNE add, ascending | groups chained continuously; result rounded toward zero |
| noise scale `δ` | 1 | 1/2 |

Difficulty does not depend on the device, so a miner picks the cheaper one: **H100**.

## Constraints a miner must respect

- `r = 32`, `32 | k`, `1024 ≤ k ≤ 65536`; `m, n ≤ 2²⁴` (`public_params.rs`).
- Tile: `h ≥ 4`, `w ≥ 16`, `256 ≤ h·w ≤ 2048`, exactly 16 lanes, each subtile within `MIN_SUBTILE_ELEMS..=MAX_SUBTILE_ELEMS` (min 16) (`layout.rs::check_lottery_layout`).
- No NaN or infinity anywhere in the pipeline.
- **Jackpot policy** (`jackpot_policy.rs`), evaluated on the opened tile:
  1. **Liveness**: at most 1/64 of entries per side may have `|x| ≥ 8·δ·‖row‖₂`. This rules out spiky or structured-sparse data.
  2. **Noise floor**: every row's noise σ must be at least 1.
  3. **Unpredictable summands**: at most `k·h·w/20` products may be ones the hardware window would drop.

## Why there is no shortcut

- **The verifier hashes the final FP32 bits of every tile cell**, and each cell depends on all k products through a non-associative truncation chain. Any reordering (Strassen, Winograd, different tree shapes) changes bits.
- **Noise is per row and per column, keyed by seeds** that commit to the whole of A and B. Duplicated or zero rows do not produce duplicate A′ rows.
- **The policy forbids sparse or low-entropy inputs**, so there is no zero-skipping. In simulation, only 0.25% of products truncate away.
- **The ticket count is proportional to MACs**: one ticket per h·w cells, each cell k MACs, and difficulty scales by h·w·k. Tile size and k don't change the work per expected win.

## Useful properties for an ASIC

- **B′ depends only on B and an ancestor header**, so it can stay resident for many blocks.
- **The miner chooses A and B freely.** Any data that passes the policy is valid, so both can be generated on chip from a seed. A is generated twice (hash pass, then compute pass) instead of being stored.
- **Opening a win is cheap.** The host can regenerate the winning rows from seeds and build the Merkle openings and `PlainProof` in software. The plonky2 ZK proof is done by the pool or gateway, as today.

## Fork history (risk context)

| fork | change | source |
|---|---|---|
| MoE (V2 cert) | MoE mining support | `docs/moe-fork-upgrade-guide.md` |
| Rank penalty (soft fork, mainnet height 96251) | closed a work-accounting gap where a larger noise rank bought an easier bound | commit `51a6e11b` |
| Salted seed (V3, mainnet height 99000) | binds Merkle roots to matrix dimensions | commit `fc5ca65a`, `docs/salted-seed-fork-upgrade-guide.md` |
| FP8 (V4) | this scheme; activation pending | branch `fp8-scheme` |
