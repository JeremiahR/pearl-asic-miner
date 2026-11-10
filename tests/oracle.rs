//! Bit-exactness against Pearl's own FP8 verifier (zk-pow @ fp8-scheme
//! 630b7316). If these pass, a winning tile found by the ASIC replays to the
//! same bits on the verifier, so its ticket is valid.

use pearl_asic_sim::{atom, engine};
use zk_pow::api::fp8::utils::{H100, xor_fold_extract};
use zk_pow::api::layout::{AxisPattern, DimType, lane_assignment};

fn reference(a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    H100 {}.matmul_fp8_policy_replay(a, b, m, n, k).expect("reference replay").0
}

fn model(a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = atom::cell(&a[i * k..(i + 1) * k], &b[j * k..(j + 1) * k]).expect("overflow");
        }
    }
    out
}

fn assert_bits_eq(x: &[f32], y: &[f32], what: &str) {
    for (i, (p, q)) in x.iter().zip(y).enumerate() {
        assert_eq!(p.to_bits(), q.to_bits(), "{what}: cell {i} model={p:e} ({:#010x}) ref={q:e} ({:#010x})", p.to_bits(), q.to_bits());
    }
}

#[test]
fn gaussian_mining_data_matches_reference() {
    let mut g = engine::OperandGen::new(1, 48.0);
    for &k in &[32usize, 96, 128, 160, 1024, 4096] {
        let (m, n) = (4, 16);
        let a: Vec<u8> = (0..m).flat_map(|_| g.row(k)).collect();
        let b: Vec<u8> = (0..n).flat_map(|_| g.row(k)).collect();
        assert_bits_eq(&model(&a, &b, m, n, k), &reference(&a, &b, m, n, k), &format!("gauss k={k}"));
    }
}

#[test]
fn adversarial_codes_match_reference() {
    // Uniform over all non-NaN codes: huge dynamic range, subnormals,
    // cancellation, signed zeros. Small k keeps FP32 from overflowing.
    let mut g = engine::OperandGen::new(7, 1.0);
    for trial in 0..3000 {
        let k = 32 * (1 + trial % 9);
        let (m, n) = (3, 5);
        let a = g.random_codes(m * k);
        let b = g.random_codes(n * k);
        assert_bits_eq(&model(&a, &b, m, n, k), &reference(&a, &b, m, n, k), &format!("random trial {trial}"));
    }
}

#[test]
fn tiny_and_cancelling_values_match_reference() {
    let mut g = engine::OperandGen::new(9, 1.0);
    for trial in 0..200 {
        let k = 128;
        // Mostly subnormal/small codes with exact cancellations.
        let mut a: Vec<u8> = g.random_codes(k).into_iter().map(|c| c & 0x8F).collect();
        let b: Vec<u8> = g.random_codes(k).into_iter().map(|c| c & 0x9F).collect();
        if trial % 2 == 0 {
            for i in 0..k / 2 {
                a[k / 2 + i] = a[i] ^ 0x80; // exact negation against same-b? only if b equal
            }
        }
        let b2: Vec<u8> = b.iter().take(k / 2).chain(b.iter().take(k / 2)).copied().collect();
        assert_bits_eq(&model(&a, &b2, 1, 1, k), &reference(&a, &b2, 1, 1, k), &format!("tiny trial {trial}"));
    }
}

#[test]
fn core_dataflow_and_ticket_match_reference() {
    let (h, w, k) = (16usize, 16usize, 1024usize);
    let mut g = engine::OperandGen::new(3, 48.0);
    let a: Vec<u8> = (0..h).flat_map(|_| g.row(k)).collect();
    let b: Vec<u8> = (0..w).flat_map(|_| g.row(k)).collect();

    let mut core = engine::Core::new(w, k, b.clone());
    let tile: Vec<f32> = (0..h).flat_map(|i| core.stream_row(&a[i * k..(i + 1) * k], false)).collect();
    assert_bits_eq(&tile, &reference(&a, &b, h, w, k), "core tile");

    // Lane layout from the protocol's own pattern machinery.
    let rows = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)]).unwrap();
    let cols = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)]).unwrap();
    let lanes = lane_assignment(&rows, &cols);
    let msg = engine::xor_fold(&tile, &lanes);
    assert_eq!(msg, xor_fold_extract(&tile, &lanes), "fold");

    let seed_a = [0x5Au8; 32];
    let t = engine::ticket(&seed_a, &msg);
    assert_eq!(t, zk_pow::api::fp8::transcript::compute_jackpot_ticket(&seed_a, &msg).jackpot, "ticket hash");
}
