//! Bit-exact, hardware-structured model of the ASIC's core datapath.
//!
//! Pearl's FP8 (cert v4) verifier replays the winning tile *bit-exactly* on a
//! committed device's tensor-core semantics (`Device::H100` or `Device::B200`).
//! Difficulty is `target * h * w * k` for both devices, so a miner is free to
//! pick the cheaper one. H100 semantics are much cheaper in silicon: the
//! accumulation window keeps 13 fractional bits (vs 25 on B200), so the
//! 33-input adder tree is ~16 bits wide instead of ~28.
//!
//! The model is written the way the RTL would be: fixed-width integer fields,
//! one function per pipeline block, no floating-point arithmetic inside the
//! atom. The only IEEE op is the per-window FP32 RNE promotion add, which is a
//! standard FP32 adder block (`f32 + f32` in Rust is exactly that).
//!
//! ```text
//!   a[0..32] b[0..32]   (E4M3 codes)        carry (fp32, window-local)
//!      │   │                                  │
//!   ┌──▼───▼──┐ x32   PRODUCT UNIT            │ CARRY UNPACK
//!   │ 4b x 4b │       sign xor, exp add,      │ top-14 significand bits
//!   └──┬──────┘       8-bit significand       │
//!      └──────────────┬───────────────────────┘
//!               ┌─────▼──────┐  MAX-EXP TREE (33 -> 1, live terms only)
//!               └─────┬──────┘
//!               ┌─────▼──────┐  ALIGN: 15-bit right shift, sat. at 15, trunc
//!               └─────┬──────┘  COND. NEGATE (into CSA carry-in)
//!               ┌─────▼──────┐  CSA TREE 33 -> 2, then 22-bit CPA
//!               └─────┬──────┘
//!               ┌─────▼──────┐  LZC + NORMALISE to 14 bits (truncate),
//!               └─────┬──────┘  subnormal clamp, repack to fp32 bits
//!                     ▼
//!               new carry (fp32)
//! ```
//!
//! A *window* chains up to 4 atoms (128 products) starting from +0, then is
//! promoted into the cell's running FP32 total with one RNE add. Windows are
//! independent until promotion, which is what lets the array compute all
//! windows of a cell in parallel and pipeline the promotions.

use crate::fp8;

/// Products per atom (hardware accumulation group).
pub const ATOM_K: usize = 32;
/// Atoms per promotion window (H100: 128 products).
pub const WINDOW_ATOMS: usize = 4;
pub const WINDOW_K: usize = ATOM_K * WINDOW_ATOMS;
/// Significand width kept by the H100 accumulation window (1 + 13 fractional).
pub const SIG_W: u32 = 14;
/// Max useful alignment shift: aligned terms are < 2^15, so a shift of 15+
/// always yields 0. The hardware shifter therefore only needs 4 control bits
/// plus a saturation flag.
pub const MAX_SHIFT: u32 = 15;

const FP32_MIN_EXP: i32 = -126;

/// One adder-tree input. Value = `sig * 2^(exp - 13)`.
///
/// `live` is the non-zero detect on the *full-width* operand: a term can be
/// live (participating in the max-exponent tree) even though its 14-bit
/// projection `sig` is zero (a tiny subnormal carry). This matches the
/// reference and costs one OR-reduce in hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub neg: bool,
    pub exp: i32,
    pub sig: u32,
    pub live: bool,
}

/// PRODUCT UNIT: exact E4M3 x E4M3 product. 4-bit x 4-bit significand
/// multiplier (hidden bit included), 4+4-bit exponent adder, sign xor.
#[inline]
pub fn product(a: u8, b: u8) -> Term {
    let (sa, ea) = fp8::unpack(a);
    let (sb, eb) = fp8::unpack(b);
    let p = sa as u32 * sb as u32; // <= 15*15 = 225, 8 bits
    Term {
        neg: ((a ^ b) & 0x80) != 0,
        exp: ea as i32 + eb as i32 - 14,
        sig: p << 7, // pure wiring
        live: p != 0,
    }
}

/// CARRY UNPACK: fp32 carry -> 14-bit window term (truncation toward zero).
#[inline]
pub fn carry_term(x: f32) -> Term {
    let bits = x.to_bits();
    let neg = bits >> 31 != 0;
    let field = (bits >> 23) & 0xFF;
    let man = bits & 0x7F_FFFF;
    if field == 0 && man == 0 {
        return Term { neg, exp: 0, sig: 0, live: false };
    }
    let (exp, sig24) = if field != 0 { (field as i32 - 127, man | 0x80_0000) } else { (FP32_MIN_EXP, man) };
    Term { neg, exp, sig: sig24 >> 10, live: true }
}

/// Result of one atom, plus side-band observables used by the activity /
/// policy statistics (not part of the datapath).
#[derive(Clone, Copy, Debug)]
pub struct AtomOut {
    pub value: f32,
    /// Live products whose aligned contribution truncated to zero.
    pub dropped: u32,
}

/// MAX-EXP TREE + ALIGN + CSA/CPA + NORMALISE: one H100 accumulation group.
/// Returns `None` only on FP32 overflow (never happens on policy-valid data;
/// the reference rejects it as non-finite).
pub fn group_sum(terms: &[Term]) -> Option<AtomOut> {
    // Max-exponent tree over live terms.
    let mut anchor: Option<i32> = None;
    for t in terms.iter().filter(|t| t.live) {
        anchor = Some(anchor.map_or(t.exp, |m: i32| m.max(t.exp)));
    }
    let Some(anchor) = anchor else {
        return Some(AtomOut { value: 0.0, dropped: 0 });
    };

    // Align (saturating shifter) + conditional negate + adder tree.
    let mut acc: i32 = 0; // 33 * 2^15 < 2^21: a 22-bit signed CPA suffices
    let mut dropped = 0;
    for t in terms.iter().filter(|t| t.live) {
        let sh = (anchor - t.exp) as u32;
        let aligned = if sh >= MAX_SHIFT { 0 } else { t.sig >> sh };
        if aligned == 0 {
            dropped += 1;
        }
        acc += if t.neg { -(aligned as i32) } else { aligned as i32 };
    }
    debug_assert!(acc.unsigned_abs() < 1 << 21);

    // Normalise: LZC, shift to 14 bits (truncate), subnormal clamp.
    let neg = acc < 0;
    let mut mag = acc.unsigned_abs();
    let signed_zero = |neg: bool| f32::from_bits((neg as u32) << 31);
    if mag == 0 {
        return Some(AtomOut { value: signed_zero(false), dropped });
    }
    let width = 32 - mag.leading_zeros() as i32;
    let mut exp = anchor + width - SIG_W as i32;
    if width > SIG_W as i32 {
        mag >>= (width - SIG_W as i32) as u32;
    } else {
        mag <<= (SIG_W as i32 - width) as u32;
    }
    if exp < FP32_MIN_EXP {
        let sh = (FP32_MIN_EXP - exp) as u32;
        mag = if sh >= 32 { 0 } else { mag >> sh };
        exp = FP32_MIN_EXP;
    }
    if mag == 0 {
        return Some(AtomOut { value: signed_zero(neg), dropped });
    }
    // Repack: 14-bit significand -> fp32 fraction field (low 10 bits zero).
    let mut field = exp + 127;
    if mag & (1 << (SIG_W - 1)) == 0 {
        field -= 1; // subnormal
    }
    if field >= 255 {
        return None;
    }
    let bits = ((neg as u32) << 31) | ((field as u32) << 23) | ((mag << 10) & 0x7F_FFFF);
    Some(AtomOut { value: f32::from_bits(bits), dropped })
}

/// One atom: carry + 32 fresh products.
#[inline]
pub fn atom(carry: f32, a: &[u8], b: &[u8]) -> Option<AtomOut> {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= ATOM_K);
    let mut terms = [Term { neg: false, exp: 0, sig: 0, live: false }; ATOM_K + 1];
    terms[0] = carry_term(carry);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        terms[i + 1] = product(x, y);
    }
    group_sum(&terms[..a.len() + 1])
}

/// One promotion window: up to 4 chained atoms from +0.
pub fn window(a: &[u8], b: &[u8]) -> Option<(f32, u32)> {
    let mut local = 0.0f32;
    let mut dropped = 0;
    for (ca, cb) in a.chunks(ATOM_K).zip(b.chunks(ATOM_K)) {
        let out = atom(local, ca, cb)?;
        local = out.value;
        dropped += out.dropped;
    }
    Some((local, dropped))
}

/// PROMOTE: ascending FP32 RNE accumulation of window results; -0 is
/// canonicalised to +0 after every add, as on the device.
#[inline]
pub fn promote(total: f32, local: f32) -> Option<f32> {
    let t = total + local;
    if !t.is_finite() {
        return None;
    }
    Some(if t == 0.0 { 0.0 } else { t })
}

/// Full output cell `<a_row, b_row>` with H100 semantics.
pub fn cell(a_row: &[u8], b_row: &[u8]) -> Option<f32> {
    assert!(a_row.len() == b_row.len() && a_row.len().is_multiple_of(ATOM_K));
    let mut total = 0.0f32;
    for (wa, wb) in a_row.chunks(WINDOW_K).zip(b_row.chunks(WINDOW_K)) {
        total = promote(total, window(wa, wb)?.0)?;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_dot_products() {
        let one = fp8::from_f32(1.0);
        let two = fp8::from_f32(2.0);
        let a = vec![one; 128];
        let b = vec![two; 128];
        assert_eq!(cell(&a, &b), Some(256.0));
        let z = vec![0u8; 64];
        assert_eq!(cell(&z, &z).unwrap().to_bits(), 0);
    }

    #[test]
    fn truncation_drops_tiny_terms() {
        // 448*448 dominates; 2^-9 * 2^-9 is ~2^-35 relative: dropped.
        let mut a = vec![0u8; 32];
        let mut b = vec![0u8; 32];
        a[0] = fp8::from_f32(448.0);
        b[0] = fp8::from_f32(448.0);
        a[1] = 0x01;
        b[1] = 0x01;
        let out = atom(0.0, &a, &b).unwrap();
        assert_eq!(out.value, 448.0 * 448.0);
        assert_eq!(out.dropped, 1);
    }
}
