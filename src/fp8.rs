//! OCP FP8 E4M3 ("E4M3FN") helpers. Bias 7, no infinities, `S.1111.111` is NaN
//! (banned by the protocol, so the ASIC never generates it).

/// Unpack to (significand with hidden bit, effective exponent field >= 1).
/// Value = `sig * 2^(exp - 10)`.
#[inline]
pub fn unpack(x: u8) -> (u8, u8) {
    let e = (x >> 3) & 0x0F;
    let m = x & 0x07;
    if e == 0 { (m, 1) } else { (m | 0x08, e) }
}

pub fn to_f32(x: u8) -> f32 {
    let (s, e) = unpack(x & 0x7F);
    let v = s as f32 * 2f32.powi(e as i32 - 10);
    if x & 0x80 != 0 { -v } else { v }
}

/// Round-to-nearest-even, saturating at +-448 (matches `f32_to_fp8_e4m3`).
pub fn from_f32(x: f32) -> u8 {
    assert!(x.is_finite());
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let m = x.abs();
    let bits = m.to_bits();
    let e = (bits >> 23) as i32 - 127;
    let mag = if e < -6 {
        (m * 512.0).round_ties_even() as u8
    } else {
        let mut exp = (e + 7) as u32;
        let fm = bits & 0x7F_FFFF;
        let mut mant = fm >> 20;
        let rem = fm & 0xF_FFFF;
        if rem > 0x8_0000 || (rem == 0x8_0000 && mant & 1 == 1) {
            mant += 1;
            if mant == 8 {
                mant = 0;
                exp += 1;
            }
        }
        if exp > 15 || (exp == 15 && mant >= 7) { 0x7E } else { ((exp << 3) | mant) as u8 }
    };
    sign | mag
}

pub fn is_nan(x: u8) -> bool {
    x & 0x7F == 0x7F
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_all_codes() {
        for c in 0u8..=255 {
            if is_nan(c) || c == 0x80 {
                continue;
            }
            assert_eq!(from_f32(to_f32(c)), c, "code {c:#04x}");
        }
        assert_eq!(to_f32(0x7E), 448.0);
        assert_eq!(to_f32(0x01), 2f32.powi(-9));
    }
}
