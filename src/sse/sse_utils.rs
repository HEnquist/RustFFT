use core::arch::x86_64::*;

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy)]
pub struct SseF32x4(__m128);

// Standard math ops
impl Add for SseF32x4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self( unsafe {_mm_add_ps(self.0, other.0)})
    }
}
impl Sub for SseF32x4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self( unsafe {_mm_sub_ps(self.0, other.0)})
    }
}
// Standard math ops
impl Mul for SseF32x4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self( unsafe {_mm_mul_ps(self.0, other.0)})
    }
}

impl SseF32x4 {
    pub fn new(values: __m128) -> Self {
        Self(values)
    }

    pub fn get(self) -> __m128 {
        self.clone().0
    }
}

//  __  __       _   _               _________  _     _ _
// |  \/  | __ _| |_| |__           |___ /___ \| |__ (_) |_
// | |\/| |/ _` | __| '_ \   _____    |_ \ __) | '_ \| | __|
// | |  | | (_| | |_| | | | |_____|  ___) / __/| |_) | | |_
// |_|  |_|\__,_|\__|_| |_|         |____/_____|_.__/|_|\__|
//

pub struct Rotate90F32 {
    //sign_1st: SseF32x4,
    sign_2nd: SseF32x4,
    sign_both: SseF32x4,
}

impl Rotate90F32 {
    pub fn new(positive: bool) -> Self {
        // There doesn't seem to be any need for rotating just the first element, but let's keep the code just in case
        //let sign_1st = unsafe {
        //    if positive {
        //        _mm_set_ps(0.0, 0.0, 0.0, -0.0)
        //    }
        //    else {
        //        _mm_set_ps(0.0, 0.0, -0.0, 0.0)
        //    }
        //};
        let sign_2nd = unsafe {
            if positive {
                SseF32x4::new(_mm_set_ps(0.0, -0.0, 0.0, 0.0))
            } else {
                SseF32x4::new(_mm_set_ps(-0.0, 0.0, 0.0, 0.0))
            }
        };
        let sign_both = unsafe {
            if positive {
                SseF32x4::new(_mm_set_ps(0.0, -0.0, 0.0, -0.0))
            } else {
                SseF32x4::new(_mm_set_ps(-0.0, 0.0, -0.0, 0.0))
            }
        };
        Self {
            //sign_1st,
            sign_2nd,
            sign_both,
        }
    }

    #[inline(always)]
    pub unsafe fn rotate_2nd(&self, values: &SseF32x4) -> SseF32x4 {
        let temp = _mm_shuffle_ps(values.get(), values.get(), 0xB4);
        SseF32x4::new(_mm_xor_ps(temp, self.sign_2nd.get()))
    }

    // There doesn't seem to be any need for rotating just the first element, but let's keep the code just in case
    //#[inline(always)]
    //pub unsafe fn rotate_1st(&self, values: SseF32x4) -> SseF32x4 {
    //    let temp = _mm_shuffle_ps(values, values, 0xE1);
    //    _mm_xor_ps(temp, self.sign_1st)
    //}

    #[inline(always)]
    pub unsafe fn rotate_both(&self, values: &SseF32x4) -> SseF32x4 {
        let temp = _mm_shuffle_ps(values.get(), values.get(), 0xB1);
        SseF32x4::new(_mm_xor_ps(temp, self.sign_both.get()))
    }
}

// Pack 1st complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r1.re, r1.im, l1.re + l1.im
#[inline(always)]
pub unsafe fn pack_1st_f32(left: &SseF32x4, right: &SseF32x4) -> SseF32x4 {
    //_mm_shuffle_ps(left, right, 0x44)
    SseF32x4::new(_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(left.get()),_mm_castps_pd(right.get()))))
}

// Pack 2nd complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r2.re, r2.im, l2.re + l2.im
#[inline(always)]
pub unsafe fn pack_2nd_f32(left: &SseF32x4, right: &SseF32x4) -> SseF32x4 {
    //_mm_shuffle_ps(left, right, 0xEE)
    SseF32x4::new(_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(left.get()),_mm_castps_pd(right.get()))))
}

// Pack 1st and 2nd complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r1.re, r1.im, l2.re + l2.im
#[inline(always)]
pub unsafe fn pack_1and2_f32(left: &SseF32x4, right: &SseF32x4) -> SseF32x4 {
    //_mm_shuffle_ps(left, right, 0xE4)
    SseF32x4::new(_mm_blend_ps(left.get(), right.get(), 0x0C))
}

// Pack 2nd and 1st complex
// left: r1.re, r1.im, r2.re, r2.im
// right: l1.re, l1.im, l2.re, l2.im
// --> r2.re, r2.im, l1.re + l1.im
#[inline(always)]
pub unsafe fn pack_2and1_f32(left: &SseF32x4, right: &SseF32x4) -> SseF32x4 {
    SseF32x4::new(_mm_shuffle_ps(left.get(), right.get(), 0x4E))
}

// Reverse complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, a.re + a.im
#[inline(always)]
pub unsafe fn reverse_complex_elements_f32(values: &SseF32x4) -> SseF32x4 {
    SseF32x4::new(_mm_shuffle_ps(values.get(), values.get(), 0x4E))
}

// Invert sign of 2nd complex
// values: a.re, a.im, b.re, b.im
// -->  a.re, a.im, -b.re, -b.im
#[inline(always)]
pub unsafe fn negate_2nd_f32(values: &SseF32x4) -> SseF32x4 {
    SseF32x4::new(_mm_xor_ps(values.get(), _mm_set_ps(-0.0, -0.0, 0.0, 0.0)))
}

// Duplicate 1st complex
// values: a.re, a.im, b.re, b.im
// --> a.re, a.im, a.re + a.im
#[inline(always)]
pub unsafe fn duplicate_1st_f32(values: &SseF32x4) -> SseF32x4 {
    SseF32x4::new(_mm_shuffle_ps(values.get(), values.get(), 0x44))
}

// Duplicate 2nd complex
// values: a.re, a.im, b.re, b.im
// --> b.re, b.im, b.re + b.im
#[inline(always)]
pub unsafe fn duplicate_2nd_f32(values: &SseF32x4) -> SseF32x4 {
    SseF32x4::new(_mm_shuffle_ps(values.get(), values.get(), 0xEE))
}

// Complex multiplication.
// Each input contains two complex values, which are multiplied in parallel.
#[inline(always)]
pub unsafe fn mul_complex_f32(left: &SseF32x4, right: &SseF32x4) -> SseF32x4 {
    //SSE3, taken from Intel performance manual
    let mut temp1 = _mm_shuffle_ps(right.get(), right.get(), 0xA0);
    let mut temp2 = _mm_shuffle_ps(right.get(), right.get(), 0xF5);
    temp1 = _mm_mul_ps(temp1, left.get());
    temp2 = _mm_mul_ps(temp2, left.get());
    temp2 = _mm_shuffle_ps(temp2, temp2, 0xB1);
    SseF32x4::new(_mm_addsub_ps(temp1, temp2))
}

//  __  __       _   _                __   _  _   _     _ _
// |  \/  | __ _| |_| |__            / /_ | || | | |__ (_) |_
// | |\/| |/ _` | __| '_ \   _____  | '_ \| || |_| '_ \| | __|
// | |  | | (_| | |_| | | | |_____| | (_) |__   _| |_) | | |_
// |_|  |_|\__,_|\__|_| |_|          \___/   |_| |_.__/|_|\__|
//

pub(crate) struct Rotate90F64 {
    sign: __m128d,
}

impl Rotate90F64 {
    pub fn new(positive: bool) -> Self {
        let sign = unsafe {
            if positive {
                _mm_set_pd(0.0, -0.0)
            } else {
                _mm_set_pd(-0.0, 0.0)
            }
        };
        Self { sign }
    }

    #[inline(always)]
    pub unsafe fn rotate(&self, values: __m128d) -> __m128d {
        let temp = _mm_shuffle_pd(values, values, 0x01);
        _mm_xor_pd(temp, self.sign)
    }
}

#[inline(always)]
pub unsafe fn mul_complex_f64(left: __m128d, right: __m128d) -> __m128d {
    // SSE3, taken from Intel performance manual
    let mut temp1 = _mm_unpacklo_pd(right, right);
    let mut temp2 = _mm_unpackhi_pd(right, right);
    temp1 = _mm_mul_pd(temp1, left);
    temp2 = _mm_mul_pd(temp2, left);
    temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
    _mm_addsub_pd(temp1, temp2)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use num_complex::Complex;

    #[inline(always)]
    unsafe fn mul_complex_f64(left: __m128d, right: __m128d) -> __m128d {
        let mul1 = _mm_mul_pd(left, right);
        let right_flipped = _mm_shuffle_pd(right, right, 0x01);
        let mul2 = _mm_mul_pd(left, right_flipped);
        let sign = _mm_set_pd(-0.0, 0.0);
        let mul1 = _mm_xor_pd(mul1, sign);
        let temp1 = _mm_shuffle_pd(mul1, mul2, 0x00);
        let temp2 = _mm_shuffle_pd(mul1, mul2, 0x03);
        _mm_add_pd(temp1, temp2)
    }

    #[test]
    fn test_mul_complex_f64() {
        unsafe {
            let right = _mm_set_pd(1.0, 2.0);
            let left = _mm_set_pd(5.0, 7.0);
            let res = mul_complex_f64(left, right);
            let expected = _mm_set_pd(2.0 * 5.0 + 1.0 * 7.0, 2.0 * 7.0 - 1.0 * 5.0);
            assert_eq!(
                std::mem::transmute::<__m128d, Complex<f64>>(res),
                std::mem::transmute::<__m128d, Complex<f64>>(expected)
            );
        }
    }

    #[test]
    fn test_mul_complex_f32() {
        unsafe {
            let val1 = Complex::<f32>::new(1.0, 2.5);
            let val2 = Complex::<f32>::new(3.2, 4.2);
            let val3 = Complex::<f32>::new(5.6, 6.2);
            let val4 = Complex::<f32>::new(7.4, 8.3);

            let nbr2 = _mm_set_ps(val4.im, val4.re, val3.im, val3.re);
            let nbr1 = _mm_set_ps(val2.im, val2.re, val1.im, val1.re);
            let res = mul_complex_f32(nbr1, nbr2);
            let res = std::mem::transmute::<SseF32x4, [Complex<f32>; 2]>(res);
            let expected = [val1 * val3, val2 * val4];
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_pack() {
        unsafe {
            let nbr2 = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
            let nbr1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
            let first = pack_1st_f32(nbr1, nbr2);
            let second = pack_2nd_f32(nbr1, nbr2);
            let first = std::mem::transmute::<SseF32x4, [Complex<f32>; 2]>(first);
            let second = std::mem::transmute::<SseF32x4, [Complex<f32>; 2]>(second);
            let first_expected = [Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)];
            let second_expected = [Complex::new(3.0, 4.0), Complex::new(7.0, 8.0)];
            assert_eq!(first, first_expected);
            assert_eq!(second, second_expected);
        }
    }
}
