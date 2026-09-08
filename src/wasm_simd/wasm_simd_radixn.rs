//! A WASM SIMD version of `RadixN`, generic over f32 and f64.
//!
//! This mirrors `src/algorithm/radixn.rs`: one flat transpose down to a base FFT, then a stack of
//! in-place cross-FFT layers over a single packed twiddle array. The only difference is that the
//! cross-FFT layers use WASM SIMD column butterflies instead of the scalar ones, so a whole vector of
//! columns is processed per butterfly call.
//!
//! Because a column butterfly consumes `COMPLEX_PER_VECTOR` columns at a time, the column count at
//! every layer has to be a whole number of vectors. The column count starts at `base_len` and only
//! ever grows by whole factors, so requiring `base_len % COMPLEX_PER_VECTOR == 0` is enough. That
//! is 1 for f64 (no restriction) and 2 for f32.

use std::any::TypeId;
use std::sync::Arc;

use core::arch::wasm32::v128;
use num_complex::Complex;

use crate::array_utils::{factor_transpose, workaround_transmute_mut, TransposeFactor};
use crate::common::{FftNum, RadixFactor};
use crate::{Direction, Fft, FftDirection, Length};

use super::wasm_simd_butterflies::{
    WasmSimdF32Butterfly3, WasmSimdF32Butterfly5, WasmSimdF32Butterfly6, WasmSimdF64Butterfly3,
    WasmSimdF64Butterfly5, WasmSimdF64Butterfly6,
};
use super::wasm_simd_prime_butterflies::{WasmSimdF32Butterfly7, WasmSimdF64Butterfly7};
use super::wasm_simd_vector::{
    Rotation90, WasmSimdArray, WasmSimdArrayMut, WasmVector, WasmVector32, WasmVector64,
};
use super::WasmNum;

/// Butterflies 3, 5 and 6 are written against raw `v128`, while `WasmVector32`/`WasmVector64`
/// are newtypes over it, so results come back needing rewrapping. Butterfly 7 already speaks the
/// wrapper types and needs none of this.
#[inline(always)]
fn wrap64<const N: usize>(values: [v128; N]) -> [WasmVector64; N] {
    values.map(WasmVector64)
}

#[inline(always)]
fn wrap32<const N: usize>(values: [v128; N]) -> [WasmVector32; N] {
    values.map(WasmVector32)
}

/// The column butterflies `WasmSimdRadixN` needs that aren't already on `WasmVector`.
///
/// Radix 2 and 4 are already vector-generic as `WasmVector::column_butterfly2` and
/// `column_butterfly4`. Radix 3, 5, 6 and 7 only exist as element-type-specific structs holding
/// precomputed twiddles, so this trait pairs each vector type with its own set of them. For f64 a
/// vector is one complex number and the plain `perform_fft_direct` is already a single column; for
/// f32 a vector is two complex numbers and `perform_parallel_fft_direct` does two columns at once.
pub trait RadixNButterflies: WasmVector + Sized {
    type Butterfly3: Send + Sync;
    type Butterfly5: Send + Sync;
    type Butterfly6: Send + Sync;
    type Butterfly7: Send + Sync;

    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3;
    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5;
    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6;
    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7;

    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3];
    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5];
    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6];
    /// Safety: The current machine must support the simd128 instruction set
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7];
}

impl RadixNButterflies for WasmVector64 {
    type Butterfly3 = WasmSimdF64Butterfly3<f64>;
    type Butterfly5 = WasmSimdF64Butterfly5<f64>;
    type Butterfly6 = WasmSimdF64Butterfly6<f64>;
    type Butterfly7 = WasmSimdF64Butterfly7<f64>;

    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        WasmSimdF64Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        WasmSimdF64Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        WasmSimdF64Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        WasmSimdF64Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        wrap64(bf.perform_fft_direct(rows[0].0, rows[1].0, rows[2].0))
    }
    #[inline(always)]
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        wrap64(bf.perform_fft_direct(rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0))
    }
    #[inline(always)]
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        wrap64(bf.perform_fft_direct([
            rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0, rows[5].0,
        ]))
    }
    #[inline(always)]
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        bf.perform_fft_direct(rows)
    }
}

impl RadixNButterflies for WasmVector32 {
    type Butterfly3 = WasmSimdF32Butterfly3<f32>;
    type Butterfly5 = WasmSimdF32Butterfly5<f32>;
    type Butterfly6 = WasmSimdF32Butterfly6<f32>;
    type Butterfly7 = WasmSimdF32Butterfly7<f32>;

    #[inline(always)]
    unsafe fn make_butterfly3(direction: FftDirection) -> Self::Butterfly3 {
        WasmSimdF32Butterfly3::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly5(direction: FftDirection) -> Self::Butterfly5 {
        WasmSimdF32Butterfly5::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly6(direction: FftDirection) -> Self::Butterfly6 {
        WasmSimdF32Butterfly6::new(direction)
    }
    #[inline(always)]
    unsafe fn make_butterfly7(direction: FftDirection) -> Self::Butterfly7 {
        WasmSimdF32Butterfly7::new(direction)
    }

    #[inline(always)]
    unsafe fn column_butterfly3(bf: &Self::Butterfly3, rows: [Self; 3]) -> [Self; 3] {
        wrap32(bf.perform_parallel_fft_direct(rows[0].0, rows[1].0, rows[2].0))
    }
    #[inline(always)]
    unsafe fn column_butterfly5(bf: &Self::Butterfly5, rows: [Self; 5]) -> [Self; 5] {
        wrap32(
            bf.perform_parallel_fft_direct(rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0),
        )
    }
    #[inline(always)]
    unsafe fn column_butterfly6(bf: &Self::Butterfly6, rows: [Self; 6]) -> [Self; 6] {
        wrap32(bf.perform_parallel_fft_direct(
            rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0, rows[5].0,
        ))
    }
    #[inline(always)]
    unsafe fn column_butterfly7(bf: &Self::Butterfly7, rows: [Self; 7]) -> [Self; 7] {
        bf.perform_parallel_fft_direct(rows)
    }
}

/// The per-layer cross-FFT kernels, holding whatever precomputed state each radix needs.
enum Layer<N: WasmNum> {
    Factor2,
    Factor3(<N::VectorType as RadixNButterflies>::Butterfly3),
    Factor4(Rotation90<N::VectorType>),
    Factor5(<N::VectorType as RadixNButterflies>::Butterfly5),
    Factor6(<N::VectorType as RadixNButterflies>::Butterfly6),
    Factor7(<N::VectorType as RadixNButterflies>::Butterfly7),
}

impl<N: WasmNum> Layer<N> {
    fn radix(&self) -> usize {
        match self {
            Layer::Factor2 => 2,
            Layer::Factor3(_) => 3,
            Layer::Factor4(_) => 4,
            Layer::Factor5(_) => 5,
            Layer::Factor6(_) => 6,
            Layer::Factor7(_) => 7,
        }
    }
}

/// FFT algorithm for lengths that factor into small radixes, NEON accelerated version.
/// This is designed to be used via a Planner, and not created directly.
pub struct WasmSimdRadixN<N: WasmNum, T> {
    twiddles: Box<[N::VectorType]>,

    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,

    factors: Box<[TransposeFactor]>,
    layers: Box<[Layer<N>]>,

    len: usize,
    direction: FftDirection,

    inplace_scratch_len: usize,
    outofplace_scratch_len: usize,
    immut_scratch_len: usize,
}

impl<N: WasmNum, T: FftNum> WasmSimdRadixN<N, T> {
    /// Constructs a WasmSimdRadixN which computes FFTs of length `factor_product * base_fft.len()`.
    pub fn new(factors: &[RadixFactor], base_fft: Arc<dyn Fft<T>>) -> Self {
        // Internal sanity check: Make sure that N == T.
        // This struct has two generic parameters N and T, but they must always be the same, and are
        // only kept separate to help work around the lack of specialization.
        assert_eq!(TypeId::of::<N>(), TypeId::of::<T>());

        let base_len = base_fft.len();
        let direction = base_fft.fft_direction();
        let complex_per_vector = <N::VectorType as WasmVector>::COMPLEX_PER_VECTOR;

        // Every cross-FFT layer processes a whole vector of columns at a time. The column count
        // starts at base_len and is only ever multiplied by a factor, so this one check covers
        // every layer.
        assert!(
            factors.is_empty() || base_len % complex_per_vector == 0,
            "WasmSimdRadixN requires a base length divisible by {}, got {}",
            complex_per_vector,
            base_len
        );

        // set up our cross FFT butterfly instances. simultaneously, compute the number of twiddles
        let mut layers = Vec::with_capacity(factors.len());
        let mut cross_fft_len = base_len;
        let mut twiddle_count = 0;

        for factor in factors {
            // twiddles are stored a vector at a time, so a layer needs one chunk per vector column
            twiddle_count += (cross_fft_len / complex_per_vector) * (factor.radix() - 1);

            layers.push(unsafe {
                match factor {
                    RadixFactor::Factor2 => Layer::Factor2,
                    RadixFactor::Factor3 => {
                        Layer::Factor3(N::VectorType::make_butterfly3(direction))
                    }
                    RadixFactor::Factor4 => Layer::Factor4(WasmVector::make_rotate90(direction)),
                    RadixFactor::Factor5 => {
                        Layer::Factor5(N::VectorType::make_butterfly5(direction))
                    }
                    RadixFactor::Factor6 => {
                        Layer::Factor6(N::VectorType::make_butterfly6(direction))
                    }
                    RadixFactor::Factor7 => {
                        Layer::Factor7(N::VectorType::make_butterfly7(direction))
                    }
                }
            });

            cross_fft_len *= factor.radix();
        }
        let len = cross_fft_len;

        // set up our list of transpose factors - it's the same list but reversed, and we want to
        // collapse duplicates. Note that we are only de-duplicating adjacent factors: if we're
        // passed 7 * 2 * 7, we can't collapse the sevens because the exact order matters.
        let mut transpose_factors: Vec<TransposeFactor> = Vec::with_capacity(factors.len());
        for f in factors.iter().rev() {
            let mut push_new = true;
            if let Some(last) = transpose_factors.last_mut() {
                if last.factor == *f {
                    last.count += 1;
                    push_new = false;
                }
            }
            if push_new {
                transpose_factors.push(TransposeFactor {
                    factor: *f,
                    count: 1,
                });
            }
        }

        // Same packing as the scalar RadixN: all layers in one array, bottom layer first, and
        // within a layer, (radix - 1) twiddles per column. The difference is that a "column" here
        // is a whole vector of columns, so each entry is a twiddle chunk rather than one twiddle.
        let mut twiddle_factors: Vec<N::VectorType> = Vec::with_capacity(twiddle_count);
        let mut cross_fft_len = base_len;
        for factor in factors {
            let num_vector_columns = cross_fft_len / complex_per_vector;
            cross_fft_len *= factor.radix();

            for i in 0..num_vector_columns {
                for k in 1..factor.radix() {
                    unsafe {
                        twiddle_factors.push(WasmVector::make_mixedradix_twiddle_chunk(
                            i * complex_per_vector,
                            k,
                            cross_fft_len,
                            direction,
                        ));
                    }
                }
            }
        }

        // figure out how much scratch space we need to request from callers
        let base_inplace_scratch = base_fft.get_inplace_scratch_len();
        let inplace_scratch_len = if base_inplace_scratch > len {
            len + base_inplace_scratch
        } else {
            len
        };
        let outofplace_scratch_len = if base_inplace_scratch > len {
            base_inplace_scratch
        } else {
            0
        };

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

            base_fft,
            base_len,

            factors: transpose_factors.into_boxed_slice(),
            layers: layers.into_boxed_slice(),

            len,
            direction,

            inplace_scratch_len,
            outofplace_scratch_len,
            immut_scratch_len: base_inplace_scratch,
        }
    }

    /// The flat transpose that reorders the input down to base-sized chunks.
    #[inline(always)]
    fn transpose(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if let Some(unroll_factor) = self.factors.first() {
            // for performance, we really, really want to unroll the transpose, but we need to make
            // sure the output length is divisible by the unroll amount. choosing the first factor
            // seems to reliably perform well
            match unroll_factor.factor {
                RadixFactor::Factor2 => {
                    factor_transpose::<Complex<T>, 2>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor3 => {
                    factor_transpose::<Complex<T>, 3>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor4 => {
                    factor_transpose::<Complex<T>, 4>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor5 => {
                    factor_transpose::<Complex<T>, 5>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor6 => {
                    factor_transpose::<Complex<T>, 6>(self.base_len, input, output, &self.factors)
                }
                RadixFactor::Factor7 => {
                    factor_transpose::<Complex<T>, 7>(self.base_len, input, output, &self.factors)
                }
            }
        } else {
            // no factors, so just pass data straight to our base
            output.copy_from_slice(input);
        }
    }

    /// The stack of in-place cross-FFT layers, run after the base FFTs.
    unsafe fn cross_ffts(&self, output: &mut [Complex<T>]) {
        let out: &mut [Complex<N>] = workaround_transmute_mut(output);

        let mut cross_fft_len = self.base_len;
        let mut layer_twiddles: &[N::VectorType] = &self.twiddles;

        for layer in self.layers.iter() {
            let num_columns = cross_fft_len;
            cross_fft_len *= layer.radix();

            for data in out.chunks_exact_mut(cross_fft_len) {
                match layer {
                    Layer::Factor2 => {
                        cross_layer::<N, 2, _>(data, layer_twiddles, num_columns, |v| {
                            WasmVector::column_butterfly2(v)
                        })
                    }
                    Layer::Factor3(bf) => {
                        cross_layer::<N, 3, _>(data, layer_twiddles, num_columns, |v| {
                            N::VectorType::column_butterfly3(bf, v)
                        })
                    }
                    Layer::Factor4(rotation) => {
                        cross_layer::<N, 4, _>(data, layer_twiddles, num_columns, |v| {
                            WasmVector::column_butterfly4(v, *rotation)
                        })
                    }
                    Layer::Factor5(bf) => {
                        cross_layer::<N, 5, _>(data, layer_twiddles, num_columns, |v| {
                            N::VectorType::column_butterfly5(bf, v)
                        })
                    }
                    Layer::Factor6(bf) => {
                        cross_layer::<N, 6, _>(data, layer_twiddles, num_columns, |v| {
                            N::VectorType::column_butterfly6(bf, v)
                        })
                    }
                    Layer::Factor7(bf) => {
                        cross_layer::<N, 7, _>(data, layer_twiddles, num_columns, |v| {
                            N::VectorType::column_butterfly7(bf, v)
                        })
                    }
                }
            }

            // skip past all the twiddle factors used in this layer
            let complex_per_vector = <N::VectorType as WasmVector>::COMPLEX_PER_VECTOR;
            let twiddle_offset = (num_columns / complex_per_vector) * (layer.radix() - 1);
            layer_twiddles = &layer_twiddles[twiddle_offset..];
        }
    }

    unsafe fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);
        self.base_fft.process_with_scratch(output, scratch);
        self.cross_ffts(output);
    }

    unsafe fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);

        // the input is free once the transpose is done, so use it as base scratch when we weren't
        // handed any of our own
        let base_scratch = if scratch.len() > 0 { scratch } else { input };
        self.base_fft.process_with_scratch(output, base_scratch);

        self.cross_ffts(output);
    }
}
boilerplate_fft_wasm_simd_oop_scratch!(WasmSimdRadixN, |this: &WasmSimdRadixN<_, _>| this.len);

/// One cross-FFT layer: for each vector of columns, gather RADIX rows strided by `num_columns`,
/// apply the twiddles, run the column butterfly, scatter back.
///
/// Unrolled two vectors at a time, which is what WasmSimdRadix4's `butterfly_4` does and is what gets
/// the two independent dependency chains needed to keep the FMA pipeline busy.
#[inline(always)]
unsafe fn cross_layer<N: WasmNum, const RADIX: usize, F>(
    mut data: &mut [Complex<N>],
    twiddles: &[N::VectorType],
    num_columns: usize,
    butterfly: F,
) where
    F: Fn([N::VectorType; RADIX]) -> [N::VectorType; RADIX],
{
    let complex_per_vector = <N::VectorType as WasmVector>::COMPLEX_PER_VECTOR;
    let num_vector_columns = num_columns / complex_per_vector;
    let tw_stride = RADIX - 1;

    debug_assert!(twiddles.len() >= num_vector_columns * tw_stride);

    // The row-0 twiddle is always 1, so it's neither stored nor applied.
    let gather = |data: &[Complex<N>], idx: usize, tw_base: usize| -> [N::VectorType; RADIX] {
        std::array::from_fn(|r| {
            let v = data.load_complex(idx + r * num_columns);
            if r == 0 {
                v
            } else {
                WasmVector::mul_complex(v, *twiddles.get_unchecked(tw_base + r - 1))
            }
        })
    };

    let mut vcol = 0;
    while vcol + 2 <= num_vector_columns {
        let idx = vcol * complex_per_vector;

        let a = gather(data, idx, vcol * tw_stride);
        let b = gather(data, idx + complex_per_vector, (vcol + 1) * tw_stride);

        let a = butterfly(a);
        let b = butterfly(b);

        for r in 0..RADIX {
            data.store_complex(a[r], idx + r * num_columns);
            data.store_complex(b[r], idx + complex_per_vector + r * num_columns);
        }

        vcol += 2;
    }

    // an odd vector column count leaves one behind
    if vcol < num_vector_columns {
        let idx = vcol * complex_per_vector;
        let a = butterfly(gather(data, idx, vcol * tw_stride));
        for r in 0..RADIX {
            data.store_complex(a[r], idx + r * num_columns);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::{check_fft_algorithm, construct_base};
    use num_traits::Float;
    use rand::distributions::uniform::SampleUniform;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_f64() {
        let factor_list = &[
            RadixFactor::Factor2,
            RadixFactor::Factor3,
            RadixFactor::Factor4,
            RadixFactor::Factor5,
            RadixFactor::Factor6,
            RadixFactor::Factor7,
        ];

        // f64 fits one complex per vector, so every base length is legal
        for base in 1..7 {
            let base_forward = construct_base(base, FftDirection::Forward);
            let base_inverse = construct_base(base, FftDirection::Inverse);

            test_radixn::<f64>(&[], Arc::clone(&base_forward));
            test_radixn::<f64>(&[], Arc::clone(&base_inverse));

            for factor_a in factor_list {
                test_radixn::<f64>(&[*factor_a], Arc::clone(&base_forward));
                test_radixn::<f64>(&[*factor_a], Arc::clone(&base_inverse));

                for factor_b in factor_list {
                    let factors = &[*factor_a, *factor_b];
                    test_radixn::<f64>(factors, Arc::clone(&base_forward));
                    test_radixn::<f64>(factors, Arc::clone(&base_inverse));
                }
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_f32() {
        let factor_list = &[
            RadixFactor::Factor2,
            RadixFactor::Factor3,
            RadixFactor::Factor4,
            RadixFactor::Factor5,
            RadixFactor::Factor6,
            RadixFactor::Factor7,
        ];

        // f32 fits two complex per vector, so the base length has to be even
        for base in [2, 4, 6] {
            let base_forward = construct_base(base, FftDirection::Forward);
            let base_inverse = construct_base(base, FftDirection::Inverse);

            test_radixn::<f32>(&[], Arc::clone(&base_forward));
            test_radixn::<f32>(&[], Arc::clone(&base_inverse));

            for factor_a in factor_list {
                test_radixn::<f32>(&[*factor_a], Arc::clone(&base_forward));
                test_radixn::<f32>(&[*factor_a], Arc::clone(&base_inverse));

                for factor_b in factor_list {
                    let factors = &[*factor_a, *factor_b];
                    test_radixn::<f32>(factors, Arc::clone(&base_forward));
                    test_radixn::<f32>(factors, Arc::clone(&base_inverse));
                }
            }
        }
    }

    /// The base doesn't have to be a scratch-free butterfly. A composite base is a recursive
    /// recipe that needs its own scratch, which is the case `design_radixn` hits whenever a length
    /// has factors above 7 (for example 11 * 13 = 143).
    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_composite_base() {
        let mut planner64 = crate::FftPlannerScalar::<f64>::new();
        let mut planner32 = crate::FftPlannerScalar::<f32>::new();

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            // odd base, f64 only
            for base_len in [143, 55, 65] {
                let base = planner64.plan_fft(base_len, direction);
                assert!(
                    base.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                test_radixn::<f64>(&[RadixFactor::Factor6, RadixFactor::Factor4], base);
            }

            // even base, usable by both element types
            for base_len in [22, 26, 110] {
                let base = planner32.plan_fft(base_len, direction);
                assert!(
                    base.get_inplace_scratch_len() > 0,
                    "base {} was expected to need scratch",
                    base_len
                );
                test_radixn::<f32>(&[RadixFactor::Factor3, RadixFactor::Factor4], base);

                let base = planner64.plan_fft(base_len, direction);
                test_radixn::<f64>(&[RadixFactor::Factor3, RadixFactor::Factor4], base);
            }
        }
    }

    /// The recipes the spike was benchmarked on, so the sizes that actually matter stay covered.
    #[wasm_bindgen_test]
    fn test_wasm_simd_radixn_large_recipes() {
        use RadixFactor::*;
        // (factors, f64 base, f32 base). f32 needs an even base, so it gets its own.
        let cases: [(&[RadixFactor], usize, usize); 6] = [
            (&[Factor6, Factor6, Factor6], 5, 6),            // 1080
            (&[Factor6, Factor5, Factor5], 7, 8),            // 1050
            (&[Factor6, Factor6, Factor4], 7, 8),            // 1008
            (&[Factor6, Factor6, Factor3], 12, 12),          // 1296
            (&[Factor6, Factor6, Factor6, Factor4], 12, 12), // 10368
            (
                &[Factor6, Factor6, Factor5, Factor5, Factor4, Factor4],
                7,
                8,
            ), // 100800
        ];
        for (factors, base64, base32) in cases {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                test_radixn::<f64>(factors, construct_base(base64, direction));
                test_radixn::<f32>(factors, construct_base(base32, direction));
            }
        }
    }

    fn test_radixn<T: WasmNum + Float + SampleUniform>(
        factors: &[RadixFactor],
        base_fft: Arc<dyn Fft<T>>,
    ) {
        let len = base_fft.len() * factors.iter().map(|f| f.radix()).product::<usize>();
        let direction = base_fft.fft_direction();
        let fft: WasmSimdRadixN<T, T> = WasmSimdRadixN::new(factors, base_fft);

        check_fft_algorithm::<T>(&fft, len, direction);
    }
}
