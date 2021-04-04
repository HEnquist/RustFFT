use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;

use crate::algorithm::butterflies::{Butterfly1, Butterfly16, Butterfly2, Butterfly4, Butterfly8};
use crate::array_utils;
use crate::common::{fft_error_inplace, fft_error_outofplace};
use crate::{
    array_utils::{RawSlice, RawSliceMut},
    common::FftNum,
    twiddles, FftDirection,
};
use crate::{Direction, Fft, Length};

const SPLIT_AT_LEN: usize = 32768;

/// FFT algorithm optimized for power-of-two sizes
///
/// ~~~
/// // Computes a forward FFT of size 4096
/// use rustfft::algorithm::Radix4;
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 4096];
///
/// let fft = Radix4::new(4096, FftDirection::Forward);
/// fft.process(&mut buffer);
/// ~~~

pub struct Radix4<T> {
    twiddles: Box<[Complex<T>]>,
    shuffle_map: Box<[(usize, usize)]>,
    base_fft: Arc<dyn Fft<T>>,
    base_len: usize,
    len: usize,
    direction: FftDirection,
}

impl<T: FftNum> Radix4<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    pub fn new(len: usize, direction: FftDirection) -> Self {
        assert!(
            len.is_power_of_two(),
            "Radix4 algorithm requires a power-of-two input size. Got {}",
            len
        );

        // figure out which base length we're going to use
        let num_bits = len.trailing_zeros();
        let (base_len, base_fft) = match num_bits {
            0 => (len, Arc::new(Butterfly1::new(direction)) as Arc<dyn Fft<T>>),
            1 => (len, Arc::new(Butterfly2::new(direction)) as Arc<dyn Fft<T>>),
            2 => (len, Arc::new(Butterfly4::new(direction)) as Arc<dyn Fft<T>>),
            _ => {
                if num_bits % 2 == 1 {
                    (8, Arc::new(Butterfly8::new(direction)) as Arc<dyn Fft<T>>)
                } else {
                    (16, Arc::new(Butterfly16::new(direction)) as Arc<dyn Fft<T>>)
                }
            }
        };

        // precompute the twiddle factors this algorithm will use.
        // we're doing the same precomputation of twiddle factors as the mixed radix algorithm where width=4 and height=len/4
        // but mixed radix only does one step and then calls itself recusrively, and this algorithm does every layer all the way down
        // so we're going to pack all the "layers" of twiddle factors into a single array, starting with the bottom layer and going up
        let mut twiddle_stride = len / (base_len * 4);
        let mut twiddle_factors = Vec::with_capacity(len * 2);
        while twiddle_stride > 0 {
            let num_rows = len / (twiddle_stride * 4);
            for i in 0..num_rows {
                for k in 1..4 {
                    let twiddle = twiddles::compute_twiddle(i * k * twiddle_stride, len, direction);
                    twiddle_factors.push(twiddle);
                }
            }
            twiddle_stride >>= 2;
        }

        // make a lookup table for the bit reverse shuffling 
        let rest_len = len/base_len;
        let bitpairs = (rest_len.trailing_zeros()/2) as usize;
        let mut shuffle_map = (0..rest_len).map(|val| (val, reverse_bits(val, bitpairs))).collect::<Vec<(usize, usize)>>();

        // if the lookup table spans a too large range, sort it into chunks
        if rest_len > SPLIT_AT_LEN {
            let chunks = rest_len/SPLIT_AT_LEN;
            let range_per_chunk = rest_len / chunks;
            shuffle_map.sort_by(|a, b| (a.1/range_per_chunk).partial_cmp(&(b.1/range_per_chunk)).unwrap());
        }

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),
            shuffle_map: shuffle_map.into_boxed_slice(),
            base_fft,
            base_len,
            len,
            direction,
        }
    }

    fn perform_fft_out_of_place(
        &self,
        signal: &[Complex<T>],
        spectrum: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        // Prepare for radix 4 by copying shuffled input values to the output
        unsafe {
            bitreversed_transpose(self.base_len, signal, spectrum, &self.shuffle_map);
        }

        // Base-level FFTs
        self.base_fft.process_with_scratch(spectrum, &mut []);

        // cross-FFTs
        let mut current_size = self.base_len * 4;
        let mut layer_twiddles: &[Complex<T>] = &self.twiddles;

        while current_size <= signal.len() {
            let num_rows = signal.len() / current_size;

            for i in 0..num_rows {
                unsafe {
                    butterfly_4(
                        &mut spectrum[i * current_size..],
                        layer_twiddles,
                        current_size / 4,
                        self.direction,
                    )
                }
            }

            //skip past all the twiddle factors used in this layer
            let twiddle_offset = (current_size * 3) / 4;
            layer_twiddles = &layer_twiddles[twiddle_offset..];

            current_size *= 4;
        }
    }
}
boilerplate_fft_oop!(Radix4, |this: &Radix4<_>| this.len);

// Reverse bits of value, in pairs.
// For 8 bits: abcdefgh -> ghefcdab
fn reverse_bits(value: usize, bitpairs: usize) -> usize {
    let mut result: usize = 0; 
    let mut value = value;
    for _ in 0..bitpairs {
        result = (result<<2) + (value & 0x03);
        value = value>>2;
    }
    result
}

// Preparing for radix 4 is similar to a transpose, where the column index is bit reversed. 
// Use a lookup table to avoid repeating the slow bit reverse operations.
unsafe fn bitreversed_transpose<T: Copy>(base_len: usize, input: &[T], output: &mut [T], shuffle_map: &[(usize, usize)]) {
    for y in 0..base_len {
        for (x, x_rev) in shuffle_map.iter() {
            let input_index = x_rev + y * shuffle_map.len();
            let output_index = y + x * base_len;
            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

unsafe fn butterfly_4<T: FftNum>(
    data: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    num_ffts: usize,
    direction: FftDirection,
) {
    let butterfly4 = Butterfly4::new(direction);

    let mut idx = 0usize;
    let mut scratch = [Zero::zero(); 4];
    for tw in twiddles.chunks_exact(3).take(num_ffts) {
        scratch[0] = *data.get_unchecked(idx);
        scratch[1] = *data.get_unchecked(idx + 1 * num_ffts) * tw[0];
        scratch[2] = *data.get_unchecked(idx + 2 * num_ffts) * tw[1];
        scratch[3] = *data.get_unchecked(idx + 3 * num_ffts) * tw[2];

        butterfly4.perform_fft_contiguous(RawSlice::new(&scratch), RawSliceMut::new(&mut scratch));

        *data.get_unchecked_mut(idx) = scratch[0];
        *data.get_unchecked_mut(idx + 1 * num_ffts) = scratch[1];
        *data.get_unchecked_mut(idx + 2 * num_ffts) = scratch[2];
        *data.get_unchecked_mut(idx + 3 * num_ffts) = scratch[3];

        idx += 1;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn test_radix4() {
        for pow in 0..8 {
            let len = 1 << pow;
            test_radix4_with_length(len, FftDirection::Forward);
            test_radix4_with_length(len, FftDirection::Inverse);
        }
    }

    fn test_radix4_with_length(len: usize, direction: FftDirection) {
        let fft = Radix4::new(len, direction);

        check_fft_algorithm::<f32>(&fft, len, direction);
    }
}
