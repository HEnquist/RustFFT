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

        Self {
            twiddles: twiddle_factors.into_boxed_slice(),

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
        // copy the data into the spectrum vector, split the copying up into chunks to make it more cache friendly
        let mut num_chunks = signal.len() / (2*8192);
        if num_chunks == 0 {
            num_chunks = 1;
        } else if num_chunks > self.base_len/4 {
            num_chunks = self.base_len/4;
        }
        for n in 0..num_chunks {
            prepare_radix4(
                signal.len(),
                self.base_len,
                signal,
                spectrum,
                1,
                n,
                num_chunks,
            );
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

fn prepare_radix4<T: FftNum>(
    size: usize,
    base_len: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
    chunk: usize,
    nbr_chunks: usize,
) {
    if size == (4 * base_len) {
        do_radix4_shuffle(size, signal, spectrum, stride, chunk, nbr_chunks);
    } else if size == base_len {
        unsafe {
            for i in (chunk * base_len / nbr_chunks)..((chunk + 1) * base_len / nbr_chunks) {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        }
    } else {
        for i in 0..4 {
            prepare_radix4(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
                chunk,
                nbr_chunks,
            );
        }
    }
}

fn do_radix4_shuffle<T: FftNum>(
    size: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
    chunk: usize,
    nbr_chunks: usize,
) {
    let stepsize = size / 4;
    let stepstride = stride * 4;
    let signal_offset = stride;
    let spectrum_offset = size / 4;
    unsafe {
        for i in (chunk * stepsize / nbr_chunks)..((chunk + 1) * stepsize / nbr_chunks) {
            let val0 = *signal.get_unchecked(i * stepstride);
            let val1 = *signal.get_unchecked(i * stepstride + signal_offset);
            let val2 = *signal.get_unchecked(i * stepstride + 2 * signal_offset);
            let val3 = *signal.get_unchecked(i * stepstride + 3 * signal_offset);
            *spectrum.get_unchecked_mut(i) = val0;
            *spectrum.get_unchecked_mut(i + spectrum_offset) = val1;
            *spectrum.get_unchecked_mut(i + 2 * spectrum_offset) = val2;
            *spectrum.get_unchecked_mut(i + 3 * spectrum_offset) = val3;
        }
    }
}

pub fn reverse_bits(value: usize, bits: usize) -> usize {
    let mut result: usize = 0; 
    let mut value = value;
    for _ in 0..bits/2 {
        result = (result<<2) + (value & 0x03);
        value = value>>2;
    }
    result
}

pub unsafe fn bitrev_transpose<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T], bits: usize) {
    for x in 0..width {
        let x_rev = reverse_bits(x, bits);
        for y in 0..height {
            let input_index = x_rev + y * width;
            let output_index = y + x * height;

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
