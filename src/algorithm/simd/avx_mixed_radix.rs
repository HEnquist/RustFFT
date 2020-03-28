use std::sync::Arc;
use std::arch::x86_64::*;
use std::cmp::min;

use num_complex::Complex;
use num_traits::Zero;

use common::{FFTnum, verify_length_inline, verify_length_minimum};

use ::{Length, IsInverse, FftInline};

use super::avx_utils::AvxComplexArrayf32;
use super::avx_utils;

pub struct MixedRadix2xnAvx<T> {
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    len: usize,
    inverse: bool,
}
impl MixedRadix2xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let half_len = inner_fft.len();
        let len = half_len * 2;

        assert_eq!(len % 2, 0, "MixedRadix2xnAvx requires its FFT length to be an even number. Got {}", len);

        let quotient = half_len / 4;
        let remainder = half_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let twiddles: Vec<_> = (0..num_twiddle_columns).map(|x| {
            let chunk_size = if x == quotient { remainder } else { 4 };
            let mut twiddle_chunk = [Complex::zero(); 4];
            for i in 0..chunk_size {
                twiddle_chunk[i] = f32::generate_twiddle_factor(x*4 + i, len, inverse);
            }
            twiddle_chunk.load_complex_f32(0)
        }).collect();

        Self {
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let half_len = len / 2;
        
        let chunk_count = half_len / 4;
        let remainder = half_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0 = buffer.load_complex_f32(i*4); 
        	let input1 = buffer.load_complex_f32(i*4 + half_len);

        	let (output0, output1_pretwiddle) = avx_utils::column_butterfly2_f32(input0, input1);
            buffer.store_complex_f32(i*4, output0);
            
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i), output1_pretwiddle);
        	buffer.store_complex_f32(i*4 + half_len, output1);
        }

        // process the remainder
        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

            let input0 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + half_len);

            let (output0, output1_pretwiddle) = avx_utils::column_butterfly2_f32(input0, input1);
            buffer.store_complex_remainder_f32(remainder_mask, output0, chunk_count*4);

            let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count), output1_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output1, chunk_count*4 + half_len);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(half_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..half_len];
        scratch.copy_from_slice(&buffer[..half_len]);

        for i in 0..chunk_count {
            let input0 = scratch.load_complex_f32(i*4); 
            let input1 = buffer.load_complex_f32(i*4 + half_len);

            // We loaded data from 2 separate arrays. inteleave the two arrays
            let (transposed0, transposed1) = avx_utils::interleave_evens_odds_f32(input0, input1);

            // store the interleaved array contiguously
            buffer.store_complex_f32(i*8, transposed0);
            buffer.store_complex_f32(i*8 + 4, transposed1);
        }

        // transpose the remainder
        if remainder > 0 {
            let load_remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

            let input0 = scratch.load_complex_remainder_f32(load_remainder_mask, chunk_count*4); 
            let input1 = buffer.load_complex_remainder_f32(load_remainder_mask, chunk_count*4 + half_len);

            // We loaded data from 2 separate arrays. inteleave the two arrays
            let (transposed0, transposed1) = avx_utils::interleave_evens_odds_f32(input0, input1);

            // store the interleaved array contiguously
            let store_remainder_mask_0 = avx_utils::RemainderMask::new_f32(min(remainder * 2, 4));
            buffer.store_complex_remainder_f32(store_remainder_mask_0, transposed0, chunk_count*8);
            if remainder > 2 {
                let store_remainder_mask_1 = avx_utils::RemainderMask::new_f32(remainder * 2 - 4);
                buffer.store_complex_remainder_f32(store_remainder_mask_1, transposed1, chunk_count*8 + 4);
            }
        }
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix2xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix2xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix2xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for MixedRadix2xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadix4xnAvx<T> {
    twiddle_config: avx_utils::Rotate90Config,
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    len: usize,
    inverse: bool,
}
impl MixedRadix4xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let quarter_len = inner_fft.len();
        let len = quarter_len * 4;

        assert_eq!(len % 4, 0, "MixedRadix4xnAvx requires its FFT length to be a multiple of 4. Got {}", len);

        let quotient = quarter_len / 4;
        let remainder = quarter_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 3);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 4 };

        	for y in 1..4 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*4 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        Self {
            twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let quarter_len = len / 4;

        let chunk_count = quarter_len / 4;
        let remainder = quarter_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0 = buffer.load_complex_f32(i*4); 
        	let input1 = buffer.load_complex_f32(i*4 + quarter_len);
        	let input2 = buffer.load_complex_f32(i*4 + quarter_len*2);
        	let input3 = buffer.load_complex_f32(i*4 + quarter_len*3);

        	let (output0, output1_pretwiddle, output2_pretwiddle, output3_pretwiddle) = avx_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

            buffer.store_complex_f32(i*4, output0);
            let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*3), output1_pretwiddle);
            buffer.store_complex_f32(i*4 + quarter_len, output1);
            let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*3+1), output2_pretwiddle);
            buffer.store_complex_f32(i*4 + quarter_len*2, output2);
            let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*3+2), output3_pretwiddle);
        	buffer.store_complex_f32(i*4 + quarter_len*3, output3);
        }

        // process the remainder
        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

            let input0 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len);
            let input2 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*2);
            let input3 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*3);

            let (output0, output1_pretwiddle, output2_pretwiddle, output3_pretwiddle) = avx_utils::column_butterfly4_f32(input0, input1, input2, input3, self.twiddle_config);

            buffer.store_complex_remainder_f32(remainder_mask, output0, chunk_count*4);
            let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*3), output1_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output1, chunk_count*4 + quarter_len);
            let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*3+1), output2_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output2, chunk_count*4 + quarter_len*2);
            let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*3+2), output3_pretwiddle);
            buffer.store_complex_remainder_f32(remainder_mask, output3, chunk_count*4 + quarter_len*3);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(quarter_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..(len - quarter_len)];
        scratch.copy_from_slice(&buffer[..(len-quarter_len)]);

        for i in 0..chunk_count {
            let input0 = scratch.load_complex_f32(i*4); 
            let input1 = scratch.load_complex_f32(i*4 + quarter_len);
            let input2 = scratch.load_complex_f32(i*4 + quarter_len*2);
            let input3 = buffer.load_complex_f32(i*4 + quarter_len*3);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(input0, input1, input2, input3);

            // store the first chunk directly back into 
            buffer.store_complex_f32(i*16, transposed0);
            buffer.store_complex_f32(i*16 + 4, transposed1);
            buffer.store_complex_f32(i*16 + 4*2, transposed2);
            buffer.store_complex_f32(i*16 + 4*3, transposed3);
        }

        // transpose the remainder
        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

            let input0 = scratch.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = scratch.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len);
            let input2 = scratch.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*2);
            let input3 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + quarter_len*3);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, _transposed3) = avx_utils::transpose_4x4_f32(input0, input1, input2, input3);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            buffer.store_complex_f32(chunk_count*16, transposed0);
            if remainder >= 2 {
                buffer.store_complex_f32(chunk_count*16 + 4, transposed1);
                if remainder >= 3 {
                    buffer.store_complex_f32(chunk_count*16 + 4*2, transposed2);
                    // the remainder will never be 4 - because if it was 4, it would have been handled in the loop above
                    // so we can just not use the final 2 values. thankfully, the optimizer is smart enough to never even generate the last 2 values
                }
            }
        }
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix4xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix4xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix4xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for MixedRadix4xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadix8xnAvx<T> {
    twiddle_config: avx_utils::Rotate90Config,
    twiddles_butterfly8: __m256,
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    len: usize,
    inverse: bool,
}
impl MixedRadix8xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let eigth_len = inner_fft.len();
        let len = eigth_len * 8;

        assert_eq!(len % 8, 0, "MixedRadix8xnAvx requires its FFT length to be a multiple of 8. Got {}", len);

        let quotient = eigth_len / 4;
        let remainder = eigth_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 7);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 4 };

        	for y in 1..8 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*4 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        Self {
        	twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
        	twiddles_butterfly8: avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 8, inverse)),
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let eigth_len = len / 8;

        let chunk_count = eigth_len / 4;
        let remainder = eigth_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0 = buffer.load_complex_f32(i*4); 
        	let input1 = buffer.load_complex_f32(i*4 + eigth_len);
        	let input2 = buffer.load_complex_f32(i*4 + eigth_len*2);
        	let input3 = buffer.load_complex_f32(i*4 + eigth_len*3);
        	let input4 = buffer.load_complex_f32(i*4 + eigth_len*4);
        	let input5 = buffer.load_complex_f32(i*4 + eigth_len*5);
        	let input6 = buffer.load_complex_f32(i*4 + eigth_len*6);
        	let input7 = buffer.load_complex_f32(i*4 + eigth_len*7);

        	let (output0, mid1, mid2, mid3, mid4, mid5, mid6, mid7) = avx_utils::column_butterfly8_fma_f32(input0, input1, input2, input3, input4, input5, input6, input7, self.twiddles_butterfly8, self.twiddle_config);

        	buffer.store_complex_f32(i*4, output0);
        	debug_assert!(self.twiddles.len() >= (i+1) * 7);
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7), mid1);
        	buffer.store_complex_f32(i*4 + eigth_len, output1);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+1), mid2);
        	buffer.store_complex_f32(i*4 + eigth_len*2, output2);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+2), mid3);
        	buffer.store_complex_f32(i*4 + eigth_len*3, output3);
        	let output4 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+3), mid4);
        	buffer.store_complex_f32(i*4 + eigth_len*4, output4);
        	let output5 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+4), mid5);
        	buffer.store_complex_f32(i*4 + eigth_len*5, output5);
        	let output6 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+5), mid6);
        	buffer.store_complex_f32(i*4 + eigth_len*6, output6);
        	let output7 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*7+6), mid7);
        	buffer.store_complex_f32(i*4 + eigth_len*7, output7);
        }

        // process the remainder, if there is a remainder to process
        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

        	let input0 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
        	let input1 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len);
        	let input2 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*2);
        	let input3 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*3);
        	let input4 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*4);
        	let input5 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*5);
        	let input6 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*6);
        	let input7 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*7);

        	let (output0, mid1, mid2, mid3, mid4, mid5, mid6, mid7) = avx_utils::column_butterfly8_fma_f32(input0, input1, input2, input3, input4, input5, input6, input7, self.twiddles_butterfly8, self.twiddle_config);
            
        	buffer.store_complex_remainder_f32(remainder_mask, output0, chunk_count*4);
        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 7);
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7), mid1);
        	buffer.store_complex_remainder_f32(remainder_mask, output1, chunk_count*4 + eigth_len);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7+1), mid2);
        	buffer.store_complex_remainder_f32(remainder_mask, output2, chunk_count*4 + eigth_len*2);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7+2), mid3);
        	buffer.store_complex_remainder_f32(remainder_mask, output3, chunk_count*4 + eigth_len*3);
        	let output4 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7+3), mid4);
        	buffer.store_complex_remainder_f32(remainder_mask, output4, chunk_count*4 + eigth_len*4);
        	let output5 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7+4), mid5);
        	buffer.store_complex_remainder_f32(remainder_mask, output5, chunk_count*4 + eigth_len*5);
        	let output6 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7+5), mid6);
        	buffer.store_complex_remainder_f32(remainder_mask, output6, chunk_count*4 + eigth_len*6);
        	let output7 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*7+6), mid7);
        	buffer.store_complex_remainder_f32(remainder_mask, output7, chunk_count*4 + eigth_len*7);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(eigth_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..len];

        for i in 0..chunk_count {
            let input0 = buffer.load_complex_f32(i * 4); 
            let input1 = buffer.load_complex_f32(i * 4 + eigth_len);
            let input2 = buffer.load_complex_f32(i * 4 + eigth_len*2);
            let input3 = buffer.load_complex_f32(i * 4 + eigth_len*3);
            let input4 = buffer.load_complex_f32(i * 4 + eigth_len*4);
            let input5 = buffer.load_complex_f32(i * 4 + eigth_len*5);
            let input6 = buffer.load_complex_f32(i * 4 + eigth_len*6);
            let input7 = buffer.load_complex_f32(i * 4 + eigth_len*7);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, transposed3) = avx_utils::transpose_4x4_f32(input0, input1, input2, input3);
            let (transposed4, transposed5, transposed6, transposed7) = avx_utils::transpose_4x4_f32(input4, input5, input6, input7);

            // store the first chunk directly back into 
            scratch.store_complex_f32(i * 32, transposed0);
            scratch.store_complex_f32(i * 32 + 4, transposed4);
            scratch.store_complex_f32(i * 32 + 4*2, transposed1);
            scratch.store_complex_f32(i * 32 + 4*3, transposed5);
            scratch.store_complex_f32(i * 32 + 4*4, transposed2);
            scratch.store_complex_f32(i * 32 + 4*5, transposed6);
            scratch.store_complex_f32(i * 32 + 4*6, transposed3);
            scratch.store_complex_f32(i * 32 + 4*7, transposed7);
        }

        // transpose the remainder, if there is a remainder to process
        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

            let input0 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len);
            let input2 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*2);
            let input3 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*3);
            let input4 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*4);
            let input5 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*5);
            let input6 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*6);
            let input7 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + eigth_len*7);

            // Transpose the 8x4 array and scatter them
            let (transposed0, transposed1, transposed2, _transposed3) = avx_utils::transpose_4x4_f32(input0, input1, input2, input3);
            let (transposed4, transposed5, transposed6, _transposed7) = avx_utils::transpose_4x4_f32(input4, input5, input6, input7);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            scratch.store_complex_f32(chunk_count*32, transposed0);
            scratch.store_complex_f32(chunk_count*32 + 4, transposed4);
            if remainder >= 2 {
                scratch.store_complex_f32(chunk_count*32 + 4*2, transposed1);
                scratch.store_complex_f32(chunk_count*32 + 4*3, transposed5);
                if remainder >= 3 {
                    scratch.store_complex_f32(chunk_count*32 + 4*4, transposed2);
                    scratch.store_complex_f32(chunk_count*32 + 4*5, transposed6);
                    // the remainder will never be 4 - because if it was 4, it would have been handled in the loop above
                    // so we can just not use the final 2 values. thankfully, the optimizer is smart enough to never even generate the last 2 values
                }
            }
        }
        buffer.copy_from_slice(scratch);
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix8xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix8xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix8xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for MixedRadix8xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

pub struct MixedRadix16xnAvx<T> {
    twiddle_config: avx_utils::Rotate90Config,
    twiddles_butterfly16: [__m256; 6],
    twiddles: Box<[__m256]>,
    inner_fft: Arc<FftInline<T>>,
    len: usize,
    inverse: bool,
}
impl MixedRadix16xnAvx<f32> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute the power-of-two FFT
    /// Returns Ok() if this machine has the required instruction sets, Err() if some instruction sets are missing
    #[inline]
    pub fn new(inner_fft: Arc<FftInline<f32>>) -> Result<Self, ()> {
        let has_avx = is_x86_feature_detected!("avx");
        let has_fma = is_x86_feature_detected!("fma");
        if has_avx && has_fma {
            // Safety: new_with_avx requires the "avx" feature set. Since we know it's present, we're safe
            Ok(unsafe { Self::new_with_avx(inner_fft) })
        } else {
            Err(())
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(inner_fft: Arc<FftInline<f32>>) -> Self {
        let inverse = inner_fft.is_inverse();
        let sixteenth_len = inner_fft.len();
        let len = sixteenth_len * 16;

        assert_eq!(len % 16, 0, "MixedRadix16xnAvx requires its FFT length to be a multiple of 16. Got {}", len);

        let quotient = sixteenth_len / 4;
        let remainder = sixteenth_len % 4;

        let num_twiddle_columns = quotient + if remainder > 0 { 1 } else { 0 };

        let mut twiddles = Vec::with_capacity(num_twiddle_columns * 15);
        for x in 0..num_twiddle_columns {
            let chunk_size = if x == quotient { remainder } else { 4 };

        	for y in 1..16 {
                let mut twiddle_chunk = [Complex::zero(); 4];
                for i in 0..chunk_size {
                    twiddle_chunk[i] = f32::generate_twiddle_factor(y*(x*4 + i), len, inverse);
                }

	            twiddles.push(twiddle_chunk.load_complex_f32(0));
        	}
        }

        Self {
        	twiddle_config: avx_utils::Rotate90Config::get_from_inverse(inverse),
        	twiddles_butterfly16: [
                avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(1, 16, inverse)),
                avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(2, 16, inverse)),
                avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(3, 16, inverse)),
                avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(4, 16, inverse)),
                avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(6, 16, inverse)),
                avx_utils::broadcast_complex_f32(f32::generate_twiddle_factor(9, 16, inverse)),
            ],
            twiddles: twiddles.into_boxed_slice(),
            inner_fft,
            len,
            inverse,
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        let len = self.len();
        let sixteenth_len = len / 16;

        let chunk_count = sixteenth_len / 4;
        let remainder = sixteenth_len % 4;

        // process the column FFTs
        for i in 0..chunk_count {
        	let input0  = buffer.load_complex_f32(i * 4); 
        	let input1  = buffer.load_complex_f32(i * 4 + sixteenth_len);
        	let input2  = buffer.load_complex_f32(i * 4 + sixteenth_len*2);
        	let input3  = buffer.load_complex_f32(i * 4 + sixteenth_len*3);
        	let input4  = buffer.load_complex_f32(i * 4 + sixteenth_len*4);
        	let input5  = buffer.load_complex_f32(i * 4 + sixteenth_len*5);
        	let input6  = buffer.load_complex_f32(i * 4 + sixteenth_len*6);
        	let input7  = buffer.load_complex_f32(i * 4 + sixteenth_len*7);
        	let input8  = buffer.load_complex_f32(i * 4 + sixteenth_len*8); 
        	let input9  = buffer.load_complex_f32(i * 4 + sixteenth_len*9);
        	let input10 = buffer.load_complex_f32(i * 4 + sixteenth_len*10);
        	let input11 = buffer.load_complex_f32(i * 4 + sixteenth_len*11);
        	let input12 = buffer.load_complex_f32(i * 4 + sixteenth_len*12);
        	let input13 = buffer.load_complex_f32(i * 4 + sixteenth_len*13);
        	let input14 = buffer.load_complex_f32(i * 4 + sixteenth_len*14);
        	let input15 = buffer.load_complex_f32(i * 4 + sixteenth_len*15);

        	let (output0, mid1, mid2, mid3, mid4, mid5, mid6, mid7, mid8, mid9, mid10, mid11, mid12, mid13, mid14, mid15)
        		= avx_utils::column_butterfly16_fma_f32(
        			input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, self.twiddles_butterfly16, self.twiddle_config
    			);

        	buffer.store_complex_f32(i * 4, output0);

        	debug_assert!(self.twiddles.len() >= (i+1) * 15);
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15),  mid1);
        	buffer.store_complex_f32(i * 4 + sixteenth_len, output1);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+1), mid2);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*2, output2);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+2), mid3);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*3, output3);
        	let output4 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+3), mid4);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*4, output4);
        	let output5 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+4), mid5);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*5, output5);
        	let output6 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+5), mid6);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*6, output6);
        	let output7 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+6), mid7);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*7, output7);
        	let output8 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+7), mid8);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*8, output8);
        	let output9 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+8), mid9);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*9, output9);
        	let output10 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+9), mid10);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*10, output10);
        	let output11 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+10), mid11);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*11, output11);
        	let output12 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+11), mid12);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*12, output12);
        	let output13 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+12), mid13);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*13, output13);
        	let output14 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+13), mid14);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*14, output14);
        	let output15 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(i*15+14), mid15);
        	buffer.store_complex_f32(i * 4 + sixteenth_len*15, output15);
        }

        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

        	let input0  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
        	let input1  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len);
        	let input2  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*2);
        	let input3  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*3);
        	let input4  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*4);
        	let input5  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*5);
        	let input6  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*6);
        	let input7  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*7);
        	let input8  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*8); 
        	let input9  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*9);
        	let input10 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*10);
        	let input11 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*11);
        	let input12 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*12);
        	let input13 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*13);
        	let input14 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*14);
        	let input15 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*15);

        	let (output0, mid1, mid2, mid3, mid4, mid5, mid6, mid7, mid8, mid9, mid10, mid11, mid12, mid13, mid14, mid15)
        		= avx_utils::column_butterfly16_fma_f32(
        			input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, self.twiddles_butterfly16, self.twiddle_config
    			);

        	buffer.store_complex_remainder_f32(remainder_mask, output0, chunk_count*4);

        	debug_assert!(self.twiddles.len() >= (chunk_count+1) * 15);
        	let output1 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15),  mid1);
        	buffer.store_complex_remainder_f32(remainder_mask, output1, chunk_count* 4 + sixteenth_len);
        	let output2 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+1), mid2);
        	buffer.store_complex_remainder_f32(remainder_mask, output2, chunk_count* 4 + sixteenth_len*2);
        	let output3 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+2), mid3);
        	buffer.store_complex_remainder_f32(remainder_mask, output3, chunk_count* 4 + sixteenth_len*3);
        	let output4 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+3), mid4);
        	buffer.store_complex_remainder_f32(remainder_mask, output4, chunk_count* 4 + sixteenth_len*4);
        	let output5 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+4), mid5);
        	buffer.store_complex_remainder_f32(remainder_mask, output5, chunk_count* 4 + sixteenth_len*5);
        	let output6 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+5), mid6);
        	buffer.store_complex_remainder_f32(remainder_mask, output6, chunk_count* 4 + sixteenth_len*6);
        	let output7 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+6), mid7);
        	buffer.store_complex_remainder_f32(remainder_mask, output7, chunk_count* 4 + sixteenth_len*7);
        	let output8 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+7), mid8);
        	buffer.store_complex_remainder_f32(remainder_mask, output8, chunk_count* 4 + sixteenth_len*8);
        	let output9 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+8), mid9);
        	buffer.store_complex_remainder_f32(remainder_mask, output9, chunk_count* 4 + sixteenth_len*9);
        	let output10 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+9), mid10);
        	buffer.store_complex_remainder_f32(remainder_mask, output10, chunk_count* 4 + sixteenth_len*10);
        	let output11 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+10), mid11);
        	buffer.store_complex_remainder_f32(remainder_mask, output11, chunk_count* 4 + sixteenth_len*11);
        	let output12 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+11), mid12);
        	buffer.store_complex_remainder_f32(remainder_mask, output12, chunk_count* 4 + sixteenth_len*12);
        	let output13 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+12), mid13);
        	buffer.store_complex_remainder_f32(remainder_mask, output13, chunk_count* 4 + sixteenth_len*13);
        	let output14 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+13), mid14);
        	buffer.store_complex_remainder_f32(remainder_mask, output14, chunk_count* 4 + sixteenth_len*14);
        	let output15 = avx_utils::complex_multiply_fma_f32(*self.twiddles.get_unchecked(chunk_count*15+14), mid15);
        	buffer.store_complex_remainder_f32(remainder_mask, output15, chunk_count* 4 + sixteenth_len*15);
        }

        // process the row FFTs
        for chunk in buffer.chunks_exact_mut(sixteenth_len) {
        	self.inner_fft.process_inline(chunk, scratch);
        }

        // transpose for the output. make sure to slice off any extra scratch first
       // transpose for the output. make sure to slice off any extra scratch first
        let scratch = &mut scratch[..len];

        for i in 0..chunk_count {
            let input0  = buffer.load_complex_f32(i * 4); 
            let input1  = buffer.load_complex_f32(i * 4 + sixteenth_len);
            let input2  = buffer.load_complex_f32(i * 4 + sixteenth_len*2);
            let input3  = buffer.load_complex_f32(i * 4 + sixteenth_len*3);
            let input4  = buffer.load_complex_f32(i * 4 + sixteenth_len*4);
            let input5  = buffer.load_complex_f32(i * 4 + sixteenth_len*5);
            let input6  = buffer.load_complex_f32(i * 4 + sixteenth_len*6);
            let input7  = buffer.load_complex_f32(i * 4 + sixteenth_len*7);
            let input8  = buffer.load_complex_f32(i * 4 + sixteenth_len*8); 
            let input9  = buffer.load_complex_f32(i * 4 + sixteenth_len*9);
            let input10 = buffer.load_complex_f32(i * 4 + sixteenth_len*10);
            let input11 = buffer.load_complex_f32(i * 4 + sixteenth_len*11);

            // Transpose the 8x4 array and scatter them
            let (transposed0,  transposed1,  transposed2,  transposed3)  = avx_utils::transpose_4x4_f32(input0, input1, input2, input3);
            scratch.store_complex_f32(i * 64, transposed0);
            let (transposed4,  transposed5,  transposed6,  transposed7)  = avx_utils::transpose_4x4_f32(input4, input5, input6, input7);
            scratch.store_complex_f32(i * 64 + 4, transposed4);
            let (transposed8,  transposed9,  transposed10, transposed11) = avx_utils::transpose_4x4_f32(input8, input9, input10, input11);
            scratch.store_complex_f32(i * 64 + 4*2, transposed8);
            let input12 = buffer.load_complex_f32(i * 4 + sixteenth_len*12);
            let input13 = buffer.load_complex_f32(i * 4 + sixteenth_len*13);
            let input14 = buffer.load_complex_f32(i * 4 + sixteenth_len*14);
            let input15 = buffer.load_complex_f32(i * 4 + sixteenth_len*15);
            let (transposed12, transposed13, transposed14, transposed15) = avx_utils::transpose_4x4_f32(input12, input13, input14, input15);

            // store the first chunk directly back into 
            scratch.store_complex_f32(i * 64 + 4*3, transposed12);
            scratch.store_complex_f32(i * 64 + 4*4, transposed1);
            scratch.store_complex_f32(i * 64 + 4*5, transposed5);
            scratch.store_complex_f32(i * 64 + 4*6, transposed9);
            scratch.store_complex_f32(i * 64 + 4*7, transposed13);
            scratch.store_complex_f32(i * 64 + 4*8, transposed2);
            scratch.store_complex_f32(i * 64 + 4*9, transposed6);
            scratch.store_complex_f32(i * 64 + 4*10, transposed10);
            scratch.store_complex_f32(i * 64 + 4*11, transposed14);
            scratch.store_complex_f32(i * 64 + 4*12, transposed3);
            scratch.store_complex_f32(i * 64 + 4*13, transposed7);
            scratch.store_complex_f32(i * 64 + 4*14, transposed11);
            scratch.store_complex_f32(i * 64 + 4*15, transposed15);
        }

        if remainder > 0 {
            let remainder_mask = avx_utils::RemainderMask::new_f32(remainder);

            let input0  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4); 
            let input1  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len);
            let input2  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*2);
            let input3  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*3);
            let input4  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*4);
            let input5  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*5);
            let input6  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*6);
            let input7  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*7);
            let input8  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*8); 
            let input9  = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*9);
            let input10 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*10);
            let input11 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*11);
            let input12 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*12);
            let input13 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*13);
            let input14 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*14);
            let input15 = buffer.load_complex_remainder_f32(remainder_mask, chunk_count*4 + sixteenth_len*15);

            // Transpose the 8x4 array and scatter them
            let (transposed0,  transposed1,  transposed2,  _transposed3)  = avx_utils::transpose_4x4_f32(input0, input1, input2, input3);
            scratch.store_complex_f32(chunk_count*64, transposed0);
            let (transposed4,  transposed5,  transposed6,  _transposed7)  = avx_utils::transpose_4x4_f32(input4, input5, input6, input7);
            scratch.store_complex_f32(chunk_count*64 + 4, transposed4);
            let (transposed8,  transposed9,  transposed10, _transposed11) = avx_utils::transpose_4x4_f32(input8, input9, input10, input11);
            scratch.store_complex_f32(chunk_count*64 + 4*2, transposed8);
            let (transposed12, transposed13, transposed14, _transposed15) = avx_utils::transpose_4x4_f32(input12, input13, input14, input15);
            scratch.store_complex_f32(chunk_count*64 + 4*3, transposed12);

            // store the transposed remainder back into the buffer -- but keep in account the fact we should only write out some of the chunks!
            if remainder >= 2 {
                scratch.store_complex_f32(chunk_count*64 + 4*4, transposed1);
                scratch.store_complex_f32(chunk_count*64 + 4*5, transposed5);
                scratch.store_complex_f32(chunk_count*64 + 4*6, transposed9);
                scratch.store_complex_f32(chunk_count*64 + 4*7, transposed13);
                if remainder >= 3 {
                    scratch.store_complex_f32(chunk_count*64 + 4*8, transposed2);
                    scratch.store_complex_f32(chunk_count*64 + 4*9, transposed6);
                    scratch.store_complex_f32(chunk_count*64 + 4*10, transposed10);
                    scratch.store_complex_f32(chunk_count*64 + 4*11, transposed14);
                    // the remainder will never be 4 - because if it was 4, it would have been handled in the loop above
                    // so we can just not use the final 2 values. thankfully, the optimizer is smart enough to never even generate the last 2 values
                }
            }
        }
        buffer.copy_from_slice(scratch);
    }
}
default impl<T: FFTnum> FftInline<T> for MixedRadix16xnAvx<T> {
    fn process_inline(&self, _buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        unimplemented!();
    }
    fn get_required_scratch_len(&self) -> usize {
        unimplemented!();
    }
}
impl FftInline<f32> for MixedRadix16xnAvx<f32> {
    fn process_inline(&self, buffer: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) {
        verify_length_inline(buffer, self.len());
        verify_length_minimum(scratch, self.get_required_scratch_len());

        unsafe { self.perform_fft_f32(buffer, scratch) };
    }
    #[inline]
    fn get_required_scratch_len(&self) -> usize {
        self.len()
    }
}

impl<T> Length for MixedRadix16xnAvx<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> IsInverse for MixedRadix16xnAvx<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_utils::check_inline_fft_algorithm;
    use std::sync::Arc;
    use algorithm::*;

    #[test]
    fn test_mixedradix_2xn_avx() {
        for pow in 2..8 {
            for remainder in 0..4 {
                let len = (1 << pow) + 2 * remainder;
                test_mixedradix_2xn_avx_with_length(len, false);
                test_mixedradix_2xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_2xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 2, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix2xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_4xn_avx() {
        for pow in 2..8 {
            for remainder in 0..4 {
                let len = (1 << pow) + 4 * remainder;
                test_mixedradix_4xn_avx_with_length(len, false);
                test_mixedradix_4xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_4xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 4, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix4xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_8xn_avx() {
        for pow in 3..9 {
            for remainder in 0..4 {
                let len = (1 << pow) + remainder * 8;
                test_mixedradix_8xn_avx_with_length(len, false);
                test_mixedradix_8xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_8xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 8, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix8xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }

    #[test]
    fn test_mixedradix_16xn_avx() {
        for pow in 4..10 {
            for remainder in 0..4 {
                let len = (1 << pow) + remainder * 16;
                test_mixedradix_16xn_avx_with_length(len, false);
                test_mixedradix_16xn_avx_with_length(len, true);
            }
        }
    }

    fn test_mixedradix_16xn_avx_with_length(len: usize, inverse: bool) {
        let inner_fft = Arc::new(DFT::new(len / 16, inverse)) as Arc<dyn FftInline<f32>>;
        let fft = MixedRadix16xnAvx::new(inner_fft).expect("Can't run test because this machine doesn't have the required instruction sets");

        check_inline_fft_algorithm(&fft, len, inverse);
    }
}