#![allow(bare_trait_objects)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;
extern crate rustfft;
use rustfft::FftNum;
use num_complex::Complex;

use std::sync::Arc;
use test::Bencher;


fn run_prepare_segmented<T: FftNum>(
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    base_len: usize,
) {
    // copy the data into the spectrum vector, split the copying up into chunks to make it more cache friendly
    let mut num_chunks = signal.len() / 8192;
    if num_chunks == 0 {
        num_chunks = 1;
    } else if num_chunks > base_len {
        num_chunks = base_len;
    }
    for n in 0..num_chunks {
        prepare_radix4_segmented(
            signal.len(),
            base_len,
            signal,
            spectrum,
            1,
            n,
            num_chunks,
        );
    }
}

fn prepare_radix4_segmented<T: FftNum>(
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
            prepare_radix4_segmented(
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

fn bench_segmented(b: &mut Bencher, len: usize) {
    let signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    // figure out which base length we're going to use
    let num_bits = len.trailing_zeros();
    let base_len = match num_bits {
        0 => len,
        1 => len,
        2 => len,
        _ => {
            if num_bits % 2 == 1 {
                8
            } else {
                16
            }
        }
    };
    b.iter(|| {run_prepare_segmented(&signal, &mut spectrum, base_len);} );
}

#[bench] fn prep_______64_segmented(b: &mut Bencher) { bench_segmented(b, 64); }
#[bench] fn prep______256_segmented(b: &mut Bencher) { bench_segmented(b, 256); }
#[bench] fn prep______512_segmented(b: &mut Bencher) { bench_segmented(b, 512); }
#[bench] fn prep_____1024_segmented(b: &mut Bencher) { bench_segmented(b, 1024); }
#[bench] fn prep_____2048_segmented(b: &mut Bencher) { bench_segmented(b, 2048); }
#[bench] fn prep_____4096_segmented(b: &mut Bencher) { bench_segmented(b, 4096); }
#[bench] fn prep_____8192_segmented(b: &mut Bencher) { bench_segmented(b, 8192); }
#[bench] fn prep____16384_segmented(b: &mut Bencher) { bench_segmented(b, 16384); }
#[bench] fn prep____32768_segmented(b: &mut Bencher) { bench_segmented(b, 32768); }
#[bench] fn prep____65536_segmented(b: &mut Bencher) { bench_segmented(b, 65536); }
#[bench] fn prep___131072_segmented(b: &mut Bencher) { bench_segmented(b, 131072); }
#[bench] fn prep___262144_segmented(b: &mut Bencher) { bench_segmented(b, 262144); }
#[bench] fn prep___524288_segmented(b: &mut Bencher) { bench_segmented(b, 524288); }
#[bench] fn prep__1048576_segmented(b: &mut Bencher) { bench_segmented(b, 1048576); }
#[bench] fn prep__2097152_segmented(b: &mut Bencher) { bench_segmented(b, 2097152); }
#[bench] fn prep__4194304_segmented(b: &mut Bencher) { bench_segmented(b, 4194304); }
//#[bench] fn radix4__8388608(b: &mut Bencher) { bench_radix4(b, 8388608); }
//#[bench] fn radix4_16777216(b: &mut Bencher) { bench_radix4(b, 16777216); }

fn run_prepare_current<T: FftNum>(
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    base_len: usize,
) {

    prepare_radix4(
        signal.len(),
        base_len,
        signal,
        spectrum,
        1,
    );
}

fn prepare_radix4<T: FftNum>(
    size: usize,
    base_len: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
) {
    if size == base_len {
        unsafe {
            for i in 0..size {
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
            );
        }
    }
}

fn bench_current(b: &mut Bencher, len: usize) {
    let signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    // figure out which base length we're going to use
    let num_bits = len.trailing_zeros();
    let base_len = match num_bits {
        0 => len,
        1 => len,
        2 => len,
        _ => {
            if num_bits % 2 == 1 {
                8
            } else {
                16
            }
        }
    };
    b.iter(|| {run_prepare_current(&signal, &mut spectrum, base_len);} );
}

#[bench] fn prep_______64_current(b: &mut Bencher) { bench_current(b, 64); }
#[bench] fn prep______256_current(b: &mut Bencher) { bench_current(b, 256); }
#[bench] fn prep______512_current(b: &mut Bencher) { bench_current(b, 512); }
#[bench] fn prep_____1024_current(b: &mut Bencher) { bench_current(b, 1024); }
#[bench] fn prep_____2048_current(b: &mut Bencher) { bench_current(b, 2048); }
#[bench] fn prep_____4096_current(b: &mut Bencher) { bench_current(b, 4096); }
#[bench] fn prep_____8192_current(b: &mut Bencher) { bench_current(b, 8192); }
#[bench] fn prep____16384_current(b: &mut Bencher) { bench_current(b, 16384); }
#[bench] fn prep____32768_current(b: &mut Bencher) { bench_current(b, 32768); }
#[bench] fn prep____65536_current(b: &mut Bencher) { bench_current(b, 65536); }
#[bench] fn prep___131072_current(b: &mut Bencher) { bench_current(b, 131072); }
#[bench] fn prep___262144_current(b: &mut Bencher) { bench_current(b, 262144); }
#[bench] fn prep___524288_current(b: &mut Bencher) { bench_current(b, 524288); }
#[bench] fn prep__1048576_current(b: &mut Bencher) { bench_current(b, 1048576); }
#[bench] fn prep__2097152_current(b: &mut Bencher) { bench_current(b, 2097152); }
#[bench] fn prep__4194304_current(b: &mut Bencher) { bench_current(b, 4194304); }
//#[bench] fn radix4__8388608(b: &mut Bencher) { bench_radix4(b, 8388608); }
//#[bench] fn radix4_16777216(b: &mut Bencher) { bench_radix4(b, 16777216); }'

/*
fn run_prepare_tweaked<T: FftNum>(
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    base_len: usize,
) {
    prepare_radix4_tweaked(
        signal.len(),
        base_len,
        signal,
        spectrum,
        1,
    );
}

fn prepare_radix4_tweaked<T: FftNum>(
    size: usize,
    base_len: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
) {
    if size == (4 * base_len) {
        do_radix4_shuffle_tweaked(size, signal, spectrum, stride);
    } else if size == base_len {
        unsafe {
            for i in 0..base_len {
                *spectrum.get_unchecked_mut(i) = *signal.get_unchecked(i * stride);
            }
        }
    } else {
        for i in 0..4 {
            prepare_radix4_tweaked(
                size / 4,
                base_len,
                &signal[i * stride..],
                &mut spectrum[i * (size / 4)..],
                stride * 4,
            );
        }
    }
}

fn do_radix4_shuffle_tweaked<T: FftNum>(
    size: usize,
    signal: &[Complex<T>],
    spectrum: &mut [Complex<T>],
    stride: usize,
) {
    let stepsize = size / 4;
    let stepstride = stride * 4;
    let signal_offset = stride;
    let spectrum_offset = size / 4;
    unsafe {
        for i in 0..stepsize {
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

fn bench_tweaked(b: &mut Bencher, len: usize) {
    let signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    // figure out which base length we're going to use
    let num_bits = len.trailing_zeros();
    let base_len = match num_bits {
        0 => len,
        1 => len,
        2 => len,
        _ => {
            if num_bits % 2 == 1 {
                8
            } else {
                16
            }
        }
    };
    b.iter(|| {run_prepare_tweaked(&signal, &mut spectrum, base_len);} );
}

#[bench] fn prep_______64_tweaked(b: &mut Bencher) { bench_tweaked(b, 64); }
#[bench] fn prep______256_tweaked(b: &mut Bencher) { bench_tweaked(b, 256); }
#[bench] fn prep______512_tweaked(b: &mut Bencher) { bench_tweaked(b, 512); }
#[bench] fn prep_____1024_tweaked(b: &mut Bencher) { bench_tweaked(b, 1024); }
#[bench] fn prep_____2048_tweaked(b: &mut Bencher) { bench_tweaked(b, 2048); }
#[bench] fn prep_____4096_tweaked(b: &mut Bencher) { bench_tweaked(b, 4096); }
#[bench] fn prep_____8192_tweaked(b: &mut Bencher) { bench_tweaked(b, 8192); }
#[bench] fn prep____16384_tweaked(b: &mut Bencher) { bench_tweaked(b, 16384); }
#[bench] fn prep____32768_tweaked(b: &mut Bencher) { bench_tweaked(b, 32768); }
#[bench] fn prep____65536_tweaked(b: &mut Bencher) { bench_tweaked(b, 65536); }
#[bench] fn prep___131072_tweaked(b: &mut Bencher) { bench_tweaked(b, 131072); }
#[bench] fn prep___262144_tweaked(b: &mut Bencher) { bench_tweaked(b, 262144); }
#[bench] fn prep___524288_tweaked(b: &mut Bencher) { bench_tweaked(b, 524288); }
#[bench] fn prep__1048576_tweaked(b: &mut Bencher) { bench_tweaked(b, 1048576); }
#[bench] fn prep__2097152_tweaked(b: &mut Bencher) { bench_tweaked(b, 2097152); }
#[bench] fn prep__4194304_tweaked(b: &mut Bencher) { bench_tweaked(b, 4194304); }
//#[bench] fn radix4__8388608(b: &mut Bencher) { bench_radix4(b, 8388608); }
//#[bench] fn radix4_16777216(b: &mut Bencher) { bench_radix4(b, 16777216); }
*/

pub fn reverse_bits(value: usize, bits: usize) -> usize {
    let mut result: usize = 0; 
    let mut value = value;
    for _ in 0..bits {
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

pub unsafe fn bitrev_transpose_2<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T], shuffled: &[(usize, usize)]) {
    for y in 0..height {
        //for x in 0..width {
        for (x, x_rev) in shuffled.iter() {
            //let x_rev = reverse_bits(x, bits);
            //let x_rev = shuffled.get_unchecked(x); 
        
            let input_index = x_rev + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

fn bench_reverse(b: &mut Bencher, len: usize) {
    let signal = vec![Complex{re: 0_f32, im: 0_f32}; len];
    let mut spectrum = signal.clone();
    // figure out which base length we're going to use
    let num_bits = len.trailing_zeros();
    let base_len = match num_bits {
        0 => len,
        1 => len,
        2 => len,
        _ => {
            if num_bits % 2 == 1 {
                8
            } else {
                16
            }
        }
    };
    let width = len/base_len;
    let rev_bits = (width.trailing_zeros()/2) as usize;
    let mut shuffled = (0..width).map(|val| (val, reverse_bits(val, rev_bits))).collect::<Vec<(usize, usize)>>();
    if width > (4*8192) {
        let chunks = width/(4*8192);
        let width_per_chunk = width / chunks;
        shuffled.sort_by(|a, b| (a.1/width_per_chunk).partial_cmp(&(b.1/width_per_chunk)).unwrap());
    }
    
    //b.iter(|| unsafe {bitrev_transpose(width, base_len, &signal, &mut spectrum, rev_bits);} );
    b.iter(|| unsafe {bitrev_transpose_2(width, base_len, &signal, &mut spectrum, &shuffled);} );
}

#[bench] fn prep_______64_reverse(b: &mut Bencher) { bench_reverse(b, 64); }
#[bench] fn prep______256_reverse(b: &mut Bencher) { bench_reverse(b, 256); }
#[bench] fn prep______512_reverse(b: &mut Bencher) { bench_reverse(b, 512); }
#[bench] fn prep_____1024_reverse(b: &mut Bencher) { bench_reverse(b, 1024); }
#[bench] fn prep_____2048_reverse(b: &mut Bencher) { bench_reverse(b, 2048); }
#[bench] fn prep_____4096_reverse(b: &mut Bencher) { bench_reverse(b, 4096); }
#[bench] fn prep_____8192_reverse(b: &mut Bencher) { bench_reverse(b, 8192); }
#[bench] fn prep____16384_reverse(b: &mut Bencher) { bench_reverse(b, 16384); }
#[bench] fn prep____32768_reverse(b: &mut Bencher) { bench_reverse(b, 32768); }
#[bench] fn prep____65536_reverse(b: &mut Bencher) { bench_reverse(b, 65536); }
#[bench] fn prep___131072_reverse(b: &mut Bencher) { bench_reverse(b, 131072); }
#[bench] fn prep___262144_reverse(b: &mut Bencher) { bench_reverse(b, 262144); }
#[bench] fn prep___524288_reverse(b: &mut Bencher) { bench_reverse(b, 524288); }
#[bench] fn prep__1048576_reverse(b: &mut Bencher) { bench_reverse(b, 1048576); }
#[bench] fn prep__2097152_reverse(b: &mut Bencher) { bench_reverse(b, 2097152); }
#[bench] fn prep__4194304_reverse(b: &mut Bencher) { bench_reverse(b, 4194304); }
//#[bench] fn radix4__8388608(b: &mut Bencher) { bench_radix4(b, 8388608); }
//#[bench] fn radix4_16777216(b: &mut Bencher) { bench_radix4(b, 16777216); }