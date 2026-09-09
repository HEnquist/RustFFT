// Temporary tuning harness: where should the Rader's cutoff sit, per float type?
//
// For each prime, the planner's choice is decided by the largest prime factor of len - 1. This
// times Rader's against Bluestein's for the same prime and reports the ratio grouped by that
// factor, so the cutoff can be read off directly: the largest factor whose bucket still has
// Rader's winning.
use rustfft::algorithm::{BluesteinsAlgorithm, RadersAlgorithm};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftNum, FftPlannerWasmSimd};
use std::sync::Arc;
use std::time::Instant;

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

fn max_prime_factor(n: usize) -> usize {
    let (mut m, mut d, mut best) = (n, 2, 1);
    while d * d <= m {
        while m % d == 0 {
            best = d;
            m /= d;
        }
        d += 1;
    }
    if m > 1 {
        best = m;
    }
    best
}

fn time_fft<T: FftNum>(fft: &Arc<dyn Fft<T>>) -> f64 {
    let mut buffer: Vec<Complex<T>> = vec![Complex::zero(); fft.len()];
    let mut scratch: Vec<Complex<T>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
    let mut iters = 1usize;
    loop {
        let s = Instant::now();
        for _ in 0..iters {
            fft.process_with_scratch(&mut buffer, &mut scratch);
        }
        if s.elapsed().as_secs_f64() > 0.02 || iters > 1 << 28 {
            break;
        }
        iters *= 2;
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let s = Instant::now();
        for _ in 0..iters {
            fft.process_with_scratch(&mut buffer, &mut scratch);
        }
        let p = s.elapsed().as_secs_f64() / iters as f64;
        if p < best {
            best = p;
        }
    }
    best * 1e9
}

fn bluestein_inner_len(len: usize) -> usize {
    let min_inner_len = 2 * len - 1;
    let pow2 = min_inner_len.checked_next_power_of_two().unwrap();
    let factor3 = pow2 / 4 * 3;
    if factor3 >= min_inner_len {
        factor3
    } else {
        pow2
    }
}

fn measure<T: FftNum>(len: usize) -> (f64, f64) {
    let mut planner = FftPlannerWasmSimd::<T>::new().unwrap();
    let raders: Arc<dyn Fft<T>> = Arc::new(RadersAlgorithm::new(planner.plan_fft_forward(len - 1)));
    let bluesteins: Arc<dyn Fft<T>> = Arc::new(BluesteinsAlgorithm::new(
        len,
        planner.plan_fft_forward(bluestein_inner_len(len)),
    ));
    (time_fft(&raders), time_fft(&bluesteins))
}

fn main() {
    // A spread of primes per bucket, across a wide range of sizes.
    let mut by_bucket: Vec<(usize, usize)> = Vec::new();
    for len in 100..80_000 {
        if is_prime(len) {
            let pf = max_prime_factor(len - 1);
            if pf <= 61 {
                by_bucket.push((pf, len));
            }
        }
    }
    by_bucket.sort();

    println!("bucket,len,f32_raders_ns,f32_bluesteins_ns,f64_raders_ns,f64_bluesteins_ns");
    let mut per_bucket: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (pf, len) in by_bucket {
        per_bucket.entry(pf).or_default().push(len);
    }
    for (pf, lens) in per_bucket {
        // Sample evenly across the size range so a bucket isn't dominated by tiny lengths.
        let step = (lens.len() / 12).max(1);
        for len in lens.iter().step_by(step).take(12) {
            let (r32, b32) = measure::<f32>(*len);
            let (r64, b64) = measure::<f64>(*len);
            println!("{},{},{:.1},{:.1},{:.1},{:.1}", pf, len, r32, b32, r64, b64);
        }
    }
}
