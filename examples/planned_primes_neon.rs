// Times whatever the planner picks for a spread of primes, so a cutoff change can be checked
// end to end rather than only as a Rader's against Bluestein's A/B.
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftNum, FftPlannerNeon};
use std::sync::Arc;
use std::time::Instant;

const PRIMES: &[usize] = &[
    229, 311, 353, 659, 661, 677, 683, 859, 883, 929, 947, 1009, 1093, 1171, 1621, 1657, 1741,
    1801, 1861, 2053, 2069, 2221, 2521, 2953, 3301, 3361, 3511, 3907, 4019, 4231, 4999, 5333, 5591,
    5851, 7411, 7591, 8269, 8273, 8527, 8971, 9521, 9769, 10151, 10333, 10657, 10831, 11593, 12097,
    13159, 13781, 14401, 14851, 15361, 16073, 16369, 16607, 17137, 17681, 18061, 19001, 19801,
    20521, 20593, 20641, 20747, 21143, 21577, 21601, 22441, 22679, 23041, 23311, 24421, 25601,
    25741, 26317, 26497, 26641, 26681, 28051, 28351, 30241, 30637, 30977, 31907, 32833, 35251,
    35281, 35617, 35911, 36191, 36709, 41023, 43793, 44201, 45943, 46691, 47521, 47737, 47807,
    48779, 49037, 51941, 53281, 54367, 55903, 57529, 58321, 58997, 59149, 59779, 59879,
];

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

fn main() {
    let mut p32 = FftPlannerNeon::<f32>::new().unwrap();
    let mut p64 = FftPlannerNeon::<f64>::new().unwrap();
    println!("len,f32_ns,f64_ns");
    for &len in PRIMES {
        let a = p32.plan_fft_forward(len);
        let b = p64.plan_fft_forward(len);
        println!("{},{:.1},{:.1}", len, time_fft(&a), time_fft(&b));
    }
}
