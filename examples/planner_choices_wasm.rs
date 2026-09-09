// Temporary tuning harness: the planner choice bench groups, as a plain program.
//
// The #[bench] harness does not run on wasm32, so this reproduces the groups from
// benches/planner_choices/body.rs with the same timing loop the other harnesses use.
use rustfft::algorithm::{BluesteinsAlgorithm, MixedRadix, RadersAlgorithm};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftNum, FftPlannerWasmSimd};
use std::sync::Arc;
use std::time::Instant;

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

fn planned<T: FftNum>(len: usize) -> f64 {
    let mut p = FftPlannerWasmSimd::<T>::new().unwrap();
    time_fft(&p.plan_fft_forward(len))
}

fn alt_mixedradix<T: FftNum>(len: usize) -> f64 {
    let mut p = FftPlannerWasmSimd::<T>::new().unwrap();
    let pow2 = 1 << len.trailing_zeros();
    let left = p.plan_fft_forward(pow2);
    let right = p.plan_fft_forward(len / pow2);
    time_fft(&(Arc::new(MixedRadix::new(left, right)) as Arc<dyn Fft<T>>))
}

fn alt_raders<T: FftNum>(len: usize) -> f64 {
    let mut p = FftPlannerWasmSimd::<T>::new().unwrap();
    let inner = p.plan_fft_forward(len - 1);
    time_fft(&(Arc::new(RadersAlgorithm::new(inner)) as Arc<dyn Fft<T>>))
}

fn alt_bluesteins<T: FftNum>(len: usize) -> f64 {
    let mut p = FftPlannerWasmSimd::<T>::new().unwrap();
    let inner = p.plan_fft_forward(bluestein_inner_len(len));
    time_fft(&(Arc::new(BluesteinsAlgorithm::new(len, inner)) as Arc<dyn Fft<T>>))
}

fn row(group: &str, which: &str, len: usize, f32_ns: f64, f64_ns: f64) {
    println!("{},{},{},{:.1},{:.1}", group, which, len, f32_ns, f64_ns);
}

fn main() {
    println!("group,which,len,f32_ns,f64_ns");

    for &len in &[1152usize, 2880, 7680, 11520, 23040, 46080] {
        row("radixn", "planned", len, planned::<f32>(len), planned::<f64>(len));
        row("radixn", "alt_mixedradix", len, alt_mixedradix::<f32>(len), alt_mixedradix::<f64>(len));
    }
    for &len in &[1215usize, 3125, 10125] {
        row("radixn_odd", "planned", len, planned::<f32>(len), planned::<f64>(len));
    }
    for &len in &[4096usize, 65536] {
        row("pow2", "planned", len, planned::<f32>(len), planned::<f64>(len));
    }
    for &len in &[1297usize, 5881, 22051] {
        row("prime_rader", "planned", len, planned::<f32>(len), planned::<f64>(len));
        row("prime_rader", "alt_bluesteins", len, alt_bluesteins::<f32>(len), alt_bluesteins::<f64>(len));
    }
    for &len in &[1301usize, 5501, 22541] {
        row("prime_split", "planned", len, planned::<f32>(len), planned::<f64>(len));
        row("prime_split", "alt_raders", len, alt_raders::<f32>(len), alt_raders::<f64>(len));
        row("prime_split", "alt_bluesteins", len, alt_bluesteins::<f32>(len), alt_bluesteins::<f64>(len));
    }
    for &len in &[1229usize, 5503, 22481] {
        row("prime_bluestein", "planned", len, planned::<f32>(len), planned::<f64>(len));
        row("prime_bluestein", "alt_raders", len, alt_raders::<f32>(len), alt_raders::<f64>(len));
    }
}
