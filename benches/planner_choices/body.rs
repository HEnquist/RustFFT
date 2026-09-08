//! The body of the planner choice benchmarks, shared by the SIMD backends.
//!
//! Each backend gets a thin bench target that invokes `planner_choice_benches!` with its own
//! planner type. See `bench_planner_choices_sse.rs`.
//!
//! The benches come in two kinds:
//!
//! * `planned_*` measures whatever the planner picks for a length. The lengths are grouped so
//!   that each group lands on one planner decision, so a run says how the shipped choices hold up
//!   on a given machine. Diffing these between two revisions is how a planner change gets checked.
//!
//! * `alt_*` measures the alternative the planner turned down for the same length, built by hand
//!   out of the same public algorithms the planner itself builds. Comparing an `alt_*` against
//!   the `planned_*` of the same length is what motivates a choice rather than just recording it.
//!
//! One gap worth naming: the SIMD RadixN is private to the crate, so a bench target can't build
//! it directly. The RadixN alternative is therefore only measurable where the planner already
//! picks it, which is f32. For f64 the `planned` and `alt_mixedradix` benches of a length are the
//! same algorithm and should agree. That the f64 planner is right to decline RadixN was measured
//! by building the crate both ways, not from here.

/// One `#[bench]` per length in a group, for both float types, for each named bench function.
///
/// The length list is forwarded as a single token tree, because a `macro_rules!` repetition can't
/// take the cross product of two lists at the same depth.
macro_rules! bench_group {
    ($group:ident, $lengths:tt, [ $($which:ident),* ]) => {
        $(
            bench_group_one!($group, $lengths, $which);
        )*
    };
}

macro_rules! bench_group_one {
    ($group:ident, { $($len:literal),* }, $which:ident) => {
        paste! {
            $(
                #[bench]
                fn [<bench_ $group _ $which _f32_ $len>](b: &mut Bencher) {
                    $which::<f32>(b, $len);
                }

                #[bench]
                fn [<bench_ $group _ $which _f64_ $len>](b: &mut Bencher) {
                    $which::<f64>(b, $len);
                }
            )*
        }
    };
}

macro_rules! planner_choice_benches {
    ($planner:ident) => {
        use pastey::paste;
        use rustfft::algorithm::{BluesteinsAlgorithm, MixedRadix, RadersAlgorithm};
        use rustfft::num_complex::Complex;
        use rustfft::num_traits::Zero;
        use rustfft::{Fft, FftNum};
        use std::sync::Arc;
        use test::Bencher;

        fn bench_fft<T: FftNum>(b: &mut Bencher, fft: Arc<dyn Fft<T>>) {
            let mut buffer: Vec<Complex<T>> = vec![Complex::zero(); fft.len()];
            let mut scratch: Vec<Complex<T>> = vec![Complex::zero(); fft.get_inplace_scratch_len()];
            b.iter(|| fft.process_with_scratch(&mut buffer, &mut scratch));
        }

        /// Whatever the planner decides for this length.
        fn planned<T: FftNum>(b: &mut Bencher, len: usize) {
            let mut planner = $planner::<T>::new().unwrap();
            bench_fft(b, planner.plan_fft_forward(len));
        }

        /// The mixed radix the planner falls back to when it turns RadixN down: the power of two
        /// peeled off the front, and the rest as the other side.
        fn alt_mixedradix<T: FftNum>(b: &mut Bencher, len: usize) {
            let mut planner = $planner::<T>::new().unwrap();
            let power_of_two = 1 << len.trailing_zeros();
            let left = planner.plan_fft_forward(power_of_two);
            let right = planner.plan_fft_forward(len / power_of_two);
            bench_fft(b, Arc::new(MixedRadix::new(left, right)) as Arc<dyn Fft<T>>);
        }

        /// Rader's for a prime, whatever the cutoff would have said.
        fn alt_raders<T: FftNum>(b: &mut Bencher, len: usize) {
            let mut planner = $planner::<T>::new().unwrap();
            let inner = planner.plan_fft_forward(len - 1);
            bench_fft(b, Arc::new(RadersAlgorithm::new(inner)) as Arc<dyn Fft<T>>);
        }

        /// Bluestein's for a prime, at the inner length the planner would have chosen.
        fn alt_bluesteins<T: FftNum>(b: &mut Bencher, len: usize) {
            let mut planner = $planner::<T>::new().unwrap();

            // The same inner length rule design_prime uses: the next power of two, or three
            // quarters of it when that still clears 2 * len - 1.
            let min_inner_len = 2 * len - 1;
            let inner_len_pow2 = min_inner_len.checked_next_power_of_two().unwrap();
            let inner_len_factor3 = inner_len_pow2 / 4 * 3;
            let inner_len = if inner_len_factor3 >= min_inner_len {
                inner_len_factor3
            } else {
                inner_len_pow2
            };

            let inner = planner.plan_fft_forward(inner_len);
            bench_fft(b, Arc::new(BluesteinsAlgorithm::new(len, inner)) as Arc<dyn Fft<T>>);
        }

        // Mixed-factor lengths with at least six factors of two and something else on top, which
        // is where RadixN is in play. f32 uses it, f64 declines and peels the power of two off
        // the front instead, which is exactly what alt_mixedradix builds.
        bench_group!(radixn, {1152, 2880, 7680, 11520, 23040, 46080}, [planned, alt_mixedradix]);

        // Odd mixed-factor lengths. An f32 vector holds two complex numbers, so every cross-FFT
        // layer needs an even column count and there is no factor of two to spare for the base.
        // Neither type can use RadixN here, so these should track alt_mixedradix's cousin, the
        // partitioned mixed radix.
        bench_group!(radixn_odd, {1215, 3125, 10125}, [planned]);

        // Pure powers of two are Radix4's job rather than the generic driver's. A control group:
        // a change to the RadixN decision should leave these alone.
        bench_group!(pow2, {4096, 65536}, [planned]);

        // Primes whose `len - 1` factors entirely into butterflies of 23 or less, so every cutoff
        // from 23 up picks Rader's. A control group for cutoff changes, and the alt says how much
        // Rader's is winning by.
        bench_group!(prime_rader, {1453, 2081, 11731}, [planned, alt_bluesteins]);

        // Primes whose `len - 1` has a prime factor of 29 or 31. These are exactly the lengths
        // max_rader_prime_factor decides: admitted they use Rader's, otherwise Bluestein's. Both
        // alternatives are benched, so the cutoff can be read straight off the numbers.
        bench_group!(prime_cutoff, {2729, 8867, 33641}, [planned, alt_raders, alt_bluesteins]);

        // Primes whose `len - 1` has a prime factor far above any butterfly, so Rader's has to
        // nest another prime algorithm inside itself and Bluestein's wins under any cutoff.
        bench_group!(prime_bluestein, {9931, 43391}, [planned, alt_raders]);
    };
}
