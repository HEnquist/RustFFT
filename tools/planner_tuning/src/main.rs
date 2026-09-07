//! Measurement tool for RustFFT's planners.
//!
//! Works against any planner that implements `TunablePlanner`, selected with `--planner`.
//! Recipes are built through the planner's own internals, so what is measured here is exactly
//! what the planner would construct.
//!
//! Subcommands:
//!   time SPEC...              time recipes against each other; a '*reps' suffix runs a recipe
//!                             over a len*reps buffer, which is how inner FFTs are invoked
//!   regret LEN...             measure how far the planner's pick is from the best available
//!   model TRAIN... 0 TEST...  calibrate a cost model and score its picks the same way
//!   residuals LEN...          show how per-element overhead varies with working set
//!   verify LEN...             check every enumerated candidate against a direct DFT
//!   emit LEN...               print the fitted model as a Rust source file

mod emit;
mod model;

use model::{bucket_of, overhead_scale, Model, FIT_ORDER};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{ToPrimitive, Zero};
use rustfft::tuning::{
    candidates_capped, parse, to_spec_string, ScalarTuner, Spec, TunablePlanner,
};
use rustfft::{Fft, FftDirection, FftNum};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Subject<T: FftNum> {
    name: String,
    fft: Arc<dyn Fft<T>>,
    reps: usize,
    buffer: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    rounds: Vec<f64>,
}

impl<T: FftNum> Subject<T> {
    fn new(name: String, fft: Arc<dyn Fft<T>>, reps: usize) -> Self {
        let buffer = vec![Complex::zero(); fft.len() * reps];
        let scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
        Subject {
            name,
            fft,
            reps,
            buffer,
            scratch,
            rounds: Vec::new(),
        }
    }

    /// Wall-clock nanoseconds for `iters` passes over the whole buffer.
    fn time_block(&mut self, iters: usize) -> f64 {
        let start = Instant::now();
        for _ in 0..iters {
            self.fft
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
        }
        start.elapsed().as_secs_f64() * 1e9
    }

    /// Nanoseconds per individual FFT, ie per chunk of `fft.len()`.
    fn time_per_fft(&mut self, iters: usize) -> f64 {
        let reps = self.reps;
        self.time_block(iters) / (iters * reps) as f64
    }

    fn best(&self) -> f64 {
        self.rounds.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn median(&self) -> f64 {
        let mut values = self.rounds.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[values.len() / 2]
    }
}

/// Time a set of subjects against each other, round-robin.
///
/// Round-robin rather than one at a time so any drift over the run hits every subject equally,
/// and min-of-rounds rather than a mean because the quantity of interest is the cost with
/// nothing else interfering.
fn measure<T: FftNum>(subjects: &mut [Subject<T>], rounds: usize, block_ms: f64) {
    let iters: Vec<usize> = subjects
        .iter_mut()
        .map(|subject| {
            let probe = subject.time_block(1);
            (((block_ms * 1e6) / probe.max(1.0)).ceil() as usize).clamp(1, 20_000_000)
        })
        .collect();

    for (subject, &n) in subjects.iter_mut().zip(iters.iter()) {
        subject.time_block(1 + n / 4);
    }

    for _ in 0..rounds {
        for (subject, &n) in subjects.iter_mut().zip(iters.iter()) {
            let per_fft = subject.time_per_fft(n);
            subject.rounds.push(per_fft);
        }
    }
}

/// Keep this thread on a performance core. Without it the macOS scheduler is free to park a
/// long-running thread on an efficiency core, which shows up as bimodal timings.
#[cfg(target_os = "macos")]
fn request_performance_core() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn request_performance_core() {}

fn median_of(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Measure every primitive the model needs: each butterfly, and each valid Radix4 shape.
///
/// Each is timed over a buffer of at least 8192 elements, because primitives are almost always
/// invoked many times over a larger buffer rather than standalone, and a single cold call would
/// price in a startup cost they do not really pay in use.
fn calibrate_primitives<T: FftNum, P: TunablePlanner<T>>(
    rounds: usize,
    block_ms: f64,
    max_len: usize,
) -> Model {
    const REFERENCE_ELEMENTS: usize = 8192;
    let mut planner = P::new();
    let mut model = Model::default();

    let mut shapes: Vec<Option<(usize, u32)>> = Vec::new();
    let mut subjects: Vec<Subject<T>> = Vec::new();

    for len in P::butterfly_lens() {
        let spec = Spec::Butterfly(len);
        let fft = planner.build(&spec, FftDirection::Forward);
        let reps = (REFERENCE_ELEMENTS / len).max(1);
        shapes.push(None);
        subjects.push(Subject::new(format!("b{}", len), fft, reps));
    }

    for base in P::radix4_bases() {
        let mut k = 1u32;
        while base * (1usize << (2 * k)) <= max_len {
            let spec = Spec::Radix4 {
                k,
                base: Arc::new(Spec::Butterfly(base)),
            };
            let len = spec.len();
            let fft = planner.build(&spec, FftDirection::Forward);
            let reps = (REFERENCE_ELEMENTS / len).max(1);
            shapes.push(Some((base, k)));
            subjects.push(Subject::new(to_spec_string(&spec), fft, reps));
            k += 1;
        }
    }

    measure(&mut subjects, rounds, block_ms);

    for (shape, subject) in shapes.iter().zip(subjects.iter()) {
        match shape {
            Some(key) => {
                model.radix4.insert(*key, subject.best());
            }
            None => {
                model.butterfly.insert(subject.fft.len(), subject.best());
            }
        }
    }
    model
}

/// Fit one overhead curve for each composing algorithm.
///
/// Kinds are fitted in dependency order, since the residual of a MixedRadix that contains a
/// MixedRadixSmall only means anything once the Small's own overhead is known.
fn fit_overheads<T: FftNum, P: TunablePlanner<T>>(
    model: &mut Model,
    lengths: &[usize],
    rounds: usize,
    block_ms: f64,
    cap: usize,
    bucketed: bool,
) -> Vec<(Arc<Spec>, f64)> {
    let mut samples: Vec<(Arc<Spec>, f64)> = Vec::new();
    for &len in lengths {
        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, cap);
        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, rounds, block_ms);
        for (spec, subject) in specs.iter().zip(subjects.iter()) {
            samples.push((Arc::clone(spec), subject.best()));
        }
    }

    let mut fitted: Vec<&'static str> = Vec::new();
    for target in FIT_ORDER {
        let usable: Vec<(u32, f64)> = samples
            .iter()
            .filter(|(spec, _)| spec.kind() == target && model.descendants_known(spec, &fitted))
            .filter_map(|(spec, measured)| {
                model
                    .inner_cost(spec)
                    .map(|inner| (bucket_of(spec), (measured - inner) / overhead_scale(spec)))
            })
            .collect();
        if usable.is_empty() {
            eprintln!("warning: no calibration samples for '{}'", target);
            continue;
        }

        // One median per log2 bucket, but only for buckets with enough samples to mean anything.
        // Sparse buckets are dropped and filled in by interpolation instead.
        let mut table: Vec<(u32, f64)> = Vec::new();
        if bucketed {
            let mut by_bucket: std::collections::BTreeMap<u32, Vec<f64>> = Default::default();
            for (bucket, residual) in usable.iter() {
                by_bucket.entry(*bucket).or_default().push(*residual);
            }
            const MIN_PER_BUCKET: usize = 3;
            table = by_bucket
                .iter()
                .filter(|(_, values)| values.len() >= MIN_PER_BUCKET)
                .map(|(bucket, values)| (*bucket, median_of(values.clone())))
                .collect();
        }

        // Too little data to describe a curve, or curves not wanted, so use one constant.
        if table.len() < 2 {
            table = vec![(0, median_of(usable.iter().map(|(_, r)| *r).collect()))];
        }

        let rendered: Vec<String> = table
            .iter()
            .map(|(bucket, value)| format!("{}:{:.2}", 1usize << bucket, value))
            .collect();
        println!(
            "  {:<5} {:>3} buckets from {:>4} samples   {}",
            target,
            table.len(),
            usable.len(),
            rendered.join(" ")
        );
        model.overhead.insert(target, table);
        fitted.push(target);
    }
    samples
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

struct Options {
    rounds: usize,
    block_ms: f64,
    cap: usize,
    verbose: bool,
    bucketed: bool,
}

fn cmd_time<T: FftNum, P: TunablePlanner<T>>(specs: &[String], opts: &Options) {
    let mut planner = P::new();
    let mut subjects: Vec<Subject<T>> = specs
        .iter()
        .map(|text| {
            let (spec_text, reps) = match text.rsplit_once('*') {
                Some((spec, reps)) => (spec, reps.parse().expect("bad repeat count")),
                None => (text.as_str(), 1usize),
            };
            let spec = parse(spec_text).unwrap_or_else(|e| {
                eprintln!("error in spec '{}': {}", spec_text, e);
                std::process::exit(2);
            });
            let fft = planner.build(&spec, FftDirection::Forward);
            Subject::new(text.clone(), fft, reps)
        })
        .collect();

    measure(&mut subjects, opts.rounds, opts.block_ms);

    println!(
        "{:<46} {:>9} {:>7} {:>13} {:>9}",
        "recipe", "len", "reps", "min ns", "spread"
    );
    for subject in subjects.iter() {
        let (min, med) = (subject.best(), subject.median());
        println!(
            "{:<46} {:>9} {:>7} {:>13.1} {:>8.2}%",
            subject.name,
            subject.fft.len(),
            subject.reps,
            min,
            (med - min) / min * 100.0
        );
    }

    if subjects.len() > 1 {
        let best = subjects
            .iter()
            .map(|s| s.best())
            .fold(f64::INFINITY, f64::min);
        println!("\nrelative to best:");
        for subject in subjects.iter() {
            println!("  {:<46} {:>6.3}x", subject.name, subject.best() / best);
        }
    }
}

fn cmd_regret<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    println!(
        "{:>9} {:>8} {:>10} {:>10}  {}",
        "len", "regret", "planner ns", "best ns", "best recipe (when it differs)"
    );

    let mut regrets: Vec<(f64, usize, String, String)> = Vec::new();

    for &len in lengths {
        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, opts.cap);
        let planner_spec = to_spec_string(&specs[0]);

        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();

        measure(&mut subjects, opts.rounds, opts.block_ms);

        let planner_time = subjects[0].best();
        let best_index = subjects
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.best().partial_cmp(&b.1.best()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let best_time = subjects[best_index].best();
        let best_spec = subjects[best_index].name.clone();
        let regret = planner_time / best_time;

        println!(
            "{:>9} {:>7.3}x {:>10.0} {:>10.0}  {}",
            len,
            regret,
            planner_time,
            best_time,
            if best_index == 0 {
                "= planner".to_string()
            } else {
                best_spec.clone()
            }
        );
        if opts.verbose {
            let mut ranked: Vec<&Subject<T>> = subjects.iter().collect();
            ranked.sort_by(|a, b| a.best().partial_cmp(&b.best()).unwrap());
            for subject in ranked.iter().take(6) {
                println!(
                    "            {:>6.3}x  {}",
                    subject.best() / best_time,
                    subject.name
                );
            }
            println!("            ({} candidates measured)", subjects.len());
        }

        regrets.push((regret, len, planner_spec, best_spec));
    }

    regrets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = regrets.len();
    let mean = regrets.iter().map(|r| r.0).sum::<f64>() / n as f64;
    println!("\n--- regret: planner's pick divided by the best recipe measured ---");
    println!("  lengths  {}", n);
    println!("  mean     {:.4}", mean);
    println!("  median   {:.4}", regrets[n / 2].0);
    println!("  p90      {:.4}", regrets[(n * 9) / 10].0);
    println!("  worst    {:.4}", regrets[n - 1].0);
    let losing = regrets.iter().filter(|r| r.0 > 1.02).count();
    println!("  more than 2% off the best: {} of {} lengths", losing, n);
    println!("\nworst offenders:");
    for (regret, len, planner_spec, best_spec) in regrets.iter().rev().take(10) {
        println!("  {:>8}  {:.3}x", len, regret);
        println!("      planner: {}", planner_spec);
        println!("      best:    {}", best_spec);
    }
}

/// Check that every enumerated candidate actually computes a correct FFT.
///
/// Timing a recipe says nothing about whether it is valid, and an invalid one would happily
/// produce fast wrong answers. Each candidate is compared against a direct DFT.
fn cmd_verify<T: FftNum + ToPrimitive, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let mut worst_overall: f64 = 0.0;
    let mut failures = 0usize;

    for &len in lengths {
        let input: Vec<Complex<T>> = (0..len)
            .map(|i| {
                let x = ((i * 2654435761usize) % 1000) as f64 / 500.0 - 1.0;
                let y = ((i * 40503usize) % 1000) as f64 / 500.0 - 1.0;
                Complex::new(T::from_f64(x).unwrap(), T::from_f64(y).unwrap())
            })
            .collect();

        // Reference: a direct DFT, which shares no code with the recipes under test.
        let reference_fft = rustfft::algorithm::Dft::<T>::new(len, FftDirection::Forward);
        let mut reference = input.clone();
        let mut reference_scratch = vec![Complex::zero(); reference_fft.get_inplace_scratch_len()];
        reference_fft.process_with_scratch(&mut reference, &mut reference_scratch);
        let reference_norm: f64 = reference
            .iter()
            .map(|c| c.norm_sqr().to_f64().unwrap())
            .sum::<f64>()
            .sqrt();

        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, opts.cap);
        let mut worst_here: f64 = 0.0;
        let mut worst_spec = String::new();

        for spec in specs.iter() {
            let fft = planner.build(spec, FftDirection::Forward);
            let mut buffer = input.clone();
            let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
            fft.process_with_scratch(&mut buffer, &mut scratch);

            let error: f64 = buffer
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).norm_sqr().to_f64().unwrap())
                .sum::<f64>()
                .sqrt()
                / reference_norm;
            if error > worst_here {
                worst_here = error;
                worst_spec = to_spec_string(spec);
            }
        }

        let bad = worst_here > 1e-6;
        if bad {
            failures += 1;
        }
        println!(
            "{:>8}  {} candidates, worst relative error {:.3e}  {}{}",
            len,
            specs.len(),
            worst_here,
            if bad { "FAIL " } else { "" },
            worst_spec
        );
        worst_overall = worst_overall.max(worst_here);
    }

    println!(
        "\nworst relative error over all lengths: {:.3e}  ({} lengths failed)",
        worst_overall, failures
    );
    if failures > 0 {
        std::process::exit(1);
    }
}

/// Show how each algorithm's per-element overhead varies with size.
fn cmd_residuals<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options) {
    let max_len = lengths.iter().copied().max().unwrap_or(1024) * 4;
    eprintln!("measuring primitives...");
    let mut model = calibrate_primitives::<T, P>(opts.rounds, opts.block_ms, max_len);
    eprintln!("fitting overheads...");
    let samples = fit_overheads::<T, P>(
        &mut model,
        lengths,
        opts.rounds,
        opts.block_ms,
        opts.cap,
        opts.bucketed,
    );

    let mut binned: std::collections::BTreeMap<
        &'static str,
        std::collections::BTreeMap<u32, Vec<f64>>,
    > = Default::default();
    for (spec, measured) in samples.iter() {
        if !FIT_ORDER.contains(&spec.kind()) {
            continue;
        }
        if let Some(inner) = model.inner_cost(spec) {
            let residual = (measured - inner) / overhead_scale(spec);
            binned
                .entry(spec.kind())
                .or_default()
                .entry(bucket_of(spec))
                .or_default()
                .push(residual);
        }
    }

    println!(
        "{:<5} {:>8} {:>8} {:>9} {:>7}",
        "kind", "len>=", "median", "p25..p75", "n"
    );
    for (kind, buckets) in binned {
        for (bucket, values) in buckets {
            if values.len() < 3 {
                continue;
            }
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            println!(
                "{:<5} {:>8} {:>8.3} {:>9} {:>7}",
                kind,
                1usize << bucket,
                sorted[n / 2],
                format!("{:.2}..{:.2}", sorted[n / 4], sorted[(n * 3) / 4]),
                n
            );
        }
        println!();
    }
}

fn cmd_model<T: FftNum, P: TunablePlanner<T>>(train: &[usize], test: &[usize], opts: &Options) {
    let max_len = test.iter().chain(train.iter()).copied().max().unwrap_or(1024) * 4;

    println!("planner: {}", P::label());
    println!("measuring primitives...");
    let mut model = calibrate_primitives::<T, P>(opts.rounds, opts.block_ms, max_len);

    println!("fitting overheads on {} training lengths...", train.len());
    fit_overheads::<T, P>(
        &mut model,
        train,
        opts.rounds,
        opts.block_ms,
        opts.cap,
        opts.bucketed,
    );
    println!("\n{}", model.describe());

    println!(
        "{:>9} {:>9} {:>9}   {}",
        "len", "model", "planner", "model's pick (when it is not the best)"
    );
    let mut model_regrets = Vec::new();
    let mut planner_regrets = Vec::new();

    for &len in test {
        let mut planner = P::new();
        let specs = candidates_capped(&mut planner, len, opts.cap);
        let mut subjects: Vec<Subject<T>> = specs
            .iter()
            .map(|spec| {
                let fft = planner.build(spec, FftDirection::Forward);
                Subject::new(to_spec_string(spec), fft, 1)
            })
            .collect();
        measure(&mut subjects, opts.rounds, opts.block_ms);

        // Fastest of the first pass, used only to pick which recipes deserve a careful re-timing.
        let best_index = subjects
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.best().partial_cmp(&b.1.best()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let model_index = specs
            .iter()
            .enumerate()
            .filter_map(|(i, spec)| model.cost(spec).map(|c| (i, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i);

        let model_spec = match model_index {
            Some(i) => subjects[i].name.clone(),
            None => "<no candidate priced>".to_string(),
        };

        // Second pass. The winner of a wide comparison is biased fast: with sub-percent noise,
        // the minimum of many draws lands below the true minimum, which puts a floor under any
        // regret measured against it. Re-timing just the recipes of interest, for longer,
        // removes most of that bias.
        let finalists: Vec<usize> = {
            let mut picked = vec![0usize, best_index];
            if let Some(i) = model_index {
                picked.push(i);
            }
            picked.sort_unstable();
            picked.dedup();
            picked
        };
        let mut finals: Vec<Subject<T>> = finalists
            .iter()
            .map(|&i| {
                let fft = planner.build(&specs[i], FftDirection::Forward);
                Subject::new(to_spec_string(&specs[i]), fft, 1)
            })
            .collect();
        measure(&mut finals, opts.rounds * 4, opts.block_ms);

        let time_of =
            |index: usize| -> f64 { finals[finalists.iter().position(|&i| i == index).unwrap()].best() };
        let planner_time = time_of(0);
        let best_time = finalists
            .iter()
            .map(|&i| time_of(i))
            .fold(f64::INFINITY, f64::min);
        let model_time = match model_index {
            Some(i) => time_of(i),
            None => f64::NAN,
        };

        let model_regret = model_time / best_time;
        let planner_regret = planner_time / best_time;
        model_regrets.push(model_regret);
        planner_regrets.push(planner_regret);

        println!(
            "{:>9} {:>8.3}x {:>8.3}x   {}",
            len,
            model_regret,
            planner_regret,
            if model_regret <= 1.001 {
                "= best".to_string()
            } else {
                model_spec
            }
        );
    }

    for (label, mut values) in [("model  ", model_regrets), ("planner", planner_regrets)] {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        println!(
            "{}: mean {:.4}  median {:.4}  p90 {:.4}  worst {:.4}",
            label,
            mean,
            values[n / 2],
            values[(n * 9) / 10],
            values[n - 1]
        );
    }
}

fn cmd_emit<T: FftNum, P: TunablePlanner<T>>(lengths: &[usize], opts: &Options, element: &str) {
    let max_len = lengths.iter().copied().max().unwrap_or(1024) * 4;
    eprintln!("measuring primitives...");
    let mut model = calibrate_primitives::<T, P>(opts.rounds, opts.block_ms, max_len);
    eprintln!("fitting overheads on {} lengths...", lengths.len());
    fit_overheads::<T, P>(
        &mut model,
        lengths,
        opts.rounds,
        opts.block_ms,
        opts.cap,
        opts.bucketed,
    );
    print!("{}", emit::emit(&model, element, P::label()));
}

// ---------------------------------------------------------------------------

enum Command {
    Time(Vec<String>),
    Regret(Vec<usize>),
    Model(Vec<usize>, Vec<usize>),
    Residuals(Vec<usize>),
    Verify(Vec<usize>),
    Emit(Vec<usize>),
}

fn run<T: FftNum + ToPrimitive, P: TunablePlanner<T>>(
    command: &Command,
    opts: &Options,
    element: &str,
) {
    match command {
        Command::Time(specs) => cmd_time::<T, P>(specs, opts),
        Command::Regret(lengths) => cmd_regret::<T, P>(lengths, opts),
        Command::Model(train, test) => cmd_model::<T, P>(train, test, opts),
        Command::Residuals(lengths) => cmd_residuals::<T, P>(lengths, opts),
        Command::Verify(lengths) => cmd_verify::<T, P>(lengths, opts),
        Command::Emit(lengths) => cmd_emit::<T, P>(lengths, opts, element),
    }
}

fn dispatch<T: FftNum + ToPrimitive>(planner: &str, command: &Command, opts: &Options, el: &str) {
    match planner {
        "scalar" => run::<T, ScalarTuner<T>>(command, opts, el),
        // The manifest gives rustfft the SIMD feature matching the target, so architecture
        // alone decides which of these exists. A tool-crate `feature = ...` cfg would refer to
        // the tool's own features and always be false.
        #[cfg(target_arch = "aarch64")]
        "neon" => run::<T, rustfft::tuning::NeonTuner<T>>(command, opts, el),
        #[cfg(target_arch = "x86_64")]
        "sse" => run::<T, rustfft::tuning::SseTuner<T>>(command, opts, el),
        #[cfg(target_arch = "wasm32")]
        "wasm_simd" => run::<T, rustfft::tuning::WasmSimdTuner<T>>(command, opts, el),
        other => {
            eprintln!(
                "unknown or unavailable planner '{}' on this build; try 'scalar'",
                other
            );
            std::process::exit(2);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: planner_tuning <command> [options] ARGS...");
        eprintln!("commands: time SPEC... | regret LEN... | model TRAIN... 0 TEST...");
        eprintln!("          residuals LEN... | verify LEN... | emit LEN...");
        eprintln!("  --planner NAME  scalar (default), neon, sse");
        eprintln!("  --rounds N      timing rounds per subject (default 9)");
        eprintln!("  --block-ms MS   wall-clock time per timed block (default 10)");
        eprintln!("  --cap N         max candidates per length (default 48)");
        eprintln!("  --f32           measure f32 instead of f64");
        eprintln!("  --bucketed      fit overhead curves over working set, not constants");
        eprintln!("  --verbose       for 'regret', list the top candidates per length");
        std::process::exit(2);
    }

    let command_name = args[0].clone();
    let mut planner = "scalar".to_string();
    let mut opts = Options {
        rounds: 9,
        block_ms: 10.0,
        cap: 48,
        verbose: false,
        bucketed: false,
    };
    let mut f32_mode = false;
    let mut rest: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--planner" => {
                i += 1;
                planner = args[i].clone();
            }
            "--rounds" => {
                i += 1;
                opts.rounds = args[i].parse().expect("--rounds wants a number");
            }
            "--block-ms" => {
                i += 1;
                opts.block_ms = args[i].parse().expect("--block-ms wants a number");
            }
            "--cap" => {
                i += 1;
                opts.cap = args[i].parse().expect("--cap wants a number");
            }
            "--f32" => f32_mode = true,
            "--bucketed" => opts.bucketed = true,
            "--verbose" => opts.verbose = true,
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    let numbers = |values: &[String]| -> Vec<usize> {
        values
            .iter()
            .map(|s| s.parse().expect("lengths must be numbers"))
            .collect()
    };

    let command = match command_name.as_str() {
        "time" => Command::Time(rest.clone()),
        "regret" => Command::Regret(numbers(&rest)),
        "residuals" => Command::Residuals(numbers(&rest)),
        "verify" => Command::Verify(numbers(&rest)),
        "emit" => Command::Emit(numbers(&rest)),
        "model" => {
            let lengths = numbers(&rest);
            let split = lengths
                .iter()
                .position(|&l| l == 0)
                .expect("model wants TRAIN... 0 TEST...");
            let (train, test) = lengths.split_at(split);
            Command::Model(train.to_vec(), test[1..].to_vec())
        }
        other => {
            eprintln!("unknown command '{}'", other);
            std::process::exit(2);
        }
    };

    request_performance_core();

    if f32_mode {
        dispatch::<f32>(&planner, &command, &opts, "f32");
    } else {
        dispatch::<f64>(&planner, &command, &opts, "f64");
    }
}
