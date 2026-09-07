//! Measurement tool for the Neon planner.
//!
//! Two subcommands:
//!
//!   time SPEC...      time recipes against each other, eg 'r4(2,b16)' 'mr(b32,b32)'
//!                     a spec may carry a '*reps' suffix to run it over a len*reps buffer
//!   regret LEN...     for each length, enumerate candidate recipes, measure them all, and
//!                     report how far the shipping planner's pick is from the best available
//!
//! Recipes are built through rustfft's own planner internals, so what is measured here is
//! exactly what the planner would construct.

mod emit;
mod model;

use model::{bucket_of, kind, overhead_scale, Model};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::tuning::{
    butterfly_lens, parse, radix4_shapes, to_spec, NeonTuner, Recipe,
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

    fn placeholder() -> Self {
        Subject {
            name: String::new(),
            fft: std::sync::Arc::new(rustfft::algorithm::Dft::new(1, FftDirection::Forward)),
            reps: 1,
            buffer: Vec::new(),
            scratch: Vec::new(),
            rounds: vec![f64::INFINITY],
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
/// Round-robin rather than one-at-a-time so that any drift over the run hits every subject
/// equally, and min-of-rounds rather than a mean because the quantity of interest is the cost
/// with nothing else interfering.
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
            let total = subject.time_block(n);
            let per_fft = total / (n * subject.reps) as f64;
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

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_time<T: FftNum>(specs: &[String], rounds: usize, block_ms: f64) {
    let mut tuner = NeonTuner::<T>::new();
    let mut subjects: Vec<Subject<T>> = specs
        .iter()
        .map(|spec| {
            let (recipe_spec, reps) = match spec.rsplit_once('*') {
                Some((recipe, reps)) => (recipe, reps.parse().expect("bad repeat count")),
                None => (spec.as_str(), 1usize),
            };
            let recipe = parse(recipe_spec).unwrap_or_else(|e| {
                eprintln!("error in spec '{}': {}", recipe_spec, e);
                std::process::exit(2);
            });
            let fft = tuner.build(&recipe, FftDirection::Forward);
            Subject::new(spec.clone(), fft, reps)
        })
        .collect();

    measure(&mut subjects, rounds, block_ms);

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

fn cmd_regret<T: FftNum>(
    lengths: &[usize],
    rounds: usize,
    block_ms: f64,
    verbose: bool,
    cap: usize,
) {
    println!(
        "{:>9} {:>8} {:>10} {:>10}  {}",
        "len", "regret", "planner ns", "best ns", "best recipe (when it differs)"
    );

    let mut regrets: Vec<(f64, usize, String, String)> = Vec::new();

    for &len in lengths {
        let mut tuner = NeonTuner::<T>::new();
        let candidates = tuner.candidates_capped(len, cap);
        let planner_spec = to_spec(&candidates[0]);

        let mut subjects: Vec<Subject<T>> = candidates
            .iter()
            .map(|recipe| {
                let fft = tuner.build(recipe, FftDirection::Forward);
                Subject::new(to_spec(recipe), fft, 1)
            })
            .collect();

        measure(&mut subjects, rounds, block_ms);

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
        if verbose {
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
    println!(
        "  more than 2% off the best: {} of {} lengths",
        losing, n
    );
    println!("\nworst offenders:");
    for (regret, len, planner_spec, best_spec) in regrets.iter().rev().take(10) {
        println!("  {:>8}  {:.3}x", len, regret);
        println!("      planner: {}", planner_spec);
        println!("      best:    {}", best_spec);
    }
}


fn median_of(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// Measure every primitive the model needs: each butterfly, and each valid Radix4 shape.
///
/// Each is timed over a buffer of at least 8192 elements, because primitives are almost always
/// invoked many times over a larger buffer rather than standalone, and a single cold call would
/// price in a startup cost they do not really pay in use.
fn calibrate_primitives<T: FftNum>(rounds: usize, block_ms: f64, max_len: usize) -> Model {
    const REFERENCE_ELEMENTS: usize = 8192;
    let mut tuner = NeonTuner::<T>::new();
    let mut model = Model::default();

    let mut subjects: Vec<(String, usize, Option<(usize, u32)>, Subject<T>)> = Vec::new();
    for len in butterfly_lens() {
        let recipe = parse(&format!("b{}", len)).expect("butterfly spec");
        let fft = tuner.build(&recipe, FftDirection::Forward);
        let reps = (REFERENCE_ELEMENTS / len).max(1);
        subjects.push((
            format!("b{}", len),
            len,
            None,
            Subject::new(format!("b{}", len), fft, reps),
        ));
    }
    for (base, k) in radix4_shapes::<T>(max_len) {
        let spec = format!("r4({},b{})", k, base);
        let recipe = parse(&spec).expect("radix4 spec");
        let len = recipe.len();
        let fft = tuner.build(&recipe, FftDirection::Forward);
        let reps = (REFERENCE_ELEMENTS / len).max(1);
        subjects.push((spec.clone(), len, Some((base, k)), Subject::new(spec, fft, reps)));
    }

    let mut just_subjects: Vec<Subject<T>> = subjects.iter_mut().map(|s| std::mem::replace(&mut s.3, Subject::placeholder())).collect();
    measure(&mut just_subjects, rounds, block_ms);

    for ((_, len, shape, _), subject) in subjects.iter().zip(just_subjects.iter()) {
        match shape {
            Some(key) => {
                model.radix4.insert(*key, subject.best());
            }
            None => {
                model.butterfly.insert(*len, subject.best());
            }
        }
    }
    model
}

/// Fit one overhead-per-element number for each composing algorithm.
///
/// Kinds are fitted in dependency order, since the residual of a MixedRadix that contains a
/// MixedRadixSmall only means anything once the Small's own overhead is known.
fn fit_overheads<T: FftNum>(
    model: &mut Model,
    lengths: &[usize],
    rounds: usize,
    block_ms: f64,
    cap: usize,
) -> Vec<(std::sync::Arc<Recipe>, f64)> {
    let mut samples: Vec<(std::sync::Arc<Recipe>, f64)> = Vec::new();
    for &len in lengths {
        let mut tuner = NeonTuner::<T>::new();
        let candidates = tuner.candidates_capped(len, cap);
        let mut subjects: Vec<Subject<T>> = candidates
            .iter()
            .map(|recipe| {
                let fft = tuner.build(recipe, FftDirection::Forward);
                Subject::new(to_spec(recipe), fft, 1)
            })
            .collect();
        measure(&mut subjects, rounds, block_ms);
        for (recipe, subject) in candidates.iter().zip(subjects.iter()) {
            samples.push((std::sync::Arc::clone(recipe), subject.best()));
        }
    }

    let mut fitted: Vec<&'static str> = Vec::new();
    for target in ["mrs", "gts", "mr", "gt", "rad", "bs"] {
        let usable: Vec<(u32, f64)> = samples
            .iter()
            .filter(|(recipe, _)| kind(recipe) == target && model.descendants_known(recipe, &fitted))
            .filter_map(|(recipe, measured)| {
                model
                    .inner_cost(recipe)
                    .map(|inner| (bucket_of(recipe), (measured - inner) / overhead_scale(recipe)))
            })
            .collect();
        if usable.is_empty() {
            eprintln!("warning: no calibration samples for '{}'", target);
            continue;
        }

        // One median per log2 bucket, but only for buckets with enough samples to mean
        // anything. Sparse buckets are dropped and filled in by interpolation instead.
        let mut by_bucket: std::collections::BTreeMap<u32, Vec<f64>> = Default::default();
        for (bucket, residual) in usable.iter() {
            by_bucket.entry(*bucket).or_default().push(*residual);
        }
        const MIN_PER_BUCKET: usize = 3;
        let mut table: Vec<(u32, f64)> = by_bucket
            .iter()
            .filter(|(_, values)| values.len() >= MIN_PER_BUCKET)
            .map(|(bucket, values)| (*bucket, median_of(values.clone())))
            .collect();

        // Too little data to describe a curve, so fall back to one constant for this kind.
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


/// Check that every enumerated candidate actually computes a correct FFT.
///
/// Timing a recipe says nothing about whether it is valid, and an invalid one (a GoodThomas on
/// non-coprime sides, say) would happily produce fast wrong answers. Each candidate is compared
/// against a direct DFT of the same input.
fn cmd_verify<T: FftNum + rustfft::num_traits::ToPrimitive>(lengths: &[usize], cap: usize) {
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
        let mut reference_scratch =
            vec![Complex::zero(); reference_fft.get_inplace_scratch_len()];
        reference_fft.process_with_scratch(&mut reference, &mut reference_scratch);
        let reference_norm: f64 = reference
            .iter()
            .map(|c| c.norm_sqr().to_f64().unwrap())
            .sum::<f64>()
            .sqrt();

        let mut tuner = NeonTuner::<T>::new();
        let candidates = tuner.candidates_capped(len, cap);
        let mut worst_here: f64 = 0.0;
        let mut worst_spec = String::new();

        for recipe in candidates.iter() {
            let fft = tuner.build(recipe, FftDirection::Forward);
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
                worst_spec = to_spec(recipe);
            }
        }

        let bad = worst_here > 1e-6;
        if bad {
            failures += 1;
        }
        println!(
            "{:>8}  {} candidates, worst relative error {:.3e}  {}{}",
            len,
            candidates.len(),
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
///
/// The model charges one constant per element per algorithm. That is only right if the constant
/// really is constant. GoodThomas and Rader's both reindex with a scatter rather than a
/// transpose, so their per-element cost has every reason to grow once the working set stops
/// fitting in cache, and if it does, a single number cannot express it.
fn cmd_residuals<T: FftNum>(lengths: &[usize], rounds: usize, block_ms: f64, cap: usize) {
    let max_len = lengths.iter().copied().max().unwrap_or(1024) * 4;
    eprintln!("measuring primitives...");
    let mut model = calibrate_primitives::<T>(rounds, block_ms, max_len);
    eprintln!("fitting overheads...");
    let samples = fit_overheads::<T>(&mut model, lengths, rounds, block_ms, cap);

    // kind -> log2 bucket -> residuals
    let mut binned: std::collections::BTreeMap<&'static str, std::collections::BTreeMap<u32, Vec<f64>>> =
        Default::default();
    for (recipe, measured) in samples.iter() {
        let k = kind(recipe);
        if !matches!(k, "mr" | "mrs" | "gt" | "gts" | "rad" | "bs") {
            continue;
        }
        if let Some(inner) = model.inner_cost(recipe) {
            let residual = (measured - inner) / overhead_scale(recipe);
            let bucket = (recipe.len() as f64).log2() as u32;
            binned.entry(k).or_default().entry(bucket).or_default().push(residual);
        }
    }

    println!("{:<5} {:>8} {:>8} {:>9} {:>7}", "kind", "len>=", "median", "p25..p75", "n");
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

fn cmd_model<T: FftNum>(
    train: &[usize],
    test: &[usize],
    rounds: usize,
    block_ms: f64,
    cap: usize,
) {
    let max_len = test.iter().chain(train.iter()).copied().max().unwrap_or(1024) * 4;

    println!("measuring primitives...");
    let mut model = calibrate_primitives::<T>(rounds, block_ms, max_len);

    println!("fitting overheads on {} training lengths...", train.len());
    fit_overheads::<T>(&mut model, train, rounds, block_ms, cap);
    println!("\n{}", model.describe());

    println!(
        "{:>9} {:>9} {:>9}   {}",
        "len", "model", "planner", "model's pick (when it is not the best)"
    );
    let mut model_regrets = Vec::new();
    let mut planner_regrets = Vec::new();

    for &len in test {
        let mut tuner = NeonTuner::<T>::new();
        let candidates = tuner.candidates_capped(len, cap);
        let mut subjects: Vec<Subject<T>> = candidates
            .iter()
            .map(|recipe| {
                let fft = tuner.build(recipe, FftDirection::Forward);
                Subject::new(to_spec(recipe), fft, 1)
            })
            .collect();
        measure(&mut subjects, rounds, block_ms);

        let planner_time_pass1 = subjects[0].best();
        let _ = planner_time_pass1;

        // Fastest of the first pass, used only to pick which recipes deserve a careful re-timing.
        let best_index = subjects
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.best().partial_cmp(&b.1.best()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let model_index = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, recipe)| model.cost(recipe).map(|c| (i, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i);

        let model_spec = match model_index {
            Some(i) => subjects[i].name.clone(),
            None => "<no candidate priced>".to_string(),
        };

        // Second pass. The winner of a 48-way comparison is biased fast: with sub-percent
        // noise, the minimum of many draws lands below the true minimum, which puts a floor
        // under any regret measured against it. Re-timing just the three recipes of interest,
        // for longer, removes most of that bias and shows whether the model is actually behind
        // the best or merely behind the luckiest measurement.
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
                let fft = tuner.build(&candidates[i], FftDirection::Forward);
                Subject::new(to_spec(&candidates[i]), fft, 1)
            })
            .collect();
        measure(&mut finals, rounds * 4, block_ms);

        let time_of = |index: usize| -> f64 {
            finals[finalists.iter().position(|&i| i == index).unwrap()].best()
        };
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
            if model_regret <= 1.001 { "= best".to_string() } else { model_spec }
        );
    }

    for (label, mut values) in [
        ("model  ", model_regrets),
        ("planner", planner_regrets),
    ] {
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

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: planner_tuning <time|regret> [options] ARGS...");
        eprintln!("  --rounds N     timing rounds per subject (default 9)");
        eprintln!("  --block-ms MS  wall-clock time per timed block (default 10)");
        eprintln!("  --f32          measure f32 instead of f64");
        eprintln!("  --verbose      for 'regret', list the top candidates per length");
        eprintln!("  --cap N        max candidates per length (default 48)");
        eprintln!("commands: time SPEC... | regret LEN... | model TRAIN... 0 TEST... | verify LEN... | emit LEN...");
        std::process::exit(2);
    }

    let command = args[0].clone();
    let mut rounds = 9usize;
    let mut block_ms = 10.0f64;
    let mut f32_mode = false;
    let mut verbose = false;
    let mut cap = 48usize;
    let mut rest: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--rounds" => {
                i += 1;
                rounds = args[i].parse().expect("--rounds wants a number");
            }
            "--block-ms" => {
                i += 1;
                block_ms = args[i].parse().expect("--block-ms wants a number");
            }
            "--cap" => {
                i += 1;
                cap = args[i].parse().expect("--cap wants a number");
            }
            "--f32" => f32_mode = true,
            "--verbose" => verbose = true,
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    request_performance_core();

    match command.as_str() {
        "time" => {
            if f32_mode {
                cmd_time::<f32>(&rest, rounds, block_ms)
            } else {
                cmd_time::<f64>(&rest, rounds, block_ms)
            }
        }
        "regret" => {
            let lengths: Vec<usize> = rest
                .iter()
                .map(|s| s.parse().expect("lengths must be numbers"))
                .collect();
            if f32_mode {
                cmd_regret::<f32>(&lengths, rounds, block_ms, verbose, cap)
            } else {
                cmd_regret::<f64>(&lengths, rounds, block_ms, verbose, cap)
            }
        }
        "residuals" => {
            let lengths: Vec<usize> = rest
                .iter()
                .map(|s| s.parse().expect("lengths must be numbers"))
                .collect();
            if f32_mode {
                cmd_residuals::<f32>(&lengths, rounds, block_ms, cap)
            } else {
                cmd_residuals::<f64>(&lengths, rounds, block_ms, cap)
            }
        }
        "emit" => {
            let lengths: Vec<usize> = rest
                .iter()
                .map(|s| s.parse().expect("lengths must be numbers"))
                .collect();
            let max_len = lengths.iter().copied().max().unwrap_or(1024) * 4;
            eprintln!("measuring primitives...");
            let mut model = calibrate_primitives::<f64>(rounds, block_ms, max_len);
            eprintln!("fitting overheads on {} lengths...", lengths.len());
            fit_overheads::<f64>(&mut model, &lengths, rounds, block_ms, cap);
            print!("{}", emit::emit(&model, "f64"));
        }
        "verify" => {
            let lengths: Vec<usize> = rest
                .iter()
                .map(|s| s.parse().expect("lengths must be numbers"))
                .collect();
            if f32_mode {
                cmd_verify::<f32>(&lengths, cap)
            } else {
                cmd_verify::<f64>(&lengths, cap)
            }
        }
        "model" => {
            let lengths: Vec<usize> = rest
                .iter()
                .map(|s| s.parse().expect("lengths must be numbers"))
                .collect();
            let split = lengths.iter().position(|&l| l == 0).unwrap_or(0);
            let (train, test) = lengths.split_at(split);
            let test = &test[1..];
            if f32_mode {
                cmd_model::<f32>(train, test, rounds, block_ms, cap)
            } else {
                cmd_model::<f64>(train, test, rounds, block_ms, cap)
            }
        }
        other => {
            eprintln!("unknown command '{}'", other);
            std::process::exit(2);
        }
    }
}
