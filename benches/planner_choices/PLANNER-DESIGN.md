# Planner design notes

Compiled 2026-09-09 from a session that split PR #11 and then went down a long planner rabbit hole.
Companion to `NOTES.md`, which holds raw measurement results. This file holds the reasoning, the
inventory of what is actually hand-tuned, and the argument about what to do next.

**Temporary, like NOTES.md. Delete before this branch merges back.**

## Start here

If you are picking this up cold:

1. Read "The constants inventory" below. It is the actionable part. Everything else is
   justification for the ordering.
2. The recommended next piece of work is **the `design_prime` cost comparison**, described under
   "Recommended order of work". It needs no new benchmarking to get a first answer.
3. The thing that blocks PR #179 leaving draft is the **f64 RadixN contradiction**, item 3 in the
   inventory.

## Where things stand

| branch / PR | what | state |
| --- | --- | --- |
| [ejmahler#178](https://github.com/ejmahler/RustFFT/pull/178) | Rader's precomputed permutation | open, 1 commit, 1 file |
| [ejmahler#179](https://github.com/ejmahler/RustFFT/pull/179) | SIMD RadixN, all three backends | **draft**, 5 commits |
| `simd_planner_tuning` | this branch, all width-dependent tuning | no PR |
| `simd_radixn`, `sse_planner_tuning`, `HEnquist/RustFFT#11` | pre-split originals | left untouched deliberately |

`#179` deliberately matches what `src/plan.rs` does: RadixN for both float types, and
`MAX_RADER_PRIME_FACTOR` a plain 23 everywhere. Every width-dependent decision lives here instead.
`#179` says in its own description that a companion planner PR has to land before it can merge.

## The constants inventory

Every decision the planner makes by hand-tuned rule rather than by measurement, with what is
actually known about each. Ordered by how much evidence there is that the rule is wrong.

### 1. `MAX_RADER_PRIME_FACTOR`, currently 23

**Known wrong for f32.** `src/plan.rs:18` and the three SIMD planners. Rader's if the largest prime
factor of `len - 1` is at or below the cutoff, else Bluestein's.

The SSE sweep in `NOTES.md` (primes 100 to 80000, bucketed by largest prime factor of `len - 1`)
says f32 wants **7**, not 23: from bucket 11 upward Bluestein's beats Rader's by 1.16x to 1.40x,
consistently across three length bands. End to end through the planner that is 1.22x to 1.43x on the
buckets it moves. f64 wants **31**, which is where it already sits on this branch.

Two structural problems beyond the value being wrong:

- It is a **step function standing in for a cost comparison**. The planner already builds both
  candidate trees. The constant exists only because nobody priced them.
- It **branches on the wrong quantity**. Largest prime factor of `len - 1` is a proxy for "is the
  Rader's inner FFT well planned". Two primes can share a bucket and have wildly different inner
  cost, `len - 1 = 2^k · 23` versus `23^3 · m`. No value of the constant fixes that.

Trap recorded in `NOTES.md` and worth repeating: an A/B of Rader's against Bluestein's is only
trustworthy **at or below 31**, the largest prime butterfly. Above that, raising the cutoff also
changes how Rader's own inner FFT is planned, so it starts nesting another Rader's. A bucket table
measured with the cutoff at 31 predicted 37/41/43 should be admitted; end to end a cutoff of 43 came
out at 0.86x. Only a two-build end-to-end comparison counts above 31.

### 2. RadixN f64 gate, `complex_per_vector < 2`

**Contradictory evidence. This is what blocks #179.** `src/simd_planner.rs` on this branch.

- SSE: RadixN for f64 measured **0.80x** of the mixed radix fallback over 109 mixed-factor lengths
  from 24 to 241920, and every base the design picks was a loss on its own.
- NEON: RadixN for f64 measured **1.13x to 1.53x** *faster* than the best MixedRadix tree the
  backend could previously build (commit message of the original NEON RadixN work).

Both cannot be right for a single width-only gate. Either it becomes backend aware, or one of the
two measurements is wrong. Resolving this is the highest-value single measurement on this branch.

If the answer really is backend-dependent, that has an architectural consequence: `simd_planner.rs`
stops being a place where all three backends agree, which is the property that justifies its
existence. See `rustfft-simd-planners-stay-separate` in memory.

### 3. RadixN base length and factor order

**Untested.** The `p2/p3/p5/p7` cascade in `simd_planner::design_radixn` and its twin in
`plan.rs::design_radixn`. There is a temporary `RUSTFFT_FORCE_RADIXN_BASE` env hook on this branch
specifically to measure it.

The scalar spike found this is where scalar's residual regret mostly lives: "a RadixN base chosen
one factor off, eg 1260 where it picks `rn(7.6.5,b6)` over `rn(6.6.5,b7)`".

Cheap approach already decided on and recorded in memory (`rustfft-radixn-base-sweep-idea`): an
`#[ignore]`d test in the backend's `*_radixn.rs` that sweeps every legal base for a length, builds
the RadixN for each, times it and prints best-vs-planner. Roughly 80 lines, no feature flag, never
ships. Do **not** port the 1700-line tuning tool for this.

### 4. Bluestein's inner length

**Untested, and the scalar spike found misses.** Next power of two above `2·len - 1`, or three
quarters of it when that still fits. Spike example: at 10007 the planner picks `r4(5,b24)` = 24576
where the best is `r4(6,b5)` = 20480.

### 5. The Small-variant cliff, `left_len < 31` scalar / `< 33` SIMD

**Measured this session. Conservative but not wrong.** See "GoodThomas earns its keep" below. The
advantage decays with size but never crosses 1.0x inside the range the planner can reach, so
nothing says the threshold is too generous.

Two loose ends: the scalar/SIMD inconsistency (31 vs 33, so scalar excludes a 31x31 split and SIMD
includes 32x32) looks like drift rather than an expressed rule; and everything **above** the cliff
is untested, because plain `GoodThomasAlgorithm` is never constructed.

### 6. `gcd(left, right) == 1` chooses GoodThomas

**Measured this session. Essentially always right.** No headroom here. Details below.

### 7. `MIN_RADIX4_BITS = 6`

**Untested.** Smallest size at which Radix4 is considered, `2^6 = 64`, in all four planners.

### 8. The butterfly-pair search range, `13 < len <= 1024`

**Untested.** `simd_planner::design_butterfly_product`. Scalar uses `len > 992 || is_power_of_two`
instead, so the two differ and neither bound is measured.

### 9. `partition_factors()` balancing

**Identified as a structural loss source, not a tunable constant.** The spike's diagnosis of NEON's
1.223 mean regret was that balancing the split deepens the recipe tree, and every extra MixedRadix
level costs two transposes plus twiddles. RadixN is the fix for exactly that, which is why the
1.223 figure is now stale (see below).

## Measured this session

### GoodThomas earns its keep, decisively

NEON, M1, every coprime pair of butterflies with `14 <= l·r <= 1024`, leaves built by the real
planner, timed both standalone and chunk-by-chunk over a ~1 MiB working set. Ratio below 1.00x means
`GoodThomasAlgorithmSmall` beats `MixedRadixSmall`. Raw data in `gt-vs-mr-neon-m1.txt` next to this
file.

| set | pairs | in-context geomean | GT wins |
| --- | --- | --- | --- |
| f32, excluding pairs with 31 | 120 | **0.845x** | 120/120 |
| f64, excluding pairs with 31 | 120 | **0.820x** | 120/120 |
| f32, pairs containing 31 | 18 | 0.987x | 6/18 |
| f64, pairs containing 31 | 18 | 0.860x | 18/18 |

Worst case in the 120 is 0.98x. So the `gcd == 1` rule is right and there is nothing to recover.

**Two findings worth keeping.**

The working-set sensitivity does not matter here. The residuals run had `gts` overhead swinging 3x
across the L1 boundary, so the expectation was that GoodThomas would look worse in context than
standalone. It doesn't: f32 goes 0.852x standalone to 0.862x in context, f64 0.811x to 0.825x. The
working set moves both algorithms about equally and the ratio is what the planner needs.

**`NeonF32Butterfly31` is an anomaly worth chasing separately.** All 12 f32 losses are pairs
containing 31, and only on f32; the same pairs on f64 sit at a perfectly normal 0.860x. So it is not
GoodThomas degrading, it is something specific to the f32 path through Butterfly31 where the index
permutation stops paying. Narrow, real, and invisible to a structural predicate.

Size trend, excluding 31-pairs, in-context geomean:

| length | f32 | f64 |
| --- | --- | --- |
| under 100 | 0.774x | 0.765x |
| 100-300 | 0.879x | 0.840x |
| 300-600 | 0.914x | 0.872x |
| 600-1024 | 0.915x | 0.917x |

Decaying, but still ~8% at the top of the reachable range.

### Plain `GoodThomasAlgorithm` is dead code, as far as planning goes

It has a `Recipe` variant, a `len()` arm and a `build_fft` arm in all four planners, and no planner
ever constructs it. Zero occurrences in 40000 scalar recipes and 40000 NEON recipes, against 10518
and 18536 `GoodThomasAlgorithmSmall`.

So above the size cliff, coprimality stops being consulted entirely and everything becomes
`MixedRadix`. Whether Good-Thomas would pay at larger sizes has never been tested, because the
planner cannot express it. Given the small-size advantage is still 8% at the top of the range and
decaying slowly, this is not obviously a dead end.

### The three SIMD planners now agree exactly

Recipes dumped for every length 1 to 20000, f32 and f64, on NEON, SSE and wasm against
`upstream/master`. All three: 26290 lengths gain a RadixN, 33 further differences are the 320/576
butterfly-pair reorder propagating through nested designs, nothing else moves. **All three backends
produce identical designs at all 40000 lengths.** Useful invariant to re-check after any change here.

### Rader's precompute

Per-element permutation cost on M1 f64 went 8.08-8.33 ns to 1.62-2.36 ns, end to end 1.24x to 1.61x
over eight smooth primes 1009 to 100801. Already in #178.

## What FFTW does, and what it tells us

Checked against primary sources this session, because the design options below hinge on it.

### FFTW_ESTIMATE is a pure operation count

`X(iestimate_cost)` in `kernel/planner.c` is the whole heuristic:

```c
double cost = pln->ops.add + pln->ops.mul
#if HAVE_FMA
            + pln->ops.fma
#else
            + 2 * pln->ops.fma
#endif
            + pln->ops.other;
```

over `typedef struct { double add; double mul; double fma; double other; } opcnt;`. No timing, no
calibration, **and no cache or working-set term at all**.

`other` is the load-bearing field: the non-arithmetic bucket (loads, stores, index arithmetic,
permutation work), estimated per-solver. Without it a flop count prices Good-Thomas as nearly free,
since eliminating twiddle multiplies is its whole point, and misses that it pays in index
permutation. Any structural model needs the same term.

Costs compose additively up the tree, and the helper comments give the model away:
`X(ops_madd)` is documented as `dst = m*a + b`, i.e. a parent running child `a` m times plus its own
ops. That is the same assumption the spike validated empirically at ±6%.

### FFTW's own verdict on op counts

> "there is an estimate mode that performs no measurements whatsoever, but instead minimizes a
> heuristic cost function: the number of floating-point operations plus the number of 'extraneous'
> loads/stores... This can reduce the planner time by several orders of magnitude, but with a
> **significant penalty observed in plan efficiency**. This penalty reinforces a conclusion of [3]:
> **there is no longer any clear connection between operation counts and FFT speed**, thanks to the
> complexity of modern computers."

This is the sentence that kills the "structural op-count model" idea for general plan ranking.

### MEASURE is dynamic programming with memoization

Solvers are tried in sequence, each may recurse into the planner for sub-problems, and the fastest
by explicit measurement wins. The paper is candid about the limitation, and it is the same one the
spike measured:

> "the FFTW planner uses dynamic programming: it optimizes each sub-problem locally, independently
> of the larger context. **Dynamic programming is not guaranteed to find the fastest plan, because
> the performance of plans is context-dependent on real machines**: this is another engineering
> tradeoff that we make for the sake of planning speed."

Memoization stores "a 128-bit hash of the problem and a pointer to the solver that generated the
plan", MD5, collisions harmless. That memo table **is** the wisdom table, which is why wisdom is a
byproduct of planning rather than a separate feature.

Timing methodology, `kernel/timer.c`: `FFTW_TIME_LIMIT 2.0`, `TIME_REPEAT 8`, `TIME_MIN 100.0`.
Start at one iteration, double until above `TIME_MIN`, repeat up to 8 times taking the **minimum**,
bail past 2 seconds. Same shape the spike converged on independently, which took cross-process
repeatability from ±50% in 2021 to 0.1-0.5%.

If there is no cycle counter, `measure_execution_time` returns negative and `evaluate_plan` does
`goto estimate`. MEASURE silently degrades to ESTIMATE. Relevant: `wasm32-unknown-unknown` has no
clock.

### Nobody uses MEASURE, and the reason is not speed

`FFTW_MEASURE` is flag value 0, so FFTW's own default is to measure. Every high-level binding
overrides it: MATLAB's `fftw('planner')` defaults to `'estimate'`, pyFFTW's `PLANNER_EFFORT`
defaults to `'FFTW_ESTIMATE'`, Julia's wrappers default to estimate.

The dominant reason is an API incompatibility, not planning time:

> "the planner overwrites the input array during planning unless a saved plan is available for that
> problem... **The only exceptions to this are the `FFTW_ESTIMATE` and `FFTW_WISDOM_ONLY` flags.**"

Any wrapper offering `y = fft(x)` cannot use MEASURE without copying, because the user's array is
the input. Add one-shot transforms, plan-dependent rounding breaking bit-reproducibility, and a
planner that is not thread-safe (only `fftw_execute` is), and estimate wins everywhere except the
plan-once-run-for-hours case FFTW was designed for.

**Consequence for RustFFT: the bar to clear is FFTW_ESTIMATE, not FFTW_MEASURE.** And RustFFT's
fixed planner is not an op count; its constants encode things that were actually timed. That is
plausibly the stronger heuristic for the mode people really run.

### Wisdom's maintenance contract is the real problem

> "It should be safe to reuse wisdom as long as the hardware **and program binaries** remain
> unchanged... It is therefore wise to recreate wisdom every time an application is recompiled."

Recompiling your own application invalidates it, via "differing code alignments". So the validity
key is (hardware, FFTW build, your binary) and FFTW can only check one of the three. Staleness is
therefore **silent**: you get a valid plan that is merely slower, with no signal.

The deployment assumptions have aged badly: containers with no persistence, autoscaling across
instance types, heterogeneous fleets. And `/etc/fftw/wisdom` needs a sysadmin who runs
`fftw-wisdom`, which in practice means nobody.

## Architectural options

### A. Better compile-time constants (status quo, improved)

Fix the constants in the inventory above, using offline measurement.

**For:** no new machinery, no I/O, no staleness, no plan-time cost. Constants are versioned with the
crate and cannot go stale relative to the binary. Works identically on wasm and in containers. This
is effectively "wisdom resolved at compile time, generated once by whoever has the hardware".

**Against:** one-size-fits-all across machines, and each constant needs a measurement campaign on
hardware you physically own.

**Verdict: the default choice.** Three separate lines of evidence in this session point here.

### B. Full estimating planner (cost model in the planner)

The `estimating_planner` branch: table-driven model, 22 butterflies and 58 Radix4 shapes measured
and stored, six fitted per-element overheads for mr/mrs/gt/gts/rad/bs.

**The headline number is stale.** Model 1.019 mean regret against planner 1.223 was measured on
NEON off upstream 6.4.1 on 2026-09-07, *before* RadixN. The recorded diagnosis of that 1.223 was
`partition_factors()` deepening the tree, which is exactly what RadixN removes. Scalar, which
already had RadixN, measured 1.048 not 1.223. **Expect NEON to have moved most of the way to
scalar's number.**

**One run settles whether this is a 20% idea or a 5% idea:** re-run the regret survey on the same 33
lengths, `--planner neon`, against `simd_radixn_split` instead of 6.4.1. The adapter already has a
`Spec::RadixN` case because scalar needed one, so pointing it at the SIMD planners is adding a match
arm, not a port.

**Also unresolved:** the model was only ever validated on f64, and f32-versus-f64 vector width is
precisely the axis causing trouble now. And the ship-a-table problem was never solved.

### C. Structural op-count model (no measurement at all)

Count passes, transposes and complex multiplies from the recipe tree; no calibration; ships anywhere.

**Verdict: drop it.** This is FFTW_ESTIMATE, FFTW has run the experiment for 25 years, and their own
conclusion is that operation counts no longer predict FFT speed. Survives only in the scoped form
below, where it is not ranking arbitrary trees but making one binary comparison between candidates
that differ a lot.

### D. Scoped estimates: replace specific constants with the comparison they approximate

Fence off the obvious arms (powers of two to Radix4, small lengths to butterflies) and use a cost
comparison **only** where the current code has a hand-tuned constant and no confidence in it.

**For:**

- It is not adding a cost model, it is **deleting a constant** and replacing a step function with
  the comparison it was approximating. Much easier to argue upstream.
- The failure mode that would sink an estimating planner is regressing cases that already work.
  Powers of two are already optimal, so fencing them off makes that structurally impossible. The
  blast radius is exactly the set of lengths where you currently have no confidence.
- All three open questions blocking #179 are in the non-obvious set.
- It fixes things a constant structurally cannot, like two primes sharing a `len - 1` bucket with
  very different inner trees.

**Against:** needs leaf costs from somewhere, so it does not fully escape calibration. And note that
FFTW's `other` counts are per-solver hand-estimates, so FFTW does not escape hand-tuning either, it
**relocates** it into many small local constants instead of a few global ones. That relocation is
probably the most useful thing to steal.

**Verdict: the interesting direction, and `design_prime` is the place to start.**

### E. Measure at plan time, persist nothing

Time both candidates for a scoped decision using the planner's own scratch buffers, pick, store
nothing. Opt-in entry point, e.g. `plan_fft_forward_measured(len)`, off by default.

**For:** correct for the machine it runs on by construction. No file, no format, no staleness, no
lifecycle, no I/O. **FFTW cannot do this** because MEASURE clobbers the caller's arrays and its
search space is far too large; RustFFT has neither problem, since the planner never sees user data
and scoped decisions are two-candidate comparisons.

**Cost is bounded and predictable:** on the order of a few dozen executions of the transform. At
100801 a Rader's runs ~2.4 ms, so tens of milliseconds; at 1009 it is microseconds. Unacceptable as
a default for one transform, trivial for anyone about to run millions, which is who opts in.

**Against:** plan-dependent results become machine-dependent, which complicates reproducibility and
testing. Still needs the scoped decision set from D.

### F. Wisdom-style persistence

**Rejected.** Two independent reasons, both decisive:

1. Storage and staleness, per FFTW's own caveats above. Silent degradation, unobservable
   invalidation key.
2. **A library should not read or write files on its own.** It breaks wasm and sandboxed and
   read-only-container users, it puts a hidden ambient input into a planner that is currently a pure
   function of (len, direction, type), and it would have broken the 40000-recipe audit this session
   relied on. The Rust ecosystem convention is that libraries take data and applications do I/O.

If persistence is ever wanted, the clean version is **not** a wisdom format: expose the `Recipe`.
It is already a small serialisable decision tree, and `recipe_for(len)` plus `plan_from_recipe(r)`
gives an application everything wisdom does with zero I/O in the library.

Caveat: that is a much bigger API commitment than it looks. `Recipe` is `pub(crate)` in `plan.rs`
and lives in private modules on the SIMD side, and making it public freezes the planner's internal
vocabulary as stable API. Upstream has pulled back from exactly this once already, in `e181738`
"Keep RadixN, but remove it from the public API for now". Treat it as a design conversation with
ejmahler, not a small addition.

## Recommended order of work

1. **Resolve the f64 RadixN contradiction (inventory item 2).** Highest value: it unblocks #179
   leaving draft, and it decides whether the gate is width aware, backend aware, or dropped. If the
   answer is backend aware, that is also the strongest single argument that the fixed planner is
   running out of road.

2. **The `design_prime` cost comparison (option D applied to inventory item 1).** Replace
   `MAX_RADER_PRIME_FACTOR` with a comparison of the two trees the planner already builds. Start by
   checking whether an estimate reproduces the SSE bucket table already in `NOTES.md` — **this needs
   no new benchmarking**. Two candidates only, so no enumeration blowup, and both sides are built
   from "obvious" shapes whose leaf costs are the cheap, stable part to measure. If it reproduces
   the table, a constant is deleted and open question 3 goes with it.

3. **Re-run the NEON regret survey against `simd_radixn_split` (option B).** One run, and it tells
   you whether the estimating planner is still a 20% idea or has become a 5% idea. Cheap, and it
   should probably happen before any large investment in B.

4. **Reconcile 576 f64 on SSE:** 2.06x in the port commit, 0.78x in `NOTES.md`. Same machine, same
   plan change, so one of the two runs is wrong.

5. **The Rader's cutoff on NEON and wasm.** If they do not land near SSE's f32 7 / f64 31, then
   `max_rader_prime_factor` has to be backend aware rather than width aware. Largely moot if 2 lands.

6. **RadixN base sweep (inventory item 3)**, using the ignored-test approach, not a tool port.

7. **`NeonF32Butterfly31`**, separately from all of the above. Narrow and probably self-contained.

## Reusable technique notes

**Recipe-diff audit.** Append a `#[test] #[ignore]` to a planner module that prints
`design_fft_for_len` for every length 1..=20000 for f32 and f64, run it in a `git worktree` of the
baseline and in the working tree, diff. Then classify every difference programmatically rather than
eyeballing it: "gained algorithm X" versus "explained by known change Y" versus "unexplained".
Catching that all 33 non-RadixN differences were two plans propagating through nested designs is
what made the split PR defensible. Cheap, and it should be run after any planner change.

**Dumping recipes under wasm.** The `wasm-bindgen` harness captures `println!` and only shows it on
failure, and a multi-MB string panics the harness. Use `wasm32-wasip1` with a plain `#[test]` and a
Node WASI runner instead:

```js
// wasi_run.mjs
import { WASI } from 'node:wasi';
import { readFile } from 'node:fs/promises';
import { argv, env } from 'node:process';
const args = argv.slice(2);
const wasi = new WASI({ version: 'preview1', args, env, preopens: { '/': '/' } });
const mod = await WebAssembly.compile(await readFile(args[0]));
process.exitCode = wasi.start(await WebAssembly.instantiate(mod, wasi.getImportObject()));
```

```sh
RUSTFLAGS="-C target-feature=+simd128" \
CARGO_TARGET_WASM32_WASIP1_RUNNER=/path/to/wasi_run.sh \
  cargo test --release --target wasm32-wasip1 --lib --features wasm_simd <name> -- --nocapture
```

Note `wasm32-unknown-unknown` has no clock, so `Instant::now()` panics there; wasip1 is required for
anything that times. And V8 tiers from Liftoff to TurboFan, so any wasm measurement at short warmup
is timing unoptimised code: always use long blocks.

**Running SSE on an ARM Mac.** `cargo test --release --target x86_64-apple-darwin --lib sse` works
under Rosetta and is enough to run the SSE test suite and dump SSE recipes. AVX tests fail there,
Rosetta does not support AVX, so filter to `sse`.

**Timing loop that actually repeats.** Round-robin the candidates and take the **minimum** of
several rounds, doubling the iteration count until a block exceeds a floor. Do not use the mean of a
`cargo bench` run. This is what took the 2021 spread from ±50% to 0.1-0.5%, and independently it is
exactly what FFTW's `measure_execution_time` does.

**Comparing algorithms in context, not standalone.** An inner FFT inside a MixedRadix runs
chunk-by-chunk over a large buffer, so time it that way: allocate `len * reps` and loop
`buffer.chunks_exact_mut(len)`. For the GoodThomas question it turned out not to change the answer,
but that was worth knowing rather than assuming.
