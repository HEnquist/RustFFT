# Counted cost model: where to pick this up

Written 2026-09-11, at the end of the spike that produced `RESULTS.md`. That document is the
evidence; this one is the plan. Read `RESULTS.md` first for numbers, and `OP-COUNTS.md` for how the
instruction counts were derived.

## Update, 2026-09-15: the crossover term, and a bigger dataset

Nothing below this section is retracted, but two numbers in `RESULTS.md` are stale and one
structural weakness in the dataset has been fixed.

**`RESULTS.md`'s headline is stale on this branch.** It records NEON f64 at mean 1.0027, worst
1.0427. Scoring the same `dump_neon_f64.tsv` with the unmodified model on `simd_radixn_split`
(9ce0980) gives **1.0125 / 1.1669**. The RadixN work moved it underneath the document. Re-measure
before quoting any figure from `RESULTS.md`.

**A sweep over every length 1..1000 found a defect the 33 length set could not see.** Timing the
planner's pick against the model's pick at all 1000 lengths showed the model losing systematically
wherever a `Small` variant was available: 127 lengths at a geometric mean of 0.945 in f64, 214 at
0.930 in f32. The cause is structural: **every one of the original 33 lengths is 1000 or larger**,
and the `Small` variants need two butterfly inners, so they stop being candidates near 1024. The
set was blind to the whole class.

**The mechanism is a crossover, not a sign error.** `MixedRadixSmall` and
`GoodThomasAlgorithmSmall` transpose with `array_utils::transpose_small`, whose read index strides
by `width` and so touches a fresh 64 byte line per 16 byte element. The general forms call the
`transpose` crate, which tiles to recover that reuse and pays per row setup for it. Dearer per row,
cheaper per element. Measured general over small on the M1:

```text
len        22     28     45    104    496    992
gt/gts   1.341  1.270  1.182  1.056  1.001  1.027
mr/mrs   1.115  1.059  1.048  1.013  0.931  0.948
```

The decay towards 1, and MixedRadix crossing under it near len 200, is a fixed cost amortising. A
per element difference could produce neither shape. Two wrong turns are worth not repeating: making
the general transpose `Sequential` lets `mr(b32,b32)` win at 1024 (regret 1.167), and charging both
variants the same pattern lets `mrs(b32,b32)` win there instead (1.290). The original `Permuted`
charge on the small variants was load-bearing, just misexplained; leave it alone.

**The fix is one weight.** `Params::general_row`, default 30, charged per row to the general
variants only, with rows taken as `1.5 * (width + height)`. All twelve pairs above are called
correctly for any value in 21 to 42, bounded by Good-Thomas at 992 below and MixedRadix at 496
above. Over the full 1..1000 sweep it takes f64 from 129 losses beyond 2% to 26, and f32 from 216
to 93.

**The dataset is now 44 lengths.** Added 22, 45, 59, 104, 120, 233, 320, 373, 496, 720, 992, dumped
with the original settings (`--rounds 7 --block-ms 10 --cap 48`) and appended to
`dump_neon_f64.tsv` and `dump_neon_f32.tsv`. `verify` is clean at all eleven. The train/test splits
were regenerated, so they are 22/22 now and any figure quoted against the old 17/16 split is stale.
Scores on the extended set, f64:

| | mean | p90 | worst |
|---|---|---|---|
| shipping planner | 1.0853 | 1.2605 | 1.4945 |
| model, `general_row` 0 | 1.0322 | 1.1268 | 1.3761 |
| model, `general_row` 30 | **1.0207** | **1.0582** | **1.2747** |

Held out: train 1.0405 -> 1.0269, test 1.0238 -> 1.0146. The f32 numbers in the session were run
with the f64 weights, so they show the direction but are not the f32 result; retune before quoting.

**What the extended set now exposes, in priority order.**

1. **Rader's versus Bluestein's is the worst remaining defect**, and it is the same decision as
   Track A below, arriving from the other side. The model takes `rad(...)` where the planner's
   `bs(...)` is faster: len 59 at 1.275, 233 at 1.146, 373 at 1.127 in f64, and up to 1.6x in f32.
   Note the model is *worse than the planner* at these three, which is new.
2. **The width/height asymmetry is now measurable.** The model ties `mr(A,B)` with `mr(B,A)`, so it
   picks between them arbitrarily, and four of the new lengths land on the wrong one: 22 picks
   `gts(b11,b2)` for 1.058, 104 `gts(b13,b8)` for 1.019, 992 `gts(b31,b32)` for 1.041. That is
   item 2 under Track B, and it was not scoreable before.
3. Lengths 120 and 320 are large shipping-planner failures (1.387 and 1.222) that the model gets
   right, so they are useful regression guards.

**Uncommitted state in this worktree.** `counted.rs` (the `general_row` term) and `main.rs` (its
`--general-row` flag) are the real change. `main.rs` also carries a throwaway `sweep` subcommand
used only for the 1..1000 comparison and its artifact; **it is not to be committed**, so drop those
hunks before committing anything else from `main.rs`. The dump and split files are regenerated data.

## State of the work

Branch `counted_cost_spike`, worktree `/Users/henrik/repos/RustFFT-counted`, based on
`simd_radixn_split` (9ce0980). Eight commits, nothing merged anywhere, library untouched outside
the `tuning` feature.

The verdict is positive: a cost model built by reading source picks within 4.3% of the fastest
recipe on NEON f64, 12.1% on NEON f32 and 15.2% on SSE f64, worst case, against a target of
"reliably within 20%". It beats the shipping planner on mean and worst on all nine datasets, holds
on held-out halves, and does not degrade when the working set leaves cache.

Datasets are kept as `dump_*.tsv` (gitignored, regenerate with `dump`). Scoring is pure replay, so
model iteration needs no machine.

## The one thing to do first

**Decide whether to pursue the scoped form or the full estimating planner.** They are different
projects and the evidence now favours the scoped one.

| | scoped (option D) | full estimating planner (option B) |
|---|---|---|
| what it does | replaces one hand-tuned constant with a two-candidate cost comparison | enumerates candidates and prices them all |
| plan-time cost | negligible | 20x to 1265x the fixed planner, 0.02-0.6 ms |
| new machinery | a cost function, no enumeration | enumeration inside every planner |
| evidence it works | Rader's vs Bluestein's: 29 of 30 correct across both backends | mean 1.003-1.052, worst 1.043-1.152 |
| risk | low, one decision, auditable by recipe diff | changes every plan, machine-dependent weights uncharacterised |

## Track A: ship the scoped win (recommended)

Replace `MAX_RADER_PRIME_FACTOR` with a comparison of the two trees `design_prime` already builds.

**The evidence is unusually clean.** Three primes have `lpf(len-1) = 23` exactly, so the current
rule must answer them identically, and the truth does not agree: 1013 wants Bluestein's by 1.13x on
NEON and 1.30x on SSE, while 9661 and 100189 want Rader's by 1.09x to 1.39x. No threshold on that
quantity can be correct at any value. 991 adds a backend-dependent answer, Rader's by 1.4% on NEON
and losing by 14.5% on SSE. The counted model gets 14 of 15 on NEON and 15 of 15 on SSE.

1. Port the minimum of `counted.rs` into the crate: the butterfly tables, `mul_complex`, and enough
   of the pass model to price a `Raders` tree against a `Bluesteins` tree. No enumeration.
2. Decide where the weights live. Per `RESULTS.md` they are per-machine, not per-backend, but this
   comparison is between two very different trees so it should be far less weight-sensitive than
   the MixedRadix-versus-GoodThomas near-tie. **Check that before porting**: sweep the weights over
   the prime datasets and confirm the 29-of-30 result is stable across the whole grid. If it is,
   ship constants and stop worrying.
3. Recipe-diff audit over 1..20000 against the current planner, as `PLANNER-DESIGN.md` describes.
4. Re-measure the 15 primes on both machines to confirm the diff is an improvement.

Delivers a deleted constant and a measurable win, with no plan-time cost and no enumeration.

## Track B: what the full planner still needs

In rough order of how much each would change the picture.

1. **A third machine.** The two in hand sit on the diagonal of {ARM, x86} x {strong, weak memory},
   so instruction set and memory system are perfectly confounded. `radixn_extra = 5` is the largest
   single term, worth 1.654 -> 1.152 on SSE, and is justified as an instruction-set property
   (16 xmm registers against 32 v) but fitted on one x86 machine.
   - **ryzen250, Zen 5**: same 16 xmm registers, strong memory system. Tests whether the register
     story is really the register story or just Coffee Lake's scheduler. Its recorded noise is not
     disqualifying: the signals are medians over hundreds of pairs, and even 40% independent
     per-pair noise pins the median to +-0.028 against effects of 0.08. Do not quote worst-case
     regret from it. Single-channel memory does not matter, because no test length reaches DRAM on
     it and the DRAM weight is inert anyway.
   - **pi5, Cortex-A76**: NEON with 32 registers, weak memory system. The op counts and the
     register term must carry over unchanged, so anything that moves is memory-system. Also the
     representative machine for NEON's real audience per `METHOD.md`.
   - Neither is reachable. Both need an address, `id_ed25519` in `authorized_keys`, and a
     toolchain. SSH was not enabled on the Ryzen as of 2026-09-09.

2. **Asymmetry in width and height.** The cost function ties `mr(A,B)` with `mr(B,A)` at all 415
   reversed pairs, yet 133 to 179 of them measure more than 2% apart. `width` and `height` play
   different roles in the transposes and in which FFT runs over contiguous rows. This is the
   clearest unexploited improvement and it is derivable from the code.

3. **MixedRadix versus GoodThomas.** The model has a fixed preference rather than a decision: it
   picks GoodThomas at 142 of 142 pairs, which matches truth 121/142 on NEON and 38/142 on SSE.
   With identical inners, `cost_gt - cost_mr` is a constant multiple of `len`, so its sign cannot
   vary with length. The needed correction is small, x1.055 on the M1 and x1.190 on the i3, about
   +1.8 and +7.0 instruction-equivalents per element per scatter. **Keep the correction small**:
   a large stride-aware rewrite aimed at this regressed both backends and was reverted.

4. **Integration and plan-time budget.** Enumerating inside the planners is new machinery, and the
   plan-time numbers above are the budget it has to fit. Memoising inner recipes across candidates
   is the obvious first optimisation; the measurement deliberately did not do it.

5. ~~**f32 beyond NEON.**~~ **Done, 2026-09-15.** SSE f32 measures worst 1.121, mean 1.031, held
   out at 1.105, against the planner's 1.927. All four cells of {NEON, SSE} x {f32, f64} now clear
   the bar. The feared interaction (even-base constraints against `radix4_bases()` filtering on a
   multiple of 4, and `design_radixn`'s f32 base fixup returning None) did not materialise:
   candidate counts track the NEON f32 run and `verify` is clean at every length.

   It threw off one finding worth having. With four datasets on one grid, `permuted` splits by
   element type rather than by backend, 1.5 at f64 and 4.0 to 6.0 at f32, which identified a
   missing term: `mem()` divided permuted accesses by `complex_per_vector`, but a gather or scatter
   computes an address per element and cannot fill a vector. Charging them per element,
   now the default, leaves both f64 datasets byte-identical, moves both f32 optima to 2.5, and
   takes SSE f32 from 51 to 117 of 216 grid points clearing 20%. Weights are still per (machine,
   element type); the correction buys robustness, not a smaller table. See `RESULTS.md`.

## Things that are settled, do not redo them

- **Cache sizes are inert.** Removing the L1/L2/DRAM distinction entirely changes no pick at any
  length, in cache or out, including at eight times the fitted DRAM weight. All candidates at a
  given length touch about the same data and differ only in pass count, so the level is a common
  factor that cannot reorder them. This removes the objection that killed option G: a shipping
  model needs no cache sizes, assumed or queried.
- **Stride-aware memory costs are a regression.** Charging by actual stride against the cache line
  fixes SSE's MR-vs-GT from 38/142 to 104/142 but costs NEON 1.043 -> 1.575. `cross_layer` walks
  `chunks_exact_mut(cross_fft_len)` and every gathered row is inside the current chunk, so a
  cache-resident chunk is touched once whatever the stride.
- **Per-element-type op counts are not worth maintaining.** The f32 table was counted properly and
  bought nothing: scored with the f64 table and retuned weights, the worst case is identical and the
  mean slightly better. f32 differs from f64 by a near-uniform scale, which the memory weights
  absorb, whereas SSE differs from NEON in shape, which is why that swap does cost 1.043 -> 1.246.
- **A pure operation count does not work**, confirming FFTW's own verdict locally: mean 1.359,
  worst 1.718, worse than the shipping planner. The memory term is the whole difference.
- **The MR-vs-GT preference follows the machine, not the backend**: wasm on the M1 tracks NEON on
  the M1 to within 0.004 while SSE on the ThinkCentre is 0.088 away.
- **Two mechanisms for that machine gap are falsified**: cache capacity (the gap is largest at
  6-25 KiB footprints, inside both L1s) and L1 set conflicts (flat across the 2-adic content of
  the scatter stride). The mechanism is still unknown.

## Operational notes

- `verify`'s reference DFT now runs in f64 whatever the recipes use, and its threshold is a
  length-scaled budget rather than a flat `1e-6`. The old same-precision reference produced false
  failures at f32 above len 10000, where the naive DFT is less accurate than the FFTs it judges.

- `dump` measures once, `score` and `costs` replay offline, `explain` prints a recipe's cost tree,
  `plantime` compares planning cost. `sweep.sh` grids the weights.
- Build SSE with `--no-default-features --features sse`, or the planner picks AVX.
- wasm: build for `wasm32-wasip1` with `RUSTFLAGS="-C target-feature=+simd128"` and run through
  `run_wasm.mjs` under node. Use long blocks; V8 tiers up from Liftoff.
- The ThinkCentre does **not** need the sudo governor step: no Turbo, so all cores sit at exactly
  3.100 GHz under load even on powersave. Verified by sampling `scaling_cur_freq` during a run.
- Do not compile while measuring. zsh does not word-split unquoted variables, so a `$LENGTHS`
  variable arrives as one argument and the tool panics; inline the numbers or use `${=VAR}`.
- Run `verify` after any change to candidate enumeration, before trusting a timing.
