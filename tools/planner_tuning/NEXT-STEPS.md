# Counted cost model: where to pick this up

Written 2026-09-11, at the end of the spike that produced `RESULTS.md`. That document is the
evidence; this one is the plan. Read `RESULTS.md` first for numbers, and `OP-COUNTS.md` for how the
instruction counts were derived.

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

5. **f32 beyond NEON.** NEON f32 is now done: worst 1.121, mean 1.024, held out, against the
   planner's 1.969. It needed its own weight set, so weights are per (machine, element type). Still
   untested: f32 on SSE, which is where the even-base constraints interact with `radix4_bases()`
   filtering on a multiple of 4, and where `design_radixn`'s f32 base fixup can return None.

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
