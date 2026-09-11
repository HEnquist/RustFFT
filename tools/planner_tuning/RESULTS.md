# Counted-op cost model: experiment result

**Verdict: the idea works, and comfortably clears the bar.** On NEON f64 over the 33-length
survey, a cost model built entirely from reading the source picks a recipe within **4.3% of the
fastest in the worst case** and within 0.3% on average, against a target of "reliably within 20%,
10% would be amazing".

## Numbers

Regret is the measured time of the chosen recipe divided by the measured time of the best
enumerated candidate. NEON, f64, Apple M1, 33 lengths, `--cap 48`.

| | mean | median | p90 | worst |
|---|---|---|---|---|
| shipping planner (post-RadixN) | 1.0925 | 1.0327 | 1.2605 | 1.4945 |
| **counted model** | **1.0027** | **1.0000** | **1.0052** | **1.0427** |
| counted model, held-out half | 1.0036 | 1.0000 | 1.0151 | 1.0427 |
| measured-table model, for reference | 1.019 | 1.000 | - | 1.074 |

The reference row is the fitted table-driven model from `estimating_planner`, measured pre-RadixN
on the same lengths, so it is indicative rather than a like-for-like comparison. The counted model
is at least as good as the thing it would replace, while needing no per-machine table.

The counted model picks the outright fastest recipe at **29 of 33 lengths**. The four misses are
1.043, 1.025, 1.015 and 1.005, all of them RadixN base choices one step off.

## What the model is

Two parts.

**Counted arithmetic, read from source.** Instruction counts for all 22 NEON f64 butterflies, and
formulas for the composing algorithms' passes, twiddle multiplies and transposes. The derivation
is in `OP-COUNTS.md`. The generated prime butterflies came out as an exact closed form,
`(h-1)(2h+5)` with `h = (len+1)/2`, so they cannot drift when sizes are added or removed.

**A coarse memory term.** Each pass is charged per element touched, scaled by how it walks memory
(sequential, strided, permuted) and by which level of an assumed cache hierarchy the working set
falls in. The working set is the whole transform's buffer, threaded down unchanged to every nested
algorithm: that is what stops an inner FFT from being priced as if it ran standalone, which is the
specific failure that sank the 2021 attempt.

The only quantities not read off the source are six weights converting memory access into
arithmetic-instruction equivalents. Those were fitted once, offline, against one recorded dataset.

## Robustness, which is the real evidence

Across the whole 108-point parameter grid, worst-case regret takes only four distinct values:

| worst | settings |
|---|---|
| 1.0427 | 12 |
| 1.0854 | 12 |
| 1.1669 | 72 |
| 1.2903 | 12 |

**96 of 108 settings clear the 20% bar**, so the answer does not depend on hitting the weights
precisely. Two specific insensitivities are worth recording:

- The **DRAM cost is irrelevant to the ranking**. Values 3.0, 6.0, 10.0 and 16.0 give byte-identical
  results. This independently reproduces the finding recorded under option G in
  `PLANNER-DESIGN.md`: a working set that moves moves both candidates about equally, so it shifts
  absolute times without reordering them.
- The Rader's weight is flat from 15 to 120. Only the order of magnitude matters.

That insensitivity is the main reason to think the weights will travel between machines, but it is
not proof: see below.

## How the memory term is calculated, and how much each part earns

The calculation is deliberately crude. For each pass:

```
cost = (number of load and store instructions)
     * (per-access cost at the cache level the working set falls in)
     * (1.0 sequential | 1.5 strided | 1.5 permuted)
```

Ablating each part against the same dump gives:

| variant | mean | worst |
|---|---|---|
| A. full model | **1.0027** | **1.0427** |
| B. pattern-blind: sequential priced the same as jumpy | 1.0196 | 1.2459 |
| C. cache-flat: one access cost, no L1/L2/DRAM distinction | 1.0027 | 1.0427 |
| D. no memory term, arithmetic plus the Rader's chain only | 1.3295 | 1.7182 |
| E. pure operation count, the FFTW_ESTIMATE analogue | 1.3594 | 1.7182 |
| F. full memory term but no Rader's serial chain | 1.0352 | 1.6151 |

Three conclusions, and the second is the important one.

**1. The memory term is the whole difference.** A pure operation count (E) scores mean 1.359 and
worst 1.718, which is *worse than the shipping planner*. That reproduces FFTW's own verdict on
`FFTW_ESTIMATE` locally and on this codebase, rather than inheriting it. Adding the memory term
takes the same op counts from 1.359 to 1.003. So the thing `PLANNER-DESIGN.md` dropped as option C
really does fail, and the thing that rescues it is exactly the term FFTW does not have.

**2. The cache hierarchy contributes nothing at all.** Variant C removes the L1/L2/DRAM distinction
entirely, charging one flat cost per access, and produces *byte-identical picks at all 33 lengths*.
The assumed cache sizes, which is the entire machinery option G was rejected over, are doing zero
work. What carries the ranking is how many elements a recipe touches and how jumpily, not where
they are served from. So the model reduces to:

```
cost = arithmetic instructions
     + 1.0 * sequential accesses
     + 1.5 * non-sequential accesses
     + the Rader's serial index chain
```

with **no cache sizes, and therefore nothing to query or assume**. This also explains why the DRAM
weight was irrelevant in the grid above: it was never being used to discriminate anything.

Do not over-read it: 33 lengths on one machine, and a machine with a very different hierarchy could
in principle behave otherwise. The cache-level code is left in place so that can be tested rather
than assumed. But on this evidence the shippable model is simpler than the one that was rejected.

**3. Sequential versus jumpy is load-bearing, but the finer distinction is not.** Variant B, which
prices every access the same, degrades worst-case from 1.043 to 1.246. Yet the best fit has
`strided == permuted == 1.5`, so the model currently does not distinguish a transpose from a
digit-reversal scatter. One knob, "is this access sequential or not", is carrying all of it.

A clear next refinement, not done here: charge by **actual stride against the cache line**. A
RadixN cross-layer strides by `num_columns`, which grows with every layer, so the early layers are
line-friendly and the late ones fetch a whole line per element. The model currently gives every
layer the same price. That is the most obvious remaining source of error, and it is derivable from
the code rather than from measurement.

## The one structural correction the experiment forced

The first run scored mean 1.040 but **worst 1.615**, and every bad length was a Rader's pick, two
of them nested Rader's. The cause was a genuine modelling error found by reading the code rather
than by fitting: on this branch `raders_algorithm.rs` recomputes `index = index * root % len` per
element, where `len` is a `StrengthReducedU64`, so that `%` is two 64x64->128 widening multiplies
plus a shift and a subtract. Worse, `index` feeds the next iteration, so the chain is loop-carried
and latency-bound rather than throughput-bound.

Pricing that chain took worst-case from 1.615 to 1.167, and the remaining memory-weight tuning took
it to 1.043. This is the pattern to expect: **a bad number means a missing term, not a wrong
weight.** It is the same diagnostic that found the doubled Rader's inner FFT in the 2026-09 work.

## Caveats

1. **One backend, one float type, one machine.** NEON f64 on an M1. Nothing here shows the weights
   transfer to SSE on a different microarchitecture, which is the claim the whole idea rests on.
   The cheap next test is one `dump` on the thinkcentre and one offline refit.
2. **33 adversarial lengths.** They are mixed-factor composites near round numbers, chosen because
   that is where a planner is weakest. Powers of two come out 1.000 for everyone.
3. **Regret is a lower bound.** Candidate inner recipes come from the same planner, so the true
   optimum is better than "best measured" and the real distance from fastest is larger.
4. **The Rader's term is tied to current code.** If the permutation precompute from
   `raders_precompute` lands, that per-element cost drops by roughly 4x and the term must be
   re-derived. That is inherent to a model read from source: it is accurate because it tracks the
   code, and it must be updated when the code changes.
5. **`--cap 48`** trims the candidate list at lengths with hundreds of divisors.

## Reproducing

```sh
cd tools/planner_tuning && cargo build --release
./target/release/planner_tuning verify --planner neon 1000 1050 1200 1296 720 840 960 1009 1013 97 397
./target/release/planner_tuning dump --planner neon --rounds 7 --cap 48 --out dump_neon_f64.tsv <the 33 lengths>
./target/release/planner_tuning score --seq-l2 1.5 --seq-dram 6.0 --strided 1.5 --permuted 1.5 \
    --rader-index 50 dump_neon_f64.tsv
./target/release/planner_tuning explain 'rad(rn(7.6,rad(rn(7.5.4,b17))))'   # cost tree, to check recursion
./sweep.sh dump_neon_f64.tsv                                               # the 108-point grid
```

`dump` measures once and writes every candidate's time; `score` and `sweep.sh` are pure replay, so
model iteration after the first run needs no machine at all. That is the part that makes this
maintainable where the measured table was not.
