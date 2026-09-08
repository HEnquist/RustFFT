# Planner tuning notes

Scratch notes for the SIMD planner tuning, shared between machines through the branch.
**Temporary. Delete this file before merging back.**

The branch is `sse_planner_tuning`. The name is historical, NEON gets tuned on it too, and it is
throwaway: squash and force push on it freely, it only merges back once we are happy with it.

## How to run

```sh
cargo +nightly bench --bench bench_planner_choices_sse
cargo +nightly bench --bench bench_planner_choices_neon
cargo +nightly bench --bench bench_planner_choices_wasm_simd
```

Pin to a core, the numbers move a lot otherwise:

```sh
taskset -c 4 cargo +nightly bench --bench bench_planner_choices_sse
```

Read a group by comparing `planned` against the `alt_*` of the same length and type.
A ratio below 1.00x means the planner's choice is the faster one.

## Machines

| id | CPU | notes |
| --- | --- | --- |
| ryzen250 | AMD Ryzen 7 250 (Zen 5, 780M) | laptop, powersave governor, run-to-run noise up to 2x per length, medians stable |

Add a row when you run on a new machine, and tag results below with the id.

## Measured so far

### RadixN, f32 only (ryzen250, SSE)

`planned` against `alt_mixedradix`, so lower is RadixN winning:

| len | f32 | f64 |
| --- | --- | --- |
| 1152 | 0.75x | 0.98x |
| 2880 | 0.48x | 1.02x |
| 7680 | 0.75x | 0.99x |
| 11520 | 0.56x | 1.00x |
| 23040 | 0.57x | 1.01x |
| 46080 | 0.57x | 1.00x |

f64 sits at 1.00x because the planner already picks mixed radix there, which is the self-check.
The f64 gate itself was measured by building the crate both ways: RadixN for f64 came out at 0.80x
median over 109 mixed-factor lengths, so it is off.

### Rader's cutoff (ryzen250, SSE)

Swept primes from 100 to 80000, bucketed by the largest prime factor of `len - 1`, which is what
`design_prime` actually branches on. Rader's over Bluestein's, so below 1.00x is Rader's winning:

| bucket | 2 | 3 | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 | 59 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32 | 0.45 | 0.68 | 0.68 | 0.98 | 1.22 | 1.28 | 1.19 | 1.16 | 1.28 | 1.11 | 1.40 | 1.86 | 2.02 | 1.99 | 2.12 | 1.93 | 2.57 |
| f64 | 0.39 | 0.65 | 0.62 | 0.69 | 0.64 | 0.75 | 0.71 | 0.81 | 0.83 | 0.86 | 0.75 | 0.84 | 0.91 | 0.82 | 1.22 | 0.92 | 1.23 |

Buckets 2 and 3 are the "no other factors" case, where the cutoff never applies.
The split holds across three length bands (under 2k, 2k to 15k, over 15k), so it is not a size
artifact.

So the cutoff was mistuned in both directions, not just between 23 and 31:

- f32 wants **7**. Bluestein's inner FFT is a power of two, so it gets the full Radix4 benefit
  when a vector holds two complex numbers, and pulls ahead as soon as `len - 1` needs a factor
  above 7.
- f64 wants **43**. With one complex number per vector that advantage largely goes away, and
  Rader's keeps winning past the largest prime butterfly, so its inner FFT nesting a prime
  algorithm of its own is still cheaper than Bluestein's.

Applied as `max_rader_prime_factor`, 7 for f32 and 43 for f64. End to end check against b631958
in progress.

The harness is `examples/tune_rader_cutoff.rs` on this branch, also temporary. Run it as
`taskset -c 4 cargo run --release --example tune_rader_cutoff > out.csv`, roughly 15 minutes.

## Open questions

- 576 f64 is 0.78x against upstream master, from `design_butterfly_product` moving ahead of the
  power of two peel, which sends it to `MixedRadixSmall(24, 24)`. The same reorder gives 576 f32
  3.40x and 320 f64 1.69x, and those are the only two lengths in the whole range it touches.
- RadixN base length choice for f32 is untested against alternatives. There is a temporary
  `RUSTFFT_FORCE_RADIXN_BASE` env hook on this branch for that, also to be removed before merge.

## For the NEON and wasm_simd machine

Both fixes live in the shared `simd_planner`, so they apply to all three backends, but only SSE
has been measured. Two things to confirm:

1. `radixn_planned_*` against `radixn_alt_mixedradix_*`. If NEON f64 shows RadixN winning where
   SSE f64 did not, the `complex_per_vector < 2` gate needs to become backend aware.
2. The `prime_cutoff` group. Better still, run `examples/tune_rader_cutoff.rs` there too and
   compare the bucket table above. The f32 and f64 answers came out very different on SSE, and
   the reasoning behind that is about vector width rather than anything x86 specific, so NEON
   should land somewhere similar. If it does not, `max_rader_prime_factor` has to become backend
   aware rather than just width aware.

The bench group constants in `body.rs` still bracket the old 23 and 31 cutoffs. They want
regrouping around the new ones: at or below 7 both types use Rader's, 11 to 43 is the band where
f32 and f64 now disagree, and above 47 both use Bluestein's.
