# Counted-op cost model: experiment result

**Verdict: the idea works, on two architectures and both element types, and clears the bar on all
four.** A cost model built entirely from reading the source picks a recipe within **4.3% of the
fastest in the worst case on NEON f64** and **15.2% on SSE f64**, against a target of "reliably
within 20%, 10% would be amazing". f32 comes in at **12.1% on NEON** and **12.1% on SSE**. All four
results hold on a held-out half of the lengths.

The op counts are read per backend, as expected for different instruction sets. Beyond that, three
weights differ between NEON and SSE, and one of them, a penalty for the generic RadixN driver
against the hand-written Radix4 kernel, is not a fudge: it is exactly zero on NEON and positive on
SSE, matching a register-count argument. See the SSE section.

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

A clear next refinement would seem to be charging by **actual stride against the cache line**,
since a RadixN cross layer strides by a `num_columns` that grows with every layer. That was tried
and it is a regression on both backends; see the MixedRadix-versus-GoodThomas section below for
the numbers and for why the reasoning was wrong.

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

## Cross-backend: SSE on the ThinkCentre

The claim the whole idea rests on is that this travels. It does, but **not with the same weights**,
and the way it failed first is the useful part.

Porting the op counts needed no measurement, only more reading. SSE4.1 has no FMA, so `fmadd` and
`nmadd` cost 2 instructions instead of 1 and `mul_complex` costs 6 instead of 4. The prime
butterfly closed form re-derives from `(h-1)(2h+5)` to `(h-1)(4h+2)`, verified exactly against all
eight generated SSE sizes. Two butterflies would have been wrong had I re-weighted the NEON numbers
mechanically instead of counting the SSE source: butterfly 3 is hand-written without FMA at 10
instructions, and butterfly 8 uses `rotate_45`/`rotate_135` where NEON uses explicit multiplies.

**First attempt: the NEON weights transferred badly.** Mean 1.183, worst 1.654, barely ahead of the
SSE planner's 1.246 / 1.734. Every miss was an `rn(...)` pick, on composites and inside Bluestein's
alike, so the model was systematically overrating RadixN on SSE. No setting of the existing five
weights fixed it: the whole 108-point grid sat at worst 1.654. A missing term, not a wrong weight,
which is the same diagnostic as the Rader's case.

Length 10007 shows it plainly:

| recipe | SSE ns |
|---|---|
| `bs(10007, r4(5,b24))`, the hand-written Radix4 kernel | 480,023 |
| `bs(10007, rn(4.4.4.4.4.4,b5))`, the generic RadixN driver | 794,077 |

Nearly the same radix-4 work, 1.65x apart. On NEON that *same* RadixN recipe is the best available.

**The missing term is the generic driver, and it is readable from code plus one ISA constant.**
`cross_layer` in `src/simd_radixn.rs` gathers two vector columns before transforming either, so a
radix-R layer holds 2R rows live plus the butterfly's temporaries, while `sse_radix4.rs` is a
hardcoded 2x unroll over six twiddles. 2R fits aarch64's 32 vector registers at every radix RadixN
supports; it does not fit x86-64's 16 xmm registers. Adding one per-element-per-layer penalty for
the generic driver gives:

| | mean | median | p90 | worst |
|---|---|---|---|---|
| SSE planner | 1.2464 | 1.2323 | 1.4949 | 1.7337 |
| counted, NEON weights | 1.1826 | 1.1495 | 1.3444 | 1.6542 |
| **counted, SSE weights** | **1.0516** | **1.0490** | **1.1248** | **1.1515** |
| counted, SSE weights, held-out half | 1.0520 | 1.0608 | 1.1318 | 1.1425 |

That clears the 20% bar on a second architecture, and it holds out.

**The penalty is not a fudge that could have been fitted anywhere.** On NEON its best value is
exactly zero, and any positive value degrades NEON sharply (worst 1.043 at 0, 1.246 at 1, 1.507 at
4). The term switches on for the backend whose register file cannot hold the working set, and off
for the one that can, which is what the physical story predicts.

### What actually differs per backend

| | NEON | SSE |
|---|---|---|
| butterfly instruction counts | read from `src/neon/` | read from `src/sse/` |
| generic-RadixN penalty | 0 | 5 |
| strided access multiplier | 1.5 | 2.5 |
| L2 access cost | 1.5 | 2.0 |
| Rader's index chain, sequential/permuted, cache sizes | identical | identical |

So the per-backend surface is one counted table plus three numbers. That is a different kind of
maintenance burden from the measured model's 22 butterflies and 58 Radix4 shapes per backend per
float type: the table is derived by reading code, and the three numbers are few enough to fit once
and sanity-check rarely.

**This table is mislabelled, and a later section corrects it.** With only two machines, "per
backend" cannot be distinguished from "per machine". A wasm run on the M1 shows the memory weights
follow the machine, not the instruction set; only the counted table and the RadixN register penalty
are genuinely per-backend. See "What MixedRadix and GoodThomas actually do differently" below.

### A side finding worth keeping

This independently reproduces, and explains, the open question in `NOTES.md`: SSE measured f64
RadixN at 0.80x over 109 lengths while NEON measured it winning 1.13-1.53x, and the discrepancy is
what blocks `simd_radixn_split` upstream. The mechanism proposed here is register pressure in the
generic `cross_layer` against x86-64's 16 xmm registers. That is a testable claim independent of
any cost model: it predicts the gap shrinks for small radixes and widens for radix 6 and 7, and
that an AVX build with 16 wider registers would not fix it while a hand-written SSE RadixN would.

## Rader's versus Bluestein's, the decision `MAX_RADER_PRIME_FACTOR` hardcodes

Tested separately on 15 primes chosen to span the rule's own input, `lpf(len-1)` from 5 to 5003,
at three size decades. This is the decision option D in `PLANNER-DESIGN.md` proposes replacing
with a cost comparison.

| | right family | mean regret | worst regret |
|---|---|---|---|
| counted model, NEON | **14 / 15** | 1.0013 | 1.0139 |
| counted model, SSE | **15 / 15** | 1.0078 | 1.1176 |
| planner, NEON | 14 / 15 | 1.0408 | 1.3566 |
| planner, SSE | 13 / 15 | 1.0310 | 1.2954 |

The model's one miss, NEON at 991, is a 1.4% near-tie and costs 1.014x. That is the right failure
mode: it goes wrong only where being wrong is nearly free.

### The constant cannot be correct at any value

Three of the primes have `lpf(len-1) = 23` exactly, so the planner's rule sees identical input and
must give them identical answers. The measurements do not agree with each other:

| len | `lpf(len-1)` | NEON | SSE |
|---|---|---|---|
| 1013 | 23 | **Bluestein's** by 1.13x | **Bluestein's** by 1.30x |
| 9661 | 23 | **Rader's** by 1.39x | **Rader's** by 1.09x |
| 100189 | 23 | **Rader's** by 1.36x | **Rader's** by 1.36x |

So no threshold on `lpf(len-1)` can be right, at 23 or at any other value, because that quantity
does not determine the answer: length matters too, and the two disagree by 1.1x to 1.4x in both
directions. Retuning the constant cannot fix this; only replacing it with a comparison can. The
counted model gets all three right on both backends.

The other cross-check is 991, where the right answer is **backend-dependent**: Rader's wins by
1.4% on NEON and loses by 14.5% on SSE. A single shared constant cannot express that either.

### Caveat specific to this test

The enumerator offers exactly **one** Rader's candidate per prime, `rad(planner.plan(len-1))`,
against six Bluestein's variants, so "best measured" is mildly biased toward Bluestein's and a
better Rader's inner could flip a close call. It does not look fatal here, since Rader's still wins
outright at 6 of 15 lengths, but the near-ties (991, 1009 on SSE, 100049 on SSE) should not be
read as settled. Widening the Rader's side of the enumeration is the obvious follow-up.

## MixedRadix versus GoodThomas, and choosing between splits

Analysed offline from the dumps already taken, comparing every pair where the same factor split
appears as both a MixedRadix and a GoodThomas (142 such pairs per backend).

| | NEON | SSE |
|---|---|---|
| MR vs GT on the same factor pair | 121 / 142 | **38 / 142** |
| mean cost of a wrong call | 1.029x | 1.041x |
| worst cost of a wrong call | 1.076x | 1.255x |
| choosing among two-way splits, 27 lengths | mean 1.024, worst 1.108 | mean 1.088, worst 1.371 |

### The model does not actually decide this; it has a fixed preference

The 38/142 is not noise, it is structural. The model prefers GoodThomas at **142 of 142** pairs on
both backends. On NEON that happens to match the truth, which also prefers GoodThomas 121 of 142
times; on SSE the truth prefers GoodThomas only 38 times, so the same fixed preference is wrong
104 times.

The reason is visible in the formula. With the same factor pair, the two inner FFTs are identical,
so the whole comparison is overhead against overhead:

```
cost_gt - cost_mr = len * [ 2s*(2*scatter - 2*strided - 1) - mul_complex ]
```

which is a constant multiple of `len`. Its sign cannot change from one length to the next, so the
model can only ever answer "always GT" or "always MR". Sweeping the weights confirms it: accuracy
takes exactly two values, 85% or 14% on NEON and 26% or 73% on SSE, with nothing in between.

### But the decision is nearly a tie, which bounds how much this matters

Measured `gt/mr` ratios sit in 0.90 to 0.98 on NEON and 0.96 to 1.07 on SSE. The two algorithms are
within a few percent almost everywhere, which is why a wrong call costs 3 to 4% on average. That is
right at the model's resolution: the 2026-09 spike established that decisions worth making are
typically 20% apart, and this one is not.

It also means a **fixed per-backend preference is close to optimal for this decision**: "prefer
GoodThomas on NEON, prefer MixedRadix on SSE" scores 85% and 73%, which is what the degenerate
model already does once its weights are fitted per backend. The residual is the 1.255x tail on SSE.

### Argument order is a genuine blind spot

`mr(A,B)` and `mr(B,A)` get **identical** cost from the model, all 415 reversed pairs tied exactly,
yet 179 of them on NEON and 133 on SSE differ by more than 2% when measured. The cost function is
symmetric in its two children while the algorithm is not: `width` and `height` play different roles
in the transposes and in which FFT runs over contiguous rows. Making the model asymmetric in width
and height is the clearest unexploited improvement.

### A refinement that was predicted to help, and did not

The previous section of this document proposed charging memory by **actual stride against the cache
line** as the obvious next step, since a RadixN cross layer strides by a `num_columns` that grows
each layer. That was implemented and measured, and it is a **regression**:

| model | NEON mean / worst | SSE mean / worst |
|---|---|---|
| committed, flat pattern multipliers | **1.003 / 1.043** | **1.052 / 1.152** |
| stride-aware, saturating at the cache line | 1.109 / 1.575 | 1.031 / 1.213 |
| stride-aware, cross layers exempt while the chunk is cache-resident | 1.062 / 1.301 | 1.106 / 1.323 |

It does fix the sub-decision it was aimed at, taking SSE's MR-vs-GT from 38/142 to 104/142, but it
loses more elsewhere than it gains there. Reading the code says why the first version was wrong:
`cross_layer` walks `chunks_exact_mut(cross_fft_len)` and every row it gathers lives inside the
current chunk, so a cache-resident chunk is touched once no matter how large the stride is.
Exempting those layers recovers part of the NEON loss but not all of it, and costs SSE.

The change was reverted. The lesson is the same one the Rader's and the SSE cases taught in the
other direction: **more detail is not automatically more accuracy**, and a term has to be checked
against the whole set rather than against the decision that motivated it.

## What MixedRadix and GoodThomas actually do differently, and which machine prefers which

With the same factor pair the two inner FFTs are identical, so the whole difference is the glue:

| | MixedRadix (`mixed_radix.rs:128-158`) | GoodThomas (`good_thomas_algorithm.rs`) |
|---|---|---|
| passes over the buffer | 4 | 3 |
| transposes | 3 x tiled `transpose::transpose` | 1 x tiled |
| permutations | none | 2 x raw scatter, CRT in and Ruritanian out |
| twiddles | `len` complex multiplies | none |

So `GT - MR = 2*(scatter - tiled transpose) - 1 twiddle pass`. GoodThomas trades away the twiddle
multiply *and* one whole pass, and pays for it with two scattered passes instead of two tiled ones.
It wins whenever the scatter penalty is less than about half a complex multiply per element.

That also sets the size of any correction. Centring the model on the truth needs GoodThomas's cost
multiplied by 1.055 on the M1 and 1.190 on the ThinkCentre, which is about +1.8 and +7.0
instruction-equivalents per element per scatter, against a complex multiply costing 4 and 6. A small
term. That is worth remembering against the temptation to rebuild the memory model: the stride-aware
rewrite above was a large change aimed at a small discrepancy, and it lost more than it gained.

### Two mechanisms proposed and both falsified

**Cache capacity: no.** Both algorithms hold a scratch of `len`, so the footprint is `2*len*16`
bytes. Lengths 210 to 780 give 6 to 25 KiB, inside the L1 of *both* machines. If capacity drove the
gap it should vanish there. It is largest there:

| len | footprint | NEON@M1 | SSE@i3 | gap |
|---|---|---|---|---|
| 210-780 | 6-25 KiB, inside both L1 | 0.926 | 1.012 | **+0.086** |
| 1k-4k | 32-128 KiB | 0.950 | 1.006 | +0.056 |
| 4k-16k | 128-512 KiB | 0.981 | 1.035 | +0.054 |
| >16k | >512 KiB | 1.002 | 1.076 | +0.074 |

**L1 set conflicts: no.** Conflicts depend on the stride, not the size, so the ratio should worsen
as the scatter stride gains factors of two. It is flat across the 2-adic content of both `width + 1`
and `height`, and if anything moves the other way.

What survives is only that it is an execution property of scattered access, visible with everything
in L1. The mechanism is **not established**.

### The preference follows the machine, not the backend

The obvious confound is that "NEON versus SSE" is also "M1 versus i3-8100T". Running the
**wasm_simd** backend on the M1 separates them: same memory system, different code generator, and
no FMA. Over the same 8 lengths and the same 94 factor pairs:

| backend | machine | median gt/mr | GT wins |
|---|---|---|---|
| NEON | M1 | 0.9241 | 88 / 94 |
| **wasm_simd** | **M1** | **0.9280** | **94 / 94** |
| SSE | i3-8100T | 1.0121 | 39 / 94 |

wasm on the M1 tracks NEON on the M1 to within 0.004 while SSE on the ThinkCentre is 0.088 away.
**The MixedRadix versus GoodThomas preference is a property of the machine, not of the instruction
set.**

### Consequence: the weights are not all per-backend

That means this document's earlier "what differs per backend" table is mislabelled. Sorting the
model's inputs by what they actually depend on:

| input | depends on | how it is obtained | effect if wrong |
|---|---|---|---|
| butterfly and primitive op counts | instruction set | read from source | large |
| generic-RadixN penalty (register file) | instruction set | 16 xmm vs 32 v registers | large: SSE worst 1.152 -> 1.654 |
| memory weights (seq, strided, permuted) | **the machine** | fitted once per machine | moderate |

Measured directly, by swapping only the memory weights while holding the architectural term:

| | own weights | the other machine's weights | no register term |
|---|---|---|---|
| SSE worst | 1.152 | 1.213 | 1.654 |
| NEON worst | 1.043 | 1.290 | n/a, the term is zero |

So the expensive, architectural part of the model is genuinely portable and read from code, and the
machine-dependent part is three numbers whose misuse costs 0.06 to 0.25 of worst-case regret. That
is a far better maintenance story than the measured table, but it is **not** "no measurement at
all", and with two machines the machine-dependent part cannot be characterised.

### The two machines sit on the diagonal of a 2x2, which is the whole problem

|  | strong memory system | weak memory system |
|---|---|---|
| **ARM / NEON** | M1 (have) | **Pi 5** (missing) |
| **x86 / SSE** | **Ryzen, Zen 5** (missing) | i3-8100T (have) |

Instruction set and memory system are perfectly confounded because only the diagonal is filled. The
wasm run breaks the confound along one axis only, by holding the machine fixed and changing the
backend. Either off-diagonal machine breaks it properly; both would complete the design.

**The Ryzen tests something the Pi cannot.** `radixn_extra = 5` is the model's largest single term,
worth 1.654 -> 1.152 on SSE, and it is justified as an instruction-set property: 16 xmm registers
against 32 v registers. It is fitted on exactly one x86 machine. Zen 5 also has 16 architectural
xmm registers, so if the register story is right it should need a similar value there. If it needs
roughly zero, then "register pressure" was really "Coffee Lake's scheduler", the term is
machine-specific, and the claim that the expensive part of the model is portable is much weaker.
Zen 5 also has a strong memory system, so it independently cross-checks the wasm result: if
MixedRadix versus GoodThomas on Zen 5 behaves like the M1 despite being x86, that confirms the
preference follows the machine.

**Its noise is not disqualifying for these questions.** The recorded objection to ryzen250 is up to
2x per-length variation under powersave. That matters for a worst-case regret number, where one bad
length decides the answer, but the structural questions here are medians over hundreds of pairs.
Simulating independent per-pair noise on the existing 306 pairs: at 10% the median is determined to
+-0.007, at 20% to +-0.013, at 40% to +-0.028, against an effect of 0.079. And much of the recorded
variation is slow drift, which the round-robin interleaving cancels within a length because both
candidates of a ratio are hit equally.

So: run it, report win counts, medians and fitted terms, and **do not quote a worst-case regret
from it**. Build with `--no-default-features --features sse` as always, and note that governor and
boost pinning matter far more on a laptop than they did on the ThinkCentre, which has no Turbo.

### The instrument this needs next is a Pi 5

A Cortex-A76 is NEON with 32 vector registers but a much weaker memory system, so the op counts and
the register term **must** carry over unchanged and anything that moves is memory-system. It is the
clean third point, and it tests the specific worry that the M1's unusually strong memory system has
been flattering the model. As of 2026-09-11 it is still not reachable: `~/.ssh/config` has a bare
`Host pi5` with no user or key and the name does not resolve. It needs a reachable address, the
`id_ed25519` public key in its `authorized_keys`, and a Rust toolchain.

## Out of cache: the regime the 2021 attempt died in

Everything above was measured in cache. The footprint of a transform is `2*len*16` bytes, since
both MixedRadix and GoodThomas hold a scratch of `len`, so the 33-length survey tops out at 3.2 MiB
at length 102400 and never leaves the last level on any machine tested. That matters because the
2021 scalar model's failures were precisely out there: it lost 30% at 100k and 60% at 1M, and the
recorded diagnosis was that it was cache-blind.

Ten lengths from 500000 to 2097152, footprints 15 to 64 MiB against the M1's 12 MiB L2, so
genuinely memory-bound:

| | mean | median | worst |
|---|---|---|---|
| counted model | **1.0110** | 1.0000 | **1.0432** |
| shipping planner | 1.0218 | 1.0030 | 1.1307 |

Worst-case 1.0432, against 1.0427 in cache. **The model does not degrade when the working set
leaves cache.** It picks the outright best recipe at 6 of 10, and its single worst length is 999999,
which carries a factor of 37 and so needs a prime algorithm nested inside a split.

Note the planner also does much better here (worst 1.131 against 1.495 on the mixed-factor
composites), because these lengths are mostly smooth and land on shapes its fixed rules handle
well. The window is narrower out here; the model still wins it.

### The cache-level term is inert even here, and now it is clear why

The ablation that removes the L1/L2/DRAM distinction entirely was run again on these ten lengths,
and on sweeps of the DRAM weight up to eight times its fitted value:

| variant | mean | worst |
|---|---|---|
| cache-flat, one cost everywhere | 1.0107 | 1.0407 |
| tuned | 1.0110 | 1.0432 |
| DRAM weight x4 | 1.0110 | 1.0432 |
| DRAM weight x8 | 1.0110 | 1.0432 |

So the earlier finding was not an artefact of testing only cache-resident sizes. The reason is
simple in hindsight: **at a given length every candidate touches about the same amount of data**,
differing only in how many passes it makes over it, so the cache level enters as a common factor
that scales all candidates together and cannot reorder them. That is the same mechanism behind the
observation recorded under option G, that a working-set change moved both of two candidates about
equally.

This strengthens the case for dropping the assumed cache sizes from any shipping version: they are
now shown to be inert across three orders of magnitude of working set, in cache and out.

## Summary across every dataset

Same model, weights fitted per (machine, element type), scored against every dump taken. All rows
are f64 unless the row says f32.

| dataset | counted model | shipping planner |
|---|---|---|
| NEON small, 210-780 | 1.005 / **1.052** | 1.071 / 1.260 |
| NEON survey, 33 mixed-factor | 1.003 / **1.043** | 1.093 / 1.495 |
| NEON 15 primes | 1.001 / **1.014** | 1.041 / 1.357 |
| NEON 10 large, 0.5-2M, out of cache | 1.011 / **1.043** | 1.022 / 1.131 |
| NEON survey, **f32** | 1.032 / **1.121** | 1.171 / 1.969 |
| SSE small, 210-780 | 1.035 / **1.069** | 1.247 / 1.524 |
| SSE survey, 33 mixed-factor | 1.052 / **1.152** | 1.246 / 1.734 |
| SSE 15 primes | 1.008 / **1.118** | 1.031 / 1.295 |
| SSE survey, **f32** | 1.031 / **1.121** | 1.207 / 1.927 |
| wasm small, 8 lengths | 1.007 / **1.055** | 1.072 / 1.358 |

(mean / worst). The model wins on both statistics on every dataset. Over the 140 length-backend
cases in the f64 rows: the planner is already optimal at 52 of them, the model is strictly better
at 77, strictly worse at 13.

The op counts are load-bearing, not decoration: scoring NEON data with SSE's counts degrades
worst-case from 1.043 to 1.246.

A note on the wasm row. `Backend::parse` knows only `neon` and `sse`, so the wasm dump silently fell
back to the NEON counts, and it scores better that way (1.055) than with SSE's (1.153) despite wasm
SIMD having no FMA. That is not luck: V8 compiles wasm SIMD **to NEON machine code** on an M1, so
the op counts follow the JIT's target, not the bytecode. A shipping version would need to decide
this deliberately rather than by fallback.

## Plan time, the one axis where the fixed planner is ahead

The fixed planner answers from a few integer operations. Enumerate-and-price has to build the
candidate set and cost every member.

| len | one FFT | fixed plan | model plan | model plan in FFT executions |
|---|---|---|---|---|
| 1260 | 5.3 us | 0.5 us | 178 us | **34x** |
| 10080 | 47 us | 0.7 us | 354 us | 7.6x |
| 100800 | 580 us | 0.5 us | 585 us | 1.0x |
| 1000000 | 9.4 ms | 0.4 us | 556 us | 0.06x |

That is 20x to 1265x the fixed planner's plan time, around 540x on average. In absolute terms it is
0.02 to 0.6 ms, which is irrelevant for RustFFT's normal plan-once-run-many usage but costs tens of
executions for a one-shot small transform. The measurement is pessimistic: it builds a fresh planner
per repetition, so no inner recipe is memoised.

This is the strongest argument for the scoped form of the idea over the full estimating planner: a
two-candidate comparison at one decision point costs almost nothing, while enumerating 48 candidates
costs this.

## f32 on NEON

The whole spike had been f64. f32 changes real structure: `complex_per_vector` becomes 2, which
turns on the even-base constraints in `design_radixn`, halves the vector butterfly call count, and
sends the cross-FFT layers through `perform_parallel_fft_direct`, a different kernel that computes
two FFTs at once. `verify --f32` passes first, 0 lengths failed at about 1e-7.

| NEON f32, 33-length survey | mean | median | p90 | worst |
|---|---|---|---|---|
| counted model | **1.024** | 1.000 | 1.083 | **1.121** |
| counted model, held-out half | 1.028 | 1.000 | 1.089 | **1.121** |
| shipping planner | 1.171 | 1.073 | 1.527 | 1.969 |

It clears the bar, and holds out. Note the f32 planner is markedly worse than the f64 one, worst
1.969 against 1.495, so there is more headroom here than on f64.

It needs its **own weights**: `seq_l2 3.0, strided 2.5, permuted 4.0` against f64's
`1.5, 1.5, 1.5`. So a weight set is per (machine, element type), not per machine. That is a third
set, and it is the cost of covering f32.

### The f32-specific op counts bought nothing, and the reason is instructive

The f32 butterflies were counted properly from `perform_parallel_fft_direct` (see `OP-COUNTS.md`).
Scored with the **f64** table instead, and weights retuned, the result is the same worst case and a
slightly better mean:

| | best mean | best worst | weights it wants |
|---|---|---|---|
| f32 counts | 1.032 | 1.121 | l2 3.0, strided 2.5, permuted 4.0 |
| f64 counts on f32 data | **1.024** | 1.121 | l2 1.5, strided 1.0, permuted 1.5 |

Both find the same optimum, parameterised differently. The reason is visible in the tables:

| | ratio range | spread | kind of difference |
|---|---|---|---|
| f32 against f64 counts | 0.50 to 1.00 | 2.00x | mostly a uniform **scale** |
| SSE against NEON counts | 1.00 to 1.78 | 1.78x | a change of **shape** |

A uniform scale on the butterfly table cannot change a ranking, because only the balance between
arithmetic and memory matters and the memory weights absorb it. A change of shape can, which is why
swapping NEON counts for SSE ones costs 1.043 -> 1.246 while swapping f64 counts for f32 ones costs
nothing.

**Consequence for a shipping model:** per-element-type op counts are probably not worth maintaining.
One table per instruction set, plus a weight set per (machine, element type), looks sufficient. The
f32 counts are kept in `counted.rs` because they are measured and correct, but the evidence says
they are not earning their maintenance.

## f32 on SSE, and the term that four datasets exposed

Measured on the ThinkCentre, 2026-09-15, same 33 lengths, `--rounds 7 --cap 48`. This was the last
untested cell of the {NEON, SSE} x {f32, f64} grid, and the one flagged in `NEXT-STEPS.md` as risky:
it is where `design_radixn`'s even-base constraints meet `radix4_bases()` filtering on a multiple of
4, and where the f32 base fixup can return None. None of that broke. Candidate counts match the
NEON f32 run closely, and `verify --planner sse --f32` is clean at every length.

| SSE f32, 33-length survey | mean | median | p90 | worst |
|---|---|---|---|---|
| counted model | **1.031** | 1.002 | 1.105 | **1.121** |
| counted model, held-out half | 1.028 | 1.000 | 1.086 | **1.105** |
| shipping planner | 1.207 | 1.152 | 1.490 | 1.927 |

Four for four, then: every backend and element type tested clears the 20% bar, and every one holds
out. As on NEON, the f32 planner is the weaker one, worst 1.927 against f64's 1.734, so f32 is where
the headroom is on both architectures.

It needs its own weights, as f32 did on NEON. The SSE **f64** weights score 1.174 mean and 1.870
worst on f32, barely ahead of the planner, and every visible miss is a `gt(...)`.

### The four best weight sets, on one identical grid

Re-gridding all four datasets over the same 216 points, rather than comparing the numbers each was
originally fitted on, makes a pattern visible that three datasets could not show:

| dataset | l2 | strided | permuted | radixn_extra | mean | worst |
|---|---|---|---|---|---|---|
| NEON f64 | any | 1.5 | **1.5** | 0 | 1.003 | 1.043 |
| NEON f32 | 3.0 | 2.5 | **4.0** | 0 | 1.032 | 1.121 |
| SSE f64 | 2.0 | 2.5 | **1.5** | 5 | 1.052 | 1.152 |
| SSE f32 | 1.5 | 2.5 | **6.0** | 3 | 1.034 | 1.121 |

`permuted` splits by element type, not by backend: 1.5 for both f64 sets, 4.0 to 6.0 for both f32
sets. A weight that tracks the element type across two unrelated instruction sets is not a fitting
artefact, it is a missing term. That is the same diagnostic that found the Rader's index chain and
`radixn_extra`.

### The mechanism, and the fix

`mem()` divided every access count by `complex_per_vector`, on the reasoning that one load moves a
whole vector. That is true of a sequential or strided pass and false of a permuted one. Digit
reversal, Good-Thomas CRT reindexing and Rader's permutation all compute a destination per element,
so there is no contiguous run to fill a vector with. The model was therefore under-charging permuted
traffic by exactly `complex_per_vector`: a factor of 1 at f64, where it is invisible, and a factor
of 2 at f32, where the fitted weight had to absorb it.

Charging permuted passes per complex number instead, which is now the default
(`--permuted-vector` restores the old behaviour):

| dataset | permuted before | after | worst before | after | grid points clearing 20% |
|---|---|---|---|---|---|
| NEON f64 | 1.5 | 1.5 | 1.0427 | 1.0427 | 95/216 -> 95/216 |
| NEON f32 | 4.0 | **2.5** | 1.1213 | 1.1213 | 4/216 -> 3/216 |
| SSE f64 | 1.5 | 1.5 | 1.1515 | 1.1515 | 2/216 -> 2/216 |
| SSE f32 | 6.0 | **2.5** | 1.1210 | 1.1210 | 51/216 -> **117/216** |

Three things to note.

1. **Both f64 datasets are byte-identical**, which they must be, because the factor is 1 there. A
   change that moved them would have been a fudge rather than a correction.
2. **Both f32 datasets move their optimum to the same value, 2.5**, from 4.0 and 6.0 respectively,
   and neither loses any accuracy doing it.
3. **SSE f32 becomes far less weight-sensitive**, from 51 of 216 settings clearing the bar to 117.
   That is the practical payoff: the model stops depending on hitting one weight precisely.

Cross-type transfer improves sharply too, though it does not close: applying each machine's f64
weights to its f32 data goes from 2.427 to 1.770 worst on NEON, and from 1.870 to 1.509 on SSE.

### What this does not do

It does not collapse the weight sets. f32 still wants `permuted 2.5` where f64 wants 1.5, and one
shared set across all four datasets (l2 2.0, strided 2.5, permuted 2.5) gives worst cases of 1.167,
1.324, 1.323 and 1.198, so three of the four miss the bar. **Weights remain per (machine, element
type).** The correction buys a better parameterisation and much more robustness, not a smaller
table.

`radixn_extra` also shrinks with the element type, 5 at f64 against 2 to 3 at f32 on the same
machine and the same backend. That is the direction the register-pressure story predicts: the
penalty is charged per complex element, a spilled xmm register holds two complex f32, so the same
count of spilled vectors amortises over twice the elements. Predicted 2.5, observed 2 to 3. It stays
0 on NEON at both element types.

### The verify gate was wrong for f32, and is now fixed

`verify` compared against a direct DFT computed in the **same** precision as the recipes under test.
A naive DFT sums `len` terms per output, so its own error grows with length, and at f32 it swamps
what is being measured: at len 100000 the f32 reference is off by 1.6e-5 while the recipes it judges
are off by 1.6e-7. Against a flat `1e-6` threshold that produced false failures, first seen as
`10125 FAIL 1.860e-6` on SSE f32. The **scalar** planner, which shares no SIMD code, failed the same
lengths by the same margins, which is what identified the reference rather than the kernels.

The reference is now computed in f64 whatever the recipes use, and the threshold is a budget of two
terms: `20 * eps * sqrt(log2 len)` for the recipe, whose error accumulates over passes, plus
`4 * eps_f64 * sqrt(len)` for the reference, whose error accumulates over elements. With an f64
reference, f32 recipe error comes out flat at 1.3e-7 to 1.6e-7 at every length from 1000 to 100000,
which is `eps_f32` and is the right answer. The same fix removed a false f64 failure at 100000,
where the f64 reference had the identical problem hidden by the loose threshold.

## Caveats

1. **Two machines, and they are confounded.** All four cells of {NEON, SSE} x {f32, f64} now clear
   the bar and hold out, so the idea travels across instruction sets and element types. But the two
   machines sit on the diagonal of {ARM, x86} x {strong, weak memory}, so instruction set and memory
   system cannot be told apart. A third machine is the instrument this needs; see `NEXT-STEPS.md`.
2. **33 adversarial lengths.** They are mixed-factor composites near round numbers, chosen because
   that is where a planner is weakest. Powers of two come out 1.000 for everyone.
3. **Regret is a lower bound.** Candidate inner recipes come from the same planner, so the true
   optimum is better than "best measured" and the real distance from fastest is larger.
4. **The Rader's term is tied to current code.** If the permutation precompute from
   `raders_precompute` lands, that per-element cost drops by roughly 4x and the term must be
   re-derived. That is inherent to a model read from source: it is accurate because it tracks the
   code, and it must be updated when the code changes.
5. **`--cap 48`** trims the candidate list at lengths with hundreds of divisors.
6. **Weights are per (machine, element type).** Four sets for two machines. The `permuted_scalar`
   correction narrowed the f32-to-f64 gap but did not close it, and no single set covers all four.

## Reproducing

```sh
cd tools/planner_tuning && cargo build --release
./target/release/planner_tuning verify --planner neon 1000 1050 1200 1296 720 840 960 1009 1013 97 397
./target/release/planner_tuning dump --planner neon --rounds 7 --cap 48 --out dump_neon_f64.tsv <the 33 lengths>
./target/release/planner_tuning score --seq-l2 1.5 --seq-dram 6.0 --strided 1.5 --permuted 1.5 \
    --rader-index 50 dump_neon_f64.tsv
./target/release/planner_tuning explain 'rad(rn(7.6,rad(rn(7.5.4,b17))))'   # cost tree, to check recursion
./sweep.sh dump_neon_f64.tsv                                               # the 108-point grid
./grid.sh dump_sse_f32.tsv --f32                                           # the 216-point grid
python3 split.py dump_sse_f32.tsv sse_f32_train.tsv sse_f32_test.tsv       # the held-out halves
```

SSE f32, measured on the thinkcentre, then scored anywhere:

```sh
./target/release/planner_tuning verify --planner sse --f32 --cap 48 <lengths>
./target/release/planner_tuning dump --planner sse --f32 --rounds 7 --cap 48 \
    --out dump_sse_f32.tsv <the 33 lengths>
./target/release/planner_tuning score --f32 --seq-l1 1.0 --seq-l2 1.5 --seq-dram 6.0 \
    --strided 2.5 --permuted 2.5 --rader-index 50 --radixn-extra 2 dump_sse_f32.tsv
```

`--f32` is needed on `score` as well as on `dump`: the dump header records the planner, so the
backend is recovered automatically, but it does not record the element type.

`dump` measures once and writes every candidate's time; `score` and `sweep.sh` are pure replay, so
model iteration after the first run needs no machine at all. That is the part that makes this
maintainable where the measured table was not.
