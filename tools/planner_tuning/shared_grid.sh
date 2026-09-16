#!/bin/sh
# Grid one memory weight set jointly against all four datasets, to find the best set that could be
# shipped if weights turned out not to be splittable per backend. This is the floor, not the
# proposal: see the "floor" section of RESULTS.md.
#
# radixn_extra is allowed to differ per backend, because it is a register-file property (32 v
# registers against 16 xmm) and the backend is known at compile time. Everything else is shared.
#
# Pure replay, needs no machine. Prints "mean/worst" per dataset, in the order
# NEON f64, NEON f32, SSE f64, SSE f32.
BIN=./target/release/planner_tuning
for l2 in 1.5 2.0 3.0; do
for st in 1.5 2.5 4.0; do
for pm in 1.5 2.5 4.0; do
for rx in 0 2 3 5; do
  out=""
  for ds in "dump_neon_f64.tsv::0" "dump_neon_f32.tsv:--f32:0" \
            "dump_sse_f64.tsv::$rx" "dump_sse_f32.tsv:--f32:$rx"; do
    f=$(echo "$ds" | cut -d: -f1)
    fl=$(echo "$ds" | cut -d: -f2)
    r=$(echo "$ds" | cut -d: -f3)
    line=$($BIN score --seq-l1 1.0 --seq-l2 $l2 --seq-dram 6.0 --strided $st --permuted $pm \
                      --rader-index 45 --radixn-extra $r $fl "$f" 2>/dev/null | grep "counted   n=")
    m=$(echo "$line" | sed 's/.*mean \([0-9.]*\).*/\1/')
    w=$(echo "$line" | sed 's/.*worst \([0-9.]*\).*/\1/')
    out="$out $m/$w"
  done
  echo "l2=$l2 st=$st pm=$pm rx_sse=$rx |$out"
done; done; done; done
