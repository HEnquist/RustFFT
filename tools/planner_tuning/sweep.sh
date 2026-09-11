#!/bin/sh
# Grid-search the counted model's math-versus-memory weights against a frozen dump.
# Pure replay: no measuring, no planner, so this needs no machine and takes seconds.
BIN=./target/release/planner_tuning
DUMP=${1:-dump_neon_f64.tsv}
for l1 in 1.0; do
for l2 in 1.5 2.0 3.0; do
for dram in 3.0 6.0 10.0 16.0; do
for st in 1.0 1.5 2.5; do
for pm in 1.5 2.5 4.0; do
  line=$($BIN score --seq-l1 $l1 --seq-l2 $l2 --seq-dram $dram \
                    --strided $st --permuted $pm --rader-index 50 "$DUMP" 2>/dev/null \
         | grep "counted   n=")
  [ -n "$line" ] && echo "l2=$l2 dram=$dram strided=$st permuted=$pm | $line"
done; done; done; done; done
