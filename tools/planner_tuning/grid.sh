#!/bin/sh
# One identical grid over all four datasets, so weight sets can be compared like with like.
BIN=./target/release/planner_tuning
DUMP=$1; shift
for l2 in 1.5 2.0 3.0; do
for st in 1.5 2.5 4.0; do
for pm in 1.5 2.5 4.0 6.0; do
for rx in 0 1 2 3 5 8; do
  line=$($BIN score --seq-l1 1.0 --seq-l2 $l2 --seq-dram 6.0 --strided $st --permuted $pm \
                    --rader-index 50 --radixn-extra $rx "$@" "$DUMP" 2>/dev/null | grep "counted   n=")
  [ -n "$line" ] && echo "l2=$l2 st=$st pm=$pm rx=$rx | $line"
done; done; done; done
