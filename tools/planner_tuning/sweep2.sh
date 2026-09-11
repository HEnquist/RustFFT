#!/bin/sh
BIN=./target/release/planner_tuning
DUMP=$1; shift
for l2 in 1.0 1.5 2.0 3.0; do
for bl in 0.5 1.0 1.5 2.0 3.0; do
for ri in 20 50; do
  line=$($BIN score --seq-l2 $l2 --seq-dram 6.0 --blocked $bl --rader-index $ri "$@" "$DUMP" 2>/dev/null | grep "counted   n=")
  [ -n "$line" ] && echo "l2=$l2 blocked=$bl rader=$ri | $line"
done; done; done
