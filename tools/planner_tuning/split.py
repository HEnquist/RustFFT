#!/usr/bin/env python3
"""Split a dump into held-out halves: even-indexed lengths train, odd-indexed test.

The split is on numerically sorted length, so each half spans the whole size range and neither
gets all the small transforms. usage: split.py DUMP TRAIN_OUT TEST_OUT
"""
import sys

dump, train_out, test_out = sys.argv[1:4]
head, rows = [], []
for line in open(dump):
    if line.startswith('#') or line.startswith('len\t'):
        head.append(line)
    elif line.strip():
        rows.append(line)

lengths = sorted({int(r.split('\t')[0]) for r in rows})
train = {l for i, l in enumerate(lengths) if i % 2 == 0}

for path, keep in ((train_out, train), (test_out, set(lengths) - train)):
    with open(path, 'w') as f:
        f.writelines(head)
        f.writelines(r for r in rows if int(r.split('\t')[0]) in keep)
    print(f"{path}: {len(keep)} lengths")
