import collections, subprocess, sys, math

def toplevel(spec):
    """('mr', 'A', 'B') for a two-way split, else None."""
    i = spec.find('(')
    if i < 0: return None
    kind = spec[:i]
    if kind not in ('mr','mrs','gt','gts'): return None
    depth=0
    for j in range(i+1, len(spec)):
        c=spec[j]
        if c=='(': depth+=1
        elif c==')':
            if depth==0: break
            depth-=1
        elif c==',' and depth==0:
            return kind, spec[i+1:j], spec[j+1:-1]
    return None

def load(dump, args):
    out = subprocess.run(['./target/release/planner_tuning','costs',*args,dump],
                         capture_output=True, text=True).stdout
    data = collections.defaultdict(dict)
    for line in out.splitlines()[1:]:
        f = line.split('\t')
        if len(f) < 5 or f[3]=='NA': continue
        data[int(f[0])][f[1]] = (float(f[2]), float(f[3]))
    return data

NEON=['--seq-l2','1.5','--seq-dram','6.0','--strided','1.5','--permuted','1.5','--rader-index','50']
SSE =['--seq-l2','2.0','--seq-dram','6.0','--strided','2.5','--permuted','1.5','--rader-index','50','--radixn-extra','5']

for name, dumps, args in (('NEON', ['dump_neon_f64.tsv','dump_neon_primes.tsv'], NEON),
                          ('SSE',  ['dump_sse_f64.tsv','dump_sse_primes.tsv'], SSE)):
    data = collections.defaultdict(dict)
    for d in dumps:
        for L, rows in load(d, args).items(): data[L].update(rows)

    # --- 1. MR vs GT on the identical factor pair ---
    ok=bad=0; losses=[]
    for L, rows in data.items():
        byargs = collections.defaultdict(dict)
        for spec,(ns,cost) in rows.items():
            t = toplevel(spec)
            if t: byargs[L, t[1], t[2]][t[0]] = (ns, cost, spec)
        for key, kinds in byargs.items():
            mr = kinds.get('mr') or kinds.get('mrs')
            gt = kinds.get('gt') or kinds.get('gts')
            if not (mr and gt): continue
            truth = 'gt' if gt[0] < mr[0] else 'mr'
            guess = 'gt' if gt[1] < mr[1] else 'mr'
            if truth == guess: ok+=1
            else:
                bad+=1
                losses.append(max(mr[0],gt[0])/min(mr[0],gt[0]))
    lm = f", mean cost of a wrong call {sum(losses)/len(losses):.3f}x, worst {max(losses):.3f}x" if losses else ""
    print(f'=== {name} ===')
    print(f'  MixedRadix vs GoodThomas on the same factor pair: {ok}/{ok+bad} correct{lm}')

    # --- 2. choosing among two-way splits ---
    regrets=[]; order_ties=0; order_real=0
    for L, rows in data.items():
        splits = {s:v for s,v in rows.items() if toplevel(s)}
        if len(splits) < 2: continue
        best_ns = min(v[0] for v in splits.values())
        pick = min(splits, key=lambda s: splits[s][1])
        regrets.append(splits[pick][0]/best_ns)
        # argument order: mr(A,B) vs mr(B,A)
        for s,(ns,cost) in splits.items():
            k,a,b = toplevel(s)
            rev = f'{k}({b},{a})'
            if rev in splits and s < rev:
                r = max(ns, splits[rev][0]) / min(ns, splits[rev][0])
                if r > 1.02: order_real += 1
                if abs(cost - splits[rev][1]) < 1e-9: order_ties += 1
    regrets.sort()
    print(f'  choosing among two-way splits ({len(regrets)} lengths): '
          f'mean {sum(regrets)/len(regrets):.4f}  median {regrets[len(regrets)//2]:.4f}  worst {regrets[-1]:.4f}')
    print(f'  argument order: {order_real} reversed pairs differ by >2% when measured; '
          f'model ties {order_ties} of them exactly')
