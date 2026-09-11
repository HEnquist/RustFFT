import collections, subprocess, sys
def toplevel(spec):
    i=spec.find('(')
    if i<0: return None
    k=spec[:i]
    if k not in ('mr','mrs','gt','gts'): return None
    d=0
    for j in range(i+1,len(spec)):
        c=spec[j]
        if c=='(': d+=1
        elif c==')':
            if d==0: break
            d-=1
        elif c==',' and d==0: return k,spec[i+1:j],spec[j+1:-1]
    return None
def load(dump,args):
    out=subprocess.run(['./target/release/planner_tuning','costs',*args,dump],capture_output=True,text=True).stdout
    d=collections.defaultdict(dict)
    for line in out.splitlines()[1:]:
        f=line.split('\t')
        if len(f)<5 or f[3]=='NA': continue
        d[int(f[0])][f[1]]=(float(f[2]),float(f[3]))
    return d
def pairs(data):
    for L,rows in data.items():
        by=collections.defaultdict(dict)
        for s,(ns,c) in rows.items():
            t=toplevel(s)
            if t: by[(t[1],t[2])][t[0]]=(ns,c)
        for key,k in by.items():
            mr=k.get('mr') or k.get('mrs'); gt=k.get('gt') or k.get('gts')
            if mr and gt: yield L,key,mr,gt
def report(name,dumps,args):
    data=collections.defaultdict(dict)
    for d in dumps:
        for L,r in load(d,args).items(): data[L].update(r)
    truth_gt=model_gt=ok=0; n=0
    wrong_dir=collections.Counter()
    for L,key,mr,gt in pairs(data):
        n+=1
        t='gt' if gt[0]<mr[0] else 'mr'
        g='gt' if gt[1]<mr[1] else 'mr'
        truth_gt += t=='gt'; model_gt += g=='gt'
        if t==g: ok+=1
        else: wrong_dir[f'model said {g}, truth {t}']+=1
    print(f'{name}: {ok}/{n} correct | truth prefers GT {truth_gt}/{n} | model prefers GT {model_gt}/{n}')
    for k,v in wrong_dir.most_common(): print(f'    {k}: {v}')
NEON=['--seq-l2','1.5','--seq-dram','6.0','--strided','1.5','--permuted','1.5','--rader-index','50']
SSE =['--seq-l2','2.0','--seq-dram','6.0','--strided','2.5','--permuted','1.5','--rader-index','50','--radixn-extra','5']
report('NEON',['dump_neon_f64.tsv','dump_neon_primes.tsv'],NEON)
report('SSE ',['dump_sse_f64.tsv','dump_sse_primes.tsv'],SSE)
