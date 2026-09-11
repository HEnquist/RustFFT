import collections, subprocess, statistics
exec(open('mrgt.py').read().split('NEON=')[0])
NEON=['--seq-l2','1.5','--seq-dram','6.0','--strided','1.5','--permuted','1.5','--rader-index','50']
SSE =['--seq-l2','2.0','--seq-dram','6.0','--strided','2.5','--permuted','1.5','--rader-index','50','--radixn-extra','5']
for name,dumps,args in (('NEON',['dump_neon_f64.tsv','dump_neon_primes.tsv'],NEON),
                        ('SSE', ['dump_sse_f64.tsv','dump_sse_primes.tsv'],SSE)):
    data=collections.defaultdict(dict)
    for d in dumps:
        for L,r in load(d,args).items(): data[L].update(r)
    meas=[]; mod=[]; per=[]
    for L,key,mr,gt in pairs(data):
        meas.append(gt[0]/mr[0]); mod.append(gt[1]/mr[1])
        per.append((L, gt[0]/mr[0], gt[1]/mr[1], mr[1]))
    print(f'=== {name} ===  {len(meas)} MR/GT pairs')
    print(f'  measured gt/mr : median {statistics.median(meas):.4f}  range {min(meas):.3f}-{max(meas):.3f}')
    print(f'  model    gt/mr : median {statistics.median(mod):.4f}  range {min(mod):.3f}-{max(mod):.3f}')
    # correction needed: multiply GT cost by k so the model's median ratio matches measured median
    k = statistics.median(meas)/statistics.median(mod)
    print(f'  -> GT cost would need multiplying by {k:.4f} to centre the model on the truth')
    # how much extra cost per element is that?
    extra = statistics.median([(k-1)*m for _,_,_,m in [(a,b,c,d) for a,b,c,d in per]])
    print(f'  spread of measured ratios (p10..p90): '
          f'{sorted(meas)[len(meas)//10]:.3f}..{sorted(meas)[9*len(meas)//10]:.3f}')
