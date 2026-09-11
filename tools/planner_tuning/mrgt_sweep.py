import collections, subprocess
exec(open('mrgt.py').read().split('NEON=')[0])   # reuse toplevel/load/pairs
def acc(dumps,args):
    data=collections.defaultdict(dict)
    for d in dumps:
        for L,r in load(d,args).items(): data[L].update(r)
    ok=n=0
    for L,key,mr,gt in pairs(data):
        n+=1
        if (('gt' if gt[0]<mr[0] else 'mr') == ('gt' if gt[1]<mr[1] else 'mr')): ok+=1
    return ok,n
for name,dumps,extra in (('NEON',['dump_neon_f64.tsv','dump_neon_primes.tsv'],['--seq-l2','1.5']),
                         ('SSE', ['dump_sse_f64.tsv','dump_sse_primes.tsv'],['--seq-l2','2.0','--radixn-extra','5'])):
    print(f'=== {name}: MR-vs-GT accuracy over strided (blocked transpose) x permuted (scatter) ===')
    print('        permuted:  ' + '  '.join(f'{p:>5}' for p in ('1.0','1.5','2.5','4.0','6.0','9.0')))
    for st in ('1.0','1.5','2.5'):
        row=[]
        for pm in ('1.0','1.5','2.5','4.0','6.0','9.0'):
            a=['--seq-dram','6.0','--strided',st,'--permuted',pm,'--rader-index','50']+extra
            ok,n=acc(dumps,a); row.append(f'{100*ok//n:>4}%')
        print(f'  strided={st}:  ' + '  '.join(f'{v:>5}' for v in row))
