import collections, subprocess
exec(open('mrgt.py').read().split('NEON=')[0])
def stats(dumps,args):
    data=collections.defaultdict(dict)
    for d in dumps:
        for L,r in load(d,args).items(): data[L].update(r)
    ok=n=mgt=tgt=0
    for L,key,mr,gt in pairs(data):
        n+=1
        t='gt' if gt[0]<mr[0] else 'mr'; g='gt' if gt[1]<mr[1] else 'mr'
        tgt+= t=='gt'; mgt+= g=='gt'; ok+= t==g
    return ok,n,tgt,mgt
for name,dumps,extra in (('NEON',['dump_neon_f64.tsv','dump_neon_primes.tsv'],['--seq-l2','1.5']),
                         ('SSE', ['dump_sse_f64.tsv','dump_sse_primes.tsv'],['--seq-l2','2.0','--radixn-extra','5'])):
    print(f'=== {name}: MR vs GT accuracy over blocked-transpose multiplier ===')
    for bl in ('0.5','1.0','1.5','2.0','3.0','4.0'):
        a=['--seq-dram','6.0','--blocked',bl,'--rader-index','50']+extra
        ok,n,tgt,mgt=stats(dumps,a)
        print(f'  blocked={bl:>4}:  {ok:>3}/{n} correct ({100*ok//n:>3}%)   truth prefers GT {tgt}, model prefers GT {mgt}')
