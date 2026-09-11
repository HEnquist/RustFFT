import collections, subprocess, sys
def lpf(n):
    m=1; f=2
    while f*f<=n:
        while n%f==0: n//=f; m=max(m,f)
        f+=1
    return max(m,n)

def load(dump):
    rows=collections.defaultdict(dict); pick={}
    for l in open(dump):
        if l.startswith('#') or l.startswith('len\t'): continue
        f=l.rstrip('\n').split('\t')
        L,spec,ns,ps,isp=int(f[0]),f[1],float(f[2]),int(f[3]),f[4]=='1'
        if not (spec in rows[L] and ps==1): rows[L][spec]=ns
        if isp: pick[L]=spec
    return rows,pick

def model_picks(dump, args):
    out=subprocess.run(['./target/release/planner_tuning','score',*args,dump],
                       capture_output=True,text=True).stdout
    res={}
    for line in out.splitlines():
        p=line.split()
        if len(p)>=4 and p[0].isdigit():
            res[int(p[0])]=(float(p[1].rstrip('x')), ' '.join(p[3:]))
    return res

NEON=['--seq-l2','1.5','--seq-dram','6.0','--blocked','1.5','--rader-index','50']
SSE =['--seq-l2','2.0','--seq-dram','6.0','--blocked','2.5','--rader-index','50','--radixn-extra','5']

for name,dump,args in (('NEON','dump_neon_primes.tsv',NEON),('SSE','dump_sse_primes.tsv',SSE)):
    rows,pick=load(dump); mp=model_picks(dump,args)
    print(f'===================== {name} =====================')
    print(f'{"len":>7} {"lpf":>6}  {"truth":<10} {"margin":>7}  {"planner":<10} {"model":<10}  {"m.regret":>8}')
    agree=dis=0; near=0
    for L in sorted(rows, key=lambda x:(len(str(x)),x)):
        r=rows[L]
        rad={k:v for k,v in r.items() if k.startswith('rad(')}
        bs ={k:v for k,v in r.items() if k.startswith('bs(')}
        best=min(r,key=r.get)
        truth='RADERS' if best.startswith('rad(') else 'BLUESTEIN'
        margin=(min(bs.values())/min(rad.values())) if rad and bs else float('nan')
        pl='RADERS' if pick[L].startswith('rad(') else 'BLUESTEIN'
        mreg,mname=mp[L]
        mkind = truth if mname.startswith('= best') else ('RADERS' if mname.startswith('rad(') else 'BLUESTEIN')
        ok = '' if mkind==truth else '  <-- WRONG KIND'
        close = ' (near tie)' if 0.95<margin<1.05 else ''
        if 0.95<margin<1.05: near+=1
        if mkind==truth: agree+=1
        else: dis+=1
        print(f'{L:>7} {lpf(L-1):>6}  {truth:<10} {margin:>7.3f}  {pl:<10} {mkind:<10}  {mreg:>7.3f}x{ok}{close}')
    print(f'  model picks the right family at {agree} of {agree+dis}; {near} of those are near ties (within 5%)')
    wrongplanner=sum(1 for L in rows if (('RADERS' if pick[L].startswith('rad(') else 'BLUESTEIN') !=
        ('RADERS' if min(rows[L],key=rows[L].get).startswith('rad(') else 'BLUESTEIN')))
    print(f'  planner picks the wrong family at {wrongplanner} of {len(rows)}')
