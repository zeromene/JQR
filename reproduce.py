"""CPU reproduction of the paper's released records and evidence construction."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,statistics,subprocess,sys
from collections import Counter

ROOT=Path(__file__).resolve().parent
def load(name):
    p=ROOT/name
    with (gzip.open(p,'rt',encoding='utf-8') if p.suffix=='.gz' else p.open(encoding='utf-8')) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_files():
    hashes=load('MANIFEST.json')['files']
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest:raise ValueError('File identity mismatch: '+name)
    print('Verified',len(hashes),'release files.',flush=True)

def summarize(bootstrap=False):
    data=load('artifacts/budget_analysis.json.gz');rows=data['records']
    old=load('artifacts/system_analysis.json.gz')
    idx={(r['dataset'],r['reader'],r['question_id'],r['budget'],r['method']):r for r in rows}
    assert len(idx)==len(rows)==30960
    replays=0
    for r in old['records']:
        if r['study']=='complete' and r['method'] in ('quote_anchor','strong_rerank'):
            v=idx[r['dataset'],r['reader'],r['question_id'],512,r['method']]
            assert all(abs(v[k]-r[k])<1e-12 for k in ['em','f1']);replays+=1
    assert replays==7740
    for g in data['groups']:
        rr=[r for r in rows if all(r[k]==g[k] for k in ['dataset','reader','budget','method'])]
        assert len(rr)==g['n'] and dict(Counter(r['status'] for r in rr))==g['statuses']
        for k in ['f1','em']:assert abs(100*statistics.mean(r[k] for r in rr)-g[k])<1e-10
        for k in ['evidence_tokens','actual_reader_evidence_tokens','prompt_tokens']:
            assert abs(statistics.mean(r.get(k,0) for r in rr)-g[k])<1e-10
    if bootstrap:recompute_intervals(data,idx)
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    with (out/'budget_scores.csv').open('w',newline='',encoding='utf-8') as f:
        keys=['dataset','reader','budget','method','n','f1','em','evidence_tokens','valid_percent']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:g[k] for k in keys} for g in data['groups'])
    # Public-panel score records are independent of the system-panel records.
    figures=load('figures/figure_data.json')
    for name,records in figures['public_records'].items():
        if isinstance(records,list) and records:
            assert all(0<=r['f1']<=1 and 0<=r['em']<=1 for r in records)
    report=dict(budget_records=len(rows),system_records=len(old['records']),original_512_replays=replays,
                absolute_budget_groups=32,bootstrap_recomputed=bootstrap,new_model_calls=0)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

def recompute_intervals(data,idx):
    import numpy as np
    from reader_memory.budget_sweep_core_v1 import bootstrap_matrix,interval_family
    audit=load('artifacts/budget_raw_completion_audit.json.gz')
    meta={r['question_id']:r for r in audit['cluster_metadata']}
    readers=['Qwen3.5-9B','Phi-4-mini-instruct'];budgets=[128,256,384,512]
    ests=[];boots=[];ies=[];ibs=[]
    for di,ds in enumerate(['hotpot','musique']):
        ids=sorted(q for q,r in meta.items() if r['dataset']==ds)
        values=[[idx[ds,r,q,b,'quote_anchor'][m]-idx[ds,r,q,b,'strong_rerank'][m]
                 for m in ['f1','em'] for r in readers for b in budgets] for q in ids]
        est,boot=bootstrap_matrix(values,[meta[q]['cluster'] for q in ids],10000,30+di)
        cases=[c for c in data['comparisons'] if c['dataset']==ds]
        for i,c in enumerate(cases):
            for offset,m in [(0,'f1'),(8,'em')]:
                np.testing.assert_allclose(c[m]['delta_points'],100*est[i+offset],atol=1e-9,rtol=0)
                np.testing.assert_allclose(c[m]['ci95_points'],100*np.quantile(boot[:,i+offset],[.025,.975]),atol=1e-9,rtol=0)
        pairs=[(i,i//4*4+3) for i in range(8) if i%4!=3]
        ie=np.array([est[i]-est[j] for i,j in pairs]);ib=np.column_stack([boot[:,i]-boot[:,j] for i,j in pairs])
        for i,c in enumerate(c for c in data['interactions'] if c['dataset']==ds):
            np.testing.assert_allclose(c['f1']['delta_points'],100*ie[i],atol=1e-9,rtol=0)
            np.testing.assert_allclose(c['f1']['ci95_points'],100*np.quantile(ib[:,i],[.025,.975]),atol=1e-9,rtol=0)
        ests.append(est[:8]);boots.append(boot[:,:8]);ies.append(ie);ibs.append(ib)
    for ee,bb,cc in [(ests,boots,data['comparisons']),(ies,ibs,data['interactions'])]:
        found,_=interval_family(np.concatenate(ee),np.concatenate(bb,axis=1))
        for a,c in zip(found,cc):np.testing.assert_allclose(a['simultaneous_ci95_points'],c['f1']['simultaneous_ci95_points'],atol=1e-9,rtol=0)
    print('Reproduced all 16 gain intervals and all 12 budget-interaction intervals.',flush=True)

def construct(tokenizer,limit=0):
    from tokenizers import Tokenizer
    from reader_memory.budget_sweep_core_v1 import construct as build
    from reader_memory.schema import fingerprint
    expected=load('environment/tokenizer.json')['sha256']
    if sha(tokenizer)!=expected:raise ValueError('Tokenizer does not match the frozen tokenizer hash')
    tok=Tokenizer.from_file(str(tokenizer))
    class Count:
        exact=True
        def __init__(self):self.memo={}
        def count(self,text):
            if text not in self.memo:self.memo[text]=len(tok.encode(text,add_special_tokens=False).ids)
            return self.memo[text]
    count=Count();done=0
    with gzip.open(ROOT/'artifacts/construction_inputs.jsonl.gz','rt',encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            x=json.loads(line);assert fingerprint(x['source'])==x['source_hash'];count.memo.clear()
            for b in (128,256,384,512):
                plans,trace=build(x['source'],x['scores'],x['seeds'],x['proposal'],count,b)
                target=x['expected'][str(b)]
                assert trace['admitted']==target['admitted']
                assert trace['effective_budget']==target['effective_budget']
                for name,p in plans.items():
                    assert fingerprint(p['evidence_text'])==target['methods'][name]['evidence_sha256']
                    assert p['evidence_tokens']==target['methods'][name]['evidence_tokens']
                done+=1
            if i%100==0:print('Reconstructed',i,'questions;',done,'paired budget conditions.',flush=True)
            if limit and i>=limit:break
    print('Exact evidence reconstruction passed:',done,'pairs;',done*2,'contexts.',flush=True)

def figures():
    dest=ROOT/'outputs/figures';dest.mkdir(parents=True,exist_ok=True)
    import importlib.util
    spec=importlib.util.spec_from_file_location('jqr_plot',ROOT/'figures/render_readable.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def save(fig,name,title):
        fig.savefig(dest/(name+'.pdf'),metadata={'Title':title,'Creator':'Matplotlib'})
        m.plt.close(fig)
    m.save=save
    for fn in [m.hops,m.budgets,m.raincloud,m.flows,m.mosaics,m.policy,m.paths]:fn()
    print('Rendered seven empirical vector PDF figures to outputs/figures.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','statistics','construct','figures'])
    p.add_argument('--bootstrap',action='store_true')
    p.add_argument('--tokenizer',type=Path,default=ROOT/'tokenizers/qwen3.5-9b.json')
    p.add_argument('--limit',type=int,default=0,help='0 means every question; use a positive value for a smoke test')
    a=p.parse_args();verify_files()
    if a.action in ['verify','statistics']:summarize(a.bootstrap)
    elif a.action=='construct':
        if not a.tokenizer:p.error('--tokenizer must point to the frozen Qwen tokenizer.json')
        construct(a.tokenizer,a.limit)
    else:figures()
