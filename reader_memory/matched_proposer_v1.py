"""Blind 2x2 proposer/reader interaction test on the preallocated 48 sources."""
from collections import Counter
from copy import deepcopy
import importlib.util,inspect,json,statistics
from . import role_quote_v1 as base
from . import role_quote_v2 as prompt_source
from . import anchored_quote_v1 as packing_source
ROOT=base.ROOT;BASE=ROOT/'research/matched_proposer_v1';PLAN=BASE/'confirmation';RUN=ROOT/'runs/matched_proposer_v1'
load,sha,immutable,fp,write_json=base.load,base.sha,base.immutable,base.fp,base.write_json
READERS=base.READERS;WRITERS=('Qwen3.5-4B','Qwen3.5-9B');BUDGETS=(256,512)
ARMS=('global','ap3','quote_anchor','quote_cover')
CONFIG=dict(version='matched-proposer-v1',questions=48,writers=list(WRITERS),readers=list(READERS),budgets=list(BUDGETS),arms=list(ARMS),
 primary='Within-source mean over both readers and budgets: matching-size proposer quote_cover minus AP3',
 mechanism='Within-source mean: matching-size proposer quote_cover minus opposite-size proposer quote_cover',
 criterion='Both primary and matching-mechanism descriptive paired bootstrap lower bounds > 0, with positive primary and matching means separately on each reader',
 max_proposer_calls=192,max_answer_conditions=1536,no_training=True,confirmation_selected_before_any_current_method_outputs=True,
 prompt='Exact role_quote_v2 prompt',packing='Exact anchored_quote_v1 rules',proposer_output=2048,reader_output=128,
 resource_caps_mib=dict(retrieve=4000,rank=9000,propose4=13000,propose9=20500,qa4=10500,qa9=19500),
 resource_headroom_mib=dict(retrieve=1000,rank=1000,propose4=1000,propose9=500,qa4=1000,qa9=1000))

def verify():
 m=load(BASE/'manifest.json')
 if m['config']!=CONFIG or fp({k:v for k,v in m.items() if k!='signature'})!=m['signature']:raise ValueError('Manifest identity')
 for n,d in m['files'].items():
  if sha(ROOT/n)!=d:raise ValueError('Frozen file changed: '+n)
 return m['signature']

def runtime(writer,shared=False):
 if writer not in WRITERS:raise ValueError('Unknown writer')
 spec=importlib.util.spec_from_file_location('reader_memory._matched_runtime_'+writer[-2:],ROOT/'reader_memory/role_quote_v1.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.BASE,m.PLAN,m.PANEL=BASE,PLAN,'confirmation'
 m.RUN=RUN/'shared' if shared else RUN/writer
 m.CONFIG=dict(packing_source.CONFIG,version=CONFIG['version'],proposer=writer,arms=list(ARMS))
 m.ARMS,m.PROMPT,m.verify,m.quote_plan=ARMS,prompt_source.PROMPT,verify,packing_source.quote_plan
 m.SHARED_CACHE=RUN/'request_cache'
 source=inspect.getsource(base.qa)
 original="cache=RUN/'request_cache'/reader/(key+'.json')"
 if original not in source:raise ValueError('QA cache hook changed')
 source=source.replace(original,"cache=SHARED_CACHE/reader/(key+'.json')")
 exec(compile(source,str(__file__)+':qa','exec'),m.__dict__)
 source=inspect.getsource(base.assemble).replace("('ap3','quote_flat','sentence_cover')","('ap3','quote_anchor')")
 exec(compile(source,str(__file__)+':assemble','exec'),m.__dict__)
 return m

def preflight():
 verify()
 for writer in WRITERS:runtime(writer).preflight()

def assemble():
 sid=verify()
 for writer in WRITERS:
  m=runtime(writer)
  for folder in ('retrieval','ranking'):
   for s in m.sources():
    row=load(RUN/'shared'/folder/(s['question_id']+'.json'))
    if row['signature']!=sid or row['source_hash']!=fp(s):raise ValueError('Shared input identity')
    immutable(m.RUN/folder/(s['question_id']+'.json'),row)
  m.assemble()
 plans=[load(RUN/w/'plans.json')['plans'] for w in WRITERS]
 for arm in ('global','ap3'):
  a={(p['question_id'],p['budget']):p['evidence_text'] for p in plans[0] if p['arm']==arm}
  b={(p['question_id'],p['budget']):p['evidence_text'] for p in plans[1] if p['arm']==arm}
  if a!=b:raise ValueError('Baseline differs by writer')
 immutable(RUN/'selection_complete.json',dict(signature=sid,plans=sum(map(len,plans)),both_writers_frozen_before_any_qa=True,baselines_identical=True))

def qa(reader):
 if load(RUN/'selection_complete.json')['signature']!=verify():raise ValueError('Unfrozen selection')
 for writer in WRITERS:runtime(writer).qa(reader)

def score():
 from .full_comparison_v1 import answer_scores
 common=base.old.common;sid=verify();refs=load(PLAN/'private/references.json');records=[]
 for writer in WRITERS:
  for reader in READERS:
   for p in load(RUN/writer/'plans.json')['plans']:
    q=p['question_id'];r=load(RUN/writer/'responses'/reader/(fp([reader,q,p['budget'],p['arm']])+'.json'))
    if r['signature']!=sid or r['plan_hash']!=fp(p):raise ValueError('Response identity')
    em,f1=answer_scores(r['text'],refs[q]);records.append(dict(writer=writer,reader=reader,question_id=q,budget=p['budget'],arm=p['arm'],
     em=em,f1=f1,hops=refs[q]['hops'],status=r['status'],reused=r.get('reused',False),usage=r['usage'],evidence_tokens=p['evidence_tokens'],finish_reason=r.get('finish_reason')))
 idx={(r['writer'],r['reader'],r['question_id'],r['budget'],r['arm']):r for r in records}
 if len(records)!=CONFIG['max_answer_conditions'] or len(idx)!=len(records):raise ValueError('Incomplete matrix')
 other=lambda reader:next(w for w in WRITERS if w!=reader)
 def value(writer,reader,q,b,arm):return idx[writer,reader,q,b,arm]['f1']
 primary_values=[statistics.mean(value(r,r,q,b,'quote_cover')-value(r,r,q,b,'ap3') for r in READERS for b in BUDGETS) for q in refs]
 mechanism_values=[statistics.mean(value(r,r,q,b,'quote_cover')-value(other(r),r,q,b,'quote_cover') for r in READERS for b in BUDGETS) for q in refs]
 primary=common.paired(primary_values);mechanism=common.paired(mechanism_values)
 groups=[];pairs=[];per_reader={};per_reader_matching={}
 for reader in READERS:
  per_reader[reader]=common.paired([statistics.mean(value(reader,reader,q,b,'quote_cover')-value(reader,reader,q,b,'ap3') for b in BUDGETS) for q in refs])
  per_reader_matching[reader]=common.paired([statistics.mean(value(reader,reader,q,b,'quote_cover')-value(other(reader),reader,q,b,'quote_cover') for b in BUDGETS) for q in refs])
  for writer in WRITERS:
   for budget in BUDGETS+('pooled',):
    bs=BUDGETS if budget=='pooled' else (budget,)
    for arm in ARMS:
     rs=[r for r in records if r['writer']==writer and r['reader']==reader and r['budget'] in bs and r['arm']==arm]
     groups.append(dict(writer=writer,reader=reader,budget=budget,arm=arm,f1_points=100*statistics.mean(r['f1'] for r in rs),em_points=100*statistics.mean(r['em'] for r in rs),mean_tokens=statistics.mean(r['evidence_tokens'] for r in rs)))
    for control in ('ap3','global','quote_anchor'):
     pairs.append(dict(writer=writer,reader=reader,budget=budget,contrast='quote_cover minus '+control,
      **common.paired([statistics.mean(value(writer,reader,q,b,'quote_cover')-value(writer,reader,q,b,control) for b in bs) for q in refs])))
 calls=[r for r in records if not r['reused'] and r['status'] in ('ok','empty')]
 confirmed=primary['ci95'][0]>0 and mechanism['ci95'][0]>0 and all(v['mean_delta_f1']>0 for v in [*per_reader.values(),*per_reader_matching.values()])
 summary=dict(signature=sid,complete=True,conditions=len(records),records=records,primary=primary,matching_mechanism=mechanism,per_reader_primary=per_reader,per_reader_matching=per_reader_matching,
  confirmation_criterion_met=confirmed,group_means=groups,paired_comparisons=pairs,status_counts=dict(Counter(r['status'] for r in records)),
  finish_reasons=dict(Counter(r['finish_reason'] for r in records)),unique_new_reader_calls=len(calls),
  reader_costs={k:sum(r['usage'].get(k,0) for r in calls) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
  proposal_gates={w:load(RUN/w/'proposal_gate.json') for w in WRITERS},
  caveats='First blind test of a development-observed model-matching hypothesis; 48 preallocated train-file sources, 24/23/1 hops. Two same-family model sizes. Descriptive unadjusted source bootstrap intervals. Not an official test.')
 immutable(RUN/'summary.json',summary)
 lines=['# Matched proposer: blind 48-source test','',summary['caveats'],'','|Writer|Reader|Arm|F1|Tokens|','|---|---|---|---:|---:|']
 for g in groups:
  if g['budget']=='pooled':lines.append(f"|{g['writer']}|{g['reader']}|{g['arm']}|{g['f1_points']:.4f}|{g['mean_tokens']:.2f}|")
 lines+=['','Primary: '+json.dumps(primary),'Matching: '+json.dumps(mechanism),'Confirmed: '+str(confirmed)]
 (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('SCORE_COMPLETE',json.dumps(dict(primary=primary,mechanism=mechanism,confirmed=confirmed)),flush=True)

def phase(name):
 if name=='preflight':preflight()
 elif name in ('retrieve','rank'):getattr(runtime(WRITERS[0],True),name)()
 elif name in ('propose4','propose9'):runtime(WRITERS[0 if name=='propose4' else 1]).propose()
 elif name in ('qa4','qa9'):qa('Qwen3.5-'+('4B' if name=='qa4' else '9B'))
 elif name in ('assemble','score'):globals()[name]()
 else:raise ValueError('Unknown phase')
