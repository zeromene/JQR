"""Joint quote retention with a reranker anchor and AP3 context completion."""
from collections import Counter
from copy import deepcopy
import statistics
from . import role_quote_v2 as parent
from .conjunctive_quote_plan import joint_plan

ROOT=parent.ROOT;BASE=ROOT/'research/anchored_quote_v1';PANEL=parent.PANEL
PLAN=BASE/PANEL;RUN=ROOT/'runs/anchored_quote_v1'/PANEL
load,sha,immutable,fp,write_json=parent.load,parent.sha,parent.immutable,parent.fp,parent.write_json
READERS=parent.READERS;BUDGETS=(256,512)
ARMS=('global','ap3','quote_or','quote_joint','quote_anchor','quote_cover')
CONFIG=dict(parent.CONFIG,version='anchored-quote-v1',arms=list(ARMS),
 primary='9B quote_cover minus ap3, source mean over both budgets',
 mechanism_controls=['quote_cover minus quote_or','quote_cover minus quote_joint','quote_cover minus quote_anchor'],
 stopping='Joint exact quotes plus the original global retained anchor; complete only with sentences from the original AP3 context.',
 fallback='Unchanged AP3 if any proposed role has no quote, or joint core plus anchor exceeds the cap.',
 development_proposals='Exact public V2 proposals reused; no further LLM generation or feedback-conditioned edits')


def verify():
 m=load(BASE/'manifest.json')
 if m['config']!=CONFIG or fp({k:v for k,v in m.items() if k!='signature'})!=m['signature']:raise ValueError('Manifest identity')
 for n,d in m['files'].items():
  if sha(ROOT/n)!=d:raise ValueError('Frozen file changed: '+n)
 return m['signature']


def quote_plan(s,scores,baseline,ap3,proposal,counter,arm):
 if arm=='quote_or':return parent.quote_plan(s,scores,baseline,ap3,proposal,counter,'quote_cover')
 core,trace=joint_plan(s,scores,baseline,ap3,proposal,counter)
 if arm=='quote_joint' or trace['fallback']:return core,trace
 if arm not in ('quote_anchor','quote_cover'):raise ValueError('Unknown arm')
 from .anchor_guard_controls_v1 import packing
 from .relation_preserving_pilot_v1 import merge_ranges,render_ranges
 by={p['source_id']:p for p in s['paragraphs']};order=packing.sentence_order(s,scores);ranks={uid:i for i,uid in enumerate(order)}
 def render(ranges):
  ranges={pid:merge_ranges(by[pid]['text'],rr) for pid,rr in ranges.items()}
  def key(pid):return min(ranks[pid+':span:'+str(i)] for i,sp in enumerate(by[pid]['spans']) if any(a<sp['end'] and b>sp['start'] for a,b in ranges[pid]))
  ranges={pid:ranges[pid] for pid in sorted(ranges,key=key)}
  text=render_ranges(s,ranges);return text,counter.count(text),ranges
 def extend(ranges,uid):
  ranges=deepcopy(ranges);pid,_,i=uid.split(':');sp=by[pid]['spans'][int(i)]
  ranges.setdefault(pid,[]).append([sp['start'],sp['end']]);return ranges
 anchor=next(uid for uid in order if uid in set(baseline['selected_spans']))
 ranges=extend(core['selected_ranges'],anchor);text,tokens,ranges=render(ranges)
 trace=dict(trace,anchor=anchor,fill=arm=='quote_cover',representation=arm)
 if tokens>baseline['evidence_tokens']:return deepcopy(ap3),dict(trace,fallback='joint_anchor_over_budget',joint_anchor_tokens=tokens)
 seed=tokens;fill=[]
 if arm=='quote_cover':
  retained=set(ap3['selected_spans'])
  for uid in order:
   if uid not in retained:continue
   candidate=extend(ranges,uid);nt,nc,nr=render(candidate)
   if nc<=baseline['evidence_tokens']:
    if nt!=text:fill.append(uid)
    text,tokens,ranges=nt,nc,nr
 return dict(evidence_text=text,evidence_tokens=tokens,selected_ranges=ranges,selected_spans=[]),dict(trace,seed_tokens=seed,fill_spans=fill,fallback=None)


def score():
 from .full_comparison_v1 import answer_scores
 common=parent.parent.old.common
 sid=verify();refs=load(PLAN/'private/references.json');plans=load(RUN/'plans.json')['plans'];records=[]
 for reader in READERS:
  for p in plans:
   q=p['question_id'];r=load(RUN/'responses'/reader/(fp([reader,q,p['budget'],p['arm']])+'.json'))
   if r['signature']!=sid or r['plan_hash']!=fp(p):raise ValueError('Answer identity')
   em,f1=answer_scores(r['text'],refs[q]);records.append(dict(reader=reader,question_id=q,budget=p['budget'],arm=p['arm'],em=em,f1=f1,hops=refs[q]['hops'],status=r['status'],reused=r.get('reused',False),usage=r['usage'],evidence_tokens=p['evidence_tokens'],finish_reason=r.get('finish_reason'),fallback=p['trace'].get('fallback')))
 idx={(r['reader'],r['question_id'],r['budget'],r['arm']):r for r in records};pairs=[];groups=[]
 for reader in READERS:
  for budget in BUDGETS+('pooled',):
   bs=BUDGETS if budget=='pooled' else (budget,)
   for arm,control in [('quote_cover',a) for a in ARMS if a!='quote_cover']+[('quote_joint','quote_or'),('quote_anchor','quote_joint')]:
    values=[statistics.mean(idx[reader,q,b,arm]['f1']-idx[reader,q,b,control]['f1'] for b in bs) for q in refs]
    pairs.append(dict(reader=reader,budget=budget,contrast=arm+' minus '+control,**common.paired(values)))
   for arm in ARMS:
    rows=[r for r in records if r['reader']==reader and r['budget'] in bs and r['arm']==arm]
    groups.append(dict(reader=reader,budget=budget,arm=arm,em_points=100*statistics.mean(r['em'] for r in rows),f1_points=100*statistics.mean(r['f1'] for r in rows),mean_tokens=statistics.mean(r['evidence_tokens'] for r in rows)))
 def delta(reader,control):return next(x for x in pairs if x['reader']==reader and x['budget']=='pooled' and x['contrast']=='quote_cover minus '+control)
 primary=delta(READERS[0],'ap3');secondary=delta(READERS[1],'ap3')
 mechanism=any(delta(READERS[0],c)['mean_delta_f1']>0 for c in ('quote_or','quote_joint','quote_anchor'))
 screen=primary['mean_delta_f1']>=.02 and primary['wins']>=3 and secondary['mean_delta_f1']>0 and mechanism
 confirmed=primary['ci95'][0]>0 and secondary['mean_delta_f1']>0 and mechanism
 calls=[r for r in records if not r['reused'] and r['status'] in ('ok','empty')]
 summary=dict(signature=sid,panel=PANEL,complete=True,conditions=len(records),records=records,group_means=groups,paired_comparisons=pairs,primary=primary,
  development_screen_passed=screen,confirmation_criterion_met=confirmed if PANEL=='confirmation' else None,
  status_counts=dict(Counter(r['status'] for r in records)),unique_new_reader_calls=len(calls),
  reader_costs={k:sum(r['usage'].get(k,0) for r in calls) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
  proposal_gate=load(RUN/'proposal_gate.json'),selection_audit=load(RUN/'selection_audit.json'),
  caveats='Exposed development panel. Same native models and frozen candidates. Confirmation is an unused train-file panel, not official test. Joint quote retention does not certify entailment. Descriptive unadjusted intervals.')
 immutable(RUN/'summary.json',summary)
 lines=['# Anchored joint quotes '+PANEL,'',summary['caveats'],'','|Reader|Arm|F1|Tokens|','|---|---|---:|---:|']
 for g in groups:
  if g['budget']=='pooled':lines.append(f"|{g['reader']}|{g['arm']}|{g['f1_points']:.4f}|{g['mean_tokens']:.2f}|")
 (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('SCORE_COMPLETE',PANEL,primary,'screen',screen,flush=True)


def runtime():
 m=parent.runtime();m.BASE,m.PLAN,m.RUN,m.PANEL=BASE,PLAN,RUN,PANEL
 m.CONFIG,m.ARMS,m.verify,m.quote_plan,m.score=CONFIG,ARMS,verify,quote_plan,score
 original_assemble=m.assemble
 # Assembly's audit names must follow this experiment's controls.
 source=(ROOT/'reader_memory/role_quote_v1.py').read_text(encoding='utf-8')
 block=source[source.index('def assemble():'):source.index('\ndef qa(')]
 block=block.replace("('ap3','quote_flat','sentence_cover')","('ap3','quote_or','quote_joint','quote_anchor')")
 exec(compile(block,str(__file__)+':assemble','exec'),m.__dict__)
 if PANEL=='development':
  def reused_propose():
   sid=verify();bundle=load(PLAN/'public/proposals.json');ss=m.sources()
   for s in ss:
    row=bundle[s['question_id']+'.json']
    if row['source_hash']!=fp(s):raise ValueError('Proposal source mismatch')
    immutable(RUN/'proposals'/(s['question_id']+'.json'),dict(row,signature=sid,reused_from=row['signature']))
   gate=load(PLAN/'public/proposal_gate.json')
   immutable(RUN/'proposal_gate.json',dict(gate,signature=sid,reused_proposal_sources=len(ss),new_proposer_calls=0))
  m.propose=reused_propose
 return m

if __name__=='__main__':runtime().main()
