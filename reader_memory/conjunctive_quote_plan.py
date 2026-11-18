"""Conservative joint retention of every proposed quote; no answer information."""
from copy import deepcopy


def joint_plan(s,scores,baseline,ap3,proposal,counter):
 from .anchor_guard_controls_v1 import packing
 from .relation_preserving_pilot_v1 import merge_ranges,render_ranges
 roles=proposal['roles'];candidates=proposal['candidates']
 represented=set().union(*(set(c['roles']) for c in candidates)) if candidates else set()
 trace=dict(roles=len(roles),quote_units=len(candidates),joint_quotes=True,fill=False,semantic_sufficiency_certified=False)
 if not candidates or represented!=set(range(len(roles))):
  return deepcopy(ap3),dict(trace,fallback='missing_role_quote')
 by={p['source_id']:p for p in s['paragraphs']};ranges={}
 for c in candidates:
  if by[c['source']]['text'][c['start']:c['end']]!=c['text']:raise ValueError('Quote provenance')
  ranges.setdefault(c['source'],[]).append([c['start'],c['end']])
 ranges={pid:merge_ranges(by[pid]['text'],rr) for pid,rr in ranges.items()}
 ranks={uid:i for i,uid in enumerate(packing.sentence_order(s,scores))}
 def key(pid):return min(ranks[pid+':span:'+str(i)] for i,sp in enumerate(by[pid]['spans']) if any(a<sp['end'] and b>sp['start'] for a,b in ranges[pid]))
 ranges={pid:ranges[pid] for pid in sorted(ranges,key=key)}
 text=render_ranges(s,ranges);tokens=counter.count(text)
 if tokens>baseline['evidence_tokens']:return deepcopy(ap3),dict(trace,fallback='joint_quotes_over_budget',joint_tokens=tokens)
 for c in candidates:
  if not any(a<=c['start'] and b>=c['end'] for a,b in ranges[c['source']]):raise ValueError('Lost conjunct')
 return dict(evidence_text=text,evidence_tokens=tokens,selected_ranges=ranges,selected_spans=[]),dict(trace,fallback=None,covered_roles=sorted(represented),seed_tokens=tokens)
