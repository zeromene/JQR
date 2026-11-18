"""Capacity diagnostic: same anchored packing, same prompt, native 9B proposals."""
from types import FunctionType
from . import anchored_quote_v1 as parent
ROOT=parent.ROOT;BASE=ROOT/'research/anchored_quote_9b_v1';PANEL=parent.PANEL
PLAN=BASE/PANEL;RUN=ROOT/'runs/anchored_quote_9b_v1'/PANEL
load,sha,immutable,fp,write_json=parent.load,parent.sha,parent.immutable,parent.fp,parent.write_json
READERS,ARMS=parent.READERS,parent.ARMS
CONFIG=dict(parent.CONFIG,version='anchored-quote-9b-v1',proposer='Qwen3.5-9B',
 development_proposals='Fresh 9B proposals, same public prompt and deterministic settings as the 4B candidate control',
 proposer_allocator_cap_mib=22000,capacity_control='Compare the complete same-source 4B-proposer predecessor; model-size gain is not a new packing mechanism')

def verify():
 m=load(BASE/'manifest.json')
 if m['config']!=CONFIG or fp({k:v for k,v in m.items() if k!='signature'})!=m['signature']:raise ValueError('Manifest identity')
 for n,d in m['files'].items():
  if sha(ROOT/n)!=d:raise ValueError('Frozen file changed: '+n)
 return m['signature']

def runtime():
 m=parent.runtime();m.BASE,m.PLAN,m.RUN,m.PANEL=BASE,PLAN,RUN,PANEL
 m.CONFIG,m.verify=CONFIG,verify
 m.propose=FunctionType(parent.parent.parent.propose.__code__,m.__dict__)
 g=dict(parent.__dict__,BASE=BASE,PLAN=PLAN,RUN=RUN,PANEL=PANEL,CONFIG=CONFIG,verify=verify)
 m.score=FunctionType(parent.score.__code__,g)
 return m

if __name__=='__main__':runtime().main()
