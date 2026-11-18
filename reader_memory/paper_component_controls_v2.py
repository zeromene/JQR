"""Additive component controls using frozen proposals and exact QA reuse."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import argparse,fcntl,json,os,statistics,subprocess,time,traceback
from . import paper_confirmation_v1 as parent
from . import paper_confirmation_runtime_v1 as runtime

u=parent.u;ROOT=parent.ROOT
BASE=ROOT/'research/paper_component_controls_v2';RUN=ROOT/'runs/paper_component_controls_v2'
CONFIG=dict(version='component-controls-v2',sources=1935,budget=512,
 methods=['quote_joint','quote_cover'],readers=parent.CONFIG['readers'],
 logical_conditions=7740,proposer_calls=0,training=False,seed=30,
 contrasts=['quote_anchor minus quote_joint','quote_anchor minus quote_cover'],
 scope='Post-confirmation supplementary analysis, same fixed sources; not an independent replication or rescue of the failed retention gate.',
 threshold='Report estimates and source-cluster 95% intervals for all eight contrasts. Do not require significance or expand after results.',
 wait_timeout=None,wall_timeout=None)

def put(path,value):runtime.put(path,value)
def state(status,**kwargs):
 value=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**kwargs)
 path=RUN/'status.json';path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,path)
 print(json.dumps(value),flush=True)

def freeze():
 parent_sig=parent.verify()
 files=['reader_memory/paper_component_controls_v2.py','research/paper_component_controls_v2/PROTOCOL.md']
 value=dict(config=CONFIG,parent_signature=parent_sig,files={x:u.sha(ROOT/x) for x in files},
  input_hashes={name:u.sha(parent.RUN/name) for name in ['plans.json','mapping.json','requests.json','analysis.json']})
 put(BASE/'manifest.json',value);return u.sha(BASE/'manifest.json')

def verify():
 m=u.load(BASE/'manifest.json');assert m['config']==CONFIG and parent.verify()==m['parent_signature']
 for path,digest in m['files'].items():assert u.sha(ROOT/path)==digest,path
 for name,digest in m['input_hashes'].items():assert u.sha(parent.RUN/name)==digest,name
 return u.sha(BASE/'manifest.json')

def prepare():
 sid=verify();tok=runtime.tokenizer('Qwen3.5-9B');phi=runtime.tokenizer('Phi-4-mini-instruct')
 class Count:
  exact=True
  def count(self,text):return len(tok.encode(text,add_special_tokens=False))
 count=Count();oldplans=u.load(parent.RUN/'plans.json');oldrequests=u.load(parent.RUN/'requests.json')
 anchor={(p['question_id'],p['reader']):p for p in oldplans if p['study']=='complete' and p['method']=='quote_anchor'}
 plans=[];requests={};mapping=[];reuse={};audit=[]
 for i,x in enumerate(parent.sources(),1):
  start=time.perf_counter();s=x['source'];q=s['question_id'];d=x['dataset']
  rank=runtime.cached('rank',q,s)['value'];ret=runtime.cached('retrieve',q,s)['value'];prop=runtime.cached('proposal',[q,'role'],s)['value']['parsed']
  global_plan,ap3=parent.old.hp.short.short_plans(s,rank['scores'],[str(k) for k in ret['paths'][0]],count,512,512)
  for method in ['quote_anchor']+CONFIG['methods']:
   if global_plan.get('selection_error') or ap3.get('selection_error'):p,trace=deepcopy(ap3),dict(fallback='baseline_selection_failure')
   else:p,trace=parent.old.hp.anchored.quote_plan(s,rank['scores'],global_plan,ap3,prop,count,method)
   assert count.count(p['evidence_text'])==p['evidence_tokens']<=512
   if method=='quote_anchor':
    assert p['evidence_text']==anchor[q,'Qwen3.5-9B']['evidence_text'],'Anchor reproduction changed'
    continue
   for reader in CONFIG['readers']:
    t=tok if reader=='Qwen3.5-9B' else phi
    plan=dict(dataset=d,question_id=q,reader=reader,method=method,budget=512,
     evidence_text=p['evidence_text'],evidence_tokens=p['evidence_tokens'],
     actual_reader_evidence_tokens=len(t.encode(p['evidence_text'],add_special_tokens=False)),trace=trace)
    req=parent.qa_request(plan,s,reader);key=req['key'];plans.append(plan);mapping.append(key)
    assert runtime.prompt_tokens(t,req['messages'])<=8192
    if key in oldrequests:
     assert req['messages']==oldrequests[key]['messages'] and req['max_output']==oldrequests[key]['max_output']
     old=u.load(parent.RUN/'outputs'/(key+'.json'));assert old['request_identity']==key
     reuse[key]=str((parent.RUN/'outputs'/(key+'.json')).relative_to(ROOT))
    requests[key]=req
  audit.append(dict(question_id=q,seconds_all_controls=time.perf_counter()-start))
  if i%100==0:state('preparing',completed=i,expected=1935)
 assert len(plans)==len(mapping)==7740
 for name,value in [('plans',plans),('mapping',mapping),('requests',requests),('reuse',reuse),('packing_costs',audit)]:put(RUN/(name+'.json'),value)
 result=dict(signature=sid,logical_conditions=len(plans),unique_requests=len(requests),exact_reused_requests=len(reuse),
  new_requests_by_reader={r:sum(k not in reuse and v['reader']==r for k,v in requests.items()) for r in CONFIG['readers']},
  anchor_reproductions=1935,references_read=False)
 put(RUN/'preparation.json',result);state('prepared',**result)

def qa_worker(reader,attempt):
 import torch
 from .paper_competition_fastqueue_v1 import NvmlMemory
 from .mainline_service_v1 import generate
 sid=verify();folder=RUN/'attempts'/attempt;folder.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 isq=reader=='Qwen3.5-9B';admission=26000 if isq else 12000;reserve=24000 if isq else 10500;cap=24500 if isq else 11000
 active_key=None;start=time.perf_counter()
 try:
  state('preloading_cpu',reader=reader,attempt=attempt)
  backend=runtime.preload('qa_qwen' if isq else 'qa_phi');monitor=NvmlMemory()
  torch.cuda.set_per_process_memory_fraction(cap*2**20/torch.cuda.get_device_properties(0).total_memory)
  with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
   last=-float('inf')
   while True:
    free=monitor.free_mib();acquired=False
    if free>=admission:
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
     except BlockingIOError:pass
    if acquired:
     try:
      if monitor.free_mib()>=admission:
       claim=torch.empty(max(0,int(reserve*2**20-torch.cuda.memory_reserved())),dtype=torch.uint8,device='cuda');del claim;break
     except torch.OutOfMemoryError:torch.cuda.empty_cache()
     fcntl.flock(lock,fcntl.LOCK_UN)
    if time.monotonic()-last>=20:state('waiting_for_memory',reader=reader,free_mib=round(free),required_mib=admission,expires_at=None);last=time.monotonic()
    time.sleep(.1)
   backend.model.to('cuda').eval();assert all(v.device.type=='cuda' for v in backend.model.parameters())
   requests=u.load(RUN/'requests.json');reuse=u.load(RUN/'reuse.json')
   pending=[(k,v) for k,v in sorted(requests.items()) if v['reader']==reader and k not in reuse]
   state('running',reader=reader,completed=0,expected=len(pending),attempt=attempt)
   for i,(key,req) in enumerate(pending,1):
    dest=RUN/'outputs'/(key+'.json')
    if not dest.exists():
     for old in (RUN/'attempts').glob('*/calls/'+key+'.json'):
      assert (old.parent.parent/'oom.json').exists() and u.load(old.parent.parent/'oom.json')['active_key']==key,'Unknown interrupted request'
     active_key=key;put(folder/'calls'/(key+'.json'),dict(key=key,started=time.time()))
     result=generate(backend,req['messages'],req['max_output'])
     assert result['usage']['prompt_tokens']==runtime.prompt_tokens(backend.tokenizer,req['messages'])
     put(dest,dict(signature=sid,key=key,request_identity=key,result=result));active_key=None
    else:assert u.load(dest)['signature']==sid
    if i%10==0 or i==len(pending):state('running',reader=reader,completed=i,expected=len(pending),attempt=attempt)
  put(RUN/'completed_readers'/(reader+'.json'),dict(signature=sid,reader=reader));return 0
 except torch.OutOfMemoryError as exc:
  put(folder/'oom.json',dict(active_key=active_key,error=repr(exc)));state('resource_retry',reader=reader);return 75
 finally:
  put(folder/'resources.json',dict(reader=reader,seconds_with_wait=time.perf_counter()-start,
    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0,
    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,includes_active_reservation=True))

def analyze():
 from . import paper_confirmation_analysis_v1 as analysis
 from .evidence_acquisition_v2 import parse_answer
 sid=verify();requests=u.load(RUN/'requests.json');reuse=u.load(RUN/'reuse.json')
 assert all(k in reuse or (RUN/'outputs'/(k+'.json')).exists() for k in requests)
 refs=analysis.references();metric=parent.old.hp.official_metrics()
 plans=u.load(RUN/'plans.json');mapping=u.load(RUN/'mapping.json');records=[]
 for p,key in zip(plans,mapping):
  result=u.load(ROOT/reuse[key] if key in reuse else RUN/'outputs'/(key+'.json'))['result'];f1=em=0;status='ok';hypothesis=''
  if p['dataset']=='hotpot':
   hypothesis=result['text'];em=float(metric['exact_match_score'](hypothesis,refs[p['question_id']]['answer']));f1=float(metric['f1_score'](hypothesis,refs[p['question_id']]['answer'])[0])
  elif result['finish_reason']!='stop':status='output_limit'
  else:
   try:hypothesis=parse_answer(result['text'])
   except (ValueError,KeyError,TypeError):status='parse_failure'
   else:em,f1=analysis.mu.answer_scores(hypothesis,refs[p['question_id']])
  records.append({k:v for k,v in p.items() if k!='evidence_text'}|dict(f1=f1,em=em,status=status,usage=result['usage'],key=key,reused=key in reuse))
 old=[r for r in u.load(parent.RUN/'analysis.json')['records'] if r['study']=='complete' and r['method']=='quote_anchor']
 metadata={('hotpot:'+r['native_id'] if r['dataset']=='hotpot' else r['native_id']+':multi'):r for r in u.load(parent.BASE/'allocation.json')['selected']}
 comparisons=[];groups=[]
 for d in ['hotpot','musique']:
  for reader in CONFIG['readers']:
   anchor={r['question_id']:r['f1'] for r in old if r['dataset']==d and r['reader']==reader}
   clusters={q:metadata[q]['component_id' if d=='hotpot' else 'support_document_component'] for q in anchor}
   for method in CONFIG['methods']:
    rr=[r for r in records if (r['dataset'],r['reader'],r['method'])==(d,reader,method)]
    assert {r['question_id'] for r in rr}==set(anchor)
    comparisons.append(dict(dataset=d,reader=reader,method='quote_anchor',control=method,**analysis.paired({r['question_id']:anchor[r['question_id']]-r['f1'] for r in rr},clusters)))
    groups.append(dict(dataset=d,reader=reader,method=method,n=len(rr),f1=100*statistics.mean(r['f1'] for r in rr),em=100*statistics.mean(r['em'] for r in rr),
     evidence_tokens=statistics.mean(r['evidence_tokens'] for r in rr),statuses=dict(Counter(r['status'] for r in rr))))
 put(RUN/'analysis.json',dict(signature=sid,complete=True,groups=groups,comparisons=comparisons,records=records,
  scope=CONFIG['scope'],original_confirmation_decision_unchanged=True,automatic_expansion=False))
 state('complete',logical_conditions=7740,comparisons=len(comparisons))

def run():
 verify()
 if not (RUN/'preparation.json').exists():prepare()
 attempt=len(list((RUN/'attempts').glob('*')))
 for reader in ['Phi-4-mini-instruct','Qwen3.5-9B']:
  while not (RUN/'completed_readers'/(reader+'.json')).exists():
   attempt+=1;name=f'{attempt:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
   with (folder/'worker.log').open('a') as log:
    proc=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_component_controls_v2','worker','--reader',reader,'--attempt',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (RUN/'active_worker.json').write_text(json.dumps(dict(pid=proc.pid,reader=reader,attempt=name)))
    code=proc.wait()
   if code==75:continue
   assert code==0,'Component worker failed: '+str(folder/'worker.log')
 if not (RUN/'analysis.json').exists():analyze()

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run','worker','analyze']);p.add_argument('--reader');p.add_argument('--attempt');a=p.parse_args()
 try:
  if a.action=='worker':raise SystemExit(qa_worker(a.reader,a.attempt))
  globals()[a.action]()
 except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
