"""One fixed public-dev benchmark: reuse AP3-era baselines, add quotes and BRIEF-Pro."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import argparse,fcntl,json,os,statistics,subprocess,time,traceback
from . import paper_confirmation_v1 as confirmation
from . import paper_confirmation_runtime_v1 as rt
from . import full_comparison_v1_repair1 as historical
from . import full_comparison_v1 as old
from . import paper_competition_v1 as screen

u=confirmation.u;ROOT=confirmation.ROOT
BASE=ROOT/'research/paper_benchmark_extension_v1';RUN=ROOT/'runs/paper_benchmark_extension_v1'
LEGACY=list(old.METHODS)+['exit_budgeted','compact_budgeted','longrefiner_budgeted_repair_v1']
NEW=['quote_anchor','current_ap3','brief_pro_auto_cap','brief_pro_auto_native']
CONFIG=dict(version='paper-benchmark-extension-v1',sources=480,hop_counts={'2':249,'3':151,'4':80},
 methods=LEGACY+NEW,reader='Qwen3.5-9B',budget=512,reader_max_output=2048,reader_max_input=8192,
 seed=30,bootstrap=10000,new_proposer_max_calls=960,brief_max_calls=480,brief_max_output=1024,
 brief_revision='287329d91a06cb8158a94254e15030bea4f0ce19',brief_mode='official Auto prompt, Transformers BF16 SDPA greedy adaptation; archive native and Qwen512 prefix-cap outputs',
 primary_scope='Fixed supplementary benchmark on previously exposed official dev; not a new independent confirmation.',
 contrasts='quote_anchor against every capped baseline; native BRIEF separately, no same-budget claim',
 failed_gates_unchanged=True,training=False,adaptive_expansion=False,wait_timeout=None,wall_timeout=None)

def put(path,value):rt.put(path,value)
def state(status,**kwargs):
 v=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**kwargs);RUN.mkdir(parents=True,exist_ok=True)
 p=RUN/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(tmp,p);print(json.dumps(v),flush=True)
def sources():return u.load(BASE/'public/sources.json')
def freeze():
 assert historical.verify()=='d94204c5f5d74f2ae5c27ef3c98cc558ef89e5795be507a4f569660ab32a7b43'
 files=['reader_memory/paper_benchmark_extension_v1.py','research/paper_benchmark_extension_v1/PROTOCOL.md','research/paper_benchmark_extension_v1/allocation.json','research/paper_benchmark_extension_v1/public/sources.json']
 m=dict(config=CONFIG,confirmation_signature=confirmation.verify(),historical_signature=historical.verify(),
  files={f:u.sha(ROOT/f) for f in files},legacy_plan_hashes={name:u.sha(ROOT/'runs'/name/'qa_plans.jsonl') for name in ['full_comparison_v1_repair1','recent_baselines_v1','longrefiner_repair_v1']},
  legacy_baseline_manifests={name:u.sha(ROOT/'research'/name/'manifest.json') for name in ['recent_baselines_v1','longrefiner_repair_v1']},
  brief_asset_metadata_sha256=u.sha(ROOT/'research/paper_extension_v1/brief_asset_metadata.json'))
 put(BASE/'manifest.json',m)
def verify():
 m=u.load(BASE/'manifest.json');assert m['config']==CONFIG and confirmation.verify()==m['confirmation_signature']
 for p,d in m['files'].items():assert u.sha(ROOT/p)==d,p
 for name,d in m['legacy_plan_hashes'].items():assert u.sha(ROOT/'runs'/name/'qa_plans.jsonl')==d,name
 for name,d in m['legacy_baseline_manifests'].items():assert u.sha(ROOT/'research'/name/'manifest.json')==d,name
 assert u.sha(ROOT/'research/paper_extension_v1/brief_asset_metadata.json')==m['brief_asset_metadata_sha256']
 return u.sha(BASE/'manifest.json')
def path_for(phase,q):return RUN/phase/(u.fp(q)+'.json')

def prepare():
 sid=verify();ss=sources();by={s['question_id']:s for s in ss};assert len(by)==480
 tokenizer=rt.tokenizer('Qwen3.5-9B');plans=[];inputs={};replay={};proposals={}
 for runname,methods in [('full_comparison_v1_repair1',list(old.METHODS)),('recent_baselines_v1',['exit_budgeted','compact_budgeted']),('longrefiner_repair_v1',['longrefiner_budgeted_repair_v1'])]:
  folder=ROOT/'runs'/runname;count=0
  for line in (folder/'qa_plans.jsonl').open():
   p=json.loads(line)
   if p['question_id'] not in by or p['evidence_budget']!=512 or p['method'] not in methods:continue
   assert len(tokenizer.encode(p['evidence_text'],add_special_tokens=False))==p['evidence_tokens']<=512
   # Kept as an entire published configuration, with its original failure handling.
   plans.append(dict(dataset='musique',question_id=p['question_id'],question=by[p['question_id']]['question'],
     method=p['method'],budget=512,evidence_text=p['evidence_text'],evidence_tokens=p['evidence_tokens'],
     status='selection_failed' if p.get('selection_error') else 'ready',trace=p.get('trace',{}),
     origin=str((folder/'qa_plans.jsonl').relative_to(ROOT)),original_plan_hash=u.fp(p)))
   count+=1
  assert count==480*len(methods),(runname,count)
 for s in ss:
  q=s['question_id'];inp={}
  for phase in ['qwen','retrieve']:
   p=historical.RUN/phase/(q+'.json');v=u.load(p)
   assert v['signature']==u.load(BASE/'manifest.json')['historical_signature'] and v['source_sha256']==u.fp(s)
   inp['ranking' if phase=='qwen' else 'retrieval']=v
  inputs[q]=inp
  prior=screen.RUN/'proposals'/(u.fp([q,'role'])+'.json')
  if prior.exists():
   p=u.load(prior);assert p['source_hash']==u.fp(s)
   assert p['attempts'][0]['messages']==screen.proposal_messages(s,'role')
   proposals[q]=dict(parsed=p['parsed'],attempts=p['attempts'],origin=str(prior.relative_to(ROOT)))
 for q,v in proposals.items():put(path_for('proposals',q),v)
 for name,value in [('legacy_plans',plans),('inputs',inputs)]:put(RUN/(name+'.json'),value)
 # Full calls include messages and native output; decode/parse failures remain real outcomes.
 for runname in ['full_comparison_v1_repair1','recent_baselines_v1','longrefiner_repair_v1']:
  p=ROOT/'runs'/runname/'qa/answer_calls.jsonl'
  if not p.exists():continue
  with p.open() as f:
   for line in f:
    if not line.endswith('\n'):break
    c=json.loads(line)
    if not c.get('result') or not c['result'].get('completion_token_ids'):continue
    assert c['max_tokens']==2048
    key=u.fp(['Qwen3.5-9B',c['messages'],c['max_tokens']])
    row=dict(result=c['result'],origin=str(p.relative_to(ROOT)),messages=c['messages'],max_output=2048)
    if key in replay:assert replay[key]['result']['completion_token_ids']==row['result']['completion_token_ids'],'Conflicting exact legacy results'
    else:replay[key]=row
 # Newer completed screen artifacts are exact common-backend calls as well.
 if (screen.RUN/'requests.json').exists():
  for key,req in u.load(screen.RUN/'requests.json').items():
   out=screen.RUN/'outputs'/(key+'.json')
   if out.exists() and req['reader']=='Qwen3.5-9B':
    row=dict(result=u.load(out)['result'],origin=str(out.relative_to(ROOT)),messages=req['messages'],max_output=req['max_output'])
    if key in replay:assert row['result']['completion_token_ids']==replay[key]['result']['completion_token_ids']
    else:replay[key]=row
 put(RUN/'legacy_replay.json',replay)
 put(RUN/'preparation.json',dict(signature=sid,sources=480,legacy_materials=len(plans),reused_proposals=len(proposals),
  available_exact_reader_cache=len(replay),new_retrieval_calls=0,new_reranking_calls=0,references_read=False))
 state('prepared',legacy_materials=len(plans),reused_proposals=len(proposals))

def durable_call(phase,key,fn,attempt):
 dest=RUN/'calls'/phase/(key+'.json')
 if dest.exists():return u.load(dest)['result']
 for oldstart in (RUN/'attempts').glob('*/calls/'+key+'.json'):
  assert (oldstart.parent.parent/'oom.json').exists() and u.load(oldstart.parent.parent/'oom.json')['active_key']==key,'Unknown interrupted call'
 folder=RUN/'attempts'/attempt;put(folder/'calls'/(key+'.json'),dict(key=key,phase=phase,time=time.time()));os.environ['BENCHMARK_ACTIVE_KEY']=key
 result=fn();put(dest,dict(result=result,key=key,phase=phase));os.environ.pop('BENCHMARK_ACTIVE_KEY',None);return result

def propose(backend,attempt):
 from .mainline_service_v1 import generate
 ss=sources()
 for i,s in enumerate(ss,1):
  q=s['question_id'];dest=path_for('proposals',q)
  if not dest.exists():
   messages=screen.proposal_messages(s,'role');attempts=[]
   for j in range(2):
    assert rt.prompt_tokens(backend.tokenizer,messages)<=16384
    key=u.fp(['Qwen3.5-9B',messages,2048]);result=durable_call('propose',key,lambda:generate(backend,messages,2048),attempt)
    parsed=screen.hp.parse_result(s,'role',result);attempts.append(dict(messages=messages,result=result,parsed=parsed))
    if parsed['status']!='invalid':break
    correction='Repair your output once. Return the complete valid JSON object only. Validation errors: '+json.dumps(parsed['errors'])+'. Use only exact quotes from the original sources; do not invent facts. If no supported evidence is identifiable, return {"roles":[]}.'
    messages=screen.proposal_messages(s,'role')+[dict(role='assistant',content=result['text']),dict(role='user',content=correction)]
   put(dest,dict(parsed=parsed,attempts=attempts,origin='new_frozen_role_proposal'))
  if i%10==0:state('running',phase='propose',completed=i,expected=480)

def brief_generate(tokenizer,model,source):
 import torch
 documents='\n\n'.join(p['title']+'\n'+p['text'] for p in source['paragraphs'])
 content=f"Write a high-quality summary of the provided documents with respect to the question.\n ### This is the question: {source['question']}\n### These are the documents:\n{documents}\n### This is the summary:"
 prompt=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False)+"<|start_header_id|>assistant<|end_header_id|>"
 # vLLM's default encoder adds special tokens; preserve it explicitly.
 inputs=tokenizer(prompt,add_special_tokens=True,return_tensors='pt').to('cuda');n=inputs.input_ids.shape[-1]
 assert n+1024<=model.config.max_position_embeddings,'BRIEF input exceeds native context; no clipping'
 stops=list(set([tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|eot_id|>')]))
 start=time.perf_counter()
 with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=1024,do_sample=False,repetition_penalty=1.0,eos_token_id=stops,pad_token_id=tokenizer.eos_token_id)
 ids=out[0,n:].tolist();raw=tokenizer.decode(ids,skip_special_tokens=True)
 text=raw.removeprefix('<|start_header_id|>assistant<|end_header_id|>').strip().removeprefix('<|start_header_id|>').strip().removeprefix('assistant').strip().removeprefix('<|end_header_id|>').strip().removeprefix(',').strip().removeprefix('.').strip()
 return dict(text=text,raw_text=raw,completion_token_ids=ids,finish_reason='stop' if ids and ids[-1] in stops else 'length',
  usage=dict(prompt_tokens=n,completion_tokens=len(ids),elapsed_seconds=time.perf_counter()-start),prompt=prompt)

def brief(backend,attempt):
 tok,model=backend
 # A synthetic fixture tests the exact deployment path without dataset scoring.
 fixture=dict(question='Which city is the museum in?',paragraphs=[dict(title='Museum',text='The museum is in Brookfield.')])
 if not (RUN/'brief_technical_smoke.json').exists():
  result=durable_call('brief_smoke',u.fp(['brief-smoke-v1',fixture]),lambda:brief_generate(tok,model,fixture),attempt)
  assert result['text'].strip(),'Empty BRIEF synthetic output';put(RUN/'brief_technical_smoke.json',dict(passed=True,result=result))
 for i,s in enumerate(sources(),1):
  dest=path_for('brief',s['question_id'])
  if not dest.exists():
   result=durable_call('brief',u.fp(['brief',CONFIG['brief_revision'],s]),lambda:brief_generate(tok,model,s),attempt)
   put(dest,dict(source_hash=u.fp(s),result=result))
  if i%10==0:state('running',phase='brief',completed=i,expected=480)

def assemble():
 verify();tok=rt.tokenizer('Qwen3.5-9B')
 class Count:
  exact=True
  def count(self,text):return len(tok.encode(text,add_special_tokens=False))
 count=Count();plans=u.load(RUN/'legacy_plans.json');inputs=u.load(RUN/'inputs.json');ss=sources();by={s['question_id']:s for s in ss}
 from experiments.recent_baselines_v1.core import cap_text
 for s in ss:
  q=s['question_id'];inp=inputs[q];scores=inp['ranking']['scores'];seeds=[str(v) for v in inp['retrieval']['paths'][0]]
  global_plan,ap3=screen.hp.short.short_plans(s,scores,seeds,count,512,512);prop=u.load(path_for('proposals',q))['parsed']
  if global_plan.get('selection_error') or ap3.get('selection_error'):quote,trace=deepcopy(ap3),dict(fallback='baseline_selection_failure')
  else:quote,trace=screen.hp.anchored.quote_plan(s,scores,global_plan,ap3,prop,count,'quote_anchor')
  raw=u.load(path_for('brief',q))['result'];capped,applied=cap_text(raw['text'],512,tok)
  newplans={'quote_anchor':dict(quote,trace=trace),'current_ap3':dict(ap3,trace={}),
   'brief_pro_auto_cap':dict(evidence_text=capped,evidence_tokens=count.count(capped),trace=dict(cap_applied=applied,native_tokens=count.count(raw['text']),compressor_finish=raw['finish_reason'])),
   'brief_pro_auto_native':dict(evidence_text=raw['text'],evidence_tokens=count.count(raw['text']),trace=dict(cap_applied=False,compressor_finish=raw['finish_reason']))}
  for method,p in newplans.items():
   plans.append(dict(dataset='musique',question_id=q,question=s['question'],method=method,budget=0 if method.endswith('_native') else 512,
    evidence_text=p['evidence_text'],evidence_tokens=p['evidence_tokens'],trace=p['trace'],status='selection_failed' if p.get('selection_error') else 'ready',origin='new_material'))
 assert len(plans)==480*len(CONFIG['methods']) and len({(p['question_id'],p['method']) for p in plans})==len(plans)
 requests={};mapping=[];reuse={};cache=u.load(RUN/'legacy_replay.json')
 for p in plans:
  assert count.count(p['evidence_text'])==p['evidence_tokens'];assert p['budget']==0 or p['evidence_tokens']<=512
  req=screen.qa_request(p,by[p['question_id']]);key=req['key']
  if p['status']=='selection_failed':mapping.append(None);continue
  if rt.prompt_tokens(tok,req['messages'])>8192:p['status']='input_infeasible';mapping.append(None);continue
  mapping.append(key);requests[key]=req
  if key in cache:
   assert cache[key]['messages']==req['messages'] and cache[key]['max_output']==req['max_output'];reuse[key]=cache[key]
 for name,value in [('plans',plans),('mapping',mapping),('requests',requests),('reuse',reuse)]:put(RUN/(name+'.json'),value)
 put(RUN/'assembly.json',dict(logical_conditions=len(plans),unique_requests=len(requests),reused_requests=len(reuse),new_requests=len(requests)-len(reuse),
  per_method_counts=dict(Counter(p['method'] for p in plans)),references_read=False))
 state('assembled',logical_conditions=len(plans),unique_requests=len(requests),reused=len(reuse),new=len(requests)-len(reuse))

def qa(backend,attempt):
 from .mainline_service_v1 import generate
 sid=verify();requests=u.load(RUN/'requests.json');reuse=u.load(RUN/'reuse.json');pending=[(k,r) for k,r in sorted(requests.items()) if k not in reuse]
 for i,(key,req) in enumerate(pending,1):
  dest=RUN/'outputs'/(key+'.json')
  if not dest.exists():
   result=durable_call('qa',key,lambda:generate(backend,req['messages'],req['max_output']),attempt)
   assert result['usage']['prompt_tokens']==rt.prompt_tokens(backend.tokenizer,req['messages'])
   put(dest,dict(signature=sid,key=key,result=result))
  if i%10==0 or i==len(pending):state('running',phase='qa',completed=i,expected=len(pending))

def worker(phase,attempt):
 import torch
 from .paper_competition_fastqueue_v1 import NvmlMemory
 sid=verify();folder=RUN/'attempts'/attempt;folder.mkdir(parents=True,exist_ok=True)
 isbrief=phase=='brief';admission,reserve,cap=(12500,11000,11500) if isbrief else (26000,24000,24500)
 torch.set_num_threads(8);torch.manual_seed(30);torch.cuda.manual_seed_all(30);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter()
 try:
  if isbrief:
   from transformers import AutoTokenizer,AutoModelForCausalLM
   verified=u.load(ROOT/'research/paper_extension_v1/brief_assets_verified.json');assert verified['model_revision']==CONFIG['brief_revision']
   modeldir=ROOT/'models/BRIEF-Pro';tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
   model=AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval();backend=(tok,model)
  else:backend=rt.preload('qa_qwen');model=backend.model
  torch.cuda.set_per_process_memory_fraction(cap*2**20/torch.cuda.get_device_properties(0).total_memory);monitor=NvmlMemory()
  with (ROOT/'runs/gpu_spare_admission.lock').open('a') as lock:
   last=-float('inf')
   while True:
    free=monitor.free_mib();held=False
    if free>=admission:
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
     except BlockingIOError:pass
    if held:
     try:
      if monitor.free_mib()>=admission:
       reservation=torch.empty(max(0,int(reserve*2**20-torch.cuda.memory_reserved())),dtype=torch.uint8,device='cuda');del reservation;break
     except torch.OutOfMemoryError:torch.cuda.empty_cache()
     fcntl.flock(lock,fcntl.LOCK_UN)
    if time.monotonic()-last>20:state('waiting_for_memory',phase=phase,free_mib=round(free),required_mib=admission,expires_at=None);last=time.monotonic()
    time.sleep(.1)
   model.to('cuda').eval();assert all(p.device.type=='cuda' for p in model.parameters())
   state('running',phase=phase,attempt=attempt);globals()[phase](backend,attempt)
  put(RUN/'completed_phases'/(phase+'.json'),dict(signature=sid,phase=phase));return 0
 except torch.OutOfMemoryError as exc:
  put(folder/'oom.json',dict(active_key=os.environ.get('BENCHMARK_ACTIVE_KEY'),error=repr(exc)));return 75
 finally:put(folder/'resources.json',dict(phase=phase,seconds_with_wait=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0,includes_active_reservation=True))

def analyze():
 from .evidence_acquisition_v2 import parse_answer
 from .paper_confirmation_analysis_v1 import paired
 sid=verify();plans=u.load(RUN/'plans.json');mapping=u.load(RUN/'mapping.json');requests=u.load(RUN/'requests.json');reuse=u.load(RUN/'reuse.json')
 assert all(k in reuse or (RUN/'outputs'/(k+'.json')).exists() for k in requests)
 wanted={s['question_id'] for s in sources()};refs={q:r for q,r in u.load(historical.PLAN/'private/references.json').items() if q in wanted};assert len(refs)==480
 records=[]
 with historical.bind():
  for p,key in zip(plans,mapping):
   em=f1=0;status=p['status'];usage={};hypothesis=''
   if key:
    result=(reuse[key] if key in reuse else u.load(RUN/'outputs'/(key+'.json')))['result'];usage=result['usage']
    if result['finish_reason']!='stop':status='output_limit'
    else:
     try:hypothesis=parse_answer(result['text'])
     except (ValueError,KeyError,TypeError):status='parse_failure'
     else:em,f1=old.answer_scores(hypothesis,refs[p['question_id']]);status='ok'
   records.append(dict(question_id=p['question_id'],method=p['method'],budget=p['budget'],f1=f1,em=em,status=status,evidence_tokens=p['evidence_tokens'],usage=usage,key=key,reused=key in reuse if key else False))
 clusters=old.source_clusters(refs);groups=[];comparisons=[];anchor={r['question_id']:r['f1'] for r in records if r['method']=='quote_anchor'}
 for method in CONFIG['methods']:
  rr=[r for r in records if r['method']==method];assert len(rr)==480
  groups.append(dict(method=method,n=480,f1=100*statistics.mean(r['f1'] for r in rr),em=100*statistics.mean(r['em'] for r in rr),evidence_tokens=statistics.mean(r['evidence_tokens'] for r in rr),statuses=dict(Counter(r['status'] for r in rr))))
  if method!='quote_anchor':comparisons.append(dict(method='quote_anchor',control=method,same_budget=method!='brief_pro_auto_native',**paired({r['question_id']:anchor[r['question_id']]-r['f1'] for r in rr},clusters)))
 put(RUN/'analysis.json',dict(signature=sid,complete=True,groups=groups,comparisons=comparisons,records=records,
  scope=CONFIG['primary_scope'],assembly=u.load(RUN/'assembly.json'),publication_ready=False,full_deployment_costs_complete=False,
  costs_note='Legacy compressor/ranker/retriever costs are not zero; recover original per-source ledgers. New calls and proposal attempts include token usage. No controlled latency claim.'))
 state('complete',logical_conditions=len(plans),comparisons=len(comparisons))

def run():
 from .paper_competition_fastqueue_v1 import NvmlMemory
 verify()
 if not (RUN/'preparation.json').exists():prepare()
 monitor=NvmlMemory();attempt=len(list((RUN/'attempts').glob('*')));last=-float('inf')
 while True:
  pending=[p for p in ['brief','propose'] if not (RUN/'completed_phases'/(p+'.json')).exists()]
  if not pending:
   if not (RUN/'assembly.json').exists():assemble()
   pending=[] if (RUN/'completed_phases/qa.json').exists() else ['qa']
  if not pending:break
  free=monitor.free_mib();ready=[]
  for phase in pending:
   if phase=='brief' and not (ROOT/'research/paper_extension_v1/brief_assets_verified.json').exists():continue
   if free>=(12500 if phase=='brief' else 26000):ready.append(phase)
  if not ready:
   if time.monotonic()-last>20:state('queued',pending=pending,free_mib=round(free),brief_assets_ready=(ROOT/'research/paper_extension_v1/brief_assets_verified.json').exists(),expires_at=None);last=time.monotonic()
   time.sleep(.1);continue
  phase=ready[0];attempt+=1;name=f'{attempt:06d}';folder=RUN/'attempts'/name;folder.mkdir(parents=True)
  with (folder/'worker.log').open('a') as log:
   proc=subprocess.Popen([os.sys.executable,'-u','-m','reader_memory.paper_benchmark_extension_v1','worker','--phase',phase,'--attempt',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   (RUN/'active_worker.json').write_text(json.dumps(dict(pid=proc.pid,phase=phase,attempt=name)));code=proc.wait()
  if code==75:continue
  assert code==0,'Worker failed: '+str(folder/'worker.log')
 if not (RUN/'analysis.json').exists():analyze()

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run','worker','analyze']);p.add_argument('--phase');p.add_argument('--attempt');a=p.parse_args()
 try:
  if a.action=='worker':raise SystemExit(worker(a.phase,a.attempt))
  globals()[a.action]()
 except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
