"""A bounded matched-teacher distillation pilot; no final-role auxiliary supervision."""
from copy import deepcopy
import importlib.util
import math
import random
import statistics
import time
from . import selector_teacher_pilot_v1 as teacher
from .itercomp_engine_v1 import BGEEncoder, units, render
from .itercomp_primitives_v1 import dual_scores
ROOT=teacher.ROOT;BASE=ROOT/'research/selector_distill_v1';PLAN=BASE;RUN=ROOT/'runs/selector_distill_v1'
load,sha,fp,immutable,write_json=teacher.load,teacher.sha,teacher.fp,teacher.immutable,teacher.write_json
BUDGETS=teacher.BUDGETS;READERS=teacher.READERS
ARMS=('student_plain','student_role','bge_fixed','teacher_plain','teacher_role','teacher_role_sentence')
CONFIG=dict(teacher.CONFIG,version='selector-distill-v1',sources=64,train_sources=64,arms=list(ARMS),
    training=True,train_seed=30,epochs=20,batch_size=8,learning_rate=.001,weight_decay=.01,
    hidden_size=128,set_layers=2,set_heads=4,dropout=.1,encoder_batch=8,encoder_dtype='float32',
    student_threshold=.5,train_targets='Sentence expansion of final common-packed teacher selection; imitation, not truth',
    max_reader_conditions=1536,automatic_expansion=False,final_role_auxiliary=False,
    resource_caps_mib=dict(embed=6500,train=3000,retrieve=4000,rank=9000,propose9=20500,predict=3000,qa4=10500,qa9=19500),
    resource_headroom_mib=dict(embed=1000,train=1000,retrieve=1000,rank=1000,propose9=500,predict=1000,qa4=1000,qa9=1000))


def verify():
    m=load(BASE/'manifest.json');assert m['config']==CONFIG and m['signature']==fp({k:v for k,v in m.items() if k!='signature'})
    assert m['parent_signature']==teacher.verify()
    for n,h in m['files'].items():assert sha(ROOT/n)==h,n
    return m['signature']


def sources(split='calibration'):
    ss=load(BASE/'public'/('train_sources.json' if split=='train' else 'sources.json'));assert len(ss)==64
    return [teacher.parent.execution_source(s) for s in ss]


def cases():
    return sorted([(s,k,b) for s in sources() for k,b in (('role',None),('budget_plain',256),('budget_plain',512))],key=lambda x:fp([30,x[0]['question_id'],x[1],x[2]]))


def runtime():
    spec=importlib.util.spec_from_file_location('reader_memory._distill_parent',ROOT/'reader_memory/hotpot_confirmation_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.BASE,m.PLAN,m.RUN,m.CONFIG,m.sources,m.verify,m.cases,m.ARMS=BASE,BASE,RUN,CONFIG,sources,verify,cases,ARMS
    return m


def labels_from_plan(s,p):
    ranges=teacher.plan_ranges(s,p)
    return [int(any(a<c['end'] and z>c['start'] for a,z in ranges.get(c['source'],[]))) for c in units(s)]


def pack(cs,scores,budget,counter,eligible=None,threshold=None):
    started=time.perf_counter()
    assert len(cs)==len(scores) and all(math.isfinite(v) for v in scores)
    pool=list(range(len(cs))) if eligible is None else list(eligible)
    order=sorted(pool,key=lambda i:(-scores[i],i));selected=[]
    for i in order:
        if threshold is not None and scores[i]<threshold:continue
        candidate=sorted(selected+[i]);text=render(cs,candidate)
        if counter.count(text)<=budget:selected=candidate
    fallback=None
    if not selected:
        for i in order:
            if counter.count(render(cs,[i]))<=budget:selected=[i];fallback='highest_fitting_sentence';break
    text=render(cs,selected)
    return dict(evidence_text=text,evidence_tokens=counter.count(text),selected_indices=selected,
                selected_ids=[cs[i]['uid'] for i in selected],selected_ranges={},
                trace=dict(fallback=fallback),packing_seconds=time.perf_counter()-started,
                status='ready' if selected else 'selection_failed',reason=None if selected else 'No whole sentence fits')


def preflight():
    from transformers import AutoTokenizer
    from .full_comparison_v1 import beam,ranker
    sid=verify()
    for n,h in load(BASE/'manifest.json')['model_files'].items():assert sha(ROOT/n)==h,n
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    lens=[len(tok.encode(tok.apply_chat_template(teacher.parent.messages(s,k,b),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)) for s,k,b in cases()]
    assert max(lens)<=CONFIG['proposer_max_input']
    bge=AutoTokenizer.from_pretrained(str(ROOT/'models/bge-m3'),local_files_only=True,trust_remote_code=False)
    maximum=0;counter=teacher.parent.CounterTokens();fit=[]
    for split in ('train','calibration'):
        for s in sources(split):
            cs=units(s);texts=[s['question']]+[c['title']+'\n'+c['text'] for c in cs]
            maximum=max(maximum,max(len(x) for x in bge(texts,truncation=False)['input_ids']))
            fit.append(any(counter.count(render(cs,[i]))<=256 for i in range(len(cs))))
    assert maximum<=8192 and all(fit)
    bt=beam.tokenizer();rt=AutoTokenizer.from_pretrained(str(ranker.MODEL),local_files_only=True,trust_remote_code=False)
    for s in sources():
        beam.codes(bt,s,'cpu')
        for _,text in ranker.documents(s):ranker.rank_input_ids(rt,s['question'],text)
    immutable(RUN/'preflight.json',dict(signature=sid,maximum_teacher_prompt=max(lens),maximum_bge_tokens=maximum,all_sources_have_fitting_sentence=True,reference_values_parsed=False))


def embed():
    from .full_comparison_v1 import torch_setup
    torch=torch_setup();sid=verify();encoder=BGEEncoder(ROOT/'models/bge-m3',device='cuda',batch_size=CONFIG['encoder_batch']);counter=teacher.parent.CounterTokens()
    for split in ('train','calibration'):
        for i,s in enumerate(sources(split),1):
            cs=units(s);texts=[s['question']]+[c['title']+'\n'+c['text'] for c in cs]
            torch.cuda.synchronize();start=time.perf_counter();dense,sparse=encoder.encode(texts);torch.cuda.synchronize();seconds=time.perf_counter()-start
            q=dense[0];mix=dual_scores(q,sparse[0],dense[1:],sparse[1:])
            features=[]
            for c,d,score in zip(cs,dense[1:],mix):
                features.append(d+q+[abs(a-b) for a,b in zip(d,q)]+[a*b for a,b in zip(d,q)]+[score,counter.count(render([c],[0]))/512])
            path=RUN/'features'/split/(s['question_id']+'.pt');assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(x=torch.tensor(features,dtype=torch.float32),scores=mix,source_hash=fp(s),signature=sid,seconds=seconds),path)
            immutable(path.with_suffix('.json'),dict(signature=sid,source_hash=fp(s),sha256=sha(path),units=len(cs),seconds=seconds,input_policy='question; title plus sentence; no truncation; no teacher labels'))
            print('EMBED',split,i,'/64',flush=True)


def feature(s,split,torch):
    path=RUN/'features'/split/(s['question_id']+'.pt');receipt=load(path.with_suffix('.json'));assert sha(path)==receipt['sha256']
    row=torch.load(path,map_location='cpu',weights_only=True)
    assert row['signature']==verify() and row['source_hash']==fp(s) and row['x'].shape[0]==len(units(s))
    return row


def build_model(torch,dim):
    nn=torch.nn
    class Selector(nn.Module):
        def __init__(self):
            super().__init__();self.project=nn.Sequential(nn.Linear(dim+1,128),nn.GELU(),nn.LayerNorm(128))
            layer=nn.TransformerEncoderLayer(128,4,256,dropout=.1,batch_first=True,activation='gelu')
            self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.out=nn.Linear(128,1)
        def forward(self,x,mask):return self.out(self.context(self.project(x),src_key_padding_mask=~mask)).squeeze(-1)
    return Selector()


def collate(torch,examples,device):
    n=max(e['x'].shape[0] for e in examples);dim=examples[0]['x'].shape[-1]
    x=torch.zeros(len(examples),n,dim+1);mask=torch.zeros(len(examples),n,dtype=torch.bool);ys=torch.zeros(len(examples),n)
    for i,e in enumerate(examples):
        k=len(e['x']);x[i,:k,:dim]=e['x'];x[i,:k,dim]=e['budget']/512;mask[i,:k]=True
        if 'y' in e:ys[i,:k]=torch.tensor(e['y'])
    return x.to(device),mask.to(device),ys.to(device)


def state_hash(torch,model):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train():
    from .full_comparison_v1 import torch_setup
    torch=torch_setup();sid=verify();bundle=load(BASE/'training/targets.json');examples={a:[] for a in ('plain','role')};audit=[]
    for s in sources('train'):
        row=feature(s,'train',torch)
        for b in BUDGETS:
            target=bundle[s['question_id']][str(b)];audit.append(dict(question_id=s['question_id'],budget=b,fit=target['fit'],reason=target['reason']))
            if not target['fit']:continue
            for arm in examples:examples[arm].append(dict(x=row['x'],budget=b,y=target[arm]))
    assert len(examples['plain'])==len(examples['role'])>=64
    initialization=None;trained=[]
    for arm in ('plain','role'):
        torch.manual_seed(30);torch.cuda.manual_seed_all(30);model=build_model(torch,examples[arm][0]['x'].shape[1]).cuda()
        initial=state_hash(torch,model)
        if initialization is None:initialization=initial
        assert initial==initialization
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=random.Random(30);history=[];updates=0
        torch.cuda.synchronize();start=time.perf_counter()
        for epoch in range(20):
            order=list(range(len(examples[arm])));rng.shuffle(order);losses=[];model.train()
            for begin in range(0,len(order),8):
                batch=[examples[arm][i] for i in order[begin:begin+8]];x,mask,y=collate(torch,batch,'cuda')
                logits=model(x,mask);loss_matrix=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                positive=mask&(y>0);negative=mask&~positive
                losses_per_case=.5*(loss_matrix*positive).sum(1)/positive.sum(1).clamp_min(1)+.5*(loss_matrix*negative).sum(1)/negative.sum(1).clamp_min(1)
                loss=losses_per_case.mean();assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1;losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch+1,loss=statistics.mean(losses)));print('TRAIN',arm,epoch+1,history[-1]['loss'],flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        path=RUN/'models'/('student_'+arm+'.pt');assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),path)
        result=dict(arm=arm,signature=sid,initial_state_sha256=initial,checkpoint_sha256=sha(path),updates=updates,examples=len(examples[arm]),epochs=20,
                    seconds=elapsed,trainable_parameters=sum(p.numel() for p in model.parameters()),history=history,checkpoint_rule='Final epoch20 only; no calibration access',validation_used=False)
        immutable(path.with_suffix('.json'),result);trained.append(result)
        del model,optimizer;torch.cuda.empty_cache()
    assert trained[0]['updates']==trained[1]['updates']
    immutable(RUN/'training_summary.json',dict(signature=sid,trained=trained,common_fit_mask=audit,source_denominator=64,auxiliary_role_labels_used=False,answer_values_used=False))


def predict():
    from .full_comparison_v1 import torch_setup
    torch=torch_setup();sid=verify();data={s['question_id']:feature(s,'calibration',torch) for s in sources()}
    for arm in ('plain','role'):
        model=build_model(torch,next(iter(data.values()))['x'].shape[1]).cuda()
        path=RUN/'models'/('student_'+arm+'.pt');assert sha(path)==load(path.with_suffix('.json'))['checkpoint_sha256']
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.eval()
        for s in sources():
            q=s['question_id']
            for b in BUDGETS:
                x,mask,_=collate(torch,[dict(x=data[q]['x'],budget=b)],'cuda');torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():scores=model(x,mask).sigmoid()[0].cpu().tolist()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                immutable(RUN/'student_scores'/(fp([q,b,arm])+'.json'),dict(signature=sid,source_hash=fp(s),arm=arm,budget=b,scores=scores,seconds=elapsed))
        del model;torch.cuda.empty_cache()
    print('STUDENT_PREDICTIONS_COMPLETE',flush=True)


def assemble():
    import torch
    sid=verify();counter=teacher.parent.CounterTokens();plans=[]
    for s in sources():
        q=s['question_id'];cs=units(s);rank=load(RUN/'ranking'/(q+'.json'));retrieval=load(RUN/'retrieval'/(q+'.json'))
        proposals={(k,b):load(RUN/'proposals'/(fp([q,k,b])+'.json')) for k,b in (('role',None),('budget_plain',256),('budget_plain',512))}
        assert all(r['signature']==sid and r['source_hash']==fp(s) for r in [rank,retrieval,*proposals.values()])
        teachers={(p['budget'],p['arm']):p for p in teacher.make_plans(s,rank['scores'],retrieval,proposals,counter)}
        f=feature(s,'calibration',torch)
        for b in BUDGETS:
            for a,ta in (('teacher_plain','budget_plain'),('teacher_role','role_common')):
                p=deepcopy(teachers[b,ta]);p['arm']=a;plans.append(p)
            eligible=[i for i,y in enumerate(labels_from_plan(s,teachers[b,'role_common'])) if y]
            score=[rank['scores'][c['uid']] for c in cs]
            material=pack(cs,score,b,counter,eligible=eligible)
            plans.append(dict(material,question_id=q,budget=b,arm='teacher_role_sentence',private_inputs_read=False))
            material=pack(cs,f['scores'],b,counter)
            plans.append(dict(material,question_id=q,budget=b,arm='bge_fixed',private_inputs_read=False))
            for arm in ('plain','role'):
                saved=load(RUN/'student_scores'/(fp([q,b,arm])+'.json'));assert saved['signature']==sid and saved['source_hash']==fp(s)
                material=pack(cs,saved['scores'],b,counter,threshold=.5)
                plans.append(dict(material,question_id=q,budget=b,arm='student_'+arm,private_inputs_read=False))
    assert len(plans)==768 and len({(p['question_id'],p['budget'],p['arm']) for p in plans})==768
    assert all(counter.count(p['evidence_text'])==p['evidence_tokens']<=p['budget'] for p in plans)
    immutable(RUN/'plans.json',dict(signature=sid,selection_before_qa=True,plans=plans));print('ASSEMBLED',len(plans),flush=True)


def score():
    sid=verify();plans=load(RUN/'plans.json')['plans'];rows={}
    for reader in READERS:
        for p in plans:
            key=reader,p['question_id'],p['budget'],p['arm'];r=load(RUN/'responses'/reader/(fp(list(key))+'.json'))
            assert r['signature']==sid and r['plan_hash']==fp(p) and r['status'] in ('ok','empty','selection_failed');rows[key]=r
    assert len(rows)==1536
    refs=load(BASE/'private/references.json');assert set(refs)=={s['question_id'] for s in sources()}
    metric=teacher.parent.official_metrics();records=[]
    for reader in READERS:
        for p in plans:
            q,b,a=p['question_id'],p['budget'],p['arm'];r=rows[reader,q,b,a];gold=refs[q]['answer']
            records.append(dict(reader=reader,question_id=q,budget=b,arm=a,f1=float(metric['f1_score'](r['text'],gold)[0]),em=float(metric['exact_match_score'](r['text'],gold)),
                evidence_tokens=p['evidence_tokens'],status=r['status'],usage=r['usage'],reused=r.get('reused',False),finish_reason=r.get('finish_reason')))
    ix={(r['reader'],r['question_id'],r['budget'],r['arm']):r for r in records}
    def contrast(a,c):return teacher.parent.paired([statistics.mean(ix[r,q,b,a]['f1']-ix[r,q,b,c]['f1'] for r in READERS for b in BUDGETS) for q in refs])
    primary=contrast('student_role','student_plain');cheap=contrast('student_role','bge_fixed');retention=contrast('student_role','teacher_role_sentence')
    passed=primary['mean_delta_f1']>=.02 and cheap['mean_delta_f1']>=0 and retention['mean_delta_f1']>=-.03
    comparisons=[dict(arm=a,control=c,**contrast(a,c)) for a,c in [('student_role','student_plain'),('student_role','bge_fixed'),('student_role','teacher_role_sentence'),('student_role','teacher_role'),('teacher_role','teacher_plain'),('teacher_role_sentence','teacher_role')]]
    groups=[dict(arm=a,f1_points=100*statistics.mean(r['f1'] for r in records if r['arm']==a),mean_tokens=statistics.mean(p['evidence_tokens'] for p in plans if p['arm']==a)) for a in ARMS]
    fresh=[r for r in records if not r['reused'] and r['status'] in ('ok','empty')]
    summary=dict(signature=sid,complete=True,records=records,groups=groups,comparisons=comparisons,primary=primary,cheap=cheap,retention=retention,
        decision='eligible_for_larger_distillation_design' if passed else 'stop_scaling_this_student_for_diagnosis',
        calibration_only=True,independent_confirmation=False,noninferiority_established=False,publication_ready=False,
        student_training=load(RUN/'training_summary.json'),proposal_gate=load(RUN/'proposal_gate.json'),
        inherited_training_teacher_costs=load(BASE/'training/provenance.json')['teacher_costs'],
        new_reader_calls=len(fresh),reader_costs={k:sum(r['usage'].get(k,0) for r in fresh) for k in ('prompt_tokens','completion_tokens','elapsed_seconds')},
        caveats='Single seed,64 training and64 fixed calibration components. No final-role auxiliary. Frozen BGE encoder plus small set selector. Whole sentences differ from quote teacher; expanded teacher diagnostic is explicit. Descriptive feasibility thresholds are not noninferiority or innovation significance. No automatic expansion.')
    immutable(RUN/'summary.json',summary);print('SCORE_COMPLETE',summary['decision'],primary,flush=True)


def phase(name):
    if name in ('preflight','embed','train','predict','assemble','score'):globals()[name]()
    elif name=='retrieve':runtime().retrieve()
    elif name=='rank':runtime().runtime('relation_bundle_pilot_v1').rank()
    elif name=='propose9':runtime().propose()
    elif name in ('qa4','qa9'):runtime().runtime('role_quote_v1').qa('Qwen3.5-'+('4B' if name=='qa4' else '9B'))
    else:raise ValueError(name)
