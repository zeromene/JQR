"""Ordinary supervised Beam adaptation; predicted hops, no inference gold labels."""
import argparse
from collections import Counter
import gc
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import time

from . import relation_preserving_pilot_v1 as recomp
from .schema import fingerprint as fp, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/path_retrieval_pilot_v1'
RUN = ROOT / 'runs/path_retrieval_pilot_v1'
MODEL = ROOT / 'models/DeBERTa-v3-base'
CODE_REV = '902980bfb5bd47a963569ab9c3621fd5ae81204a'
ARMS = ('recomp_sentence', 'beam_path_fixed')
CONFIG = dict(version='path-retrieval-pilot-v1', seed='path-retrieval-20261030-v1', torch_seed=30,
    allocation=dict(development={'3':16, '2':48}, calibration={'3':8, '2':24},
                    reserved={'3':8, '2':24}, train={'3':256, '2':768}),
    epochs=6, train_queries=2048, gradient_accumulation=8, learning_rate=1e-5,
    weight_decay=0.01, warmup_ratio=0.1, gradient_clip=1., beam_size=2, max_seq_len=512,
    precision='FP32 parameters/AdamW; BF16 autocast; TF32 disabled',
    tokenizer='native pinned SentencePiece, case preserved; no transformers5 automatic conversion',
    retriever='official Beam Retriever class, two binary heads, ordered MuSiQue supervision, early-stop training',
    code_revision=CODE_REV, backbone='microsoft/deberta-v3-base',
    backbone_revision='8ccc9b6f36199bec6961081d44eb72fb3f7353f3',
    hop_prediction='additional linear three-class head, question-only CLS; train count labels; no gold inference hops',
    pilot_hops=[1,2,3], four_hop_efficacy_claimed=False,
    hop_loss_weight=0.2, checkpoint_selection='calibration support-set F1, ties retain earlier epoch',
    packing='whole best predicted path if it fits; otherwise ordinary RECOMP sentence compression within that path',
    retriever_clipping='author max512 prefix policy; candidate clipping and predicted length logged, reader original text preserved',
    reader='Qwen3.5-9B', reader_variant='format_only', evidence_budget=2048, output_tokens=2048,
    decoding='greedy-thinking_false', arms=list(ARMS), development_sources=64, development_questions=128,
    max_new_reader_calls=256, retries=0, independent_confirmation=False,
    dependency_protection_implemented=False, official_paper_replication=False,
    protected_final_data_accessed=False, automatic_followup=False)

load, immutable, sha = recomp.load, recomp.immutable, recomp.base.sha


def blind_input(source):
    """Drop even metadata that could encode gold hop count or single/multi mode."""
    return dict(question=source['question'], paragraphs=[dict(title=p['title'], text=p['text']) for p in source['paragraphs']])


def validate_source(source):
    if set(source) != {'question_id','native_id','mode','question','paragraphs'}:
        raise ValueError('Unexpected public source fields')
    if len(source['paragraphs']) != 20:
        raise ValueError('All20 candidate paragraphs required')
    for i,p in enumerate(source['paragraphs']):
        if set(p) != {'source_id','title','timestamp','text','spans'} or p['source_id'] != str(i):
            raise ValueError('Private candidate fields or source reordering')
        for s in p['spans']:
            if set(s) != {'start','end','text'} or p['text'][s['start']:s['end']] != s['text']:
                raise ValueError('Source offset mismatch')


def freeze():
    if (PLAN/'manifest.json').exists(): raise ValueError('Already frozen')
    inherited = load(recomp.PLAN/'manifest.json')
    paths = [p for p in inherited['files'] if p.endswith('.py')]
    paths += ['reader_memory/path_retrieval_pilot_v1.py','tools/prepare_path_retrieval_pilot_v1.py',
        'tools/stage_path_retrieval_pilot_v1.py','scripts/path_retrieval_pilot_v1_queue.py',
        'tests/test_path_retrieval_pilot_v1.py','tools/check_path_runtime_v1.py','models/DeBERTa-v3-base/artifact_manifest.json',
        'models/RECOMP-NQ-extractive/artifact_manifest.json']
    paths += [p.relative_to(ROOT).as_posix() for p in PLAN.rglob('*') if p.is_file()
        and p.name not in {'manifest.json','CURRENT_STATUS.md','submission_receipt.json'} and p.suffix != '.pyc']
    value = dict(config=CONFIG, files={p:sha(ROOT/p) for p in sorted(set(paths))})
    value['signature'] = fp(value)
    immutable(PLAN/'manifest.json', value)
    print('FROZEN_PATH_PILOT', value['signature'], flush=True)


def verify():
    m = load(PLAN/'manifest.json')
    if m['config'] != CONFIG or fp({k:v for k,v in m.items() if k!='signature'}) != m['signature']:
        raise ValueError('Scientific identity mismatch')
    for name,digest in m['files'].items():
        if sha(ROOT/name) != digest: raise ValueError('Frozen input changed: '+name)
    return m['signature']


def check_models():
    for model in (MODEL, recomp.MODEL):
        manifest = load(model/'artifact_manifest.json')
        for f in manifest['files']:
            p = model/f['name']
            if p.stat().st_size != f['size'] or sha(p) != f['sha256']:
                raise ValueError('Offline model artifact mismatch: '+str(p))


class NativeSentencePiece:
    """Use the original DeBERTa slow tokenizer's default SentencePiece semantics.

    Upstream defaults: no lowercasing, no punctuation splitting, spm.encode.
    The pinned upstream reference is kept in sources/; the new transformers5
    tokenizer conversion was found not to preserve these IDs in a train-only check.
    """
    def __init__(self,path):
        import sentencepiece
        config=load(Path(path)/'tokenizer_config.json')
        if config.get('do_lower_case') is not False or config.get('vocab_type')!='spm':
            raise ValueError('Unexpected native tokenizer settings')
        self.spm=sentencepiece.SentencePieceProcessor(model_file=str(Path(path)/'spm.model'))
        self.pad_token_id,self.cls_token_id,self.sep_token_id,self.unk_token_id=[self.spm.piece_to_id(v) for v in ('[PAD]','[CLS]','[SEP]','[UNK]')]
        if [self.pad_token_id,self.cls_token_id,self.sep_token_id,self.unk_token_id]!=[0,1,2,3]:
            raise ValueError('Unexpected DeBERTa special IDs')

    def encode(self,text,add_special_tokens=True,truncation=False):
        if truncation:raise ValueError('Clipping must be explicit and audited by the retriever')
        if any(v in text for v in ('[PAD]','[CLS]','[SEP]','[UNK]','[MASK]')):
            raise ValueError('Literal model special token in native data; unsupported, not silently converted')
        ids=self.spm.encode(text,out_type=int)
        return [self.cls_token_id]+ids+[self.sep_token_id] if add_special_tokens else ids


def tokenizer():
    return NativeSentencePiece(MODEL)


def codes(tok, source, device):
    import torch
    s = blind_input(source)
    question = s['question'][:-1] if s['question'].endswith('?') else s['question']
    q = tok.encode(question, add_special_tokens=False, truncation=False)
    if not 0 < len(q) < 256: raise ValueError('Unusually long/empty question; refuse silent clipping')
    full = [tok.encode(p['title']+'.'+p['text'], add_special_tokens=False, truncation=False) for p in s['paragraphs']]
    cs = [v[:CONFIG['max_seq_len']-len(q)] for v in full]
    return dict(q_codes=[torch.tensor(q,device=device,dtype=torch.long)],
                c_codes=[[torch.tensor(v,device=device,dtype=torch.long) for v in cs]]), dict(
                question_tokens=len(q), original_candidate_tokens=list(map(len,full)),
                retained_candidate_tokens=list(map(len,cs)),
                clipped_candidates=sum(len(a)!=len(b) for a,b in zip(full,cs)),
                internal_prefix_clipping='author policy max512, additionally divides prefix space by predicted hops during inference')


def make_model(device):
    import torch
    from transformers import AutoConfig, AutoModel
    tok = tokenizer()
    conf = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    conf.cls_token_id, conf.sep_token_id = tok.cls_token_id, tok.sep_token_id
    path = PLAN/'sources/beam_retriever/retrieval/retriever_model.py'
    spec = importlib.util.spec_from_file_location('pinned_beam_retriever',path)
    vendor = importlib.util.module_from_spec(spec);spec.loader.exec_module(vendor)
    model = vendor.Retriever(conf,str(MODEL),AutoModel,max_seq_len=512,beam_size=2,
        gradient_checkpointing=True,use_label_order=True,use_early_stop=True)
    model.hop_count_classifier_layer = torch.nn.Linear(conf.hidden_size,3)
    return tok, model.to(device)


def count_logits(torch, model, q):
    ids = torch.cat([q.new_tensor([model.config.cls_token_id]),q,q.new_tensor([model.config.sep_token_id])])[None,:]
    hidden = model.encoder(input_ids=ids,attention_mask=torch.ones_like(ids))[0][:,0,:]
    return model.hop_count_classifier_layer(hidden)


def encoder_meter(model):
    costs=dict(encoder_forward_calls=0,encoder_active_tokens=0,encoder_padded_positions=0)
    def hook(module,args,kwargs):
        ids=kwargs.get('input_ids')
        if ids is None and args:ids=args[0]
        if ids is None:raise ValueError('Unmetered encoder invocation')
        mask=kwargs.get('attention_mask')
        costs['encoder_forward_calls']+=1
        costs['encoder_padded_positions']+=ids.numel()
        costs['encoder_active_tokens']+=int(mask.sum()) if mask is not None else ids.numel()
    return costs,model.encoder.register_forward_pre_hook(hook,with_kwargs=True)


def infer(torch, tok, model, source):
    # This function never loads a reference file or consumes source IDs/modes.
    encoded, audit = codes(tok, source, 'cuda')
    started = time.perf_counter()
    costs,handle=encoder_meter(model)
    try:
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            logits = count_logits(torch, model, encoded['q_codes'][0]).float()
            hops = int(logits.argmax(dim=-1).item()) + 1
            result = model(**encoded, sf_idx=[[]], hop=hops)
    finally:handle.remove()
    torch.cuda.synchronize()
    paths = result['current_preds']
    if not paths or any(len(p)!=hops or len(set(p))!=len(p) or any(not 0<=j<20 for j in p) for p in paths):
        raise ValueError('Invalid blind retrieval path')
    return dict(predicted_hops=hops, hop_logits=logits[0].cpu().tolist(),paths=paths,
                tokenization=audit,seconds=time.perf_counter()-started,costs=costs)


def support_f1(path, support):
    a,b = set(path),set(support)
    return 2*len(a&b)/(len(a)+len(b)) if a or b else 1.


def calibration(torch,tok,model,sources,refs):
    model.eval(); rows=[]
    for source in sources:
        prediction = infer(torch,tok,model,source)
        target = refs[source['question_id']]['support_indices']
        rows.append(dict(question_id=source['question_id'],**prediction,
            support_set_f1=support_f1(prediction['paths'][0],target),
            exact_support_set=set(prediction['paths'][0])==set(target),
            hop_correct=prediction['predicted_hops']==len(target)))
    return dict(rows=rows, support_set_f1=statistics.mean(r['support_set_f1'] for r in rows),
        exact_support_set=statistics.mean(r['exact_support_set'] for r in rows),
        hop_accuracy=statistics.mean(r['hop_correct'] for r in rows))


def preflight():
    sid=verify();check_models()
    from transformers import AutoTokenizer
    tok=tokenizer()
    comp=AutoTokenizer.from_pretrained(str(recomp.MODEL),local_files_only=True)
    maxima={'question':0,'recomp':0};counts={};native_checked=0
    for split,expected in [('train',2048),('calibration',64),('development',128)]:
        sources=load(PLAN/'public'/(split+'_inputs.json'))
        if len(sources)!=expected: raise ValueError('Wrong split size')
        counts[split]=len(sources)
        for s in sources:
            validate_source(s)
            for text in [s['question']]+[p['title']+'.'+p['text'] for p in s['paragraphs']]:
                expected=[tok.spm.piece_to_id(piece) for piece in tok.spm.encode(text,out_type=str)]
                if tok.encode(text,add_special_tokens=False,truncation=False)!=expected:
                    raise ValueError('Native IDs differ from original pieces-to-IDs tokenizer semantics')
                native_checked+=1
            q=s['question'][:-1] if s['question'].endswith('?') else s['question']
            n=len(tok.encode(q,add_special_tokens=False,truncation=False))
            maxima['question']=max(maxima['question'],n)
            if n>=256: raise ValueError('Question exceeds explicit retriever limit')
            if split=='development':
                lengths=[len(comp.encode(t,add_special_tokens=True,truncation=False))
                    for t in [s['question']]+[t for _,t in recomp.units(s)]]
                maxima['recomp']=max(maxima['recomp'],max(lengths))
    if maxima['recomp']>512: raise ValueError('RECOMP input exceeds512; no silent clipping')
    if sha(ROOT/'models/Qwen3.5-9B/.verified.json')!=load(PLAN/'model_references.json')['reader_marker_sha256']:
        raise ValueError('Reader verification marker changed')
    immutable(RUN/'preflight.json',dict(signature=sid,passed=True,questions=counts,max_tokens=maxima,
        native_sentencepiece_matching_texts=native_checked,
        inference_has_gold_hops=False,protected_final_data_accessed=False))
    print('PATH_PREFLIGHT_OK',counts,maxima,flush=True)


def train():
    sid=verify()
    if (RUN/'training_started.json').exists(): raise ValueError('Training already started; no silent rerun/resume')
    import torch
    torch.manual_seed(CONFIG['torch_seed']); random.seed(CONFIG['torch_seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok,model=make_model('cuda')
    training_costs,meter_handle=encoder_meter(model)
    sources=load(PLAN/'public/train_inputs.json');refs=load(PLAN/'private/train_references.json')
    calibration_sources=load(PLAN/'public/calibration_inputs.json')
    calibration_refs=load(PLAN/'private/calibration_references.json')
    encoded=[codes(tok,s,'cpu')[0] for s in sources]
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and p.ndim>=2],weight_decay=0.01),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and p.ndim<2],weight_decay=0.)]
    optimizer=torch.optim.AdamW(groups,lr=CONFIG['learning_rate'])
    total=CONFIG['epochs']*math.ceil(len(sources)/CONFIG['gradient_accumulation'])
    warmup=max(1,int(total*CONFIG['warmup_ratio']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: step/warmup if step<warmup else max(0,(total-step)/(total-warmup)))
    immutable(RUN/'training_started.json',dict(signature=sid,parameters=sum(p.numel() for p in model.parameters()),
        training_queries=len(sources),epochs=CONFIG['epochs'],optimizer_steps=total,training_supervision='support paragraph order + count only, no reader-answer feedback'))
    best=-1.;best_epoch=0;global_step=0;started=time.perf_counter();history=[]
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        order=list(range(len(sources)));random.Random(CONFIG['torch_seed']+epoch).shuffle(order)
        epoch_start=time.perf_counter();loss_sum=0.
        for position,index in enumerate(order,1):
            batch={k: [[v.to('cuda') for v in arr] for arr in values] if k=='c_codes'
                   else [v.to('cuda') for v in values] for k,values in encoded[index].items()}
            chain=refs[sources[index]['question_id']]['support_indices']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                retrieval_loss=model(**batch,sf_idx=[chain])['loss']
                logits=count_logits(torch,model,batch['q_codes'][0])
                hop_loss=torch.nn.functional.cross_entropy(logits.float(),torch.tensor([len(chain)-1],device='cuda'))
                raw_loss=retrieval_loss+CONFIG['hop_loss_weight']*hop_loss
                loss=raw_loss/CONFIG['gradient_accumulation']
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();loss_sum+=float(raw_loss.detach())
            if position%CONFIG['gradient_accumulation']==0 or position==len(order):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'])
                if not torch.isfinite(norm): raise ValueError('Nonfinite gradients')
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
                status=dict(signature=sid,epoch=epoch,epochs=CONFIG['epochs'],query=position,queries=len(order),
                    optimizer_step=global_step,total_optimizer_steps=total,mean_loss=loss_sum/position,
                    elapsed_seconds=time.perf_counter()-started,epoch_seconds=time.perf_counter()-epoch_start,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),lr=scheduler.get_last_lr()[0],
                    cumulative_encoder_costs=dict(training_costs))
                write_json(RUN/'training_status.json',status)
                if global_step%8==0 or global_step==1: print('PATH_TRAIN',json.dumps(status),flush=True)
        metrics=calibration(torch,tok,model,calibration_sources,calibration_refs)
        immutable(RUN/'calibration'/f'epoch_{epoch}.json',dict(signature=sid,epoch=epoch,**metrics))
        if metrics['support_set_f1']>best+1e-12:
            best,best_epoch=metrics['support_set_f1'],epoch
            temporary=RUN/'best.pt.tmp';torch.save(model.state_dict(),temporary);temporary.replace(RUN/'best.pt')
        history.append(dict(epoch=epoch,mean_loss=loss_sum/len(order),
            **{k:v for k,v in metrics.items() if k!='rows'},seconds=time.perf_counter()-epoch_start))
        write_json(RUN/'training_history.json',history)
        print('PATH_EPOCH_COMPLETE',json.dumps(history[-1]),'best_epoch',best_epoch,flush=True)
    immutable(RUN/'training_complete.json',dict(signature=sid,best_epoch=best_epoch,calibration_support_f1=best,
        checkpoint_sha256=sha(RUN/'best.pt'),optimizer_steps=global_step,seconds=time.perf_counter()-started,
        history=history,development_feedback_used=False,encoder_costs=training_costs,
        encoder_meter_includes_calibration=True,backward_compute_not_represented_by_forward_tokens=True))
    meter_handle.remove()
    print('PATH_TRAINING_COMPLETED',best_epoch,best,flush=True)


def retrieve():
    sid=verify()
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    complete=load(RUN/'training_complete.json')
    if sha(RUN/'best.pt')!=complete['checkpoint_sha256']: raise ValueError('Trained checkpoint changed')
    tok,model=make_model('cuda')
    model.load_state_dict(torch.load(RUN/'best.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
    rows=[]
    for i,source in enumerate(load(PLAN/'public/development_inputs.json'),1):
        row=dict(question_id=source['question_id'],source_sha256=fp(source),signature=sid,**infer(torch,tok,model,source))
        rows.append(row);write_json(RUN/'retrieval_status.json',dict(completed=i,expected=128))
        print('PATH_RETRIEVE',i,'/128','predicted_hops',row['predicted_hops'],flush=True)
    immutable(RUN/'retrieval.json',rows)


def pack_path(source,path,scores,counter,budget=2048):
    if not path or len(path)!=len(set(path)) or any(not 0<=p<20 for p in path):
        raise ValueError('Invalid predicted path')
    paragraphs=[source['paragraphs'][j] for j in path]
    ranges={p['source_id']:[[0,len(p['text'])]] for p in paragraphs}
    text=recomp.render_ranges(source,ranges)
    if counter.count(text)<=budget:
        plan=dict(**{k:source[k] for k in ('question_id','native_id','mode','question')},
            arm='beam_path_fixed',reader_variant='format_only',selected_ids=['quote:'+p['source_id'] for p in paragraphs],
            selected_ranges=ranges,selected_spans=[p['source_id']+':span:'+str(i) for p in paragraphs for i in range(len(p['spans']))],
            fully_preserved_sources=list(ranges),evidence_text=text,evidence_tokens=counter.count(text),
            evidence_sha256=fp(text),private_inputs_read=False,diagnostic_only=False)
        return plan,dict(packing='whole_path',predicted_path=path)
    sub=dict(source,paragraphs=paragraphs)
    allowed={u for u,_ in recomp.units(sub)}
    plan,trace=recomp.pack(sub,{u:v for u,v in scores.items() if u in allowed},'recomp_sentence',counter,budget)
    plan['arm']='beam_path_fixed'
    return plan,dict(packing='ordinary_sentence_compression',predicted_path=path,trace=trace)


def select():
    sid=verify()
    import torch
    from transformers import AutoModel,AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter=ExactCounter();tok=AutoTokenizer.from_pretrained(str(recomp.MODEL),local_files_only=True)
    reader_tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(str(recomp.MODEL),local_files_only=True,dtype=torch.float32,attn_implementation='eager').to('cpu').eval()
    paths={r['question_id']:r for r in load(RUN/'retrieval.json')};all_plans=[];maximum_prompt=0
    for i,s in enumerate(load(PLAN/'public/development_inputs.json'),1):
        row=paths[s['question_id']]
        if row['signature']!=sid or row['source_sha256']!=fp(s): raise ValueError('Retrieval source mismatch')
        texts=[s['question']]+[t for _,t in recomp.units(s)]
        encoded=[tok.encode(t,add_special_tokens=True,truncation=False) for t in texts]
        if max(map(len,encoded))>512: raise ValueError('No RECOMP truncation permitted')
        chunks=[];started=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(encoded),32):
                batch=tok.pad(dict(input_ids=encoded[start:start+32]),padding=True,return_tensors='pt')
                chunks.append(recomp.mean_pooling(torch,model(**batch).last_hidden_state,batch['attention_mask']))
        em=torch.cat(chunks)
        scores={u:float(em[0]@em[j]) for j,(u,_) in enumerate(recomp.units(s),1)}
        flat,flat_trace=recomp.pack(s,scores,'recomp_sentence',counter)
        path,path_trace=pack_path(s,row['paths'][0],scores,counter)
        for plan in (flat,path):
            full=reader_tok.apply_chat_template(recomp.messages_for(plan),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            length=counter.count(full);maximum_prompt=max(maximum_prompt,length)
            if length+2048>32768:raise ValueError('Reader prompt overflow; no silent truncation')
        immutable(RUN/'selected'/(s['question_id'].replace(':','_')+'.json'),dict(signature=sid,source_sha256=fp(s),
            scores=scores,plans=[flat,path],flat_trace=flat_trace,path_trace=path_trace,
            shared_encoding=dict(forward_calls=math.ceil(len(encoded)/32),input_tokens=sum(map(len,encoded)),seconds=time.perf_counter()-started)))
        all_plans.extend([flat,path]);write_json(RUN/'selection_status.json',dict(completed=i,expected=128))
        print('PATH_PACK',i,'/128',path_trace['packing'],flush=True)
    if len(all_plans)!=256: raise ValueError('Incomplete development matrix')
    immutable(RUN/'qa_plans.json',all_plans)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=sha(RUN/'qa_plans.json'),conditions=256,max_prompt_tokens=maximum_prompt))


def qa(endpoint):
    sid=verify()
    ledger=recomp.ledger
    old=ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for
    ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=RUN,lambda:sid,lambda:{},recomp.messages_for
    try: ledger.qa(endpoint)
    finally: ledger.RUN,ledger.verify,ledger.saved_cache,ledger.messages_for=old


def score():
    sid=verify();refs=load(PLAN/'private/development_references.json')
    plans=load(RUN/'qa_plans.json')
    scored=recomp.interface.parent.score_rows(plans,list(read_jsonl(RUN/'qa/predictions.jsonl')),refs,sid)
    summaries=[];comparisons=[]
    for scope in ('overall','multi','single'):
        def by(arm): return {r['question_id']:r for r in scored if r['arm']==arm and (scope=='overall' or r['mode']==scope)}
        for arm in ARMS:
            rows=list(by(arm).values())
            summaries.append(dict(arm=arm,scope=scope,n=len(rows),f1_points=100*statistics.mean(r['local_native_f1'] for r in rows),
                mean_evidence_tokens=statistics.mean(r['evidence_tokens'] for r in rows),statuses=dict(Counter(r['status'] for r in rows)),
                all_support_sources_touched=sum(set(map(str,refs[r['question_id']]['support_indices'])) <= {u.split(':')[0] for u in r['selected_spans']} for r in rows)))
        comparisons.append(dict(scope=scope,**recomp.interface.paired(by('beam_path_fixed'),by('recomp_sentence'))))
    retrieval=load(RUN/'retrieval.json')
    set_f1=[support_f1(r['paths'][0],refs[r['question_id']]['support_indices']) for r in retrieval]
    calls=list(read_jsonl(RUN/'qa/answer_calls.jsonl'))
    if len(calls)>256: raise ValueError('Call budget exceeded')
    changes={scope:next(c for c in comparisons if c['scope']==scope) for scope in ('overall','multi','single')}
    resource_signal=(changes['overall']['delta_points']>0 and changes['multi']['delta_points']>0
        and changes['multi']['wins']>=3 and changes['single']['delta_points']>=-3.125
        and changes['overall']['normal_pair_delta_points'] is not None
        and changes['overall']['normal_pair_delta_points']>0)
    immutable(RUN/'scored.json',scored)
    immutable(RUN/'report.json',dict(signature=sid,complete=True,scores=summaries,comparisons=comparisons,
        retrieval_support_set_f1=statistics.mean(set_f1),
        hop_accuracy=statistics.mean(r['predicted_hops']==len(refs[r['question_id']]['support_indices']) for r in retrieval),
        new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in scored),
        reader_cost={k:sum(c['result'].get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens')},
        training=load(RUN/'training_complete.json'),selection_costs_retained_in_selected_files=True,
        resource_continue_signal=resource_signal,resource_signal_requires_manual_evidence_case_audit=True,
        calibration_and_development_distinct=True,independent_confirmation=False,paper_ready=False,
        dependency_protection_implemented=False,automatic_followup=False,config=CONFIG))
    print('PATH_PILOT_RESULTS',json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','verify','preflight','train','retrieve','select','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8023');args=p.parse_args()
    if args.mode=='qa': qa(args.endpoint)
    else:
        result=globals()[args.mode]()
        if result: print(result,flush=True)
