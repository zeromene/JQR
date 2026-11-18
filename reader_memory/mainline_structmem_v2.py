"""Pinned official StructMem event + cross-event synthesis, common-model 2K adaptation."""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace
import uuid

from . import baseline_bootstrap_v2 as base
from . import evidence_acquisition_v1 as evidence
from .evidence_acquisition_v2 import parse_answer
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/mainline_structmem_v2'
OFFICIAL = ROOT/'third_party/structmem_official'
CONFIG = dict(method='StructMem-event-cross-event-common-model-2k', reader='Qwen3.5-9B',
    embedding='Qwen3-Embedding-0.6B', embedding_dims=1024, embedding_device='cpu',
    compression_model='llmlingua2-small', compression_rate=.6,
    compression_device='cuda', compression_attention='eager for native attention segmentation',
    topic_shared=True, extraction_mode='event', extract_threshold=.1,
    writer_max_tokens=16000, internal_buffer_tokenizer='local common Qwen tokenizer',
    summary_time_window=3600, summary_top_k_seeds=15, summary_scope='global',
    summary_before_update=True, offline_update_threshold=.9,
    event_top_k=60, summary_top_k=5, evidence_budget=2048, summary_reserve=1024,
    packing='whole-entry first-fit summary <=1024; events <=2048; remaining summaries <=2048',
    answer_prompt='common SimpleMem QA protocol; native StructMem event date/speaker content',
    answer_max_tokens=2048, decoding='bf16-sdpa-greedy-thinking_false',
    official_revision='8449d574df6bae1bdf3314a1564da65e2f37e046', heldout_accessed=False)
ACTIVE = {}
AUDIT_LOCK = threading.RLock()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity():
    base.verify_inputs()
    provenance = load(OFFICIAL/'PROVENANCE.json')
    if provenance['revision'] != CONFIG['official_revision']:
        raise ValueError('Official revision mismatch')
    for name, meta in provenance['files'].items():
        if base.sha(OFFICIAL/name) != meta['sha256']:
            raise ValueError('Official source modified: '+name)
    paths = ['reader_memory/mainline_structmem_v2.py', 'reader_memory/mainline_service_v1.py',
        'scripts/mainline_structmem_queue_v2.py', 'research/mainline_baselines_v1/STRUCTMEM_PROTOCOL_V2.md',
        'reader_memory/backend.py', 'reader_memory/evidence_acquisition_v1.py',
        'reader_memory/evidence_acquisition_v2.py', 'reader_memory/tokenization.py']
    value = dict(config=CONFIG, code={p:base.sha(ROOT/p) for p in paths},
        public_inputs={p:base.sha(base.PLAN/p) for p in ['public/queries.json','public/conversations.json']},
        upstream_provenance_sha256=base.sha(OFFICIAL/'PROVENANCE.json'),
        runtime_lock_sha256=base.sha(ROOT/'offline/structmem-installed-lock.txt'),
        model_configs={p:base.sha(ROOT/'models'/p/'config.json') for p in
            ['Qwen3.5-9B','Qwen3-Embedding-0.6B','llmlingua2-small']})
    target = RUN/'identity.json'
    if target.exists() and load(target) != value:
        raise ValueError('Frozen StructMem identity changed; use new version')
    write_json(target, value)
    return fingerprint(value)


def author_def(path, name, env):
    """Load a single unmodified author definition without CLI/import side effects."""
    text = path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name == name)
    import textwrap
    exec(compile('from __future__ import annotations\n'+textwrap.dedent(ast.get_source_segment(text,node)),
        str(path),'exec'), env)
    return env[name]


def adapters():
    """Keep native algorithms; substitute audited local transport and eager BERT attention."""
    sys.path.insert(0,str(OFFICIAL/'src'))
    sys.path.insert(0,str(OFFICIAL/'experiments/locomo'))
    from lightmem.factory.memory_manager.openai import OpenaiManager
    from lightmem.factory.memory_manager.factory import MemoryManagerFactory
    from lightmem.factory.pre_compressor.llmlingua_2 import LlmLingua2Compressor
    from lightmem.factory.pre_compressor.factory import PreCompressorFactory
    from transformers import AutoTokenizer

    class Client:
        def __init__(self):
            self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            if kw.get('tools'):
                raise ValueError('Unexpected tool request from native manager')
            # Official manager can use threads. Serialize audited inference calls;
            # this affects wall time, not the independent prompts/responses.
            with AUDIT_LOCK:
                call_id = str(uuid.uuid4())
                out = RUN/ACTIVE['conversation_id']
                call = dict(**ACTIVE, call_id=call_id, request=kw)
                intent = out/'writer_inflight.json'
                if intent.exists():
                    raise ValueError('Uncommitted native write; no hidden retry')
                write_json(intent,call)
                started=time.perf_counter()
                result={}
                try:
                    result=base.rpc(ACTIVE['endpoint'],'/complete',dict(messages=kw['messages'],max_tokens=kw['max_tokens']))
                    text=base.require_complete(result)
                except Exception as exc:
                    base.append(out/'writer_calls.jsonl',dict(**call, result=result,
                        status='failed',error=f'{type(exc).__name__}: {exc}',seconds=time.perf_counter()-started))
                    intent.unlink()
                    raise
                base.append(out/'writer_calls.jsonl',dict(**call,result=result,
                    status='complete',seconds=time.perf_counter()-started))
                intent.unlink()
                usage=result['usage']
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text,tool_calls=None))],
                    usage=SimpleNamespace(prompt_tokens=usage['prompt_tokens'],completion_tokens=usage['completion_tokens'],
                        total_tokens=usage['prompt_tokens']+usage['completion_tokens']))

    class LocalManager(OpenaiManager):
        def __init__(self,config):
            # Native manager's algorithms use only these initialized members.
            self.config=config
            self.context_windows=32768
            self.client=Client()
            self.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),local_files_only=True)

    class EagerCompressor(LlmLingua2Compressor):
        def __init__(self,config):
            self.config=config
            from llmlingua import PromptCompressor
            self._compressor=PromptCompressor(
                model_name=config.llmlingua_config['model_name'],
                device_map=config.llmlingua_config['device_map'],use_llmlingua2=True,
                llmlingua2_config=config.llmlingua2_config,
                model_config=dict(attn_implementation='eager'))
            # LLMLingua's word grouping uses its model_name string after loading.
            # Supply original architecture identity without reloading or downloading.
            self._compressor.model_name='iofu728/llmlingua-2-bert-base-multilingual-cased-meetingbank'
            self._compressor.model.eval()

    # Factories import names from this very module instance (also under -m).
    globals()['LocalManager']=LocalManager
    globals()['EagerCompressor']=EagerCompressor
    sys.modules['reader_memory.mainline_structmem_v2']=sys.modules[__name__]
    MemoryManagerFactory._MODEL_MAPPING['openai']='reader_memory.mainline_structmem_v2.LocalManager'
    PreCompressorFactory._MODEL_MAPPING['llmlingua-2']='reader_memory.mainline_structmem_v2.EagerCompressor'


def configuration(cid,device='cuda',directory=None):
    out=directory or RUN/cid
    return dict(pre_compress=True,
        pre_compressor=dict(model_name='llmlingua-2',configs=dict(llmlingua_config=dict(
            model_name=str(ROOT/'models/llmlingua2-small'),device_map=device,use_llmlingua2=True),
            compress_config=dict(instruction='',rate=.6,target_token=-1))),
        topic_segment=True,precomp_topic_shared=True,topic_segmenter=dict(model_name='llmlingua-2'),
        messages_use='user_only',metadata_generate=True,text_summary=True,
        memory_manager=dict(model_name='openai',configs=dict(model='Qwen3.5-9B',max_tokens=16000,
            temperature=0,top_p=1,api_key='offline-local-only',openai_base_url='http://127.0.0.1:8016')),
        extract_threshold=.1,index_strategy='embedding',
        text_embedder=dict(model_name='huggingface',configs=dict(model=str(ROOT/'models/Qwen3-Embedding-0.6B'),
            embedding_dims=1024,model_kwargs=dict(device='cpu',local_files_only=True))),
        retrieve_strategy='embedding',
        embedding_retriever=dict(model_name='qdrant',configs=dict(collection_name=cid,
            embedding_model_dims=1024,path=str(out/'qdrant_events'),on_disk=True)),
        summary_retriever=dict(model_name='qdrant',configs=dict(collection_name=cid+'_summary',
            embedding_model_dims=1024,path=str(out/'qdrant_summaries'),on_disk=True)),
        update='offline',extraction_mode='event',history_db_path=str(out/'history.db'),
        logging=dict(level='INFO',file_enabled=True,log_dir=str(out/'native_logs')))


def preflight():
    """Only compatibility: real compression/attention, no QA or teacher generation."""
    adapters()
    from lightmem.configs.base import BaseMemoryConfigs
    from lightmem.factory.pre_compressor.factory import PreCompressorFactory
    from lightmem.factory.topic_segmenter.factory import TopicSegmenterFactory
    config=BaseMemoryConfigs(**configuration('fixture',device='cpu',directory=RUN/'preflight'))
    comp=PreCompressorFactory.from_config(config.pre_compressor)
    seg=TopicSegmenterFactory.from_config(config.topic_segmenter,True,comp)
    matrix=seg.sentence_level_attention(['Lena works in Berlin.','I will visit her work city.'])
    compressed=comp.inner_compressor.compress_prompt(context=['Lena works in Berlin and will visit her family next week.'],rate=.6,target_token=-1)
    if matrix.shape != (2,2) or not compressed['compressed_prompt']:
        raise ValueError('Native compressor/segmenter fixture failed')
    write_json(RUN/'preflight.json',dict(status='passed',matrix_shape=list(matrix.shape),
        compressed_fixture=compressed['compressed_prompt'],compression_inference_calls=1,teacher_generation_calls=0,qa_inputs_read=False))
    print('STRUCTMEM_COMPATIBILITY_OK',flush=True)


def writer(cid,conv,sid,endpoint):
    out=RUN/cid
    out.mkdir(parents=True,exist_ok=True)
    if (out/'build_manifest.json').exists():
        m=load(out/'build_manifest.json')
        if m['signature'] != sid or any(base.sha(out/p) != digest for p,digest in m['exports'].items()):
            raise ValueError('Native build cache identity mismatch')
        return
    if (out/'build_started.json').exists():
        raise ValueError('Native build interrupted; preserve partial output for audit, no automatic rerun')
    write_json(out/'build_started.json',dict(signature=sid,conversation_id=cid))
    import lightmem.memory.lightmem as native
    from prompts import LoCoMo_Event_Binding_factual,LoCoMo_Event_Binding_relational
    native.GLOBAL_TOPIC_IDX=0
    native.GLOBAL_LAST_SUMMARY_TIME=None
    ACTIVE.clear()
    ACTIVE.update(signature=sid,conversation_id=cid,endpoint=endpoint,phase='extract',turn=0)
    config=configuration(cid)
    write_json(out/'native_config.json',config)
    # Native save_memory_entries uses a relative default filename. Isolate it.
    saved_cwd=Path.cwd()
    os.chdir(out)
    try:
        memory=native.LightMemory.from_config(config)
        speakers=list(dict.fromkeys(t['speaker'] for t in conv['turns']))
        prompts=dict(factual=LoCoMo_Event_Binding_factual,relational=LoCoMo_Event_Binding_relational)
        for index,t in enumerate(conv['turns']):
            ACTIVE['turn']=index+1
            stamp=datetime.strptime(t['timestamp'].strip('()'),'%I:%M %p on %d %B, %Y').strftime('%Y-%m-%d %H:%M:%S')
            metadata=dict(speaker_id='speaker_a' if t['speaker']==speakers[0] else 'speaker_b',
                speaker_name=t['speaker'],time_stamp=stamp)
            pair=[dict(role='user',content=t['content'],**metadata),dict(role='assistant',content='',**metadata)]
            last=index==len(conv['turns'])-1
            result=memory.add_memory(messages=pair,METADATA_GENERATE_PROMPT=prompts,
                force_segment=last,force_extract=last)
            base.append(out/'turns.jsonl',dict(turn=index+1,source_id=t['source_id'],result=result))
            if index%10==0 or last:
                write_json(RUN/'status.json',dict(status='running',phase='event-extraction',
                    conversation_id=cid,turn=index+1,total_turns=len(conv['turns'])))
                print('STRUCTMEM_WRITE',cid,index+1,'/',len(conv['turns']),flush=True)
        events_before=memory.embedding_retriever.get_all(with_vectors=True,with_payload=True)
        write_json(out/'events_before_update.json',events_before)
        if not events_before:
            raise ValueError('Official event extraction produced no entries')
        ACTIVE['phase']='cross-event-summary'
        write_json(RUN/'status.json',dict(status='running',phase='cross-event-summary',conversation_id=cid))
        summary_result=memory.summarize(retrieval_scope='global',time_window=3600,
            top_k_seeds=15,process_all=True)
        write_json(out/'summary_result.json',summary_result)
        summaries=memory.summary_retriever.get_all(with_vectors=True,with_payload=True)
        write_json(out/'summaries.json',summaries)
        ACTIVE['phase']='offline-update'
        write_json(RUN/'status.json',dict(status='running',phase='offline-update',conversation_id=cid))
        memory.construct_update_queue_all_entries()
        memory.offline_update_all_entries(score_threshold=.9)
        events=memory.embedding_retriever.get_all(with_vectors=True,with_payload=True)
        write_json(out/'events.json',events)
        write_json(out/'native_token_statistics.json',memory.get_token_statistics())
        calls=list(read_jsonl(out/'writer_calls.jsonl')) if (out/'writer_calls.jsonl').exists() else []
        write_json(out/'build_manifest.json',dict(signature=sid,events_before_update=len(events_before),
            events=len(events),summaries=len(summaries),calls=len(calls),
            failed_calls=sum(c['status']!='complete' for c in calls),
            exports={p:base.sha(out/p) for p in ['events_before_update.json','events.json',
                'summaries.json','summary_result.json','native_token_statistics.json']},
            summary_on_pre_update_events=True,private_inputs_read=False))
        memory.embedding_retriever.client.close()
        memory.summary_retriever.client.close()
        del memory
        import gc,torch
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        os.chdir(saved_cwd)


def make_plans(cid,queries,sid):
    import numpy as np
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    from .tokenization import TokenCounter
    out=RUN/cid
    if (out/'plan_manifest.json').exists():
        return verify_plans(cid,sid)
    counter=TokenCounter(ROOT/'models/Qwen3.5-9B')
    embedder=TextEmbedderHuggingface(BaseTextEmbedderConfig(model=str(ROOT/'models/Qwen3-Embedding-0.6B'),
        embedding_dims=1024,model_kwargs=dict(device='cpu',local_files_only=True)))
    env=dict(np=np)
    cls=author_def(OFFICIAL/'experiments/locomo/retrievers.py','VectorRetriever',env)
    retrieve=cls(embedder)
    format_native=author_def(OFFICIAL/'experiments/locomo/retrievers.py','format_related_memories',dict(datetime=datetime))
    events,summaries=load(out/'events.json'),load(out/'summaries.json')
    plans=[]
    for q in queries:
        er=retrieve.retrieve(events,q['question'],limit=60)
        sr=retrieve.retrieve(summaries,q['question'],limit=5) if summaries else []
        if not er:
            raise ValueError('No retrieved StructMem events; inspect embedding')
        event_entries=[]
        for r in er:
            speaker=r['payload'].get('speaker_name','Unknown')
            event_entries.append(evidence.memory_entry('event:'+r['id'],
                speaker+': '+format_native([r])))
        summary_entries=[evidence.memory_entry('summary:'+r['id'],
            'Cross-event summary: '+str(r['payload'].get('summary',r['payload'].get('memory','')))) for r in sr]
        chosen,seen=[],set()
        def pack(entries,cap):
            for e in entries:
                key=e['entry_id']
                if key in seen:
                    continue
                trial=chosen+[e]
                if counter.count(evidence.FORMAT(None,[SimpleNamespace(**v) for v in trial]))<=cap:
                    chosen.append(e)
                    seen.add(key)
        pack(summary_entries,1024)
        pack(event_entries,2048)
        pack(summary_entries,2048)
        text=evidence.FORMAT(None,[SimpleNamespace(**v) for v in chosen])
        plans.append(dict(**q,signature=sid,method=CONFIG['method'],contexts=chosen,
            evidence_tokens=counter.count(text),evidence_budget=2048,
            retrieved_event_ids=[r['id'] for r in er],retrieved_summary_ids=[r['id'] for r in sr],
            selected_event_count=sum(e['entry_id'].startswith('event:') for e in chosen),
            selected_summary_count=sum(e['entry_id'].startswith('summary:') for e in chosen)))
    write_json(out/'plans.json',plans)
    write_json(out/'plan_manifest.json',dict(signature=sid,plans_sha256=base.sha(out/'plans.json'),
        build_manifest_sha256=base.sha(out/'build_manifest.json'),n=len(plans),private_inputs_read=False))
    return plans


def verify_plans(cid,sid):
    out=RUN/cid
    manifest=load(out/'plan_manifest.json')
    if manifest['signature']!=sid or manifest['plans_sha256']!=base.sha(out/'plans.json') or \
        manifest['build_manifest_sha256']!=base.sha(out/'build_manifest.json'):
        raise ValueError('StructMem plan identity mismatch')
    return load(out/'plans.json')


def answer(plans,sid,endpoint):
    rows=list(read_jsonl(RUN/'predictions.jsonl')) if (RUN/'predictions.jsonl').exists() else []
    calls=list(read_jsonl(RUN/'answer_calls.jsonl')) if (RUN/'answer_calls.jsonl').exists() else []
    done={r['question_id'] for r in rows}
    if len(done)!=len(rows) or any(r['signature']!=sid for r in rows+calls) or len(calls)!=len(rows):
        raise ValueError('Invalid, duplicate or uncommitted QA log')
    intent=RUN/'answer_inflight.json'
    if intent.exists():
        if load(intent)['question_id'] not in done:
            raise ValueError('Interrupted QA intent; no automatic retry')
        intent.unlink()
    for p in plans:
        if p['question_id'] in done:
            continue
        text=evidence.FORMAT(None,[SimpleNamespace(**v) for v in p['contexts']])
        tokens=base.rpc(endpoint,'/tokenize',dict(text=text))['tokens']
        if tokens!=p['evidence_tokens'] or tokens>2048:
            raise ValueError('Evidence tokenizer/budget mismatch')
        messages=[dict(role='system',content='You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.'),
            dict(role='user',content=evidence.PROMPT(None,p['question'],text))]
        call=dict(signature=sid,question_id=p['question_id'],messages=messages,max_tokens=2048)
        write_json(intent,call)
        result={}
        start=time.perf_counter()
        hypothesis,error,status='',None,'failed'
        try:
            result=base.rpc(endpoint,'/complete',dict(messages=messages,max_tokens=2048))
            hypothesis=parse_answer(base.require_complete(result))
            status='ok'
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        base.append(RUN/'answer_calls.jsonl',dict(**call,result=result,status=status,error=error,
            seconds=time.perf_counter()-start))
        base.append(RUN/'predictions.jsonl',dict(**p,hypothesis=hypothesis,status=status,error=error))
        intent.unlink()
        done.add(p['question_id'])
        print('STRUCTMEM_QA',len(done),'/64',p['question_id'],status,flush=True)
        if not result:
            raise RuntimeError('No QA service result; stopping after recorded failure')


def run(endpoint):
    sid=identity()
    if not (RUN/'preflight.json').exists() or load(RUN/'preflight.json')['status']!='passed':
        raise ValueError('Native compatibility preflight missing')
    import torch
    torch.set_num_threads(8)
    adapters()
    queries=load(base.PLAN/'public/queries.json')
    for conv in load(base.PLAN/'public/conversations.json'):
        cid=conv['conversation_id']
        writer(cid,conv,sid,endpoint)
        plans=make_plans(cid,[q for q in queries if q['conversation_id']==cid],sid)
        answer(plans,sid,endpoint)
    write_json(RUN/'status.json',dict(status='completed',expected=64,heldout_accessed=False))


def score():
    base.verify_inputs(private=True)
    sid=identity()
    refs={r['question_id']:r for r in load(base.PLAN/'private/references.json')}
    rows=list(read_jsonl(RUN/'predictions.jsonl'))
    if len(rows)!=len({r['question_id'] for r in rows}) or any(r['signature']!=sid for r in rows):
        raise ValueError('Invalid StructMem scoring rows')
    groups=defaultdict(list)
    scored=[]
    for r in rows:
        ref=refs[r['question_id']]
        f1=base.native_f1(r['hypothesis'],ref['answer'],ref['category'])
        scored.append(dict(**r,official_native_f1=f1))
        for scope in ['overall','category:'+str(r['category']),'conversation:'+r['conversation_id']]:
            groups[scope].append((f1,r))
    summaries=[dict(scope=k,n=len(v),native_f1=statistics.mean(x[0] for x in v),
        mean_evidence_tokens=statistics.mean(x[1]['evidence_tokens'] for x in v),
        status_counts=dict(Counter(x[1]['status'] for x in v)),
        questions_with_summary=sum(x[1]['selected_summary_count']>0 for x in v)) for k,v in sorted(groups.items())]
    calls=[c for p in RUN.glob('conv-*/writer_calls.jsonl') for c in read_jsonl(p)]
    costs={phase:dict(calls=sum(c['phase']==phase for c in calls),
        failed=sum(c['phase']==phase and c['status']!='complete' for c in calls),
        **{k:sum(c.get('result',{}).get('usage',{}).get(k,0) for c in calls if c['phase']==phase)
            for k in ['prompt_tokens','completion_tokens']}) for phase in sorted({c['phase'] for c in calls})}
    write_json(RUN/'scored.json',scored)
    report=dict(signature=sid,method=CONFIG['method'],scores=summaries,costs=costs,
        complete={r['question_id'] for r in rows}==set(refs),
        scope='two exposed dev histories; common-model budget adaptation, not native paper-score replication',
        heldout_accessed=False)
    write_json(RUN/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','run','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8016')
    args=parser.parse_args()
    if args.mode=='run':
        run(args.endpoint)
    else:
        globals()[args.mode]()
