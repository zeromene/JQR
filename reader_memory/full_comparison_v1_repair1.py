"""Support native17--20 candidates without changing the frozen method or evaluation set."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from . import full_comparison_v1 as parent

ROOT=parent.ROOT
PLAN=ROOT/'research/full_comparison_v1_repair1'
RUN=ROOT/'runs/full_comparison_v1_repair1'
PARENT_PLAN=parent.PLAN
ORIGINAL_VERIFY_CODE=parent.verify_code
CONFIG=dict(version='native-candidate-interface-repair1',scientific_config=parent.CONFIG,
    correction='preserve actual native17--20 candidates; no padding, dropping, labels or altered methods',
    original_code_identity_source='research/full_comparison_v1/code_manifest.json',
    native_candidate_counts={'17':2,'18':5,'19':9,'20':2401},new_model_calls_in_failed_preparation=0)
CODE_FILES=['reader_memory/full_comparison_v1_repair1.py','tools/prepare_full_comparison_v1_repair1.py',
    'tools/full_comparison_v1_repair1_remote.py','scripts/full_comparison_v1_repair1_queue.py',
    'tests/test_full_comparison_v1_repair1.py','research/full_comparison_v1_repair1/PROTOCOL.md',
    'research/full_comparison_v1_repair1/sources/metrics_answer.py']


def native_public(row,question,mode):
    paragraphs=[]
    for p in row['paragraphs']:
        text=p['paragraph_text']
        spans=parent.recomp.core.sentence_spans(text)
        for s in spans:
            while s['start']<s['end'] and text[s['start']].isspace():s['start']+=1
            while s['end']>s['start'] and text[s['end']-1].isspace():s['end']-=1
            if text[s['start']:s['end']]!=s['text']:raise ValueError('Source offset changed')
        if ''.join(''.join(s['text'] for s in spans).split())!=''.join(text.split()):
            raise ValueError('Sentence split lost source characters')
        paragraphs.append(dict(source_id=str(p['idx']),title=p['title'],timestamp='undated',text=text,spans=spans))
    if not 17<=len(paragraphs)<=20 or [p['source_id'] for p in paragraphs]!=[str(i) for i in range(len(paragraphs))]:
        raise ValueError('Unexpected native candidate count/order')
    return dict(question_id=row['id']+':'+mode,native_id=row['id'],mode=mode,question=question,paragraphs=paragraphs)


def native_validate(source):
    if set(source)!={'question_id','native_id','mode','question','paragraphs'}:
        raise ValueError('Private fields entered selector')
    if not 17<=len(source['paragraphs'])<=20:
        raise ValueError('Native candidate count outside declared range')
    for i,p in enumerate(source['paragraphs']):
        if set(p)!={'source_id','title','timestamp','text','spans'} or p['source_id']!=str(i):
            raise ValueError('Candidate schema/order changed')
        for span in p['spans']:
            if set(span)!={'start','end','text'} or p['text'][span['start']:span['end']]!=span['text']:
                raise ValueError('Candidate span changed')


def verify_code():
    value=parent.load(PLAN/'code_manifest.json')
    if value['config']!=CONFIG or value['signature']!=parent.fp({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Repair identity changed')
    for n,digest in value['files'].items():
        if parent.sha(ROOT/n)!=digest:raise ValueError('Repair file changed: '+n)
    old=parent.PLAN
    parent.PLAN=PARENT_PLAN
    try:
        previous=ORIGINAL_VERIFY_CODE()
    finally:
        parent.PLAN=old
    if previous!=value['parent_code_signature']:
        raise ValueError('Original frozen method changed')
    return value['signature']


def freeze_code():
    original_signature=ORIGINAL_VERIFY_CODE()
    value=dict(config=CONFIG,parent_code_signature=original_signature,files={n:parent.sha(ROOT/n) for n in CODE_FILES})
    value['signature']=parent.fp(value)
    parent.immutable(PLAN/'code_manifest.json',value)
    print('FULL_NATIVE_INTERFACE_FROZEN',value['signature'],flush=True)


def verify():
    value=parent.load(PLAN/'manifest.json')
    if value['code_signature']!=verify_code() or value['signature']!=parent.fp({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Repaired evaluation binding changed')
    for n,digest in value['files'].items():
        if parent.sha(PLAN/n)!=digest:raise ValueError('Evaluation data changed: '+n)
    return value['signature']


@contextmanager
def bind():
    old=(parent.PLAN,parent.RUN,parent.verify_code,parent.verify,parent.beam.validate_source)
    parent.PLAN,parent.RUN,parent.verify_code,parent.verify,parent.beam.validate_source=PLAN,RUN,verify_code,verify,native_validate
    parent.official_metric.cache_clear()
    try:
        yield
    finally:
        parent.PLAN,parent.RUN,parent.verify_code,parent.verify,parent.beam.validate_source=old
        parent.official_metric.cache_clear()


def run(mode,endpoint='http://127.0.0.1:8037'):
    with bind():
        if mode=='qa':parent.qa(endpoint)
        elif mode in ('recomp_sentence','recomp_hotpot_sentence'):parent.recomp_select(mode)
        else:getattr(parent,mode)()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze_code','freeze_data','verify','preflight','retrieve','dense','qwen',
        'llmlingua','recomp_sentence','recomp_hotpot_sentence','assemble','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8037')
    a=p.parse_args()
    if a.mode=='freeze_code':freeze_code()
    elif a.mode=='verify':print(verify(),flush=True)
    else:run(a.mode,a.endpoint)
