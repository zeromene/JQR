"""Independent question confirmation of frozen systems; no new trained method."""
from . import paper_competition_v1 as old
from . import proposal_retention_v1 as retention
u=old.u; ROOT=u.ROOT; BASE=ROOT/'research/paper_confirmation_v1'; RUN=ROOT/'runs/paper_confirmation_v1'
CONFIG=dict(version='paper-confirmation-v1',seed=30,allocation_seed='paper-confirmation-v1-20261114-final',
    hotpot_sources=1400,musique_sources=535,diagnostic_per_dataset=256,budget=512,
    readers=['Qwen3.5-9B','Phi-4-mini-instruct'],methods=['quote_anchor','rank_add','ap3','strong_rerank'],
    extractors=['role','direct'],diagnostic_arms=list(retention.ARMS),diagnostic_reader='Qwen3.5-9B',
    main_logical_conditions=15480,diagnostic_logical_conditions=4096,logical_conditions=19576,
    proposer='Qwen3.5-9B',proposer_max_input=16384,proposer_max_output=2048,
    reader_max_input=8192,hotpot_max_output=128,musique_max_output=2048,
    max_proposer_calls=5805,bootstrap=10000,training=False,automatic_expansion=False,
    primary='Hotpot Qwen quote_anchor minus rank_add: delta >=2 points and two-sided 95% lower bound >0',
    transfer='Same Qwen-budgeted evidence strings to Phi; report actual Phi tokens, no Phi equality claim',
    musique_scope='New questions and disjoint atomic subquestions; reused document facts allowed and disclosed',
    diagnostic='Same-sentence-pool, exact Qwen evidence and full input token match, role and direct separately',
    admission_mib=dict(retrieve=5000,rank=11000,propose=26000,qa_qwen=26000,qa_phi=12000),
    reserve_mib=dict(retrieve=4000,rank=9500,propose=24000,qa_qwen=24000,qa_phi=10500),
    cap_mib=dict(retrieve=4200,rank=10000,propose=24500,qa_qwen=24500,qa_phi=11000),
    poll_seconds=.1,retry_seconds=1,wait_timeout=None,wall_timeout=None,oom_retry_limit=None)

def sources():
    value=u.load(BASE/'public/sources.json')
    assert len(value)==CONFIG['hotpot_sources']+CONFIG['musique_sources']
    assert len({v['source']['question_id'] for v in value})==len(value)
    return value

def verify(models=False):
    m=u.load(BASE/'manifest.json');assert m['config']==CONFIG
    for name,digest in m['files'].items():assert u.sha(ROOT/name)==digest,name
    if models:
        for name,digest in m['model_files'].items():assert u.sha(ROOT/name)==digest,name
    return u.sha(BASE/'manifest.json')

def qa_request(plan,source,reader):
    r=old.qa_request(plan,source);r['reader']=reader
    r['key']=u.fp([reader,r['messages'],r['max_output']])
    return r

def native_mu(row):
    # Train contains one selected native 16-context question; retain it unchanged.
    from .replacement_core_v1 import sentence_spans
    paragraphs=[]
    for p in row['paragraphs']:
        text=p['paragraph_text'];spans=sentence_spans(text)
        for v in spans:
            while v['start']<v['end'] and text[v['start']].isspace():v['start']+=1
            while v['end']>v['start'] and text[v['end']-1].isspace():v['end']-=1
            assert text[v['start']:v['end']]==v['text']
        assert ''.join(''.join(v['text'] for v in spans).split())==''.join(text.split())
        paragraphs.append(dict(source_id=str(p['idx']),title=p['title'],timestamp='undated',text=text,spans=spans))
    assert 16<=len(paragraphs)<=20
    assert [p['source_id'] for p in paragraphs]==[str(i) for i in range(len(paragraphs))]
    return dict(question_id=row['id']+':multi',native_id=row['id'],mode='multi',question=row['question'],paragraphs=paragraphs)
