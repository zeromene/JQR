"""Score only a completed, identity-checked screen; no feedback into selection."""
from collections import Counter
import statistics
from . import paper_competition_v1 as p
from . import full_comparison_v1 as mu
from .evidence_acquisition_v2 import parse_answer
u=p.u


def analyze():
    sid=p.verify();completion=u.load(p.RUN/'completion.json');assert completion['signature']==sid
    plans=u.load(p.RUN/'plans.json');mapping=u.load(p.RUN/'mapping.json');requests=u.load(p.RUN/'requests.json')
    assert len(plans)==len(mapping)==1536
    outputs={k:u.load(p.RUN/'outputs'/(k+'.json')) for k in requests}
    for key,v in outputs.items():
        req=requests[key]
        assert v['signature']==sid and v['key']==key==u.fp([req['reader'],req['messages'],req['max_output']])
        assert isinstance(v['result']['text'],str) and isinstance(v['result']['usage'],dict)
    index={(v['dataset'],v['question_id'],v['budget'],v['method']):v for v in mapping}
    assert len(index)==1536
    for v in plans:
        m=index[v['dataset'],v['question_id'],v['budget'],v['method']]
        assert m['plan_hash']==u.fp(v)
        if m['key']:assert m['key'] in outputs
        else:assert v['status'] in ('selection_failed','input_infeasible')
    # Reference access is deliberately below the complete-matrix checks.
    ids={d:{x['source']['question_id'] for x in p.sources() if x['dataset']==d} for d in ['hotpot','musique']}
    refs={}
    for d,name in [('hotpot','hotpot_confirmation_v1'),('musique','full_comparison_v1_repair1')]:
        raw=u.load(p.ROOT/'research'/name/'private/references.json')
        if isinstance(raw,list):raw={v['question_id']:v for v in raw}
        refs[d]={q:raw[q] for q in sorted(ids[d])}
    metric=p.hp.official_metrics();records=[]
    for plan in plans:
        d,q,b,method=(plan[k] for k in ('dataset','question_id','budget','method'))
        key=index[d,q,b,method]['key'];hypothesis='';em=f1=0.;status=plan['status'];usage={};origin=None
        if key:
            output=outputs[key];result=output['result'];usage=result['usage'];origin=output['origin']
            if d=='hotpot':
                hypothesis=result['text'];status='ok' if hypothesis.strip() else 'empty'
                em=float(metric['exact_match_score'](hypothesis,refs[d][q]['answer']))
                f1=float(metric['f1_score'](hypothesis,refs[d][q]['answer'])[0])
            else:
                if result['finish_reason']!='stop':status='output_limit'
                else:
                    try:hypothesis=parse_answer(result['text'])
                    except (ValueError,TypeError,KeyError):status='parse_failure'
                    else:em,f1=mu.answer_scores(hypothesis,refs[d][q]);status='ok'
        records.append(dict(dataset=d,question_id=q,budget=b,method=method,f1=f1,em=em,
            hypothesis=hypothesis,status=status,evidence_tokens=plan['evidence_tokens'],usage=usage,origin=origin,key=key))
    groups=[];comparisons=[]
    ix={(v['dataset'],v['question_id'],v['budget'],v['method']):v for v in records}
    for d in ids:
        clusters={q:q for q in ids[d]} if d=='hotpot' else mu.source_clusters(refs[d])
        for b in (256,512,0):
            methods=p.ARMS if b else ['itercomp_native','uncompressed']
            for method in methods:
                rr=[ix[d,q,b,method] for q in sorted(ids[d])]
                groups.append(dict(dataset=d,budget=b,method=method,n=len(rr),
                    f1=100*statistics.mean(v['f1'] for v in rr),em=100*statistics.mean(v['em'] for v in rr),
                    evidence_tokens=statistics.mean(v['evidence_tokens'] for v in rr),
                    reader_input=statistics.mean(v['usage'].get('prompt_tokens',0) for v in rr),
                    reader_output=statistics.mean(v['usage'].get('completion_tokens',0) for v in rr),
                    statuses=dict(Counter(v['status'] for v in rr))))
            if b:
                for method in ('quote_anchor','rank_add'):
                    for control in p.ARMS[2:]:
                        comparisons.append(dict(dataset=d,budget=b,method=method,control=control,
                            **mu.paired_stats({q:ix[d,q,b,method]['f1'] for q in ids[d]},
                                              {q:ix[d,q,b,control]['f1'] for q in ids[d]},clusters)))
    costs={}
    for phase in ('propose','itercomp','qa'):
        calls=[u.load(f) for f in (p.RUN/'calls'/phase).glob('*.json') if not f.name.endswith('.started.json')]
        costs[phase]=dict(new_calls=len(calls),model='Qwen3.5-9B',
            prompt_tokens=sum(v['result']['usage']['prompt_tokens'] for v in calls),
            output_tokens=sum(v['result']['usage']['completion_tokens'] for v in calls))
    for phase in ('compact','longrefiner'):
        calls=[u.load(f) for f in (p.RUN/'compressor_calls'/phase).glob('*.json') if not f.name.endswith('.started.json')]
        costs[phase]=dict(new_calls_including_synthetic=len(calls),
            model='CompAct-7b' if phase=='compact' else 'Qwen2.5-3B-Instruct+LongRefiner adapters',
            prompt_tokens=sum(v['result']['input_tokens'] for v in calls),
            output_tokens=sum(len(v['result']['token_ids']) for v in calls))
    resources={f.stem:u.load(f) for f in (p.RUN/'resources').glob('*.json')}
    result=dict(signature=sid,complete=True,logical_conditions=1536,groups=groups,comparisons=comparisons,
        records=records,assembly=u.load(p.RUN/'assembly.json'),new_compute=costs,resources=resources,
        inherited_stage_costs={q:{k:v.get('costs',{}) for k,v in data.items()} for q,data in p.inputs()['inputs'].items()},
        full_deployment_costs_complete=False,
        historical_cost_gaps=['Some inherited proposal/compressor costs require separate provenance-ledger recovery; never count them as zero.'],
        interpretation='Exposed development screen; per-dataset protocols; 2000 cluster bootstrap draws; descriptive unadjusted intervals; no speed or mechanism claim.',
        decision='competition_screen_complete_requires_review_before_any_further_experiment')
    u.immutable(p.RUN/'analysis.json',result)
    lines=['# 开发竞争力筛选结果','',result['interpretation'],'',
        '|数据|预算|方法|F1|EM|证据token|读者输入|','|---|---:|---|---:|---:|---:|---:|']
    for v in groups:lines.append(f"|{v['dataset']}|{v['budget']}|{v['method']}|{v['f1']:.2f}|{v['em']:.2f}|{v['evidence_tokens']:.1f}|{v['reader_input']:.1f}|")
    lines+=['','完整历史部署成本尚有缺项，不能用本表宣称全流程效率占优。新计算、历史复用、显存和阶段时间单列保存在 analysis.json。',
        '这是开发筛选，尚未达到投稿就绪；下一步由全部对照结果决定，不自动开下一轮。']
    with (p.RUN/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    p.state('analyzed',logical_conditions=1536,decision=result['decision'])
